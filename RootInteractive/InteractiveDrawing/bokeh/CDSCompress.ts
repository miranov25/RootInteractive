import {ColumnDataSource} from "models/sources/column_data_source"
import * as p from "core/properties"
//import * as pako from './pako/'
//declare var Buffer : any
declare const  pako : any
//const pako = require('pako');
//declare namespace pako {
//  function inflate(arrayOut:Uint8Array): Uint8Array;
//}

const is_little_endian = (() => {
  const buf = new ArrayBuffer(4);
  const buf8 = new Uint8Array(buf);
  const buf32 = new Uint32Array(buf);
  buf32[1] = 0x0a0b0c0d;
  let little_endian = true;
  if (buf8[4] == 0x0a && buf8[5] == 0x0b && buf8[6] == 0x0c && buf8[7] == 0x0d) {
      little_endian = false;
  }
  return little_endian;
})();

const BYTE_ORDER = is_little_endian ? "little" : "big";

function swap16(buffer: ArrayBuffer) {
  const x = new Uint8Array(buffer);
  for (let i = 0, end = x.length; i < end; i += 2) {
      const t = x[i];
      x[i] = x[i + 1];
      x[i + 1] = t;
  }
}
function swap32(buffer: ArrayBuffer) {
  const x = new Uint8Array(buffer);
  for (let i = 0, end = x.length; i < end; i += 4) {
      let t = x[i];
      x[i] = x[i + 3];
      x[i + 3] = t;
      t = x[i + 1];
      x[i + 1] = x[i + 2];
      x[i + 2] = t;
  }
}
function swap64(buffer: ArrayBuffer) {
  const x = new Uint8Array(buffer);
  for (let i = 0, end = x.length; i < end; i += 8) {
      let t = x[i];
      x[i] = x[i + 7];
      x[i + 7] = t;
      t = x[i + 1];
      x[i + 1] = x[i + 6];
      x[i + 6] = t;
      t = x[i + 2];
      x[i + 2] = x[i + 5];
      x[i + 5] = t;
      t = x[i + 3];
      x[i + 3] = x[i + 4];
      x[i + 4] = t;
  }
}
function swap(buffer: ArrayBuffer, dtype: string) {
  switch (dtype) {
      case "uint16":
      case "int16":
          swap16(buffer);
          break;
      case "uint32":
      case "int32":
      case "float32":
          swap32(buffer);
          break;
      case "float64":
          swap64(buffer);
          break;
  }
}

/*function encodeArcsinh(x: number, mu: number, sigma0: number, sigma1: number): number {
  return Math.asinh((x - mu) / sigma1) / sigma0;
}*/

function to_fixed_point(x: number, scale: number, origin: number): number {
  return Math.round((x - origin) / scale);
}

export namespace CDSCompress {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnDataSource.Props & {
      inputData :    p.Property<any>
      sizeMap :    p.Property<any>
      enableDithering : p.Property<boolean>
  }
}

export interface CDSCompress extends CDSCompress.Attrs {}

export class CDSCompress extends ColumnDataSource {
  properties: CDSCompress.Props

  constructor(attrs?: Partial<CDSCompress.Attrs>) {
    super(attrs)
  }

  static __name__ = "CDSCompress"

  static {

    this.define<CDSCompress.Props>(({Any, Boolean})=>({
        inputData:    [ Any, {} ],
        sizeMap:    [ Any, {} ],
        enableDithering: [ Boolean, false ]
    }))
  }

  _length: number
  intermediateData: Record<string, any>
  freshColumns: Set<string>
  invalidateOnDitheringToggle: Set<string>

  initialize(): void {
    super.initialize()
    console.info("CDSCompress::initialize")
    this.intermediateData = {}
    this.data = {}
    this.freshColumns = new Set<string>()
    this.invalidateOnDitheringToggle = new Set<string>()
    this._length = -1
  }

  connect_signals(): void {
    super.connect_signals()
    this.connect(this.properties.enableDithering.change, () => this.toggle_dithering())
  }

  inflateCompressedBokehBase64(arrayIn: any, key: string, startIndex: number = -1, earlyExit: boolean = false): any {
    let arrayOut=arrayIn.array
    if(startIndex < 0){
      startIndex = arrayIn.history.length || 0;
    }
    for(var i = startIndex-1;i>=0; i--) {
      console.log((i + 1) + " --> " + arrayIn.history[i])
      const action = Object.prototype.toString.call(arrayIn.history[i]) === '[object String]' ? arrayIn.history[i] : arrayIn.history[i][0]
      const actionParams = Object.prototype.toString.call(arrayIn.history[i]) === '[object String]' ? null : arrayIn.history[i][1]
      if (action == "base64_decode"){
        const s = atob(arrayOut)
        arrayOut = new Uint8Array(s.length)
        for (let j = 0; j < s.length; j++) { 
          arrayOut[j] = s.charCodeAt(j)
        }
      }
      if (action == "inflate") {
        arrayOut = pako.inflate(arrayOut)
        const dtype = arrayIn.dtype
        if(arrayIn.byteorder !== BYTE_ORDER){
          swap(arrayOut.buffer, dtype)
        }
        if (dtype == "int8"){
          arrayOut = new Int8Array(arrayOut.buffer)
        }
        if (dtype == "int16"){
          arrayOut = new Int16Array(arrayOut.buffer)
        }
        if (dtype == "uint16"){
          arrayOut = new Uint16Array(arrayOut.buffer)
        }
        if (dtype == "int32"){
          arrayOut = new Int32Array(arrayOut.buffer)
        }
        if (dtype == "uint32"){
          arrayOut = new Uint32Array(arrayOut.buffer)
        }
        if (dtype == "float32"){
          arrayOut = new Float32Array(arrayOut.buffer)
          arrayOut = new Float64Array(arrayOut)
        }
        if (dtype == "float64"){
          arrayOut = new Float64Array(arrayOut.buffer)
        }
        console.log(arrayOut)
      }
      if (action == "code") {
        let size = arrayOut.length
        let arrayOutNew = new Array(arrayOut.length)
        for (let i = 0; i < size; i++) {
          arrayOutNew[i] = arrayIn.valueCode[arrayOut[i]]
        }
        arrayOut=arrayOutNew
      }
      if (action == "linear") {
        if (actionParams == null || actionParams.origin == null || actionParams.scale == null) {
          console.error("Not enough parameters");
          continue;
        } 
        if(earlyExit){
          arrayIn.arrayOut = arrayOut
          return {"arrayOut": arrayOut, "dtype": arrayIn.dtype, "byteorder": arrayIn.byteorder, "encode": (x: number) => to_fixed_point(x, actionParams.scale, actionParams.origin)}
        }
        const origin = actionParams.origin
        const scale = actionParams.scale
        const dither = actionParams.dither || this.enableDithering
        const arrayOutNew = new Float64Array(arrayOut.length)
        this.invalidateOnDitheringToggle.add(key)
        if(dither){
          for (let i = 0; i < arrayOut.length; i++) {
            arrayOutNew[i] = origin + scale * (arrayOut[i] + Math.random() - .5)
          }
        } else {
          for (let i = 0; i < arrayOut.length; i++) {
            arrayOutNew[i] = origin + scale * arrayOut[i]
          }
        }
        arrayOut = arrayOutNew;
      }
      if (action == "sinh"){
        const mu = actionParams.mu
        const sigma0 = actionParams.sigma0
        const sigma1 = sigma0 / actionParams.sigma1
        const dither = actionParams.dither || this.enableDithering
        let arrayOutNew = new Array(arrayOut.length)
        this.invalidateOnDitheringToggle.add(key)
        if(dither){
          arrayOutNew = (Array.from(arrayOut) as number[]).map((x: number) => sigma1 * Math.sinh(sigma0 * (x + Math.random() - .5) + mu))
        } else {
          arrayOutNew = (Array.from(arrayOut) as number[]).map((x: number) => sigma1 * Math.sinh(sigma0 * x + mu))
        }
        arrayOut = arrayOutNew
      }
    }
    return arrayOut
  }

  encode_number_to_column(_key: string, value: number) {
    return value
  }

  toggle_dithering() {
    for (const key of this.invalidateOnDitheringToggle){
      this.freshColumns.delete(key)
    }
    if(this.invalidateOnDitheringToggle.size > 0){
      this.invalidateOnDitheringToggle.clear()
      this.change.emit()
    }
  }

  get_intermediate_column(key: string) {
    // This is used to get the inflated column before applying transformations
    if (this.intermediateData[key] != null) {
      return this.intermediateData[key].arrayOut
    }
    const arrOut = this.inflateCompressedBokehBase64(this.inputData[key], key, -1, true)
    this.intermediateData[key] = arrOut
    return arrOut
  }

  get_column(key: string){
    if (this.freshColumns.has(key)) {
      return this.data[key];
    }
    const arrOut = this.inflateCompressedBokehBase64(this.inputData[key], key);
    this.data[key]=arrOut;
    this.freshColumns.add(key);
    if(this._length === -1) this._length = arrOut.length
    if(arrOut.length !== this._length){
      throw Error("Corrupted length of column " + key + ": " + arrOut.length + " expected: " + this._length)
    }
    console.log(key);    
    return arrOut;
  }

  get_array(key: string) {
    let column = this.get_column(key)
    if (column == null)
        return []
    else if (!Array.isArray(column))
        return Array.from(column)
    return column;
  }

  get_length(){
    if (this._length !== -1) return this._length
    for(const key in this.inputData){
      this.get_column(key)
      break
    }
    if(this._length === -1) this._length = 0;
    return this._length;
  }

}
