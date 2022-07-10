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

export namespace CDSCompress {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnDataSource.Props & {
      inputData :    p.Property<Record<string, any>>
      sizeMap :    p.Property<Record<string, any>>
  }
}

export interface CDSCompress extends CDSCompress.Attrs {}

export class CDSCompress extends ColumnDataSource {
  properties: CDSCompress.Props

  constructor(attrs?: Partial<CDSCompress.Attrs>) {
    super(attrs)
  }

  static __name__ = "CDSCompress"

  static init_CDSCompress() {

    this.define<CDSCompress.Props>(()=>({
        inputData:    [ p.Instance ],
        sizeMap:    [ p.Instance ]
    }))
  }

  initialize(): void {
    super.initialize()
    //this.data = {}
    console.info("CDSCompress::initialize")
    this.inflateCompressedBokehObjectBase64()
  }
  inflateCompressedBokehBase64(arrayIn: any ) {
    let arrayOut=arrayIn.array
    for(var i =  arrayIn.actionArray.length-1;i>=0; i--) {
      console.log((i + 1) + " --> " + arrayIn.actionArray[i])
      if (arrayIn.actionArray[i][0] == "base64") {
        //arrayOut=Buffer.from(arrayOut, 'base64');
        arrayOut = atob(arrayOut).split("").map(function (x) {
          return x.charCodeAt(0)
        });
      }
      if (arrayIn.actionArray[i][0] == "zip") {
        //arrayOut=Buffer.from(arrayOut, 'base64');
        //var myArr = new Uint8Array(arrayOut)
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
      if (arrayIn.actionArray[i][0] == "code") {
        if(arrayIn.skipCode){
          continue
        }
        let size = arrayOut.length
        let arrayOutNew = new Array()
        for (let i = 0; i < size; i++) {
          arrayOutNew.push(arrayIn.valueCode[arrayOut[i]])
        }
        arrayOut=arrayOutNew
      }
    }
    return arrayOut
    //arrayIn.actionArray.reverse().forEach(x => console.log(x))
  }

  inflateCompressedBokehObjectBase64() {
    this.data = {};
    let objectOut0 = Object;
    let objectOut = (objectOut0 as any);
    let objectIn = (this.inputData as any);
    let length=0;
    for (let arr in objectIn) {
      let arrOut = this.inflateCompressedBokehBase64(objectIn[arr]);
      this.data[arr]=arrOut;
      //if (arrOut.length!=length)
      length=arrOut.length
      console.log(arr);
    }
    this.data["index"]=new Uint32Array(length)
     for (let i = 0; i < length; i++) {
         this.data["index"][i]=i;
        }
    return objectOut
  }

}
