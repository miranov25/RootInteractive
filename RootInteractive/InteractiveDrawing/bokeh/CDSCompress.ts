import {ColumnDataSource} from "models/sources/column_data_source"
import * as p from "core/properties"

import {decodeArray} from "./SerializationUtils"
import {ri_debug} from "./RIDebug"
declare const  pako : any

export namespace CDSCompress {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnDataSource.Props & {
      inputData :    p.Property<any>
      sizeMap :    p.Property<any>
      enableDithering : p.Property<boolean>
      nRows : p.Property<number>
  }
}

function collect_column_deps(actionArray: any[]){
  let deps = new Set<string>()
  for(let i=0; i<actionArray.length; i++){
    let action = actionArray[i]
    if(Object.prototype.toString.call(action) === '[object String]'){
      continue
    }
    const actionParams = action[1]
    action = action[0]
    if(action === "linear" || action === "sinh"){
      if(actionParams.dither == null || actionParams.dither === "toggle"){
        if(ri_debug()) console.debug("adding dither toggle dependency")   // PHASE_5 group 6 (G4 b)
        deps.add("dither")
      }
    }
  }
  return deps
}

export interface CDSCompress extends CDSCompress.Attrs {}

export class CDSCompress extends ColumnDataSource {
  properties: CDSCompress.Props

  constructor(attrs?: Partial<CDSCompress.Attrs>) {
    super(attrs)
  }

  static __name__ = "CDSCompress"

  static {

    this.define<CDSCompress.Props>(({Any, Boolean, Int})=>({
        inputData:    [ Any, {} ],
        sizeMap:    [ Any, {} ],
        enableDithering: [ Boolean, false ],
        nRows: [ Int, -1 ]    // PHASE_5 group 6: the table's own row count, sent by the page builder; -1: unknown
    }))
  }

  _length: number
  intermediateData: Record<string, any>
  freshColumns: Set<string>
  invalidateOnDitheringToggle: Set<string>

  initialize(): void {
    super.initialize()
    if(ri_debug()) console.info("CDSCompress::initialize")
    this.intermediateData = {}
    this.data = {}
    this.freshColumns = new Set<string>()
    this.invalidateOnDitheringToggle = new Set<string>()
    // PHASE_5 group 6: the length is the table's own row count when the page builder sends it; before, the first
    // decoded column set it, so a corrupted first column made every good column be refused
    this._length = this.nRows >= 0 ? this.nRows : -1
  }

  connect_signals(): void {
    super.connect_signals()
    this.connect(this.properties.enableDithering.change, () => this.toggle_dithering())
    // PHASE_5 group 6 (answer G8 a, register TR-D1): new data replace the old. Every decoded column is dropped, so
    // it is decoded again from the new data, and the models that read the table recompute on the change signal.
    // Before, the columns decoded before the replacement kept their old values without a message. New data of
    // another length must come with their row count (nRows); otherwise every column is refused by its length.
    this.connect(this.properties.inputData.change, () => this.replace_data())
    this.connect(this.properties.nRows.change, () => this.replace_data())
  }

  replace_data() {
    this.intermediateData = {}
    this.data = {}
    this.freshColumns.clear()
    this.invalidateOnDitheringToggle.clear()
    this._length = this.nRows >= 0 ? this.nRows : -1
    this.change.emit()
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
    // Function removed during refactor.
    return this.get_column(key)
  }

  get_column(key: string){
    if (this.freshColumns.has(key)) {
      return this.data[key];
    }
    const deps = collect_column_deps(this.inputData[key].decodeProgram);
    const builtins = {"inflate": pako.inflate}
    const env = {"builtins": builtins, "enableDithering": this.enableDithering, "seed": this.name+"_"+key, "dtype": this.inputData[key].dtype, "byteorder": this.inputData[key].byteorder}
    const arrOut = decodeArray(this.inputData[key].array, this.inputData[key].decodeProgram, env)
    if(deps.has("dither")){
      this.invalidateOnDitheringToggle.add(key)
    }
    // PHASE_5 Group 1 (TR-D2): the length is checked before the column is kept; before, a refused column was kept
    // and returned without an error on the next read
    if(this._length === -1) this._length = arrOut.length
    if(arrOut.length !== this._length){
      throw Error("Corrupted length of column " + key + ": " + arrOut.length + " expected: " + this._length)
    }
    this.data[key]=arrOut;
    this.freshColumns.add(key);
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
