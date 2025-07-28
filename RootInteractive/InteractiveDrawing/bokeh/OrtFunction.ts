import {Model} from "model"
import * as p from "core/properties"

declare const ort:any

export namespace OrtFunction {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    parameters: p.Property<Record<string, any>>
    fields: p.Property<Array<string>>
    v_func: p.Property<string>
  }
}

export interface OrtFunction extends OrtFunction.Attrs {}

function transposeFlat(data: any[]): any[] {
  if (data.length === 0) return [];
  const numRows = data.length;
  const numCols = data[0].length;
  const length = numRows * numCols;
  let transposed: any[] = new Array(length).fill(0)
  for (let i = 0; i < numRows; i++) {
    for (let j = 0; j < numCols; j++) {
      transposed[j * numRows + i] = data[i][j];
    }
  }
  return transposed;
}

export class OrtFunction extends Model {
  properties: OrtFunction.Props

  constructor(attrs?: Partial<OrtFunction.Attrs>) {
    super(attrs)
  }

  static __name__ = "OrtFunction"

  static {
    this.define<OrtFunction.Props>(({Array, String, Any})=>({
      parameters:  [Any, {}],
      fields: [Array(String), []],
      v_func:  [String, ""],
    }))
  }

  initialize(){
    super.initialize()
    console.log(ort)
    window.requestAnimationFrame(() => this.initialize_ort(new Uint8Array(atob(this.v_func).split("").map(function (x) {
      return x.charCodeAt(0)
    }))))
    this._dirty_flag = true
  }

  async initialize_ort(bytes: Uint8Array) {
    if(!ort){
      window.requestAnimationFrame(() => this.initialize_ort(bytes))
      return
    }
    this._session = await ort.InferenceSession.create(bytes)
    this.change.emit()
  }

  args_keys: Array<string>
  args_values: Array<any>

  effective_fields: Array<string>

  _session: any

  _results: any
  _results_back: any

  _dirty_flag: boolean

  connect_signals(): void {
    super.connect_signals()
  }

  update_args(){
    this.args_keys = Object.keys(this.parameters)
    this.args_values = Object.values(this.parameters)
    this.change.emit()
  }

  // TODO: Add a get_value function, we want to also support ND functions
  async v_compute(xs: Record<string, any>, _data_source: any, _output: any[] | null =null){
      if(this._session){
        console.log(xs)
        const xs_tensors = Object.keys(xs).reduce((acc: Record<string, any>, key: string) => {
          const data = xs[key]
          if(Array.isArray(data)){
            if(Array.isArray(data[0]) || data[0] instanceof Float32Array || data[0] instanceof Float64Array){
              const flat = transposeFlat(data)
              acc[key] = new ort.Tensor("float32", flat, [data[0].length, data.length])
            } else {
              console.warn("Unsupported data type for key:", key, "Data:", data)
              acc[key] = new ort.Tensor("float32", data, [data.length])
            }
          } 
          return acc}, {})
          try {
            console.log(this._session.inputNames)
            console.log(this._session.outputNames)
            console.log(xs_tensors)
            const new_results = await this._session.run(xs_tensors, ["output_label"])
            console.log(new_results)
            if(new_results["output_label"].data instanceof BigInt64Array){
              return Array.from(new_results["output_label"].data).map(x => Number(x))
            }
            return new_results["output_label"].data as any[]
          } catch (error) {
            console.error("Error during ONNX inference:", error)
            return null
          }

      } else {
          return null
      }
  }

  get_fields():string[]{
    return this.fields
  }

}
