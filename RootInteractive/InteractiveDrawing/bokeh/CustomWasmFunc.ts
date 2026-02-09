import {Model} from "model"
import * as p from "core/properties"

const SENTINEL_OUTPUT = {
  version: "0.0.1",
  pointer: 0,
  args: [],
  length: 0,
  dtype: "error"
}

export namespace CustomWasmFunc {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    parameters: p.Property<Record<string, any>>
    fields: p.Property<Array<string>>
    func: p.Property<string | null>
    module: p.Property<any>
    dtype_out: p.Property<string | null>
  }
}
export interface CustomWasmFunc extends CustomWasmFunc.Attrs {}
export class CustomWasmFunc extends Model {
  properties: CustomWasmFunc.Props

  constructor(attrs?: Partial<CustomWasmFunc.Attrs>) {
    super(attrs)
  }

  static {
    this.define<CustomWasmFunc.Props>(({Any, String, Dict}) => ({
      parameters: [Dict(Any), {}],
      fields: [p.Array(String), []],
      func: [String, null],
      module: [Any, null],
      dtype_out: [String, null]
    }))
  }

  compute(){
    throw new Error("Scalar functions not implemented for WebAssembly")
  }

  v_compute(xs: any[],  data_source: any, output: number | null): {version: string, pointer: number, args: number[], length: number, dtype: string} {
    const func = this.module.exports[this.func]
    const len = data_source.length
    const dtype_out = this.dtype_out
    if (func) {
      let output_ptr = 0
      if (typeof output === "number" && Number.isInteger(output)) {
        // Treat as index into an array
        output_ptr = output
      } else {
        // Create new destination array
        if(dtype_out === "int32" ){
          output_ptr = this.module.exports.malloc(len * 4)
        } else if(dtype_out === "float64"){
          output_ptr = this.module.exports.malloc(len * 8)
        }
        if (output_ptr === 0) {
          console.error("WASM malloc failed")
          return SENTINEL_OUTPUT
        }
      } 
      const xs_copy = []
      for(let i=0; i<xs.length; i++){
        if (typeof xs[i] === "number" && Number.isInteger(xs[i])){
          xs_copy[i] = xs[i]
          if(xs[i] === 0){
            console.error(`Got null pointer as argument ${i}`)
          }
        } else if(xs[i] instanceof Int32Array){
          xs_copy[i] = this.module.exports.malloc(len * 4)
          for(let j=0; j<len; j++){
            this.module.memory[xs_copy[i]+j] = xs[i][j]
          }
        } else if(xs[i] instanceof Float64Array){
          xs_copy[i] = this.module.exports.malloc(len * 8)
          const dst = xs_copy[i]
          const src = xs[i]
          for(let j=0; j<len; j++){
            this.module.memory[dst+j] = src[j]
          }
        }
      }
      const result = func(...xs_copy, output_ptr, len)
      if (result) {
        console.log("WASM function executed successfully")
      } else {
        console.error("WASM function execution failed")
      }
      // Lazily allocates memory, it's consuimer's responsibility to free it. Maybe use reference counting?
      return {version: "0.0.1", pointer: output_ptr, args: xs_copy, length: len, dtype: dtype_out}
    } else {
      console.error("WASM function not found")
      return SENTINEL_OUTPUT
    }
  }
}