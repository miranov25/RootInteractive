import {Model} from "model"
import * as p from "core/properties"

export namespace CustomWasmFunc {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    parameters: p.Property<Record<string, any>>
    fields: p.Property<Array<string>>
    func: p.Property<string | null>
    module: p.Property<any>
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
    }))
  }

  compute(xs: any[],  data_source: any, output: number | any[]): number[] {
    const func = this.module.exports[this.func]
    const len = data_source.length
    if (func) {
      let output_ptr = 0
      if (typeof output === "number" && Number.isInteger(output)) {
        // Treat as index into an array
        output_ptr = output
      } else if (Array.isArray(output)) {
        // Treat as destination array
        output_ptr = this.module.exports.malloc(len * 4)
        if (output_ptr === 0) {
          console.error("WASM malloc failed")
          return []
        }
      } else {
        console.error("Output must be an integer index or an array")
        return []
      }
      const result = func(xs, len, output_ptr)
      if (result) {
        console.log("WASM function executed successfully")
      } else {
        console.error("WASM function execution failed")
      }
    } else {
      console.error("WASM function not found")
      return []
    }
  }
}