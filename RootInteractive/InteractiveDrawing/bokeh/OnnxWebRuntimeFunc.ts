import {Model} from "model"
import * as p from "core/properties"

let ort:any

export namespace OnnxWebRuntimeSession {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    parameters: p.Property<Record<string, any>>
    fields: p.Property<Array<string>>
    v_func: p.Property<string | null>
  }
}

export interface CustomJSNAryFunction extends CustomJSNAryFunction.Attrs {}

export class CustomJSNAryFunction extends Model {
  properties: CustomJSNAryFunction.Props

  constructor(attrs?: Partial<CustomJSNAryFunction.Attrs>) {
    super(attrs)
  }

  static __name__ = "CustomJSNAryFunction"

  static {
    this.define<CustomJSNAryFunction.Props>(({Array, String, Any, Nullable})=>({
      parameters:  [Any, {}],
      fields: [Array(String), []],
      v_func:  [Nullable(String), null],
    }))
  }

  initialize(){
    super.initialize()
    
  }

  async initialize_ort(bytes: Uint8Array) {
    this._session = await ort.InferrenceSession.create(bytes)
  }

  args_keys: Array<string>
  args_values: Array<any>

  effective_fields: Array<string>

  vector_func: Function | null
  _session: any

  connect_signals(): void {
    super.connect_signals()
  }

  update_vfunc(){
    if(!this.v_func){
	    this.vector_func = null
	    return
    }
    this.compute_effective_fields(this.v_func)
    this.args_keys = Object.keys(this.parameters)
    this.args_values = Object.values(this.parameters)
    this.vector_func = new Function(...this.args_keys, ...this.effective_fields, "data_source", "$output",'"use strict";\n'+this.v_func)
    this.change.emit()
  }

  update_args(){
    this.args_keys = Object.keys(this.parameters)
    this.args_values = Object.values(this.parameters)
    this.change.emit()
  }

  v_compute(xs: any[], data_source: any, output: any[] | null =null){
      if(this.vector_func){
          return this.vector_func(...this.args_values, ...xs, data_source, output)
      } else {
          return null
      }
  }

  get_fields():string[]{
    return this.fields
  }

}
