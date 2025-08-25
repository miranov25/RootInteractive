import {Model} from "model"
import * as p from "core/properties"

export namespace CustomJSNAryFunction {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    parameters: p.Property<Record<string, any>>
    fields: p.Property<Array<string>>
    func: p.Property<string | null>
    v_func: p.Property<string | null>
    auto_fields:p.Property<boolean>
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
    this.define<CustomJSNAryFunction.Props>(({Array, String, Any, Nullable, Boolean, Int})=>({
      parameters:  [Any, {}],
      fields: [Array(String), []],
      func:    [ Nullable(String), null ],
      v_func:  [Nullable(String), null],
      auto_fields: [Boolean, false],
      n_out: [Int, 1]
    }))
  }

  initialize(){
    super.initialize()
    this.update_func()
    this.update_vfunc()
  }

  args_keys: Array<string>
  args_values: Array<any>

  effective_fields: Array<string>

  scalar_func: Function | null
  vector_func: Function | null

  connect_signals(): void {
    super.connect_signals()
  }

  update_func(){
    if(!this.func){
	    this.scalar_func = null
	    return
    }
    this.compute_effective_fields(this.func)
    this.args_keys = Object.keys(this.parameters)
    this.args_values = Object.values(this.parameters)
    this.scalar_func = new Function(...this.args_keys, ...this.effective_fields, '"use strict";\n'+this.func)
    this.change.emit()
  }

  compute(x: any[]){
    return this.scalar_func!(...this.args_values, ...x)
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

  compute_effective_fields(search:string){
    if(!this.auto_fields){
      this.effective_fields = this.fields
      return
    }
    this.effective_fields = []
    for (const field of this.fields) {
      if(search.includes(field)){
        this.effective_fields.push(field)
      }
    }
  }

  get_fields():string[]{
    return this.effective_fields
  }

}
