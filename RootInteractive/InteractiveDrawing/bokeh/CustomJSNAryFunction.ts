import {Model} from "model"
import * as p from "core/properties"

export namespace CustomJSNAryFunction {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    parameters: p.Property<Record<string, any>>
    fields: p.Property<Array<string>>
    func: p.Property<string>
    v_func: p.Property<string>
  }
}

export interface CustomJSNAryFunction extends CustomJSNAryFunction.Attrs {}

export class CustomJSNAryFunction extends Model {
  properties: CustomJSNAryFunction.Props

  constructor(attrs?: Partial<CustomJSNAryFunction.Attrs>) {
    super(attrs)
  }

  static __name__ = "CustomJSNAryFunction"

  static init_CustomJSNAryFunction() {
    this.define<CustomJSNAryFunction.Props>(({Array, String})=>({
      parameters:  [p.Instance, {}],
      fields: [Array(String), []],
      func:    [ String ],
      v_func:  [String]
    }))
  }

  initialize(){
    super.initialize()
    this.update_func()
    this.update_vfunc()
  }

  args_keys: Array<string>
  args_values: Array<any>

  scalar_func: Function
  vector_func: Function

  connect_signals(): void {
    super.connect_signals()
  }

  update_func(){
    this.args_keys = Object.keys(this.parameters)
    this.args_values = Object.values(this.parameters)
    this.scalar_func = new Function(...this.args_keys, ...this.fields, '"use strict";\n'+this.func)
    this.change.emit()
  }

  compute(x: any[]){
    return this.scalar_func!(...this.args_values, ...x)
  }

  update_vfunc(){
    this.args_keys = Object.keys(this.parameters)
    this.args_values = Object.values(this.parameters)
    this.vector_func = new Function(...this.args_keys, ...this.fields, "data_source", "$output",'"use strict";\n'+this.v_func)
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

}
