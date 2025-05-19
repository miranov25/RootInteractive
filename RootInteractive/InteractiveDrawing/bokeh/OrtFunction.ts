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
  async v_compute(xs: any[], _data_source: any, _output: any[] | null =null){
      if(this._session){
         // const feeds = Object.fromEntries(this.fields.map((name:string, i:number) => [name, xs[i]]))
          const new_results = await this._session.run(xs)
          console.log(new_results)
          return new_results
      } else {
          return null
      }
  }

  get_fields():string[]{
    return this.fields
  }

}
