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
    this.initialize_ort(new Uint8Array(atob(this.v_func).split("").map(function (x) {
      return x.charCodeAt(0)
    })))
  }

  async initialize_ort(bytes: Uint8Array) {
    this._session = await ort.InferrenceSession.create(bytes)
    this.change.emit()
  }

  args_keys: Array<string>
  args_values: Array<any>

  effective_fields: Array<string>

  _session: any

  _results: any
  _results_back: any

  connect_signals(): void {
    super.connect_signals()
  }

  update_args(){
    this.args_keys = Object.keys(this.parameters)
    this.args_values = Object.values(this.parameters)
    this.change.emit()
  }

  async actually_compute(feeds){
    this._results = await this._session.run(feeds)
    this.change.emit()
  }

  // TODO: Add a get_value function, we want to also support ND functions
  v_compute(xs: any[], _data_source: any, _output: any[] | null =null){
      if(this._session){
          const feeds = Object.fromEntries(this.fields.map((name:string, i:number) => [name, xs[i]]))
          this.actually_compute(feeds)
          return this._results
      } else {
          return null
      }
  }

  get_fields():string[]{
    return this.fields
  }

}
