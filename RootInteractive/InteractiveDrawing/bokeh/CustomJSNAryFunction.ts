import {Model} from "model"
import * as p from "core/properties"

// PHASE_5 Group 5 (review finding G4-P1-A): a parameter whose name cannot be a variable of strict-mode JavaScript
// (a reserved word such as default, class, static or eval, or a name such as n-bins) is not passed to the function.
// Before, every such name of the page broke the text query, even when the query did not use it. The JavaScript engine
// itself decides, once per name. A function cannot use such a name in any case.
const bindable_names = new Map<string, boolean>()
function is_bindable(name: string): boolean {
  let ok = bindable_names.get(name)
  if(ok === undefined){
    ok = /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name)
    if(ok){
      try {
        new Function(name, '"use strict";')
      } catch(_error) {
        ok = false
      }
    }
    bindable_names.set(name, ok)
  }
  return ok
}

// the name as a whole word of the code, not as a part of a longer name
function uses_name(code: string, name: string): boolean {
  const escaped = name.replace(/\$/g, "\\$")
  return new RegExp("(^|[^A-Za-z0-9_$])" + escaped + "($|[^A-Za-z0-9_$])").test(code)
}

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

  // the name refused by decision N1 (a name that is both a column and a parameter of this function), or null
  refused: string | null

  effective_fields: Array<string>

  scalar_func: Function | null
  vector_func: Function | null

  connect_signals(): void {
    super.connect_signals()
  }

  update_func(){
    if(!this.func){
	    this.scalar_func = null
	    if(!this.v_func) this.refused = null
	    return
    }
    this.compute_effective_fields(this.func)
    this.set_args()
    if(this.check_refused(this.func)){
      this.scalar_func = null
    } else {
      this.scalar_func = new Function(...this.args_keys, ...this.effective_fields, '"use strict";\n'+this.func)
    }
    this.change.emit()
  }

  compute(x: any[]){
    if(this.refused != null){
      return NaN
    }
    return this.scalar_func!(...this.args_values, ...x)
  }

  update_vfunc(){
    if(!this.v_func){
	    this.vector_func = null
	    return
    }
    this.compute_effective_fields(this.v_func)
    this.set_args()
    if(this.check_refused(this.v_func)){
      this.vector_func = null
    } else {
      this.vector_func = new Function(...this.args_keys, ...this.effective_fields, "data_source", "$output",'"use strict";\n'+this.v_func)
    }
    this.change.emit()
  }

  update_args(){
    this.set_args()
    this.change.emit()
  }

  set_args(){
    // PHASE_5 Group 4 (TQ-D1): a parameter with the name of one of the function's fields is not passed as a second
    // argument: the function is built in strict mode, which refuses two arguments with one name. A function that
    // uses such a name is refused (check_refused, PHASE_5 Group 5). Group 5 (G4-P1-A): only names that can be
    // variables of strict-mode JavaScript are passed.
    const fields = new Set(this.effective_fields ?? [])
    this.args_keys = Object.keys(this.parameters).filter((k) => !fields.has(k) && is_bindable(k))
    this.args_values = this.args_keys.map((k) => this.parameters[k])
  }

  check_refused(code: string): boolean {
    // PHASE_5 Group 5, the Architect's decision N1 (review finding G4-POL-1): a function that uses a name that is
    // both a column (a field) and a parameter of the function is refused by name. It gives NaN for every row (a
    // query then selects nothing), and the error names the function and the name. Before (Group 4) the column won
    // without a message. A name that is both, but not used by the function, changes nothing.
    const fields = new Set(this.effective_fields ?? [])
    this.refused = null
    for(const k of Object.keys(this.parameters)){
      if(fields.has(k) && is_bindable(k) && uses_name(code, k)){
        this.refused = k
        break
      }
    }
    if(this.refused != null){
      console.error(new Error(`CustomJSNAryFunction: the name "${this.refused}" is both a column and a parameter ` +
        `of the function "${code}"; the function is refused and gives NaN; rename the column or the parameter`))
      return true
    }
    return false
  }

  v_compute(xs: any[], data_source: any, output: any[] | null =null){
      if(this.refused != null){
        return null             // the column is computed row by row, by compute: NaN
      }
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
