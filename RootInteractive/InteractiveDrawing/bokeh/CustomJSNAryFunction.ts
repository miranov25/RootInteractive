import {Model} from "model"
import * as p from "core/properties"

// PHASE_5 Group 5 (review finding G4-P1-A): a parameter whose name cannot be a variable of strict-mode JavaScript
// (a reserved word such as default, class, static or eval, or a name such as n-bins) is not passed to the function.
// Before, every such name of the page broke the text query, even when the query did not use it. A name must be plain
// ASCII (the Architect's rule of 2026-11-18; the page builder refuses other parameter names), and the JavaScript
// engine itself decides, once per name. A function that uses such a name is refused by name (refusal, Group 5a).
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

// PHASE_5 Group 5a (review finding F2): the names that a function's code uses as variables. The text of comments, of
// strings and of template literals is not a use, nor is a name after a dot (a property, as in data.cut); the code
// inside ${...} of a template literal is. Names are read with Unicode's identifier classes, so that a name such as
// "σcut" is not mistaken for "cut". Two known limits, both on the safe side (the name counts as used, and a function
// may be refused by name, never computed wrongly): a name written as an object key ({cut: 1}), and a name inside a
// regular-expression literal (/cut/).
const ID_START = /[\p{ID_Start}$_]/u
const ID_PART = /[\p{ID_Continue}$‌‍]/u
export function used_names(code: string): Set<string> {
  const names = new Set<string>()
  const n = code.length
  let i = 0
  const char_at = (k: number) => String.fromCodePoint(code.codePointAt(k)!)
  const scan = (in_template: boolean): void => {
    let depth = 0
    let after_dot = false
    while(i < n){
      const c = code[i]
      if(c === "/" && code[i+1] === "/"){
        const end = code.indexOf("\n", i)
        i = end < 0 ? n : end
        continue
      }
      if(c === "/" && code[i+1] === "*"){
        const end = code.indexOf("*/", i+2)
        i = end < 0 ? n : end+2
        continue
      }
      if(c === "'" || c === '"'){
        for(++i; i < n && code[i] !== c; ++i){
          if(code[i] === "\\") ++i
        }
        ++i
        after_dot = false
        continue
      }
      if(c === "`"){
        for(++i; i < n && code[i] !== "`";){
          if(code[i] === "\\"){
            i += 2
          } else if(code[i] === "$" && code[i+1] === "{"){
            i += 2
            scan(true)
          } else {
            ++i
          }
        }
        ++i
        after_dot = false
        continue
      }
      if(in_template && c === "{"){
        ++depth
      } else if(in_template && c === "}"){
        if(depth === 0){
          ++i
          return
        }
        --depth
      }
      if(c === "." && code.startsWith("...", i)){
        i += 3
        after_dot = false
        continue
      }
      if((c >= "0" && c <= "9") || (c === "." && code[i+1] >= "0" && code[i+1] <= "9")){
        // a number with its letters and point (1e5, 0x1f, 1n, .5)
        while(i < n && /[0-9A-Za-z_.]/.test(code[i])) ++i
        after_dot = false
        continue
      }
      if(c === "."){
        after_dot = true
        ++i
        continue
      }
      const ch = char_at(i)
      if(ID_START.test(ch)){
        let j = i + ch.length
        while(j < n){
          const next = char_at(j)
          if(!ID_PART.test(next)) break
          j += next.length
        }
        if(!after_dot) names.add(code.slice(i, j))
        i = j
        after_dot = false
        continue
      }
      if(!/\s/.test(ch)) after_dot = false
      i += ch.length
    }
  }
  scan(false)
  return names
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

  // the name for which the scalar or the vector function is refused (PHASE_5 Groups 5 and 5a), or null
  refused_func: string | null
  refused_vfunc: string | null

  // the names of the parameters when the functions were last built (Group 5a, F3)
  parameter_keys: string

  effective_fields: Array<string>

  scalar_func: Function | null
  vector_func: Function | null

  connect_signals(): void {
    super.connect_signals()
  }

  update_func(){
    this.parameter_keys = Object.keys(this.parameters).join("\u0000")
    if(!this.func){
	    this.scalar_func = null
	    this.refused_func = null
	    return
    }
    this.compute_effective_fields(this.func)
    this.set_args()
    this.refused_func = this.refusal(this.func)
    if(this.refused_func != null){
      this.scalar_func = null
    } else {
      this.scalar_func = new Function(...this.args_keys, ...this.effective_fields, '"use strict";\n'+this.func)
    }
    this.change.emit()
  }

  compute(x: any[]){
    if(this.refused_func != null){
      return NaN
    }
    return this.scalar_func!(...this.args_values, ...x)
  }

  update_vfunc(){
    this.parameter_keys = Object.keys(this.parameters).join("\u0000")
    if(!this.v_func){
	    this.vector_func = null
	    this.refused_vfunc = null
	    return
    }
    this.compute_effective_fields(this.v_func)
    this.set_args()
    this.refused_vfunc = this.refusal(this.v_func)
    if(this.refused_vfunc != null){
      this.vector_func = null
    } else {
      this.vector_func = new Function(...this.args_keys, ...this.effective_fields, "data_source", "$output",'"use strict";\n'+this.v_func)
    }
    this.change.emit()
  }

  update_args(){
    // PHASE_5 Group 5a (review finding F3): when the parameters have other names than when the functions were built,
    // the functions are built again, which checks the refusals again; otherwise only the values change.
    if(Object.keys(this.parameters).join("\u0000") !== this.parameter_keys){
      this.update_func()
      this.update_vfunc()
      return
    }
    this.set_args()
    this.change.emit()
  }

  set_args(){
    // PHASE_5 Group 4 (TQ-D1): a parameter with the name of one of the function's fields is not passed as a second
    // argument: the function is built in strict mode, which refuses two arguments with one name. A function that
    // uses such a name is refused (refusal, PHASE_5 Group 5). Group 5 (G4-P1-A): only names that can be
    // variables of strict-mode JavaScript are passed.
    const fields = new Set(this.effective_fields ?? [])
    this.args_keys = Object.keys(this.parameters).filter((k) => !fields.has(k) && is_bindable(k))
    this.args_values = this.args_keys.map((k) => this.parameters[k])
  }

  refusal(code: string): string | null {
    // PHASE_5 Group 5, the Architect's decision N1 (review finding G4-POL-1): a function that uses a name that is
    // both a column (a field) and a parameter of the function is refused by name. Group 5a (F1): so is a function that
    // uses a parameter that cannot be passed to it (a name that is not a plain ASCII JavaScript name). A refused
    // function gives NaN for every row (a query then selects nothing), and one error names the function and the
    // name. A name that is not used (Group 5a, F2: see used_names) changes nothing.
    const used = used_names(code)
    const fields = new Set(this.effective_fields ?? [])
    for(const k of Object.keys(this.parameters)){
      if(!used.has(k)) continue
      if(fields.has(k)){
        console.error(new Error(`CustomJSNAryFunction: the name "${k}" is both a column and a parameter of the ` +
          `function "${code}"; the function is refused and gives NaN; rename the column or the parameter`))
        return k
      }
      if(!is_bindable(k)){
        console.error(new Error(`CustomJSNAryFunction: the parameter "${k}" cannot be used in the function ` +
          `"${code}": a parameter used in a function needs a plain ASCII JavaScript name; the function is ` +
          `refused and gives NaN`))
        return k
      }
    }
    return null
  }

  v_compute(xs: any[], data_source: any, output: any[] | null =null){
      if(this.refused_vfunc != null){
        const len = xs.length > 0 ? xs[0].length : (data_source?.get_length?.() ?? 0)
        const out = output != null && output.length === len ? output : new Array(len)
        for(let i=0; i < len; ++i) out[i] = NaN
        return out
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
