import {Model} from "model"
import * as p from "core/properties"

// PHASE_5 Group 5 (review finding G4-P1-A): a parameter whose name cannot be a variable of strict-mode JavaScript
// (a reserved word such as default, class, static or eval, or a name such as n-bins) is not passed to the function.
// Before, every such name of the page broke the text query, even when the query did not use it. A name must be plain
// ASCII (the Architect's rule of 2026-11-18; the page builder refuses other parameter names), and the JavaScript
// engine itself decides, once per name. A function that uses such a name is refused by name (refusal, Group 5a).
// PHASE_5 Group 6 (the review of group 5d, F1; the Architect, 2026-11-21: "Column names has to be plain ascii. We
// should force it."): the same rule holds for columns, and a function that uses a column whose name is not plain
// ASCII, such as σcut, is refused by name (refusal). Before, such a column was left out, and the function failed with
// a ReferenceError.
const ASCII_NAME = /^[A-Za-z_$][A-Za-z0-9_$]*$/
const bindable_names = new Map<string, boolean>()
function is_bindable(name: string): boolean {
  let ok = bindable_names.get(name)
  if(ok === undefined){
    ok = ASCII_NAME.test(name)
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

// PHASE_5 Group 6 (the review of group 5d: Claude10.RI F1; GPT33.RI's summary F2, F4 and F5): a function that
// JavaScript cannot build. The causes: a typing error in the code (a text query typed by the user); a name given twice
// (a column or a parameter named data_source or $output, the arguments of the vector function, or a column listed
// twice); a column listed by hand whose name cannot be a variable. Before, the error left build() and the only sign
// was an unhandled rejection in the page. Now one console error says why, and build() decides what happens: the
// functions in force before stay in force (F4, a text query keeps its previous selection), or, when there are none,
// the function is refused and gives NaN. Returns the function, or the cause as a text.
function build_function(names: string[], body: string): Function | string {
  const twice = names.find((name, k) => names.indexOf(name) !== k)
  let cause: string
  if(twice !== undefined){
    cause = `the name "${twice}" is given to it twice (a column or a parameter named data_source or $output, ` +
      `the arguments of the vector function, or a column listed twice)`
  } else {
    try {
      return new Function(...names, '"use strict";\n' + body)
    } catch(error) {
      cause = String(error)
    }
  }
  return cause
}

// Group 6 (F4): what build() made from the bodies func and v_func, kept while a later body cannot be built
type InForce = {func: string | null, v_func: string | null, parameter_keys: string, effective_fields: Array<string>,
                scalar_func: Function | null, vector_func: Function | null,
                refused_func: string | null, refused_vfunc: string | null}

// PHASE_5 Group 5b (review findings G5a-F2-A and G5a-F2-B): the names that a function's code uses as variables, read
// with the lexical rules of JavaScript. Not a use: the text of comments, of strings, of template literals and of
// regular expressions, and a name after a dot (a property, as in data.cut). A use: a name in the code, also inside
// ${...} of a template literal, also written with escapes (\u0041 and \u{41} are the name A). A "/" starts a
// regular expression where a value is expected (after an operator, "(", ",", a word such as return, or the ")" of
// if, while, for and with), and is a division after a value. Where JavaScript's rules cannot be decided from the
// text alone (a "/" after "}", after the words of, await, yield, break and continue or the label after these two;
// group 5c: code with the marks <!-- or --> of HTML-like comments) or the code is not valid, the reading is not sure,
// and every name anywhere in the text counts as used: the function may then be refused by name, never computed with
// a name that was missed. A function that uses eval or arguments may read any name (group 5c, reads_any_name).
// Group 5d: such a function gets every column whose name can be a variable (compute_effective_fields).
// Known extra uses, all on the safe side: object keys ({cut: 1}), labels, and the names that a function declares
// itself.
const ID_START = /[\p{ID_Start}$_]/u
const ID_PART = /[\p{ID_Continue}$\u200C\u200D]/u
const ESCAPE = /\\u(?:([0-9A-Fa-f]{4})|\{([0-9A-Fa-f]+)\})/y
const ANY_NAME = /(?:[\p{ID_Start}$_]|\\u[0-9A-Fa-f]{4}|\\u\{[0-9A-Fa-f]+\})(?:[\p{ID_Continue}$\u200C\u200D]|\\u[0-9A-Fa-f]{4}|\\u\{[0-9A-Fa-f]+\})*/gu
const REGEX_AFTER_WORD = new Set(["return", "typeof", "instanceof", "in", "new", "delete", "void", "throw", "case",
                                  "do", "else", "extends", "debugger"])
// Group 5c: after break and continue, and after their label, a "/" on the next line starts a regular expression and
// on the same line is not valid; the reading is then not sure, on the safe side.
const UNSURE_AFTER_WORD = new Set(["of", "await", "yield", "break", "continue"])
// Group 5c (review of group 5b, Claude10.RI F2 and F3): eval reads the function's own variables from a string, whose
// names may even be built at run time, and arguments reads them by position. A function that uses one of them may
// read any name: see reads_any_name.
const READS_ANY_NAME = ["eval", "arguments"]
const REGEX_AFTER_PAREN_OF = new Set(["if", "while", "for", "with"])
type Next = "regex" | "division" | "unsure"     // what a "/" means at this point of the code

function is_line_end(c: string): boolean {
  return c === "\n" || c === "\r" || c === "\u2028" || c === "\u2029"
}

function decode_escapes(name: string): string {
  return name.replace(/\\u(?:([0-9A-Fa-f]{4})|\{([0-9A-Fa-f]+)\})/g, (_m, four, braced) => {
    const cp = parseInt(four ?? braced, 16)
    return cp <= 0x10FFFF ? String.fromCodePoint(cp) : ""
  })
}

export function used_names(code: string): Set<string> {
  const names = new Set<string>()
  const n = code.length
  let i = 0
  let unsure = false
  const char_at = (k: number) => String.fromCodePoint(code.codePointAt(k)!)

  // a name at i, with its escapes decoded; null (i unchanged) if an escape is not valid there
  const read_name = (): string | null => {
    let name = ""
    while(i < n){
      let ch: string
      let len: number
      if(code[i] === "\\"){
        ESCAPE.lastIndex = i
        const m = ESCAPE.exec(code)
        const cp = m == null ? -1 : parseInt(m[1] ?? m[2], 16)
        if(m == null || cp > 0x10FFFF) return name === "" ? null : name
        ch = String.fromCodePoint(cp)
        len = m[0].length
        if(!(name === "" ? ID_START : ID_PART).test(ch)) return name === "" ? null : name
      } else {
        ch = char_at(i)
        len = ch.length
        if(!(name === "" ? ID_START : ID_PART).test(ch)) break
      }
      name += ch
      i += len
    }
    return name
  }

  // a string literal from its opening quote at i
  const read_string = (quote: string): void => {
    for(++i; i < n; ++i){
      const c = code[i]
      if(c === "\\"){
        ++i
        if(code[i] === "\r" && code[i+1] === "\n") ++i
      } else if(c === quote){
        ++i
        return
      } else if(c === "\n" || c === "\r"){
        break
      }
    }
    unsure = true
  }

  // a regular-expression literal from its "/" at i, then its flags
  const read_regex = (): void => {
    let in_class = false
    for(++i; ; ++i){
      if(i >= n || is_line_end(code[i])){
        unsure = true
        return
      }
      const c = code[i]
      if(c === "\\"){
        ++i
        if(i >= n || is_line_end(code[i])){
          unsure = true
          return
        }
      } else if(c === "["){
        in_class = true
      } else if(c === "]"){
        in_class = false
      } else if(c === "/" && !in_class){
        ++i
        break
      }
    }
    while(i < n){
      const ch = char_at(i)
      if(!ID_PART.test(ch)) break
      i += ch.length
    }
  }

  // a template literal from its "`" at i; the code inside ${...} is read as code
  const read_template = (): void => {
    for(++i; i < n;){
      const c = code[i]
      if(c === "\\"){
        i += 2
      } else if(c === "`"){
        ++i
        return
      } else if(c === "$" && code[i+1] === "{"){
        i += 2
        if(!scan(true)) return
      } else {
        ++i
      }
    }
    unsure = true
  }

  // code up to the end, or (in_template) up to the "}" that closes ${; false if that "}" is missing
  const scan = (in_template: boolean): boolean => {
    let next: Next = "regex"
    let depth = 0
    let after_dot = false
    let word = ""                       // the word just read, for the "(" of if, while, for and with
    const parens: Next[] = []           // for each open "(": what a "/" after its ")" means
    while(i < n){
      const c = code[i]
      if(/\s/.test(c)){
        ++i
        continue
      }
      if(c === "/" && code[i+1] === "/"){
        for(i += 2; i < n && !is_line_end(code[i]); ++i);
        continue
      }
      if(c === "/" && code[i+1] === "*"){
        const end = code.indexOf("*/", i+2)
        if(end < 0) unsure = true
        i = end < 0 ? n : end+2
        continue
      }
      const was_word = word
      word = ""
      if(c === "/"){
        if(next === "regex"){
          read_regex()
          next = "division"
        } else {
          if(next === "unsure") unsure = true
          ++i
          next = "regex"
        }
        after_dot = false
        continue
      }
      if(c === "'" || c === '"'){
        read_string(c)
        next = "division"
        after_dot = false
        continue
      }
      if(c === "`"){
        read_template()
        next = "division"
        after_dot = false
        continue
      }
      if(c === "." && code.startsWith("...", i)){
        i += 3
        next = "regex"
        after_dot = false
        continue
      }
      if((c >= "0" && c <= "9") || (c === "." && code[i+1] >= "0" && code[i+1] <= "9")){
        // a number with its letters and points (1e5, 0x1f, 1n, 1_000, .5, and 1..p, whose p is a property)
        while(i < n && /[0-9A-Za-z_.]/.test(code[i])) ++i
        next = "division"
        after_dot = false
        continue
      }
      if(c === "."){
        ++i
        after_dot = true
        next = "regex"
        continue
      }
      if(c === "\\" || ID_START.test(char_at(i))){
        const name = read_name()
        if(name == null){
          unsure = true
          ++i
          next = "regex"
          after_dot = false
          continue
        }
        if(after_dot){
          next = "division"
        } else {
          names.add(name)
          word = name
          next = REGEX_AFTER_WORD.has(name) ? "regex"
            : UNSURE_AFTER_WORD.has(name) || was_word === "break" || was_word === "continue" ? "unsure" : "division"
        }
        after_dot = false
        continue
      }
      after_dot = false
      if(c === "("){
        parens.push(REGEX_AFTER_PAREN_OF.has(was_word) ? "regex" : was_word === "await" ? "unsure" : "division")
        next = "regex"
      } else if(c === ")"){
        const after = parens.pop()
        if(after === undefined) unsure = true
        next = after ?? "unsure"
      } else if(c === "]"){
        next = "division"
      } else if(c === "{"){
        ++depth
        next = "regex"
      } else if(c === "}"){
        if(in_template && depth === 0){
          ++i
          return true
        }
        --depth
        next = "unsure"                 // the end of a block (a "/" then starts a regular expression) or of a value
      } else if((c === "+" || c === "-") && code[i+1] === c){
        i += 2
        next = "division"               // a "/" after x++ divides; a regular expression cannot follow a prefix ++
        continue
      } else {
        next = "regex"                  // an operator or a punctuator: , ; : ? = ! ~ & | ^ < > * % + - # @
      }
      ++i
    }
    if(in_template) unsure = true
    return false
  }

  // Group 5c (review of group 5b, Claude10.RI F1): <!-- anywhere and --> at the start of a line are line comments in
  // a browser (HTML-like comments of JavaScript); where they may occur, the reading is not sure, on the safe side.
  if(code.includes("<!--") || code.includes("-->")) unsure = true
  scan(false)
  if(unsure){
    for(const m of code.matchAll(ANY_NAME)) names.add(decode_escapes(m[0]))
  }
  return names
}

export function reads_any_name(used: Set<string>): boolean {
  return READS_ANY_NAME.some((name) => used.has(name))
}

export namespace CustomJSNAryFunction {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    parameters: p.Property<Record<string, any>>
    fields: p.Property<Array<string>>
    func: p.Property<string | null>
    v_func: p.Property<string | null>
    auto_fields:p.Property<boolean>
    problem: p.Property<string>
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
      n_out: [Int, 1],
      // PHASE_5 Group 6c (TQ-D3): why the text is not in force or computes nothing; "" when it works. The page shows
      // it on the text field (bokehTools.py); it is set only here, without the change signal of the model.
      problem: [String, ""]
    }))
  }

  initialize(){
    super.initialize()
    this.build()
  }

  args_keys: Array<string>
  args_values: Array<any>

  // the name for which the scalar or the vector function is refused (PHASE_5 Groups 5 and 5a), or why it cannot be
  // built (Group 6), or null
  refused_func: string | null
  refused_vfunc: string | null

  // Group 6 (F4): the functions in force, or undefined before a build in which every function could be built. Declared
  // only, with no value here: Bokeh builds the model from its constructor, before a class field would be set.
  in_force: InForce | undefined

  // Group 6c (TQ-D3): the problem of the last build and of the last computation, and why refusal() refused
  build_problem: string | undefined
  compute_problem: string | undefined
  refusal_reason: string | undefined

  // the names of the parameters when the functions were last built (Group 5a, F3; Group 5b: built together)
  parameter_keys: string

  effective_fields: Array<string>

  scalar_func: Function | null
  vector_func: Function | null

  connect_signals(): void {
    super.connect_signals()
  }

  update_func(){
    this.build()
  }

  update_vfunc(){
    this.build()
  }

  build(){
    // PHASE_5 Group 5b (review finding G5a-F3-P2): the scalar and the vector function are built together, with one
    // list of fields and one list of arguments, chosen for both bodies; before, the function built last chose them,
    // and with automatic fields the other function could then be called with the columns of the wrong names.
    this.parameter_keys = Object.keys(this.parameters).join("\u0000")
    this.compute_effective_fields([this.func, this.v_func].filter((body): body is string => !!body))
    this.set_args()
    // Group 6: a function that JavaScript cannot build (build_function)
    const kept = this.in_force
    const names = [...this.args_keys, ...this.effective_fields]
    const not_built: Array<[string, string]> = []       // the body, and why it cannot be built
    this.refused_func = this.func ? this.refusal(this.func) : null
    this.scalar_func = null
    if(this.func && this.refused_func == null){
      const built = build_function(names, this.func)
      if(typeof built === "string"){
        this.refused_func = built
        not_built.push([this.func, built])
      } else this.scalar_func = built
    }
    this.refused_vfunc = this.v_func ? this.refusal(this.v_func) : null
    this.vector_func = null
    if(this.v_func && this.refused_vfunc == null){
      const built = build_function([...names, "data_source", "$output"], this.v_func)
      if(typeof built === "string"){
        this.refused_vfunc = built
        not_built.push([this.v_func, built])
      } else this.vector_func = built
    }
    // Group 6 (F4, the Architect, 2026-11-21: "Keep the previous selections."): when a body cannot be built, the
    // functions in force before stay in force, with their columns, and follow their parameters; a text query keeps its
    // previous selection. One console error says why. With no functions in force before (a model whose first build
    // fails), or with parameters of other names, the function is refused by name, as rule N1 refuses a shared name:
    // NaN for every row (a query then selects nothing). A function refused by N1 is built as before, and is in force.
    if(not_built.length > 0 && kept !== undefined && kept.parameter_keys === this.parameter_keys){
      this.effective_fields = kept.effective_fields
      this.scalar_func = kept.scalar_func
      this.vector_func = kept.vector_func
      this.refused_func = kept.refused_func
      this.refused_vfunc = kept.refused_vfunc
      this.set_args()
      for(const [body, cause] of not_built)
        console.error(new Error(`CustomJSNAryFunction: a function cannot be built: ${cause}; the previous function ` +
          `is kept, and the result does not change; the function: "${body}"; the previous: ` +
          `"${body === this.v_func ? kept.v_func : kept.func}"`))
      this.set_build_problem(`not used: ${not_built[0][1]}; the previous one is kept`)
      return
    }
    for(const [body, cause] of not_built)
      console.error(new Error(`CustomJSNAryFunction: a function cannot be built: ${cause}; it is refused and gives ` +
        `NaN; the function: "${body}"`))
    if(not_built.length === 0)
      this.in_force = {func: this.func, v_func: this.v_func, parameter_keys: this.parameter_keys,
                       effective_fields: this.effective_fields, scalar_func: this.scalar_func,
                       vector_func: this.vector_func, refused_func: this.refused_func,
                       refused_vfunc: this.refused_vfunc}
    // Group 6c (TQ-D3): what the page shows on the text field; a new function has not been computed yet
    this.compute_problem = ""
    if(not_built.length > 0)
      this.set_build_problem(`not used: ${not_built[0][1]}; no result (NaN)`)
    else if(this.refused_func != null || this.refused_vfunc != null)
      this.set_build_problem(`refused: ${this.refusal_reason ?? "see the console"}; no result (NaN)`)
    else
      this.set_build_problem("")
    this.change.emit()
  }

  // Group 6c (TQ-D3, the Architect, 2026-11-21: "red frame or something similar"): the property problem, which the
  // page shows on the text field, is the problem of the build, else that of the last computation. It is set without
  // the change signal of the model (no_change), so that showing it does not compute the column again.
  set_build_problem(text: string){
    this.build_problem = text
    this.show_problem()
  }

  set_compute_problem(error: unknown, kept: boolean){
    this.compute_problem = error == null ? "" :
      `error while computing: ${String(error)}; ${kept ? "the previous result is kept" : "no result (NaN)"}`
    this.show_problem()
  }

  show_problem(){
    const text = this.build_problem || this.compute_problem || ""
    if(this.problem !== text)
      this.setv({problem: text}, {no_change: true})
  }

  compute(x: any[]){
    if(this.refused_func != null){
      return NaN
    }
    return this.scalar_func!(...this.args_values, ...x)
  }

  update_args(){
    // PHASE_5 Group 5a (review finding F3): when the parameters have other names than when the functions were built,
    // the functions are built again, which checks the refusals again; otherwise only the values change.
    if(Object.keys(this.parameters).join("\u0000") !== this.parameter_keys){
      this.build()
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
    // name. A name that is not used (Group 5a, F2: see used_names) changes nothing. Group 5b (G5a-F2-A): the columns
    // are all the fields the function may get, also those that automatic fields leave out because the code writes
    // the name with escapes (\u0041 for A).
    // Group 5c (review of group 5b): a function that uses eval or arguments may read any name, so every name that is
    // both a column and a parameter counts as used by it.
    const used = used_names(code)
    const any = reads_any_name(used)
    const fields = new Set([...(this.fields ?? []), ...(this.effective_fields ?? [])])
    for(const k of Object.keys(this.parameters)){
      if(!used.has(k) && !(any && fields.has(k))) continue
      if(fields.has(k)){
        console.error(new Error(`CustomJSNAryFunction: the name "${k}" is both a column and a parameter of the ` +
          `function "${code}"; the function is refused and gives NaN; rename the column or the parameter`))
        this.refusal_reason = `the name "${k}" is both a column and a parameter`
        return k
      }
      if(!is_bindable(k)){
        console.error(new Error(`CustomJSNAryFunction: the parameter "${k}" cannot be used in the function ` +
          `"${code}": a parameter used in a function needs a plain ASCII JavaScript name; the function is ` +
          `refused and gives NaN`))
        this.refusal_reason = `the parameter "${k}" has no plain ASCII name`
        return k
      }
    }
    // Group 6 (F1, the Architect, 2026-11-21: column names are plain ASCII, and this is enforced): a function that
    // uses a column whose name is not plain ASCII, such as σcut, is refused by name. The page builder lists such a
    // column for a function with automatic columns only for this check (bokehTools.py); it is never given to it.
    for(const field of this.fields ?? []){
      if(used.has(field) && !ASCII_NAME.test(field)){
        console.error(new Error(`CustomJSNAryFunction: the column "${field}" cannot be used in the function ` +
          `"${code}": a column used in a function needs a plain ASCII name; the function is refused and gives NaN; ` +
          `rename the column`))
        this.refusal_reason = `the column "${field}" has no plain ASCII name`
        return field
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

  compute_effective_fields(bodies: string[]){
    // Group 5c (review of group 5b, GPT32.RI and GPT33.RI): a field is given when a body's text contains its name, as
    // before, or when a body uses it written with escapes (\u0041 for A); every field is given to a body that may
    // read any name (eval, arguments). Before, a column written with escapes was not given (ReferenceError).
    // Group 5d (review of group 5c, P1-G5c-01): a field is given only if its name can be a variable of the function,
    // by the rule for parameters (is_bindable: a plain ASCII name of strict-mode JavaScript). The page builder gives
    // automatic fields only such names anyway (RE_VALID_NAME in bokehTools.py), except the words of JavaScript, such
    // as default, static, let or arguments. A function cannot read such a field as a variable, and building the
    // function with it failed with a SyntaxError: with every field for eval and arguments (group 5c), and before with
    // a name merely found in the text, for example the column default and a comment that says default. The refusal
    // still looks at every field.
    // Group 6: fields listed by hand are all given, by position, and a function that cannot then be built is refused
    // by name (build_function). An automatic field whose name is not plain ASCII is never given; a function that uses
    // such a column, automatic or listed by hand, is refused by name (refusal, F1).
    if(!this.auto_fields){
      this.effective_fields = this.fields
      return
    }
    const used = bodies.map((body) => used_names(body))
    this.effective_fields = this.fields.filter((field) => is_bindable(field) && bodies.some((body, k) =>
      body.includes(field) || used[k].has(field) || reads_any_name(used[k])))
  }

  uses_parameter(name: string): boolean {
    // Group 5c (review of group 5b, GPT32.RI): whether a change of the parameter "name" concerns the functions, read
    // as the refusal reads them (also written with escapes; every name for eval and arguments). The page builder asks
    // this when a parameter changes; before, it searched the text of the function, and a parameter written with
    // escapes left the old result in place without a message.
    // Group 6 (F4): the bodies of the functions in force, which are kept while a later body cannot be built
    const bodies = this.in_force !== undefined ? [this.in_force.func, this.in_force.v_func] : [this.func, this.v_func]
    return bodies.some((body) => {
      if(!body) return false
      const used = used_names(body)
      return used.has(name) || reads_any_name(used)
    })
  }

  get_fields():string[]{
    return this.effective_fields
  }

}
