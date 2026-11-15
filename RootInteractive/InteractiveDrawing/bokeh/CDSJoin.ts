import {ColumnarDataSource} from "models/sources/columnar_data_source"
import * as p from "core/properties"

// A join shows the columns of two tables side by side.
//   on_left = [],    on_right = []     row i of the left table beside row i of the right table
//   on_left = [col], on_right = []     index join: the left table holds, in col, the row number of its partner in the
//                                      right table (the left table is the child, the right table the parent)
//   on_left = [],    on_right = [col]  the same with the row numbers held by the right table
//   otherwise                          key join: rows whose key columns are equal within `tolerance`
//
// _indices_left and _indices_right say which row of each table stands in each row of the join:
//   null    every row of that table, in its own order
//   array   one row number per row of the join; -1 means "no partner", shown as NaN (null in a text column)

// A join that cannot be computed says so by name on every read (get_length and get_column), not while the page is
// built.
class JoinRefusal extends Error {}

// What a row number pointing INTO a table means. For a plain table: its rows as they stand. For an index join that
// dropped child rows it is still the row number of the child table before any row was dropped.
type Origin = {rows: number[] | null, count: number}

function origin_of(table: ColumnarDataSource): Origin {
  if (table instanceof CDSJoin) return table.original_rows()
  return {rows: null, count: table.get_length() ?? 0}
}

// A key takes part in a join only if it is a number; NaN never matches, not even another NaN.
function is_key(v: any): boolean {
  return typeof v === "number" && v === v
}

// What stands in a row without a partner: null if the column holds text, NaN otherwise.
function missing_value(column: any): number | null {
  if (!Array.isArray(column)) return NaN
  let has_number = false
  let has_null = false
  for (const v of column) {
    if (typeof v === "string") return null
    if (typeof v === "number") has_number = true
    else if (v === null) has_null = true
  }
  return !has_number && has_null ? null : NaN
}

function gather(column: any, rows: number[] | null): any[] {
  if (rows === null) return Array.isArray(column) ? column : Array.from(column)
  const out = new Array(rows.length)
  let fill: number | null = NaN
  let fill_known = false
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i]
    if (r >= 0) out[i] = column[r]
    else {
      if (!fill_known) {
        fill = missing_value(column)
        fill_known = true
      }
      out[i] = fill
    }
  }
  return out
}

// `order` lists rows by rising value of `col`. First position in order[lo..hi) not below x by more than tol.
function first_within(col: any, order: number[], lo: number, hi: number, x: number, tol: number): number {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (x - col[order[mid]] > tol) lo = mid + 1
    else hi = mid
  }
  return lo
}

// First position in order[lo..hi) above x by more than tol.
function first_above(col: any, order: number[], lo: number, hi: number, x: number, tol: number): number {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (col[order[mid]] - x > tol) hi = mid
    else lo = mid + 1
  }
  return lo
}

export namespace CDSJoin {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    left: p.Property<ColumnarDataSource>
    right: p.Property<ColumnarDataSource>
    on_left: p.Property<string[]>
    on_right: p.Property<string[]>
    prefix_left: p.Property<string>
    prefix_right: p.Property<string>
    how: p.Property<string>
    tolerance: p.Property<number>
  }
}

export interface CDSJoin extends CDSJoin.Attrs {}

export class CDSJoin extends ColumnarDataSource {
  properties: CDSJoin.Props

  constructor(attrs?: Partial<CDSJoin.Attrs>) {
    super(attrs)
  }

  static __name__ = "CDSJoin"

  static {
    this.define<CDSJoin.Props>(({Ref, Array, String, Number})=>({
      left:  [Ref(ColumnarDataSource)],
      right: [Ref(ColumnarDataSource)],
      on_left: [ Array(String), [] ],
      on_right: [ Array(String), [] ],
      prefix_left: [String, ""],
      prefix_right: [String, ""],
      how: [ String, "inner" ],
      tolerance: [Number, 1e-5]
    }))
  }

  _indices_left: number[] | null
  _indices_right: number[] | null

  // why the join cannot be computed, or null
  _refusal: string | null
  // index joins only: the table that holds the row numbers, and which of its rows are kept (null: all)
  _holder: ColumnarDataSource | null
  _holder_rows: number[] | null

  cached_columns: Set<string>

  initialize(){
    super.initialize()
    this.data = {}
    this.cached_columns = new Set()
    this.compute_indices()
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.left.change, () => this.compute_indices())
    this.connect(this.right.change, () => this.compute_indices())

    //this.connect(this.selected.change, () => this.update_selection())
  }

  join_column(column: any[], indices: number[] | null): any[]{
    return gather(column, indices)
  }

  compute_indices(): void{
    this._refusal = null
    this._holder = null
    this._holder_rows = null
    try {
      this.compute_rows()
    } catch (e) {
      if (!(e instanceof JoinRefusal)) throw e
      this._refusal = e.message
      this._indices_left = []
      this._indices_right = []
    }
    // selected.indices = this.source.selected.indices
    this.cached_columns.clear()
    this.change.emit()
  }

  compute_rows(): void{
    const {left, right, on_left, on_right, how} = this
    const is_left_join = how === "left" || how === "outer"
    const is_right_join = how === "right" || how === "outer"
    if (on_left.length === 0 && on_right.length === 0){
      this._indices_left = null
      this._indices_right = null
    } else if (on_left.length === 1 && on_right.length === 0){
      const rows = this.index_join(left, right, on_left[0], is_left_join)
      this._indices_left = rows.holder
      this._indices_right = rows.target
    } else if (on_left.length === 0 && on_right.length === 1){
      const rows = this.index_join(right, left, on_right[0], is_right_join)
      this._indices_left = rows.target
      this._indices_right = rows.holder
    } else if (on_left.length !== on_right.length){
      console.warn("Cannot make join: incompatible numbers of columns")
      this._indices_left = []
      this._indices_right = []
    } else {
      this.key_join(is_left_join, is_right_join)
    }
  }

  // Index join. `holder` keeps, in `column`, the row number of each row's partner in `target`.
  //   a whole number from 0 up to the last row of `target`:  that row; if `target` is a join that dropped the row,
  //                                                          there is no partner
  //   a number past the end of `target`:                     the join is refused by name
  //   anything else (-1, NaN, no value):                     no partner
  // A row without a partner is kept only if `keep_all`.
  index_join(holder: ColumnarDataSource, target: ColumnarDataSource, column: string,
             keep_all: boolean): {holder: number[] | null, target: number[]} {
    const pointers: any = holder.get_column(column)
    if (pointers == null){
      console.warn("Cannot make join: column doesn't exist")
      return {holder: [], target: []}
    }
    const {rows, count} = origin_of(target)
    let position: Int32Array | null = null
    if (rows !== null){
      position = new Int32Array(count).fill(-1)
      for (let i = 0; i < rows.length; i++) position[rows[i]] = i
    }
    const n = pointers.length
    const kept: number[] = []
    const partner: number[] = []
    for (let i = 0; i < n; i++){
      const x = pointers[i]
      let row = -1
      if (typeof x === "number" && x >= 0 && Number.isInteger(x)){
        if (x >= count){
          const name = this.name ?? this.id
          throw new JoinRefusal(
            `CDSJoin "${name}": row number ${x} in column "${column}" is outside the parent table (${count} rows)`)
        }
        row = position === null ? x : position[x]
      }
      if (row >= 0 || keep_all){
        kept.push(i)
        partner.push(row)
      }
    }
    this._holder = holder
    this._holder_rows = kept.length === n ? null : kept
    return {holder: this._holder_rows, target: partner}
  }

  // Key join. The rows follow the left table's order; the partners of one left row follow the right table's order;
  // right rows without a partner come last (right and outer joins). The right table is sorted once here, so neither
  // table has to arrive sorted, and it does not matter which key is listed first.
  key_join(is_left_join: boolean, is_right_join: boolean): void{
    const {left, right, on_left, on_right} = this
    const indices_left: number[] = []
    const indices_right: number[] = []
    this._indices_left = indices_left
    this._indices_right = indices_right
    const keys_left: any[] = on_left.map((x: string) => left.get_column(x))
    const keys_right: any[] = on_right.map((x: string) => right.get_column(x))
    if (keys_left.some(x => x == null) || keys_right.some(x => x == null)){
      console.warn("Cannot make join: column doesn't exist")
      return
    }
    const len_left = left.get_length() ?? 0
    const len_right = right.get_length() ?? 0
    const tolerance = this.tolerance >= 0 ? this.tolerance : 0
    const last = keys_left.length - 1

    // the right rows that can have a partner, by rising key (first listed key first, then the row number)
    const order: number[] = []
    for (let j = 0; j < len_right; j++){
      if (keys_right.every(col => is_key(col[j]))) order.push(j)
    }
    order.sort((a: number, b: number) => {
      for (const col of keys_right){
        if (col[a] < col[b]) return -1
        if (col[a] > col[b]) return 1
      }
      return a - b
    })

    let found: number[] = []
    let i_left = 0
    // order[lo..hi) holds the same values of the keys before `depth` and is sorted by the keys from `depth` on
    const collect = (lo: number, hi: number, depth: number): void => {
      const col = keys_right[depth]
      const x = keys_left[depth][i_left]
      const first = first_within(col, order, lo, hi, x, tolerance)
      const end = first_above(col, order, first, hi, x, tolerance)
      if (depth === last){
        for (let q = first; q < end; q++) found.push(order[q])
        return
      }
      let start = first
      while (start < end){
        const stop = first_above(col, order, start, end, col[order[start]], 0)
        collect(start, stop, depth + 1)
        start = stop
      }
    }

    const has_partner = new Uint8Array(len_right)
    for (i_left = 0; i_left < len_left; i_left++){
      found = []
      if (keys_left.every(col => is_key(col[i_left]))) collect(0, order.length, 0)
      if (found.length > 1) found.sort((a: number, b: number) => a - b)
      for (const i_right of found){
        indices_left.push(i_left)
        indices_right.push(i_right)
        has_partner[i_right] = 1
      }
      if (found.length === 0 && is_left_join){
        indices_left.push(i_left)
        indices_right.push(-1)
      }
    }
    if (is_right_join){
      for (let i_right = 0; i_right < len_right; i_right++){
        if (has_partner[i_right] === 0){
          indices_left.push(-1)
          indices_right.push(i_right)
        }
      }
    }
  }

  refuse_if_refused(): void{
    if (this._refusal !== null) throw new JoinRefusal(this._refusal)
  }

  // What a row number pointing into this join means (see Origin).
  original_rows(): Origin{
    this.refuse_if_refused()
    if (this._holder === null) return {rows: null, count: this.get_length() ?? 0}
    const origin = origin_of(this._holder)
    const kept = this._holder_rows
    if (kept === null) return origin
    const rows = origin.rows
    return {rows: rows === null ? kept : kept.map(i => rows[i]), count: origin.count}
  }

  get_column(key: string){
    this.refuse_if_refused()
    const {left, right, data, prefix_left, prefix_right} = this
    if (this.cached_columns.has(key)) return data[key]
    let column = null
    // a left column follows the rows of the left table, a right column the rows of the right table
    let indices = this._indices_left
    if (prefix_left && key.startsWith(prefix_left)){
      let key_new = key.replace(prefix_left,"")
      column = left.get_column(key_new)
    } else if(prefix_right && key.startsWith(prefix_right)){
      let key_new = key.replace(prefix_right,"")
      column = right.get_column(key_new)
      indices = this._indices_right
    } else {
      try {
        column = left.get_column(key)
      }
      catch (e) {
        if (e instanceof JoinRefusal) throw e
        column = null
      }
      if (column == null){
        column = right.get_column(key)
        indices = this._indices_right
      }
    }
    if(column != null) {
      data[key] = gather(column, indices)
    }
    this.cached_columns.add(key)
    return data[key]
  }

  get_length(){
    this.refuse_if_refused()
    if (this._indices_left == null){
      if(this._indices_right == null){
        let l = this.left.get_length()
        if(l == null){
          return this.right.get_length()
        }
        let r = this.right.get_length()
        if(r == null){
          return l
        }
        return Math.min(l, r)
      } else {
        return this._indices_right.length
      }
    }
    return this._indices_left.length
  }

}
