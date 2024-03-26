import {ColumnarDataSource} from "models/sources/columnar_data_source"
import * as p from "core/properties"

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
    if(indices === null){
      return column
    }
    return indices.map(x => column[x])
  }

  compute_indices(): void{
    const {left, right, on_left, on_right, how, change, tolerance} = this
    const is_left_join = how === "left" || how === "outer"
    const is_right_join = how === "right" || how === "outer"
    if (on_left.length === 0){
      if (on_right.length === 0){
        this._indices_left = null
        this._indices_right = null
        this.cached_columns.clear()
        change.emit()
        return
      }
    } else if(on_left.length === 1 && on_right.length === 0){
      this._indices_left = null
      this._indices_right = left.get_array(on_left[0])
      if(this._indices_right === null){
        console.warn("Cannot make join: column doesn't exist")
        return
      }
      if (!is_left_join){
        const l = right.get_length()
        if (l !== null) this._indices_right = this._indices_right.filter(x => x > 0 && x < l)
      }
    } else if(on_left.length === 0 && on_right.length === 1){
      this._indices_left = right.get_array(on_right[0])
      this._indices_right = null
      if(this._indices_left === null){
        console.warn("Cannot make join: column doesn't exist")
        return
      }
      if (!is_right_join){
        const l = left.get_length()
        if (l !== null) this._indices_left = this._indices_left.filter(x => x > 0 && x < l)
      }
    } else if (on_left.length != on_right.length){
        console.warn("Cannot make join: incompatible numbers of columns")
        return
    } else {
      // So far, we assume sorted keys
      // Adding bisect and some way of avoiding invalidating indices would possibly help - or possibly using hash tables
      // Something more elegant would also be easier to maintain
      const len_left = left.length
      const len_right = right.length
      let down_left = 0
      let down_right = 0
      let indices_left = this._indices_left ?? []
      indices_left.length = 0
      let indices_right = this._indices_right ?? []
      indices_right.length = 0
      const arr_left: number[][] = on_left.map((x: string) => left.get_array(x))
      const arr_right: number[][] = on_right.map((x: string) => right.get_array(x))
      while(down_left < len_left && down_right < len_right){
        const comparison_result = arr_left.reduceRight(
          (acc: number, cur: number[], idx: number) => {
            let r = acc === 0 ? cur[down_left] - arr_right[idx][down_right] : acc
            if(Math.abs(r) > tolerance){
              return r
            }
            return 0
          }, 0)
        if(comparison_result > 0){
          if(is_left_join){
            indices_left.push(down_left)
            indices_right.push(-1)
          }
          down_left ++
        } else if(comparison_result < 0){
          if(is_right_join){
            indices_left.push(-1)
            indices_right.push(down_right)
          }
          down_right ++
        } else {
          let up_left = down_left
          let up_right = down_right
          let is_greater = 0
          while(is_greater === 0 && up_left < len_left){
            up_left ++
            is_greater = arr_left.reduceRight((acc: number, cur: number[]) => acc === 0 ? cur[up_left] - cur[down_left] : acc, 0)
          }
          is_greater = 0
          while(is_greater === 0 && up_right < len_right){
            up_right ++
            is_greater = arr_right.reduceRight((acc: number, cur: number[]) => acc === 0 ? cur[up_right] - cur[down_right] : acc, 0)
          }

          for(let i_left = down_left; i_left < up_left; i_left ++){
            for(let i_right = down_right; i_right < up_right; i_right ++){
              indices_left.push(i_left)
              indices_right.push(i_right)
            }
          }

          down_left = up_left
          down_right = up_right
        }
      }
      if(is_left_join){
        while(down_left < len_left){
          indices_left.push(down_left)
          indices_right.push(-1)
          down_left ++   
        }       
      }
      if(is_right_join){
        while(down_right < len_right){
          indices_left.push(-1)
          indices_right.push(down_right) 
          down_right ++         
        }       
      }
      this._indices_left = indices_left
      this._indices_right = indices_right
    }
    // selected.indices = this.source.selected.indices
    this.cached_columns.clear()
    change.emit()
  }

  get_column(key: string){
    const {left, right, data, prefix_left, prefix_right} = this
    if (this.cached_columns.has(key)) return data[key]
    let column = null
    if (prefix_left && key.startsWith(prefix_left)){
      let key_new = key.replace(prefix_left,"")
      column = left.get_column(key_new)
    } else if(prefix_right && key.startsWith(prefix_right)){
      let key_new = key.replace(prefix_right,"")
      column = right.get_column(key_new)      
    } else {
      try {
        column = left.get_column(key)
      }
      catch {
        column = right.get_column(key)
      }
      if (column == null){
        column = right.get_column(key)
      }
    }
    if(column != null) {
      if (!Array.isArray(column)){
        data[key] = this.join_column(Array.from(column), this._indices_right)
      } else {
        data[key] = this.join_column(column, this._indices_right)
      }
    }
    this.cached_columns.add(key)
    console.log(data[key])
    return data[key]
  }

  get_length(){
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
