import {ColumnarDataSource} from "models/sources/columnar_data_source"
import * as p from "core/properties"

export namespace CDSJoin {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    left: p.Property<ColumnarDataSource>
    right: p.Property<ColumnarDataSource>
    on_left: p.Property<string[]>
    on_right: p.Property<string[]>
    join_type: p.Property<string>
  }
}

export interface CDSJoin extends CDSJoin.Attrs {}

export class CDSJoin extends ColumnarDataSource {
  properties: CDSJoin.Props

  constructor(attrs?: Partial<CDSJoin.Attrs>) {
    super(attrs)
  }

  static __name__ = "CDSJoin"

  static init_CDSJoin() {
    this.define<CDSJoin.Props>(({Ref, Array, String})=>({
      left:  [Ref(ColumnarDataSource)],
      right: [Ref(ColumnarDataSource)],
      on_left: [ Array(String), [] ],
      on_right: [ Array(String), [] ],
      join_type: [ String ]
    }))
  }

  _indices_left: number[] | null
  _indices_right: number[] | null

  initialize(){
    super.initialize()
    this.data = {}
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
    const {left, right, on_left, on_right, join_type, change} = this
    if (on_left.length === 0){
      if (on_right.length === 0){
        this._indices_left = null
        this._indices_right = null
      }
    } else if(on_left.length === 1 && on_right.length === 0){
      this._indices_left = null
      this._indices_right = left.get_array(on_left[0])
      if(this._indices_right === null){
        console.warn("Cannot make join: column doesn't exist")
        return
      }
      if (join_type == "inner" || join_type == "right"){
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
      if (join_type == "inner" || join_type == "left"){
        const l = left.get_length()
        if (l !== null) this._indices_left = this._indices_left.filter(x => x > 0 && x < l)
      }
    } else if (on_left.length != on_right.length){
        console.warn("Cannot make join: incompatible numbers of columns")
        return
    } else {
      // TODO: Make the logic for nontrivial joins
    }
    for (const key in left.data) {
      const col = left.get_array(key)
      if(col !== null) this.data[key] = this.join_column(col, this._indices_left)
    }
    for (const key in right.data) {
      const col = right.get_array(key)
      if(col !== null) this.data[key] = this.join_column(col, this._indices_right)
    }
    // selected.indices = this.source.selected.indices
    change.emit()
  }

  /*update_selection(){
    this.source.selected.indices = this.selected.indices
  }*/

}
