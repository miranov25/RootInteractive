import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {RIFilter} from "./RIFilter"
import * as p from "core/properties"

export namespace ColumnFilter {
  export type Attrs = p.AttrsOf<Props>

  export type Props = RIFilter.Props & {
    source: p.Property<ColumnarDataSource>
    field:p.Property<string>
  }
}

export interface ColumnFilter extends ColumnFilter.Attrs {}

export class ColumnFilter extends RIFilter {
  properties: ColumnFilter.Props

  constructor(attrs?: Partial<ColumnFilter.Attrs>) {
    super(attrs)
  }

  static __name__ = "ColumnFilter"

  static init_ColumnFilter() {
    this.define<ColumnFilter.Props>(({Ref, String})=>({
      source:  [Ref(ColumnarDataSource)],
      field: [String]
    }))
  }

  private cached_vector: boolean[]
  private dirty_source: boolean

  initialize(){
    super.initialize()
    this.dirty_source = true
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.source.change, this.mark_dirty_source)
  }

  mark_dirty_source(){
    this.dirty_source = true
    this.change.emit()
  }

  public v_compute(): boolean[]{
    const {dirty_source, cached_vector, source, field} = this
    if (!dirty_source){
        return cached_vector
    }
    let col = source.get_column(field)
    if(col == null) return []
    let new_vector: boolean[] = this.cached_vector
    if (new_vector == null){
        new_vector = Array(col.length)
    } else {
        new_vector.length = col.length
    }
    for(let i=0; i<col.length; i++){
        new_vector[i] = !!col[i]
      }
    this.dirty_source = false
    this.cached_vector = new_vector
    return new_vector
  }
}
