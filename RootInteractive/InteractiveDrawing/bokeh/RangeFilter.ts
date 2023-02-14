import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {RIFilter} from "./RIFilter"
import * as p from "core/properties"

export namespace RangeFilter {
  export type Attrs = p.AttrsOf<Props>

  export type Props = RIFilter.Props & {
    source: p.Property<ColumnarDataSource>
    field: p.Property<string>
    range: p.Property<number[]>
  }
}

export interface RangeFilter extends RangeFilter.Attrs {}

export class RangeFilter extends RIFilter {
  properties: RangeFilter.Props

  constructor(attrs?: Partial<RangeFilter.Attrs>) {
    super(attrs)
  }

  static __name__ = "RangeFilter"

  static init_RangeFilter() {
    this.define<RangeFilter.Props>(({Ref, Number, String, Array})=>({
      source:  [Ref(ColumnarDataSource)],
      field: [String],
      range: [Array(Number)]
    }))
  }

  private cached_vector: boolean[]
  private dirty_source: boolean
  private dirty_widget: boolean

  public index_low: number
  public index_high: number

  initialize(){
    super.initialize()
    this.dirty_source = true
    this.dirty_widget = true
    this.index_low = 0
    this.index_high = -1
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.properties.range.change, this.mark_dirty_widget)
    this.connect(this.source.change, this.mark_dirty_source)
  }

  mark_dirty_widget(){
    this.dirty_widget = true
//    this.change.emit()
  }

  mark_dirty_source(){
    this.dirty_source = true
//    this.change.emit()
  }

  public v_compute(): boolean[]{
    const {dirty_source, dirty_widget, cached_vector, source, field} = this
    if (!dirty_source && !dirty_widget){
        return cached_vector
    }
    let col = source.get_column(field) as number[]
    let new_vector: boolean[] = this.cached_vector
    if (new_vector == null){
        new_vector = Array(col.length)
    } else {
        new_vector.length = col.length
    }
    const index_low = this.index_low
    const index_high = this.index_high === -1 ? col.length : this.index_high
    const [low, high] = this.range
    let has_element = false
    for(let i=index_low; i<index_high; i++){
      const a = (col[i] >= low) && (col[i] <= high)
      new_vector[i] = a
      has_element ||= a
    }
    if(!has_element){
      console.warn("Range empty: " + field)
    }
    this.dirty_source = false
    this.dirty_widget = false
    this.cached_vector = new_vector
    return new_vector
  }
}
