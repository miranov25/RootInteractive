import {ColumnarDataSource} from "models/sources/columnar_data_source"
import { MultiSelect } from "models/widgets/multiselect"
import {Model} from "model"
import * as p from "core/properties"

export namespace MultiSelectFilter {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    source: p.Property<ColumnarDataSource>
    widget: p.Property<MultiSelect>
    mapping: p.Property<Record<string, number>>
    mask: p.Property<number>
    field:p.Property<string>
    how: p.Property<string>
  }
}

export interface MultiSelectFilter extends MultiSelectFilter.Attrs {}

export class MultiSelectFilter extends Model {
  properties: MultiSelectFilter.Props

  constructor(attrs?: Partial<MultiSelectFilter.Attrs>) {
    super(attrs)
  }

  static __name__ = "MultiSelectFilter"

  static init_MultiSelectFilter() {
    this.define<MultiSelectFilter.Props>(({Ref, Int, String})=>({
      source:  [Ref(ColumnarDataSource)],
      widget:    [ Ref(MultiSelect) ],
      mapping:    [p.Instance],
      mask: [Int, -1],
      field: [String],
      how: [String, "any"]
    }))
  }

  private cached_vector: boolean[]
  private dirty_source: boolean
  private dirty_widget: boolean

  initialize(){
    super.initialize()
    this.dirty_source = true
    this.dirty_widget = true
  }

  public v_compute(): boolean[]{
    const {dirty_source, dirty_widget, cached_vector, widget, source, mapping, mask, field, how} = this
    if (!dirty_source && !dirty_widget){
        return cached_vector
    }
    let col = source.get_array(field)
    const mask_new = widget.value.map((a: string) => mapping[a]).reduce((acc: number, cur: number) => acc | cur, mask)
    let new_vector: boolean[] = []
    if (how == "any"){
        new_vector = col.map((x: number) => (x & mask_new) != 0)
    } else if(how == "all"){
        new_vector = col.map((x:number) => (x & mask_new) == mask_new)
    }
    this.dirty_source = false
    this.dirty_widget = false
    this.cached_vector = new_vector
    return new_vector
  }
}
