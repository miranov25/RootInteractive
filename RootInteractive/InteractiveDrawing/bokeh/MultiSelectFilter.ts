import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {RIFilter} from "./RIFilter"
import * as p from "core/properties"

export namespace MultiSelectFilter {
  export type Attrs = p.AttrsOf<Props>

  export type Props = RIFilter.Props & {
    source: p.Property<ColumnarDataSource>
    selected: p.Property<string[]>
    mapping: p.Property<Record<string, number>>
    mask: p.Property<number>
    field:p.Property<string>
    how: p.Property<string>
  }
}

export interface MultiSelectFilter extends MultiSelectFilter.Attrs {}

export class MultiSelectFilter extends RIFilter {
  properties: MultiSelectFilter.Props

  constructor(attrs?: Partial<MultiSelectFilter.Attrs>) {
    super(attrs)
  }

  static __name__ = "MultiSelectFilter"

  static init_MultiSelectFilter() {
    this.define<MultiSelectFilter.Props>(({Ref, Int, Array, String})=>({
      source:  [Ref(ColumnarDataSource)],
      selected:    [ Array(String), [] ],
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

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.properties.selected.change, this.mark_dirty_widget)
    this.connect(this.source.change, this.mark_dirty_source)
  }

  mark_dirty_widget(){
    this.dirty_widget = true
  }

  mark_dirty_source(){
    this.dirty_source = true
  }

  public v_compute(): boolean[]{
    const {dirty_source, dirty_widget, cached_vector, selected, source, mapping, mask, field, how} = this
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
    if (how == "whitelist"){
        const accepted_codes = selected.map((a: string) => mapping[a])
        let count=0
        for(let i=0; i<col.length; i++){
            const x = col[i]
            let isOK = false
            for(let j=0; j<accepted_codes.length; j++){
              if(accepted_codes[j] == x){
                isOK = true
                break
              }
            }
            new_vector[i] = isOK
        }        
        console.log(count)
        this.dirty_source = false
        this.dirty_widget = false
        this.cached_vector = new_vector
        return new_vector
    }
    const mask_new = selected.map((a: string) => mapping[a]).reduce((acc: number, cur: number) => acc | cur, 0) & mask
    if (how == "any"){
        for(let i=0; i<col.length; i++){
          const x = col[i] as number
          new_vector[i] = (x & mask_new) != 0
        }
    } else if(how == "all"){
      for(let i=0; i<col.length; i++){
        const x = col[i] as number
        new_vector[i] = (x & mask_new) == mask_new
      }
    } else if(how == "neither"){
      for(let i=0; i<col.length; i++){
        const x = col[i] as number
        new_vector[i] = (x & mask_new) == 0
      }
    } else if(how == "eq"){
      for(let i=0; i<col.length; i++){
        const x = col[i] as number
        new_vector[i] = x == mask_new
      }
    }
    this.dirty_source = false
    this.dirty_widget = false
    this.cached_vector = new_vector
    return new_vector
  }
}
