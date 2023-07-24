import {ColumnarDataSource} from "models/sources/columnar_data_source"
import * as p from "core/properties"

export namespace CDSStack {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    sources: p.Property<ColumnarDataSource[]>
    mapping: p.Property<Record<string, number>>
    activeSources: p.Property<string[]> 
  }
}

export interface CDSStack extends CDSStack.Attrs {}

export class CDSStack extends ColumnarDataSource {
  properties: CDSStack.Props

  constructor(attrs?: Partial<CDSStack.Attrs>) {
    super(attrs)
  }

  static __name__ = "CDSStack"

  static init_CDSStack() {
    this.define<CDSStack.Props>(({Ref, Array, String})=>({
      sources:  [Array(Ref(ColumnarDataSource)), []],
      mapping: [p.Instance],
      activeSources: [Array(String), []]
    }))
  }

  initialize(){
    super.initialize()
    this.data = {}
    this._selected = new Set()
    this._selected_invalid = true
  }
 
  private _cached_lengths: number[]
  private _cached_offsets: number[]
  private _selected: Set<number>
  private _selected_invalid: boolean

  connect_signals(): void {
    super.connect_signals()
    for (let i=0; i < this.sources.length; i++){
        this.connect(this.sources[i].change, () => {
		    this._selected_invalid = true
		    this.change.emit()})
    }
    this.connect(this.properties.activeSources.change, () => {
		this._selected_invalid = true
		this.change.emit()})
  }

  join_column(key: string): any[]{
    if(this._selected_invalid){
      this.change_active()
    }
    const {sources, mapping, activeSources,  _cached_offsets} = this
    let col = Array(this.get_length()) 
    if(key === "_source_index"){
      for(let i=0; i < activeSources.length; i++){
        const j = mapping[activeSources[i]] 
        if(j < sources.length-1){
          col.fill(activeSources[i], _cached_offsets[j], _cached_offsets[j+1])
        } else {
          col.fill(activeSources[i], _cached_offsets[j])
        }
      }     
    } else {
      for(let i=0; i < activeSources.length; i++){
        const j = mapping[activeSources[i]] 
        const colSmall = sources[j].get_column(key)
        if(colSmall != null) {
          for(let k=0; k < colSmall.length; k++){
            col[k+_cached_offsets[j]] = colSmall[k]
          }
        }
      }
    }
    return col
  }

  change_active(): void{
    this.data = {}
    let length_acc = 0
    this._cached_lengths = []
    this._cached_offsets = []
    const {_selected, sources, activeSources, mapping} = this
    _selected.clear()
    for(const x of activeSources){
      _selected.add(mapping[x])
    }
    for(let i=0; i < sources.length; i++){
      this._cached_offsets.push(length_acc)
      let l = 0
      if(_selected.has(i)){
        l = sources[i].get_length() ?? 0
	length_acc += l
      }
      this._cached_lengths.push(l)
    } 
    this._selected_invalid = false
  }

  get_column(key: string){
    const {data} = this
    if (data[key] != null) return data[key]
    data[key] = this.join_column(key)
    return data[key]
  }

  get_length(){
    if(this._selected_invalid){
      this.change_active()
    }
    if(this._cached_offsets.length === 0){
      return 0
    }
    return this._cached_offsets[this._cached_offsets.length-1] + this._cached_lengths[this._cached_lengths.length-1]
  }

}
