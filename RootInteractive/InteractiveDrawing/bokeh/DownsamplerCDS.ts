import {ColumnarDataSource} from "models/sources/columnar_data_source"
import * as p from "core/properties"

export namespace DownsamplerCDS {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnarDataSource>
    nPoints: p.Property<number>
    watched: p.Property<boolean>
  }
}

export interface DownsamplerCDS extends DownsamplerCDS.Attrs {}

export class DownsamplerCDS extends ColumnarDataSource {
  properties: DownsamplerCDS.Props

  constructor(attrs?: Partial<DownsamplerCDS.Attrs>) {
    super(attrs)
  }

  static __name__ = "DownsamplerCDS"

  static init_DownsamplerCDS() {
    this.define<DownsamplerCDS.Props>(({Ref, Int, Boolean})=>({
      source:  [Ref(ColumnarDataSource)],
      nPoints:    [ Int, 300 ],
      watched: [Boolean, true]
    }))
  }

  public booleans: number[] | null// This is a hack to avoid expensive validation

  private _indices: number[]

  private _downsampled_indices: number[]

  private _needs_update: boolean

  initialize(){
    super.initialize()
    this.booleans = null
    this._indices = []
    this.data = {}
    this.shuffle_indices()
    this.update()
  }

  shuffle_indices(){
    const {_indices, source} = this
    _indices.length = source.get_length()!
    // Deck shuffling algorithm - for each card drawn, swap it with a random card in hand
    for(let i=0; i<_indices.length; ++i){
      let random_index = (Math.random()*(i+1)) | 0
      _indices[i] = i
      _indices[i] = _indices[random_index]
      _indices[random_index] = i
    }
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.selected.change, () => this.update_selection())
    // TODO: Add the use case when source grows in size
    this.connect(this.source.change, () => {this.invalidate()})
    this.connect(this.properties.watched.change, () => {this.toggle_watched()})
  }

  update(){
    const {source, nPoints, selected, booleans, _indices} = this
    const l = source.length
    if(this._indices.length < l){
      this.shuffle_indices()
    }
    // Maybe add different downsampling strategies for small or large nPoints?
    // This is only efficient if the downsampling isn't too aggressive.
    this._downsampled_indices = []
    for(let i=0; i < this._indices.length && this._downsampled_indices.length < nPoints; i++){
      if (_indices[i]<l && (booleans == null || booleans[_indices[i]])){
        this._downsampled_indices.push(_indices[i])
      }
    }

    this._downsampled_indices.sort((a,b)=>a-b)
    this.data = {}

    const selected_indices: number[] = []
    const original_indices = this.source.selected.indices
    let j=0
    for(let i=0; i < original_indices.length; i++){
      // TODO: Maybe do binary search, this won't assume selected indices are sorted and might be more performant
      while(this._downsampled_indices[j] < original_indices[i]){
        j++
      }
      if(this._downsampled_indices[j] === original_indices[i]){
        selected_indices.push(j)
        j++
      }
    }
//    change.emit()
    selected.indices = selected_indices
    this._needs_update = false
  }


  update_selection(){
    const downsampled_indices = this.data.index
    const selected_indices = this.selected.indices.map((x:number)=>downsampled_indices[x])
    selected_indices.sort((a,b)=>a-b)
    const original_indices = this.source.selected.indices
    // TODO: Change original CDS selection indices
    const old_indices: number[] = []
    let iDownsampled=0
    for(let i=0; i < original_indices.length; i++){
      // TODO: Maybe do binary search, this won't assume selected indices are sorted and might be more performant
      while(downsampled_indices[iDownsampled] < original_indices[i]){
        iDownsampled++
      }
      if(iDownsampled == downsampled_indices.length || downsampled_indices[iDownsampled] > original_indices[i]){
        old_indices.push(original_indices[i])
      } 
    }
    // Mergesort - the arrays are guaranteed to be disjoint and values in each to be unique
    const merged_indices = []
    let iSelected = 0
    let iOld = 0
    while(iOld < old_indices.length || iSelected < selected_indices.length){
      if (iOld < old_indices.length){
        if(iSelected < selected_indices.length && selected_indices[iSelected] < old_indices[iOld]){
          merged_indices.push(selected_indices[iSelected])
          iSelected++
        } else {
          merged_indices.push(old_indices[iOld])
          iOld++
        }
      } else {
        merged_indices.push(selected_indices[iSelected])
        iSelected++
      }
    }
    this.source.selected.indices = merged_indices
        
  }

  get_column(columnName: string){
    if(this.watched && this._needs_update){
      this.update()
    }
    const {data, source, _downsampled_indices} = this
    if (data[columnName] != undefined){
      return data[columnName]
    }
    const column_orig = source.get_column(columnName)
    if (column_orig == null){
      throw ReferenceError("Invalid column name " + columnName)
    }
    data[columnName] = _downsampled_indices.map((x: number) => column_orig[x])
    return data[columnName]
  }

  get_length(){
    if(this.watched && this._needs_update){
      this.update()
    }
    return this._downsampled_indices.length
  }

  invalidate(){
    this._needs_update = true
    if(this.watched){
      this.change.emit()
    }
  }

  toggle_watched(){
    if(this.watched && this._needs_update){
      this.change.emit()
    }
  }

  //Needed because of typescript, is supposed to be a nop
  on_visible_change(){}
}
