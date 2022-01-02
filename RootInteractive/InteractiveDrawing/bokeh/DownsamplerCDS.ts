import {ColumnarDataSource} from "models/sources/columnar_data_source"
import * as p from "core/properties"

export namespace DownsamplerCDS {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnarDataSource>
    nPoints: p.Property<number>
    selectedColumns: p.Property<string[]>
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
    this.define<DownsamplerCDS.Props>(({Ref, Int, Array, String})=>({
      source:  [Ref(ColumnarDataSource)],
      nPoints:    [ Int, 300 ],
      selectedColumns:    [ Array(String), [] ]
    }))
  }

  public booleans: number[] | null// This is a hack to avoid expensive validation

  private _indices: number[]

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
    this.connect(this.source.change, () => this.update())
  }

  update(){
    const {source, nPoints, selectedColumns, data, change, booleans, _indices} = this
    const l = source.length
    // Maybe add different downsampling strategies for small or large nPoints?
    // This is only efficient if the downsampling isn't too aggressive.
    const downsampled_indices: number[] = []
    for(let i=0; i<l && downsampled_indices.length < nPoints; i++){
      if (booleans === null || booleans[_indices[i]]){
        downsampled_indices.push(_indices[i])
      }
    }

    downsampled_indices.sort((a,b)=>a-b)

    for(const columnName of selectedColumns){
      if (!source.get_column(columnName)){
        throw ReferenceError("Invalid column name " + columnName)
      }
      data[columnName] = []
      const colSource = source.get_array(columnName)
      const colDest = data[columnName]
      for(let i=0; i < downsampled_indices.length; i++){
        colDest[i] = (colSource[downsampled_indices[i]])
      }
    }

    data.index = downsampled_indices
    const selected_indices: number[] = []
    const original_indices = this.source.selected.indices
    let j=0
    for(let i=0; i < original_indices.length; i++){
      // TODO: Maybe do binary search, this won't assume selected indices are sorted and might be more performant
      while(downsampled_indices[j] < original_indices[i]){
        j++
      }
      if(downsampled_indices[j] === original_indices[i]){
        selected_indices.push(j)
        j++
      }
    }
    change.emit()
    this.selected.indices = selected_indices
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

}
