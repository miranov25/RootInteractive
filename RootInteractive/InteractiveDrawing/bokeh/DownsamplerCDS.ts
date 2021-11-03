import {ColumnDataSource} from "models/sources/column_data_source"
import * as p from "core/properties"

export namespace DownsamplerCDS {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnDataSource.Props & {
    source: p.Property<ColumnDataSource>
    nPoints: p.Property<number>
    selectedColumns: p.Property<string[]>
  }
}

export interface DownsamplerCDS extends DownsamplerCDS.Attrs {}

export class DownsamplerCDS extends ColumnDataSource {
  properties: DownsamplerCDS.Props

  constructor(attrs?: Partial<DownsamplerCDS.Attrs>) {
    super(attrs)
  }

  static __name__ = "DownsamplerCDS"

  static init_DownsamplerCDS() {
    this.define<DownsamplerCDS.Props>(({Ref, Int, Array, String})=>({
      source:  [Ref(ColumnDataSource)],
      nPoints:    [ Int ],
      selectedColumns:    [ Array(String), [] ]
    }))
  }

  public booleans: number[] | null// This is a hack to avoid expensive validation

  private _indices: number[]

  initialize(){
    super.initialize()
    this.booleans = null
    this._indices = []
    this.shuffle_indices()
    this.update()
  }

  shuffle_indices(){
    const {_indices, source} = this
    _indices.length = source.get_length()!
    // Deck shuffling algorithm - for each card drawn, swap it with a random card in hand
    for(let i=0; i<_indices.length; ++i){
      let random_index = (Math.random()*i) | 0
      _indices[i] = i
      _indices[i] = _indices[random_index]
      _indices[random_index] = i
    }
  }

  update(){
    const {source, nPoints, selectedColumns, data, change, booleans, _indices} = this
    const l = source.length
    // Maybe add different downsampling strategies for small or large nPoints?
    // This is only efficient if the downsampling isn't too aggressive.
    const selected_indices: number[] = []
    for(let i=0; i<l && selected_indices.length < nPoints; i++){
      if (booleans === null || booleans[_indices[i]]){
        selected_indices.push(_indices[i])
      }
    }
    
    for(const columnName of selectedColumns){
      data[columnName] = []
      const colSource = source.data[columnName]
      const colDest = data[columnName]
      for(let i=0; i < selected_indices.length; i++){
        colDest[i] = (colSource[selected_indices[i]])
      }
    }
//    this.data.index = []
//    for(let i=0; i < _indices.length; i++){
//      this.data.index[i] = this._indices[i]
//    }
    data.index = selected_indices
    change.emit()
  }
/*
  let j=0
  for(let i=0; i < _indices.length; i++){
  }
  */

  update_selection(){
    //const downsampled_indices = this.selected.indices
    //const original_indices = this.source.selected.indices
    // TODO: Change original CDS selection indices
  }

}
