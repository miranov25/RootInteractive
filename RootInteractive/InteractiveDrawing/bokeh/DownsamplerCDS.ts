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
    this.update()
  }

  update(){
    const {source, nPoints, selectedColumns, booleans, _indices} = this
    let l = source.length
    // Maybe add different downsampling strategies for small or large nPoints?
    // This is only efficient if the downsampling isn't too aggressive.
    let nSelected = 0
    _indices.length = 0
    for (let i = 0; i < l; i++){
      if(booleans === null || booleans[i]){
        if(nSelected < nPoints){
          _indices.push(i);
        } else if(Math.random() < nPoints / (nSelected+1)) {
          let randomIndex = Math.floor(Math.random()*nPoints)|0;
          _indices[randomIndex] = i;
        }
        nSelected++;
      }
    }
    _indices.sort() // Might not be needed
    for(const column of selectedColumns){
      this.data[column] = []
      for(let i=0; i < _indices.length; i++){
        this.data[column][i] = (source.data[column][_indices[i]])
      }
    }
    this.data.index = []
    for(let i=0; i < _indices.length; i++){
      this.data.index[i] = this._indices[i]
    }
    this.change.emit()
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
