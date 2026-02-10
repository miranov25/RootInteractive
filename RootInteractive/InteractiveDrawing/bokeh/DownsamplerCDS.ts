import {ColumnDataSource} from "models/sources/column_data_source"
import {Model} from "model"
import * as p from "core/properties"

export namespace DownsamplerCDS {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnDataSource.Props & {
    source: p.Property<any>
    nPoints: p.Property<number>
    watched: p.Property<boolean>
    filter: p.Property<any>
  }
}

function nextpoweroftwo(x: number){
  let y=1; while(y<=x){y <<= 1} return y
}

function hash32(x: number) {
  x |= 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
}

function noise(seed: number, idx: number){
  return hash32(seed ^ Math.imul(idx, 0x9e3779b9))
}

function shuffle(n: number, arrayOut: number[], seed: number){
  let j=0
  for(let i=0; i<n; j++){
    const x = (nextpoweroftwo(i) - 1) & noise(seed,j)
    if(x > i){
      continue
    }
    arrayOut[i] = i
    arrayOut[i] = x
    arrayOut[x] = i
    i++
  }
}

export interface DownsamplerCDS extends DownsamplerCDS.Attrs {}

export class DownsamplerCDS extends ColumnDataSource {
  properties: DownsamplerCDS.Props

  constructor(attrs?: Partial<DownsamplerCDS.Attrs>) {
    super(attrs)
  }

  static __name__ = "DownsamplerCDS"

  static {
    this.define<DownsamplerCDS.Props>(({Ref, Int, Boolean, Nullable})=>({
      source:  [Ref(Model)],
      nPoints:    [ Int, 300 ],
      watched: [Boolean, true],
      filter: [Nullable(Ref(Model)), null]
    }))
  }

  public booleans: number[] | null// This is a hack to avoid expensive validation
  public low: number
  public high: number

  private _indices: number[]

  private _downsampled_indices: number[]

  private _needs_update: boolean

  private _is_trivial: boolean

  private _cached_columns: Set<string>

  initialize(){
    super.initialize()
    this.booleans = null
    this._indices = []
    this._downsampled_indices = []
    this.data = {}
    this.low = 0
    this.high = -1
    this._needs_update = true
    this._is_trivial = this.nPoints < 0 && this.filter == null
    this._cached_columns = new Set()
  }

  shuffle_indices(){
    const {_indices, source} = this
    _indices.length = source.get_length()!
    // Fisher-Yates random permutation
    shuffle(_indices.length, _indices, 42)
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.selected.change, () => this.update_selection())
    this.connect(this.source.change, () => {this.invalidate()})
    if(this.filter != null){
      this.connect(this.filter.change, () => {this.invalidate()})
    }
    this.connect(this.properties.watched.change, () => {this.toggle_watched()})
  }

  update(){
    const {source, nPoints, selected, filter, _indices} = this
    const l = source.length
    console.log(this.name)
    if(filter == null && (nPoints < 0 || nPoints >= l)){
      this._is_trivial = true
      this._needs_update = false
      return
    }
    if(nPoints < 0 || nPoints >= l){
      this._downsampled_indices = []
      const booleans = filter!.v_compute()
      this._downsampled_indices = []
      for(let i=0; i < l; i++){
        if (booleans[i]){
          this._downsampled_indices.push(i)
        }
      }      
    } else {
      if(this._indices.length < l){
        this.shuffle_indices()
      }
      // Maybe add different downsampling strategies for small or large nPoints?
      // This is only efficient if the downsampling isn't too aggressive.
      const low = this.low
      const high = this.high >= 0 && this.high < l ? this.high : l
      //const booleans = this.booleans
      const booleans = filter != null ? filter.v_compute() : null
      this._downsampled_indices = []
      for(let i=0; i < _indices.length && this._downsampled_indices.length < nPoints; i++){
        if (_indices[i] <= high && _indices[i] >= low && (booleans == null || booleans[_indices[i]])){
          this._downsampled_indices.push(_indices[i])
        }
      }
      this._downsampled_indices.sort((a,b)=>a-b)
    }
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
    selected.indices = selected_indices
    this._needs_update = false
  }


  update_selection(){
    if(this._is_trivial) return
    const downsampled_indices = this.data.index
    const selected_indices = Array.from(this.selected.indices).map((x:number)=>downsampled_indices[x])
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
    const {data, source, _downsampled_indices, _is_trivial, _cached_columns} = this
    if(_is_trivial){
      return source.get_column(columnName) 
    }
    if(_cached_columns.has(columnName)){
      return data[columnName]
    }
    const column_orig = source.get_column(columnName)
    if (column_orig == null){
      throw ReferenceError("Invalid column name " + columnName)
    }
    let new_column = data[columnName] as any[]
    if(new_column == null){
      new_column = Array(_downsampled_indices.length)
    }
    if(new_column.length < _downsampled_indices.length) new_column.length = _downsampled_indices.length
    for(let i=0; i <_downsampled_indices.length; i++){
      new_column[i] = column_orig[_downsampled_indices[i]]
    }
    data[columnName] = new_column
    return data[columnName]
  }

  get_length(){
    if(this.watched && this._needs_update){
      this.update()
    }
    if(this._is_trivial){
      return this.source.get_length()
    }
    return this._downsampled_indices.length
  }

  invalidate(){
    this._needs_update = true
    this._cached_columns.clear()
    if(this.watched){
      this.change.emit()
    }
  }

  toggle_watched(){
    if(this.watched && this._needs_update){
      this.change.emit()
    }
  }

  on_visible_change(){
    if(this.watched && this._needs_update){
      this.change.emit()
    }   
  }
}
