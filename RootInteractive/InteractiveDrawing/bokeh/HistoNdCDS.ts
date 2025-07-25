import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {Model} from "model"
import * as p from "core/properties"

export namespace HistoNdCDS {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnarDataSource>
    filter: p.Property<any | null>
    nbins:        p.Property<number[]>
    range:    p.Property<(number[] | null)[] | null>
    sample_variables:      p.Property<string[]>
    weights:      p.Property<string | null>
    histograms: p.Property<Record<string, Record<string, any>>>
  }
}

export interface HistoNdCDS extends HistoNdCDS.Attrs {}

export class HistoNdCDS extends ColumnarDataSource {
  properties: HistoNdCDS.Props

  constructor(attrs?: Partial<HistoNdCDS.Attrs>) {
    super(attrs)

    this._range_min = []
    this._range_max = []
    this._transform_origin = []
    this._transform_scale = []
    this._strides = []
    this._nbins = []
  }

  static __name__ = "HistoNdCDS"

  static {

    this.define<HistoNdCDS.Props>(({Ref, Array, Nullable, Number, Int, String, Any})=>({
      source:  [Ref(ColumnarDataSource)],
      filter:         [Nullable(Ref(Model)), null], 
      nbins:        [Array(Int)],
      range:    [Nullable(Array(Nullable(Array(Number))))],
      sample_variables:      [Array(String)],
      weights:      [Nullable(String), null],
      histograms:  [Any, {}]
    }))
  }

  initialize(): void {
    super.initialize()

    this.data = {}
    this.view = null
    this._bin_indices = null
    this._bin_indices_dirty = true
    this.invalidate_cached_bins()
    this.update_nbins()
    this._stale_range = true
    this.changed_histogram = false
    this.cached_columns = new Set()
    this.change_selection()
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.source.change, () => {
      this.invalidate_cached_bins()
      this.change_selection()
    })
    this.connect(this.properties.weights.change, () => {
      this.change_weights()
    })
    if(this.filter != null){
      this.connect(this.filter.change, () => {this.change_selection()})
    }
  }

  public view: number[] | null
  private _sorted_indices: number[] | null

  public dim: number

  private _bin_indices: Int32Array | null
  private _bin_indices_dirty: boolean

  private _range_min: number[]
  private _range_max: number[]
  private _nbins: number[]
  private _transform_origin: number[]
  private _transform_scale: number[]
  private _strides: number[]

  private _stale_range: boolean

  private _unweighted_histogram: number[] | null

  private _sorted_column_pool: number[][]

  public changed_histogram: boolean

  private cached_columns: Set<string>

  update_range(): void {
    this._nbins = this.nbins;
    this._sorted_indices = null
    this._unweighted_histogram = null

    this._sorted_column_pool = []

    let sample_array: ArrayLike<number>[] = []
    if(this.range === null || this.range.reduce((acc: boolean, cur) => acc || (cur === null), false))
    for (const column_name of this.sample_variables) {
      const column = this.source.get_column(column_name)
      if (column == null){
        throw new Error("Column " + column_name + " not found in source " + this.source.name);
      }
      sample_array.push(column)
    }
      // This code seems stupid
      if(this.view === null){
        if(this.range === null){
          for (let i = 0; i < this._nbins.length; i++) {
            const column = sample_array[i]
            this.auto_range(column, i)
          }
        } else {
          for (let i = 0; i < this.range.length; i++) {
            const r = this.range[i]
            if(r === null) {
              const column = sample_array[i]
              this.auto_range(column, i)
            } else {
              this._range_min[i] = r[0]
              this._range_max[i] = r[1]
            }
          }
        }
      } else {
        if(this.range === null){
          for (let i = 0; i < this._nbins.length; i++) {
            const column = sample_array[i]
            this.auto_range_indices(column, i)
          }
        } else {
          for (let i = 0; i < this.range.length; i++) {
            const r = this.range[i]
            if(r === null) {
              const column = sample_array[i]
              this.auto_range_indices(column, i)
            } else {
              this._range_min[i] = r[0]
              this._range_max[i] = r[1]
            }
          }
        }        
      }
      const dim = this.nbins.length
      this._nbins.length = dim
      this._transform_scale.length = dim
      this._transform_origin.length = dim
      this._strides.length = dim+1
      this._strides[0] = 1
      for(let i=0; i<dim; i++){
        this._nbins[i] = this.nbins[i]
        this._transform_scale[i] = this._nbins[i]/(this._range_max[i]-this._range_min[i])
        this._transform_origin[i] = this._range_min[i]*this._transform_scale[i]
        this._strides[i+1] = this._strides[i]*this._nbins[i]
      }

      const length = this._strides[dim]
      for(let i=0; i<dim; i++){
        let bin_bottom = []
        let bin_center = []
        let bin_top = []
        let inv_scale = 1/this._transform_scale[i]
        for (let index = 0; index < length; index++) {
          let true_index = ((index%this._strides[i+1])/this._strides[i])|0
          bin_bottom.push(this._range_min[i]+true_index*inv_scale)
          bin_center.push(this._range_min[i]+(true_index+.5)*inv_scale)
          bin_top.push(this._range_min[i]+(true_index+1)*inv_scale)
        }
        this.data["bin_bottom_"+i] = bin_bottom
        this.data["bin_center_"+i]  = bin_center
        this.data["bin_top_"+i]  = bin_top
        this.cached_columns.add("bin_bottom_"+i)
        this.cached_columns.add("bin_center_"+i)
        this.cached_columns.add("bin_top_"+i)
      }

      this.dim = dim
      this._stale_range = false
      this._bin_indices_dirty = true
  }

  histogram(weights: string | null, weights_transform: ((x:number) => number) | null=null): number[]{
    if(weights == null){
      if(this._unweighted_histogram != null){
        return this._unweighted_histogram
      }
    }
    console.log("Histogram " + this.name + " " + weights)
    if(this._sorted_indices != null && weights != null){
      return this.histogram_sorted(weights, weights_transform)
    }
    const length = this._strides[this._strides.length-1]
    let sample_array: ArrayLike<number>[] = []
    for (const column_name of this.sample_variables) {
      sample_array.push(this.source.get_column(column_name)!)
    }
    for(let i=0; i<this._nbins.length; i++){
      if(Math.abs(this._range_min[i]*this._transform_scale[i]-this._transform_origin[i])>1e-6){
        console.log(this._range_min[i]*this._transform_scale[i]-this._transform_origin[i])
        throw Error("Assertion error: Range minimum in histogram" + this.name + "is broken")
      }
      if(Math.abs(this._range_max[i]*this._transform_scale[i]-this._transform_origin[i] - this._nbins[i])>1e-6){
        throw Error("Assertion error: Range maximum in histogram" + this.name + "is broken")
      }
    }
    let bincount: number[] = Array(length)
    bincount.fill(0)
    const view_indices = this.view
    if(this._bin_indices_dirty){
      this._bin_indices = this.compute_bins(sample_array)
      this._bin_indices_dirty = false
    }
    if(view_indices === null){
      const n_indices = this.source.length
      if(weights != null){
        const weights_array = this.source.get_column(weights)
        if (weights_array == null){
          throw ReferenceError("Column "+ weights + " not found in " + this.source.name)
        }
        if(weights_transform != null){
          for(let i=0; i<n_indices; i++){
            const bin = this.getbin(i, sample_array)
            if(bin >= 0 && bin < length){
              bincount[bin] += weights_transform(weights_array[i])
            }
          }
        } else {
          for(let i=0; i<n_indices; i++){
            const bin = this.getbin(i, sample_array)
            if(bin >= 0 && bin < length){
              bincount[bin] += weights_array[i]
            }
          }
        }
      } else {
        for(let i=0; i<n_indices; i++){
          const bin = this.getbin(i, sample_array)
          if(bin >= 0 && bin < length){
            bincount[bin] += 1
          }
        }
      }
    } else {
      const n_indices = this.source.length
      if(weights != null){
        const weights_array = this.source.get_column(weights)
        if (weights_array == null){
          throw ReferenceError("Column "+ weights + " not found in " + this.source.name)
        }
        for(let i=0; i<n_indices; i++){
          if (!view_indices[i])
            continue
          const bin = this.getbin(i, sample_array)
          if(bin >= 0 && bin < length){
            bincount[bin] += weights_array[i]
          }
        }
      } else {
        for(let i=0; i<n_indices; i++){
          if (!view_indices[i])
            continue
          const bin = this.getbin(i, sample_array)
          if(bin >= 0 && bin < length){
            bincount[bin] += 1
          }
        }
      }
    }
    if(weights == null){
      this._unweighted_histogram = bincount
    }
    return bincount
  }

  histogram_sorted(weights: string, weights_transform: ((x:number) => number)| null){
    let bincount: number[] = Array(length)
    const {_sorted_indices, source} = this
    const cumulative_histogram = this.cumulative_histogram_noweights()
    const weights_column = source.get_column(weights)
    if(weights_column == null){
      return this.histogram(null)
    }
    for(let i=0; i<cumulative_histogram.length; i++){
      let acc = 0
      const begin = i ? cumulative_histogram[i-1] : 0
      const end = cumulative_histogram[i]
      if(weights_transform != null){
        for(let j=begin; j<end; j++){
          acc += weights_transform(weights_column[_sorted_indices![j]])
        }
      } else{
        for(let j=begin; j<end; j++){
          acc += weights_column[_sorted_indices![j]]
        }
      }
      bincount[i] = acc
    }
    return bincount
  }

  getbin(idx: number, _sample: ArrayLike<number>[]): number{
      // Removed during
      return this._bin_indices![idx]
  }

  compute_bins(sample: ArrayLike<number>[]): Int32Array {
    const length = this.source.get_length()!
    const bins = this._bin_indices && this._bin_indices.length >= length ? this._bin_indices : new Int32Array(length)
    bins.fill(0, 0, length)
    for(let axis_idx=0; axis_idx<this._nbins.length; axis_idx++){
      const column = sample[axis_idx]
      const scale = this._transform_scale[axis_idx]
      const origin = this._transform_origin[axis_idx]
      const stride = this._strides[axis_idx]
      const nbins = this._nbins[axis_idx]
      const range_min = this._range_min[axis_idx]
      const range_max = this._range_max[axis_idx]
      console.log("HistoNdCDS: compute_bins", this.name, "axis_idx", axis_idx, "nbins", nbins, "range_min", range_min, "range_max", range_max)
      for(let i=0; i<length; i++){
        const val = column[i]
        // Overflow bins 
        if(val < range_min || val > range_max || !isFinite(val) || bins[i] < 0) {
          bins[i] = -1
          continue
        }
        // Make the max value inclusive
        if(val === range_max){
          bins[i] += (nbins - 1) * stride
        } else {
          bins[i] += ((val * scale - origin) | 0) * stride
        }
      }
    }
    return bins
  }

  public get_stride(idx: number): number{
    return this._strides[idx]
  }

  get_length(){
    const dim = this._strides.length-1
    return this._strides[dim]
  }

  invalidate_cached_bins(){
    this._bin_indices_dirty = true
  }

  get_column(key: string){
    if(this._stale_range){
      this.update_range()
    }
    if(this.cached_columns.has(key)){
      return this.data[key]
    }
    this.compute_function(key)
    if(this.cached_columns.has(key)){
      return this.data[key]
    }
    return null
  }

  compute_function(key: string){
    const {histograms, data} = this 
    if(key == "bin_count"){
      console.time("HistoNdCDS compute_function bin_count")
      data[key] = this.histogram(this.weights)
      console.timeEnd("HistoNdCDS compute_function bin_count")
    } else if(key == "errorbar_high"){
      const bincount = this.get_column("bin_count")!
      const l = this.get_length()
      const errorbar_edge = Array<number>(l)
      for(let i=0; i<l; i++){
        errorbar_edge[i] = bincount[i] + Math.sqrt(bincount[i])
      }
      data[key] = errorbar_edge
    } else if(key == "errorbar_low"){
      const bincount = this.get_column("bin_count")!
      const l = this.get_length()
      const errorbar_edge = Array<number>(l)
      for(let i=0; i<l; i++){
        errorbar_edge[i] = bincount[i] - Math.sqrt(bincount[i])
      }
      data[key] = errorbar_edge
    } else if(histograms != null){
      console.time("HistoNdCDS compute_function " + key)
      if(histograms[key] == null){
        data[key] = this.histogram(null)
      } else {
        data[key] = this.histogram(histograms[key].weights)
      }
      console.timeEnd("HistoNdCDS compute_function " + key)
    }
   this.cached_columns.add(key) 
  }

  auto_range(column: ArrayLike<number>, axis_idx: number){
    let range_min = Infinity
    let range_max = -Infinity
    const l = column.length
    for(let x=0; x<l; x++){
      const v = column[x]
      if(isFinite(v)){
        if (v < range_min) range_min = v
        if (v > range_max) range_max = v
      }
    }
    if(range_min == range_max){
      range_min -= 1
      range_max += 1
    }
    if(!isFinite(range_min)){
      range_min = 0
      range_max = 1
    }
    this._range_min[axis_idx] = range_min
    this._range_max[axis_idx] = range_max    
  }

  auto_range_indices(column: ArrayLike<number>, axis_idx: number){
    let range_min = Infinity
    let range_max = -Infinity
    const view = this.view!
    const l = view!.length
    for(let x=0; x<l; x++){
      const v = column[x]
      if(view[x] && isFinite(v)){
        if (v < range_min) range_min = v
        if (v > range_max) range_max = v
      }
    }
    if(range_min == range_max){
      range_min -= 1
      range_max += 1
    }
    if(!isFinite(range_min)){
      range_min = 0
      range_max = 1
    }
    this._range_min[axis_idx] = range_min
    this._range_max[axis_idx] = range_max    
  }

  public change_selection(){
    this._stale_range = true
    this._sorted_indices = null
    this._unweighted_histogram = null
    this.cached_columns.clear()
    if(this.filter != null && this.filter.active){
      this.view = this.filter.v_compute()
    } else {
      this.view = null
    }
    this.changed_histogram = true
    this.change.emit()
    this.changed_histogram = false
  }

  public change_weights(){
    this.cached_columns.delete("bin_count")
    this.change.emit()
  }

  update_nbins(){
    const dim = this.nbins.length
    this._nbins.length = dim
    this._strides.length = dim+1
    this._strides[0] = 1
    for(let i=0; i<dim; i++){
      this._nbins[i] = this.nbins[i]
      this._strides[i+1] = this._strides[i]*this._nbins[i]
    }
    this.dim = dim
  }

  public sorted_column_orig(key: string){
    // This is only used for non-associative histograms - exposed to speed up computing projections with stable quantiles / range sums
    const sorted_indices = this.compute_sorted_indices()
    const col = this.source.get_column(key)
    if(col == null) return null
    const l = sorted_indices.length
    let col_new = this._sorted_column_pool.pop()
    if(col_new == undefined){
      col_new = Array(l)
    }
    for(let i=0; i<l; i++){
      col_new[i] = col[sorted_indices[i]]
    }
    return col_new
  }

  public cumulative_histogram_noweights(){
    // TODO: Add caching - recomputing this histogram might be a bottleneck
    const histogram = this.histogram(null)
    let cumulativeHistogram = [...histogram]
    let acc = 0
    let l = histogram.length
    for (let i=0; i < l; i++){
      acc += histogram[i]
      cumulativeHistogram[i] = acc
    }
    return cumulativeHistogram
  }

  public compute_sorted_indices(){
    if(this._sorted_indices != null){
      return this._sorted_indices
    }
    const {view, source, sample_variables} = this
    let sample_array: ArrayLike<number>[] = []
    for (const column_name of sample_variables) {
      sample_array.push(source.get_column(column_name)!)
    }
    let working_indices = [...this.cumulative_histogram_noweights()]
    const n_entries = working_indices[working_indices.length-1]
    let histogram = this.histogram(null)
    for(let i=0; i<working_indices.length; i++){
      working_indices[i] -= histogram[i]
    }
    const view_sorted: number[] = Array(n_entries).fill(0)
    const l = source.get_length()!
    if(view == null){
      for(let i=0; i<l; i++){
        let bin_idx = this.getbin(i, sample_array)
        if(bin_idx >= 0){
          view_sorted[working_indices[bin_idx]] = i
          working_indices[bin_idx] ++
        }
      }
    } else {
      for(let i=0; i<l; i++){
        if(!view[i]) continue
        let bin_idx = this.getbin(i, sample_array)
        if(bin_idx >= 0){
          view_sorted[working_indices[bin_idx]] = i
          working_indices[bin_idx] ++
        }
      }
    }
    this._sorted_indices = view_sorted
    return view_sorted
  }

  public return_column_to_pool(column: number[]){
    this._sorted_column_pool.push(column)
  }

}
