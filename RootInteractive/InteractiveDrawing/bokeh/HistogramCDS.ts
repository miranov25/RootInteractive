import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {ColumnDataSource} from "models/sources/column_data_source"
import {RIFilter} from "./RIFilter"
import * as p from "core/properties"
//import { kth_value, weighted_kth_value } from "./MathUtils"

export namespace HistogramCDS {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnDataSource>
    filter: p.Property<RIFilter | null>
    nbins:        p.Property<number>
    range:    p.Property<number[] | null>
    sample:      p.Property<string>
    weights:      p.Property<string | null>
    histograms: p.Property<Record<string, Record<string, any> | null>>
  }
}

export interface HistogramCDS extends HistogramCDS.Attrs {}

export class HistogramCDS extends ColumnarDataSource {
  properties: HistogramCDS.Props

  constructor(attrs?: Partial<HistogramCDS.Attrs>) {
    super(attrs)
  }

  static __name__ = "HistogramCDS"

  static init_HistogramCDS() {

    this.define<HistogramCDS.Props>(({Ref, Number, Array, Nullable, String, Any})=>({
      source:  [Ref(ColumnDataSource)],
      filter:         [Nullable(Ref(RIFilter)), null],
      nbins:        [Number],
      range:    [Nullable(Array(Number))],
      sample:      [String],
      weights:      [Nullable(String), null],
      histograms:  [Any, {}]
    }))
  }

  initialize(): void {
    super.initialize()

    this.data = {}
    this.view = null
    this._stale_range = true
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.source.change, () => {this.change_selection()})
    this.connect(this.properties.weights.change, () => {this.change_selection()})
    this.connect(this.properties.sample.change, () => {this.change_selection()})
    this.connect(this.properties.nbins.change, () => {this.change_selection()})
    this.connect(this.properties.range.change, () => {this.change_selection()})
    if(this.filter != null){
      this.connect(this.filter.change, () => {this.change_selection()})
    }
  }

  private _transform_origin: number
  private _transform_scale: number

  private _range_min: number
  private _range_max: number
  private _nbins: number

  public view: number[] | null

  _stale_range: boolean

  update_range(): void {
      // TODO: This is a hack and can be done in a much more efficient way that doesn't save bin edges as an array
      const bin_left: number[] = []
      const bin_center: number[] = []
      const bin_right: number[] = []
      if(this.view === null){
        if(this.range === null ){
          const sample_arr = this.source.get_column(this.sample)
          if(sample_arr == null){
            throw ReferenceError("Column " + this.sample + " not found in source " + this.source.name)
          }
          let range_min = Infinity
          let range_max = -Infinity
          const l = sample_arr.length
          for(let x=0; x<l; x++){
            if(isFinite(sample_arr[x])){
              range_min = Math.min(range_min, sample_arr[x])
              range_max = Math.max(range_max, sample_arr[x])
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
          this._range_min = range_min
          this._range_max = range_max
        } else {
          this._range_min = this.range[0]
          this._range_max = this.range[1]
        }
      } else {
        if(this.range === null ){
          const sample_arr = this.source.get_column(this.sample)
          if(sample_arr == null){
            throw ReferenceError("Column " + this.sample + " not found in source " + this.source.name)
          }
          let range_min = Infinity
          let range_max = -Infinity
          const view = this.view
          const l = this.view.length
          for(let x=0; x<l; x++){
            const y = view[x]
            if(isFinite(sample_arr[y])){
              range_min = Math.min(range_min, sample_arr[y])
              range_max = Math.max(range_max, sample_arr[y])
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
          this._range_min = range_min
          this._range_max = range_max
        } else {
          this._range_min = this.range[0]
          this._range_max = this.range[1]
        }
      }
      this._nbins = this.nbins
      this._transform_scale = this._nbins/(this._range_max-this._range_min)
      this._transform_origin = -this._range_min*this._transform_scale
      for (let index = 0; index < this._nbins; index++) {
        bin_left.push(this._range_min+index*(this._range_max-this._range_min)/this._nbins)
        bin_center.push(this._range_min+(index+.5)*(this._range_max-this._range_min)/this._nbins)
        bin_right.push(this._range_min+(index+1)*(this._range_max-this._range_min)/this._nbins)
      }
      this.data["bin_bottom"] = bin_left
      this.data["bin_top"] = bin_right
      this.data["bin_center"] = bin_center
      this._stale_range = false
  }

  histogram(weights: string | null = null, weights_transform: ((x:number) => number) | null=null, initial_value:number = 0 ): number[]{
    const bincount = Array<number>(this._nbins)
    bincount.fill(initial_value)
    const sample_array = this.source.get_column(this.sample)
    if(sample_array == null){
      throw ReferenceError("Column " + this.sample + " not found in source")
    }
    const view_indices = this.view
    if(view_indices === null){
      const n_indices = this.source.length
      if(weights != null){
        const weights_array = this.source.get_column(weights)
        if (weights_array == null){
          throw ReferenceError("Column not defined: "+ weights)
        }
        if (weights_transform != null){
          for(let i=0; i<n_indices; i++){
            const bin = this.getbin(sample_array[i])
            if(bin >= 0 && bin < this._nbins){
              bincount[bin] += weights_transform(weights_array[i])
            }
          }          
        } else {
          for(let i=0; i<n_indices; i++){
            const bin = this.getbin(sample_array[i])
            if(bin >= 0 && bin < this._nbins){
              bincount[bin] += weights_array[i]
            }
          }
        }
      } else {
        for(let i=0; i<n_indices; i++){
          const bin = this.getbin(sample_array[i])
          if(bin >= 0 && bin < this._nbins){
            bincount[bin] += 1
          }
        }
      }
    } else {
      const n_indices = view_indices.length
      if(weights != null){
        const weights_array = this.source.get_column(weights)
        if (weights_array == null){
          throw ReferenceError("Column not defined: "+ weights)
        }
        for(let i=0; i<n_indices; i++){
          let j = view_indices[i]
          const bin = this.getbin(sample_array[j])
          if(bin >= 0 && bin < this._nbins){
            bincount[bin] += weights_array[j]
          }
        }
      } else {
        for(let i=0; i<n_indices; i++){
          const bin = this.getbin(sample_array[view_indices[i]])
          if(bin >= 0 && bin < this._nbins){
            bincount[bin] += 1
          }
        }
      }
    }
    return bincount
  }

  getbin(val: number): number{
      // Overflow bins
      if(val > this._range_max) return this._nbins
      if(val < this._range_min) return -1
      // Make the max value inclusive
      if(val === this._range_max) return this._nbins-1
      // Compute the bin in the normal case
      return (val*this._transform_scale+this._transform_origin)|0
  }

  get_length(){
    return this.nbins
  }

  public change_selection(){
    this._stale_range = true
    this.data = {}
    if(this.filter != null && this.filter.active){
      this.view = this.filter.get_indices()
    } else {
      this.view = null
    }
    this.change.emit()
  }

  get_column(key: string){
    if(this._stale_range){
      this.update_range()
    }
    if(this.data[key] != null){
      return this.data[key]
    }
    this.compute_function(key)
    if(this.data[key] != null){
      return this.data[key]
    }
    return null
  }

  compute_function(key: string){
    const {histograms, data} = this 
    if(key == "bin_count"){
      data[key] = this.histogram(this.weights)
    } else if(key === "errorbar_high"){
      const bincount = this.get_column("bin_count")!
      const errorbar_edge = Array<number>(this._nbins)
      for(let i=0; i<this._nbins; i++){
        errorbar_edge[i] = bincount[i] + Math.sqrt(bincount[i])
      }
      data[key] = errorbar_edge
    } else if(key === "errorbar_low"){
      const bincount = this.get_column("bin_count")!
      const errorbar_edge = Array<number>(this._nbins)
      for(let i=0; i<this._nbins; i++){
        errorbar_edge[i] = bincount[i] - Math.sqrt(bincount[i])
      }
      data[key] = errorbar_edge
    } else if(histograms != null){
      if(histograms[key] == null){
        data[key] = this.histogram(null)
      } else {
        const h = histograms[key]!
        let histogram
        if(h.aggregation){
          if(h.aggregation === "mean"){
            histogram = this.histogram(h.y)
            let sum_weights
            if(h.weights){
              sum_weights = this.histogram(h.weights)
            } else {
              sum_weights = this.histogram()
            }
            for(let i=0; i<histogram.length; ++i){
              histogram[i] /= sum_weights[i]
            }
          } else {
            histogram = this.histogram(h.weights)
          }
        } else {
          histogram = this.histogram(h.weights)
        }
        if(h.cumulative){
          let acc = 0
          let cumulative = []
          for(const x of histogram){
            acc += x
            cumulative.push(acc)
          }
          if(h.density){
            for(let i=0; i<cumulative.length; i++){
              cumulative[i] /= acc
            }
          }
          data[key] = cumulative
        } else {
          if(h.density){
            const c = histogram.reduce((acc, cur) => acc+cur) / this._transform_scale
            for(let i=0; i<histogram.length; i++){
              histogram[i] /=  c
            }
          }          
          data[key] = histogram
        }
      }
    } 
  }

}
