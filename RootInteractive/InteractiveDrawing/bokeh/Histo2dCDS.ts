import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {ColumnDataSource} from "models/sources/column_data_source"
import * as p from "core/properties"

export namespace Histo2dCDS {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnDataSource>
//    view: p.Property<number[] | null>
    nbins:        p.Property<number[]>
    range:    p.Property<(number[] | null)[] | null>
    sample_x:      p.Property<string>
    sample_y:      p.Property<string>
    weights:      p.Property<string | null>
  }
}

export interface Histo2dCDS extends Histo2dCDS.Attrs {}

export class Histo2dCDS extends ColumnarDataSource {
  properties: Histo2dCDS.Props

  constructor(attrs?: Partial<Histo2dCDS.Attrs>) {
    super(attrs)
  }

  static __name__ = "Histo2dCDS"

  static init_Histo2dCDS() {

    this.define<Histo2dCDS.Props>(({Ref, Array, Nullable, Number, Int})=>({
      source:  [Ref(ColumnDataSource)],
//      view:         [Nullable(Array(Int)), null], - specifying this as a bokeh property causes a drastic drop in performance
      nbins:        [Array(Int)],
      range:    [Nullable(Array(Nullable(Array(Number))))],
      sample_x:      [p.String],
      sample_y:      [p.String],
      weights:      [p.String, null]
    }))
  }

  initialize(): void {
    super.initialize()

    this.data = {"bin_count":[], "bin_left":[], "bin_right":[], "bin_top":[], "bin_bottom":[]}
    this.view = null
    this._bin_indices = []
    this.update_range()
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.source.change, () => {
      this._bin_indices.length = this.source.length
      this._bin_indices.fill(-2, 0, this.source.length)
    })
  }

  update_data(indices: number[] | null = null): void {
      let bincount = this.data["bin_count"] as number[]
      const length = this.nbins[0] * this.nbins[1]
      bincount.length = length
      if(indices != null){
        //TODO: Make this actually do something
      } else {
        bincount.fill(0, 0, length)
        const sample_arr_x = this.source.data[this.sample_x] as number[]
        const sample_arr_y = this.source.data[this.sample_y] as number[]
        const view_indices = this.view
        if(view_indices === null){
          const n_indices = this.source.length
          if(this.weights != null){
            const weights_array = this.source.data[this.weights]
            for(let i=0; i<n_indices; i++){
              const bin = this.getbin(i, sample_arr_x, sample_arr_y)
              if(bin >= 0 && bin < length){
                bincount[bin] += weights_array[i]
              }
            }
          } else {
            for(let i=0; i<n_indices; i++){
              const bin = this.getbin(i, sample_arr_x, sample_arr_y)
              if(bin >= 0 && bin < length){
                bincount[bin] += 1
              }
            }
          }
        } else {
          const n_indices = view_indices.length
          if(this.weights != null){
            const weights_array = this.source.data[this.weights]
            for(let i=0; i<n_indices; i++){
              let j = view_indices[i]
              const bin = this.getbin(j, sample_arr_x, sample_arr_y)
              if(bin >= 0 && bin < length){
                bincount[bin] += weights_array[j]
              }
            }
          } else {
            for(let i=0; i<n_indices; i++){
              const bin = this.getbin(view_indices[i], sample_arr_x, sample_arr_y)
              if(bin >= 0 && bin < length){
                bincount[bin] += 1
              }
            }
          }
        }

      }
      this.data["bin_count"] = bincount
      this.change.emit()
  }

  private _transform_origin_x: number
  private _transform_origin_y: number
  private _transform_scale_x: number
  private _transform_scale_y: number
  private _stride: number

  private _range_min: number[]
  private _range_max: number[]
  private _nbins: number[]

  public view: number[] | null

  private _bin_indices: number[] // Bin index caching

  update_range(): void {
      // TODO: This is a hack and can be done in a much more efficient way that doesn't save bin edges as an array
      const bin_left = (this.data["bin_left"] as number[])
      const bin_right = (this.data["bin_right"] as number[])
      const bin_top = (this.data["bin_top"] as number[])
      const bin_bottom = (this.data["bin_bottom"] as number[])
      const sample_arr_x = this.source.data[this.sample_x] as number[]
      const sample_arr_y = this.source.data[this.sample_y] as number[]
      bin_left.length = 0
      bin_right.length = 0
      bin_top.length = 0
      bin_bottom.length = 0
      this._range_min = [0, 0]
      this._range_max = [0, 0]
      this._nbins = [0, 0]
      // This code seems stupid
      if(this.range === null){
        this._range_min[0] = sample_arr_x.reduce((acc, cur) => Math.min(acc, cur), sample_arr_x[0])
        this._range_max[0] = sample_arr_x.reduce((acc, cur) => Math.max(acc, cur), sample_arr_x[0])
        this._range_min[1] = sample_arr_y.reduce((acc, cur) => Math.min(acc, cur), sample_arr_y[0])
        this._range_max[1] = sample_arr_y.reduce((acc, cur) => Math.max(acc, cur), sample_arr_y[0])
      } else {
        if(this.range[0] === null) {
          this._range_min[0] = sample_arr_x.reduce((acc, cur) => Math.min(acc, cur), sample_arr_x[0])
          this._range_max[0] = sample_arr_x.reduce((acc, cur) => Math.max(acc, cur), sample_arr_x[0])
        } else {
          this._range_min[0] = this.range[0][0]
          this._range_max[0] = this.range[0][1]
        }
        if(this.range[1] === null) {
          this._range_min[1] = sample_arr_y.reduce((acc, cur) => Math.min(acc, cur), sample_arr_y[0])
          this._range_max[1] = sample_arr_y.reduce((acc, cur) => Math.max(acc, cur), sample_arr_y[0])
        } else {
          this._range_min[1] = this.range[1][0]
          this._range_max[1] = this.range[1][1]
        }
      }
      this._nbins[0] = this.nbins[0]
      this._nbins[1] = this.nbins[1]
      this._transform_scale_x = this._nbins[0]/(this._range_max[0]-this._range_min[0])
      this._transform_scale_y = this._nbins[1]/(this._range_max[1]-this._range_min[1])
      this._transform_origin_x = -this._range_min[0]*this._transform_scale_x
      this._transform_origin_y = -this._range_min[1]*this._transform_scale_y
      this._stride = this._nbins[0]
      const length = this._nbins[0] * this._nbins[1]
      for (let index = 0; index < length; index++) {
        bin_left.push(this._range_min[0]+((index/this._stride)|0)*(this._range_max[0]-this._range_min[0])/this._nbins[0])
        bin_right.push(this._range_min[0]+(((index/this._stride)|0)+1)*(this._range_max[0]-this._range_min[0])/this._nbins[0])
        bin_bottom.push(this._range_min[1]+(index%this._stride)*(this._range_max[1]-this._range_min[1])/this._nbins[1])
        bin_top.push(this._range_min[1]+(index%this._stride+1)*(this._range_max[1]-this._range_min[1])/this._nbins[1])
      }
      this._bin_indices.length = this.source.length
      this._bin_indices.fill(-2, 0, this.source.length)
      this.update_data()
  }

  getbin(idx: number, x_arr: number[], y_arr: number[]): number{
      // This can be optimized using loop fission, but in our use case the data doeswn't change often, which means
      // that the cached bin indices are invalidated infrequently.
      const cached_value = this._bin_indices[idx]
      if(cached_value != -2) return cached_value
      const val_x = x_arr[idx]
      const val_y = y_arr[idx]
      // Overflow bins
      if(val_x < this._range_min[0] || val_x > this._range_max[0]) return -1
      if(val_y < this._range_min[1] || val_y > this._range_max[1]) return -1

      let bin = 0
      // Make the max value inclusive
      if(val_x === this._range_max[0]){
        bin += this._nbins[0] * this._stride
      } else {
        bin += ((val_x * this._transform_scale_x - this._transform_origin_x) | 0) * this._stride
      }
      if(val_y === this._range_max[1]){
        bin += this._nbins[1]
      } else {
        bin += (val_y * this._transform_scale_y - this._transform_origin_y) | 0
      }
      this._bin_indices[idx] = bin
      return bin
  }

}
