import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {ColumnDataSource} from "models/sources/column_data_source"
import * as p from "core/properties"

export namespace HistogramCDS {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnDataSource>
    view: p.Property<number[] | null>
    nbins:        p.Property<number>
    range_min:    p.Property<number>
    range_max:    p.Property<number>
    sample:      p.Property<string>
    weights:      p.Property<string | null>
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

    this.define<HistogramCDS.Props>(({Nullable, Ref, Array, Int})=>({
      source:  [Ref(ColumnDataSource)],
      view:         [Nullable(Array(Int)), null],
      nbins:        [p.Number],
      range_min:    [p.Number],
      range_max:    [p.Number],
      sample:      [p.String],
      weights:      [p.String, null]
    }))
  }

  initialize(): void {
    super.initialize()

    this.data = {"bin_count":[], "bin_left":[], "bin_right":[]}
    this.update_range()
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.source.change, () => this.update_data())
  }

  update_data(indices: number[] | null = null): void {
      let bincount = null
      if (this.data["bin_count"] != null){
        bincount = this.data["bin_count"] as number[]
        bincount.length = this.nbins
      } else {
        bincount = new Array<number>(this.nbins, 0)
      }
      if(indices != null){
        //TODO: Make this actually do something
      } else {
        bincount.fill(0, 0, this.nbins)
        const sample_array = this.source.data[this.sample]
        const view_indices = this.view
        if(view_indices === null){
          const n_indices = this.source.length
          if(this.weights != null){
            const weights_array = this.source.data[this.weights]
            for(let i=0; i<n_indices; i++){
              const bin = this.getbin(sample_array[i])
              if(bin >= 0 && bin < this.nbins){
                bincount[bin] += weights_array[i]
              }
            }
          } else {
            for(let i=0; i<n_indices; i++){
              const bin = this.getbin(sample_array[i])
              if(bin >= 0 && bin < this.nbins){
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
              const bin = this.getbin(sample_array[j])
              if(bin >= 0 && bin < this.nbins){
                bincount[bin] += weights_array[j]
              }
            }
          } else {
            for(let i=0; i<n_indices; i++){
              const bin = this.getbin(sample_array[view_indices[i]])
              if(bin >= 0 && bin < this.nbins){
                bincount[bin] += 1
              }
            }
          }
        }

      }
      this.data["bin_count"] = bincount
      this.change.emit()
  }

  private _transform_origin: number
  private _transform_scale: number

  private _range_min: number
  private _range_max: number

  update_range(): void {
      // TODO: This is a hack and can be done in a much more efficient way that doesn't save bin edges as an array
      const bin_left = (this.data["bin_left"] as number[])
      const bin_right = (this.data["bin_right"] as number[])
      bin_left.length = 0
      bin_right.length = 0
      this._range_min = this._range_min
      this._range_max = this._range_max
      this._transform_scale = this.nbins/(this.range_max-this.range_min)
      this._transform_origin = -this.range_min*this._transform_scale
      for (let index = 0; index < this.nbins; index++) {
        bin_left.push(this.range_min+index*(this.range_max-this.range_min)/this.nbins)
        bin_right.push(this.range_min+(index+1)*(this.range_max-this.range_min)/this.nbins)
      }
      this.update_data()
  }

  getbin(val: number): number{
      // Overflow bins
      if(val > this._range_max) return this.nbins
      if(val < this._range_min) return -1
      // Make the max value inclusive
      if(val === this._range_max) return this.nbins-1
      // Compute the bin in the normal case
      return (val*this._transform_scale+this._transform_origin)|0
  }

}
