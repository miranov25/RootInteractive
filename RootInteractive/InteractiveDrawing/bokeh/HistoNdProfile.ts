import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {HistoNdCDS} from "./HistoNdCDS"
import * as p from "core/properties"

export namespace HistoNdProfile {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<HistoNdCDS>
    axis_idx: p.Property<number>
    quantiles: p.Property<number[]>
    sum_range: p.Property<number[][]>
  }
}

export interface HistoNdProfile extends HistoNdProfile.Attrs {}

export class HistoNdProfile extends ColumnarDataSource {
  properties: HistoNdProfile.Props

  _stale: boolean

  constructor(attrs?: Partial<HistoNdProfile.Attrs>) {
    super(attrs)
  }

  static __name__ = "HistoNdProfile"

  static init_HistoNdProfile() {

    this.define<HistoNdProfile.Props>(({Ref, Array, Number, Int})=>({
      source:  [Ref(HistoNdCDS)],
      axis_idx: [Int, 0],
      quantiles: [Array(Number), []], // This is the list of all quantiles to compute, length is NOT equal to CDS length
      sum_range: [Array(Array(Number)), []]
    }))
  }

  initialize(): void {
    super.initialize()

    this.data = {}
    this._stale = true
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.source.change, () => {
      this._stale = true
      this.change.emit()
    })
  }

  update(): void {
        let mean_column = []
        let std_column = []
        let entries_column = []
        let isOK_column = []
        let quantile_columns:Array<Array<number>> = []
        for (let i = 0; i < this.quantiles.length; i++) {
          quantile_columns.push([])
        }
        let integral_columns:Array<Array<number>> = []
        for (let i = 0; i < this.sum_range.length; i++) {
          integral_columns.push([])
        }
        let efficiency_columns:Array<Array<number>> = []
        for (let i = 0; i < this.sum_range.length; i++) {
          efficiency_columns.push([])
        }

        const stride_low = this.source.get_stride(this.axis_idx)
        const stride_high = this.source.get_stride(this.axis_idx+1)

        for(let i=0; i<this.source.dim; i++){
          if(i != this.axis_idx){
              this.data["bin_bottom_"+i] = []
              this.data["bin_center_"+i] =[]
              this.data["bin_top_"+i] = []
          }
        }

        let bin_centers_all:Array<Array<number>> = []
        let bin_top_all:Array<Array<number>> = []
        let bin_bottom_all:Array<Array<number>> = []
        let bin_centers_filtered:Array<Array<number>> = []
        let bin_top_filtered:Array<Array<number>> = []
        let bin_bottom_filtered:Array<Array<number>> = []
        for (let i = 0; i < this.source.dim; i++) {
          bin_centers_all.push(this.source.get_column("bin_center_"+i) as number[])
          bin_centers_filtered.push([])
          bin_top_all.push(this.source.get_column("bin_top_"+i) as number[])
          bin_top_filtered.push([])
          bin_bottom_all.push(this.source.get_column("bin_bottom_"+i) as number[])
          bin_bottom_filtered.push([])
        }

        const bin_count = this.source.get_column("bin_count") as number[]
        const bin_centers = this.source.get_column("bin_center_"+this.axis_idx) as number[]
        const edges_left = this.source.get_column("bin_bottom_"+this.axis_idx) as number[]
        const edges_right = this.source.get_column("bin_top_"+this.axis_idx) as number[]

        const nbins_total = this.source.length

        for(let x = 0; x < nbins_total; x += stride_high){
          for(let z = 0; z < stride_low; z ++){
      //      console.log(x)
      //      console.log(z)
            for(let i=0; i<this.source.dim; i++){
              if(i != this.axis_idx){
              //    (this.data["bin_bottom_"+i] as any[]).push(this.source.get_array("bin_bottom_"+i)[z/length+x])
                  bin_bottom_filtered[i].push(bin_bottom_all[i][x+z])
                  bin_centers_filtered[i].push(bin_centers_all[i][x+z])
                  bin_top_filtered[i].push(bin_top_all[i][x+z])
              //    (this.data["bin_top_"+i] as any[]).push(this.source.get_array("bin_top_"+i)[z/length+x])
              }
            }
            if(bin_count === null || bin_centers === null ){
              mean_column.push(NaN)
              std_column.push(NaN)
              entries_column.push(NaN)
              isOK_column.push(false)
              continue
            }
      //      let histogram_axis = []
            let cumulative_histogram = []
            let entries = 0
            for(let y=0; y < stride_high; y += stride_low){
              entries += bin_count[x+y+z]
              cumulative_histogram.push(entries)
            }
            if(entries > 0){
              let mean = 0
              for (let y = 0; y < stride_high; y+= stride_low) {
                mean += bin_count[x+y+z] * bin_centers[x+y+z]
              }
              mean /= entries
              let std = 0
              for (let y = 0; y < stride_high; y+= stride_low) {
                std += (bin_centers[x+y+z] - mean) * (bin_centers[x+y+z] - mean) * bin_count[x+y+z]
              }
              std /= entries
              std = Math.sqrt(std)
              mean_column.push(mean)
              std_column.push(std)
              entries_column.push(entries)
              isOK_column.push(true)
              let low = 0
              let high = 0
              let iQuantile = 0
              while(iQuantile < this.quantiles.length && this.quantiles[iQuantile] < 0){
                quantile_columns[iQuantile].push(NaN)
                iQuantile++
              }
              for (let j = 0; j < cumulative_histogram.length && iQuantile < this.quantiles.length; j++) {
                high = cumulative_histogram[j]
                while(iQuantile < this.quantiles.length && high > this.quantiles[iQuantile] * entries){
                  const m = (this.quantiles[iQuantile] * entries - low)/(high-low)
                  quantile_columns[iQuantile].push(edges_left[j*stride_low]*(1-m)+edges_right[j*stride_low]*m)
                  iQuantile++
                }
                low = high
              }
              while(iQuantile < this.quantiles.length){
                quantile_columns[iQuantile].push(NaN)
                iQuantile++
              }
            } else {
              mean_column.push(NaN)
              std_column.push(NaN)
              entries_column.push(entries)
              isOK_column.push(false)
              for (let iQuantile = 0; iQuantile < quantile_columns.length; iQuantile++) {
                quantile_columns[iQuantile].push(NaN)
              }
            }
            // XXX: This assumes uniform binning
            const low = edges_left[0]
            const high = edges_right[edges_right.length-1]
            const scale = cumulative_histogram.length / (high - low)
            const origin = -low * scale
            for (let iBox = 0; iBox < this.sum_range.length; iBox++) {
              const bounding_box = this.sum_range[iBox];
              const index_left = bounding_box[0] * scale + origin
              let val_left = 0
              let val_right = 0
              if(index_left < 0){
                val_left = 0
              } else if(index_left < 1){
                val_left = index_left * cumulative_histogram[0]
              } else if(index_left < cumulative_histogram.length){
                const m_left = index_left % 1
                val_left = cumulative_histogram[(index_left|0)-1] * (1-m_left) + cumulative_histogram[index_left|0] * m_left
              } else {
                val_left = entries
              }
              const index_right = bounding_box[1] * scale + origin
              if(index_right < 0){
                val_right = 0
              } else if(index_right < 1){
                val_right = index_right * cumulative_histogram[0]
              } else if(index_right < cumulative_histogram.length){
                const m_right = index_right % 1
                val_right = cumulative_histogram[(index_right|0)-1] * (1-m_right) + cumulative_histogram[index_right|0] * m_right
              } else {
                val_right = entries
              }
              const integral = val_right - val_left
              integral_columns[iBox].push(integral)
              efficiency_columns[iBox].push(integral/entries)
            }
          }
        }

        this.data["mean"] = mean_column
        this.data["std"] = std_column
        this.data["entries"] = entries_column
        this.data["isOK"] = isOK_column
        for (let iQuantile = 0; iQuantile < quantile_columns.length; iQuantile++) {
          this.data["quantile_"+iQuantile] = quantile_columns[iQuantile]
        }
        for (let iBox = 0; iBox < integral_columns.length; iBox++) {
          this.data["sum_"+iBox] = integral_columns[iBox]
          this.data["sum_normed_"+iBox] = efficiency_columns[iBox]
        }
        for (let i = 0; i < this.source.dim; i++) {
          if(i != this.axis_idx){
              this.data["bin_bottom_"+i] = bin_bottom_filtered[i]
              this.data["bin_center_"+i] = bin_centers_filtered[i]
              this.data["bin_top_"+i] = bin_top_filtered[i]
          }
        }
        this._stale = false
  }

  get_column(key: string){
    if(this._stale){
      this.update()
    }
    return this.data[key]
  }

  get_length(){
    return this.source.get_length() / this.source.nbins[this.axis_idx]
  }

}
