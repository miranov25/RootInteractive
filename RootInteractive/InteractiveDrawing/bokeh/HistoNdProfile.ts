import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {HistoNdCDS} from "./HistoNdCDS"
import * as p from "core/properties"
import { kth_value, weighted_kth_value } from "./MathUtils"

export namespace HistoNdProfile {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<HistoNdCDS>
    axis_idx: p.Property<number>
    quantiles: p.Property<number[]>
    sum_range: p.Property<number[][]>
    unbinned: p.Property<boolean>
    weights: p.Property<(string | null)[] | string | null>
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

    this.define<HistoNdProfile.Props>(({Ref, Array, Number, Int, Boolean, String, Nullable, Or})=>({
      source:  [Ref(HistoNdCDS)],
      axis_idx: [Int, 0],
      quantiles: [Array(Number), []], // This is the list of all quantiles to compute, length is NOT equal to CDS length
      sum_range: [Array(Array(Number)), []],
      unbinned: [Boolean, false],
      weights: [Nullable(Or(Array(Nullable(String)), String)), null]
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
      if(!this.source.changed_histogram) return
      this._stale = true
      this.change.emit()
    })
    this.connect(this.properties.weights.change, () => {
      this._stale = true
      this.change.emit()
    })
  }

  update(): void {
        const {source, axis_idx, unbinned, quantiles, sum_range, weights} = this
        const mean_column = []
        const std_column = []
        const entries_column = []
        const isOK_column = []
        const cumulative_column: number[] = []
        const cdf_column = []
        const weights_column = []
        let quantile_columns:Array<Array<number>> = []
        for (let i = 0; i < quantiles.length; i++) {
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

        const stride_low = source.get_stride(axis_idx)
        const stride_high = source.get_stride(axis_idx+1)

        for(let i=0; i < source.dim; i++){
          if(i != axis_idx){
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
        for (let i = 0; i < source.dim; i++) {
          bin_centers_all.push(source.get_column("bin_center_"+i) as number[])
          bin_centers_filtered.push([])
          bin_top_all.push(source.get_column("bin_top_"+i) as number[])
          bin_top_filtered.push([])
          bin_bottom_all.push(source.get_column("bin_bottom_"+i) as number[])
          bin_bottom_filtered.push([])
        }

        const axis_name = source.sample_variables[axis_idx]

        const global_cumulative_histogram = unbinned ? source.cumulative_histogram_noweights() : null
        // const sorted_indices = unbinned && source.compute_sorted_indices()

        const bin_centers = source.get_column("bin_center_"+axis_idx) as number[]
        const edges_left = source.get_column("bin_bottom_"+axis_idx) as number[]
        const edges_right = source.get_column("bin_top_"+axis_idx) as number[]

        const nbins_total = source.length

        let weights_array
        if(!Array.isArray(weights)){
          weights_array = [weights]
        } else {
          weights_array = weights
        }

        const sorted_entries = unbinned ? source.sorted_column_orig(axis_name) : null

        for(const current_weights of weights_array){
          const sorted_weights = unbinned && current_weights != null? source.sorted_column_orig(current_weights) : null

          let entries_total = 0
          const offset: number = cdf_column.length

          const bin_count = source.histogram(current_weights) as number[]
  
          for(let x = 0; x < nbins_total; x += stride_high){
            for(let z = 0; z < stride_low; z ++){
        //      console.log(x)
        //      console.log(z)
              for(let i=0; i < source.dim; i++){
                if(i != axis_idx){
                    bin_bottom_filtered[i].push(bin_bottom_all[i][x+z])
                    bin_centers_filtered[i].push(bin_centers_all[i][x+z])
                    bin_top_filtered[i].push(bin_top_all[i][x+z])
                }
              }
              if(bin_count === null || bin_centers === null ){
                mean_column.push(NaN)
                std_column.push(NaN)
                entries_column.push(NaN)
                isOK_column.push(false)
                cumulative_column.push(NaN)
                continue
              }
              let cumulative_histogram = []
              let entries = 0
              for(let y=0; y < stride_high; y += stride_low){
                entries += bin_count[x+y+z]
                cumulative_histogram.push(entries)
              }
              cumulative_column.push(entries_total + entries/2)
              entries_total += entries
              if(entries > 0){
                let mean = 0
                if(unbinned){
                  for (let y = 0; y < stride_high; y+= stride_low) {
                    const index_low = x+y+z ? global_cumulative_histogram![x+y+z-1] : 0
                    const index_high = global_cumulative_histogram![x+y+z]
                    if(sorted_weights == null){
                      for(let i=index_low; i < index_high; ++i){
                        if (isFinite(sorted_entries![i]))
                        mean += sorted_entries![i]
                      }
                    } else {
                      for(let i=index_low; i < index_high; ++i){
                        if (isFinite(sorted_entries![i]))
                        mean += sorted_entries![i] * sorted_weights[i]
                      }
                    }
                  }
                } else 
                {
                  for (let y = 0; y < stride_high; y+= stride_low) {
                    mean += bin_count[x+y+z] * bin_centers[x+y+z]
                  }
                }
                let std = 0
                mean /= entries
                if(unbinned){
                  for (let y = 0; y < stride_high; y+= stride_low) {
                    const index_low = x+y+z ? global_cumulative_histogram![x+y+z-1] : 0
                    const index_high = global_cumulative_histogram![x+y+z]
                    if(sorted_weights == null){
                      for(let i=index_low; i < index_high; ++i){
                        if (isFinite(sorted_entries![i]))
                        std += sorted_entries![i] * sorted_entries![i]
                      }
                    } else {
                      for(let i=index_low; i < index_high; ++i){
                        if (isFinite(sorted_entries![i]))
                        std += sorted_entries![i] * sorted_entries![i] * sorted_weights[i]
                      }
                    }
                  }
                  std -= mean*mean*entries
                } else 
                {
                  for (let y = 0; y < stride_high; y+= stride_low) {
                    std += (bin_centers[x+y+z] - mean) * (bin_centers[x+y+z] - mean) * bin_count[x+y+z]
                  }
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
                // Invalid quantiles - negative
                while(iQuantile < quantiles.length && quantiles[iQuantile] < 0){
                  quantile_columns[iQuantile].push(NaN)
                  iQuantile++
                }
                for (let j = 0; j < cumulative_histogram.length && iQuantile < quantiles.length; j++) {
                  const y = j*stride_low
                  high = cumulative_histogram[j]
                  if(unbinned){
                    const index_low = x+y+z ? global_cumulative_histogram![x+y+z-1] : 0
                    const index_high = global_cumulative_histogram![x+y+z]
                    if(sorted_weights == null){
                      // In this case maybe we could use kth value instead
                      while(iQuantile < quantiles.length && high > quantiles[iQuantile] * entries){
                        const k = quantiles[iQuantile] * entries - low
                        const k_floor = k | 0
                        //const m = k % 1
                        kth_value(sorted_entries!, k_floor+index_low, (a,b) => a-b, index_low, index_high)
                        //TODO: Use lerp not nearest
                        quantile_columns[iQuantile].push(sorted_entries![k_floor+index_low])
                        iQuantile++
                      }                   
                    } else {
                      while(iQuantile < quantiles.length && high > quantiles[iQuantile] * entries){
                        const k = quantiles[iQuantile] * entries - low
                        const idx = weighted_kth_value(sorted_entries!, sorted_weights, k, index_low, index_high)
                        quantile_columns[iQuantile].push(sorted_entries![idx])
                        iQuantile++
                      }
                    }
                  } else {
                    // Lerp within bin
                    while(iQuantile < quantiles.length && high > quantiles[iQuantile] * entries){
                      const m = (quantiles[iQuantile] * entries - low)/(high-low)
                      quantile_columns[iQuantile].push(edges_left[y]*(1-m)+edges_right[y]*m)
                      iQuantile++
                    }
                  }
                  low = high
                }
                // Invalid quantiles - more than 100% or empty bin
                while(iQuantile < quantiles.length){
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
              for (let iBox = 0; iBox < sum_range.length; iBox++) {
                const bounding_box = sum_range[iBox];
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
              weights_column.push(current_weights ?? "None")
            }
          }
          for(let i=offset; i<cumulative_column.length; ++i){
            cdf_column.push(cumulative_column[i] / entries_total)
          }
          if(sorted_weights != null){
            source.return_column_to_pool(sorted_weights)
          }
        }

        this.data["mean"] = mean_column
        this.data["std"] = std_column
        this.data["entries"] = entries_column
        this.data["isOK"] = isOK_column
        this.data["cumulative"] = cumulative_column
        this.data["cdf"] = cdf_column
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
        this.data["weights"] = weights_column
        this._stale = false
        if(sorted_entries != null){
          source.return_column_to_pool(sorted_entries)
        }
  }

  get_column(key: string){
    if(this._stale){
      this.update()
    }
    return this.data[key]
  }

  get_length(){
    if(Array.isArray(this.weights)){
      return this.source.get_length() * this.weights.length / this.source.nbins[this.axis_idx]
    }
    return this.source.get_length() / this.source.nbins[this.axis_idx]
  }

}
