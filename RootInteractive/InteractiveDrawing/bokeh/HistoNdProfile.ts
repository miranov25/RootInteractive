import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {Model} from "model"
import * as p from "core/properties"
import { quantile, weighted_kth_value } from "./MathUtils"

export namespace HistoNdProfile {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<any>
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

  static {

    this.define<HistoNdProfile.Props>(({Ref, Array, Number, Int, Boolean, String, Nullable, Or})=>({
      source:  [Ref(Model)],
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
        // PHASE_5 Batch 2B (ST-D4): the requested levels in ascending order (a stable sort of their positions)
        const order = quantiles.map((_q: number, i: number) => i).sort((a: number, b: number) => quantiles[a] - quantiles[b] || a - b)
        const mean_column = []
        const std_column = []
        const entries_column = []
        const isOK_column = []
        const cumulative_column: number[] = []
        const cdf_column = []
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

        let sorted_entries = unbinned ? source.sorted_column_orig(axis_name) : null

        let first_series = true
        for(const current_weights of weights_array){
          if(!first_series && sorted_entries != null){
            // PHASE_5 Batch 2A (WT-D4): the quantiles of the series before reordered these values in place together
            // with that series' weights; this series pairs its own weights with the values in bin order again
            source.return_column_to_pool(sorted_entries)
            sorted_entries = source.sorted_column_orig(axis_name)
          }
          first_series = false
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
		let std = 0
		let entries_ = 0
                if(unbinned){
		  let old_mean = 0
                  for (let y = 0; y < stride_high; y+= stride_low) {
                    const index_low = x+y+z ? global_cumulative_histogram![x+y+z-1] : 0
                    const index_high = global_cumulative_histogram![x+y+z]
                    if(sorted_weights == null){
                      for(let i=index_low; i < index_high; ++i){
                        if (isFinite(sorted_entries![i])){
				entries_ ++
				old_mean = mean
				mean += (sorted_entries![i] - mean) / entries_
				std += (sorted_entries![i] - mean) * (sorted_entries![i] - old_mean)
			}
                      }
                    } else {
                      for(let i=index_low; i < index_high; ++i){
                        // PHASE_5 Batch 2A (WT-D2): a row of weight 0 adds nothing; as the first row of a bin it made
                        // the running mean 0/0 = NaN
                        if (isFinite(sorted_entries![i]) && sorted_weights[i] !== 0){
				entries_ += sorted_weights[i]
				old_mean = mean
                        	mean += (sorted_entries![i] - mean) * sorted_weights[i] / entries_
				std += (sorted_entries![i] - mean) * (sorted_entries![i] - old_mean) * sorted_weights[i]
			}
                      }
                    }
                  }
                } else 
                {
                  for (let y = 0; y < stride_high; y+= stride_low) {
                    mean += bin_count[x+y+z] * bin_centers[x+y+z]
                  }
		  mean /= entries
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
                // PHASE_5 Batch 2B (ST-D4): the levels are found in ascending order, each written to its own column -
                // a request list in another order (.55, .15, ...) gave wrong values or "Quantile out of range"
                const cell_quantiles: number[] = Array(quantiles.length).fill(NaN)
                let iq = 0
                // Invalid quantiles - negative
                while(iq < order.length && quantiles[order[iq]] < 0){
                  iq++
                }
                for (let j = 0; j < cumulative_histogram.length && iq < order.length; j++) {
                  const y = j*stride_low
                  high = cumulative_histogram[j]
                  if(unbinned){
                    const index_low = x+y+z ? global_cumulative_histogram![x+y+z-1] : 0
                    const index_high = global_cumulative_histogram![x+y+z]
                    if(sorted_weights == null){
                      // In this case maybe we could use kth value instead
                      while(iq < order.length && high > quantiles[order[iq]] * entries){
                        const q_scaled = (quantiles[order[iq]] * entries - low) / (high - low)
                        cell_quantiles[order[iq]] = quantile(sorted_entries!, q_scaled, index_low, index_high)
                        iq++
                      }
                    } else {
                      while(iq < order.length && high > quantiles[order[iq]] * entries){
                        const k = quantiles[order[iq]] * entries - low
                        const idx = weighted_kth_value(sorted_entries!, sorted_weights, k, index_low, index_high)
                        cell_quantiles[order[iq]] = sorted_entries![idx]
                        iq++
                      }
                    }
                  } else {
                    // Lerp within bin
                    while(iq < order.length && high > quantiles[order[iq]] * entries){
                      const m = (quantiles[order[iq]] * entries - low)/(high-low)
                      cell_quantiles[order[iq]] = edges_left[y]*(1-m)+edges_right[y]*m
                      iq++
                    }
                  }
                  low = high
                }
                // PHASE_5 Batch 2B (q = 1): "high > q * entries" is never true at q = 1 - the level is the maximum: of
                // the cell's values (unbinned), of its values with positive weight (weighted), or the upper edge of its
                // last non-empty sub-bin (binned). Levels above 1, and an empty cell, stay NaN.
                while(iq < order.length && quantiles[order[iq]] === 1){
                  let top = NaN
                  for (let j = 0; j < cumulative_histogram.length; j++) {
                    const y = j*stride_low
                    if(unbinned){
                      const index_low = x+y+z ? global_cumulative_histogram![x+y+z-1] : 0
                      const index_high = global_cumulative_histogram![x+y+z]
                      for(let i=index_low; i < index_high; ++i){
                        const v = sorted_entries![i]
                        if(isFinite(v) && (sorted_weights == null || sorted_weights[i] > 0) && !(v <= top)){
                          top = v
                        }
                      }
                    } else if(bin_count[x+y+z] > 0){
                      top = edges_right[y]
                    }
                  }
                  cell_quantiles[order[iq]] = top
                  iq++
                }
                for (let i = 0; i < quantiles.length; i++) {
                  quantile_columns[i].push(cell_quantiles[i])
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
        this._stale = false
        if(sorted_entries != null){
          source.return_column_to_pool(sorted_entries)
        }
  }

  get_column(key: string){
    if(key == "weights"){
      return this.get_weights_label()
    }
    if(this._stale){
      this.update()
    }
    return this.data[key]
  }

  get_weights_label(){
    const l = this.source.get_length() / this.source.nbins[this.axis_idx]
    if(Array.isArray(this.weights)){
      return this.weights.flatMap((w: string | null) => Array(l).fill(w ?? "None"))
    } else if(this.weights != null){
      return Array(l).fill(this.weights)
    }
    return Array(l).fill("None")
  }

  get_length(){
    if(Array.isArray(this.weights)){
      return this.source.get_length() * this.weights.length / this.source.nbins[this.axis_idx]
    }
    return this.source.get_length() / this.source.nbins[this.axis_idx]
  }

}
