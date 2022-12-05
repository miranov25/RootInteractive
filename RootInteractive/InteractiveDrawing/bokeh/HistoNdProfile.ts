import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {HistoNdCDS} from "./HistoNdCDS"
import * as p from "core/properties"

function kth_value(sample:any[], k:number, compare:(a:number, b:number)=>number, begin=0, end=-1){
  // TODO: Use a smarter algorithm
  // This algorithm should be linear average case, but worst case is quadratic
  if(end < 0) end += sample.length + 1
  if(!(k>=begin && k<end)) {
    throw Error("Element out of range: " + (k-begin))
  }
  while(begin<end){
    //debugger;
    let pivot = sample[k]
    let pivot_idx = begin
    for(let i=begin; i<end; i++){
      pivot_idx += isNaN(sample[i]) || compare(sample[i], pivot) < 0 ? 1 : 0
    }
    if(pivot_idx >= end && isNaN(pivot)) return
    let pointer_mid = pivot_idx
    let pointer_high = end-1
    let pointer_low=begin
    while(pointer_low<pivot_idx){
      let x = sample[pointer_low]
      let cmp = compare(x, pivot)
      if(cmp>0){
        sample[pointer_low] = sample[pointer_high]
        sample[pointer_high] = x
        --pointer_high
      } else if(cmp === 0){
        sample[pointer_low] = sample[pointer_mid]
        sample[pointer_mid] = x
        ++pointer_mid
      } else {
        ++pointer_low
      }
    }
    while(pointer_mid <= pointer_high){
      let x = sample[pointer_mid]
      let cmp = compare(x, pivot)     
      if(cmp === 0){
        ++pointer_mid
      } else {
        sample[pointer_mid] = sample[pointer_high]
        sample[pointer_high] = x
        --pointer_high
      }
    }
    if(k < pivot_idx){
      end = pivot_idx
    } else if(k >= pointer_mid){
      begin = pointer_mid
    } else {
      return
    }
  }
}

function weighted_kth_value(sample:number[], weights: number[], k:number, begin=0, end=-1){
  // TODO: Use a smarter algorithm
  // This algorithm should be linear average case, but worst case is quadratic
  if(end < 0) end += sample.length + 1
  const end_orig = end
  while(true){
    let pivot = sample[(begin+end)>>1]
    let pivot_idx = begin
    let pivot_cumsum_lower = 0
    let pivot_cumsum_upper = 0
    for(let i=begin; i<end; i++){
      pivot_idx += sample[i] < pivot ? 1 : 0
      pivot_cumsum_lower += sample[i] < pivot ? weights[i] : 0
      pivot_cumsum_upper += sample[i] <= pivot ? weights[i] : 0
    }
    let pointer_mid = pivot_idx
    let pointer_high = end-1
    let pointer_low=begin
    while(pointer_low<pivot_idx){
      let x = sample[pointer_low]
      let tmp = weights[pointer_low]
      if(x>pivot){
        sample[pointer_low] = sample[pointer_high]
        sample[pointer_high] = x
        weights[pointer_low] = weights[pointer_high]
        weights[pointer_high] = tmp       
        --pointer_high
      } else if(x === pivot){
        sample[pointer_low] = sample[pointer_mid]
        sample[pointer_mid] = x
        weights[pointer_low] = weights[pointer_mid]
        weights[pointer_mid] = tmp
        ++pointer_mid
      } else {
        ++pointer_low
      }
    }
    while(pointer_mid <= pointer_high){
      let x = sample[pointer_mid]
      let tmp = weights[pointer_mid]    
      if(x === pivot){
        ++pointer_mid
      } else {
        sample[pointer_mid] = sample[pointer_high]
        sample[pointer_high] = x
        weights[pointer_mid] = weights[pointer_high]
        weights[pointer_high] = tmp
        --pointer_high
      }
    }
    if(k < pivot_cumsum_lower){
      end = pivot_idx
    } else if (k > pivot_cumsum_upper) {
      k -= pivot_cumsum_upper
      begin = pointer_mid
      if (pointer_mid === end_orig) return pointer_mid-1
    } else {
      return pivot_idx
    }
  }
}

export namespace HistoNdProfile {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<HistoNdCDS>
    axis_idx: p.Property<number>
    quantiles: p.Property<number[]>
    sum_range: p.Property<number[][]>
    unbinned: p.Property<boolean>
    weights: p.Property<string | null>
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

    this.define<HistoNdProfile.Props>(({Ref, Array, Number, Int, Boolean, String, Nullable})=>({
      source:  [Ref(HistoNdCDS)],
      axis_idx: [Int, 0],
      quantiles: [Array(Number), []], // This is the list of all quantiles to compute, length is NOT equal to CDS length
      sum_range: [Array(Array(Number)), []],
      unbinned: [Boolean, false],
      weights: [Nullable(String), null]
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
        const {source, axis_idx, unbinned, quantiles, sum_range, weights} = this
        let mean_column = []
        let std_column = []
        let entries_column = []
        let isOK_column = []
        let cumulative_column: number[] = []
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

        for(let i=0; i<source.dim; i++){
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
        const sorted_weights = unbinned && weights != null? source.sorted_column_orig(weights) : null
        const sorted_entries = unbinned ? source.sorted_column_orig(axis_name) : null

        const bin_count = source.get_column("bin_count") as number[]
        const bin_centers = source.get_column("bin_center_"+axis_idx) as number[]
        const edges_left = source.get_column("bin_bottom_"+axis_idx) as number[]
        const edges_right = source.get_column("bin_top_"+axis_idx) as number[]

        const nbins_total = source.length

        let entries_total = 0

        for(let x = 0; x < nbins_total; x += stride_high){
          for(let z = 0; z < stride_low; z ++){
      //      console.log(x)
      //      console.log(z)
            for(let i=0; i<source.dim; i++){
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
                    for(let i=index_low; i<index_high; ++i){
                      if (isFinite(sorted_entries![i]))
                      mean += sorted_entries![i]
                    }
                  } else {
                    for(let i=index_low; i<index_high; ++i){
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
                    for(let i=index_low; i<index_high; ++i){
                      if (isFinite(sorted_entries![i]))
                      std += sorted_entries![i] * sorted_entries![i]
                    }
                  } else {
                    for(let i=index_low; i<index_high; ++i){
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
          }
        }

        this.data["mean"] = mean_column
        this.data["std"] = std_column
        this.data["entries"] = entries_column
        this.data["isOK"] = isOK_column
        this.data["cumulative"] = cumulative_column
        this.data["cdf"] = cumulative_column.map((x) => x / entries_total)
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
        if(sorted_weights != null){
          source.return_column_to_pool(sorted_weights)
        }
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
    return this.source.get_length() / this.source.nbins[this.axis_idx]
  }

}
