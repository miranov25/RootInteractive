import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {ColumnDataSource} from "models/sources/column_data_source"
import * as p from "core/properties"

export namespace HistoStatsCDS {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    sources: p.Property<ColumnarDataSource[]>
    names: p.Property<string[]>
    bincount_columns: p.Property<string[]>
    bin_centers: p.Property<string[]>
    rowwise: p.Property<boolean>
    quantiles: p.Property<number[]> // This is the list of all quantiles to compute, length is NOT equal to CDS length
    compute_quantile: p.Property<boolean[] | null>
    edges_left: p.Property<string[]>
    edges_right: p.Property<string[]>
  }
}

export interface HistoStatsCDS extends HistoStatsCDS.Attrs {}

export class HistoStatsCDS extends ColumnDataSource {
  properties: HistoStatsCDS.Props

  constructor(attrs?: Partial<HistoStatsCDS.Attrs>) {
    super(attrs)
  }

  static __name__ = "HistoStatsCDS"

  static init_HistoStatsCDS() {

    this.define<HistoStatsCDS.Props>(({Ref, Array, String, Boolean, Number, Nullable})=>({
      sources:  [Array(Ref(ColumnarDataSource))],
      names: [Array(String)],
      bincount_columns: [Array(String)],
      bin_centers: [Array(String)],
      rowwise: [Boolean],
      quantiles: [Array(Number), []], // This is the list of all quantiles to compute, length is NOT equal to CDS length
      compute_quantile: [Nullable(Array(Boolean))],
      edges_left: [Array(String)],
      edges_right: [Array(String)],
    }))
  }

  initialize(): void {
    super.initialize()
    // Hack to make this work with bokeh datatable - updating data triggers datatable update no matter what
    if(this.rowwise){
      this.data = {"description":["mean", "std", "entries"]}
    }
    this.update()
  }

  connect_signals(): void {
    super.connect_signals()

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
        for (let i = 0; i < this.sources.length; i++) {
          const histoCDS = this.sources[i]
          const bin_count = histoCDS.get_column(this.bincount_columns[i])
          const bin_centers = histoCDS.get_column(this.bin_centers[i])
          if(bin_count === null || bin_centers === null ){
            mean_column.push(NaN)
            std_column.push(NaN)
            entries_column.push(NaN)
            isOK_column.push(false)
            continue
          }
          // There might be a smart way of using reduce() for inner products of two vectors
          let cumulative_histogram = []
          let entries = 0
          for (let j = 0; j < bin_count.length; j++) {
            entries += bin_count[j]
            cumulative_histogram.push(entries)
          }
          if(entries > 0){
            let mean = 0
            for (let j = 0; j < bin_count.length; j++) {
              mean += bin_count[j] * bin_centers[j]
            }
            mean /= entries
            let std = 0
            for (let j = 0; j < bin_count.length; j++) {
              std += (bin_centers[j] - mean) * (bin_centers[j] - mean) * bin_count[j]
            }
            std /= entries
            std = Math.sqrt(std)
            mean_column.push(mean)
            std_column.push(std)
            entries_column.push(entries)
            isOK_column.push(true)
          } else{
            mean_column.push(NaN)
            std_column.push(NaN)
            entries_column.push(entries)
            isOK_column.push(false)
          }
          if(entries > 0 && (this.compute_quantile === null || this.compute_quantile[i]) ){
            let low = 0
            let high = 0
            let iQuantile = 0
            const edges_left = histoCDS.get_column(this.edges_left[i])
            const edges_right = histoCDS.get_column(this.edges_right[i])
            if(edges_left === null || edges_right === null ){
              throw "Cannot compute quantiles without specifying bin edges"
            }
            while(iQuantile < this.quantiles.length && this.quantiles[iQuantile] < 0){
              quantile_columns[iQuantile].push(NaN)
              iQuantile++
            }
            for (let j = 0; j < cumulative_histogram.length && iQuantile < this.quantiles.length; j++) {
              high = cumulative_histogram[j]
              while(iQuantile < this.quantiles.length && high > this.quantiles[iQuantile] * entries){
                // TODO: Use lerp
                const m = (this.quantiles[iQuantile] * entries - low)/(high-low)
                quantile_columns[iQuantile].push(edges_left[j]*(1-m)+edges_right[j]*m)
                iQuantile++
              }
              low = high
            }
            while(iQuantile < this.quantiles.length){
              quantile_columns[iQuantile].push(NaN)
              iQuantile++
            }
          } else {
            for (let iQuantile = 0; iQuantile < quantile_columns.length; iQuantile++) {
              quantile_columns[iQuantile].push(NaN)
            }
          }
        }
        if(this.rowwise){
          let description:string[] = ["mean", "std", "entries"]
          for (let j = 0; j < this.quantiles.length; j++) {
            description.push("Quantile " + this.quantiles[j])
          }
          this.data.description = description
          for (let i = 0; i < this.names.length; i++) {
            let row:number[] = [mean_column[i], std_column[i], entries_column[i]]
            for (let j = 0; j < this.quantiles.length; j++) {
              row.push(quantile_columns[j][i])
            }
            this.data[this.names[i]] = row
          }
        } else {
          this.data["name"] = this.names
          this.data["mean"] = mean_column
          this.data["std"] = std_column
          this.data["entries"] = entries_column
          this.data["isOK"] = isOK_column
          for (let iQuantile = 0; iQuantile < quantile_columns.length; iQuantile++) {
            this.data["quantile_"+iQuantile] = quantile_columns[iQuantile]
          }
        }
        this.change.emit()
  }

}
