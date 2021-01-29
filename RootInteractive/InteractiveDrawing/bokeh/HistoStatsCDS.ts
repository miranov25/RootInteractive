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

    this.define<HistoStatsCDS.Props>(({Ref, Array, String, Boolean})=>({
      sources:  [Array(Ref(ColumnarDataSource))],
      names: [Array(String)],
      bincount_columns: [Array(String)],
      bin_centers: [Array(String)],
      rowwise: [Boolean]
    }))
  }

  initialize(): void {
    super.initialize()
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
          //let cumulative_histogram = []
          let entries = 0
          for (let j = 0; j < bin_count.length; j++) {
            entries += bin_count[j]
          //  cumulative_histogram += [entries]
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

        }
        if(this.rowwise){
          this.data = {"description":["mean", "std", "entries"]}
          for (let i = 0; i < this.names.length; i++) {
            this.data[this.names[i]] = [mean_column[i], std_column[i], entries_column[i]]
          }
        } else {
          this.data = {"name": this.names, "mean": mean_column, "std": std_column, "entries": entries_column, "isOK": isOK_column}
        }
        this.change.emit()
  }

}
