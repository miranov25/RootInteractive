import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {ColumnDataSource} from "models/sources/column_data_source"
import {CDSView} from "models/sources/cds_view"
import * as p from "core/properties"

export namespace HistogramCDS {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnDataSource>
    view: p.Property<CDSView>
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

    this.define<HistogramCDS.Props>(({Ref})=>({
      source:  [Ref(ColumnDataSource)],
      view:         [Ref(CDSView), () => new CDSView()],
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
    this.view = new CDSView({source:this.source})

    this.update_range()
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.source.change, () => this.update_data())
    this.connect(this.view.change, () => this.update_data())
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
        // Caching view indices might save some time. Blame bokehjs
        let view_indices = [...this.view.indices]
        const sample_array = this.source.data[this.sample]
        // Hack to make a trivial function perform better - can be optimized further
        let weights_getter
        if(this.weights != null){
          const weights_array = this.source.data[this.weights]
          weights_getter = (i: number):number=>weights_array[i]
        } else {
          weights_getter = (_dummy: number):number=>1
        }
        for(let i=0; i<view_indices.length; i++){
          const bin = this.getbin(sample_array[i])
          if(bin >= 0 && bin < this.nbins){
            bincount[bin] += weights_getter(i)
          }
        }
      }
      this.data["bin_count"] = bincount
      this.change.emit()
  }

  update_range(): void {
      const bin_left = (this.data["bin_left"] as number[])
      const bin_right = (this.data["bin_right"] as number[])
      bin_left.length = 0
      bin_right.length = 0
      for (let index = 0; index < this.nbins; index++) {
        bin_left.push(this.range_min+index*(this.range_max-this.range_min)/this.nbins)
        bin_right.push(this.range_min+(index+1)*(this.range_max-this.range_min)/this.nbins)
      }
      this.update_data()
  }

  getbin(val: number): number{
      // Make the max value inclusive
      if(val === this.range_max) return this.nbins-1
      // Overflow bins
      if(val > this.range_max) return this.nbins
      if(val < this.range_min) return -1
      // We can avoid an arithmetic operation there if we cache origin/scale points - TODO: Do that
      return ((val-this.range_min)*this.nbins/(this.range_max-this.range_min))|0
  }

}
