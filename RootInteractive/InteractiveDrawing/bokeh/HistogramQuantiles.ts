import {ColumnDataSource} from "models/sources/column_data_source"
import {Model} from "model"
import * as p from "core/properties"
import {sorted_with_weights, quantile_of_sorted} from "./MathUtils"

// PHASE_5_2 stage 4 (FR-RI-3): the exact quantiles of one histogram's column over its selected rows, as rows to draw:
// one row per requested level whose value is finite and inside the shown range. The values are all the selected
// finite values of the column, not only those inside the histogram's range, and do not depend on its bins.
// Estimators (MathUtils.quantile_of_sorted, dfdraw's): without weights NumPy's linear; with weights "inverted_cdf" by
// default, "linear" as an option. An empty selection, a negative weight or no positive weight gives no row.
// Columns: x (the value), level, label ("median", "q=10%"), dash (the median solid, the others dashed), text, edge (the
// far end of the other axis' shown range, where the text stands: the top, or the right with flipped axes).

export namespace HistogramQuantiles {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnDataSource.Props & {
    histogram: p.Property<any>
    levels: p.Property<number[]>
    method: p.Property<string>
    text: p.Property<string>
    x_range: p.Property<any>
    edge_range: p.Property<any>
    column_names: p.Property<string[]>
  }
}

export interface HistogramQuantiles extends HistogramQuantiles.Attrs {}

export class HistogramQuantiles extends ColumnDataSource {
  properties: HistogramQuantiles.Props

  private _stale: boolean
  public values: Map<number, number>          // the last computed value of each level (also those not drawn)

  constructor(attrs?: Partial<HistogramQuantiles.Attrs>) {
    super(attrs)
  }

  static __name__ = "HistogramQuantiles"

  static {
    this.define<HistogramQuantiles.Props>(({Ref, Array, Number, String, Nullable})=>({
      column_names: [Array(String), ["x", "level", "label", "dash", "text", "edge"]],  // for Bokeh's legend check
      edge_range: [Nullable(Ref(Model)), null],   // the other axis' range: the texts stand at its far end
      histogram: [Ref(Model)],
      levels: [Array(Number), []],
      method: [String, "default"],
      text: [String, "none"],
      x_range: [Nullable(Ref(Model)), null]
    }))
  }

  initialize(){
    super.initialize()
    this.data = {x: [], level: [], label: [], dash: [], text: [], edge: []}
    this.values = new Map()
    this._stale = true
  }

  connect_signals(): void {
    super.connect_signals()
    const stale = () => { this._stale = true; this.change.emit() }
    this.connect(this.histogram.change, stale)
    for (const range of [this.x_range, this.edge_range]) {
      if (range != null) {
        this.connect(range.properties.start.change, stale)
        this.connect(range.properties.end.change, stale)
      }
    }
  }

  columns(): string[] {
    return ["x", "level", "label", "dash", "text", "edge"]
  }

  static label(q: number): string {
    return q === 0.5 ? "median" : `q=${+(q * 100).toPrecision(6)}%`
  }

  static number(v: number): string {
    return (+v.toPrecision(4)).toString()
  }

  update(){
    const h = this.histogram
    const values = h.source.get_column(h.sample) ?? []
    const weights = h.weights != null ? h.source.get_column(h.weights) : null
    const indices = h.view                    // the histogram's selected rows (null: all), as the histogram uses them
    const method = this.method === "default" ? (weights != null ? "inverted_cdf" : "linear") : this.method
    this.values = new Map()
    // one sort of the selected finite rows (with positive weight): 10^6 rows about 0.3 s without weights and 0.7 s with
    // them, 10^7 rows 3 s and 9 s (measured in Node, PHASE_5_2 stage 4); 8 or 20 bytes per row
    const sorted = sorted_with_weights(values, weights, indices)
    for (const q of this.levels) {
      if (!this.values.has(q)) this.values.set(q, sorted == null ? NaN : quantile_of_sorted(sorted.xs, sorted.ws, q, method))
    }
    const range = this.x_range
    // Nothing is drawn while the shown range is not known yet (no view has computed it): a line drawn then would
    // widen the automatic range to itself (a span takes part in it). Once the range is computed from the histogram,
    // its change recomputes the rows, and a line inside the range does not widen it.
    const known = range == null || isFinite(range.start) && isFinite(range.end)
    const low = range != null && known ? Math.min(range.start, range.end) : -Infinity
    const high = range != null && known ? Math.max(range.start, range.end) : Infinity
    const data: {[key: string]: any[]} = {x: [], level: [], label: [], dash: [], text: [], edge: []}
    const edge = this.edge_range != null ? Math.max(this.edge_range.start, this.edge_range.end) : NaN
    const done = new Set<number>()
    for (const q of this.levels) {
      if (done.has(q)) continue
      done.add(q)
      const v = this.values.get(q)!
      if (!known || !isFinite(v) || !(v >= low && v <= high)) continue
      const label = HistogramQuantiles.label(q)
      data.x.push(v)
      data.level.push(q)
      data.label.push(label)
      data.dash.push(q === 0.5 ? "solid" : "dashed")
      data.edge.push(edge)
      const number = HistogramQuantiles.number(v)
      const shown = q === 0.5 ? "median" : `${+(q * 100).toPrecision(6)}%`
      data.text.push(this.text === "both" ? `${shown}: ${number}` : this.text === "level" ? shown :
                     this.text === "value" ? number : "")
    }
    for (const key of Object.keys(data)) (this.data as any)[key] = data[key]
    this._stale = false
  }

  get_column(name: string){
    if (this._stale) this.update()
    return (this.data as any)[name] ?? null
  }

  get_length(){
    if (this._stale) this.update()
    return (this.data as any).x.length
  }
}
