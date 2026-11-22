import {ColumnDataSource} from "models/sources/column_data_source"
import {Model} from "model"
import * as p from "core/properties"

// PHASE_5_2 stage 7 (the Architect's option (y)): the boxed labels of quantile curves on a 2-D histogram, as dfdraw
// draws them: one label per curve whose value is finite, all at the X bin nearest a fixed fraction of the shown X
// range (`fraction`, 0.4 by default), at the curve's value there. Columns: x, y, text, color, level. It follows the
// profile (the selection) and the shown X range; nothing while the range is not known.

export namespace QuantileCurveLabels {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnDataSource.Props & {
    source: p.Property<any>
    x: p.Property<string>
    curves: p.Property<string[]>
    texts: p.Property<string[]>
    colors: p.Property<string[]>
    levels: p.Property<number[]>
    x_range: p.Property<any>
    fraction: p.Property<number>
    column_names: p.Property<string[]>
  }
}

export interface QuantileCurveLabels extends QuantileCurveLabels.Attrs {}

export class QuantileCurveLabels extends ColumnDataSource {
  properties: QuantileCurveLabels.Props

  private _stale: boolean

  constructor(attrs?: Partial<QuantileCurveLabels.Attrs>) {
    super(attrs)
  }

  static __name__ = "QuantileCurveLabels"

  static {
    this.define<QuantileCurveLabels.Props>(({Ref, Array, Number, String, Nullable})=>({
      source: [Ref(Model)],
      x: [String, "bin_center_0"],
      curves: [Array(String), []],
      texts: [Array(String), []],
      colors: [Array(String), []],
      levels: [Array(Number), []],
      x_range: [Nullable(Ref(Model)), null],
      fraction: [Number, 0.4],
      column_names: [Array(String), ["x", "y", "text", "color", "level"]]   // for Bokeh's checks
    }))
  }

  initialize(){
    super.initialize()
    this.data = {x: [], y: [], text: [], color: [], level: []}
    this._stale = true
  }

  connect_signals(): void {
    super.connect_signals()
    const stale = () => { this._stale = true; this.change.emit() }
    this.connect(this.source.change, stale)
    if (this.x_range != null) {
      this.connect(this.x_range.properties.start.change, stale)
      this.connect(this.x_range.properties.end.change, stale)
    }
  }

  columns(): string[] {
    return ["x", "y", "text", "color", "level"]
  }

  update(){
    const data: {[key: string]: any[]} = {x: [], y: [], text: [], color: [], level: []}
    const range = this.x_range
    const known = range != null && isFinite(range.start) && isFinite(range.end)
    const xs = this.source.get_column(this.x) ?? []
    if (known && xs.length > 0) {
      const target = range.start + this.fraction * (range.end - range.start)
      let best = -1
      for (let i = 0; i < xs.length; i++) {
        if (isFinite(xs[i]) && (best < 0 || Math.abs(xs[i] - target) < Math.abs(xs[best] - target))) best = i
      }
      if (best >= 0) {
        for (let k = 0; k < this.curves.length; k++) {
          const ys = this.source.get_column(this.curves[k]) ?? []
          const y = ys[best]
          if (y == null || !isFinite(y)) continue
          data.x.push(xs[best])
          data.y.push(y)
          data.text.push(this.texts[k])
          data.color.push(this.colors[k])
          data.level.push(this.levels[k])
        }
      }
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
