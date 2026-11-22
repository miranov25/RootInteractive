import {ColumnDataSource} from "models/sources/column_data_source"
import {Model} from "model"
import * as p from "core/properties"

// PHASE_5_2 stage 3 (FR-RI-2): the polygons of a filled band between two columns of a source, one polygon per run of
// rows whose x, lower and upper values are all finite, so that a NaN breaks the band. Bokeh 3.3.2's VArea joins the
// neighbours of a NaN instead. Columns: xs, ys (the upper edge forward, then the lower edge backward), for Patches.
// Computed when a glyph reads them, after any change of the source.

export namespace BandPolygons {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnDataSource.Props & {
    source: p.Property<any>
    x: p.Property<string>
    lower: p.Property<string>
    upper: p.Property<string>
  }
}

export interface BandPolygons extends BandPolygons.Attrs {}

export class BandPolygons extends ColumnDataSource {
  properties: BandPolygons.Props

  private _stale: boolean

  constructor(attrs?: Partial<BandPolygons.Attrs>) {
    super(attrs)
  }

  static __name__ = "BandPolygons"

  static {
    this.define<BandPolygons.Props>(({Ref, String})=>({
      source: [Ref(Model)],
      x: [String],
      lower: [String],
      upper: [String]
    }))
  }

  initialize(){
    super.initialize()
    this.data = {xs: [], ys: []}
    this._stale = true
  }

  connect_signals(): void {
    super.connect_signals()
    this.connect(this.source.change, () => {
      this._stale = true
      this.change.emit()
    })
  }

  columns(): string[] {
    return ["xs", "ys"]
  }

  update(){
    const {source} = this
    const x = source.get_column(this.x) ?? []
    const lower = source.get_column(this.lower) ?? []
    const upper = source.get_column(this.upper) ?? []
    const n = Math.min(x.length, lower.length, upper.length)
    const xs: number[][] = []
    const ys: number[][] = []
    let run: number[] = []
    const flush = () => {
      if (run.length > 0) {
        const back = run.slice().reverse()
        xs.push(run.map(i => x[i]).concat(back.map(i => x[i])))
        ys.push(run.map(i => upper[i]).concat(back.map(i => lower[i])))
        run = []
      }
    }
    for (let i = 0; i < n; i++) {
      if (isFinite(x[i]) && isFinite(lower[i]) && isFinite(upper[i])) run.push(i)
      else flush()
    }
    flush()
    this.data.xs = xs
    this.data.ys = ys
    this._stale = false
  }

  get_column(name: string){
    if (this._stale) this.update()
    return (this.data as any)[name] ?? null
  }

  get_length(){
    if (this._stale) this.update()
    return (this.data as any).xs.length
  }
}
