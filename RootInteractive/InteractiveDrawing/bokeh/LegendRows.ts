import {ColumnDataSource} from "models/sources/column_data_source"
import {Model} from "model"
import * as p from "core/properties"

// PHASE_5_2 stage 5 (FR-RI-5): the rows of a grid legend's stand-in renderer. One row per distinct value of `field` in
// the cells' sources (all cells with this legend entry), in the order they appear (as Bokeh's legend lists a legend on
// a column); every other column of `names` is NaN, so the stand-in draws nothing on its own plot, only its legend
// swatches. Without a field, one row. It follows every change of the cells' sources.

export namespace LegendRows {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnDataSource.Props & {
    sources: p.Property<any[]>
    field: p.Property<string | null>
    names: p.Property<string[]>
    column_names: p.Property<string[]>
  }
}

export interface LegendRows extends LegendRows.Attrs {}

export class LegendRows extends ColumnDataSource {
  properties: LegendRows.Props

  private _stale: boolean

  constructor(attrs?: Partial<LegendRows.Attrs>) {
    super(attrs)
  }

  static __name__ = "LegendRows"

  static {
    this.define<LegendRows.Props>(({Ref, String, Nullable, Array})=>({
      sources: [Array(Ref(Model)), []],
      field: [Nullable(String), null],
      names: [Array(String), []],
      column_names: [Array(String), []]   // declared for Bokeh's legend check
    }))
  }

  initialize(){
    super.initialize()
    this.data = {}
    this._stale = true
  }

  connect_signals(): void {
    super.connect_signals()
    for (const source of this.sources) {
      this.connect(source.change, () => { this._stale = true; this.change.emit() })
    }
  }

  columns(): string[] {
    return this.field != null ? [this.field, ...this.names] : [...this.names]
  }

  update(){
    let values: any[] = [null]
    if (this.field != null) {
      values = []
      const seen = new Set<string>()
      for (const source of this.sources) {
        for (const v of source.get_column(this.field) ?? []) {
          const key = String(v)
          if (!seen.has(key)) { seen.add(key); values.push(v) }
        }
      }
    }
    const data: {[key: string]: any[]} = {}
    for (const name of this.names) data[name] = values.map(() => NaN)
    if (this.field != null) data[this.field] = values
    for (const key of Object.keys(data)) (this.data as any)[key] = data[key]
    this._stale = false
  }

  get_column(name: string){
    if (this._stale) this.update()
    return (this.data as any)[name] ?? null
  }

  get_length(){
    if (this._stale) this.update()
    const first = this.field ?? this.names[0]
    return first != null ? ((this.data as any)[first] ?? []).length : 1
  }
}
