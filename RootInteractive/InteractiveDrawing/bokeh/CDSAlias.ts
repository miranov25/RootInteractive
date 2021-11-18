import {ColumnarDataSource} from "models/sources/columnar_data_source"
import * as p from "core/properties"

export namespace CDSAlias {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnarDataSource>
    mapping: p.Property<Record<string, any>>
  }
}

export interface CDSAlias extends CDSAlias.Attrs {}

export class CDSAlias extends ColumnarDataSource {
  properties: CDSAlias.Props

  constructor(attrs?: Partial<CDSAlias.Attrs>) {
    super(attrs)
  }

  static __name__ = "CDSAlias"

  static init_CDSAlias() {
    this.define<CDSAlias.Props>(({Ref})=>({
      source:  [Ref(ColumnarDataSource)],
      mapping:    [ p.Instance ]
    }))
  }

  initialize(){
    super.initialize()
    this.data = {}
    this.compute_functions()
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.source.change, () => this.compute_functions())
    for( const key in this.mapping){
      const column = this.mapping[key]
      if(column.hasOwnProperty("transform")){
        this.connect(column.transform.change, () => {
          this.compute_function(key)
          this.change.emit()
        })
      }
    }

    this.connect(this.selected.change, () => this.update_selection())
  }

  compute_functions(){
    const {mapping, change, selected} = this
    for (const key in mapping) {
      this.compute_function(key)
    }
    selected.indices = this.source.selected.indices
    change.emit()
  }

  compute_function(columnName: string){
    const {source, mapping, data} = this
    const column = mapping[columnName]
    if(column.hasOwnProperty("field")){
        if(column.hasOwnProperty("transform")){
            const field = source.data[column.field] as any[]
            const new_column = column.transform.v_compute(field)
            if(new_column){
                data[columnName] = new_column
            } else {
                data[columnName] = field.map(column.transform.compute)
            }
        } else {
            data[columnName] = source.data[column.field] as any[]
        }
    } else if(column.hasOwnProperty("fields")){
        const fields = column.fields.map((x: number) => source.data[x])
        let new_column = column.transform.v_compute(fields)
        if(new_column){
            data[columnName] = new_column
        } else {
            new_column = []
            for (let i = 0; i < fields[0].length; i++) {
                const row = fields.map((x: any[]) => x[i])
                // This will likely have very bad performance
                new_column.push(column.transform.compute(row))
            }
            data[columnName] = new_column
        }
    } else if(Object.prototype.toString.call(column) === '[object String]'){
      data[columnName] = source.data[column] as any[]
    }
  }

  update_selection(){
    this.source.selected.indices = this.selected.indices
  }

}
