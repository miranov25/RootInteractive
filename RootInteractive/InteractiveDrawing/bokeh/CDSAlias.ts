import {ColumnarDataSource} from "models/sources/columnar_data_source"
import * as p from "core/properties"

export namespace CDSAlias {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnarDataSource>
    mapping: p.Property<Record<string, any>>
    includeOrigColumns: p.Property<boolean>
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
    this.define<CDSAlias.Props>(({Ref, Boolean})=>({
      source:  [Ref(ColumnarDataSource)],
      mapping:    [ p.Instance, {} ],
      includeOrigColumns: [Boolean, true]
    }))
  }

  initialize(){
    super.initialize()
    this.data = {}
   // this.compute_functions()
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.source.change, () => {this.data={}; this.change.emit()})
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
    const {mapping, change, selected, source, data, includeOrigColumns} = this
    for (const key in mapping) {
      this.compute_function(key)
    }
    if(includeOrigColumns)
    for (const key in source.data) {
      if(mapping[key] === undefined){
        data[key] = source.get_array(key)
      }
    }
    selected.indices = source.selected.indices
    change.emit()
  }

  compute_function(columnName: string){
    const {source, mapping, data} = this
    const column = mapping[columnName]
    if(column == null){
      let new_column = source.get_column(columnName)
      if(new_column == null){
        throw ReferenceError("Column not defined: "+ columnName + " in data source " + source.name)
      }
      else if (!Array.isArray(new_column)){
        data[columnName] = Array.from(new_column)
      } else {
        data[columnName] = new_column
      }
      return
    }
    if(column.hasOwnProperty("field")){
        if(column.hasOwnProperty("transform")){
            let field = this.get_array(column.field)
            const new_column = column.transform.v_compute(field)
            if(new_column){
                data[columnName] = new_column
            } else {
                data[columnName] = field.map(column.transform.compute)
            }
        } else {
            data[columnName] = this.get_column(column.field) as any[]
        }
    } else if(column.hasOwnProperty("fields")){
        const fields = column.fields.map((x: string) => this.get_array(x))
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
      let new_column = this.get_column(column)
      if(new_column == null){
        throw ReferenceError("Column not defined: "+ column)
      }
      else if (!Array.isArray(new_column)){
        data[columnName] = Array.from(new_column)
      } else {
        data[columnName] = new_column
      }
    }
  }

  get_array(key: string) {
    let column = this.get_column(key)
    if (column == null)
        return []
    else if (!Array.isArray(column))
        return Array.from(column)
    return column;
  }

  get_column(key: string){
    if(this.data[key] != null){
      return this.data[key]
    }
    this.compute_function(key)
    if(this.data[key] != null){
      return this.data[key]
    }
    return null
  }

  update_selection(){
    this.source.selected.indices = this.selected.indices
  }

  get_length(){
    return this.source.get_length()
  }

  columns(){
    const old_columns = this.source.columns().filter(x => this.mapping[x] == null)
    const new_columns = Object.keys(this.mapping)
    return old_columns.concat(new_columns)
  }

}
