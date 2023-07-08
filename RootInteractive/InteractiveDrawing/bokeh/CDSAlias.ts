import {ColumnarDataSource} from "models/sources/columnar_data_source"
import * as p from "core/properties"

export namespace CDSAlias {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnarDataSource>
    mapping: p.Property<Record<string, any>>
    includeOrigColumns: p.Property<boolean>
    columnDependencies: p.Property<any[]>
  }
}

export interface CDSAlias extends CDSAlias.Attrs {}

export class CDSAlias extends ColumnarDataSource {
  properties: CDSAlias.Props

  constructor(attrs?: Partial<CDSAlias.Attrs>) {
    super(attrs)
  }

  static __name__ = "CDSAlias"

  cached_columns: Set<string>

  _locked_columns: Set<string>

  static init_CDSAlias() {
    this.define<CDSAlias.Props>(({Any, Boolean, Array, Ref})=>({
      source:  [Ref(ColumnarDataSource)],
      mapping:    [ p.Instance, {} ],
      includeOrigColumns: [Boolean, true],
      columnDependencies: [Array(Any), []]
    }))
  }

  initialize(){
    super.initialize()
    this.cached_columns = new Set()
    this._locked_columns = new Set()
    this.data = {}
   // this.compute_functions()
  }

  connect_signals(): void {
    super.connect_signals()

    this.connect(this.source.change, () => {this.cached_columns.clear(); this.change.emit()})
    for( const key in this.mapping){
      const column = this.mapping[key]
      if(column != null && column.hasOwnProperty("transform")){
        this.connect(column.transform.change, () => {
          this.invalidate_column(key)
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

  compute_function(key: string){
    const {source, mapping, data, cached_columns, _locked_columns} = this
    const column = mapping[key]
    const len = this.get_length()
    if(len == null) return
    if(column == null){
      let new_column = source.get_column(key)
      if(new_column == null){
        throw ReferenceError("Column not defined: "+ key + " in data source " + source.name)
      }
      /*else if (!Array.isArray(new_column)){
        data[key] = Array.from(new_column)
      } */else {
        data[key] = new_column
      }
      cached_columns.add(key)
      return
    }
    if(column.hasOwnProperty("field")){
        if(column.hasOwnProperty("transform")){
            let field = this.get_column(column.field)
            if(field == null){
              throw ReferenceError("Column not defined: "+ column.field + " in data source " + this.name)
            }
            const new_column = column.transform.v_compute(field)
            if(new_column){
                data[key] = new_column
            } else {
                data[key] = Array.from(field).map(column.transform.compute)
            }
        } else {
            data[key] = this.get_column(column.field) as any[]
        }
    } else if(column.hasOwnProperty("fields")){
        if(_locked_columns.has(key)){
          return
        }
        _locked_columns.add(key)
        const fields = column.fields.map((x: string) => isNaN(Number(x)) ? this.get_column(x)! : Array(len).fill(Number(x)))
        let new_column = column.transform.v_compute(fields, this.source, data[key])
        if(new_column){
            data[key] = new_column
        } else if(data[key] != null){
            new_column = data[key]
            new_column.length = source.get_length()
            const nvars = fields.length
            let row = new Array(nvars)
            for (let i = 0; i < len; i++) {
              for(let j=0; j < nvars; j++){
                row[j] = fields[j][i]
              }
              new_column[i] = column.transform.compute(row)
            }            
        } else{
            new_column = new Array(len).fill(.0)
            for (let i = 0; i < len; i++) {
                const row = fields.map((x: any[]) => x[i])
                // This will likely have very bad performance
                new_column[i] = column.transform.compute(row)
            }
            data[key] = new_column
        }
    } else if(Object.prototype.toString.call(column) === '[object String]'){
      let new_column = this.get_column(column)
      if(new_column == null){
        _locked_columns.delete(key)
        throw ReferenceError("Column not defined: "+ column)
      }
      else if (!Array.isArray(new_column)){
        data[key] = Array.from(new_column)
      } else {
        data[key] = new_column
      }
    }
    cached_columns.add(key)
    _locked_columns.delete(key)
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
    if(this.cached_columns.has(key)){
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

  invalidate_column(key: string, emit_change=true){
    if(!this.cached_columns.has(key)){
      // Seems like a bogus change, but might be there for some reason
      if(emit_change){
        // this.change.emit()
      }
      return
    }
    this.cached_columns.delete(key)
    // A bruteforce solution should work, there shouldn't be that many columns that it's problematic
    const candidate_columns = Object.keys(this.mapping)
    for(const new_key of candidate_columns){
      const column = this.mapping[new_key]
      let should_invalidate = false
      if(column.hasOwnProperty("fields")){
        for(let i=0; i<column.fields.length; i++){
          if(key == column.fields[i]){
            should_invalidate = true
            break
          }
        }
      } else if(Object.prototype.toString.call(column) === '[object String]'){
        if(key === column){
          should_invalidate = true
        }
      }
      if(should_invalidate){
        this.invalidate_column(new_key, false)
      }
    }
    if(emit_change){
      this.change.emit()
    }
  }

}
