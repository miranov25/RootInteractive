import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {ColumnDataSource} from "models/sources/column_data_source"
import * as p from "core/properties"

export namespace CDSCompress {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnDataSource>
      inputData :    p.Property<Record<string, any>>
  }
}

export interface CDSCompress extends CDSCompress.Attrs {}

export class CDSCompress extends ColumnarDataSource {
  properties: CDSCompress.Props

  constructor(attrs?: Partial<CDSCompress.Attrs>) {
    super(attrs)
  }

  static __name__ = "CDSCompress"

  static init_CDSCompress() {

    this.define<CDSCompress.Props>(({Ref})=>({
      source:  [Ref(ColumnDataSource)],
        inputData:    [ p.Instance ]
    }))
  }

  initialize(): void {
    super.initialize()
    this.data = {}
    console.info("CDSCompress::initialize")
    //this.update_range()
  }


  public view: number[] | null

}
