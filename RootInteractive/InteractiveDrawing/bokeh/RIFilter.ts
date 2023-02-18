import {Model} from "model"
import * as p from "core/properties"

export namespace RIFilter {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    active: p.Property<boolean>
    invert: p.Property<boolean>
  }
}

export interface RIFilter extends RIFilter.Attrs {}

export class RIFilter extends Model {
  properties: RIFilter.Props

  constructor(attrs?: Partial<RIFilter.Attrs>) {
    super(attrs)
  }

  static __name__ = "RIFilter"

  static init_RIFilter() {
    this.define<RIFilter.Props>(({Boolean})=>({
      active:  [Boolean, true],
      invert: [Boolean, false]
    }))
  }

  initialize(){
    super.initialize()
  }

  connect_signals(): void {
    super.connect_signals()

  }

  public v_compute(): boolean[]{
    return []
  }

  public get_indices(): number[]{
    return []
  }

}
