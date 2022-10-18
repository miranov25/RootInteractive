import {Model} from "model"
import * as p from "core/properties"

export namespace ConcatenatedString {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    components: p.Property<string[]>
  }
}

export interface ConcatenatedString extends ConcatenatedString.Attrs {}

export class ConcatenatedString extends Model {
  properties: ConcatenatedString.Props

  constructor(attrs?: Partial<ConcatenatedString.Attrs>) {
    super(attrs)
  }

  static __name__ = "ConcatenatedString"

  #is_dirty: boolean
  #value: string

  static init_ConcatenatedString() {
    this.define<ConcatenatedString.Props>(({Array, String})=>({
      components:  [Array(String), []]
    }))
  }

  initialize(){
    super.initialize()
    this.#is_dirty = false
  }

  connect_signals(): void {
    super.connect_signals()
    this.connect(this.properties.components.change, () => {this.#is_dirty = true})
  }

  compute(): string{
    if (this.#is_dirty){
        this.#value = String.prototype.concat(...this.components)
    }
    return this.#value
  }

  get value(){
    return this.compute()
  }

}
