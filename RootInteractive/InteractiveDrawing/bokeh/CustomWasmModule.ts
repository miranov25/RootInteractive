import {Model} from "model"
import * as p from "core/properties"

export namespace CustomWasmMosule {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    wasmBytes: p.Property<string>
  }
}

export interface CustomWasmModule extends CustomWasmModule.Attrs {}

export class CustomWasmModule extends Model {
  properties: CustomWasmModule.Props
  exports: WebAssembly.Exports

  constructor(attrs?: Partial<CustomWasmModule.Attrs>) {
    super(attrs)
  }

  static __name__ = "CustomWasmModule"

  static {
    this.define<CustomWasmModule.Props>(({String})=>({
      wasmBytes:  [String, ""],
    }))
  }

  initialize(){
    super.initialize()
    const bytes = new Uint8Array(atob(this.wasmBytes).split("").map(function (x) {
      return x.charCodeAt(0)
    }))
    WebAssembly.instantiate(bytes, {}).then((result) => {
      const instance = result.instance
      const exports = instance.exports
      if (exports) {
        // TODO: Add functionalty for calling the function and malloc/free
        this.exports = exports
        console.log("WASM module loaded successfully")
      }
    })
  }

  connect_signals(): void {
    super.connect_signals()
  }

}
