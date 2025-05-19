from bokeh.core.properties import String, Dict, List, Any, Bool, Int
from bokeh.model import Model

class CustomWasmModule(Model):
    __implementation__ = "CustomWasmModule.ts"

    wasmBytes = String(help="Wasm module to be used - right now is converted to UInt8Array")
