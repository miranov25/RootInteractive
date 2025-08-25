from bokeh.model import Model
from bokeh.core.properties import String, Dict, List, Any, Bytes

class OrtFunction(Model):
    __javascript__ = ["https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"]
    __implementation__ = "OrtFunction.ts"

    parameters = Dict(String, Any, help="Extra arguments to call the function with")
    fields = List(String, default=[], help="List of positional arguments - might be made optional in the future")
    v_func = String(help="The ONNX model to use - right now is converted to UInt8Array")