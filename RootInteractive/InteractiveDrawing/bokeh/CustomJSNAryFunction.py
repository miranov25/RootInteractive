from bokeh.model import Model
from bokeh.core.properties import String, Dict, List, Any

class CustomJSNAryFunction(Model):
    __implementation__ = "CustomJSNAryFunction.ts"

    parameters = Dict(String, Any, help="Extra arguments to call the function with")
    fields = List(String, default=["x"], help="List of positional arguments - might be made optional in the future")
    func = String(serialized=True, help="Code to be computed on the client - scalar case")
    v_func = String(serialized=True, help="Code to be computed on the client - vector case")

