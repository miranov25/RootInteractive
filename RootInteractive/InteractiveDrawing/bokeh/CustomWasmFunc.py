from bokeh.model import Model
from bokeh.core.properties import String, Dict, List, Any, Int, Instance
from RootInteractive.InteractiveDrawing.bokeh.CustomWasmModule import CustomWasmModule  

class CustomWasmFunc(Model):
    __implementation__ = "CustomWasmFunc.ts"

    parameters = Dict(String, Any, help="Extra arguments to call the function with")
    fields = List(String, default=[], help="List of positional arguments - might be made optional in the future")
    #n_out = Int(default=1, help="Number of output parameters - in vector case they are merged into one matrix")
    func = String(default="", help="The function to call")
    module = Instance(CustomWasmModule, help="The module to call the function from")