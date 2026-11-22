from bokeh.model import Model
from bokeh.core.properties import String, Dict, List, Any, Bool, Int

class CustomJSNAryFunction(Model):
    __implementation__ = "CustomJSNAryFunction.ts"

    parameters = Dict(String, Any, help="Extra arguments to call the function with")
    fields = List(String, default=[], help="List of positional arguments - might be made optional in the future")
    func = String(help="Code to be computed on the client - scalar case")
    v_func = String(help="Code to be computed on the client - vector case")
    auto_fields = Bool(default=False, help="Automatically try to figure out used variables using regular expression - only used for text widget, not general")
    # PHASE_5 Group 6c (TQ-D3): set only in the browser; the page shows it on the text field of the function
    problem = String(default="", help="Why the text is not in force or computes nothing; empty when it works")
    n_out = Int(default=1, help="Number of output parameters - in vector case they are merged into one matrix")

