from bokeh.model import Model
from bokeh.core.properties import String, List

# Simple wrapper for javascript String.prototype.concat

class ConcatenatedString(Model):
    __implementation__ = "ConcatenatedString.ts"

    components = List(String)