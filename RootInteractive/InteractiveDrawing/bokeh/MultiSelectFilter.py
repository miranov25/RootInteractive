from RootInteractive.InteractiveDrawing.bokeh.RIFilter import RIFilter
from bokeh.core.properties import Instance, String, Dict, List, Int
from bokeh.models.sources import ColumnarDataSource

class MultiSelectFilter(RIFilter):
    __implementation__ = "MultiSelectFilter.ts"

    source = Instance(ColumnarDataSource, help="Column data source to select from")
    selected = List(String, help="Selected values")
    mapping = Dict(String, Int, help="Dictionary converting labels to values to use")
    mask = Int(default=-1, help="AND mask to apply before using multiselect")
    field = String(help="The field to check")
    how = String(default="any", help="The operation to use for bitmask - supportedo ptions so far are any and all")

