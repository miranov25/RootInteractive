from bokeh.model import Model
from bokeh.core.properties import Instance, String, Dict, List, Any, Int
from bokeh.models.widgets import MultiSelect
from bokeh.models.sources import ColumnarDataSource

class MultiSelectFilter(Model):
    __implementation__ = "MultiSelectFilter.ts"

    source = Instance(ColumnarDataSource, help="Column data source to select from")
    widget = Instance(MultiSelect, help="MultiSelect controlling the filter")
    mapping = Dict(String, Int, help="Dictionary converting labels to values to use")
    mask = Int(default=-1, help="AND mask to apply before using multiselect")
    field = String(help="The field to check")
    how = String(default="any", help="The operation to use for bitmask - supportedo ptions so far are any and all")

