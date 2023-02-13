from RootInteractive.InteractiveDrawing.bokeh.RIFilter import RIFilter
from bokeh.core.properties import Instance, String, Tuple, Float
from bokeh.models.sources import ColumnarDataSource

class RangeFilter(RIFilter):
    __implementation__ = "RangeFilter.ts"

    source = Instance(ColumnarDataSource, help="Column data source to select from")
    field = String(help="The field to check") # May change it to a formula instead
    range = Tuple(Float(), Float(), help="The range to use - default options are [-Infinity, Infinity]")
