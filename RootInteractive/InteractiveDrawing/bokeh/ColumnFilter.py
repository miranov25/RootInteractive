from RootInteractive.InteractiveDrawing.bokeh.RIFilter import RIFilter
from bokeh.core.properties import Instance, String
from bokeh.models.sources import ColumnarDataSource

class ColumnFilter(RIFilter):
    __implementation__ = "ColumnFilter.ts"

    source = Instance(ColumnarDataSource, help="Column data source to select from")
    field = String(help="The field to check")

