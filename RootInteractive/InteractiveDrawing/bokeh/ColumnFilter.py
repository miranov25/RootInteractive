from bokeh.model import Model
from bokeh.core.properties import Instance, String
from bokeh.models.sources import ColumnarDataSource

class ColumnFilter(Model):
    __implementation__ = "ColumnFilter.ts"

    source = Instance(ColumnarDataSource, help="Column data source to select from")
    field = String(help="The field to check")

