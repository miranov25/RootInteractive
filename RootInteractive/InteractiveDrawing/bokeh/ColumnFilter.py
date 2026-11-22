from RootInteractive.InteractiveDrawing.bokeh.RIFilter import RIFilter
from bokeh.core.properties import Instance, Nullable, String
from bokeh.models.sources import ColumnarDataSource

class ColumnFilter(RIFilter):
    __implementation__ = "ColumnFilter.ts"

    source = Instance(ColumnarDataSource, help="Column data source to select from")
    field = String(help="The field to check")
    # PHASE_5_2 stage 1: one line per weight series. With a value, a row passes when its field, as text, equals it.
    value = Nullable(String, default=None, help="If set, a row passes when str(field) == value; else when field is true")

