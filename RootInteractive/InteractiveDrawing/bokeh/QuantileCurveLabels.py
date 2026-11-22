from bokeh.core.properties import Float, Instance, List, Nullable, String
from bokeh.model import Model
from bokeh.models import ColumnarDataSource


class QuantileCurveLabels(ColumnarDataSource):
    """PHASE_5_2 stage 7 (option (y)): the boxed labels of quantile curves on a 2-D histogram, one per curve with a
    finite value, all at the X bin nearest a fraction of the shown X range. Columns x, y, text, color, level."""

    __implementation__ = "QuantileCurveLabels.ts"

    source = Instance(Model, help="The profile (HistoNdProfile) that holds the curves")
    x = String(default="bin_center_0", help="The profile's X column")
    curves = List(String, help="The profile's column of each curve (quantile_0, ...)")
    texts = List(String, help="The label of each curve (\"5%\", \"50%\")")
    colors = List(String, help="The colour of each curve")
    levels = List(Float, help="The level of each curve")
    x_range = Nullable(Instance(Model), default=None, help="The shown X range")
    fraction = Float(default=0.4, help="Where the labels stand: a fraction of the shown X range")
    # the computed columns, declared for Bokeh's checks
    column_names = List(String, default=["x", "y", "text", "color", "level"])
