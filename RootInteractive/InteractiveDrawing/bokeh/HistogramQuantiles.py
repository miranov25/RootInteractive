from bokeh.core.properties import Enum, Float, Instance, List, Nullable, String
from bokeh.model import Model
from bokeh.models import ColumnarDataSource


class HistogramQuantiles(ColumnarDataSource):
    """PHASE_5_2 stage 4 (FR-RI-3): the exact quantiles of a histogram's column over its selected rows, one row per
    level whose value is finite and inside the shown range (x_range). Columns x, level, label, dash, text."""

    __implementation__ = "HistogramQuantiles.ts"

    histogram = Instance(Model, help="The HistogramCDS: its source, sample, weights and filter give the rows")
    levels = List(Float, help="The levels, from 0 to 1")
    method = Enum("default", "linear", "inverted_cdf", default="default",
                  help="default: linear without weights, inverted_cdf with weights")
    text = Enum("none", "level", "value", "both", default="none", help="The text drawn at each line")
    x_range = Nullable(Instance(Model), default=None, help="The shown range; a value outside it draws nothing")
    # the computed columns, declared for Bokeh's legend check (the legend is on the column "label")
    column_names = List(String, default=["x", "level", "label", "dash", "text", "edge"])
    edge_range = Nullable(Instance(Model), default=None, help="The other axis' range: the texts stand at its far end")
