from bokeh.core.properties import Instance, List, Float, Int, Bool, String, Either, Dict, Enum
from bokeh.models import ColumnarDataSource
from RootInteractive.InteractiveDrawing.bokeh.HistoNdCDS import HistoNdCDS

try:
    from bokeh.core.properties import Nullable
    nullable_available = True
except ImportError:
    nullable_available = False

class HistoNdProfile(ColumnarDataSource):

    __implementation__ = "HistoNdProfile.ts"

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties

    source = Instance(HistoNdCDS)
    axis_idx = Int(default=0)
    quantiles = List(Float)
    sum_range = List(List(Float))
    unbinned = Bool(default=False)
    if nullable_available:
        weights = Nullable(Either(List(Nullable(String())), String()), default=None)
        # the display text of a weight column, {column: text}; the legend shows it instead of the column's name
        weights_labels = Nullable(Dict(String, String), default=None)
    else:
        weights = Either(List(String), String(), default=None)
        weights_labels = Dict(String, String)
    # PHASE_5_2 stage 3: the quantile of a cell. "subbin" is RootInteractive's present method (DEC-10); "linear" and
    # "inverted_cdf" are dfdraw's estimators over all the cell's rows (unbinned projections only)
    quantile_method = Enum("subbin", "linear", "inverted_cdf", default="subbin")
