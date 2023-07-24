from bokeh.core.properties import Instance, List, Float, Int, Bool, String, Either
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
    else:
        weights = Either(List(String), String(), default=None)
