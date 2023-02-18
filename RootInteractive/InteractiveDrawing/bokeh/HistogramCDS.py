from bokeh.core.properties import Instance, String, Float, Int, List, Dict, Any
from RootInteractive.InteractiveDrawing.bokeh.RIFilter import RIFilter
from bokeh.models import ColumnarDataSource

try:
    from bokeh.core.properties import Nullable
    nullable_available = True
except ImportError:
    nullable_available = False

class HistogramCDS(ColumnarDataSource):

    __implementation__ = "HistogramCDS.ts"

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties

    source = Instance(ColumnarDataSource, help="Source from which to take the data to histogram")
    sample = String()
    if nullable_available:
        filter = Nullable(Instance(RIFilter))
        weights = Nullable(String(), default=None)
        range = Nullable(List(Float))
        histograms = Dict(String, Nullable(Dict(String, Any)), default={}, help="""
            Dictionary of the values to histogram.
            Keys are the names of the resulting columns, values are dictionaries with the only option supported being weights, the value of which is the column name with weights.
        """)
    else:
        filter = Instance(RIFilter)
        weights = String(default=None)
        range = List(Float)
        histograms = Dict(String, Dict(String, Any), help="""
            Dictionary of the values to histogram.
            Keys are the names of the resulting columns, values are dictionaries with the only option supported being weights, the value of which is the column name with weights.
        """)
    nbins = Int()

    print("x", __implementation__)
