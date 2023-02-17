from bokeh.core.properties import Instance, String, Float, Int, List, Dict, Any
from bokeh.models import ColumnarDataSource
from RootInteractive.InteractiveDrawing.bokeh.RIFilter import RIFilter

try:
    from bokeh.core.properties import Nullable
    nullable_available = True
except ImportError:
    nullable_available = False

class HistoNdCDS(ColumnarDataSource):

    __implementation__ = "HistoNdCDS.ts"

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties

    source = Instance(ColumnarDataSource, help="Source from which to take the data to histogram")
    sample_variables = List(String, help="Names of the columns used for binning")
    # TODO: Support auto nbins in the future - 2n-th root of total entries?
    nbins = List(Int, help="Number of bins")
    if nullable_available:
        filter = Nullable(Instance(RIFilter))
        range = Nullable(List(Nullable((List(Float)))), help="Ranges in the same order as sample_variables")
        weights = Nullable(String(), default=None)
        histograms = Dict(String, Nullable(Dict(String, Any)), default={}, help="""
        Dictionary of the values to histogram.
        Keys are the names of the resulting columns, values are dictionaries with the only option supported being weights, the value of which is the column name with weights.
            """)
    else:
        filter = Instance(RIFilter)
        range = List(List(Float), help="Ranges in the same order as sample_variables")
        weights = String(default=None)
        histograms = Dict(String, Dict(String, Any), help="""
        Dictionary of the values to histogram.
        Keys are the names of the resulting columns, values are dictionaries with the only option supported being weights, the value of which is the column name with weights.
            """)