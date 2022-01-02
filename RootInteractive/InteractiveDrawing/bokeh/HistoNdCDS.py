from bokeh.core.properties import Instance, String, Float, Int, List, Dict, Any
from bokeh.models import ColumnarDataSource


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
    weights = String(default=None)
    # TODO: Support auto nbins in the future - 2n-th root of total entries?
    nbins = List(Int, help="Number of bins")
    # TODO: When migrating to new version of bokeh, make this Nullable
    range = List(List(Float), help="Ranges in the same order as sample_variables")
    # TODO: Make this nullable too
    histograms = Dict(String, Dict(String, Any), default={"entries": {}}, help="""
    Dictionary of the values to histogram.
    Keys are the names of the resulting columns, values are dictionaries with the only option supported being weights, the value of which is the column name with weights.
        """)