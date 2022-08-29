from bokeh.core.properties import Instance, String, List, Bool, Float
from bokeh.models import ColumnarDataSource


class HistoStatsCDS(ColumnarDataSource):

    __implementation__ = "HistoStatsCDS.ts"

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties

    sources = List(Instance(ColumnarDataSource))
    names = List(String)
    bin_centers = List(String)
    bincount_columns = List(String)
    rowwise = Bool()
    quantiles = List(Float)
    compute_quantile = List(Bool)
    edges_left = List(String)
    edges_right = List(String)
    sum_range = List(List(Float))
