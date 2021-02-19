from bokeh.core.properties import Instance, String, Float, Int, List
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

    source = Instance(ColumnarDataSource)
    sample_variables = List(String)
    weights = String(default=None)
    nbins = List(Int)
    range = List(List(Float))
