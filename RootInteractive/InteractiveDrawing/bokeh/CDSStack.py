from bokeh.core.properties import Instance, String, Int, Dict, List
from bokeh.models import ColumnarDataSource

class CDSStack(ColumnarDataSource):
    __implementation__ = "CDSStack.ts"

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties
    sources = List(Instance(ColumnarDataSource))
    mapping = Dict(String, Int)
    activeSources = List(String, help="selected data sources")
    print("Import ", __implementation__)
