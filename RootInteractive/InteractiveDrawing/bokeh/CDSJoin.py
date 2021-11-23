from bokeh.core.properties import Instance, Any, List, String, Either, Dict
from bokeh.models import ColumnarDataSource


class CDSJoin(ColumnarDataSource):
    __implementation__ = "CDSJoin.ts"

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties
    left = Instance(ColumnarDataSource)
    right = Instance(ColumnarDataSource)
    on_left = List(String)
    on_right = List(String)
    join_type = String
    print("Import ", __implementation__)
