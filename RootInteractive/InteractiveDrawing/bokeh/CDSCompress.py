from bokeh.core.properties import Instance, String, Float, Int, List, Any, Dict
from bokeh.models import ColumnDataSource


class CDSCompress(ColumnDataSource):
    __javascript__ = ["https://cdnjs.cloudflare.com/ajax/libs/pako/2.0.2/pako.min.js","https://cdnjs.cloudflare.com/ajax/libs/Base64/1.1.0/base64.js"]
    __implementation__ = "CDSCompress.ts"

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties
    inputData=Dict(String, Any)
    sizeMap=Dict(String, Any)
    print("Import ", __implementation__)
