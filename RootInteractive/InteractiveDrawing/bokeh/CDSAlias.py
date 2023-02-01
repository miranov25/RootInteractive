from bokeh.core.properties import Instance, String, Any, Dict, Bool, List
from bokeh.models import ColumnarDataSource
from bokeh.model import Model


class CDSAlias(ColumnarDataSource):
    __implementation__ = "CDSAlias.ts"

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties
    source=Instance(ColumnarDataSource, help="The source to draw from")
    mapping=Dict(String, Any, help="The mapping from new columns to old columns and possibly mappers")
    includeOrigColumns=Bool()
    columnDependencies=List(Instance(Model), [], help="Hacky way of improving performance and fixing a weird serialization bug")
    print("Import ", __implementation__)
