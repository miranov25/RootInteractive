from bokeh.core.properties import Instance, String, Int, List
from bokeh.models import ColumnarDataSource


class DownsamplerCDS(ColumnarDataSource):

    __implementation__ = "DownsamplerCDS.ts"

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties

    source = Instance(ColumnarDataSource)
    nPoints = Int(default=300, help="Number of points to downsample CDS to")
    selectedColumns = List(String, default=[], help="Redundant, do not use this.")
    print("Import ", __implementation__)