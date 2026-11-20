from bokeh.core.properties import Instance, Int, Bool, Nullable, List, String
from bokeh.models import ColumnarDataSource
from RootInteractive.InteractiveDrawing.bokeh.CDSAlias import CDSAlias
from RootInteractive.InteractiveDrawing.bokeh.RIFilter import RIFilter

class DownsamplerCDS(ColumnarDataSource):

    __implementation__ = "DownsamplerCDS.ts"

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties

    source = Instance(CDSAlias)
    nPoints = Int(default=300, help="Number of points to downsample CDS to")
    watched = Bool()
    filter = Nullable(Instance(RIFilter))
    # Columns that the chain behind this source computes only when the page draws, for example a profile's "weights"
    # or a stack's "_source_index". The page builder declares a legend's column here, so that Bokeh's legend checks
    # find it: in Python (LegendItem reads source.column_names) and when the page loads (DownsamplerCDS.ts columns()).
    column_names = List(String)
    print("Import ", __implementation__)