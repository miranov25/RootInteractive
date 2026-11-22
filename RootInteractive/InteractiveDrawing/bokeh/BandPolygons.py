from bokeh.core.properties import Instance, String
from bokeh.models import ColumnarDataSource


class BandPolygons(ColumnarDataSource):
    """PHASE_5_2 stage 3 (FR-RI-2): the polygons of a filled band between two columns of a source, one per run of rows
    whose x, lower and upper values are all finite (a NaN breaks the band). Columns xs and ys, for Patches."""

    __implementation__ = "BandPolygons.ts"

    source = Instance(ColumnarDataSource, help="The rows of the band, in order (a profile's downsampler)")
    x = String(help="The column of the X positions")
    lower = String(help="The column of the lower edge")
    upper = String(help="The column of the upper edge")
