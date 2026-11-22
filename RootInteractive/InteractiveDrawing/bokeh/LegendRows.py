from bokeh.core.properties import Instance, List, Nullable, String
from bokeh.model import Model
from bokeh.models import ColumnarDataSource


class LegendRows(ColumnarDataSource):
    """PHASE_5_2 stage 5 (FR-RI-5): the rows of a grid legend's stand-in renderer: one per distinct value of field in
    the cells' sources (all cells with this legend entry; in order of appearance), the other columns (names) NaN;
    without a field, one row."""

    __implementation__ = "LegendRows.ts"

    sources = List(Instance(Model), help="The cell renderers' sources (for a legend on a column)")
    field = Nullable(String, default=None, help="The legend's column")
    names = List(String, help="The other columns the glyph reads (all NaN)")
    # declared for Bokeh's legend check, which reads column_names
    column_names = List(String)
