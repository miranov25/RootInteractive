from RootInteractive.InteractiveDrawing.bokeh.RIFilter import RIFilter
from bokeh.core.properties import List, Instance

class LazyIntersectionFilter(RIFilter):
    __implementation__ = "LazyIntersectionFilter.ts"

    filters = List(Instance(RIFilter))

