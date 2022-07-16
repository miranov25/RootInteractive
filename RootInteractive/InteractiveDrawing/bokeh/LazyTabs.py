from bokeh.models.widgets import Tabs
from bokeh.core.properties import List, Instance
from bokeh.model import Model

class LazyTabs(Tabs):

    __implementation__ = "LazyTabs.ts"

    renderers = List(List(Instance(Model)))
