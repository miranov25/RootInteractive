from bokeh.model import Model
from bokeh.core.properties import Bool

class RIFilter(Model):
    __implementation__ = "RIFilter.ts"
    # This class is a utility class that makes no sense to instantiate on its own

    active = Bool(True, help="The linear part of the transformation")
    invert = Bool(False, help="The constant component")

