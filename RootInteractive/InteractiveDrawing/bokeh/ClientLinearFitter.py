from bokeh.model import Model
from bokeh.core.properties import String, List, Float, Instance, Nullable
from bokeh.models.sources import ColumnarDataSource

class ClientLinearFitter(Model):
    __implementation__ = "ClientLinearFitter.ts"

    source = Instance(ColumnarDataSource, help="The data source to fit from")
    varX = List(String, default=[], help="List of predictors")
    varY = String(help="The variable to predict")
    alpha = Float(default=0, help="Regularization constant for ridge regression")
    weights = Nullable(String(), help="Optional weights")
