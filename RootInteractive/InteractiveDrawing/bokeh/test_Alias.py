from bokeh.io.showing import show
from bokeh.models.callbacks import CustomJS
from numpy.lib.utils import source
from RootInteractive.InteractiveDrawing.bokeh.CDSAlias import CDSAlias
from RootInteractive.InteractiveDrawing.bokeh.CustomJSNAryFunction import CustomJSNAryFunction
from RootInteractive.InteractiveDrawing.bokeh.DownsamplerCDS import DownsamplerCDS

from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.models.layouts import Column

from bokeh.plotting import Figure, output_file

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.random_sample(size=(200000, 6)), columns=list('ABCDEF'))

jsFunction = """
"use strict";
return a*x+y;
"""

sliderWidget = Slider(title='X', start=-20, end=20, step=1, value=10)

jsMapper = CustomJSNAryFunction(parameters={"a": sliderWidget.value}, fields=["x", "y"], func=jsFunction)

cdsOrig = ColumnDataSource(df)
cdsAlias = CDSAlias(source=cdsOrig, mapping={"a":{"field":"A"}, "b":{"field":"B"}, "a*x+b": {"fields":["A", "B"], "transform": jsMapper}})
cdsDownsampled = DownsamplerCDS(source = cdsAlias, selectedColumns=["a", "b", "a*x+b"])

sliderWidget.js_on_change("value", CustomJS(args = {"jsMapper": jsMapper, "cdsAlias": cdsAlias}, code="""
    jsMapper.parameters = {value: this.value}
    jsMapper.update_args()
    cdsAlias.compute_functions()
"""))

output_file("test_Alias.html")
fig = Figure()
fig.scatter(x="a", y="b", source=cdsDownsampled)
fig.scatter(x="a", y="a*x+b", source=cdsDownsampled)
show(Column(fig, sliderWidget))

