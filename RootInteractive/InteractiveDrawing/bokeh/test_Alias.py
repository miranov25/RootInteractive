from bokeh.io.showing import show
from bokeh.models.callbacks import CustomJS
from RootInteractive.InteractiveDrawing.bokeh.CDSAlias import CDSAlias
from RootInteractive.InteractiveDrawing.bokeh.CustomJSNAryFunction import CustomJSNAryFunction

from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.models.layouts import Column

from bokeh.plotting import Figure, output_file

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.random_sample(size=(2000, 6)), columns=list('ABCDEF'))

jsFunction = """
"use strict";
return a*x+y;
"""

sliderWidget = Slider(title='X', start=-20, end=20, step=1, value=10)

jsMapper = CustomJSNAryFunction(parameters={"a": sliderWidget.value}, fields=["x", "y"], func=jsFunction)

cdsOrig = ColumnDataSource(df)
cdsAlias = CDSAlias(source=cdsOrig, mapping={"a":{"field":"A"}, "b":{"field":"B"}, "a*x+b": {"fields":["A", "B"], "transform": jsMapper}})

sliderWidget.js_on_change("value", CustomJS(args = {"jsMapper": jsMapper, "cdsAlias": cdsAlias}, code="""
    jsMapper.parameters = {value: this.value}
    jsMapper.update_args()
    console.log("Boo!")
    cdsAlias.compute_functions()
"""))

output_file("test_Alias.html")
fig = Figure()
fig.scatter(x="a", y="b", source=cdsAlias)
fig.scatter(x="a", y="a*x+b", source=cdsAlias)
show(Column(fig, sliderWidget))

