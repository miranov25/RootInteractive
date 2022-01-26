from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
from RootInteractive.InteractiveDrawing.bokeh.bokehTools import bokehDrawArray
from RootInteractive.Tools.compressArray import arrayCompressionRelative16
import numpy as np
import pandas as pd
from bokeh.plotting import output_file

output_file("test_join.html")


x = np.arange(10000)
y = np.sqrt(x)
c = np.square(np.arange(100))
sourceArray = [
    {
        "name": "df2",
        "data": pd.DataFrame({"A": x, "B": y})
    },
    {
        "name": "joinA",
        "left": "df2",
        "right": None,
        "left_on": ["A"],
        "right_on": ["A"]
    },
    {
        "name": "histo0",
        "variables": ["B"],
        "weights": "A",
        "source": "joinA"
    }
]

df = pd.DataFrame({"A": c})

figureArray = [
    [["A"], ["B"],{"source":"df2"}],
    [["joinA.A"], ["joinA.df2.B"]],
    [["histo0.bin_center"], ["histo0.bin_count"]]
]

widgetParams = [
    ["multiSelect", ["A",1,2,3,4]]
]

figureLayout = [[0,1, {"height":200}], [2, {"height":50}], {"sizing_mode":"scale_width"}]
widgetDesc = [[0]]
def test_join():
    output_file("test_join_inner.html")
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, sourceArray=sourceArray, layout=figureLayout, widgetLayout=widgetDesc)

test_join()