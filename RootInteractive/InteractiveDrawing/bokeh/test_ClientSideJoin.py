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
        "data": {"X":x,"Y":y}
    },
    {
        "name": "joinA",
        "left": "df2",
        "right": None,
        "left_on": ["X"],
        "right_on": ["A"]
    },
    {
        "name": "histo0",
        "variables": ["Y"],
        "weights": "A",
        "source": "joinA"
    }
]

df = pd.DataFrame({"A": c})

figureArray = [
    [["df2.X"], ["df2.Y"]],
    [["joinA.X"], ["joinA.df2.X"]],
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

def test_gather():
    output_file("test_gather.html")
    sourceArrayGather = [
        {
            "name": "df2",
            "data": {"X":x,"Y":y}
        },
        {
            "name": "histo0",
            "variables": ["df2.Y[A]"],
            "weights": "A",
        }
    ]
    figureArray = [
        [["df2.X"], ["df2.Y"]],
        [["histo0.bin_center"], ["histo0.bin_count"]],
        [["df2.X[A]"], ["df2.Y[A]"]]
    ]
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, sourceArray=sourceArrayGather, layout=figureLayout, widgetLayout=widgetDesc)

test_gather()