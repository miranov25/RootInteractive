import pytest
import numpy as np
import pandas as pd
from bokeh.plotting import output_file

from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
from RootInteractive.InteractiveDrawing.bokeh.bokehTools import bokehDrawArray
from RootInteractive.Tools.compressArray import arrayCompressionRelative16

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
    ["range", ["A"]]
]

figureLayout = [[0,1, {"height":200}], [2, {"height":50}], {"sizing_mode":"scale_width"}]
widgetDesc = [[0]]


@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.feature("JOIN.cdsjoin.index0")
@pytest.mark.backend("browser")
@pytest.mark.layer("integration")
@pytest.mark.regression("PR-371")
def test_join():
    """Test CDSJoin with index join including index-0 (PR #371 regression)."""
    output_file("test_join_inner.html")
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, sourceArray=sourceArray, layout=figureLayout, widgetLayout=widgetDesc)


@pytest.mark.feature("DSL.gather_operation")
@pytest.mark.backend("browser")
@pytest.mark.layer("integration")
def test_gather():
    """Test gather notation (df2.Y[A] syntax)."""
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


# Remove standalone call - tests should only run via pytest
# test_join()
