import pytest
import numpy as np
import pandas as pd
from bokeh.plotting import output_file

from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
from RootInteractive.InteractiveDrawing.bokeh.bokehTools import mergeFigureArrays
from RootInteractive.Tools.compressArray import arrayCompressionRelative16
from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate import getDefaultVarsNormAll

from RootInteractive.Tools.generators.toy_event_generator import generate_event_display

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


@pytest.mark.feature("DSL.gather_operation")
@pytest.mark.backend("browser")
@pytest.mark.layer("integration")
def test_gather_realistic():
    output_file("test_gather_realistic.html")
    (events, tracks, clusters) = generate_event_display()
    cdsArray = [
        {"name": "events", "data": events},
        {"name": "tracks", "data": tracks}
    ]
    aliasArray, jsFunctionArray, variables, parameterArray, widgetParams, widgetLayoutDesc, \
        histoArray, figureArray, figureLayoutDesc = getDefaultVarsNormAll(
            variables=list(clusters.keys()) + ["events.vertex_x[event_id]", "events.vertex_y[event_id]","events.vertex_z[event_id]","events.n_tracks[event_id]",
                                           "tracks.pt[track_id]", "tracks.eta[track_id]", "tracks.phi[track_id]", "tracks.charge[track_id]"], 
            multiAxis="weights")
    widgetsSelect = [
        ['range', ['events.n_tracks[event_id]', events["n_tracks"].min(), events["n_tracks"].max()], {"name":"n_tracks", "bins":20}],
        ['range', ['tracks.pt[track_id]', tracks["pt"].min(), tracks["pt"].max()], {"name":"pt", "bins":20}],
        ['range', ['tracks.eta[track_id]', tracks["eta"].min(), tracks["eta"].max()], {"name":"eta", "bins":20}]
        ]
    selectionTab = [
        ["n_tracks", "pt", "eta"]
    ]
    widgetParams = mergeFigureArrays(widgetParams, widgetsSelect)
    widgetLayoutDesc["Select"] = selectionTab
    bokehDrawSA.fromArray(clusters, None, figureArray, widgetParams, sourceArray=histoArray + cdsArray, layout=figureLayoutDesc,
                           widgetLayout=widgetLayoutDesc, aliasArray=aliasArray, arrayCompression=arrayCompressionRelative16,
                           parameterArray=parameterArray, jsFunctionArray=jsFunctionArray)


# Remove standalone call - tests should only run via pytest
test_gather_realistic()
