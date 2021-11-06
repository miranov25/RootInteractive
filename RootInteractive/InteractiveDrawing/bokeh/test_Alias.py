from bokeh.io.showing import show
from bokeh.models.callbacks import CustomJS
from RootInteractive.InteractiveDrawing.bokeh.CDSAlias import CDSAlias
from RootInteractive.InteractiveDrawing.bokeh.CustomJSNAryFunction import CustomJSNAryFunction
from RootInteractive.InteractiveDrawing.bokeh.DownsamplerCDS import DownsamplerCDS

from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA

from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.models.layouts import Column

from bokeh.plotting import Figure, output_file

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.random_sample(size=(200000, 6)), columns=list('ABCDEF'))

parameterArray = [
    {"name": "size", "value":7, "range":[0, 30]},
    {"name": "legendFontSize", "value":"13px", "options":["9px", "11px", "13px", "15px"]},
    {"name": "paramX", "value":10, "range": [-20, 20]},
    {"name": "C_cut", "value": 1, "range": [0, 1]}
]

# This interface with two arrays is only for the case when functions are reused with different columns as target
jsFunctionArray = [
    {
        "name": "saxpy",
        "parameters": ["paramX"],
        "fields": ["x", "y"],
        "func": "return paramX*x+y"
    }
]

#This array shows the two options in action
aliasArray = [
    {
        "name": "A_mul_paramX_plus_B",
        "variables": ["A", "B"],
        "transform": "saxpy" 
    },
    {
        "name": "C_cut",
        "variables": ["C"],
        "parameters": ["C_cut"],
        "func": "return C < C_cut"
    },
    {
        "name": "efficiency_A",
        "variables": ["entries", "entries_C_cut"],
        "func": "return entries_C_cut / entries",
        "context": "histoA"
    },
    {
        "name": "efficiency_AC",
        "variables": ["entries", "entries_C_cut"],
        "func": "return entries_C_cut / entries",
        "context": "histoAC"
    }
]

figureArray = [
    [['A'], ['B', '4*A+B', 'A_mul_paramX_plus_B'], {"size":"size"}],
    [['histoA.bin_center'], ['efficiency_A'], {"context":"histoA", "size":"size"}],
    [['histoA.bin_center'], ['histoA.entries', 'histoA.entries_C_cut'], {"context":"histoA", "size":"size"}],
    [['histoAC.bin_center_0'], ['efficiency_AC'], {"context":"histoAC", "size":"size", "colorZvar": "histoAC.bin_center_1"}],
    {"size":"size", "legend_options": {"label_text_font_size": "legendFontSize"}}
]

histoArray = [
    {
        "name": "histoA", "variables": ["A"], "nbins": 10, "histograms": {
            "entries": None,
            "entries_C_cut": {
                "weights": "C_cut"
            }
        }
    },
    {
        "name": "histoAC", "variables": ["A", "C"], "nbins": [6, 6], 
        "histograms": {
            "entries": None,
            "entries_C_cut": {
                "weights": "C_cut"
            }
        }
    }
]

widgetParams=[
    ['range', ['A']],
    ['range', ['B', 0, 1, 0.1, 0, 1]],

    ['range', ['C'], {'type': 'minmax'}],
    ['range', ['D'], {'type': 'sigma', 'bins': 10, 'sigma': 3}],
    ['range', ['E'], {'type': 'sigmaMed', 'bins': 10, 'sigma': 3}],
    #['slider','F', ['@min()','@max()','@med','@min()','@median()+3*#tlm()']], # to be implmneted
    ['slider',["size"]],
    ['select',["legendFontSize"]],
    ['slider',["C_cut"]],
    ['slider',["paramX"]],
]

widgetLayoutDesc={
    "Selection": [[0, 1, 2], [3, 4], {'sizing_mode': 'scale_width'}],
    "Graphics": [[5, 6], {'sizing_mode': 'scale_width'}],
    "CustomJS functions": [[7, 8]]
    }

figureLayoutDesc=[
        [0, 1, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
        [2, 3, {'plot_height': 200, 'sizing_mode': 'scale_width'}]
        ]

tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]

def test_customJsFunction():
    jsFunction = "return a*x+y"

    sliderWidget = Slider(title='X', start=-20, end=20, step=1, value=10)

    jsMapper = CustomJSNAryFunction(parameters={"a": sliderWidget.value}, fields=["x", "y"], func=jsFunction)

    cdsOrig = ColumnDataSource(df)
    cdsAlias = CDSAlias(source=cdsOrig, mapping={"a":"A", "b":"B", "a*x+b": {"fields":["A", "B"], "transform": jsMapper}})
    cdsDownsampled = DownsamplerCDS(source = cdsAlias, selectedColumns=["a", "b", "a*x+b"])

    sliderWidget.js_on_change("value", CustomJS(args = {"jsMapper": jsMapper, "cdsAlias": cdsAlias}, code="""
        jsMapper.parameters = {x: this.value}
        jsMapper.update_args()
        cdsAlias.compute_functions()
    """))

    output_file("test_Alias.html")
<<<<<<< HEAD
    fig = Figure()
    fig.scatter(x="a", y="b", source=cdsDownsampled)
    fig.scatter(x="a", y="a*x+b", source=cdsDownsampled)
    show(Column(fig, sliderWidget))

def test_customJsFunctionBokehDrawArray():
    jsFunctionArray = [
        {
            "name": "saxpy",
            "parameters": ["paramX"],
            "fields": ["x", "y"],
            "func": "return paramX*x+y"
        }
    ]
    output_file("test_AliasBokehDraw.html")
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300, jsFunctionArray=jsFunctionArray,
                           aliasArray=aliasArray, histogramArray=histoArray)

def test_customJsFunctionBokehDrawArray_v():
    jsFunctionArray = [
        {
            "name": "saxpy",
            "parameters": ["paramX"],
            "fields": ["a", "b"],
            "v_func": "return a.map((el, idx) => paramX*a[idx]+b[idx])"
        }
    ]
    output_file("test_AliasBokehDraw_v.html")
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300, jsFunctionArray=jsFunctionArray,
                           aliasArray=aliasArray, histogramArray=histoArray)

test_customJsFunctionBokehDrawArray()
=======
    xxx=bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, widgetLayout=widgetLayoutDesc, parameterArray=parameterArray, aliasArray=aliasArray)

def test_makeColumns():
    paramDict = {"paramA": {"value": "5"}}
    aliasDict = {"saxpy": {"name": "saxpy", "fields": ["a", "x", "y"]}}
    cdsDict = {"histoA": {"nbins": 10}, None: df}
    varList, ctx_updated, memoized_columns = getOrMakeColumns(["1", "Y", "10*X+Y", "saxpy(paramA, X, Y+1)", "histoA.entries"], None, cdsDict, paramDict, aliasDict)
    assert len(varList) == 5
    assert ctx_updated[-1] == "histoA"
    print(ctx_updated)
    print(memoized_columns)

test_makeColumns()
>>>>>>> ced6460 (Added new interface for string parsing of variable names)
