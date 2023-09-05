from bokeh.io.showing import show
from bokeh.models.callbacks import CustomJS
from RootInteractive.InteractiveDrawing.bokeh.CDSAlias import CDSAlias
from RootInteractive.InteractiveDrawing.bokeh.CustomJSNAryFunction import CustomJSNAryFunction
from RootInteractive.InteractiveDrawing.bokeh.DownsamplerCDS import DownsamplerCDS

from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
from RootInteractive.InteractiveDrawing.bokeh.compileVarName import getOrMakeColumns

from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.models.layouts import Column

from bokeh.plotting import Figure, output_file

import pytest

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.random_sample(size=(200000, 6)), columns=list('ABCDEF'))

parameterArray = [
    {"name": "size", "value":7, "range":[0, 30]},
    {"name": "legendFontSize", "value":"13px", "options":["9px", "11px", "13px", "15px"]},
    {"name": "paramX", "value":10, "range": [-20, 20]},
    {"name": "C_cut", "value": 1, "range": [0, 1]},
    {"name": "CustomFunc", "value":"return Math.sin(A_mul_paramX_plus_B)"},
    {"name": "predictors", "value":["A","A_mul_paramX_plus_B"], "options":["A", "A_mul_paramX_plus_B", "B", "C", "custom_column"]}
]

# This interface with two arrays is only for the case when functions are reused with different columns as target
jsFunctionArray = [
    {
        "name": "saxpy",
        "parameters": ["paramX"],
        "fields": ["x", "y"],
        "func": "return paramX*x+y"
    },
    {
        "name": "linearFit",
        "type": "linearFit",
        "varX": "predictors",
        "varY": "A_mul_paramX_plus_B",
        "alpha": 1e-4,
        "weights": "C"
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
        "name": "C_accepted",
        "expr": "C < C_cut"
    },
    {
        "name": "efficiency_A",
        "variables": ["bin_count", "entries_C_cut"],
        "func": "return entries_C_cut / bin_count",
        "context": "histoA"
    },
    {
        "name": "efficiency_AC",
        "variables": ["bin_count", "entries_C_cut"],
        "func": "return entries_C_cut / bin_count",
        "context": "histoAC"
    },
    {
        "name": "custom_column",
        "func": "CustomFunc"
    },
    {
        "name": "BPred",
        "variables": "predictors",
        "transform": "linearFit"
    }
]

figureArray = [
    [['A'], ['B', '4*A+B', 'A_mul_paramX_plus_B', "custom_column", "BPred"], {"size":"size"}],
    [['bin_center'], ['efficiency_A', 'entries_C_cut / bin_count', "C_accepted/bin_count"], {"source":"histoA", "size":"size"}],
    [['bin_center'], ['bin_count', 'entries_C_cut', "C_accepted"], {"source":"histoA", "size":"size", "errY":["sqrt(bin_count)","sqrt(entries_C_cut)","sqrt(C_accepted)"]}],
    [['bin_center_0'], ['efficiency_AC'], {"source":"histoAC", "size":"size", "colorZvar": "bin_center_1"}],
    {"size":"size", "legend_options": {"label_text_font_size": "legendFontSize"}}
]

histoArray = [
    {
        "name": "histoA", "variables": ["A"], "nbins": 10, "histograms": {
            "entries": None,
            "entries_C_cut": {
                "weights": "C_accepted"
            },
            "C_accepted": {
                "weights": "C < C_cut"
            }
        }
    },
    {
        "name": "histoAC", "variables": ["A", "C"], "nbins": [6, 6], 
        "histograms": {
            "entries": None,
            "entries_C_cut": {
                "weights": "C_accepted"
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
    ['multiSelect', ["predictors"]],
    ['text', ["CustomFunc"], {"name": "CustomFunc"}]
]

widgetLayoutDesc={
    "Selection": [[0, 1, 2], [3, 4], {'sizing_mode': 'scale_width'}],
    "Graphics": [[5, 6], {'sizing_mode': 'scale_width'}],
    "CustomJS functions": [[7, 8, 9],["CustomFunc"]]
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
    cdsDownsampled = DownsamplerCDS(source = cdsAlias)

    sliderWidget.js_on_change("value", CustomJS(args = {"jsMapper": jsMapper, "cdsAlias": cdsAlias}, code="""
        jsMapper.parameters = {x: this.value}
        jsMapper.update_args()
    """))

    output_file("test_Alias.html")
    fig = Figure()
    fig.scatter(x="a", y="b", source=cdsDownsampled)
    fig.scatter(x="a", y="a*x+b", source=cdsDownsampled)
    show(Column(fig, sliderWidget))

def test_customJsFunctionBokehDrawArray():
    output_file("test_AliasBokehDraw.html")
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300, jsFunctionArray=jsFunctionArray,
                           aliasArray=aliasArray, histogramArray=histoArray)

def test_customJsFunctionBokehDrawArray_v():
    jsFunctionArray[0] = {
            "name": "saxpy",
            "parameters": ["paramX"],
            "fields": ["a", "b"],
            "v_func": """
                if($output == null || $output.length !== a.length){
                    $output = new Float64Array(a.length)
                }
                for(let i=0; i<$output.length; i++){
                    $output[i] = paramX*a[i]+b[i]
                }
                return $output
            """ 
        }
    
    output_file("test_AliasBokehDraw_v.html")
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300, jsFunctionArray=jsFunctionArray,
                           aliasArray=aliasArray, histogramArray=histoArray)
 
def test_Error():
    aliasArrayBad = aliasArray.copy()
    aliasArrayBad[1] = ("C_accepted", "C < C_accepted")
    with pytest.raises(KeyError):
        bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300, jsFunctionArray=jsFunctionArray,
                           aliasArray=aliasArrayBad, histogramArray=histoArray)


def test_makeColumns():
    varList, ctx_updated, memoized_columns, sources = (None, None, None, None)
    df = pd.DataFrame(np.random.random_sample(size=(200000, 3)), columns=list('XYZ'))
    df["boolY"] = df.eval("Y<Z")
    paramDict = {"paramA": {"value": "5"}}
    functionDict = {"saxpy": {"name": "saxpy", "fields": ["a", "x", "y"]}}
    cdsDict = {"histoA": {"nbins": 10, "type": "histogram", "variables": ["X"], "source": None}, None: {"data": df, "type": "source"}}
    aliasDict={}
    varList, ctx_updated, memoized_columns, sources = getOrMakeColumns(["1", "Y", "10*X+Y", "Y", "X*(Y**(5/2))", "X*(Y**(5/2))/Z", "sqrt(X)","paramA", "X if Y<paramA else Z","histoA.bin_count", "X<paramA or boolY"], None, cdsDict, paramDict, functionDict, aliasDict=aliasDict)
    assert len(varList) == 11
    assert len(sources) == 9
    assert ctx_updated[-2] == "histoA"
    varList, ctx_updated, memoized_columns, sources = getOrMakeColumns(["log(1+bin_count)", "-1<=-bin_count<1"], ["histoA"], cdsDict, paramDict, functionDict, aliasDict=aliasDict)
    print(ctx_updated)
    print(memoized_columns)
    print(sources)
    print(aliasDict)

