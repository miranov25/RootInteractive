from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.Tools.aliTreePlayer import *
from bokeh.io import curdoc
import os
import sys
import pytest
#import pickle
from pandas import CategoricalDtype
import typing as ty
import math

if "ROOT" in sys.modules:
    from ROOT import TFile, gSystem

output_file("test_bokehDrawSA.html")
# import logging

if "ROOT" in sys.modules:
    try: # TODO -use uproot to convert from panda to tree
        TFile.SetCacheFileDir("../../data/")
        tree, treeList, fileList = LoadTrees("echo http://rootinteractive.web.cern.ch/RootInteractive/data/tutorial/bokehDraw/treeABCD.root", ".*", ".*ABCDE.*", ".*", 0)
        tree.SetAlias("errY","A*0.02+0.02")
        AddMetadata(tree, "A.AxisTitle", "A (cm)")
        AddMetadata(tree, "B.AxisTitle", "B (cm/s)")
        AddMetadata(tree, "C.AxisTitle", "B (s)")
        AddMetadata(tree, "D.AxisTitle", "D (a.u.)")
        AddMetadata(tree, "A.Description", "Lorem ipsum")
        AddMetadata(tree, "B.Description", "The velocity B")
    except:
        pass


df = pd.DataFrame(np.random.random_sample(size=(20000, 6)), columns=list('ABCDEF'))
initMetadata(df)
MARKERS = ['hex', 'circle_x', 'triangle','square']
markerFactor=factor_mark('DDC', MARKERS, ["A0","A1","A2","A3","A4"] )
colorFactor=factor_cmap('DDC', 'Category10_6', ["A0","A1","A2","A3","A4"] )

mapDDC={0:"A0",1:"A1",2:"A2",3:"A3",4:"A4"}
df["B"]=np.linspace(0,1,20000)
df.eval("Bool=A>0.5", inplace=True)
df.eval("BoolB=B>0.5", inplace=True)
df.eval("BoolC=C>0.1", inplace=True)
df["A"]=df["A"].round(3)
df.loc[15, "A"] = math.nan
df["B"]=df["B"].round(3)
df["C"]=df["C"].round(3)
df["D"]=df["D"].round(3)
df["AA"]=((df.A*10).round(0)).astype(CategoricalDtype(ordered=True))
df["CC"]=((df.C*5).round(0)).astype(int)
df["DD"]=((df.D*4).round(0)).astype(int)
df["DDC"]=((df.D*4).round(0)).astype(int).map(mapDDC)
df["EE"]=(df.E*4).round(0)
df['errY']=df.A*0.02+0.02
df['maskAC']=2*(df['A']>.5)|1*(df['C']>.5)
df['ones']=1
df.head(10)
df.meta.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)", 'D.AxisTitle': "D (a.u.)", 'E.AxisTitle': "Category", "A.Description": "The distance A"}

parameterArray = [
    {"name": "colorZ", "value":"A", "options":["A", "B", "EE"]},
    {"name": "X", "value":"A", "options":["A", "B", "D"]},
    {"name": "size", "value":7, "range":[0, 30]},
    {"name": "legendFontSize", "value":"13px", "options":["9px", "11px", "13px", "15px"]},
    {"name": "legendVisible", "value":True},
    {"name": "nPoints", "range":[0, 1200], "value": 1000},
    {"name": "transformX", "value":None, "options":[None, "sqrt", "arctan2", "lambda x,y: y*cos(x+paramX)"]},
    {"name": "transformY", "value":None, "options":[None, "sqrt", "lambda x,y: sqrt(x*x+y*y)", "lambda x,y: y*sin(x+paramX)", "lambda x,y: y/x"]},
    {"name": "paramX", "value":0, "range":[-6,6]}
]

figureArray = [
#   ['A'], ['C-A'], {"color": "red", "size": 7, "colorZvar":"C", "filter": "A<0.5"}],
    [['A'], ['A*A-C*C'], {"color": "red", "size": 2, "colorZvar": "A", "varZ": "C", "errY": "errY", "errX":"0.01"}],
    [['X'], ['C+A', 'C-A', 'A/A'], {"name": "fig1"}],
    [['B'], ['C+B', 'C-B'], { "colorZvar": "B", "errY": "errY", "rescaleColorMapper": True, "colorAxisLabel": "B (cm/s)"}],
    [['D'], ['(A+B+C)*DD'], {"colorZvar": "colorZ", "size": 10, "errY": "errY"} ],
#    [['D'], ['D*10'], {"size": 10, "errY": "errY","markers":markerFactor, "color":colorFactor,"legend_field":"DDC"}],
    #marker color works only once - should be constructed in wrapper
    [['D'], ['D*10'], {"size": 10, "errY": "errY"}],
    ['selectionTable', {"name":"selection"}],
    ['descriptionTable', {"name":"description"}],
    {"size":"size", "y_transform":"transformY", "x_transform":"transformX", "legend_options": {"label_text_font_size": "legendFontSize", "visible": "legendVisible"}}
]

widgets="slider.A(0,1,0.05,0,1), slider.B(0,1,0.05,0,1), slider.C(0,1,0.01,0.1,1), slider.D(0,1,0.01,0,1), checkbox.Bool(1)"
tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)"), ("ErrY", "@errY")]

widgetParams=[
    ['range', ['A']],
    ['range', ['B', 0, 1, 0.1, 0, 1], {"type":"user", "index": True}],

    ['spinnerRange', ['C'], {"name": "widgetC"}],
    ['range', ['D'], {'type': 'sigma', 'bins': 10, 'sigma': 3}],
    ['range', ['E'], {'type': 'sigmaMed', 'bins': 10, 'sigma': 3}],
    ['slider', ['AA'], {'bins': 10, "toggleable":True, "name": "widgetAA"}],
    ['multiSelect', ["DDC", "A2", "A3", "A4", "A0", "A1"]],
    ['multiSelectBitmask', ["maskAC"], {"mapping": {"A": 2, "C": 1}, "how":"all", "title": "maskAC(all)"}],
    ['select',["CC", 0, 1, 2, 3], {"default": 1}],
    ['multiSelect',["BoolB"]],
    ['textQuery', {"title": "selection", "name":"selectionText"}],
    #['slider','F', ['@min()','@max()','@med','@min()','@median()+3*#tlm()']], # to be implmneted
    ['select',["colorZ"], {"name": "colorZ"}],
    ['select',["X"], {"name": "X"}],
    ['slider',["size"], {"name": "markerSize"}],
    ['select',["legendFontSize"], {"name": "legendFontSize"}],
    ['toggle', ['legendVisible'], {"name": "legendVisible"}],
    ['spinner', ['nPoints'], {"name": "nPointsRender"}],
    ['select', ['transformX'], {"name": "transformX"}],
    ['select', ['transformY'], {"name": "transformY"}],
    ['slider', ['paramX'], {"name":"paramX"}],
    ['spinnerRange', ['ones'], {"name":"ones"}]
]

widgetLayoutDesc={
    "Selection": [[0, 1, "widgetC"], [3, 4, 'ones'], ["widgetAA", 6],[7,8, "selectionText"], {'sizing_mode': 'scale_width'}],
    "Graphics": [["colorZ", "X", "markerSize"], ["legendFontSize", "legendVisible", "nPointsRender"], ["transformX", "transformY", "paramX"], {'sizing_mode': 'scale_width'}]
    }

figureLayoutDesc={
    "A": [
        [0, "fig1", 2, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
        ],
    "B": [
        [3, 4, {'commonX': 1, 'y_visible': 3, 'x_visible':1, 'plot_height': 100}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
        ],
    "Description": [["description"],{'plot_height': 100, 'sizing_mode': 'scale_width'} ],
    "selectionTable": [["selection"],{'plot_height': 100, 'sizing_mode': 'scale_width'}]
}

def test_invalidWidget():
    widgetParamsBroken = widgetParams.copy()
    widgetParamsBroken[-1] = ['spinneRange', ['ones'], {"name":"ones"}]
    with pytest.raises(NotImplementedError):
        fig=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParamsBroken, layout=figureLayoutDesc, tooltips=tooltips,widgetLayout=widgetLayoutDesc,sizing_mode=None, parameterArray=parameterArray, nPointRender="nPoints")

def test_record(record_property: ty.Callable[[str, ty.Any], None]):
    record_property("value1", "value1")
    record_property("value2", "value2")

def testBokehDrawArrayWidget(record_property: ty.Callable[[str, ty.Any], None]):
    output_file("test_BokehDrawArrayWidget.html")
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", parameterArray=parameterArray, nPointRender="nPoints", meta={"A.Description":"Test description for overriding metadata"})
    record_property("html_size",os.stat("test_BokehDrawArrayWidget.html").st_size)
    record_property("cdsOrig_size",len(fig.cdsOrig.column_names))


def testBokehDrawArrayWidgetNoScale(record_property: ty.Callable[[str, ty.Any], None]):
    output_file("test_BokehDrawArrayWidgetNoScale.html")
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips,widgetLayout=widgetLayoutDesc,sizing_mode=None, parameterArray=parameterArray, nPointRender="nPoints")
    record_property("html_size",os.stat("test_BokehDrawArrayWidgetNoScale.html").st_size)
    record_property("cdsOrig_size",len(fig.cdsOrig.column_names))

def testBokehDrawArrayDownsample(record_property: ty.Callable[[str, ty.Any], None]):
    output_file("test_BokehDrawArrayDownsample.html")
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, widgetLayout=widgetLayoutDesc, parameterArray=parameterArray, nPointRender="nPoints")
    record_property("html_size",os.stat("test_BokehDrawArrayWidget.html").st_size)
    record_property("cdsOrig_size",len(fig.cdsOrig.column_names))


def testBokehDrawArrayQuery(record_property: ty.Callable[[str, ty.Any], None]):
    output_file("test_BokehDrawArrayQuery.html")
    df0 = df.copy()
    fig=bokehDrawSA.fromArray(df0, "BoolC == True", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, widgetLayout=widgetLayoutDesc, nPointRender="nPoints", parameterArray=parameterArray)
    record_property("html_size",os.stat("test_BokehDrawArrayWidget.html").st_size)
    record_property("cdsOrig_size",len(fig.cdsOrig.column_names))
    assert (df0.keys() == df.keys()).all()

# output_file("test_BokehDrawArrayWidget.html")
# fig=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", parameterArray=parameterArray, nPointRender="nPoints", meta={"A.Description":"Test description for overriding metadata"})
