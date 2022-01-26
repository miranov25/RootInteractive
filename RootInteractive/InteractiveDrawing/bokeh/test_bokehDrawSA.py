from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.Tools.aliTreePlayer import *
from bokeh.io import curdoc
import os
import sys
import pytest
#import pickle
from pandas import CategoricalDtype
import typing as ty

if "ROOT" in sys.modules:
    from ROOT import TFile, gSystem

output_file("test_bokehDrawSA.html")
# import logging

if "ROOT" in sys.modules:
    TFile.SetCacheFileDir("../../data/")
    tree, treeList, fileList = LoadTrees("echo http://rootinteractive.web.cern.ch/RootInteractive/data/tutorial/bokehDraw/treeABCD.root", ".*", ".*ABCDE.*", ".*", 0)
    tree.SetAlias("errY","A*0.02+0.02")
    AddMetadata(tree, "A.AxisTitle", "A (cm)")
    AddMetadata(tree, "B.AxisTitle", "B (cm/s)")
    AddMetadata(tree, "C.AxisTitle", "B (s)")
    AddMetadata(tree, "D.AxisTitle", "D (a.u.)")

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
df.head(10)
df.meta.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)", 'D.AxisTitle': "D (a.u.)", 'Bool.AxisTitle': "A>half", 'E.AxisTitle': "Category"}

parameterArray = [
    {"name": "colorZ", "value":"EE", "options":["A", "B", "EE"]},
    {"name": "X", "value":"A", "options":["A", "B", "D"]},
    {"name": "size", "value":7, "range":[0, 30]},
    {"name": "legendFontSize", "value":"13px", "options":["9px", "11px", "13px", "15px"]},
]

figureArray = [
#   ['A'], ['C-A'], {"color": "red", "size": 7, "colorZvar":"C", "filter": "A<0.5"}],
    [['A'], ['A*A-C*C'], {"color": "red", "size": 2, "colorZvar": "A", "varZ": "C", "errY": "errY", "errX":"0.01"}],
    [['X'], ['C+A', 'C-A', 'A/A']],
    [['B'], ['C+B', 'C-B'], { "colorZvar": "colorZ", "errY": "errY", "rescaleColorMapper": True}],
    [['D'], ['(A+B+C)*DD'], {"colorZvar": "colorZ", "size": 10, "errY": "errY"} ],
#    [['D'], ['D*10'], {"size": 10, "errY": "errY","markers":markerFactor, "color":colorFactor,"legend_field":"DDC"}],
    #marker color works only once - should be constructed in wrapper
    [['D'], ['D*10'], {"size": 10, "errY": "errY"}],
    {"size":"size", "legend_options": {"label_text_font_size": "legendFontSize"}}
]

widgets="slider.A(0,1,0.05,0,1), slider.B(0,1,0.05,0,1), slider.C(0,1,0.01,0.1,1), slider.D(0,1,0.01,0,1), checkbox.Bool(1)"
tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)"), ("ErrY", "@errY")]

widgetParams=[
    ['range', ['A']],
    ['range', ['B', 0, 1, 0.1, 0, 1]],

    ['range', ['C'], {'type': 'minmax'}],
    ['range', ['D'], {'type': 'sigma', 'bins': 10, 'sigma': 3}],
    ['range', ['E'], {'type': 'sigmaMed', 'bins': 10, 'sigma': 3}],
    ['slider', ['AA'], {'bins': 10}],
    ['multiSelect', ["DDC"]],
    ['select',["CC", 0, 1, 2, 3], {"default": 1}],
    ['multiSelect',["BoolB"]],
    #['slider','F', ['@min()','@max()','@med','@min()','@median()+3*#tlm()']], # to be implmneted
    ['select',["colorZ"]],
    ['select',["X"]],
    ['slider',["size"]],
    ['select',["legendFontSize"]],
]
widgetLayoutDesc={
    "Selection": [[0, 1, 2], [3, 4], [5, 6],[7,8], {'sizing_mode': 'scale_width'}],
    "Graphics": [[9, 10, 11, 12], {'sizing_mode': 'scale_width'}]
    }

figureLayoutDesc={
    "A": [
        [0, 1, 2, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
        ],
    "B": [
        [3, 4, {'commonX': 1, 'y_visible': 3, 'x_visible':1, 'plot_height': 100}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
        ]
}

def test_record(record_property: ty.Callable[[str, ty.Any], None]):
    record_property("value1", "value1")
    record_property("value2", "value2")

def testBokehDrawArrayWidget(record_property: ty.Callable[[str, ty.Any], None]):
    output_file("test_BokehDrawArrayWidget.html")
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", parameterArray=parameterArray)
    record_property("html_size",os.stat("test_BokehDrawArrayWidget.html").st_size)
    record_property("cdsOrig_size",len(fig.cdsOrig.column_names))


def testBokehDrawArrayWidgetNoScale(record_property: ty.Callable[[str, ty.Any], None]):
    output_file("test_BokehDrawArrayWidgetNoScale.html")
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips,widgetLayout=widgetLayoutDesc,sizing_mode=None, parameterArray=parameterArray)
    record_property("html_size",os.stat("test_BokehDrawArrayWidgetNoScale.html").st_size)
    record_property("cdsOrig_size",len(fig.cdsOrig.column_names))

def testBokehDrawArrayDownsample(record_property: ty.Callable[[str, ty.Any], None], data_regression):
    output_file("test_BokehDrawArrayDownsample.html")
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, widgetLayout=widgetLayoutDesc, parameterArray=parameterArray)
    record_property("html_size",os.stat("test_BokehDrawArrayWidget.html").st_size)
    record_property("cdsOrig_size",len(fig.cdsOrig.column_names))
    # number of cdsSel columns is no longer relevant
    # record_property("cdsSel_size",len(fig.cdsSel.selectedColumns))
    # data_regression.check(list(fig.cdsSel.selectedColumns),"test_bokehDrawSA.testBokehDrawArrayDownsample")


def testBokehDrawArrayQuery(record_property: ty.Callable[[str, ty.Any], None]):
    output_file("test_BokehDrawArrayQuery.html")
    df0 = df.copy()
    fig=bokehDrawSA.fromArray(df0, "BoolC == True", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, widgetLayout=widgetLayoutDesc, nPointRender=200, parameterArray=parameterArray)
    record_property("html_size",os.stat("test_BokehDrawArrayWidget.html").st_size)
    record_property("cdsOrig_size",len(fig.cdsOrig.column_names))
    record_property("cdsSel_size",len(fig.cdsSel.selectedColumns))

    assert (df0.keys() == df.keys()).all()

def testBokehDrawArraySA_tree():
    if "ROOT" not in sys.modules:
        pytest.skip("no ROOT module")
    output_file("test_bokehDrawSAArray_fromTTree.html")
    fig=bokehDrawSA.fromArray(tree, "A>0", figureArray, widgets, tooltips=tooltips, layout=figureLayout)

