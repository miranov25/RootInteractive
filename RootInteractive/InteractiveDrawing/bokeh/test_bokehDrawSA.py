from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.Tools.aliTreePlayer import *
from bokeh.io import curdoc
import os
import sys
import pytest
from pandas import CategoricalDtype

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
df.eval("Bool=A>0.5", inplace=True)
df.eval("BoolB=B>0.5", inplace=True)
df["AA"]=((df.A*10).round(0)).astype(CategoricalDtype(ordered=True))
df["CC"]=((df.C*5).round(0)).astype(int)
df["DD"]=((df.D*4).round(0)).astype(int)
df["DDC"]=((df.D*4).round(0)).astype(int).map(mapDDC)
df["EE"]=(df.E*4).round(0)
df['errY']=df.A*0.02+0.02;
df.head(10)
df.meta.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)", 'D.AxisTitle': "D (a.u.)", 'Bool.AxisTitle': "A>half", 'E.AxisTitle': "Category"}



figureArray = [
#   ['A'], ['C-A'], {"color": "red", "size": 7, "colorZvar":"C", "filter": "A<0.5"}],
    [['A'], ['A*A-C*C'], {"color": "red", "size": 2, "colorZvar": "A", "varZ": "C", "errY": "errY", "errX":"0.01"}],
    [['A'], ['C+A', 'C-A', 'A/A']],
    [['B'], ['C+B', 'C-B'], { "size": 7, "colorZvar": "C", "errY": "errY", "rescaleColorMapper": True }],
    [['D'], ['(A+B+C)*D'], {"size": 10, "errY": "errY"} ],
#    [['D'], ['D*10'], {"size": 10, "errY": "errY","markers":markerFactor, "color":colorFactor,"legend_field":"DDC"}],
    #marker color works only once - should be constructed in wrapper
    [['D'], ['D*10'], {"size": 10, "errY": "errY"}],
]
#widgets="slider.A(0,1,0.05,0,1), slider.B(0,1,0.05,0,1), slider.C(0,1,0.01,0.1,1), slider.D(0,1,0.01,0,1), checkbox.Bool(1), multiselect.E(0,1,2,3,4)"
widgets="slider.A(0,1,0.05,0,1), slider.B(0,1,0.05,0,1), slider.C(0,1,0.01,0.1,1), slider.D(0,1,0.01,0,1), checkbox.Bool(1)"
figureLayout: str = '((0,1,2, plot_height=300),(3, x_visible=1),commonX=1,plot_height=300,plot_width=1200)'
tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]

widgetParams=[
    ['range', ['A']],
    ['range', ['B', 0, 1, 0.1, 0, 1]],
    ['range', ['C'], {'type': 'minmax'}],
    ['range', ['D'], {'type': 'sigma', 'bins': 10, 'sigma': 3}],
    ['range', ['E'], {'type': 'sigmaMed', 'bins': 10, 'sigma': 3}],
    ['slider', ['AA'], {'bins': 10}],
    ['multiSelect', ["DDC"]],
    ['select',["CC", 0, 1, 2, 3]],
    ['select',["Bool"]],
    ['multiSelect',["BoolB"]],
    #['slider','F', ['@min()','@max()','@med','@min()','@median()+3*#tlm()']], # to be implmneted
]
widgetLayoutDesc=[[0, 1, 2], [3, 4, 5], [6, 7],[8,9], {'sizing_mode': 'scale_width'}]

figureLayoutDesc=[
    [0, 1, 2, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
    [3, 4, {'plot_height': 100, 'x_visible': 1, 'y_visible': 2}],
    {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
]

def testOldInterface():
    output_file("test_bokehDrawSAOldInterface.html")
    fig=bokehDrawSA(df, "A>0", "A", "A:B:C:D", "C", widgets, 0, tooltips=tooltips, layout=figureLayout)
    #fig=bokehDrawSA(tree, "A>0", "A", "A:B:C:D", "C",widgets,0,tooltips=tooltips, layout=figureLayout)

def testBokehDrawArraySA():
    output_file("test_bokehDrawSAArray.html")
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray, widgets, tooltips=tooltips, layout=figureLayout)

def testBokehDrawArrayWidget():
    output_file("test_BokehDrawArrayWidget.html")
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips,widgetLayout=widgetLayoutDesc,sizing_mode="scale_width")

def testBokehDrawArrayWidgetNoScale():
    output_file("test_BokehDrawArrayWidgetNoScale.html")
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips,widgetLayout=widgetLayoutDesc,sizing_mode=None)


def testBokehDrawArrayDownsample():
    output_file("test_BokehDrawArrayDownsample.html")
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, widgetLayout=widgetLayoutDesc, nPointRender=200)

def testBokehDrawArraySA_tree():
    if "ROOT" not in sys.modules:
        pytest.skip("no ROOT module")
    output_file("test_bokehDrawSAArray_fromTTree.html")
    fig=bokehDrawSA.fromArray(tree, "A>0", figureArray, widgets, tooltips=tooltips, layout=figureLayout)


#testOldInterface()           # axis not working since new Bokeh - will be depecated soon
#testBokehDrawArraySA()
#testBokehDrawArraySA_tree()
#testBokehDrawArrayWidget()
#testBokehDrawArrayWidgetNoScale()
#testBokehDrawArrayDownsample()