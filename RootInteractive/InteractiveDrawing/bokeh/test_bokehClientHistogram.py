from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.Tools.aliTreePlayer import *
from bokeh.io import curdoc
import os
import sys
import pytest
from pandas import CategoricalDtype

output_file("test_bokehClientHistogram.html")
# import logging

df = pd.DataFrame(np.random.random_sample(size=(200000, 4)), columns=list('ABCD'))
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
df['errY']=df.A*0.02+0.02;
df.head(10)
df.meta.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)",
                    'D.AxisTitle': "D (a.u.)", 'Bool.AxisTitle': "A>half"}

#widgets="slider.A(0,1,0.05,0,1), slider.B(0,1,0.05,0,1), slider.C(0,1,0.01,0.1,1), slider.D(0,1,0.01,0,1), checkbox.Bool(1), multiselect.E(0,1,2,3,4)"
widgets="slider.A(0,1,0.05,0,1), slider.B(0,1,0.05,0,1), slider.C(0,1,0.01,0.1,1), slider.D(0,1,0.01,0,1), checkbox.Bool(1)"
figureLayout: str = '((0,1,2, plot_height=300),commonX=1,plot_height=300,plot_width=1200)'
tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]

widgetParams=[
    ['range', ['A']],
    ['range', ['B', 0, 1, 0.1, 0, 1]],
    ['range', ['C'], {'type': 'minmax'}],
    ['range', ['D'], {'type': 'sigma', 'bins': 10, 'sigma': 3}],
    ['multiSelect', ["DDC"]],
    ['select',["CC", 0, 1, 2, 3]],
    ['multiSelect',["BoolB"]],
]
widgetLayoutDesc=[[0, 1, 2], [3, 4], [5, 6], {'sizing_mode': 'scale_width'}]

figureLayoutDesc=[
    [0, 1, 2, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
    {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
]

histoArray = [
    {"name": "histoA", "variables": ["A"],"nbins":20},
    {"name": "histoB", "variables": ["B"],"nbins":20},
    {"name": "histoTransform", "variables": ["(A+B)/2"],"nbins":20},
    {"name": "histoAB", "variables": ["A", "B"], "nbins": [20, 20], "weights": "D"},
]

def testBokehClientHistogram():
    output_file("test_BokehClientHistogram.html")
    figureArray = [
        #   ['A'], ['C-A'], {"color": "red", "size": 7, "colorZvar":"C", "filter": "A<0.5"}],
        [['A'], ['histoA', '(A*A-C*C)*100'], {"size": 2, "colorZvar": "A", "errY": "errY", "errX": "0.01"}],
        [['(A+B)/2'], ['histoTransform', '(C+A)*200', '(C-A)*200']],
        [['B'], ['histoB', '(C+B)*10', '(C-B)*10'], {"size": 7, "colorZvar": "C", "errY": "errY",
                                                    "rescaleColorMapper": True}]
    ]
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips,
                              widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointsRender=3000, histogramArray=histoArray)

def testBokehClientHistogramOnlyHisto():
    output_file("test_BokehClientHistogramOnlyHisto.html")
    figureArray = [
        [['A'], ['histoA']],
        [['(A+B)/2'], ['histoTransform']],
        [['A'], ['histoAB']],
        [['B'], ['histoB']]
    ]
    figureLayoutDesc=[
        [0, 1,  {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [2, 3, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
    ]
    xxx = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips,
                                widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray)

#testBokehClientHistogram()
#testBokehClientHistogramOnlyHisto()