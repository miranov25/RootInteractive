from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
from RootInteractive.Tools.compressArray import arrayCompressionRelative16
import numpy as np
import pandas as pd
from bokeh.plotting import output_file

def makePanda(step,sigma):
    noise = 0.1
    aa = np.arange(-1, 1, step)
    bb = np.arange(-1, 1, step)
    cc = np.arange(-2, 2, step)
    a, b, c = np.meshgrid(aa, bb, cc, sparse=False)
    w = np.exp(-(np.abs(a - b) ** 2) / (2 * sigma ** 2))
    #w[0] = np.NaN
    wA = np.exp(-(a ** 2) / (2 * sigma ** 2))
    df0 = pd.DataFrame(data={'A': a.flatten(), 'B': b.flatten(), 'C': c.flatten(), 'W': w.flatten(),'Wa': w.flatten()})
    return df0,a,b,c,w

figureArray = [
    [['A'], ['histoA']],
    [['A'], ['histoA1']],
    [['B'], ['histoB']],
    [['D'], ['histoD']],
    [['histoB.bin_center'], ['histoB.bin_count']],
    [['histoD.bin_center'], ['histoD.bin_count']],
]
histogramArray = [
    {"name": "histoA", "variables": ["A"], "weights": "Wa", "nbins": 50, "range": [-1, 1]},
    {"name": "histoA1", "variables": ["A"], "weights": "W", "nbins": 50, "range": [-1, 1]},
    {"name": "histoB", "variables": ["B"], "weights": "W", "nbins": 50, "range": [-1, 1]},
    {"name": "histoD", "variables": ["D"], "weights": "W", "nbins": 50, "range": [-1, 1]},
]
widgetParams = [
    ['range', ['A', -1., 1., 0.1, -1., 1.]],
    ['range', ['B']],
    ['range', ['C']],
    ['multiSelect', ['D']],
    ['range', ['index'], {"index": True}],
]
widgetLayoutDesc = [[0, 1], [2, 3, 4], {'sizing_mode': 'scale_width'}]

figureLayoutDesc = [
    [0, 1, {'y_visible': 1, 'x_visible': 1, 'plot_height': 150}],
    [2, 3, {'y_visible': 1, 'x_visible': 1, 'plot_height': 150}],
    [4, 5, {'y_visible': 1, 'x_visible': 1, 'plot_height': 150}],
    {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible': 2}
]
tooltips = [("VarA", "(@A)"), ("VarB", "(@B)")]
df,a,b,c,d=makePanda(0.02,0.1)
df['D']=df["A"]-df["B"]

def test_clientHistogramWeight():
    output_file("test_bokehClientHistogramWeight.html")
    xxx = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips,
                                widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histogramArray)

def test_clientHistogramWeightCompressed():
    output_file("test_bokehClientHistogramWeight_Compressed.html")
    xxx = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips,
                                widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histogramArray,
                                arrayCompression=arrayCompressionRelative16)
