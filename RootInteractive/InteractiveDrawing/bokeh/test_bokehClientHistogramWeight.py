from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.Tools.aliTreePlayer import *
from bokeh.io import curdoc
import pytest
import matplotlib.pyplot as plt
def makePanda(step,sigma):
    noise = 0.1
    aa = np.arange(-1, 1, step)
    bb = np.arange(-1, 1, step)
    cc = np.arange(-2, 2, step)
    a, b, c = np.meshgrid(aa, bb, cc, sparse=False)
    w = np.exp(-(np.abs(a - b) ** 2) / (2 * sigma ** 2))
    wA = np.exp(-(a ** 2) / (2 * sigma ** 2))
    df0 = pd.DataFrame(data={'A': a.flatten(), 'B': b.flatten(), 'C': c.flatten(), 'W': w.flatten(),'Wa': w.flatten()})
    return df0,a,b,c,w


figureArray = [
    [['A'], ['histo'], { "weights": "Wa","nbins": 50, "range": [-1, 1]}],
    [['A'], ['histo'], { "weights": "W","nbins": 50, "range": [-1, 1]}],
    [['B'], ['histo'], { "weights": "W","nbins": 50, "range": [-1, 1]}],
    [['D'], ['histo'], { "weights": "W","nbins": 50, "range": [-1, 1]}],
]
widgetParams = [
    ['range', ['A',-1., 1., 0.1, -1., 1.]],
    ['range', ['B']],
    ['range', ['C']],
    ['range', ['D']],
]
widgetLayoutDesc = [[0, 1], [2,3], {'sizing_mode': 'scale_width'}]

figureLayoutDesc = [
    [0, 1, {'y_visible': 1, 'x_visible': 1, 'plot_height': 150}],
    [2, 3, {'y_visible': 1, 'x_visible': 1, 'plot_height': 150}],
    {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible': 2}
]
tooltips = [("VarA", "(@A)"), ("VarB", "(@B)")]
df,a,b,c,d=makePanda(0.02,0.1)
df['D']=df["A"]-df["B"]
output_file("test_bokehClientHistogramWeight.html")
xxx = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips,
                            widgetLayout=widgetLayoutDesc, sizing_mode="scale_width")
