from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.Tools.aliTreePlayer import *
from bokeh.io import curdoc
from pandas import CategoricalDtype

output_file("test_bokehClientHistogram.html")
# import logging

df = pd.DataFrame(np.random.random_sample(size=(20000, 4)), columns=list('ABCD'))
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
df['errY']=df.A*0.02+0.02
df.head(10)
df.meta.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)",
                    'D.AxisTitle': "D (a.u.)", 'Bool.AxisTitle': "A>half"}

#widgets="slider.A(0,1,0.05,0,1), slider.B(0,1,0.05,0,1), slider.C(0,1,0.01,0.1,1), slider.D(0,1,0.01,0,1), checkbox.Bool(1), multiselect.E(0,1,2,3,4)"
widgets="slider.A(0,1,0.05,0,1), slider.B(0,1,0.05,0,1), slider.C(0,1,0.01,0.1,1), slider.D(0,1,0.01,0,1), checkbox.Bool(1)"
figureLayout: str = '((0,1,2, plot_height=300),commonX=1,plot_height=300,plot_width=1200)'
tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]

parameterArray=[
    {'name':"size", "value":7, "range": [0, 20]}
]

widgetParams=[
    ['range', ['A']],
    ['range', ['B', 0, 1, 0.1, 0, 1]],
    ['range', ['C'], {'type': 'minmax'}],
    ['range', ['D'], {'type': 'sigma', 'bins': 10, 'sigma': 3}],
    ['multiSelect', ["DDC"]],
    ['slider',["size"]],
  #  ['select',["CC", 0, 1, 2, 3]],
  #  ['multiSelect',["BoolB"]],
]
widgetLayoutDesc=[[0, 1, 2], [3, 4], [5], {'sizing_mode': 'scale_width'}]

figureLayoutDesc=[
    [0, 1, 2, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
    {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
]

histoArray = [
    {"name": "histoA", "variables": ["A"], "nbins":20, "quantiles": [.05, .5, .95], "sum_range": [[.25, .75], [.4, .6]]},
    {"name": "histoB", "variables": ["B"], "nbins":20, "range": [0, 1]},
    {"name": "histoABC", "variables": ["A", "B", "C"], "nbins":[10, 5, 10], "quantiles": [.5], "sumRange": [[.25, .75]], "axis": [0, 2]},
    {"name": "histoAB", "variables": ["A", "(A+B)/2"], "nbins": [20, 20], "weights": "D", "quantiles": [.25, .5, .75], "axis": [0, 1]},
]

def testBokehClientHistogram():
    output_file("test_BokehClientHistogram.html")
    figureArray = [
        #   ['A'], ['C-A'], {"color": "red", "size": 7, "colorZvar":"C", "filter": "A<0.5"}],
        [['A'], ['histoA', '(A*A-C*C)*100'], {"size": 2, "colorZvar": "A", "errY": "errY", "errX": "0.01", "size":"size"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.mean'], {"colorZvar": "histoABC_0.bin_center_2",
                                                            "rescaleColorMapper": True, "size":"size"}],
        [['B'], ['histoB', '(C+B)*10', '(C-B)*10'], {"size": 7, "colorZvar": "C", "errY": "errY",
                                                    "rescaleColorMapper": True, "size":"size"}]
    ]
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                              widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300, histogramArray=histoArray)

def testBokehClientHistogramOnlyHisto():
    output_file("test_BokehClientHistogramOnlyHisto.html")
    figureArray = [
        [['A'], ['histoA']],
        [['A'], ['histoAB'], {"visualization_type": "colZ", "show_histogram_error": True}],
        [['A'], ['histoAB'], {"yAxisTitle": "(A+B)/2"}],
        [['B'], ['histoB'], {"flip_histogram_axes": True}],
        ["tableHisto", {"rowwise": False}]
    ]
    figureLayoutDesc=[
        [0, 1,  {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [2, 3, {'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [4, {'plot_height': 40}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2, "size": 5}
    ]
    xxx = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                                widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray)

def testBokehClientHistogramProfileA():
    output_file("test_BokehClientHistogramProfileA.html")
    figureArray = [
        [['histoAB_1.bin_center_0'], ['histoAB_1.quantile_0', 'histoAB_1.quantile_1', 'histoAB_1.quantile_2'], {"size":"size"}],
        [['histoAB_1.bin_center_0'], ['histoAB_1.quantile_1', 'histoAB_1.mean'], {"size":"size"}],
        [['A'], ['histoAB'], {"yAxisTitle": "(A+B)/2", "size":"size"}],
        [['histoAB_1.bin_center_0'], ['histoAB_1.std'], {"size":"size"}],
        ["tableHisto", {"rowwise": False}]
    ]
    figureLayoutDesc=[
        [0, 1,  {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [2, 3, {'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [4, {'plot_height': 40}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2, "size": 5}
    ]
    xxx = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                                widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray)

def testBokehClientHistogramProfileB():
    output_file("test_BokehClientHistogramProfileB.html")
    figureArray = [
        [['histoAB_0.bin_center_1'], ['histoAB_0.quantile_0', 'histoAB_0.quantile_1', 'histoAB_0.quantile_2'], {"size":"size"}],
        [['histoAB_0.bin_center_1'], ['histoAB_0.quantile_1', 'histoAB_0.mean'], {"size":"size"}],
        [['A'], ['histoAB'], {"yAxisTitle": "(A+B)/2"}],
        [['histoAB_0.bin_center_1'], ['histoAB_0.std'], {"size":"size"}],
        ["tableHisto", {"rowwise": False}]
    ]
    figureLayoutDesc=[
        [0, 1,  {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [2, 3, {'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [4, {'plot_height': 40}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2, "size": 5}
    ]
    xxx = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                                widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray)


def testBokehClientHistogramRowwiseTable():
    output_file("test_BokehClientHistogramRowwiseTable.html")
    figureArray = [
        [['A'], ['histoA']],
        [['A'], ['histoAB'], {"visualization_type": "colZ", "show_histogram_error": True}],
        [['A'], ['histoAB'], {"yAxisTitle": "(A+B)/2"}],
        [['B'], ['histoB'], {"flip_histogram_axes": True}],
        ["tableHisto", {"rowwise": True}]
    ]
    figureLayoutDesc=[
        [0, 1,  {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [2, 3, {'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [4, {'plot_height': 40}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2, "size": 5}
    ]
    xxx = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                                widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray)

def testBokehClientHistogram3d():
    output_file("test_BokehClientHistogram.html")
    histoArray = [
        {"name": "histoABC", "variables": ["(A+C)/2", "B", "C"], "nbins": [8, 10, 12], "weights": "D", "axis": [0], "sum_range": [[.25, .75]]},
    ]
    figureArray = [
        [['histoABC_0.bin_center_1'], ['histoABC_0.mean'], {"colorZvar": "histoABC_0.bin_center_2", "size": "size"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.sum_0'], {"colorZvar": "histoABC_0.bin_center_2", "size": "size"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.sum_normed_0'], {"colorZvar": "histoABC_0.bin_center_2", "size": "size"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.std'], {"colorZvar": "histoABC_0.bin_center_2", "size": "size"}]
    ]
    figureLayoutDesc=[
        [0, 1, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
        [2, 3, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
    ]
    
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                              widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=3000, histogramArray=histoArray)

def testBokehClientHistogram3d_colormap():
    output_file("test_BokehClientHistogram_colormap.html")
    histoArray = [
        {"name": "histoABC", "variables": ["(A+C)/2", "B", "C"], "nbins": [8, 10, 12], "weights": "D", "axis": [0], "sum_range": [[.25, .75]],
        "range": [[0,1],[0,1],[0,1]]},
    ]
    figureArray = [
        [['histoABC_0.bin_center_1'], ['histoABC_0.mean'], {"colorZvar": "histoABC_0.bin_center_2" }],
        [['histoABC_0.bin_center_1'], ['histoABC_0.sum_0'], {"colorZvar": "histoABC_0.bin_center_2" }],
        [['histoABC_0.bin_center_1'], ['histoABC_0.sum_normed_0'], {"colorZvar": "histoABC_0.bin_center_2" }],
        [['histoABC_0.bin_center_1'], ['histoABC_0.std'], {"colorZvar": "histoABC_0.bin_center_2" }],
        {"size": "size", "rescaleColorMapper": True}
    ]
    figureLayoutDesc=[
        [0, 1, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
        [2, 3, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
    ]
    
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                              widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=3000, histogramArray=histoArray)

def testBokehClientHistogram3d_colormap_noscale():
    output_file("test_BokehClientHistogram_colormap_noscale.html")
    histoArray = [
        {"name": "histoABC", "variables": ["(A+C)/2", "B", "C"], "nbins": [8, 10, 12], "weights": "D", "axis": [0], "sum_range": [[.25, .75]],
        "range": [[0,1],[0,1],[0,1]]}
    ]
    figureArray = [
        [['histoABC_0.bin_center_1'], ['histoABC_0.mean'], {"colorZvar": "histoABC_0.bin_center_2"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.sum_0'], {"colorZvar": "histoABC_0.bin_center_2"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.sum_normed_0'], {"colorZvar": "histoABC_0.bin_center_2"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.std'], {"colorZvar": "histoABC_0.bin_center_2"}],
        {"size": "size"}
    ]
    figureLayoutDesc=[
        [0, 1, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
        [2, 3, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
    ]
    
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                              widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=3000, histogramArray=histoArray)

def testJoin():
    output_file("test_BokehClientHistogramJoin.html")
    aliasArray = [
        {
            "name": "unbinned_mean_A",
            "variables": ["sum_A", "bin_count"],
            "func": "return sum_A / bin_count",
            "context": "histoB"
        },
        {
            "name": "delta_mean",
            "variables": ["mean", "unbinned_mean_A"],
            "func": "return mean - unbinned_mean_A",
            "context": "histoB_join_histoAB_0"
        }
    ]
    histoArray = [
        {"name": "histoAB", "variables": ["A", "B"], "nbins": [10, 10], "axis": [0]},
        {"name": "histoB", "variables": ["B"], "nbins": 10,
            "histograms": {
                "sum_A": {"weights": "A"}
            }
        }
    ]
    sourceArray = [
        {"name": "histoB_join_histoAB_0", "left": "histoAB_0", "right":"histoB", "left_on":["bin_center_1"], "right_on": ["bin_center"]}
    ]
    figureArray = [
        [['histoAB_0.bin_center_1', 'histoB.bin_center'], ['histoAB_0.mean', 'unbinned_mean_A']],
        [['histoAB_0.bin_center_1', 'histoB.bin_center'], ['histoAB_0.entries', 'histoB.bin_count']],
        [['histoB_join_histoAB_0.bin_center_1'], ['histoB_join_histoAB_0.delta_mean']],
        [['histoAB_0.bin_center_1'], ['histoAB_0.std']],
        {"size": "size"}
    ]
    figureLayoutDesc=[
        [0, 1, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [2, 3, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
    ]
    
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                              widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=3000, histogramArray=histoArray, sourceArray=sourceArray, aliasArray=aliasArray)

testBokehClientHistogramProfileA()