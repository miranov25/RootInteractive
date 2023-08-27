from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.Tools.aliTreePlayer import *
from RootInteractive.InteractiveDrawing.bokeh.bokehTools import mergeFigureArrays
from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate import getDefaultVarsDiff, getDefaultVarsRatio, getDefaultVarsNormAll
from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveParameters import figureParameters
from RootInteractive.Tools.compressArray import arrayCompressionRelative16
from pandas import CategoricalDtype

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
    {'name':"size", "value":7, "range": [0, 20]},
    {'name':"histoRangeA", "value": [0, 1], "range": [0, 1]},
    {'name':"nBinsB", "value": 20, "options":[5, 10, 20, 40]},
    {'name':"transformY", "value":None, "options":[None, "sqrt", "lambda x: log(x+eps)"]},
    {'name':"eps", "value":1, "range": [.1, 5]},
    {'name':"histo1Ds", "value":["histoA"], "options":["histoA", "histoB"]}
]

widgetParams=[
    ['range', ['A']],
    ['range', ['B', 0, 1, 0.1, 0, 1]],
    ['range', ['C'], {'type': 'minmax'}],
    ['range', ['D'], {'type': 'sigma', 'bins': 10, 'sigma': 3}],
    ['multiSelect', ["DDC"]],
    ['slider',["size"]],
    ['range', ['histoRangeA']],
    ['select', ['nBinsB']],
    ['select', ['transformY']],
    ['slider', ['eps'], {"name": "eps"}],
    ['multiSelect', ['histo1Ds'], {"name": "histo1Ds"}],
  #  ['select',["CC", 0, 1, 2, 3]],
  #  ['multiSelect',["BoolB"]],
]
widgetLayoutDesc=[[0, 1, 2], [3, 4], [5, 6], [7, 8, "eps", "histo1Ds"], {'sizing_mode': 'scale_width'}]

figureLayoutDesc=[
    [0, 1, 2, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
    {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
]

histoArray = [
    {"name": "histoA", "variables": ["A"], "nbins":20, "range": "histoRangeA", "quantiles": [.05, .5, .95], "sum_range": [[.25, .75], [.4, .6]],
         "histograms": {
            "cumulative": {"cumulative":True},
            "cdf": {"cumulative":True, "density":True}
         }},
    {"name": "histoB", "variables": ["B"], "nbins":"nBinsB", "range": [0, 1]},
    {"name": "histoABC", "variables": ["A", "B", "C"], "nbins":[10, "nBinsB", 10], "range": ["histoRangeA", None, None], "quantiles": [.5], "sumRange": [[.25, .75]], "axis": [0, 2]},
    {"name": "histoAB", "variables": ["A", "(A+B)/2"], "nbins": [20, "nBinsB"], "range": ["histoRangeA", None], "unbinned_projections":True, "weights": "D", "quantiles": [.25, .5, .75], "axis": [0, 1]},
    {"name": "histo1Ds", "sources": "histo1Ds"}
]

def testBokehClientHistogram():
    output_file("test_BokehClientHistogram.html")
    figureArray = [
        #   ['A'], ['C-A'], {"color": "red", "size": 7, "colorZvar":"C", "filter": "A<0.5"}],
        [['A'], ['histoA', '(A*A-C*C)*100'], {"size": 2, "colorZvar": "A", "errY": "errY", "errX": "0.01", "size":"size"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.mean'], {"colorZvar": "histoABC_0.bin_center_2",
                                                            "rescaleColorMapper": True, "size":"size"}],
        [['B'], ['histoB', '(C+B)*10', '(C-B)*10'], {"size": 7, "colorZvar": "C", "errY": "errY",
                                                    "rescaleColorMapper": True, "size":"size"}],
        {"y_transform":"transformY"}
    ]
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                              widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300, histogramArray=histoArray)

def testBokehClientHistogramOnlyHisto():
    output_file("test_BokehClientHistogramOnlyHisto.html")
    figureArray = [
        [['bin_center'], ['bin_count'], {"source": "histo1Ds", "errY": [("sqrt(bin_count)", "sqrt(bin_count+1)")], "yAxisTitle":"{transformY}(N)"}],
        [['bin_center_0'], ['bin_count'], {"colorZvar":"bin_center_1", "errY": [("sqrt(bin_count)", "sqrt(bin_count+1)")], "source":"histoAB"}],
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "log(bin_count+1)", "source":"histoAB"}],
        [['bin_count'], ['bin_center'], {"source": "histoB"}],
        ["tableHisto", {"rowwise": False, "include": "histoA$|histoB$"}],
        [['bin_center'], ['cumulative'], {"source": "histoA"}],
        [['bin_center_0'], ['cumulative'], {"source": "histoAB_1"}],
        [['bin_center'], ['cdf'], {"source": "histoA"}],
        [['bin_center_0'], ['cdf'], {"source": "histoAB_1"}],
        {"y_transform":"transformY"}
    ]
    figureLayoutDesc={
            "Histograms":[
                [0, 1,  {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
                [2, 3, {'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
                [4, {'plot_height': 40}],
                {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2, "size": 5}
            ],
            "Cumulative":[
                [5,6, {'commonX': 5, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
                [7,8, {'commonX': 5, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}]
            ]

        }

    xxx = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                                widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray)

def testBokehClientHistogramProfileA():
    output_file("test_BokehClientHistogramProfileA.html")
    figureArray = [
        [['bin_center_0'], ['quantile_0', 'quantile_1', 'quantile_2'], {"size":"size", "source":"histoAB_1"}],
        [['bin_center_0'], ['quantile_1', 'mean'], {"size":"size", "source":"histoAB_1"}],
        [['A'], ['histoAB'], {"yAxisTitle": "(A+B)/2", "size":"size"}],
        [['bin_center_0'], ['std'], {"size":"size", "source":"histoAB_1"}],
        ["tableHisto", {"rowwise": False}],
        {"y_transform":"transformY"}
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
        [['bin_center_1'], ['quantile_0', 'quantile_1', 'quantile_2'], {"size":"size", "source":"histoAB_0"}],
        [['bin_center_1'], ['quantile_1', 'mean'], {"size":"size", "source":"histoAB_0"}],
        [['A'], ['histoAB'], {"yAxisTitle": "(A+B)/2"}],
        [['bin_center_1'], ['std'], {"size":"size", "source":"histoAB_0"}],
        ["tableHisto", {"rowwise": False}],
        {"y_transform":"transformY"}
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
        ["tableHisto", {"rowwise": True, "exclude": r".*_.*"}],
        {"y_transform":"transformY"}
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
    output_file("test_BokehClientHistogram3d.html")
    histoArray = [
        {"name": "histoABC", "variables": ["(A+C)/2", "B", "C"], "nbins": [8, "nBinsB", 12], "weights": "D", "axis": [0], "sum_range": [[.25, .75]]},
    ]
    figureArray = [
        [['histoABC_0.bin_center_1'], ['histoABC_0.mean'], {"colorZvar": "histoABC_0.bin_center_2", "size": "size"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.sum_0'], {"colorZvar": "histoABC_0.bin_center_2", "size": "size"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.sum_normed_0'], {"colorZvar": "histoABC_0.bin_center_2", "size": "size"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.std'], {"colorZvar": "histoABC_0.bin_center_2", "size": "size"}],
        {"y_transform":"transformY"}
    ]
    figureLayoutDesc=[
        [0, 1, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [2, 3, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
    ]
    widgetLayoutDesc = [[0, 1, 2], [3, 4], [5], [7], {'sizing_mode': 'scale_width'}]
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                              widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=3000, histogramArray=histoArray)

def testBokehClientHistogram3d_colormap_noscale():
    output_file("test_BokehClientHistogram_colormap_noscale.html")
    histoArray = [
        {"name": "histoABC", "variables": ["(A+C)/2", "B", "C"], "nbins": [8, "nBinsB", 12], "weights": "D", "axis": [0], "sum_range": [[.25, .75]],
        "range": [[0,1],[0,1],[0,1]]}
    ]
    figureArray = [
        [['histoABC_0.bin_center_1'], ['histoABC_0.mean'], {"colorZvar": "histoABC_0.bin_center_2"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.sum_0'], {"colorZvar": "histoABC_0.bin_center_2"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.sum_normed_0'], {"colorZvar": "histoABC_0.bin_center_2"}],
        [['histoABC_0.bin_center_1'], ['histoABC_0.std'], {"colorZvar": "histoABC_0.bin_center_2"}],
        {"size": "size", "y_transform":"transformY"}
    ]
    figureLayoutDesc=[
        [0, 1, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [2, 3, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
    ]
    widgetLayoutDesc = [[0, 1, 2], [3, 4], [5], [7], {'sizing_mode': 'scale_width'}]
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
        {"name": "histoAB", "variables": ["A", "B"], "nbins": [10, "nBinsB"], "range": ["histoRangeA", None], "axis": [0]},
        {"name": "histoB", "variables": ["B"], "nbins": "nBinsB",
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
        {"size": "size", "y_transform":"transformY"}
    ]
    figureLayoutDesc=[
        [0, 1, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [2, 3, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
    ]
    
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                              widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=3000, histogramArray=histoArray, sourceArray=sourceArray, aliasArray=aliasArray)
    
def test_StableQuantile():
    output_file("test_BokehClientHistogramQuantile.html")
    histoArray = [
        {"name": "histoAB", "variables": ["A", "A*A*B"], "nbins": ["nBinsA", "nBinsAAB"], "axis": [1], "quantiles": [.05, .5, .95]},
        {"name": "histoABWeight", "variables": ["A", "A*A*B"], "nbins": ["nBinsA", "nBinsAAB"], "weights":"1+A", "axis": [1], "quantiles": [.05, .5, .95]},
        {"name": "histo3D", "variables": ["A", "B", "A*A*B"], "nbins": ["nBinsA", "nBinsB", "nBinsAAB"], "axis": [2], "quantiles": [.05, .5, .95]},
        {"name": "histo3D_weight", "variables": ["A", "B", "A*A*B"], "nbins": ["nBinsA", "nBinsB", "nBinsAAB"], "weights":"1+A", "axis": [2], "quantiles": [.05, .5, .95]},
        {"name": "projectionA", "axis_idx":1, "source": "histoAB", "unbinned":True, "type":"projection", "quantiles": [.05, .5, .95]},
        {"name": "projectionAWeight", "axis_idx":1, "source": "histoABWeight", "unbinned":True, "type":"projection", "quantiles": [.05, .5, .95], "weights":"1+A"},
        {"name": "projection3D", "axis_idx":2, "source": "histo3D", "unbinned":True, "type":"projection", "quantiles": [.05, .5, .95]},
        {"name": "projection3D_weight", "axis_idx":2, "source": "histo3D_weight", "unbinned":True, "type":"projection", "quantiles": [.05, .5, .95], "weights":"1+A"}
    ]
    aliasArray = [
        (f"{i}_normed", f"{i} / bin_center_0**2", j) for i in ["mean", "std", "quantile_0",  "quantile_1", "quantile_2"] for j in ["histoAB_1", "histoABWeight_1", "projectionA", "projectionAWeight"]
    ]
    aliasArray += [
        {
            "name": "std_true",
            "variables": ["bin_bottom_0", "bin_top_0", "bin_bottom_1", "bin_top_1", "bin_center_0", "bin_center_1"],
            "func": "return Math.sqrt((4*(bin_center_0*bin_center_1*(bin_top_0-bin_bottom_0))**2+((bin_top_1-bin_bottom_1)*bin_center_0**2)**2)/12)",
            "context": j
        } for j in ["histo3D_2", "histo3D_weight_2", "projection3D", "projection3D_weight"]
    ]
    figureArray = [
        [['bin_center_0'], ['mean', 'quantile_0', 'quantile_1', 'quantile_2'], {"source": "histoAB_1"}],
        [['bin_center_0'], ['std'], {"source": "histoAB_1"}],
        [['bin_center_0'], ['mean_normed', 'quantile_0_normed', 'quantile_1_normed', 'quantile_2_normed'], {"source": "histoAB_1"}],
        [['bin_center_0'], ['std_normed'], {"source": "histoAB_1"}],
        [['bin_center_0'], ['mean', 'quantile_0', 'quantile_1', 'quantile_2'], {"source": "projectionA"}],
        [['bin_center_0'], ['std'], {"source": "projectionA"}],
        [['bin_center_0'], ['mean_normed', 'quantile_0_normed', 'quantile_1_normed', 'quantile_2_normed'], {"source": "projectionA"}],
        [['bin_center_0'], ['std_normed'], {"source": "projectionA"}],
        [['bin_center_0'], ['quantile_0', 'quantile_1', 'quantile_2', 'quantile_0', 'quantile_1', 'quantile_2'], {"source": ["histoAB_1", "histoAB_1", "histoAB_1", "projectionA", "projectionA", "projectionA"]}],
        [['bin_center_0'], ['std'], {"source": ["histoAB_1", "projectionA"]}],
        [['bin_center_0'], ['quantile_0_normed', 'quantile_1_normed', 'quantile_2_normed', 'quantile_0_normed', 'quantile_1_normed', 'quantile_2_normed'], {"source": ["histoAB_1", "histoAB_1", "histoAB_1", "projectionA", "projectionA", "projectionA"]}],
        [['bin_center_0'], ['std_normed'], {"source": ["histoAB_1", "projectionA"]}],
        [['bin_center_0'], ['mean', 'quantile_0', 'quantile_1', 'quantile_2'], {"source": "histoABWeight_1"}],
        [['bin_center_0'], ['std'], {"source": "histoABWeight_1"}],
        [['bin_center_0'], ['mean_normed', 'quantile_0_normed', 'quantile_1_normed', 'quantile_2_normed'], {"source": "histoABWeight_1"}],
        [['bin_center_0'], ['std_normed'], {"source": "histoABWeight_1"}],
        [['bin_center_0'], ['mean', 'quantile_0', 'quantile_1', 'quantile_2'], {"source": "projectionAWeight"}],
        [['bin_center_0'], ['std'], {"source": "projectionAWeight"}],
        [['bin_center_0'], ['mean_normed', 'quantile_0_normed', 'quantile_1_normed', 'quantile_2_normed'], {"source": "projectionAWeight"}],
        [['bin_center_0'], ['std_normed'], {"source": "projectionAWeight"}],
        [['bin_center_0'], ['quantile_0', 'quantile_1', 'quantile_2', 'quantile_0', 'quantile_1', 'quantile_2'], {"source": ["histoABWeight_1", "histoABWeight_1", "histoABWeight_1", "projectionAWeight", "projectionAWeight", "projectionAWeight"]}],
        [['bin_center_0'], ['std'], {"source": ["histoABWeight_1", "projectionAWeight"]}],
        [['bin_center_0'], ['quantile_0_normed', 'quantile_1_normed', 'quantile_2_normed', 'quantile_0_normed', 'quantile_1_normed', 'quantile_2_normed'], {"source": ["histoABWeight_1", "histoABWeight_1", "histoABWeight_1", "projectionAWeight", "projectionAWeight", "projectionAWeight"]}],
        [['bin_center_0'], ['std_normed'], {"source": ["histoABWeight_1", "projectionAWeight"]}],
        [['bin_center_0'], ['quantile_1'], {"source": "histo3D_2", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['std'], {"source": "histo3D_2", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['quantile_1/(bin_center_0**2)'], {"source": "histo3D_2", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['std/std_true'], {"source": "histo3D_2", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['quantile_1'], {"source": "projection3D", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['std'], {"source": "projection3D", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['quantile_1/(bin_center_0**2)'], {"source": "projection3D", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['std/std_true'], {"source": "projection3D", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['quantile_1'], {"source": "histo3D_weight_2", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['std'], {"source": "histo3D_weight_2", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['quantile_1/(bin_center_0**2)'], {"source": "histo3D_weight_2", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['std/std_true'], {"source": "histo3D_weight_2", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['quantile_1'], {"source": "projection3D_weight", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['std'], {"source": "projection3D_weight", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['quantile_1/(bin_center_0**2)'], {"source": "projection3D_weight", "colorZvar": "bin_center_1"}],
        [['bin_center_0'], ['std/std_true'], {"source": "projection3D_weight", "colorZvar": "bin_center_1"}],
        {"size": "size"}
    ]
    figureLayoutDesc={
        "Without weights":{
            "binned": [[0,1], [2,3], {'plot_height': 200}],
            "unbinned": [[4,5],[6,7], {'plot_height': 200}],
            "both": [[8,9],[10,11], {'plot_height': 200}],
            "3D binned": [[24,25],[26,27], {'plot_height': 200}],
            "3D unbinned": [[28,29],[30,31], {'plot_height': 200}]
        },
        "With weights":{
            "binned": [[12,13],[14,15], {'plot_height': 200}],
            "unbinned": [[16,17],[18,19], {'plot_height': 200}],
            "both": [[20,21],[22,23], {'plot_height': 200}],
            "3D binned": [[32,33],[34,35], {'plot_height': 200}],
            "3D unbinned": [[36,37],[38,39], {'plot_height': 200}]
        }
    }
    parameterArray=[
        {'name':"size", "value":7, "range": [0, 20]},
        {'name':"nBinsB", "value": 10},
        {'name':"nBinsA", "value": 20},
        {'name':"nBinsAAB", "value": 20},
    ]
    widgetParams=[
        ['range', ['A']],
        ['range', ['B', 0, 1, 0.1, 0, 1]],
        ['range', ['C'], {'type': 'minmax'}],
        ['range', ['D'], {'type': 'sigma', 'bins': 10, 'sigma': 3}],
        ['multiSelect', ["DDC"]],
        ['spinner', ['nBinsA']],
        ['spinner', ['nBinsB']],
        ['spinner', ['nBinsAAB']],
        ['slider',["size"]],
    #  ['select',["CC", 0, 1, 2, 3]],
    #  ['multiSelect',["BoolB"]],
    ]
    widgetLayoutDesc={
        "selection":[[0,1,2],[3,4]],
        "histograms":[[5,6,7]],
        "graphics":[[8]]
    }
    
    xxx=bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                              widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=3000, histogramArray=histoArray, aliasArray=aliasArray, arrayCompression=arrayCompressionRelative16)

def test_interactiveTemplateWeights():
    output_file("test_histogramTemplate.html")
    aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVarsDiff(variables=["A", "B", "C", "D", "A*A", "A*A+B", "B/(1+C)"], weights=[None, "A>.5", "B>C"], multiAxis="weights")
    parameterArray = parameterArray + [
                {"name":"varXMulti", "value":["A", "B"], "options":variables},
                {"name":"varYMulti", "value":["A", "B"], "options":variables}
            ]
    widgetsExtra = [
            ['range', ['A'], {"name":"A"}],
            ['range', ['B'], {"name":"B"}],
            ['range', ['C'], {"name":"C"}],
            ['range', ['D'], {"name":"D"}],
            ['multiSelect', ['varXMulti'], {"name":"varXMulti"}],
            ['multiSelect', ['varYMulti'], {"name":"varYMulti"}],
            ]
    widgetParams = mergeFigureArrays(widgetParams, widgetsExtra)
    widgetLayoutDesc["Select"] = [["A","B"],["C","D"]]
    widgetLayoutDesc["Histograms"][0].extend(["varXMulti", "varYMulti"])
    histoArray = histoArray + [
            {"name":"histo1D", "variables":["varX"], "nbins":"nbinsX"},
            {"name":"histo1DMulti", "variables":["varXMulti"], "nbins":"nbinsX"},
            {"name":"histoNDMultiX", "variables":["varXMulti", "varY"], "quantiles":[0.35, 0.5], "unbinned_projections":True, "nbins":["nbinsX", "nbinsY"]},
            {"name":"histoNDMultiY", "variables":["varX", "varYMulti"], "quantiles":[0.35, 0.5], "unbinned_projections":True, "nbins":["nbinsX", "nbinsY"]},
            ]
    figureArray1D = [
            [["bin_center"], ["bin_count"], {"source":"histo1D", "name":"histo1D"}],
            [["bin_center"], ["bin_count"], {"source":"histo1DMulti", "name":"histo1DMulti"}],
            [["bin_center_0"], ["mean"], {"source":"histoNDMultiX_1", "name":"histoNDMultiX_1_Mean", "errY": "std/sqrt(entries)"}],
            [["bin_center_0"], ["std"], {"source":"histoNDMultiX_1", "name":"histoNDMultiX_1_Median", "errY": "std/sqrt(entries)"}],
            [["mean"], ["bin_center_1"], {"source":"histoNDMultiX_0", "name":"histoNDMultiX_0_Mean", "errX": "std/sqrt(entries)"}],
            [["std"], ["bin_center_1"], {"source":"histoNDMultiX_0", "name":"histoNDMultiX_0_Median", "errX": "std/sqrt(entries)"}],
            [["bin_center_0"], ["mean"], {"source":"histoNDMultiY_1", "name":"histoNDMultiY_1_Mean", "errY": "std/sqrt(entries)"}],
            [["bin_center_0"], ["std"], {"source":"histoNDMultiY_1", "name":"histoNDMultiY_1_Median", "errY": "std/sqrt(entries)"}],
            [["mean"], ["bin_center_1"], {"source":"histoNDMultiY_0", "name":"histoNDMultiY_0_Mean", "errX": "std/sqrt(entries)"}],
            [["std"], ["bin_center_1"], {"source":"histoNDMultiY_0", "name":"histoNDMultiY_0_Median", "errX": "std/sqrt(entries)"}],
            ]
    figureArray = mergeFigureArrays(figureArray, figureArray1D)
    figureLayoutDesc["Histo1D"] = [["histo1D", "histo1DMulti"], {"plot_height":350}]
    figureLayoutDesc["HistoMultiX"] = [["histoNDMultiX_1_Mean", "histoNDMultiX_1_Median"], ["histoNDMultiX_0_Mean", "histoNDMultiX_0_Median"], {"plot_height":240}]
    figureLayoutDesc["HistoMultiY"] = [["histoNDMultiY_1_Mean", "histoNDMultiY_1_Median"], ["histoNDMultiY_0_Mean", "histoNDMultiY_0_Median"], {"plot_height":240}]
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray, aliasArray=aliasArray, arrayCompression=arrayCompressionRelative16)

def test_interactiveTemplateMultiX():
    output_file("test_histogramTemplateMultiX.html")
    aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVarsRatio(variables=["A", "B", "C", "D", "A*A", "A*A+B", "B/(1+C)"], defaultVariables={"varX":["A"]})
    widgetsSelect = [
        ['range', ['A'], {"name":"A"}],
        ['range', ['B'], {"name":"B"}],
        ['range', ['C'], {"name":"C"}],
        ['range', ['D'], {"name":"D"}],
        ]
    widgetParams = mergeFigureArrays(widgetParams, widgetsSelect)
    widgetLayoutDesc["Select"] = [["A","B"],["C","D"]]
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray, aliasArray=aliasArray, arrayCompression=arrayCompressionRelative16)
    
def test_interactiveTemplateMultiY():
    output_file("test_histogramTemplateMultiY.html")
    aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVarsDiff(variables=["A", "B", "C", "D", "A*A", "A*A+B", "B/(1+C)"], defaultVariables={"varY":["B", "A*A+B"]})
    widgetsSelect = [
        ['range', ['A'], {"name":"A"}],
        ['range', ['B'], {"name":"B"}],
        ['range', ['C'], {"name":"C"}],
        ['range', ['D'], {"name":"D"}],
        ]
    widgetParams = mergeFigureArrays(widgetParams, widgetsSelect)
    widgetLayoutDesc["Select"] = [["A","B"],["C","D"]]
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray, aliasArray=aliasArray, arrayCompression=arrayCompressionRelative16)
 
def test_interactiveTemplateMultiDiff():
    output_file("test_histogramTemplateMultiDiff.html")
    aliasArray, jsFunctionArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVarsNormAll(variables=["A", "B", "C", "D", "A*A", "A*A+B", "B/(1+C)"])
    widgetsSelect = [
        ['range', ['A'], {"name":"A"}],
        ['range', ['B'], {"name":"B"}],
        ['range', ['C'], {"name":"C"}],
        ['range', ['D'], {"name":"D"}],
        ]
    widgetParams = mergeFigureArrays(widgetParams, widgetsSelect)
    widgetLayoutDesc["Select"] = [["A","B"],["C","D"]]
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray, aliasArray=aliasArray, arrayCompression=arrayCompressionRelative16,
                          jsFunctionArray=jsFunctionArray)

def test_interactiveTemplateMultiYDiff():
    output_file("test_histogramTemplateMultiY.html")
    aliasArray, jsFunctionArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVarsNormAll(variables=["A", "B", "C", "D"], multiAxis="varY")
    widgetsSelect = [
        ['range', ['A'], {"name":"A"}],
        ['range', ['B'], {"name":"B"}],
        ['range', ['C'], {"name":"C"}],
        ['range', ['D'], {"name":"D"}],
        ]
    widgetParams = mergeFigureArrays(widgetParams, widgetsSelect)
    widgetLayoutDesc["Select"] = [["A","B"],["C","D"]]
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray, aliasArray=aliasArray, arrayCompression=arrayCompressionRelative16,
                          jsFunctionArray=jsFunctionArray)
    
