import pandas as pd
import numpy as np
# from pandas import CategoricalDtype
from scipy.stats import entropy
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *


def estimateEntropy():
    random = np.random.randn(10000, 3)
    nBins = 10
    H, edges = np.histogramdd(random, bins=(nBins, nBins, nBins), range=[[-6, 6], [-6, 6], [-6, 6]])
    a = np.empty(H.size)
    b = np.empty(H.size)
    c = np.empty(H.size)
    for i in range(H.size):
        a[i] = edges[0][i % nBins]
        b[i] = edges[1][(i // nBins) % nBins]
        c[i] = edges[2][i // (nBins * nBins)]
    #    c[i]=edges[2][i%(nBins*nBins)]
    dfE = pd.DataFrame(data={'A': a, 'B': b, 'C': c, 'H': H.flatten()})
    # Test - send to cds
    entropy(dfE["A"].value_counts(), base=2)
    entropy(((dfE[0:-1] - dfE[1:])["A"])[1:-1].value_counts(), base=2)
    # entropy=0
    entropy(dfE["B"].value_counts(), base=2)
    entropy(((dfE[0:-1] - dfE[1:])["B"])[1:-1].value_counts(), base=2)
    # entropy=0
    entropy(dfE["H"].value_counts(), base=2)


# entropy=3.1

# TODO
#      if (User coding){
#          float-> integer
#          entropy coding
#      }else{
#       1. Unique gives amount of distinct value
#       1. if Unique<<size
#           1. Entropy for value_counts
#           2. Entropy for delta.value-counts
#           3. Use coding with smaller entropy
#   }


def simulatePandaDCA():
    """
    simulate DCA 5 D histograms :
    DCA: qPt,tgl,mdEd,alpha
    :return:  panda data frame with histogram and some PDF variables
    """
    sigma0 = 0.1
    sigma1 = 1
    qPtSlope = 0.1
    tglSlope = 0.1
    entries = 1000
    # qPt,tgl.mdEdx.alpha, dCA
    range = ([-5, 5], [-1, 1], [0, 1], [0, 2 * np.pi], [-10 * sigma0 - 1, 10 * sigma0 + 1])
    bins = [50, 20, 20, 12, 100]
    H, edges = np.histogramdd(sample=np.array([[0, 0, 0, 0, 0]]), bins=bins, range=range)
    indexH = np.arange(H.size)
    indexC = np.unravel_index(indexH, bins)
    qPtCenter = (edges[0][indexC[0]] + edges[0][indexC[0] + 1]) * .5
    tgl = (edges[1][indexC[1]] + edges[1][indexC[1] + 1]) * .5
    mdEdx = (edges[2][indexC[2]] + edges[2][indexC[2] + 1]) * .5
    alpha = (edges[3][indexC[3]] + edges[3][indexC[3] + 1]) * .5
    #
    valueMean = qPtCenter * qPtSlope + tgl * tglSlope
    value = edges[4][indexC[4]]
    valueSigma = sigma0 * np.sqrt(1 + sigma1 * mdEdx * qPtCenter * qPtCenter)
    weight = np.exp(-(value - valueMean) ** 2 / (2 * valueSigma * valueSigma))
    weightPoisson = np.random.poisson(weight * entries, H.size)
    H = weightPoisson
    df = pd.DataFrame({"qPtCenter": qPtCenter, "tglCenter": tgl, "mdEdxCenter": mdEdx, "V": value, "alphaCenter": alpha,
                       "mean": valueMean, "rms": valueSigma, "weight": H})
    return df


def histogramDCAPlot(df):
    """
    :param df:
    :return:
    """
    df = simulatePandaDCA()
    output_file("dcaCompressionDemo.html")
    optionsAll = {"colorZvar": "mdEdxCenter"}
    df["Vcorr"] = df["V"] - df["mean"]
    figureArray = [
        [['qPtCenter'], ['mean'], optionsAll],
        [['tglCenter'], ['mean'], optionsAll],
        [['qPtCenter'], ['tglCenter'], {"color": "red", "size": 2, "colorZvar": "mean", "varZ": "mean"}],
        [['V'], ['weight'], optionsAll],
        [['V'], ['histo'], {"range_min": -2, "range_max": 2, "weights": "weight", "nbins": 100}],
        [['Vcorr'], ['histo'], {"range_min": -2, "range_max": 2, "weights": "weight", "nbins": 100}],
    ]

    widgetParams = [
        ['range', ['tglCenter']],
        ['range', ['qPtCenter']],
        ['range', ['mdEdxCenter']],
        ['range', ['alphaCenter']],
    ]
    tooltips = [("qPtMean", "@qPtMean")]
    widgetLayoutDesc = [[0, 1], [2, 3], {'sizing_mode': 'scale_width'}]
    figureLayoutDesc = [
        [0, 1, {'plot_height': 150, 'x_visible': 1}],
        [2, 3, {'plot_height': 150, 'x_visible': 1}],
        [4, 5, {'plot_height': 150, 'x_visible': 1}],
        {'plot_height': 250, 'sizing_mode': 'scale_width', "legend_visible": True}
    ]
    fig = bokehDrawSA.fromArray(df.sample(100000), "rms>0", figureArray, widgetParams, layout=figureLayoutDesc,
                                tooltips=tooltips, sizing_mode='scale_width', widgetLayout=widgetLayoutDesc,
                                nPointRender=10000, rescaleColorMapper=True)

df = simulatePandaDCA()
histogramDCAPlot(df)