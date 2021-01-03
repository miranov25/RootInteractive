import pandas as pd
import numpy as np
from scipy.stats import entropy
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.Tools.compressArray import *

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


def simulatePandaDCA(fastTest=False):
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
    rangeH = ([-5, 5], [-1, 1], [0, 1], [0, 2 * np.pi], [-10 * sigma0 - 1, 10 * sigma0 + 1])
    bins = [50, 20, 20, 12, 100]
    if (fastTest):
        bins = [50, 10, 10, 12, 50]
    H, edges = np.histogramdd(sample=np.array([[0, 0, 0, 0, 0]]), bins=bins, range=rangeH)
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
    fig = bokehDrawSA.fromArray(df.query("(index%27)==0"), "weight>0", figureArray, widgetParams, layout=figureLayoutDesc,
                                tooltips=tooltips, sizing_mode='scale_width', widgetLayout=widgetLayoutDesc,
                                nPointRender=5000, rescaleColorMapper=True)

def mitest_roundRelativeBinary(df):
    """
    :param df:
    :return:
    """
    df["weightR5"]=roundRelativeBinary(df["weight"],5)
    df["weightR7"]=roundRelativeBinary(df["weight"],7)
    df["weightR8"]=roundRelativeBinary(df["weight"],8)
    mapIndex,mapCodeI=codeMapDF(df,0.5,1)
    sizeR0=getCompressionSize(df["weight"].to_numpy())/getSize(df["weight"].to_numpy())
    sizeR5=getCompressionSize(df["weightR5"].to_numpy().astype("int8"))/getSize(df["weight"].to_numpy())
    sizeR7=getCompressionSize(df["weightR7"].to_numpy().astype("int8"))/getSize(df["weight"].to_numpy())
    sizeR8=getCompressionSize(df["weightR8"].to_numpy().astype("int16"))/getSize(df["weight"].to_numpy())
    entropy0=entropy(df["weight"].value_counts(), base=2)
    entropy5=entropy(df["weightR5"].value_counts(), base=2)
    entropy7=entropy(df["weightR7"].value_counts(), base=2)
    entropy8=entropy(df["weightR8"].value_counts(), base=2)
    print("mitest_roundRelativeBinary(df)")
    print(sizeR0,sizeR8,sizeR7,sizeR5)
    print(entropy0/64,entropy8/64,entropy7/64,entropy5/64)
    print(entropy0,entropy8,entropy7,entropy5)

def test_Compression0():
    df = simulatePandaDCA(True)
    mitest_roundRelativeBinary(df)
    #histogramDCAPlot(df)

def testCompressionDecompression0(stop=10,step=1):
    # compress pipeline
    V0 = np.arange(start=0, stop=stop, step=step).astype("int16")
    V1=base64.b64encode(V0.data)
    V1C=zlib.compress(V1)
    V1CB=base64.b64encode(V1C).decode('utf-8')
    # decompres pipline
    V1CBI=base64.b64decode(V1CB)
    V1D=zlib.decompress(V1CBI)
    V2= base64.b64decode(V1D)
    V3=np.frombuffer(V2, dtype='int16')
    print("V0",V0);
    print("V1",V1);
    print("V1C",V1C);
    print("V1CB",V1CB);
    print("V1CBI",V1CBI);
    print("V1D",V1D);
    print("V2",V2);
    print("V3",V3);
    diffSum=(V3-V0).sum()
    if diffSum>0:
        raise ValueError("testCompressionDecompression0 error")