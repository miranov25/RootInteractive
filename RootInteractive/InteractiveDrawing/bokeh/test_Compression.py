import pandas as pd
import numpy as np
from scipy.stats import entropy
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.Tools.compressArray import *
import time

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
    if fastTest:
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
    df = pd.DataFrame({"qPtCenter": qPtCenter, "tglCenter": tgl, "mdEdxCenter": mdEdx, "alphaCenter": alpha, "V": value,
                       "mean": valueMean, "rms": valueSigma, "weight": H})
    df["qPtMean"]=np.random.normal(df["qPtCenter"],0.05)
    df["tglMean"]=np.random.normal(df["tglCenter"],0.05)
    df["mdEdxMean"]=np.random.normal(df["mdEdxCenter"],0.05)
    df["alphaMean"]=np.random.normal(df["alphaCenter"],0.05)
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
    histogramArray = [
        {"name": "histoV", "variables": ["V"], "range": [-2, 2], "weights": "weight", "nbins": 100},
        {"name": "histoVcorr", "variables": ["Vcorr"], "range": [-2, 2], "weights": "weight", "nbins": 100}
    ]
    figureArray = [
        [['qPtCenter'], ['mean'], optionsAll],
        [['tglCenter'], ['mean'], optionsAll],
        [['qPtCenter'], ['tglCenter'], {"color": "red", "size": 2, "colorZvar": "mean", "varZ": "mean"}],
        [['V'], ['weight'], optionsAll],
        [['V'], ['histoV']],
        [['Vcorr'], ['histoVcorr']],
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
    fig = bokehDrawSA.fromArray(df.query("(index%27)==0"), "weight>0", figureArray, widgetParams,
                                layout=figureLayoutDesc,
                                tooltips=tooltips, sizing_mode='scale_width', widgetLayout=widgetLayoutDesc,
                                nPointRender=5000, rescaleColorMapper=True, histogramArray=histogramArray)


def miTest_roundRelativeBinary(df):
    """
    :param df:
    :return:
    """
    sizeCompress={}
    sizeOrig={}
    entropyM={}
    for iBit in range(0,10):
        dfR=roundRelativeBinary(df["weight"], iBit)
        df["weightR{}".format(iBit)] = dfR
        sizeCompress[iBit]=getCompressionSize(dfR.to_numpy())
        sizeOrig[iBit]=getSize(dfR.to_numpy())
        entropyM[iBit]=entropy(dfR.value_counts(), base=2)
        print(iBit, entropyM[iBit], sizeCompress[iBit]/sizeOrig[iBit], entropyM[iBit]/64.)
    mapIndex, mapCodeI = codeMapDF(df, 0.5, 1)
    return mapIndex, mapCodeI


def test_Compression0():
    df = simulatePandaDCA(True)
    mapIndexI, mapCodeI=miTest_roundRelativeBinary(df)
    # histogramDCAPlot(df)

def test_CompressionSequence0(arraySize=10000):
    actionArray=[("zip",0), ("base64",0), ("debase64",0),("unzip","int8")]
    for coding in ["int8","int16", "int32", "int64", "float32", "float64"]:
        actionArray[3]=("unzip",coding)
        arrayInput=pd.Series(np.arange(0,arraySize,1,dtype=coding))
        inputSize=getSize(arrayInput)
        tic = time.perf_counter()
        arrayC = compressArray(arrayInput,actionArray, True)
        toc = time.perf_counter()
        compSize=getSize(arrayC["history"][2])
        print("test_CompressionSequence0: {}\t{}\t{:04f}\t{}\t{}\t{}".format(coding, arraySize, toc - tic, inputSize, compSize/inputSize, (arrayC["array"]-arrayInput).sum()))

def test_CompressionSequenceRel(arraySize=255,nBits=5):
    actionArray=[("relative",nBits), ("zip",0), ("base64",0), ("debase64",0),("unzip","int8")]
    for coding in ["int8","int16", "int32", "int64", "float32", "float64"]:
        actionArray[4]=("unzip",coding)
        arrayInput=pd.Series(np.arange(0,arraySize,1,dtype=coding))
        inputSize=getSize(arrayInput)
        tic = time.perf_counter()
        arrayC = compressArray(arrayInput,actionArray, True)
        toc = time.perf_counter()
        compSize=getSize(arrayC["history"][2])
        print("test_CompressionSequenceRel: {}\t{}\t{:04f}\t{}\t{}\t{}".format(coding, arraySize, toc - tic, inputSize, compSize/inputSize, (arrayC["array"]-arrayInput).sum()/inputSize))

def test_CompressionSequenceAbs(arraySize=255,delta=0.1):
    actionArray=[("delta",delta), ("zip",0), ("base64",0), ("debase64",0),("unzip","int8")]
    for coding in ["int8","int16", "int32", "int64", "float32", "float64"]:
        actionArray[4]=("unzip",coding)
        arrayInput=pd.Series(np.arange(0,arraySize,1,dtype=coding))
        inputSize=getSize(arrayInput)
        tic = time.perf_counter()
        arrayC = compressArray(arrayInput,actionArray, True)
        toc = time.perf_counter()
        compSize=getSize(arrayC["history"][2])
        print("test_CompressionSequenceRel: {}\t{}\t{:04f}\t{}\t{}\t{}".format(coding, arraySize, toc - tic, inputSize, compSize/inputSize, (arrayC["array"]-arrayInput).sum()/inputSize))



def test_CompressionSample0(arraySize=10000,scale=255):
    actionArray=[("zip",0), ("base64",0), ("debase64",0),("unzip","int8")]
    for coding in ["int8","int16", "int32", "int64", "float32", "float64"]:
        actionArray[3]=("unzip",coding)
        arrayInput=pd.Series((np.random.random_sample(size=arraySize)*scale).astype(coding))
        inputSize=getSize(arrayInput)
        tic = time.perf_counter()
        arrayC = compressArray(arrayInput,actionArray, True)
        toc = time.perf_counter()
        compSize=getSize(arrayC["history"][1])
        print("test_Compression0: {}\t{}\t{:04f}\t{}\t{}\t{}".format(coding, arraySize, toc - tic, inputSize, compSize/inputSize, (arrayC["array"]-arrayInput).sum()))


def test_CompressionSampleRel(arraySize=10000,scale=255, nBits=7):
    actionArray=[("relative",nBits), ("zip",0), ("base64",0), ("debase64",0),("unzip","int8")]
    for coding in ["float32", "float64"]:
        actionArray[4]=("unzip",coding)
        arrayInput=pd.Series((np.random.random_sample(size=arraySize)*scale).astype(coding))
        inputSize=getSize(arrayInput)
        tic = time.perf_counter()
        arrayC = compressArray(arrayInput,actionArray, True)
        toc = time.perf_counter()
        compSize=getSize(arrayC["history"][2])
        print("test_CompressionSampleRel: {}\t{}\t{:04f}\t{}\t{}\t{}".format(coding, arraySize, toc - tic, inputSize, compSize/inputSize, (np.abs(arrayC["array"]-arrayInput)/(arrayC["array"]+arrayInput)).sum()/arraySize))

def test_CompressionSampleDelta(arraySize=10000,scale=255, delta=1):
    actionArray=[("delta",delta), ("zip",0), ("base64",0), ("debase64",0),("unzip","int8")]
    for coding in ["float32", "float64"]:
        actionArray[4]=("unzip",coding)
        arrayInput=pd.Series((np.random.random_sample(size=arraySize)*scale).astype(coding))
        inputSize=getSize(arrayInput)
        tic = time.perf_counter()
        arrayC = compressArray(arrayInput,actionArray, True)
        toc = time.perf_counter()
        compSize=getSize(arrayC["history"][2])
        print("test_CompressionSampleRel: {}\t{}\t{:04f}\t{}\t{}\t{}".format(coding, arraySize, toc - tic, inputSize, compSize/inputSize, np.sqrt(((arrayC["array"]-arrayInput)**2).sum()/arraySize)))

def test_CompressionSampleDeltaCode(arraySize=10000,scale=255, delta=1):
    actionArray=[("delta",delta), ("code",0), ("zip",0), ("base64",0), ("debase64",0),("unzip","int8"),("decode",0)]
    for coding in ["float32", "float64"]:
        #actionArray[5]=("unzip",coding)
        arrayInput=pd.Series((np.random.random_sample(size=arraySize)*scale).astype(coding))
        inputSize=getSize(arrayInput)
        tic = time.perf_counter()
        arrayC = compressArray(arrayInput,actionArray, True)
        toc = time.perf_counter()
        compSize=getSize(arrayC["history"][3])
        print("test_CompressionSampleRel: {}\t{}\t{:04f}\t{}\t{}\t{}".format(coding, arraySize, toc - tic, inputSize, compSize/inputSize, np.sqrt(((arrayC["array"]-arrayInput)**2).sum()/arraySize)))


def testCompressionDecompressionInt8(stop=10, step=1):
    # compress pipeline
    t0 = np.arange(start=0, stop=stop, step=step).astype("int8")
    t1 = zlib.compress(t0)
    t2 = base64.b64encode(t1)
    # decompres pipline
    t3 = base64.b64decode(t2)
    t4 = zlib.decompress(t3)
    t5 = np.frombuffer(t4, dtype='int8')
    print("t0", t0)
    print("t1", t1)
    print("t2", t2)
    print("t3", t3)
    print("t4", t4)
    print("t5", t5)
    diffSum = (t5 - t0).sum()
    if diffSum > 0:
        raise ValueError("testCompressionDecompressionInt8 error")


def testCompressionDecompressionInt16(stop=10, step=1):
    # compress pipeline
    t0 = np.arange(start=0, stop=stop, step=step).astype("int16")
    t1 = zlib.compress(t0)
    t2 = base64.b64encode(t1)
    # decompres pipline
    t3 = base64.b64decode(t2)
    t4 = zlib.decompress(t3)
    t5 = np.frombuffer(t4, dtype='int16')
    print("t0", t0)
    print("t1", t1)
    print("t2", t2)
    print("t3", t3)
    print("t4", t4)
    print("t5", t5)
    diffSum = (t5 - t0).sum()
    if diffSum > 0:
        raise ValueError("testCompressionDecompressionInt16 error")
