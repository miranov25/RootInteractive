#  from RootInteractive.MLpipeline.test_RandomForestErrPDF import *
import pandas as pd
import numpy as np
import pickle
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.MLpipeline.NDFunctionInterface import *
#from bokeh.io import output_notebook
from RootInteractive.MLpipeline.RandoForestErrPDF import *
from RootInteractive.MLpipeline.MIForestErrPDF import *
from RootInteractive.MLpipeline.local_linear_forest import LocalLinearForestRegressor


widgetParams = [
    ['range', ['A']],
    ['range', ['B']],
    ['range', ['C']],
    ['range', ['D']],
    ['range', ['csin']],
    ['range', ['ccos']],
]
widgetLayoutDesc = [[0, 1, 2], [3, 4,5], {'sizing_mode': 'scale_width'}]
tooltips = [("A", "@A"), ("B", "@B"), ("C", "@C"), ("RF", "@RF")]
methods = ['RF']
rfErrPDF=0
df=0
fitter=0
def generateInput(nPoints, outFraction=0.0):
    """
    Generate random panda+tree random vectors A,B,C,D
        * generate function value = A+exp(3B)*sin(6.28C)
        * generate noise vector
    """
    df = pd.DataFrame(np.random.random_sample(size=(nPoints, 4)), columns=list('ABCD'))
    # df["noise"] = np.random.normal(0, 0.05, nPoints) + np.random.normal(0, 0.5, nPoints)*(np.random.random(nPoints)>0.8)
    df["noise"] = np.random.normal(0, 0.2, nPoints)
    df["noise"] += (np.random.random(nPoints)<outFraction)*np.random.normal(0, 2, nPoints)
    df["csin"] = np.sin(6.28 * df["C"])
    df["ccos"] = np.cos(6.28 * df["C"])
    df["valueOrig"] = 3 * df["A"] + np.exp(2 * df["B"]) * df["csin"]
    df["value"] = df["valueOrig"] + df["noise"]
    # df["value"] = df["valueOrig"] + df["noise"]
    df["expB"] =np.exp(2 * df["B"])
    return df


def makeFitsRF(df,n_jobs=-1):
    varFit = 'value'
    #variableX = ['A', "B", "C", 'D']
    variableX = ['A', "expB", "csin", 'D']
    nPoints = df.shape[0]
    dataContainer = DataContainer(df, variableX, varFit, [nPoints // 2, nPoints // 2])
    fitter = Fitter(dataContainer)
    #
    rfErrPDF=RandomForestErrPDF("Regressor",{"nSplit": 8, "max_depthBegin":-1, "n_jobs":n_jobs, "n_estimatorsEnd":500 })
    fitter.Register_Model('RFErrPDF',rfErrPDF)
    fitter.Fit()
    return fitter

def doFitRF(nPoints, n_jobs):
    df =generateInput(nPoints)
    makeFitsRF(df,n_jobs)


def makeFitsErrPDF(df,n_jobs=-1, mean_depth=14, niter_max=8):
    varFit = 'value'
    #variableX = ['A', "B", "C", 'D']
    variableX = ['A', "expB", "csin", 'D']
    nPoints = df.shape[0]
    dataContainer = DataContainer(df, variableX, varFit, [nPoints // 2, nPoints // 2])
    fitter = Fitter(dataContainer)
    #
    miErrPDF=MIForestErrPDF("Regressor",{"mean_depth":mean_depth, "n_jobs":n_jobs,"niter_max":niter_max })
    llfForest=LocalLinearForestRegressor(n_estimators=mean_depth,n_jobs=n_jobs)
    #fitter.Register_Model('RFErrPDF',miErrPDF)
    fitter.Register_Model('llfForest',llfForest)
    fitter.Fit()
    return fitter

def doFitErrRF(nPoints, n_jobs, mean_depth, niter_max):
    df =generateInput(nPoints,0)
    fitter=makeFitsErrPDF(df,10,mean_depth, niter_max)
    return fitter,df

def doFitLLR(n_estimators=200):
    llf=  LocalLinearForestRegressor(n_estimators=5)


def appendPrediction(nPoints):
    fitter,df=doFitErrRF(nPoints,1,8,10)
    reducibleErrorEst = {}
    statDict={"mean":0,"median":0,"std":0}
    dataIn=df[fitter.data.X_values]
    fitter.Models[0].predict(dataIn)
    return
    #
    yPred0 = fitter.Models[0].predictStat(dataIn, 0, statDict, slice(0,dataIn.shape[0]))
    df["pred0"]=statDict["mean"]
    df["predMedian0"]=statDict["median"]
    yPred1 = fitter.Models[0].predictStat(dataIn, 1, statDict, slice(0,dataIn.shape[0]))
    df["pred1"]=statDict["mean"]
    df["predMedian1"]=statDict["median"]
    df["pred"]=0.5*(df["pred0"]+df["pred1"])
    df["predMedian"]=0.5*(df["predMedian0"]+df["predMedian1"])
    df["dpred"]=0.5*(df["pred0"]+df["pred1"])-df["valueOrig"]
    df["dpredF"]=0.5*(df["pred0"]+df["pred1"])-df["value"]
    df["dpred01"]=(df["pred0"]-df["pred1"])
    df["dpredMedian"]=0.5*(df["predMedian0"]+df["predMedian1"])-df["valueOrig"]

    fitter.Models[0].predictReducibleError(dataIn,reducibleErrorEst,64,[0.5,0.7,0.8,0.9])
    deltaPred = fitter.Models[0].predictStat(dataIn, -1, statDict, slice(0,dataIn.shape[0]))
    df["redErrEst05"]=reducibleErrorEst[0.5]
    df["redErrEst07"]=reducibleErrorEst[0.7]
    df["redErrEst08"]=reducibleErrorEst[0.8]
    df["redErrEst09"]=reducibleErrorEst[0.9]
    df["stdPoint"]=statDict["std"]
    df["irrError"]=np.sqrt(df["stdPoint"]**2-df["redErrEst08"]**2)
    #
    df["ppredF"]=df["dpredF"]/df["stdPoint"]
    df["ppred"]=df["dpred"]/df["stdPoint"]
    df["ppred01"]=df["dpred01"]/df["redErrEst08"]
    return fitter, df

def makeDashboard(fitter,df):
    #fitter,df=appendPrediction()
    histoArray = [
        {"name": "hisDpredF", "variables": ["dpredF"], "nbins": 50},
        {"name": "hisDpred", "variables":   ["dpred"], "nbins":50},
        {"name": "hisDpred01", "variables": ["dpred01"], "nbins":50},
        {"name": "hisppredF", "variables": ["ppredF"], "nbins": 50},
        {"name": "hisppred", "variables":   ["ppred"], "nbins":50},
        {"name": "hisppred01", "variables": ["ppred01"], "nbins":50},
    ]
    figureArray = [
        [['value'], ['dpredF'], {"errY":"stdPoint"}],
        [['valueOrig'], ['dpred'], {"errY":"redErrEst09"}],
        [['valueOrig'], ['dpred01'], {"errY":"redErrEst09"}],
        #
        [['hisDpredF'], ['hisDpredF']],
        [['hisDpred'], ['hisDpred']],
        [['hisDpred01'], ['hisDpred01']],
        #
        [['hisppredF'], ['hisppredF']],
        [['hisppred'], ['hisppred']],
        [['hisppred01'], ['hisppred01']],
        ["tableHisto", {"rowwise": True}],
        #
        [['value'], ['redErrEst08'], {"errY":"stdPoint"}],
        [['value'], ['stdPoint'], {"errY":"stdPoint"}],
        [['expB'], ['redErrEst08'], {"errY":"stdPoint"}],
        [['expB'], ['stdPoint'], {"errY":"stdPoint"}],
    ]

    figureLayoutDesc=[
        [0, 1, 2,  {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [3, 4, 5,  {'commonX': 4, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [6, 7, 8,  {'commonX': 6, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
        [9,{'plot_height': 40}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2, "size": 5}
    ]
    figureLayoutDescT1=[
        [10,11],
        [12,13]
    ]
    figureLayoutDescT={
        "Value":figureLayoutDesc,
        "Errors":figureLayoutDescT1
    }


    widgetParams=[
        ['range', ['A']],
        ['range', ['B']],
        ['range', ['C']],
        ['range', ['D']],
        #
        ['range', ['expB']],
        ['range', ['csin']],
        ['range', ['ccos']],
        ['range',["noise"]]
    ]
    widgetLayoutDesc=[[0, 1, 2,3], [4, 5,6,7], {'sizing_mode': 'scale_width'}]
    output_file("test_ErrPdf500000.html")
    xxx = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDescT, tooltips=tooltips,
                                widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=500, histogramArray=histoArray)
    return 0


appendPrediction(10000)

#doFitErrRF(400000,1,12,8)