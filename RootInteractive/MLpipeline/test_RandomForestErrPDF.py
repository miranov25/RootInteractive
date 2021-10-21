#  from RootInteractive.MLpipeline.test_RandomForestErrPDF import *
import pandas as pd
import numpy as np
import pickle
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.MLpipeline.NDFunctionInterface import *
#from bokeh.io import output_notebook
from RootInteractive.MLpipeline.RandoForestErrPDF import *

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

def generateInput(nPoints):
    """
    Generate random panda+tree random vectors A,B,C,D
        * generate function value = A+exp(3B)*sin(6.28C)
        * generate noise vector
    """
    df = pd.DataFrame(np.random.random_sample(size=(nPoints, 4)), columns=list('ABCD'))
    # df["noise"] = np.random.normal(0, 0.05, nPoints) + np.random.normal(0, 0.5, nPoints)*(np.random.random(nPoints)>0.8)
    df["noise"] = np.random.normal(0, 0.2, nPoints)
    df["csin"] = np.sin(6.28 * df["C"])
    df["ccos"] = np.cos(6.28 * df["C"])
    df["valueOrig"] = 3 * df["A"] + np.exp(2 * df["B"]) * df["csin"]
    df["value"] = df["valueOrig"] + df["noise"]
    # df["value"] = df["valueOrig"] + df["noise"]
    df["expB"] =np.exp(2 * df["B"])
    return df


def makeFits(df,n_jobs=-1):
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

def doFit(nPoints, n_jobs):
    df =generateInput(nPoints)
    makeFits(df,n_jobs)