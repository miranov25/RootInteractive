#  from RootInteractive.MLpipeline.test_MIForestErrPDF import *
import pandas as pd
import numpy as np
import pickle
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.MLpipeline.NDFunctionInterface import *
#from bokeh.io import output_notebook
from RootInteractive.MLpipeline.RandoForestErrPDF import *
from RootInteractive.MLpipeline.MIForestErrPDF import *
from RootInteractive.MLpipeline.local_linear_forest import LocalLinearForestRegressor
import pdb;
import sys
import os;
sys.path.insert(1, os.environ[f"RootInteractive"]+'/RootInteractive/MLpipeline/') # enable deubug symbols in path

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


def test_MIForestErrPDF(nPoints=100000,outFraction=0.1,n_jobs=2):
    df=generateInput(nPoints,outFraction)
    varFit = 'value'
    variableX = ['A', "expB", "csin", 'D']
    miErrPDF=MIForestErrPDF("Regressor",{"mean_depth":mean_depth, "n_jobs":n_jobs,"niter_max":niter_max })
    miErrPDF.fit(df[variableX][0:nPoints//2],df[varFit][0:nPoints//2])
    return miErrPDF,df, df[variableX]

def drawReducible():
    outputErr={}
    x=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    miErrPDF.predictReducibleError(dfVar,  outputErr, 32, x)
    f=pd.DataFrame([x,x]).transpose()
    for i in range(9) : f[1][i]=outputErr[f[0][i]].mean()



#%%time
nPoints=100000; outFraction=0.30; mean_depth=16; niter_max=16; n_jobs=16;
miErrPDF,df,dfVar=test_MIForestErrPDF(nPoints,outFraction,n_jobs)
