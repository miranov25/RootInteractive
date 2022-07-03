#  from RootInteractive.MLpipeline.scan.scan_MIxgboostErrPDF import *
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
import xgboost as xgb
from RootInteractive.MLpipeline.MIxgboostErrPDF import *
from scipy.signal import medfilt
from xgboost import XGBRegressor


def generateF1(nPoints, norm, nSin, outFraction,stdIn):
    """
    Generate random panda+tree random vectors A,B,C,D  - A and C used to define function
        * generate function value = 2*A*sin(n*6.28*C)
        * generate noise vector
        * calculate local gradient of function
    """
    df = pd.DataFrame(np.random.random_sample(size=(nPoints, 4)), columns=list('ABCD'))
    df["B"]=df["B"]+0.5
    df["noise"] = np.random.normal(0, stdIn, nPoints)
    #df["noise"]+=np.roll(df["noise"],1)   - adding correlated noise?
    df["noise"] += (np.random.random(nPoints)<outFraction)*np.random.normal(0, 2, nPoints)
    df["csin"] = np.sin(nSin*6.28 * df["C"])
    df["ccos"] = np.cos(nSin*6.28 * df["C"])
    df["valueOrig"] = norm*df["A"]*df["csin"]
    df["value"] = df["valueOrig"] + df["noise"]
    df["gradA"] = df["csin"]
    df["gradC"] = df["A"]*df["ccos"]*nSin*6.28
    df["grad"]  =np.sqrt(df["gradA"]**2+df["gradC"]**2)
    # df["value"] = df["valueOrig"] + df["noise"]
    return df

def makeFit(df,dfRef):
    varFit = 'value'
    variableX = ['A', "B", "C"]
    paramTrain = {'learning_rate':0.2, 'max_depth':10,"n_estimators":200,"subsample":0.50,"coeff_learning_rate":0.2,"max_learning_rate":0.2}
    xgbErrPDF=MIxgboostErrPDF(paramTrain)
    xgbErrPDF.fit3Fold(df[variableX].to_numpy(),df["value"].to_numpy(),df["value"])
    xgbErrPDF.fitReducible()
    outStatRed=xgbErrPDF.predictReducibleError(dfRef[variableX].to_numpy())
    outStat=xgbErrPDF.predictStat(dfRef[variableX].to_numpy())
    for stat in outStatRed:
        dfRef[stat]=outStatRed[stat]
    for stat in outStat:
        dfRef[stat]=outStat[stat]
    return xgbErrPDF,dfRef

def makeScan(nIter=100):
    for iter in range(nIter):
        #nPoints=1000000;  stdIn=0.1; nSin=2; norm=1
        params={}
        nPoints=np.random.randint(50000,500000)
        stdIn=np.random.uniform(0.05,0.2)
        nSin=np.random.randint(1,4)
        norm=np.random.uniform(0.2,2)
        params["nPoints"]=nPoints
        params["stdIn"]=stdIn
        params["nsin"]=nSin
        params["norm"]=norm
        outFraction=0.0; n_jobs=6;
        #
        print(f"\n\nIter - {iter} {params}",)
        df   =generateF1(nPoints, nSin=nSin, outFraction=outFraction,stdIn=stdIn,norm=norm)
        dfRef=generateF1(nPoints, nSin=nSin, outFraction=outFraction,stdIn=stdIn,norm=norm)
        xgbErrPDF,dfRef=makeFit(df,dfRef)
        dfRef["nPoints"]=nPoints
        dfRef["stdIn"]=stdIn
        dfRef["nSin"]=nSin
        dfRef["norm"]=norm
        # pick out
        summary={}
        summary["xgbErrPDF"]=xgbErrPDF
        summary["dfRef"]=dfRef.sample(min(25000,dfRef.shape[0]))
        fOutput=f"xgbErrPdf_{nPoints}_{stdIn:0.4}_{nSin}_{norm}.pickle"
        with open(fOutput, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(summary["dfRef"], f, pickle.HIGHEST_PROTOCOL)

def joinScan():
    with open('scan.list') as f:
        scanList = [line.rstrip('\n') for line in f]
    #
    dfMap={}
    dfList=[]
    for i, inFile in enumerate(scanList):
        with open(inFile, 'rb') as f:
            data = pickle.load(f)
            dfMap[i]=data
            dfList.append(data)
    df = pd.concat(dfList)