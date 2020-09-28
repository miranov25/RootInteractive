import pandas as pd
import numpy as np
import pickle
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.MLpipeline.NDFunctionInterface import *
from bokeh.io import output_notebook

widgetParams = [
    ['range', ['A']],
    ['range', ['B']],
    ['range', ['C']],
    ['range', ['D']],
    ['range', ['csin']],
    ['range', ['ccos']],
]
widgetLayoutDesc = [[0, 1, 2], [3, 4,5], {'sizing_mode': 'scale_width'}]
tooltips = [("A", "@A"), ("B", "@B"), ("C", "@C"), ("RF", "@RF"), ("RFRMS", "@RFRMS"), ("RF14RMS", "@RF14RMS"),
            ("GBR", "@GBR"), ("GBRRMS", "@GBRRMS")]



def generateInput(nPoints, sigma=0.2, tailFraction=0.0,tailWidth=5.):
    """
    :param nPoints:          number of points to generate
    :param sigma:            RMS of distribution
    :param tailFraction:     fraction of poiitns with tail
    :param tailWidth:        multiplicative factor tail vs nominal sigma
    :return:
    Generate random panda+tree random vectors A,B,C,D
        * generate function value = A+exp(3B)*sin(6.28C)
        * generate noise vector
    """
    df = pd.DataFrame(np.random.random_sample(size=(nPoints, 4)), columns=list('ABCD'))
    df["noise"] =  np.random.normal(0, sigma, nPoints)*(tailWidth*(np.random.random(nPoints)<tailFraction)+1)
    df["csin"]  = np.sin(6.28 * df["C"])
    df["ccos"]  = np.cos(6.28 * df["C"])
    df["valueOrig"] = 3 * df["A"] + np.exp(2 * df["B"]) * df["csin"]
    df["value"] = df["valueOrig"] + df["noise"]
    df["expB"] =np.exp(2 * df["B"])
    return df

def makeFits(df,n_jobs=-1):
    varFit = 'value'
    variableX = ['A', "expB", "csin", 'D']
    nPoints = df.shape[0]
    dataContainer = DataContainer(df, variableX, varFit, [nPoints // 2, nPoints // 2])
    fitter = Fitter(dataContainer)
    fitter.Register_Model('RF', RandomForest('Regressor', n_estimators=900, max_depth=None, n_jobs=-1))
    fitter.Fit()
    return fitter

def makeStatSummary(rf, X, Y, Yorig):
    dfSummary=pd.DataFrame(index=range(100),
        columns=["nPoints","sigma","fraction",
                 "group","n_groups",
                 "RMSMean","MeanRMSMean",
                 "RMSMeanMean","RMSMedianMean",
                 "fdMeanStd","dMeanStd",
                 "fdMedianStd","dMedianStd",
                 #"fdMeanMeanStd","dMeanMeanStd",
                 "fdMeanMedStd","dMeanMedStd",
                 "fdMedMedStd","dMedMedStd",
                 ])
    dfIndex=0;
    for n_groups in range(10,50,20):
        for group in range(5, 40, 5):
            if group*n_groups>len(rf.model.estimators_):
                continue
            #dfSummary["nPoints"][dfIndex]=nPoints
            dfSummary["group"][dfIndex]=group
            dfSummary["n_groups"][dfIndex]=n_groups
            stat=rf.predictStat(X,group=group, n_groups=n_groups)
            #dfSummary["MeanRMS"][dfIndex]=
            dfSummary["RMSMean"][dfIndex]=stat["RMS"].mean()                 # mean rms using all points
            dfSummary["MeanRMSMean"][dfIndex]=stat["MeanRMS"].mean()
            dfSummary["RMSMeanMean"][dfIndex]=stat["RMSMean"].mean()
            dfSummary["RMSMedianMean"][dfIndex]=stat["RMSMedian"].mean()
            #
            dfSummary["fdMeanStd"][dfIndex]=(stat["Mean"]-Yorig).std()
            dfSummary["dMeanStd"][dfIndex]=(stat["Mean"]-Y).std()
            dfSummary["fdMedianStd"][dfIndex]=(stat["Median"]-Yorig).std()
            dfSummary["dMedianStd"][dfIndex]=(stat["Median"]-Y).std()
            #dfSummary["fdMeanMeanStd"][dfIndex]=(stat["MeanMean"]-Yorig).std()
            #dfSummary["dMeanMeanStd"][dfIndex]=(stat["MeanMean"]-Y).std()
            dfSummary["fdMeanMedStd"][dfIndex]=(stat["MeanMedian"]-Yorig).std()
            dfSummary["dMeanMedStd"][dfIndex]=(stat["MeanMedian"]-Y).std()
            dfSummary["fdMedMedStd"][dfIndex]=(stat["MedianMedian"]-Yorig).std()
            dfSummary["dMedMedStd"][dfIndex]=(stat["MedianMedian"]-Y).std()
            dfIndex+=1;
    dfSummary=dfSummary[0:dfIndex]
    return dfSummary

def makeScan(nPoints0, nPoints1, nPointsD, nPointsTest=5000):
    tables=[]
    for nPoints in range(nPoints0,nPoints1, nPointsD):
        for fraction in np.arange(0, 0.5, 0.1):
            for sigma in np.arange(0.1, 0.6, 0.1):
                print("Scan\t",nPoints,fraction,sigma)
                df = generateInput(nPoints,sigma, fraction,10.)
                fitter=makeFits(df)
                df = generateInput(nPointsTest,sigma, fraction,10.)
                rf=fitter.Models[0]
                X=df[fitter.data.X_values].to_numpy()
                Y=df["value"].to_numpy()
                Yorig=df["valueOrig"].to_numpy()
                dfOut=makeStatSummary(rf,X,Y,Yorig)
                dfOut["nPoints"]=nPoints
                dfOut["fraction"]=fraction
                dfOut["sigma"]=sigma
                print(dfOut)
                tables.append(dfOut)
    dfSummary=pd.concat(tables)
    return dfSummary

def scanStat(dfSummary):
    #xxx
    dfSummary["irreducible0"]=np.sqrt((dfSummary['deltaMeanStd']**2-dfSummary['fdeltaMeanStd']**2).astype(np.float64)) # this should be equal to sigma

def scanSummary():
    dfSummary=makeScan(20000,160001,20000,4000)
    dfSummary=dfSummary.astype(np.float64)
    pickle.dump(dfSummary,open('dfSummary.pkl', 'wb'))
# fSummary=makeScan(10000,40001,10000,2000)
#dfSummary=makeScan(2000,11000,40000,2000)
#dfSummary=makeScan(20000,11000,20000,2000)
#pickle.dump(dfSummary,open('dfSummary.pkl', 'wb'))