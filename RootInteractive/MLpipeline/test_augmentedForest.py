import numpy as np
import pandas as pd
from bokeh.io import output_file
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
from RootInteractive.InteractiveDrawing.bokeh.bokehTools import mergeFigureArrays
from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate import getDefaultVarsNormAll
from RootInteractive.Tools.compressArray import arrayCompressionRelative16
from RootInteractive.MLpipeline.augmentedForest import AugmentedRandomForestArray
from RootInteractive.MLpipeline.MIForestErrPDF import predictRFStat
from sklearn.ensemble import RandomForestRegressor

def generateInput(nPoints, outFraction=0.0, noise=1):
    """
    Generate random panda+tree random vectors A,B,C,D
        * generate function value = A+exp(3B)*sin(6.28C)
        * generate noise vector
    """
    df = pd.DataFrame(np.random.random_sample(size=(nPoints, 4)), columns=list('ABCD'))
    df["noise"] = np.random.normal(0, noise, nPoints)
    df["noise"] += (np.random.random(nPoints)<outFraction)*np.random.normal(0, 2, nPoints)
    df["noiseC"] = np.random.normal(0, 0.01, nPoints)
    df["csin"] = np.sin(6.28 * df["C"])
    df["ccos"] = np.cos(6.28 * df["C"])
    df["csinDistorted"] = np.sin(6.28 * (df["C"] + df["noiseC"]))
    df["valueOrig"] = 3 * df["A"] + np.exp(2 * df["B"]) * df["csin"]
    df["value"] = 3 * df["A"] + np.exp(2 * df["B"]) * df["csinDistorted"] + df["noise"]
    df["expB"] =np.exp(2 * df["B"])
    return df

def makeFits(dfTrain, dfTest):
    # common parameters for test
    n_estimators=100
    n_jobs=8
    max_depth=31
    n_repetitions=25
    sigmaAugment = np.array([0.01, 0.01, 0.005, 2.0])
    #
    fitter1 = RandomForestRegressor(n_estimators,max_depth=max_depth,n_jobs=n_jobs)
    fitterAugmented =  AugmentedRandomForestArray(100, n_repetitions=n_repetitions, n_jobs=n_jobs,max_depth=max_depth)
    XTrain = dfTrain[["A","B","C","D"]].to_numpy()
    yTrain = dfTrain["value"].to_numpy()
    fitter1.fit(XTrain,yTrain)
    fitterAugmented.fit(XTrain,yTrain,sigmaAugment)
    XTest = dfTest[["A","B","C","D"]].to_numpy().astype(np.float32)
    dfFit = dfTest.copy()
    #dfFit["valuePred"] = fitter1.predict(XTest)
    dfFit["valuePred"] = predictRFStat(fitter1, XTest, {"mean":[]}, n_jobs)["mean"]
    dfFit["n_trees"] = 100
    dfFit["fitter"] = 0
    dfFitAugmented = dfTest.copy()
    dfFitAugmented["valuePred"] = fitterAugmented.predict(XTest)
    dfFitAugmented["n_trees"] = 100
    dfFitAugmented["fitter"] = 1
    dfNew = pd.concat([dfFit, dfFitAugmented])
    return dfNew, fitter1, fitterAugmented

def makeDashboard(df):
    output_file("test_histogramTemplateMultiYDiff.html")
    aliasArray, jsFunctionArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVarsNormAll(variables=["A", "B", "C", "D", "value", "valueOrig", "valuePred"], weights=["fitter==0", "fitter==1", "1"], multiAxis="weights")
    widgetsSelect = [
        ['range', ['A'], {"name":"A"}],
        ['range', ['B'], {"name":"B"}],
        ['range', ['C'], {"name":"C"}],
        ['range', ['D'], {"name":"D"}],
        ['multiSelect', ['fitter'], {"name":"fitter"}],
        ]
    widgetParams = mergeFigureArrays(widgetParams, widgetsSelect)
    widgetLayoutDesc["Select"] = [["A","B"],["C","D"],["fitter"]]
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray, aliasArray=aliasArray, arrayCompression=arrayCompressionRelative16,
                          jsFunctionArray=jsFunctionArray)

def test_augmentedForest():
    dfTrain = generateInput(100000)
    dfTest = generateInput(100000)

    dfNew, fitter1, fitterAugmented = makeFits(dfTrain, dfTest)
    makeDashboard(dfNew)