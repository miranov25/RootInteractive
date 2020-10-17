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
methods = ['RF', 'RF10','RF14', "GBR4", "GBR", 'GBR8', "GBR10"]


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
    df["value"] = df["valueOrig"] + df["D"]*df["noise"]
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
    fitter.Register_Model('GBR', GradientBoostingPredictionIntervals(
        {"n_bootstrap": 50, "n_bootstrap_iter": 50, "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9], "n_jobs": -1},
        n_estimators=50, max_depth=6,subsample=0.8,learning_rate=0.25))
    fitter.Register_Model('GBR4', GradientBoostingPredictionIntervals(
        {"n_bootstrap": 50, "n_bootstrap_iter": 50, "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9], "n_jobs": -1},
        n_estimators=50, max_depth=4,subsample=0.8,learning_rate=0.25))
    fitter.Register_Model('GBR8', GradientBoostingPredictionIntervals(
        {"n_bootstrap": 50, "n_bootstrap_iter": 50, "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9], "n_jobs": -1},
        n_estimators=50, max_depth=8,subsample=0.8,learning_rate=0.25))
    fitter.Register_Model('GBR10', GradientBoostingPredictionIntervals(
        {"n_bootstrap": 50, "n_bootstrap_iter": 50, "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9], "n_jobs": -1},
        n_estimators=50, max_depth=10,subsample=0.8,learning_rate=0.25))
    fitter.Register_Model('RF', RandomForest('Regressor', n_estimators=1600, max_depth=None, n_jobs=-1))
    fitter.Register_Model('RF10', RandomForest('Regressor', n_estimators=1600, max_depth=10, n_jobs=-1))
    fitter.Register_Model('RF14', RandomForest('Regressor', n_estimators=1600, max_depth=14, n_jobs=-1))
    # fitter.Register_Model('RFQuant',RandomForest('RegressorQuantile',n_estimators=200, max_depth=10))
    fitter.Fit()
    return fitter


def appendFit(df, fitter):
    for method in methods:
        df = fitter.AppendOtherPandas(method, df)
        fitter.AppendStatPandas(method, df)
        df = SetAlias(df, "fpull" + method, "({}-valueOrig)/{}RMS".format(method, method))
        df = SetAlias(df, "fdelta" + method, "({}-valueOrig)".format(method))
        df = SetAlias(df, "fdeltaMean" + method, "({}Mean-valueOrig)".format(method))
        df = SetAlias(df, "fdeltaMedian" + method, "({}Median-valueOrig)".format(method))
        df = SetAlias(df, "pull" + method, "({}-value)/{}RMS".format(method, method))
        df = SetAlias(df, "delta" + method, "({}-value)".format(method))
        df = SetAlias(df, "deltaMean" + method, "({}Mean-value)".format(method))
        df = SetAlias(df, "deltaMedian" + method, "({}Median-value)".format(method))
    return df


def drawFits(df):
    output_file("test_NDRegressionFit.html")
    figureArray = [
        [['A'], ['valueOrig'], {"size": 2, "colorZvar": "csin"}],
        [['B'], ['valueOrig'], {"size": 2, "colorZvar": "csin"}],
        [['csin'], ['valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['GBRQ_70-GBRQ_30'], ['valueOrig-valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['A'], ['RF'], {"size": 2, "colorZvar": "csin"}],
        [['B'], ['RF'], {"size": 2, "colorZvar": "csin"}],
        [['csin'], ['RF'], {"size": 2, "colorZvar": "B"}],
        [['RF14RMS'], ['valueOrig-RF'], {"size": 2, "colorZvar": "B"}],
        [['A'], ['GBR4'], {"size": 2, "colorZvar": "csin"}],
        [['B'], ['GBR4'], {"size": 2, "colorZvar": "csin"}],
        [['csin'], ['GBR4'], {"size": 2, "colorZvar": "B"}],
        [['RF14RMS'], ['valueOrig-GBR4'], {"size": 2, "colorZvar": "B"}],
        [['A'], ['GBR8'], {"size": 2, "colorZvar": "csin"}],
        [['B'], ['GBR8'], {"size": 2, "colorZvar": "csin"}],
        [['csin'], ['GBR8'], {"size": 2, "colorZvar": "B"}],
        [['RF14RMS'], ['valueOrig-GBR8'], {"size": 2, "colorZvar": "B"}],
        ['table']
    ]
    figureLayoutDesc = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],

        {'plot_height': 200, 'sizing_mode': 'scale_width', "plot_width": 1800}
    ]
    fig = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, tooltips=tooltips, layout=figureLayoutDesc,
                                widgetLayout=widgetLayoutDesc)


def drawRMS(df):
    output_file("test_NDRegressionRMS.html")
    figureArray = [
        [['B'], ['RFRMS'], {"size": 2, "colorZvar": "ccos"}],
        [['B'], ['RF14RMS'], {"size": 2, "colorZvar": "ccos"}],
        [['B'], ['GBR4RMS'], {"size": 2, "colorZvar": "ccos"}],
        [['B'], ['GBRRMS'], {"size": 2, "colorZvar": "ccos"}],
        [['B'], ['GBR8RMS'], {"size": 2, "colorZvar": "ccos"}],
        [['B'], ['GBR10RMS'], {"size": 2, "colorZvar": "ccos"}],
        [['RFRMS'], ['fdeltaRF'], {"size": 2, "colorZvar": "ccos"}],
        [['RF14RMS'], ['fdeltaRF14'], {"size": 2, "colorZvar": "ccos"}],
        [['GBR4RMS'], ['fdeltaGBR4'], {"size": 2, "colorZvar": "ccos"}],
        [['GBRRMS'], ['fdeltaGBR'], {"size": 2, "colorZvar": "ccos"}],
        [['GBR8RMS'], ['fdeltaGBR8'], {"size": 2, "colorZvar": "ccos"}],
        [['GBR10RMS'], ['fdeltaGBR10'], {"size": 2, "colorZvar": "ccos"}],
        ['table']
    ]
    figureLayoutDesc = [
        [0, 1, 2, 3, 4, 5,{"commonX":0,"commonY":0}],
        [6, 7, 8, 9, 10, 11, {"commonX":6,"commonY":6}],
        {'plot_height': 250, 'sizing_mode': 'scale_width', "plot_width": 1800}
    ]
    bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, tooltips=tooltips, layout=figureLayoutDesc,
                                widgetLayout=widgetLayoutDesc)


def drawFitDelta(df):
    output_file("test_NDRegressionDelta.html")
    figureArray = [
        [['csin'], ['RF14-valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['csin'], ['RF-valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['csin'], ['GBR4-valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['csin'], ['GBR-valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['csin'], ['GBR8-valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['RF14-valueOrig'], ['RF14-valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['RF14-valueOrig'], ['RF-valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['RF14-valueOrig'], ['GBR4-valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['RF14-valueOrig'], ['GBR-valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['RF14-valueOrig'], ['GBR8-valueOrig'], {"size": 2, "colorZvar": "B"}],
        [['RF14-valueOrig'], ['RF14-RF14'], {"size": 2, "colorZvar": "B"}],
        [['RF14-valueOrig'], ['RF-RF14'], {"size": 2, "colorZvar": "B"}],
        [['RF14-valueOrig'], ['GBR4-RF14'], {"size": 2, "colorZvar": "B"}],
        [['RF14-valueOrig'], ['GBR-RF14'], {"size": 2, "colorZvar": "B"}],
        [['RF14-valueOrig'], ['GBR8-RF14'], {"size": 2, "colorZvar": "B"}],
        ['table']
    ]
    figureLayoutDesc = [
        [0, 1, 2, 3, 4, {"commonY": 0, "commonX": 0}],
        [5, 6, 7, 8, 9, {"commonY": 5}],
        [10, 11, 12, 13, 14, {"commonY": 10}],
        {"commonY": 1, "commonX": 4, 'plot_height': 250, 'sizing_mode': 'scale_width', "plot_width": 1800}
    ]
    output_file("NDFunctionInterace_1.html")
    fig = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, tooltips=tooltips, layout=figureLayoutDesc,
                                widgetLayout=widgetLayoutDesc)

def drawFitRF(df):
    output_file("test_NDRegressionDeltaMean.html")
    figureArray = [
        [['RFRMSMean'], ['fdeltaRF'], {"size": 2, "colorZvar": "B"}],
        [['RFRMSMean'], ['deltaRF'], {"size": 2, "colorZvar": "B"}],
        [['RFRMS'], ['fdeltaRF'], {"size": 2, "colorZvar": "B"}],
        [['RFRMS'], ['deltaRF'], {"size": 2, "colorZvar": "B"}],
    ]
    figureLayoutDesc = [
        [0, 1, {"commonY": 0, "commonX": 0}],
        [2, 3, {"commonY": 2}],
        {"commonY": 1, "commonX": 0, 'plot_height': 250, 'sizing_mode': 'scale_width', "plot_width": 1800}
    ]
    output_file("NDFunctionInterace_1.html")
    fig = bokehDrawSA.fromArray(df.sample(2000), "A>0", figureArray, widgetParams, tooltips=tooltips, layout=figureLayoutDesc,
                                widgetLayout=widgetLayoutDesc)


def test_NDRegression(nPoints=50, generate=True, n_jobs=-1):
    nDraw=min(nPoints,2000)
    if generate:
        df = generateInput(nPoints)
        fitter = makeFits(df,n_jobs)
        df = appendFit(df, fitter)
        # fitter.printImportance(reverse=True,nLines=3)
        fitter.printImportance(reverse=False, nLines=4)
        #fitter.to_pickle("test_NDRegressionFit.pkl")
        df.to_pickle("test_NDRegression.pkl")
        pickle.dump(fitter, file = open("test_NDRegressionFit.pkl", "wb"))
    else:
        df = pd.read_pickle("test_NDRegression.pkl")
    print(df.columns)
    drawFits(df.sample(nDraw))
    drawRMS(df.sample(nDraw))
    drawFitDelta(df.sample(nDraw))

    print("pullRF:", df["pullRF"].mean(), df["pullRF"].std())
    print("pullRF14:", df["pullRF14"].mean(), df["pullRF14"].std())
    print("pullGBR4:", df["pullGBR4"].mean(), df["pullGBR4"].std())
    print("pullGBR:", df["pullGBR"].mean(), df["pullGBR"].std())
    print("pullGBR8:", df["pullGBR8"].mean(), df["pullGBR8"].std())
    print("pullGBR10:", df["pullGBR10"].mean(), df["pullGBR10"].std())
    #
    print("rmsRF:", df["RFRMS"].mean(), df["RFRMS"].std())
    print("rmsRF14:", df["RF14RMS"].mean(), df["RF14RMS"].std())
    print("rmsGBR4:", df["GBR4RMS"].mean(), df["GBR4RMS"].std())
    print("rmsGBR:", df["GBRRMS"].mean(), df["GBRRMS"].std())
    print("rmsGBR8:", df["GBR8RMS"].mean(), df["GBR8RMS"].std())
    print("rmsGBR10:", df["GBR10RMS"].mean(), df["GBR10RMS"].std())
    print(df.columns)
    printSummary(df)
    return df
    # for method in ['RF', 'RF14',"GBR4", "GBR", 'GBR8',  "GBR10"]:
    # print("rmsRF:",df["{}"].mean(),df["RFRMS"].std())


def printSummary(df):
    # df=pd.read_pickle("test_NDRegression.pkl")
    #methods = ['RF', 'RF14', "GBR4", "GBR", 'GBR8', "GBR10"]
    print(df.std()[["delta{}".format(x) for x in methods]])
    print(df.std()[["deltaMean{}".format(x) for x in methods]])
    print(df.mean()[["{}RMS".format(x) for x in methods]])
    print(df.std()[["fdelta{}".format(x) for x in methods]])
    print(df.std()[["fdeltaMean{}".format(x) for x in methods]])
    print(df.std()[["fdeltaMedian{}".format(x) for x in methods]])


#    df[df.std()[ ["delta{}".format(x) for x in methods] ]].describe().to_markdown()

def printSummary2():
    dfOut = test_NDRegression(20000, True,32)
    dfOut.to_pickle("test_NDRegression.pkl")
    dfReport = pd.DataFrame(columns=['fit', 'deltaMean', 'deltaSTD', 'fdeltaMean', 'fdeltaSTD', 'fpullMean', 'fpullStd'])
    dfReport = pd.DataFrame(columns=[])
    # dfReport['fit']=["delta{}".format(x) for x in methods]
    dfReport['deltaMean'] = dfOut[["delta{}".format(x) for x in methods]].mean()
    dfReport['deltaSTD'] = dfOut[["delta{}".format(x) for x in methods]].std()
    dfReport['fdeltaMean'] = dfOut[["fdelta{}".format(x) for x in methods]].mean()
    dfReport['fdeltaSTD'] = dfOut[["fdelta{}".format(x) for x in methods]].std()
    dfReport['fpullMean'] = dfOut[["fpull{}".format(x) for x in methods]].mean()
    dfReport['fpullSTD'] = dfOut[["fpull{}".format(x) for x in methods]].std()

#logging.basicConfig(level=logging.DEBUG,  format="%(asctime)s %(levelname)s %(filename)s:%(funcName)s%(message)s")
#test_NDRegression()