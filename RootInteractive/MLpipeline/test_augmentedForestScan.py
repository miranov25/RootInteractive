"""
This code is non inteted to be used for the Unit test as it takes too long time to run.
It is used for the integration test in the MLpipeline, to test Augment Kernel Random forest and augmented kernel xgboost
from RootInteractive.MLpipeline.test_augmentedForestScan import *
"""
import numpy as np
import pandas as pd
from bokeh.io import output_file
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
from RootInteractive.InteractiveDrawing.bokeh.bokehTools import mergeFigureArrays
from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate import getDefaultVarsNormAll
from RootInteractive.Tools.compressArray import arrayCompressionRelative16
from RootInteractive.MLpipeline.augmentedForest import AugmentedRandomForestArray, makeAugmentedRF,makeAugmentXGBoost
from RootInteractive.MLpipeline.MIForestErrPDF import predictRFStat, predictRFStatNew
from sklearn.ensemble import RandomForestRegressor
import logging
import psutil
import os
import xgboost as xgb

# Optional: Configure logging as you prefer
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_resource_usage(phase: str):
    """
    Logs memory usage (RSS) and CPU times at a given phase in the workflow.
    """
    process = psutil.Process(os.getpid())
    # Memory in MB
    mem_rss_mb = process.memory_info().rss / (1024 * 1024)
    # CPU times (user + system); alternatively, you could use cpu_percent(interval=1.0) for instantaneous usage
    cpu_times = process.cpu_times()
    logging.info(
        f"[{phase}] Memory RSS: {mem_rss_mb:.2f} MB | "
        f"CPU Times: user={cpu_times.user:.2f}s, system={cpu_times.system:.2f}s"
    )

def generateInput(nPoints, outFraction=0.2, noise=1):
    """
    Generate random panda+tree random vectors A,B,C,D
        * generate function value = A+exp(3B)*sin(6.28C)
        * generate noise vector
    """
    df = pd.DataFrame(np.random.random_sample(size=(nPoints, 4)), columns=list('ABCD'))
    df["noise"] = np.random.normal(0, noise, nPoints)
    df["noise"] += (np.random.random(nPoints)<outFraction)*np.random.normal(0, 5*noise, nPoints)
    df["noiseC"] = np.random.normal(0, 0.01, nPoints)
    df["csin"] = np.sin(6.28 * df["C"])
    df["ccos"] = np.cos(6.28 * df["C"])
    df["csinDistorted"] = np.sin(6.28 * (df["C"] + df["noiseC"]))
    df["valueOrig"] = 3 * df["A"] + np.exp(2 * df["B"]) * df["csin"]
    df["value"] = 3 * df["A"] + np.exp(2 * df["B"]) * df["csinDistorted"] + df["noise"]
    df["expB"] =np.exp(2 * df["B"])
    return df

def create_xgb_models(n_models=20):
    xgb_params = {
        'objective': 'reg:squarederror',  # Regression task
        'n_estimators': 20,              # Number of boosting rounds
        'max_depth': 5,                   # Maximum depth of trees
        'learning_rate': 0.2,             # Step size shrinkage
        'subsample': 0.5,                 # Row sampling
        'colsample_bytree': 0.8,          # Feature sampling per tree
        'reg_lambda': 0,                  # L2 regularization
        'reg_alpha': 1                    # L1 regularization
    }
    # Create a list of XGBRegressor models with the same parameters
    xgb_models = [xgb.XGBRegressor(**xgb_params) for _ in range(n_models)]
    return xgb_models

def makeFits(dfTrain, dfTest):
    log_resource_usage("Start makeFits")
    # Common parameters for the test
    n_estimators = 1000
    n_jobs = 8
    max_depth = 31
    n_repetitions = 50
    max_samples=1./n_estimators
    sigmaAugment = np.array([0.05, 0.01, 0.01, 2.0])

    # Instantiate models
    fitter1 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs,max_samples=max_samples)
    #fitterAugmented = AugmentedRandomForestArray(   n_estimators,n_repetitions=n_repetitions,n_jobs=n_jobs,max_depth=max_depth,max_samples = max_samples)
    # Prepare data
    XTrain = dfTrain[["A", "B", "C", "D"]].to_numpy()
    yTrain = dfTrain["value"].to_numpy()
    # Fit RandomForest
    log_resource_usage("Before fitting standard RF")
    fitter1.fit(XTrain, yTrain)
    log_resource_usage("After fitting standard RF")
    # Fit AugmentedRandomForest
    #log_resource_usage("Before fitting augmented RF")
    #fitterAugmented.fit(XTrain, yTrain, sigmaAugment)
    #log_resource_usage("After fitting augmented RF")
    #
    rf0 = RandomForestRegressor(n_estimators =n_estimators,n_jobs=n_jobs,max_depth=max_depth,max_samples=max_samples)
    rf1 = RandomForestRegressor(n_estimators =n_estimators,n_jobs=n_jobs,max_depth=max_depth,max_samples=max_samples)
    makeAugmentedRF(XTrain,yTrain,[rf0,rf1],n_repetitions,sigmaAugment,0.01)
    #
    # Prepare test data & predictions
    log_resource_usage("Before predictions")
    XTest = dfTest[["A", "B", "C", "D"]].to_numpy().astype(np.float32)
    dfFit = dfTest.copy()
    predFit= predictRFStatNew(fitter1, XTest, {"mean": [],"median":[],"std":[]}, n_jobs)
    dfFit["valuePredRFMean"] = predFit["mean"]
    dfFit["valuePredRFMedian"] = predFit["median"]
    dfFit["valuePredRFStd"] = predFit["std"]
    dfFit["n_trees"] = n_estimators
    #
    #log_resource_usage("Before predict augmented RF")
    #predFit= fitterAugmented.predictRFStat(XTest,{"mean": [],"median":[],"std":[]})
    #dfFit["valuePredAMean"] = predFit["mean"]
    #dfFit["valuePredAMedian"] = predFit["median"]
    ##dfFit["valuePredAStd"] = predFit["std"]
    #log_resource_usage("After predict augmented RF")
    predFit= predictRFStatNew(rf0, XTest, {"mean": [],"median":[],"std":[]}, n_jobs)
    dfFit["valuePredAF0Mean"] = predFit["mean"]
    dfFit["valuePredAF0Median"] = predFit["median"]
    dfFit["valuePredAF0Std"] = predFit["std"]
    predFit= predictRFStatNew(rf1, XTest, {"mean": [],"median":[],"std":[]}, n_jobs)
    dfFit["valuePredAF1Mean"] = predFit["mean"]
    dfFit["valuePredAF1Median"] = predFit["median"]
    dfFit["valuePredAF1Std"] = predFit["std"]
    dfFit["valuePredAFMean"] = (dfFit["valuePredAF0Mean"]+dfFit["valuePredAF1Mean"])/2
    dfFit["valuePredAFMedian"] = (dfFit["valuePredAF0Median"]+dfFit["valuePredAF1Median"])/2
    #
    log_resource_usage("End makeFits")
    return dfFit, fitter1, rf0,rf1


def makeDashboard(df):
    output_file("test_histogramTemplateMultiYDiff.html")
    variables=sorted(list(df.columns))
    aliasArray, jsFunctionArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVarsNormAll(variables=variables)
    widgetsSelect = [
        ['range', ['A'], {"name":"A"}],
        ['range', ['B'], {"name":"B"}],
        ['range', ['C'], {"name":"C"}],
        ['range', ['D'], {"name":"D"}],
        #['multiSelect', ['fitter'], {"name":"fitter"}],
        ]
    widgetParams = mergeFigureArrays(widgetParams, widgetsSelect)
    widgetLayoutDesc["Select"] = [["A","B"],["C","D"]]
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray, aliasArray=aliasArray, arrayCompression=arrayCompressionRelative16,
                          jsFunctionArray=jsFunctionArray)

def printStat(df):
    """
    Print statistical summaries comparing standard Random Forest (RF) and Augmented Random Forest (AF) with robust statistics.
    This function evaluates and prints the standard deviation and median of predicted values compared to the original values.
    It highlights the performance differences between non-augmented and augmented models.
    Key comparisons include:
    - Standard deviation of residuals (difference between predicted and original values).
    - Median of predicted standard deviations.
    - Pull distributions, assessing how well the models' predictions align with the original values.
    Parameters:
    df : pandas.DataFrame
        DataFrame containing columns for predicted values (mean and median) and original values.

    Columns required:
    - valuePredRFMean: Predicted mean from Random Forest.
    - valuePredAFMean: Predicted mean from Augmented Forest.
    - valuePredRFMedian: Predicted median from Random Forest.
    - valuePredAFMedian: Predicted median from Augmented Forest.
    - valueOrig: Original true values.
    - valuePredRFStd: Standard deviation of predictions from Random Forest.
    - valuePredAF1Std: Standard deviation of predictions from Augmented Forest.

    Returns:
    None
    """
    print("Performance Comparison: Random Forest vs Augmented Random Forest with Robust Statistics")
    print("Goal: Pull values should be close to 1 for medians and ~1.2 for means.")
    print("Non-augmented RF tends to underperform compared to the augmented version.")
    print("Performance degradation depends on statistical methods and the number of trees.")

    # Standard deviation comparisons
    print("\nStandard Deviation of Residuals (Predicted - Original):")
    print("RF Mean:", df.eval("valuePredRFMean - valueOrig").std())
    print("AF Mean:", df.eval("valuePredAFMean - valueOrig").std())
    print("RF Median:", df.eval("valuePredRFMedian - valueOrig").std())
    print("AF Median:", df.eval("valuePredAFMedian - valueOrig").std())

    # Median of predicted standard deviations
    print("\nMedian of Predicted Standard Deviations:")
    print("RF Std:", df.eval("valuePredRFStd").median())
    print("AF Std:", df.eval("valuePredAF1Std").median())

    # Pull calculations
    print("\nPull Distributions:")
    print("Pull (RF Median):", df.eval("(valuePredRFMedian - valueOrig) / (valuePredRFStd / sqrt(1000))").std())
    print("Pull (AF Median):", df.eval("(valuePredAFMedian - valueOrig) / (valuePredAF1Std / sqrt(1000))").std())
    print("Pull (AF Mean):", df.eval("(valuePredAFMean - valueOrig) / (valuePredAF1Std / sqrt(1000))").std())


def makeXGB(df,n_models=10,max_depth=5,subsample=0.5,nRepetitions=20,learning_rate=0.2):
    # n_models=10;max_depth=5;subsample=0.5;nRepetitions=20;learning_rate=0.2
    n_estimators = int(2./(0.02+learning_rate))
    xgb_params = {
        'objective': 'reg:squarederror',  # Regression task
        'n_estimators': n_estimators,     # Number of boosting rounds
        'max_depth': max_depth,           # Maximum depth of trees
        'learning_rate': learning_rate,   # Step size shrinkage
        'subsample': subsample,           # Row sampling
        'colsample_bytree': 0.8,          # Feature sampling per tree
        'reg_lambda': 0,                  # L2 regularization
        'reg_alpha': 0,                   # L1 regularization
        "n_jobs":-2,
            #        'tree_method': 'hist'             # Fast histogram method
    }
    # Create a list of XGBRegressor models with the same parameters
    xgb_models = [xgb.XGBRegressor(**xgb_params) for _ in range(n_models)]
    #xgb_models = create_xgb_models(100)
    XTrain = df[["A", "expB", "csin", "D"]].to_numpy()
    yTrain = df["value"].to_numpy()
    sigmaAugment = np.array([0.025, 0.1, 0.01, 2.0])
    xgb2,pred,df_hist,xgb_std,xgb_median = makeAugmentXGBoost(XTrain,yTrain,xgb_models,nRepetitions,sigmaAugment,0.05,0.0001,15,df["valueOrig"])
    df_hist[f"max_depth"]=max_depth
    df_hist["subsample"]=subsample
    df_hist["nRepetitions"]=nRepetitions
    df_hist["learning_rate"]=learning_rate
    df_hist["n_models"]=n_models
    df_hist.to_csv(f"xgbhist_{max_depth}_nM{n_models}_s{int(subsample*100)}_nR{nRepetitions}_l{int(learning_rate*100)}.csv")
    df[f"xgbD_D{max_depth}_nM{n_models}_s{int(subsample*100)}_nR{nRepetitions}_l{int(learning_rate*100)}"]=pred
    df[f"xgbS_D{max_depth}_nM{n_models}_s{int(subsample*100)}_nR{nRepetitions}_l{int(learning_rate*100)}"]=xgb_std
    df[f"xgbM_D{max_depth}_nM{n_models}_s{int(subsample*100)}_nR{nRepetitions}_l{int(learning_rate*100)}"]=xgb_median
    return df, xgb2,pred,df_hist

def makeXGBParamScan(df):
    df_histAll=None
    for scan in range(4):
        n_models=np.random.randint(1, 50)   # number of xgb models used
        max_depth=np.random.randint(4, 7)            # deepnes of the tree
        nRepetitions=np.random.randint(1, 50)        # number of repetitions for the augmented forest
        subsample=2*(0.5*np.random.random()+0.1)/(1+np.sqrt(nRepetitions))      # fraction of the data used for training
        learning_rate=0.3*(np.random.random()+0.5)/max_depth   # learning rate for the xgb inversely proportional to the depth
        #
        logging.info(f"Param scan: {scan}, n_models={n_models}, max_depth={max_depth}, subsample={subsample}, nRepetitions={nRepetitions}, learning_rate={learning_rate}")
        df,xgb2,pred,df_hist=makeXGB(df,n_models,max_depth,subsample,nRepetitions,learning_rate)
        if df_histAll is None:
            df_histAll=df_hist
        else:
            df_histAll=pd.concat([df_histAll,df_hist])
        df_histAll.to_csv("xgbhistAll.csv")
        df_histAll.to_pickle("xgbhistAll.pkl")
        df.to_pickle("xgbdf.pkl")
    df_histAll.to_csv("xgbhistAll.csv")

def printScanXGB(df_histAll):
    df_histAll["rmsN"]=df_histAll.eval("RMS/sqrt(n_models)")
    df_histAll["rmsRatio"]=df_histAll.eval("rmsN/Prediction_Std_T")
    df_histAll.query("isLast&max_depth==5").sort_values("Prediction_Std_T",ascending=False)[["Prediction_Std_T","rmsN","Prediction_Std","n_models","nRepetitions","max_depth","learning_rate","subsample"]]


def test_augmentedForest(nPoints=10000):
    dfTrain = generateInput(nPoints)
    dfTest = generateInput(nPoints)
    dfNew, fitter1, rf0, rf1 = makeFits(dfTrain, dfTest)
    makeDashboard(dfNew)
    printStat(dfNew)
    return dfNew