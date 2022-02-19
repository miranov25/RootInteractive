from MLpipeline.NDFunctionInterface import DataContainer, Fitter
from InteractiveDrawing.bokeh.bokehTools import *
from InteractiveDrawing.bokeh.bokehDrawPanda import *
import sys
import pytest

try:
    import ROOT
except ImportError:
    pytest.skip("ROOT module is not imported", allow_module_level=True)
from distortionStudy import *
from TTreeHnInteractive.TTreeHnBrowser import *

    

def Distortion():
    ###
    inputPath = os.path.expandvars("$NOTESData/JIRA/ATO-336/DistortionsTimeSeries/distortionAll.csv")
    df = readDataFrame(inputPath)
    dfSplit = splitDistortionFrame(df)
    print("load csv file", input, df.shape, dfSplit.shape)
    dfSplit.head(3)
    dfSplit = dfSplit.query("iz2x>-1")
    dfSplit.dropna(inplace=True)
    ###
    assert isinstance(dfSplit, object)
    bokehDrawPanda(dfSplit,"time>0","drphiMean","drphiSector2","gascompH2O","trdMeanCurrent(0,1,0.1,0,1):gascompH2O(0,500,100,100,300)",None,ncols=2)
    exit
    ###
    deltaColumns = [col for col in dfSplit.columns if "drphiNorm" in col]
    factorColumns = ['drphiMean', 'bz', 'bckg0Mean', 'iz2x']
    variableX = [x for x in deltaColumns if (x != 'drphiNorm2') & (x != 'drphiNorm20')]
    variableX += factorColumns

    #
    x = DataContainer(dfSplit, variableX, ['drphiSector2'], [1000, 1000])
    fitter = Fitter(x)  # type: Fitter
    #fitter.Register_Method('KM', 'KerasModel', 'Regressor', layout = [100, 100, 100], loss='mean_absolute_error')
    #fitter.Register_Method('KMLog', 'KerasModel', 'Regressor', layout = [100, 100, 100], loss='mean_squared_logarithmic_error')
    fitter.Register_Method('KNN', 'KNeighbors', 'Regressor')
    fitter.Register_Method('RF', 'RandomForest', 'Regressor',  criterion="mae", n_estimators=100, max_depth=10)
    fitter.Register_Method('RF200', 'RandomForest', 'Regressor', n_estimators=200, max_depth=10)
    # list(variableX)
    print("HALLO WORLD")
    ##
    fitter.Fit()
    #fitter.Compress('KM')
    #

    #for method in ['RF', 'KNN', 'RF200', 'KM' , 'KMLog']:
    for method in ['RF', 'KNN', 'RF200']:
        dfSplit = fitter.AppendOtherPandas(method, dfSplit)
    fitter.AppendStatPandas("RF",dfSplit)

    dfSplit.head(4)
    print(dfSplit.shape)
    p = figure(plot_width=600, plot_height=600, title="s2s4")
    #plot = drawColzArray(dfSplit, " trdIntMedianL0<30 ", "drphiSector2", "RF:KM:KMLog", "year", p, ncols=2)
    plot = drawColzArray(dfSplit, " trdIntMedianL0<30 ", "drphiSector2", "RF", "year", p, ncols=2)
