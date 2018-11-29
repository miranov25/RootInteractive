from MLpipeline.NDFunctionInterface import PrepareData, Fitter
from TTreeHnInteractive.TTreeHnBrowser import *
from TTreeHnInteractive.bokehTools import *
from distortionStudy import *

###
inputPath = os.path.expandvars("$NOTESData/JIRA/ATO-336/DistortionsTimeSeries/distortionAll.csv")
df = readDataFrame(inputPath)
dfSplit = splitDistortionFrame(df)
print("load csv file", input, df.shape, dfSplit.shape)
dfSplit.head(3)
dfSplit = dfSplit.query("iz2x>-1")
dfSplit.dropna(inplace=True)
###
deltaColumns = [col for col in dfSplit.columns if "drphiSec" in col]
factorColumns = ['bz', 'bckg0Mean', 'iz2x']
variableX = [x for x in deltaColumns if x != 'drphiSector2']
variableX += factorColumns
#
x = PrepareData(dfSplit, variableX, ['drphiSector2'], [500, 500])
fitter = Fitter(x)  # type: Fitter 
fitter.Register_Method('KM', 'KerasModel', 'Regressor')
fitter.Register_Method('KNN', 'KNeighbors', 'Regressor')
fitter.Register_Method('RF', 'RandomForest', 'Regressor', n_estimators=100, max_depth=10)
fitter.Register_Method('RF200', 'RandomForest', 'Regressor', n_estimators=200, max_depth=10)
# list(variableX)

##
fitter.Fit()
fitter.Compress('KM')
for method in ['RF', 'KNN', 'RF200', 'KM']:
    fitter.AppendOtherPandas(method, dfSplit)
p = figure(plot_width=600, plot_height=600, title="s2s4")
plot = drawColz(dfSplit, " trdIntMedianL0<30 ", "drphiSector2", "RF", "year", p)

