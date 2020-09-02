import pandas as pd
import numpy as np
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.MLpipeline.NDFunctionInterface import DataContainer, Fitter
from bokeh.io import output_notebook


def generateInput(nPoints):
    """
    Generate random panda+tree random vectors A,B,C,D
        * generate function value = A+exp(3B)+sin(6.28C)
        * generate noise vector
    """
    df = pd.DataFrame(np.random.random_sample(size=(nPoints, 4)), columns=list('ABCD'))
    df["noise"] = np.random.normal(0, 0.1, nPoints)
    df["csin"] = np.sin(6.28 * df["C"])
    df["valueOrig"] = df["A"] + np.exp(3 * df["B"]) * df["csin"]
    df["value"] = df["valueOrig"] + df["noise"]
    return df

def drawInput(df):
    output_file("test_NDRegressionInput.html")
    tooltips = [("A", "@A"), ("B", "@B"), ("C", "@C")]
    figureArray = [
        [['A'], ['valueOrig'], {"size": 2, "colorZvar": "csin"}],
        [['B'], ['valueOrig'], {"size": 2, "colorZvar": "csin"}],
        ['table']
    ]
    widgets = "query.custom(), slider.A(0,1,0.1,0,1), slider.B(0,1,0.1,0,1), slider.C(0,1,0.1,0,1), slider.csin(-1,1,0.1,-1,1)"
    figureLayout: str = '((0,1),(2, plot_height=150),commonY=1, x_visible=1,y_visible=0,plot_height=300,plot_width=1200)'
    fig = bokehDrawSA.fromArray(df, "A>0", figureArray, widgets, tooltips=tooltips, layout=figureLayout)

def makeFits(df):
    varFit='value'
    variableX= ['A',"B", "C",'D']
    nPoints=df.shape[0]
    dataContainer = DataContainer(df, variableX, varFit, [nPoints//2,nPoints//2])
    fitter = Fitter(dataContainer)
    fitter.Register_Method('KNN','KNeighbors', 'Regressor')
    fitter.Register_Method('RF','RandomForest', 'Regressor', n_estimators=100, max_depth=10)
    fitter.Register_Method('RF200','RandomForest', 'Regressor', n_estimators=200, max_depth=10)
    #fitter.Register_Method('KM','KerasModel', 'Regressor', layout = [50, 50, 50], epochs=100, dropout=0.2)
    fitter.Fit()
    return fitter

def appendFit(df, fitter):
    #fitter.Compress('KM')
    for method in ['RF', 'KNN', 'RF200']:
        df = fitter.AppendOtherPandas(method,df)
        fitter.AppendStatPandas("RF",df)
        fitter.AppendStatPandas("RF200",df)
        df=SetAlias(df,"pullRF","(RF-value)/RFRMS")

def drawFit(df, fitter):
    tooltips=[("A","@A"), ("B","@B"), ("C","@C")]
    figureArray= [
        [['csin'], ['RF-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['csin'], ['RFMedian-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['csin'], ['KM-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['csin'], ['KNN-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['RF-valueOrig'], ['KNN-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['RF-valueOrig'], ['KM-valueOrig'], {"size": 2, "colorZvar":"B"}],
        ['table']
    ]
    widgets="query.custom(), slider.A(0,1,0.1,0,1), slider.B(0,1,0.1,0,1), slider.C(0,1,0.1,0,1)"
    figureLayout: str = '((0,1,2,3),(4,5, commonX=4),(6, plot_height=150),commonY=1, commonX=0, x_visible=1,y_visible=0,plot_height=250,plot_width=1400)'
    output_file("NDFunctionInterace_1.html")
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray,widgets,tooltips=tooltips, layout=figureLayout)

df = generateInput(10000)
#drawInput(df)
fitter=makeFits(df)
fitter.printImportance(reverse=True,nLines=3)
fitter.printImportance(reverse=False,nLines=3)
appendFit(df,fitter)