import pandas as pd
import numpy as np
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.MLpipeline.NDFunctionInterface import *
from bokeh.io import output_notebook


def generateInput(nPoints):
    """
    Generate random panda+tree random vectors A,B,C,D
        * generate function value = A+exp(3B)+sin(6.28C)
        * generate noise vector
    """
    df = pd.DataFrame(np.random.random_sample(size=(nPoints, 4)), columns=list('ABCD'))
    #df["noise"] = np.random.normal(0, 0.05, nPoints) + np.random.normal(0, 0.5, nPoints)*(np.random.random(nPoints)>0.8)
    df["noise"] = np.random.normal(0, 0.05, nPoints)
    df["csin"] = np.sin(6.28 * df["C"])
    df["valueOrig"] = 3*df["A"] + np.exp(3 * df["B"]) * df["csin"]
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
    figureLayoutDesc=[
        [0,1],
        {'plot_height': 200, 'sizing_mode': 'scale_width'}
    ]
    fig = bokehDrawSA.fromArray(df, "A>0", figureArray, widgets, tooltips=tooltips, layout=figureLayoutDesc)

def makeFits(df):
    varFit='value'
    variableX= ['A',"B", "C",'D']
    nPoints=df.shape[0]
    dataContainer = DataContainer(df, variableX, varFit, [nPoints//2,nPoints//2])
    fitter = Fitter(dataContainer)
    fitter.Register_Model('KNN',KNeighbors('Regressor'))
    #fitter.Register_Method('KNN','KNeighbors', 'Regressor')
    fitter.Register_Model('RF',RandomForest('Regressor',n_estimators=100, max_depth=12))
    #fitter.Register_Method('RF','RandomForest', 'Regressor', n_estimators=100, max_depth=10)
    fitter.Register_Model('RF200',RandomForest('Regressor',n_estimators=200, max_depth=12))
    fitter.Register_Model('GBR',GradientBoostingRegressorModel(n_estimators=200,max_depth=12))
    fitter.Register_Model('RFQuant',RandomForest('RegressorQuantile',n_estimators=200, max_depth=10))

    #fitter.Register_Method('RF200','RandomForest', 'Regressor', n_estimators=200, max_depth=10)
    #fitter.Register_Method('KM','KerasModel', 'Regressor', layout = [50, 50, 50], epochs=100, dropout=0.2)
    fitter.Fit()
    fitter.Fit()
    return fitter


def appendFit(df, fitter):
    #fitter.Compress('KM')
    for method in ['RF', 'KNN', 'RF200','RFQuant', "GBR"]:
        df = fitter.AppendOtherPandas(method,df)
        fitter.AppendStatPandas("RF",df)
        fitter.AppendStatPandas("RF200",df)
        df=SetAlias(df,"pullRF","(RF-value)/RFRMS")
    return df

def drawFit(df, fitter):
    output_file("test_NDRegressionOutput.html")
    tooltips=[("A","@A"), ("B","@B"), ("C","@C")]
    figureArray= [
        [['csin'], ['RF200-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['csin'], ['RF-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['csin'], ['GBR-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['csin'], ['KNN-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['RF200-valueOrig'], ['RF200-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['RF200-valueOrig'], ['RF-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['RF200-valueOrig'], ['GBR-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['RF200-valueOrig'], ['KNN-valueOrig'], {"size": 2, "colorZvar":"B"}],
        [['RF200-valueOrig'], ['RF200-RF200'], {"size": 2, "colorZvar":"B"}],
        [['RF200-valueOrig'], ['RF-RF200'], {"size": 2, "colorZvar":"B"}],
        [['RF200-valueOrig'], ['GBR-RF200'], {"size": 2, "colorZvar":"B"}],
        [['RF200-valueOrig'], ['KNN-RF200'], {"size": 2, "colorZvar":"B"}],
        ['table']
    ]
    widgetParams=[
        ['range', ['A']],
        ['range', ['B']],
        ['range', ['C']],
        ['range', ['csin']],
    ]
    widgetLayoutDesc=[[0, 1], [2, 3], {'sizing_mode': 'scale_width'}]
    figureLayoutDesc=[
        [0,1,2,3,{"commonY":0,"commonX":0}],
        [4,5,6,7,{"commonY":4}],
        [8,9,10,11,{"commonY":8}],
        {"commonY":1, "commonX":4,'plot_height': 250, 'sizing_mode': 'scale_width', "plot_width":1400}
    ]
    output_file("NDFunctionInterace_1.html")
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray,widgetParams,tooltips=tooltips, layout=figureLayoutDesc, widgetLayout=widgetLayoutDesc)

df = generateInput(4000)
fitter=makeFits(df)
df=appendFit(df,fitter)
print(df.columns)
drawInput(df.sample(2000))
drawFit(df.sample(2000),fitter)
fitter.printImportance(reverse=True,nLines=3)
fitter.printImportance(reverse=False,nLines=3)
