#!/usr/bin/env python
"""
Python script to create an interactive dashboard for the toy simulation with gas gain, using the macro $NOTES/JIRA/ATO-614/code/toydEdxSimul.C

cd $NOTES/JIRA/ATO-614
export RootInteractive=/u/miranov/github/RootInteractive/
source /lustre/alice/users/miranov/NOTES/alice-tpc-notes2/JIRA/ATO-500/setDefaultEnv.sh
singAliceLocalMI 0
source /envO2Physics
python createSimulDashboards.py

"""

import ROOT
import awkward as ak
import pandas as pd
import numpy as np
import sys

from RootInteractive.Tools.aliTreePlayer import *
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
from RootInteractive.Tools.compressArray import arrayCompressionRelative16
from bokeh.plotting import output_file
from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveParameters import *
from RootInteractive.InteractiveDrawing.bokeh.palette import kBird256, kRainbow256

def loadMacro(macro):
    """
    Function to load the macro.

    :param macro: relative path of the macro.
    """ 
    ROOT.gROOT.LoadMacro(macro)

def simulRDFComplex(numTracks):
    """
    Function to simulate Q with secondary production and gas gain fluctuations using RDataFrame. 

    :param numTracks: number of tracks to simulate
    :return: RDataFrame object
    """ 
    ROOT.initCache(10000000)
    ROOT.EnableImplicitMT(128)
    rdf = ROOT.simulTracksRDFGasGain(numTracks)
    print("Empty RDataFrame is created for the toy simulation.")
    return rdf

def rdfToAwkward(rdf, columnNames=[]):
    """
    Function to convert RDataFrame to Awkward array. 

    :param rdf: simulated RDataFrame object
    :param columnNames: list of columns to be converted to Awkward Array. If not specified, all columns from the dataframe will be converted.
    :return: awkward array
    """ 
    if len(columnNames) == 0:
        columnNames == rdf.columnNames
    array = ak.from_rdataframe(rdf, columns=(columnNames))
    print("Awkward array is created from the RDataFrame object.")
    return array

def awkwardToPandas(array):
    """
    Function to convert Awkward array to Pandas dataframe. 

    :param array: awkward array
    :return: pandas dataframe
    """ 
    df =ak.to_dataframe(array)
    print("Pandas dataframe is created from the awkward array.")
    return df

def getDefaultVars():
    """
    Function to get default RootInteractive variables for the simulated complex data.

    :return: aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc
    """ 
    # defining custom java script function to query  (used later in variable list)
    variables=["funCustom0","funCustom1","funCustom1"]

    aliasArray=[
        {"name": "funCustom0", "variables": [i for i in variables if  "ustom" not in i ],"func":"funCustomForm0",},
        {"name": "funCustom1", "variables": [i for i in variables if   "ustom" not in i], "func":"funCustomForm1",},
        {"name": "funCustom2", "variables": [i for i in variables if   "ustom" not in i], "func":"funcCustomForm2",},
    ]

    parameterArray = [
        # histo vars
        {"name": "nbinsX", "value":100, "range":[10, 200]},
        {"name": "nbinsY", "value":120, "range":[10, 200]},
        {"name": "nbinsZ", "value":5, "range":[1,10]},
        # transformation
        {"name": "exponentX", "value":1, "range":[-5, 5]},
        {'name': "xAxisTransform", "value":None, "options":[None, "sqrt", "lambda x: log(1+x)","lambda x: 1/sqrt(x)", "lambda x: x**exponentX","lambda x,y: x/y" ]},
        {'name': "yAxisTransform", "value":None, "options":[None, "sqrt", "lambda x: log(1+x)","lambda x: 1/sqrt(x)", "lambda x: x**exponentX","lambda x,y: y/x" ]},
        {'name': "zAxisTransform", "value":None, "options":[None, "sqrt", "lambda x: log(1+x)","lambda x: 1/sqrt(x)", "lambda x: x**exponentX" ]},
        # custom selection
        {'name': 'funCustomForm0', "value":"return 1"},
        {'name': 'funCustomForm1', "value":"return 1"},
        {'name': 'funCustomForm2', "value":"return 1"},
        #
        {"name": "sigmaNRel", "value":3.35, "range":[1,5]},
    ]

    parameterArray.extend(figureParameters["legend"]['parameterArray'])   
    parameterArray.extend(figureParameters["markers"]['parameterArray'])    

    widgetParams=[
        # custom selection
        ['textQuery', {"name": "customSelect0","value":"return 1"}],
        ['textQuery', {"name": "customSelect1","value":"return 1"}],
        ['textQuery', {"name": "customSelect2","value":"return 1"}],
        ['text', ['funCustomForm0'], {"name": "funCustomForm0"}],
        ['text', ['funCustomForm1'], {"name": "funCustomForm1"}],
        ['text', ['funCustomForm2'], {"name": "funCustomForm2"}],
        # histogram selection
        ['select', ['varX'], {"name": "varX"}],
        ['select', ['varY'], {"name": "varY"}],
        ['select', ['varYNorm'], {"name": "varYNorm"}],
        ['select', ['varZ'], {"name": "varZ"}],
        ['select', ['varZNorm'], {"name": "varZNorm"}],
        ['slider', ['nbinsY'], {"name": "nbinsY"}],
        ['slider', ['nbinsX'], {"name": "nbinsX"}],
        ['slider', ['nbinsZ'], {"name": "nbinsZ"}],
        # transformation
        ['spinner', ['exponentX'],{"name": "exponentX"}],
        ['spinner', ['sigmaNRel'],{"name": "sigmaNRel"}],
        ['select', ['yAxisTransform'], {"name": "yAxisTransform"}],
        ['select', ['xAxisTransform'], {"name": "xAxisTransform"}],
        ['select', ['zAxisTransform'], {"name": "zAxisTransform"}],
    ]                       

    widgetParams.extend(figureParameters["legend"]["widgets"])
    widgetParams.extend(figureParameters["markers"]["widgets"])

    widgetLayoutDesc={
        "Select": [],
        "Custom":[["customSelect0","customSelect1","customSelect2"],["funCustomForm0","funCustomForm1","funCustomForm2"]],
        "Histograms":[["nbinsX","nbinsY", "nbinsZ", "varX","varY","varYNorm","varZ","varZNorm"], {'sizing_mode': 'scale_width'}],
        "Transform":[["exponentX","xAxisTransform", "yAxisTransform","zAxisTransform"],{'sizing_mode': 'scale_width'}],
        "Legend": figureParameters['legend']['widgetLayout'],
        "Markers":["markerSize"]
    }

    figureGlobalOption={}
    figureGlobalOption=figureParameters["legend"]["figureOptions"]
    figureGlobalOption["size"]="markerSize"
    figureGlobalOption["x_transform"]="xAxisTransform"
    figureGlobalOption["y_transform"]="yAxisTransform"
    figureGlobalOption["z_transform"]="zAxisTransform"

    histoArray=[    
        {
            "name": "histoXYData",
            "variables": ["varX","varY"],
            "nbins":["nbinsX","nbinsY"], "axis":[1],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYNormData",
            "variables": ["varX","varY/varYNorm"],
            "nbins":["nbinsX","nbinsY"], "axis":[1],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYZData",
            "variables": ["varX","varY","varZ"],
            "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYNormZData",
            "variables": ["varX","varY/varYNorm","varZ"],
            "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYZNormData",
            "variables": ["varX","varY","varZ/varZNorm"],
            "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
    ]

    figureArray=[
        # histo XY
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "bin_count", "source":"histoXYData"}],
        [["bin_center_1"], ["bin_count"], { "source":"histoXYData", "colorZvar": "bin_center_0"}],
        [["bin_center_0"], ["mean","quantile_1",], { "source":"histoXYData_1","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYData_1","errY":"std/sqrt(entries)"}],
        # histoXYNorm
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "bin_count", "source":"histoXYNormData"}],
        [["bin_center_1"], ["bin_count"], { "source":"histoXYNormData", "colorZvar": "bin_center_0"}],
        [["bin_center_0"], ["mean","quantile_1",], { "source":"histoXYNormData_1","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYNormData_1","errY":"std/sqrt(entries)"}],
        # histoXYZ
        [["bin_center_0"], ["mean"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["quantile_0"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"2*std/sqrt(entries)"}],
        [["bin_center_0"], ["quantile_1"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"3*std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)"}],
        # histoXYNormZ
        [["bin_center_0"], ["mean"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)","yAxisTitle":"{varY}/{varYNorm}"}],
        [["bin_center_0"], ["quantile_0"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"2*std/sqrt(entries)","yAxisTitle":"{varY}/{varYNorm}"}],
        [["bin_center_0"], ["quantile_1"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"3*std/sqrt(entries)","yAxisTitle":"{varY}/{varYNorm}"}],
        [["bin_center_0"], ["std"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)","yAxisTitle":"{varY}/{varYNorm}"}],
        # histoXYNormZMedian
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "quantile_1", "source":"histoXYZData_2"}],
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "quantile_1", "source":"histoXYZNormData_2"}],
        # histoXYNormZMean
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "mean", "source":"histoXYZData_2"}],
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "mean", "source":"histoXYZNormData_2"}],  
        #
        figureGlobalOption
    ]
    figureLayoutDesc={
        "histoXY":[[0,1],[2,3],{"plot_height":200}],
        "histoXYNorm":[[4,5],[6,7],{"plot_height":200}],
        "histoXYZ":[[8,9],[10,11],{"plot_height":200}],
        "histoXYNormZ":[[12,13],[14,15],{"plot_height":200}],
        "histoXYNormZMedian":[[16,17],{"plot_height":200}],
        "histoXYNormZMean":[[18,19],{"plot_height":200}],
    }

    print("Default RootInteractive variables are defined.")
    return aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc 

def addMetaData(df):
    initMetadata(df)
    df.meta.metaData = {"qVector.AxisTitle": "Q",
                        "nPrimVector.AxisTitle": "Nprim",
                        "nTotVector.AxisTitle": "Ntot",  
                        "dNprimdx.AxisTitle": "dNprim/dx (1/cm)", 
                        "padLength.AxisTitle": "pad length (cm)",
                        #
                        "q2dNprimdx.AxisTitle": "Q/(dNprim/dx)",
                        "logq2dNprimdx.AxisTitle": "log(Q/(dNprim/dx))",
                        #
                        "q2dNprim.AxisTitle": "Q/dNprim",
                        "logq2dNprim.AxisTitle": "log(Q/dNprim)",
                        #
                        "q2NPrim.AxisTitle": "Q/Nprim",
                        "q2NTot.AxisTitle": "Q/Ntot",
                        "logq2NPrim.AxisTitle": "log(Q/Nprim)",
                        "logq2NTot.AxisTitle": "log(Q/Ntot)",
                        #
                        "nTot2Nprim.AxisTitle": "Ntot/Nprim",
                        "lognTot2Nprim.AxisTitle": "log(Ntot/Nprim)",
                        "nTot2dNprimdx.AxisTitle": "Ntot/(dNprim/dx)",
                        "lognTot2dNprimdx.AxisTitle": "log(Ntot/(dNprim/dx))",
                        "nTot2dNprim.AxisTitle": "Ntot/dNprim",
                        "lognTot2dNprim.AxisTitle": "log(Ntot/dNprim)",
                        #
                        "dNprimdxScaled.AxisTitle": "dNprimdx^exponentX",
                        "dNprimScaled.AxisTitle": "dNprim^exponentX",
                        #
                        "nPrimMean.AxisTitle": "<Nprim>",
                        "nTotMean.AxisTitle": "<Ntot>",
                        "qMean.AxisTitle": "<Q>",
                        #
                        "nTotVector2nPrimMean.AxisTitle": "Ntot/<Nprim>",
                        "qVector2nPrimMean.AxisTitle": "Q/<Nprim>",
                        "qVector2nTotMean.AxisTitle": "Q/<Ntot>",
                        #
                        "nPrimStdRel.AxisTitle": "sigmaNprim/<Nprim>",
                        "nTotStdRel.AxisTitle": "sigmaNtot/<Ntot>",
                        "qStdRel.AxisTitle": "sigmaQ/<Q>",
                        #
                        "TransGEMStdRel.AxisTitle": "1/TransGEM",
                        "nTotStdRelExp.AxisTitle": "expected sigmaNtot/<Ntot>",
                        "qStdRelExp.AxisTitle": "expected sigmaQ/<Q>",
                        "nTotStdRelExp2.AxisTitle": "expected sigmaNtot/<Ntot> 2",
                    }
    return df
                    
def createSimulDashboard(df, outputName):
    # add more columns
    df["q2dNprimdx"]=df["qVector"]/df["dNprimdx"]
    df["logq2dNprimdx"] = np.log(df["q2dNprimdx"], where=df["qVector"]/df["dNprimdx"] > 0)
    df["dNprim"]=df["dNprimdx"]*df["padLength"]
    df["q2dNprim"]=df["qVector"]/(df["dNprim"])
    df["logq2dNprim"] = np.log(df["q2dNprim"], where=df["qVector"]/df["dNprim"] > 0)
    df["lognSecSatur"] = np.log10(df["nSecSatur"])
    # add metadata
    df=addMetaData(df)
    
    aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVars()

    aliases = [
        ("q2NPrim", "qVector/nPrimVector"),
        ("q2NTot", "qVector/nTotVector"),
        ("logq2NPrim", "log(qVector/nPrimVector)"),
        ("logq2NTot", "log(qVector/nTotVector)"),
        #
        ("nTot2Nprim", "nTotVector/nPrimVector"), 
        ("lognTot2Nprim", "log(nTotVector/nPrimVector)"),
        ("nTot2dNprimdx", "nTotVector/dNprimdx"),                        
        ("lognTot2dNprimdx", "log(nTotVector/dNprimdx)"),
        ("nTot2dNprim", "nTotVector/dNprim"),                        
        ("lognTot2dNprim", "log(nTotVector/dNprim)"),
        #
        ("dNprimdxScaled", "dNprimdx**exponentX"),
        ("dNprimScaled", "dNprim**exponentX"),
        #
        ("nTotVector2nPrimMean", "nTotVector/nPrimMean"),
        ("qVector2nPrimMean", "qVector/nPrimMean"),
        ("qVector2nTotMean", "qVector/nTotMean"),
        ("lognTotVector2nPrimMean", "log(nTotVector/nPrimMean)"),
        ("logqVector2nPrimMean", "log(qVector/nPrimMean)"),
        ("logqVector2nTotMean", "log(qVector/nTotMean)"),
        #
        ("nPrimStdRel", "nPrimStd/nPrimMean"),
        ("nTotStdRel", "nTotStd/nTotMean"),
        ("qStdRel", "qStd/qMean"),
        #
        ("TransGEMStdRel","1./(TransGEM)"),
        ("nTotStdRelExp","(1./sqrt(nPrimMean))*sqrt(1+sigmaNRel**2)"),
        ("qStdRelExp","(1./sqrt(nPrimMean))*sqrt(1+sigmaNRel**2 +4.2*(TransGEMStdRel)**2)"),
        ("nTotStdRelExp2","sqrt(nPrimMean*((nTotMean/nPrimMean)**2)+((sigmaNRel**2)/nPrimMean)*nTotMean**2)/(nTotMean)"),
        #
        ("Unit","1+dNprimdx*0"),
    ]
    aliasArray.extend(aliases)
    variables.extend(df.columns.to_list() + [i[0] for i in aliases])
    variables.sort()
    
    parameterArray+=[
        {"name": "varX", "value":"dNprimdx", "options":variables},
        {"name": "varY", "value":"qVector", "options":variables},
        {"name": "varZ", "value":"region", "options":variables},
        {"name": "varYNorm", "value":"dNprim", "options":variables},
        {"name": "varZNorm", "value":"Unit", "options":variables},
    ]

    widgetParams+=[    
        ['multiSelect',["region"],{"name":"region"}],
        ['multiSelect',["SatOn"],{"name":"SatOn"}],
        ['range',["TransGEM"],{"name":"TransGEM"}],
        ['range',["lognSecSatur"],{"name":"lognSecSatur"}],
        ['range',["dNprimdx"],{"name":"dNprimdx"}],
        ['range',["dNprim"],{"name":"dNprim"}],
    ]
    
    widgetLayoutDesc["Select"] = [["region","SatOn"],["dNprim","dNprimdx","lognSecSatur","TransGEM"]]

    
    output_file(outputName) 
    bokehDrawSA.fromArray(df, "qVector>0", figureArray, widgetParams, layout=figureLayoutDesc, sizing_mode='scale_width', nPointRender=50000, widgetLayout=widgetLayoutDesc,
                            parameterArray=parameterArray, histogramArray=histoArray, rescaleColorMapper=True, arrayCompression=arrayCompressionRelative16, aliasArray=aliasArray,palette=kRainbow256)

def createTruncDashboard(df, outputName):
    aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVars()
    
    aliases = [
        ("Unit","1+dNprimdx*0"),
        ("qTruncROC0_to_dNprimdx","qTruncROC0/dNprimdx"),
        ("qTruncROC1_to_dNprimdx","qTruncROC1/dNprimdx"),
        ("qTruncROC2_to_dNprimdx","qTruncROC2/dNprimdx"),
        ("qTruncROC3_to_dNprimdx","qTruncROC3/dNprimdx"),
        ("qTruncROC4_to_dNprimdx","qTruncROC4/dNprimdx"),
    ]

    aliasArray.extend(aliases)
    variables.extend(df.columns.to_list() + [i[0] for i in aliases])
    variables.sort()

    parameterArray.extend([
        {"name": "varX", "value":"dNprimdx", "options":variables},
        {"name": "varY", "value":"qTruncROC4_to_dNprimdx", "options":variables},
        {"name": "varYNorm", "value":"Unit", "options":variables},
        {"name": "varZ", "value":"percentage", "options":variables},
        {"name": "varZNorm", "value":"Unit", "options":variables},
    ])

    widgetParams.extend([    
        ['multiSelect',["percentage"],{"name":"percentage"}],
        ['multiSelect',["SatOn"],{"name":"SatOn"}],
        ['range',["TransGEM"],{"name":"TransGEM"}],
        ['range',["nSecSatur"],{"name":"nSecSatur"}],
        ['range',["dNprimdx"],{"name":"dNprimdx"}],
    ])
    
    widgetLayoutDesc["Select"] = [["percentage", "SatOn", "dNprimdx", "nSecSatur", "TransGEM"], {'sizing_mode': 'scale_width'}]

    output_file(outputName) 
    bokehDrawSA.fromArray(df, "qMean>0", figureArray, widgetParams, layout=figureLayoutDesc, sizing_mode='scale_width', nPointRender=50000, widgetLayout=widgetLayoutDesc,
                            parameterArray=parameterArray, histogramArray=histoArray, rescaleColorMapper=True, arrayCompression=arrayCompressionRelative16, aliasArray=aliasArray,palette=kRainbow256)


def main():
    loadMacro(macro="$NOTES/JIRA/ATO-614/code/toydEdxSimul.C")
    rdf = simulRDFComplex(100000)
    array1 = rdfToAwkward(
        rdf, 
        columnNames=[
            "qVector",
            "nPrimVector",          
            "nTotVector",
            "dNprimdx",
            "padLength",
            "region",
            "SatOn",
            "TransGEM",
            "nSecSatur",
            "nPrimMean",
            "nTotMean",
            "qMean",
            "qMedian",
            "nPrimStd",
            "nTotStd",
            "qStd",
            "lognPrimStd",
            "lognTotStd",
            "logqStd",
        ]
    )
    df = awkwardToPandas(array1)
    createSimulDashboard(df=df, outputName="toySimul.html")
    print("toySimul.html is created.")
    
    array2 = rdfToAwkward(
        rdf, 
        columnNames=[
            "dNprimdx",
            "SatOn",
            "TransGEM",
            "nSecSatur",
            "nPrimMean",
            "nTotMean",
            "qMean",
            "qMedian",
            "nPrimStd",
            "nTotStd",
            "qStd",
            "lognPrimStd",
            "lognTotStd",
            "logqStd",
            "qTruncROC4",
            "qTruncROC0",
            "qTruncROC1",
            "qTruncROC2",
            "qTruncROC3",
        ]
    )
    dfTrunc = awkwardToPandas(array2)
    dfTrunc.reset_index([1], inplace=True)
    truncMap = {0:30, 1:40, 2:50, 3:60, 4:70, 5:80, 6:90, 7:100}
    dfTrunc['percentage']=dfTrunc['subentry'].round(0).astype(int).map(truncMap).astype(CategoricalDtype(ordered=True))
    dfTrunc = dfTrunc.drop(['subentry'], axis=1)
    createTruncDashboard(df=dfTrunc, outputName="trunc.html")
    print("trunc.html is created.")

if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     globals()[sys.argv[1]]()
