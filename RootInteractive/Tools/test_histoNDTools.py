### 0. Imports
import numpy as np
import math
import pandas as pd
from functools import partial
import re
from TTreeHnInteractive.TTreeHnBrowser import *
#from InteractiveDrawing.bokeh.bokehDraw import *
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawHisto import *

#output_notebook()
import ROOT

ROOT.gSystem.Load("$ALICE_ROOT/lib/libSTAT.so")
from RootInteractive.Tools.aliTreePlayer import *
from RootInteractive.Tools.histoNDTools import *
import logging
from RootInteractive.Tools.Alice.BetheBloch import *
from bokeh.palettes import *

#
logging.getLogger().setLevel(1)
finput=ROOT.TFile(" ~/github/RootInteractive3/tutorial/data/RootInteractive/testData/JIRA/PWGPP-485/hisPull.root")
hisArray=finput.Get("hisArray")


def runTOYMC(nPoints=100000):
    # 1.) Setup and run  TOY MC
    # nPoints=1000000
    pdg = ROOT.TDatabasePDG.Instance()
    particleList = ["e+", "mu+", "pi+", "K+", "proton"]
    massList = [pdg.GetParticle(a).Mass() for a in particleList]

    def GetMass(iPart):
        return [massList[i] for i in iPart]

    detectors = ["ITS", "TPC", "TPC0", "TPC1", "TPC2", "TRD"]
    ### 2.) Run  Toy MC
    p = np.random.random(nPoints)
    p *= 5; p += 0.1
    tgl = np.random.random(nPoints)
    tgl*=2; tgl+=-1;

    particle = np.random.randint(0, 5, size=nPoints)
    mass = np.asarray(GetMass(particle))
    lbg = np.log(p / mass)
    data = {'p': p, 'particle': particle, 'lbg': lbg, "tgl" :tgl}
    df = pd.DataFrame(data)
    for det in detectors:
        df[det] = BetheBlochAlephNP(lbg)
        df[det] *= np.random.normal(1, 0.1, nPoints)
    return df


def testHistoPanda(nPoints=10000):
    dataFrame = runTOYMC(nPoints)
    dataFrame.head(5)
    histoStringArray = [
        "TRD:tgl:p:particle:#TRD>0>>hisTRDTgl_P_P(200,0.5,3,5,-1,1, 200,0.3,5,5,0,5)",
        "TPC:tgl:p:particle:#TPC>0>>hisTPCTgl_P_P(200,0.5,3,5, -1,1, 200,0.3,5,5,0,5)",
        "TPC0:tgl:p:particle:#TPC>0>>hisTPC0Tgl_P_P(200,0.5,3,5, -1,1, 200,0.3,5,5,0,5)"
    ]
    output_file("test_histoNDTools.html")
    histogramArray = makeHistogramArray(dataFrame, histoStringArray)
    fig0,data0=bokehDrawHistoSliceColz(histogramArray[0],np.index_exp[0:200, 0:5, 10:11,0:5], 0,3, 1, {'plot_width':500, 'plot_height':300},{'size':5})
    fig1,data1=bokehDrawHistoSliceColz(histogramArray[1],np.index_exp[0:200, 0:5, 10:11,0:5], 0,3, 1, {'plot_width':500, 'plot_height':300},{'size':5})
    show(column(fig0,fig1))

def testTHnDraw():
    output_file("test_histoNDTools_THnDraw.html")
    finput=ROOT.TFile(" ~/github/RootInteractive3/tutorial/data/RootInteractive/testData/JIRA/PWGPP-485/hisPull.root")
    hisArray=finput.Get("hisArray")
    hisArray.ls()
    hisArray.FindObject("hisdY").Print("all")
    histogramArray={}
    for his in hisArray:
        ddHis=thnToNumpyDD(his)
        histogramArray[ddHis['name']]=ddHis
    figdY,sourcedY=bokehDrawHistoSliceColz(histogramArray['hisdY'],np.index_exp[0:100, 0:5, 0:6,1:9,1:3], 0,3,1, {'plot_width':800, 'plot_height':300},{'size':5})
    figdZ,sourcedY=bokehDrawHistoSliceColz(histogramArray['hisdZ'],np.index_exp[0:100, 0:5, 0:6,1:9,1:3], 0,3,1, {'plot_width':800, 'plot_height':300},{'size':5})
    show(column(figdY,figdZ))

def testBokehDrawHistoTHn():
    output_file("test_histoNDTools_testBokehDrawHistoTHn.html")
    hisArray.ls()
    bokehDrawHisto.fromTHnArray(hisArray,{},{},{})



#testHistoPanda(1000000)
#testTHnDraw()
#testBokehDrawHistoTHn()