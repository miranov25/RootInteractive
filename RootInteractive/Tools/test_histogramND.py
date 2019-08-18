try:
    import ROOT
    ROOT.gSystem.Load("$ALICE_ROOT/lib/libSTAT.so")
except ImportError:
    pass

from RootInteractive.Tools.histogramND import *
from RootInteractive.Tools.aliTreePlayer import *
from RootInteractive.Tools.Alice.BetheBloch import *

histogramArray=[]

def testHistoPanda(nPoints=10000):
    dataFrame = toyMC(nPoints)
    dataFrame.head(5)
    histoStringArray = [
        "TRD:tgl:p:particle:#TRD>0>>hisTRDTgl_P_P(200,0.5,3,5,-1,1, 200,0.3,5,5,0,5)",
        "TPC:tgl:p:particle:#TPC>0>>hisTPCTgl_P_P(200,0.5,3,5, -1,1, 200,0.3,5,5,0,5)",
        "TPC0:tgl:p:particle:#TPC>0>>hisTPC0Tgl_P_P(200,0.5,3,5, -1,1, 200,0.3,5,5,0,5)"
    ]
    output_file("test_histoNDTools.html")
    histogramArray = histogramND.makeHistogramArray(dataFrame, histoStringArray)
    return histogramArray

