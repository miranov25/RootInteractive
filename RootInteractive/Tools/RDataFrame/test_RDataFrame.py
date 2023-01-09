
import ROOT



def initTest():
    ROOT.gSystem.AddIncludePath("-I$RootInteractive/")
    ROOT.gROOT.LoadMacro("$RootInteractive/RootInteractive/Tools/RDataFrame/test_RDataFrame.C+g")


def test0():
    ROOT.gROOT.ProcessLine(".L $NOTES/JIRA/ATO-615/drawCounters.C+g")

def testTimeSeries():
    rdf=ROOT.testRDFSeries(100)





