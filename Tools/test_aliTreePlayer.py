from Tools.aliTreePlayer import *
import ROOT


def test_AliExternalInfo():
    """ test if the tree could be read """
    info = ROOT.AliExternalInfo()
    info.fVerbose = 0
    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1")
    # tree.Show(0)
    assert tree.GetEntries() > 0


def test_TTreeSredirectorWrite():
    """TTreeSredirector does not work in python"""
    # pcstream=ROOT.TTreeSRedirector("test_AliTreePlayer.root","recreate")
    # for i in range(0,10):
    return 0


#def test_Tree():
#    """test reading and dumping example tree"""
#    f = ROOT.TFile.Open("data/mapOutputEvent.root")
#    tree = f.Get("hisTPCOnElectronDist")
#    assert tree.GetEntries() > 0


def test_AnyTree():
    info = ROOT.AliExternalInfo()
    info.fVerbose = 0
    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1")
    branchTree = treeToAnyTree(tree)
    print(findSelectedBranch(branchTree, "bz"))
    assert (findSelectedBranch(branchTree, "bz"))
    print(findSelectedBranch(branchTree, "MIP.*arning$"))
    assert (findSelectedBranch(branchTree, "MIP.*arning$"))


def test_Aliases():
    info = ROOT.AliExternalInfo()
    info.fVerbose = 0
    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1")
    aliases = aliasToDictionary(tree)
    base = makeAliasAnyTree("global_Warning", aliases)
    print(RenderTree(base))
    print(findSelectedBranch(base, ".*PID.*"))


def test_TreeParsing():
    info = ROOT.AliExternalInfo()
    info.fVerbose = 0
    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1")
    selection = "meanMIP>0&resolutionMIP>0&time>0"
    varDraw = "meanMIP:meanMIPele:resolutionMIP:xxx"
    tooltips = [("MIP", "(@meanMIP)"), ("Electron", "@meanMIPele"), ("Global status", "(@global_Outlier,@global_Warning)"),
                ("MIP status(Warning,Outlier,Acc.)", "@MIPquality_Warning,@MIPquality_Outlier,@MIPquality_PhysAcc")]
    widgets = "tab.sliders(slider.meanMIP(45,55,0.1,45,55),slider.meanMIPele(50,80,0.2,50,80), slider.resolutionMIP(0,0.15,0.01,0,0.15)),"
    widgets += "tab.checkboxGlobal(slider.global_Warning(0,1,1,0,1),checkbox.global_Outlier(0)),"
    widgets += "tab.checkboxMIP(slider.MIPquality_Warning(0,1,1,0,1),checkbox.MIPquality_Outlier(0), checkbox.MIPquality_PhysAcc(1))"
    toRemove = ["^tab\..*"]
    toReplace = ["^slider.", "^checkbox."]
    print(getAndTestVariableList([selection, varDraw, widgets, "xxx"], toRemove, toReplace, tree))


def test_Parsing():
    query = "tailF>1"
    variables = "pileUp:tailF:commonF:typeF:fraction:norm:slope:isMax:MB:P0:P1:PA"
    tooltips = [('Is maximum', '@isMax'), ("Pad type", "@typeF")]
    slider = "accordion.first(slider.P0(0,1,0.5,0,1),slider.commonF(0,15,5,0,5)),accordion.second(dropdown.MB(0,0.5,1)),accordion.second(checkbox.isMax(False)),slider.typeF(0,4,1,0)"
    varSource = ["x", "varX", "varY", slider, query]
    toRemove = ["^tab\..*", "^accordion\..*", "False", "True"]
    toReplace = ["^slider.", "^checkbox.", "^dropdown."]
    # getAndTestVariableList(varSource,toRemove,toReplace,variables,slider)
    counts = dict()
    for expression in varSource:
        parseTreeVariables(expression, counts)


# test_AnyTree()
# test_Aliases()

test_Parsing()
