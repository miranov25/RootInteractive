import logging
import pytest
#from RootInteractive.InteractiveDrawing.bokeh.bokehDraw import *
from RootInteractive.Tools.aliTreePlayer import *
from bokeh.io import curdoc

if "ROOT" in sys.modules:
    ROOT.gSystem.Load("$ALICE_ROOT/lib/libSTAT.so")
else:
    pytest.skip("ROOT module is not imported", allow_module_level=True)



#   This test is useless since AliExternalInfo is no longer used in test...
#def test_AliExternalInfo():
#    """ test if the tree could be read """
#    info = ROOT.AliExternalInfo()
#    info.fVerbose = 0
#    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1")
#    # tree.Show(0)
#    assert tree.GetEntries() > 0


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
#    info = ROOT.AliExternalInfo()
#    info.fVerbose = 0
#    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1")
    tree, dummy = makeABCD(10000)
    tree.Show(0)
    branchTree = treeToAnyTree(tree)
    print(findSelectedBranch(branchTree, "A"))
    assert (findSelectedBranch(branchTree, "A"))
    print(findSelectedBranch(branchTree, "bigA"))
    assert (findSelectedBranch(branchTree, "bigA"))
    print(findSelectedBranch(branchTree, "smallA"))
    assert (findSelectedBranch(branchTree, "smallA"))


def test_Aliases():
#    info = ROOT.AliExternalInfo()
#    info.fVerbose = 0
#    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1")
    tree, dummy = makeABCD(10000)
    aliases = aliasToDictionary(tree)
    base = makeAliasAnyTree("bigA", aliases)
    print(RenderTree(base))
    print(findSelectedBranch(base, "bigA"))


def test_TreeParsing():
#    info = ROOT.AliExternalInfo()
#    info.fVerbose = 0
#    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1")
    tree, dummy = makeABCD(10000)
    selection = "B>0"
    varDraw = "A:B:C"
    tooltips = [("A", "(@A)"), ("B", "@B"), ("C", "(@C)"), ("D", "@D")]
    widgets = "tab.sliders(slider.A(0,1,0.1,0.2,0.8),slider.B(0,1,0.1,0.2,0.8), slider.C(0,1,0.1,0.2,0.8)),"
#    widgets += "tab.checkboxGlobal(slider.global_Warning(0,1,1,0,1),checkbox.global_Outlier(0)),"
#    widgets += "tab.checkboxMIP(slider.MIPquality_Warning(0,1,1,0,1),checkbox.MIPquality_Outlier(0), checkbox.MIPquality_PhysAcc(1))"
    toRemove = [r"^tab\..*"]
    toReplace = ["^slider.", "^checkbox."]
    logging.info(getAndTestVariableList([selection, varDraw, widgets, "xxx"], toRemove, toReplace, tree))


def test_Parsing():
    query = "C>0.31"
    variables = "A:B:C"
    tooltips = [("A", "(@A)"), ("B", "@B"), ("C", "(@C)"), ("D", "@D")]
    slider = "tab.sliders(slider.A(0,1,0.1,0.2,0.8),slider.B(0,1,0.1,0.2,0.8), slider.C(0,1,0.1,0.2,0.8)),"
#    slider = "accordion.first(slider.P0(0,1,0.5,0,1),slider.commonF(0,15,5,0,5)),accordion.second(dropdown.MB(0,0.5,1)),accordion.second(checkbox.isMax(False)),slider.typeF(0,4,1,0)"
    varSource = ["A", "B", "C", slider, query]
    toRemove = [r"^tab\..*", r"^accordion\..*", "False", "True"]
    toReplace = ["^slider.", "^checkbox.", "^dropdown."]
    # getAndTestVariableList(varSource,toRemove,toReplace,variables,slider)
    counts = dict()
    for expression in varSource:
        parseTreeVariables(expression, counts)

def testTree2Panda():
#    info = ROOT.AliExternalInfo()
#    info.fVerbose = 0
#    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1", "QA.ITS")
    tree, dummy = makeABCD(10000)
    df=tree2Panda(tree, ["A", "B"], "D>0")
    print(df.head(5))
    df=tree2Panda(tree, ["A", "B", "C"], "D>0", columnMask=[["A", "a"]])
    print(df.head(5))
    df=tree2Panda(tree, ["A", "B"], "D>0", exclude=["C"], columnMask=[["B", "b"]])
    print(df.head(5))

def testLoadTree():
    tree, treeList, fileList=LoadTrees("cat ../tutorial/bokehDraw/performance.list", "identFit", "xxx", ".*", 0)
    if tree == None:
        logging.error("Input file not accessible performance.list")
        pass
    tree.SetAlias("norm", "param.fElements[0]")
    tree.SetAlias("slope", "param.fElements[1]")
    tree.SetAlias("isMax", "name.String().Contains(\"PionMax\")==0")
    tree.SetAlias("isTot", "name.String().Contains(\"PionTot\")==0&&name.String().Contains(\"PionMaxTot\")==0")
    tree.SetAlias("isMaxTot", "name.String().Contains(\"PionMaxTot\")>0")
    tree.SetAlias("P0", "name.String().Contains(\"P0\")>0")
    tree.SetAlias("P1", "name.String().Contains(\"P1\")>0")
    tree.SetAlias("PA", "name.String().Contains(\"PA\")>0")
    tree.SetAlias("MB", "name.String().Contains(\"MB\")>0")
    tree.Show(0)
    print(tree)
    print(treeList)
    print(fileList)
    anyTree=treeToAnyTree(tree)
    print(anyTree)
    print(tree.GetListOfFriends())
    tree.GetListOfFriends().ls()
    print(findSelectedBranch(anyTree, ".*"))
#    logging.info(getTreeInfo(tree))

# def testlaser():
#     tree, treeList, fileList = LoadTrees("cat /data/NOTESdata/alice-tpc-notes/JIRA/ATO-493/laser.list",".*","xxx",".*",0)
#     anyTree=treeToAnyTree(tree)
#     print(findSelectedBranch(anyTree, ".*"))
#     display(anyTree)
#     display(tree.anyTree)
#     tree.Show(0)
#     df=tree2Panda(tree, [".*sector","stack","bundle"], "stack!=0")
#     print(df.columns)
#     return tree


#test_AnyTree()
#test_Aliases()
#test_Parsing()
#testTree2Panda()
#testLoadTree()
#testlaser()