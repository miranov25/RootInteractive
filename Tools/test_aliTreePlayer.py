from Tools.aliTreePlayer import *
import ROOT


def test_AliExternalInfo():
    """ test if the tree could be read """
    info = ROOT.AliExternalInfo()
    info.fVerbose = 0
    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1")
    # tree.Show(0)
    assert tree.GetEntries() > 0;


def test_TTreeSredirectorWrite():
    """TTreeSredirector does not work in python"""
    # pcstream=ROOT.TTreeSRedirector("test_AliTreePlayer.root","recreate")
    # for i in range(0,10):
    return 0


def test_Tree():
    """test reading and dumping example tree"""
    f = ROOT.TFile.Open("data/mapOutputEvent.root")
    tree = f.Get("hisTPCOnElectronDist")
    assert tree.GetEntries() > 0


def test_AnyTree():
    info = ROOT.AliExternalInfo()
    info.fVerbose = 0
    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1")
    branchTree = treeToAnyTree(tree)
    print(findSelectedBranch(branchTree, "bz"))
    assert (findSelectedBranch(branchTree, "bz"))


def test_Aliases():
    info = ROOT.AliExternalInfo()
    info.fVerbose = 0
    tree = info.GetTree("QA.TPC", "LHC15o", "cpass1_pass1")
    aliases = aliasToDictionary(tree)
    base = Node("global_Warning")
    makeAliasAnyTree("global_Warning", base, aliases)
    print(RenderTree(base))
    print(findSelectedBranch(base, ".*PID.*"))


test_AnyTree()
test_Aliases()
