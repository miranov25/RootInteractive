from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from ROOT import TFile, TStatToolkit, gSystem, AliTreePlayer
from bokeh.io import curdoc

output_file("test_bokehDrawSA.html")
# import logging

gSystem.Load("$ALICE_ROOT/lib/libSTAT.so")

#curdoc().theme = 'caliber'
TFile.SetCacheFileDir("../../data/")
tree = AliTreePlayer.LoadTrees("echo http://rootinteractive.web.cern.ch/RootInteractive/data/tutorial/bokehDraw/treeABCD.root", ".*", ".*ABCD.*", ".*", "", "")

df = pd.DataFrame(np.random.random_sample(size=(500, 4)), columns=list('ABCD'))
df.head(10)
df.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)", 'D.AxisTitle': "D (a.u.)"}



figureArray = [
    [['A'], ['C-A'], {"color": "red", "size": 7, "colorZvar":"C"}],
    [['A'], ['C+A', 'C-A']],
    [['B'], ['C+B', 'C-B'],{"color": "red", "size": 7, "colorZvar":"C"}],
    [['D'], ['(A+B+C)*D'], {"size": 10}],
]
widgets="slider.A(0,1,0.05,0,1),slider.B(0,1,0.05,0,1),slider.C(0,1,0.01,0.1,1):slider.D(0,1,0.01,0,1)"
figureLayout: str = '((0,1,2, plot_height=300),(3, x_visible=1),commonX=1,plot_height=300,plot_width=1200)'


def testOldInterface():
    output_file("test_bokehDrawSAOldInterface.html")
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
    fig=bokehDrawSA(df, "A>0", "A","A:B:C:D","C",widgets,0,tooltips=tooltips, layout=figureLayout)

def testBokehDrawArraySA():
    output_file("test_bokehDrawSAArray.html")
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray,widgets,tooltips=tooltips, layout=figureLayout)

def testOldInterface_tree():
    output_file("test_bokehDrawSAOldInterface_fromTTree.html")
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
    fig=bokehDrawSA(tree, "A>0", "A","A:B:C:D","C",widgets,0,tooltips=tooltips, layout=figureLayout)

def testBokehDrawArraySA_tree():
    output_file("test_bokehDrawSAArray_fromTTree.html")
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
    fig=bokehDrawSA.fromArray(tree, "A>0", figureArray, widgets, tooltips=tooltips, layout=figureLayout,columnMask="")

#testOldInterface()
#testBokehDrawArraySA()
testOldInterface_tree()
testBokehDrawArraySA_tree()
