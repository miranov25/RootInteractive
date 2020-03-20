# CODE AND TEST NOT FINISHED YET

try:
    import ROOT
except ImportError:
    pass

from RootInteractive.Tools.histogramND import *
from bokeh.plotting import figure, ColumnDataSource, curdoc
import sys
import pytest

if "ROOT" in sys.modules:
    ROOT.gSystem.Load("$ALICE_ROOT/lib/libSTAT.so")
    from RootInteractive.Tools.Alice.BetheBloch import *
    


histogramMap={}
histogramMapABCD={}
controlList=[]
tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]
layout
fig0=None
fig1=None

def initControlList():
    #slider = Slider(start=0, end=100, step=5, value=50, title="A")
    controlList.append(RangeSlider(start=0, end=100, step=5, value=(10,80), title="A"))
    controlList.append(RangeSlider(start=0, end=100, step=5, value=(20,90), title="B"))
    controlList.append(RangeSlider(start=0, end=100, step=5, value=(30,100), title="C"))
    controlList.append(RangeSlider(start=0, end=100, step=5, value=(40,120), title="D"))


def makeMapABCD(nPoints=100000):
    df = pd.DataFrame(np.random.randint(0,100,size=(nPoints, 4)), columns=list('ABCD'))
    histoStringArray = [
        "A:B:C:D:#A+B>100>>hABCD0(100,0,100,50,0,100,50,0,100,50,0,100)",
        "A:B:C:D:#A+C>100>>hABCD1(100,0,100,50,0,100,50,0,100,50,0,100)",
        "A:B:C:D:#A+D>100>>hABCD2(100,0,100,50,0,100,50,0,100,50,0,100)"
    ]
    global histogramMapABCD
    histogramMapABCD = histogramND.makeHistogramMap(df, histoStringArray)


def testHistoPanda(nPoints=10000):
    if "ROOT" not in sys.modules:
        pytest.skip("no ROOT module")
    dataFrame = toyMC(nPoints)
    dataFrame.head(5)
    histoStringArray = [
        "TRD:tgl:p:particle:#TRD>0>>hisTRD(200,0.5,3,5,-1,1, 200,0.3,5,5,0,5)",
        "TPC:tgl:p:particle:#TPC>0>>hisTPCT(200,0.5,3,5, -1,1, 200,0.3,5,5,0,5)",
        "TPC0:tgl:p:particle:#TPC>0>>hisTPC0(200,0.5,3,5, -1,1, 200,0.3,5,5,0,5)"
    ]
    output_file("test_histoNDTools.html")
    global histogramMap
    histogramMap = histogramND.makeHistogramMap(dataFrame, histoStringArray)
    assert isinstance(histogramMap, dict)
    makeMapABCD(nPoints)
    global histogramMapABCD
    assert isinstance(histogramMapABCD, dict)
    return histogramMap

def testHistPandaDraw():
    if "ROOT" not in sys.modules:
        pytest.skip("no ROOT module")
    #output_file("test_histogramND_testHistPandaDrawColz.html")
    initControlList()
    histogram= histogramMap["hisTRD"]
    fig0, data0 = histogram.bokehDrawColz(np.index_exp[0:200, 0:5, 10:20,0:5],0,3, 1, {'plot_width':600, 'plot_height':600},{'size': 5})
    show(fig0)
    fig1 = histogramMapABCD['hABCD1'].bokehDraw1D(np.index_exp[0:100, 0:100, 0:100, 0:100], 0, {'plot_width':600, 'plot_height':600}, {'tooltips': tooltips})
    fig2 = histogramMapABCD['hABCD2'].bokehDraw2D(np.index_exp[0:100, 0:100, 0:100, 0:100], 0, 3, {'plot_width':600, 'plot_height':600}, {'tooltips': tooltips})
    #show(row(fig1,fig2))
    global layout
    layout=column(row(fig1,fig2),row(controlList))
    return fig0

def testHistoProjection():
    #pass
    projection=histogramNDProjection.fromMap("((hABCD0+hABCD1+hABCD1) (0:100,1:10,0:10,0:100) (0,1) ()))",histogramMapABCD)
    return projection

def mycallback(attr, old, new):
    print(controlList[0].value,controlList[1].value)

def runAll():
    testHistoPanda(100000)
    testHistPandaDraw()
    projection=testHistoProjection()
    print(projection)
    hisExpresion=projection.evaluateHistogram()
    hisExpresion=projection.evaluateHistogram()
    print(hisExpresion)
    for rslider in controlList:
        rslider.on_change('value', testcallback)
    s,h = projection.makeProjection(controlList,[0,1],"0:100,1:100,:,:")
    show(layout)
    curdoc().add_root(layout)
