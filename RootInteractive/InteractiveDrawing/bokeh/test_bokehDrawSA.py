from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA_2 import *
output_file("test_bokehDrawSA.html")
# import logging

df = pd.DataFrame(np.random.randint(0, 100, size=(500, 4)), columns=list('ABCD'))
df.head(10)
df.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)", 'D.AxisTitle': "D (a.u.)"}

figureArray = [
    [['A'], ['C-A'], {"color": "red", "size": 7, "colorZvar":"C"}],
    [['A'], ['C+A', 'C-A']],
    [['B'], ['C+B', 'C-B'],{"color": "red", "size": 7, "colorZvar":"C"}],
    [['D'], ['sin(D/10)', 'sin(D/20)*0.5', 'sin(D/40)*0.25'], {"size": 10}],
]
widgets="slider.A(0,100,0.5,0,100),slider.B(0,100,5,0,100),slider.C(0,100,1,1,100):slider.D(0,100,1,1,100)"
figureLayout: str = '((0,1,2, plot_height=300),(3, x_visible=1),commonX=1,plot_height=300,plot_width=1200)'


def testOldInterface():
    output_file("test_bokehDrawSAOldInterface.html")
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
    fig=bokehDrawSA(df, "A>0", "A","A:B:C:D","C",widgets,0,tooltips=tooltips, layout=figureLayout)

def testBokehDrawArraySA():
    output_file("test_bokehDrawSAArray.html")
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray,widgets,tooltips=tooltips, layout=figureLayout)

#testOldInterface()
#testBokehDrawArraySA()