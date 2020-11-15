from RootInteractive.InteractiveDrawing.bokeh.bokehDraw import *
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
# import logging

df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
initMetadata(df)
df.head(10)
df.meta.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)", 'D.AxisTitle': "D (a.u.)"}

figureArray = [
    [['A'], ['C-A'], {"color": "red", "size": 7, "colorZvar":"C"}],
    [['A'], ['C+A', 'C-A']],
    [['B'], ['C+B', 'C-B'],{"color": "red", "size": 7, "colorZvar":"C"}],
    [['D'], ['sin(D/10)', 'sin(D/20)*0.5', 'sin(D/40)*0.25'], {"size": 10}],
    ['table']
]

def test_DrawFormula():
    """
    Test of the bokehDrawArray to draw figure array -
    :return:
    """
    print(df.meta.metaData)
    #
    output_file("test_BokehDrawArray_DraFormula.html")
    figureLayout: str = '((0,1,2),(3),(4, x_visible=1),commonX=1,x_visible=1,y_visible=1,plot_height=250,plot_width=1000)'
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
    pAll,source,figureList, df2, cmapDict =bokehDrawArray(df, "A>0", figureArray, layout=figureLayout, color="blue", size=4, tooltips=tooltips)
    show(pAll)

def test_DrawfromArray():
    output_file("test_BokehDrawArray_test_DrawfromArray.html")
    figureLayout: str = '((0,1,2),(3),(4, x_visible=1),commonX=1,x_visible=1,y_visible=1,plot_height=250,plot_width=1000)'
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
    fig=bokehDraw.fromArray(df, "A>0", figureArray,"slider.A(0,100,0,0,100)",tooltips=tooltips, layout=figureLayout)

def test_DrawSAfromArray():
    output_file("test_BokehRDrawArray_DrawSAfromArray.html")
    figureLayout: str = '((0,1,2),(3),(4, x_visible=1),commonX=1,x_visible=1,y_visible=1,plot_height=250,plot_width=1000)'
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
    widgets=",query.(), slider.A(0,100,0,0,100),slider.B(0,100,0,0,100),slider.C(0,100,0,0,100),slider.D(0,100,0,0,100)"
    fig=bokehDrawSA.fromArray(df, "A>0", figureArray,widgets,tooltips=tooltips, layout=figureLayout)



#test_DrawFormula()
#test_DrawfromArray()
#test_DrawSAfromArray()
