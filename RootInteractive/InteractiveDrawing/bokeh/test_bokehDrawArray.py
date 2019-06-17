from RootInteractive.InteractiveDrawing.bokeh.bokehDraw import *


# import logging


def test_DrawFormula():
    """
    Test of the bokehDrawArray to draw figure array -
    :return:
    """
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    df.head(10)
    df.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)", 'D.AxisTitle': "D (a.u.)"}
    print(df.metaData)
    #
    figureArray = [
        [['A'], ['C-A'], {"color": "red", "size": 1}],
        [['A'], ['C+A', 'C-A']],
        [['B'], ['C+B', 'D+B']],
        [['D'], ['sin(D/10)', 'sin(D/20)*0.5', 'sin(D/40)*0.25'], {"size": 10}],
        ['table']
    ]
    figureLayout: str = '((0,1,2),(3),(4, x_visible=1),commonX=1,x_visible=1,y_visible=0,plot_height=250,plot_width=1000)'
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
    bokehDrawArray(df, "A>0", figureArray, layout=figureLayout, color="blue", size=4, tooltips=tooltips)


test_DrawFormula()
