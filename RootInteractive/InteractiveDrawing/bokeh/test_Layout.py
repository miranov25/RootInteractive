from RootInteractive.InteractiveDrawing.bokeh.bokehDraw import *

import pandas as pd
import numpy as np




def test_Draw():
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    initMetadata(df)
    df.head(10)
    df.meta.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)", 'D.AxisTitle': "D (a.u.)"}
    testLayout = "((0,1),(2,x_visible=0),(3), plot_height=200,plot_width=800,commonX=3,commonY=3,y_visible=0)"
    bokehFigure=bokehDraw(df, "A>0", "A", "A:B:C:D", "C", "slider.A(0,100.1,0,1),slider.B(0,100,100,100,300)", None, layout=testLayout)


def test_Layout():
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    initMetadata(df)
    df.head(10)
    df.meta.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)", 'D.AxisTitle': "D (a.u.)"}
    testLayout1 = "((0,1),(2,x_visible=0),(3), plot_height=200,plot_width=800,commonX=3,commonY=3,y_visible=0)"
    testLayout2 = "((0),(1),(2,x_visible=0),(3), plot_height=200,plot_width=800,commonX=3,commonY=3,y_visible=1)"
    fig1=drawColzArray(df, "A>0", "A", "A:B:C:D", "C", None, layout=testLayout1)
    fig2=drawColzArray(df, "A>0", "A", "A:B:C:D", "C", None, layout=testLayout2)

def diff(A,B):
    return A/B

def test_DrawFormula():
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    initMetadata(df)
    df.head(10)
    df.meta.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)", 'D.AxisTitle': "D (a.u.)"}
    testLayout = "((0,1),(2,x_visible=0),(3), plot_height=200,plot_width=800,commonX=3,commonY=3,y_visible=0)"
    bokehFigure, xxx, yyy=drawColzArray(df, "A>0", "A", "A:B:C:D", "C", None, ncols=2)
    show(bokehFigure)


test_Draw()
#test_Layout()
#test_DrawFormula()
