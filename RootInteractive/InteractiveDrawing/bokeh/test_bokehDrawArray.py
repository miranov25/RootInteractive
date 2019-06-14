from RootInteractive.InteractiveDrawing.bokeh.bokehDraw import *
import pandas as pd
import numpy as np
import logging
from bokeh.layouts import *
from bokeh.palettes import *
from bokeh.io import push_notebook

def drawGraphArray(df,variables):
    """
    :param df:
    :param variables:
    :return:
    """
    graphArray=[]
    return graphArray

def bokehDrawArray(dataFrame, query, figureArray, **kwargs):
    """

    :param dataFrame:         - input data frame
    :param query:             - query
    :param figureArray:       - figure array
    :param kwargs:
    :return:
    """

    dfQuery = dataFrame.query(query)
    try:
        source = ColumnDataSource(dfQuery)
    except:
        logging.error("Invalid source:", source)
    # define default options
    options = {
        'line': -1,
        'size': 2,
        'tooltips': 'pan,box_zoom, wheel_zoom,box_select,lasso_select,reset',
        'y_axis_type': 'auto',
        'x_axis_type': 'auto',
        'plot_width': 600,
        'plot_height': 400,
        'errX': '',
        'errY': '',
        'commonX': -1,
        'commonY': -1,
        'ncols': -1,
        'layout': '',
        'palette': Spectral6
    }
    options.update(kwargs)
    plotArray=[]
    for i, variables in enumerate(figureArray):
        print(i, variables)
        figureI = figure(plot_width=options['plot_width'], plot_height=options['plot_height'], title="xxx",
                    tools=options['tooltips'], x_axis_type=options['x_axis_type'], y_axis_type=options['y_axis_type'])
        #graphArray=drawGraphArray(df, variables)
        #plotArray.append(figureI)



def test_DrawFormula():
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    df.head(10)
    # testLayout = "((0,1),(2,x_visible=0),(3), plot_height=200,plot_width=800,commonX=3,commonY=3,y_visible=0)"
    df.metaData = {}
    #"A.title": "title A", "B.title": "title B","C":"C"}
    df.metaData['A.title']="A"
    df.metaData['C.title']="C"
    print(df.metaData)
    print(df.metaData['A.title'])
    print(df.metaData['C.title'])
    figureArray=[]
    figureArray.append([['A'],['C'],{"color:blue"}])
    figureArray.append([['B'],['C','D']])

    bokehDrawArray(df,"A>0",figureArray)
    # bokehFigure=drawColzArray(df, "A>0", "A", "A:B:C:D", "C", None, ncols=2)


#test_DrawFormula()
