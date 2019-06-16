from RootInteractive.InteractiveDrawing.bokeh.bokehDraw import *
from RootInteractive.Tools.pandaTools import *
import pandas as pd
import numpy as np
import logging
from bokeh.layouts import *
from bokeh.palettes import *
from bokeh.io import push_notebook
import re
import ROOT
import copy
#from bokeh.models.widgets import DataTable, DateFormatter, TableColumn



def bokehDrawArray(dataFrame, query, figureArray, **kwargs):
    """

    :param dataFrame:         - input data frame
    :param query:             - query
    :param figureArray:       - figure array
    :param kwargs:
    :return:
    """

    dfQuery = dataFrame.query(query)
    dfQuery.metaData = dataFrame.metaData
    print(dfQuery.metaData)
    # Check/resp. load derived variables
    i: int
    for i, variables in enumerate(figureArray):
        if len(variables) > 1:
            lengthX = len(variables[0])
            lengthY = len(variables[1])
            length = max(len(variables[0]), len(variables[1]))
            for i in range(0, length):
                dfQuery, varName = pandaGetOrMakeColumn(dfQuery, variables[0][i % lengthX])
                dfQuery, varName = pandaGetOrMakeColumn(dfQuery, variables[1][i % lengthY])

    try:
        source = ColumnDataSource(dfQuery)
    except:
        logging.error("Invalid source:", source)
    # define default options
    options = {
        'line': -1,
        'size': 2,
        'tools': 'pan,box_zoom, wheel_zoom,box_select,lasso_select,reset',
        'tooltips': [],
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
        'palette': Spectral6,
        "marker": "square",
        "color": "#000000",
        "colors": 'Category10'
    }
    options.update(kwargs)
    plotArray = []
    colorAll = all_palettes[options['colors']]
    for i, variables in enumerate(figureArray):
        print(i, variables)
        if variables[0] == 'table':
            plotArray.append(makeBokehDataTable(dfQuery,source))
            continue
        figureI = figure(plot_width=options['plot_width'], plot_height=options['plot_height'], title="xxx",
                         tools=options['tools'], tooltips=options['tooltips'], x_axis_type=options['x_axis_type'], y_axis_type=options['y_axis_type'])

        # graphArray=drawGraphArray(df, variables)
        lengthX = len(variables[0])
        lengthY = len(variables[1])
        length = max(len(variables[0]), len(variables[1]))
        for i in range(0, length):
            dfQuery, varNameY = pandaGetOrMakeColumn(dfQuery, variables[1][i % lengthY])
            optionLocal = copy.copy(options)
            optionLocal['color'] = colorAll[max(length, 4)][i]
            if len(variables) > 2:
                print("Option", variables[2])
                optionLocal.update(variables[2])
            varX= variables[0][i % lengthX]
            varY= variables[1][i % lengthY]
            figureI.scatter(x=varX, y=varNameY, fill_alpha=1, source=source, size=optionLocal['size'], color=optionLocal["color"],
                            marker=optionLocal["marker"], legend=varY + " vs " + variables[0][i % lengthX]);
            figureI.xaxis.axis_label=dfQuery.metaData.get(varX+".AxisTitle",varX)
            figureI.yaxis.axis_label=dfQuery.metaData.get(varY+".AxisTitle",varY)

        figureI.legend.click_policy = "hide"
        plotArray.append(figureI)
    if len(options['layout']) > 0:  # make figure according layout
        x, layoutList, optionsLayout = processBokehLayout(options["layout"], plotArray)
        pAll = gridplotRow(layoutList, **optionsLayout)
        show(pAll)


def test_DrawFormula():
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    df.head(10)
    # testLayout = "((0,1),(2,x_visible=0),(3), plot_height=200,plot_width=800,commonX=3,commonY=3,y_visible=0)"
    df.metaData = {}
    # "A.title": "title A", "B.title": "title B","C":"C"}
    df.metaData['A.AxisTitle'] = "A (cm)"
    df.metaData['B.AxisTitle'] = "B (cm/s)"
    df.metaData['C.AxisTitle'] = "C (s)"
    df.metaData['D.AxisTitle'] = "D (a.u.)"
    print(df.metaData)
    #
    figureArray = [
        [['A'], ['C+A', 'C-A']],
        [['B'], ['C+B', 'D+B']],
        [['D'], ['sin(D/10)', 'sin(D/20)*0.5', 'sin(D/40)*0.25']],
        ['table']
    ]
    layout: str = '((0,1),(2),(3, x_visible=1),commonX=1,x_visible=1,y_visible=0,plot_height=250,plot_width=1000)'
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)")]
    bokehDrawArray(df, "A>0", figureArray, layout=layout, color="blue", size=4, tooltips=tooltips)


test_DrawFormula()
