from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar, HoverTool
from bokeh.transform import *
#from bokehTools import *
from bokeh.layouts import *
from bokeh.palettes import *
from bokeh.io import push_notebook
# import copy
import pyparsing
from IPython import get_ipython

# tuple of Bokeh markers
bokehMarkers = ["square", "circle", "triangle", "diamond", "squarecross", "circlecross", "diamondcross", "cross", "dash", "hex", "invertedtriangle",  "asterisk", "squareX","X"]


def __processBokehLayoutRow(layoutRow, figureList, layoutList, optionsMother, verbose=0):
    """
    :param layoutRow:
    :param figureList:
    :param layoutList:
    :param optionsMother:
    :param verbose:
    :return:
    """
    if verbose > 0: print("Raw", layoutRow)
    array = []
    layoutList.append(array)
    option = __processBokehLayoutOption(layoutRow)
    if verbose > 0: print("Option", option)
    for key in optionsMother:
        if not (key in option):
            option[key] = optionsMother[key]
    for idx, y in enumerate(layoutRow):
        if not y.isdigit(): continue
        fig = figureList[int(y)]
        array.append(fig)
        if 'commonY' in option:
            if type(option["commonY"]) == str:
                fig.y_range = array[0].y_range
            else:
                try:
                    fig.y_range = figureList[int(option["commonY"])].y_range
                except ValueError:
                    continue
        if 'commonX' in option:
            if type(option["commonX"]) == str:
                fig.x_range = array[0].x_range
            else:
                try:
                    fig.x_range = figureList[int(option["commonX"])].x_range
                except ValueError:
                    if verbose > 0: print('Failed: to process option ' + option["commonX"])

        if (idx > 0) & ('y_visible' in option): fig.yaxis.visible = bool(option["y_visible"])
        if 'x_visible' in option:     fig.xaxis.visible = bool(option["x_visible"])
    nCols = len(array)
    for fig in array:
        if 'plot_width' in option:
            fig.plot_width = int(option["plot_width"]/ nCols)
        if 'plot_height' in option:
            fig.plot_height = int(option["plot_height"])


def __processBokehLayoutOption(layoutOptions):
    """
    :param layoutOptions:
    :return:
    """
    # https://stackoverflow.com/questions/9305387/string-of-kwargs-to-kwargs
    options = {}
    for x in layoutOptions:
        if not (type(x) == str): continue
        if "=" in str(x):  # one of the way to see if it's list
            try:
                k, v = x.split("=")
            except ValueError:
                continue
            options[k] = v
            if v.isdigit():
                options[k] = int(v)
            else:
                try:
                    options[k] = float(v)
                except ValueError:
                    options[k] = v
    return options


def processBokehLayout(layoutString, figList, verbose=0):
    r"""
    :param layoutString:
        * layout string   see example
            https://github.com/miranov25/RootInteractiveTest/blob/870533dee18e528d0716a7e6feff8c8289c172dc/JIRA/PWGPP-485/parseLayout.ipynb
        * syntax:
            >>> layout="((row0),<(row1)>, ..., globalOptions)"
            >>> rowX="(id0,<id1>, ...,rowOptions)"
        * raw option derived from the global option, could be locally overwritten
        * layout options :
            >>> ["plot_width", "plot_height", "commonX", "commonY", "x_visible", "y_visible"]
        * Example layout syntax:
            >>> layout="((0,2,3,x_visible=1,y_visible=0), (1,plot_height=80, x_visible=0),"
            >>> layout+="(4,plot_height=80), plot_width=900, plot_height=200, commonY=1,commonX=1,x_visible=0)"
    :param figList:         array of figures to draw
    :param verbose:  verbosity
    :return: result as a string, layout list, options
    """
    # optionParse are propagated to daughter and than removed from global list
    optionsParse = ["plot_width", "plot_height", "commonX", "commonY", "x_visible", "y_visible"]
    theContent = pyparsing.Word(pyparsing.alphanums + ".+-=_") | pyparsing.Suppress(',')
    parents = pyparsing.nestedExpr('(', ')', content=theContent)
    res = parents.parseString(layoutString)[0]
    layoutList = []
    if verbose > 0: print(res)
    options = __processBokehLayoutOption(res)
    if verbose > 0: print(options)
    for x in res:
        if type(x) != str:
            __processBokehLayoutRow(x, figList, layoutList, options, verbose)
    for key in optionsParse:
        if key in options: del options[key]
    return res.asList(), layoutList, options


def drawColzArray(dataFrame, query, varX, varY, varColor, p, **options):
    r"""
    drawColzArray
    :param dataFrame: data frame
    :param query:
        selection e.g:
            >>> "varX>1&abs(varY)<2&A>0"
    :param varX:      x query
    :param varY:
        y query array of queries
            >>> "A:B:C:D:A"
    :param varColor:  z query
    :param p:         figure template
    :param options:
        optional drawing parameters
            * option - ncols - number fo columns in drawing
            * option - commonX=?,commonY=? - switch share axis
            * option - size
            * option errX  - query for errors on X
            * option errY  - array of queries for errors on Y
            * option tooltip - tooltip to show
    :return:
        figure, handle (for bokeh notebook), bokeh CDS
        drawing example - functionality like the tree->Draw( colz)
    :Example:
        https://github.com/miranov25/RootInteractiveTest/blob/master/JIRA/PWGPP-518/layoutPlay.ipynb
            >>>  df = pd.DataFrame(np.random.randint(0,100,size=(200, 4)), columns=list('ABCD'))
            >>>  drawColzArray(df,"A>0","A","A:B:C:D:A","C",None,ncols=2,plot_width=400,commonX=1, plot_height=200)
    """
    dfQuery = dataFrame.query(query)
    source = ColumnDataSource(dfQuery)
    mapper = linear_cmap(field_name=varColor, palette=Spectral6, low=min(dfQuery[varColor]), high=max(dfQuery[varColor]))
    isNotebook=get_ipython().__class__.__name__=='ZMQInteractiveShell'
    varYArray = varY.split(":")
    varXArray = varX.split(":")
    plotArray = []
    pFirst = None
    size = 2
    if 'line' in options.keys():
        line = options['line']
    else:
        line = 0
    if 'size' in options.keys(): size = options['size']
    tools = 'pan,box_zoom, wheel_zoom,box_select,lasso_select,reset'
    if 'tooltip' in options.keys(): tools = [HoverTool(tooltips=options['tooltip']), tools]
    if 'y_axis_type' in options.keys():
        y_axis_type = options['y_axis_type']
    else:
        y_axis_type = 'auto'
    if 'x_axis_type' in options.keys():
        x_axis_type = 'datetime'
    else:
        x_axis_type = 'auto'
    if 'errX' in options.keys(): varXerr = options['errX']
    if 'errY' in options.keys():
        varYerrArray = options['errY'].split(":")
    else:
        varYerrArray = varYArray
    plot_width = 400
    plot_height = 400
    if p:
        plot_width = p.plot_width
        plot_height = p.plot_height
    if 'plot_width' in options.keys(): plot_width = options['plot_width']
    if 'plot_height' in options.keys(): plot_height = options['plot_height']

    for idx, (yS, yErrorS) in enumerate(zip(varYArray, varYerrArray)):
        yArray=yS.strip('()').split(",")
        yArrayErr=yErrorS.strip('[]').split(",")
        p2 = figure(plot_width=plot_width, plot_height=plot_height, title=yS + " vs " + varX + "  Color=" + varColor, tools=tools, x_axis_type=x_axis_type, y_axis_type=y_axis_type)
        fIndex=0
        varX=varXArray[min(idx,len(varXArray)-1)]

        for y, yError in zip(yArray, yArrayErr):
            if 'varXerr' in locals():
                err_x_x = []
                err_x_y = []
                for coord_x, coord_y, x_err in zip(source.data[varX], source.data[y], source.data[varXerr]):
                    err_x_y.append((coord_y, coord_y))
                    err_x_x.append((coord_x - x_err, coord_x + x_err))
                p2.multi_line(err_x_x, err_x_y)
            if 'errY' in options.keys():
                err_y_x = []
                err_y_y = []
                for coord_x, coord_y, y_err in zip(source.data[varX], source.data[y], source.data[yError]):
                    err_y_x.append((coord_x, coord_x))
                    err_y_y.append((coord_y - y_err, coord_y + y_err))
                p2.multi_line(err_y_x, err_y_y)
            p2.scatter(x=varX, y=y, line_color=mapper, color=mapper, fill_alpha=1, source=source, size=size, marker=bokehMarkers[fIndex % 4])
            if line == 1: p2.line(x=varX, y=y, source=source)
            fIndex+=1

        if pFirst:
            if 'commonX' in options.keys(): p2.x_range = pFirst.x_range
            if 'commonY' in options.keys(): p2.y_range = pFirst.y_range
        else:
            pFirst = p2
        plotArray.append(p2)
        color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0, 0))
        p2.add_layout(color_bar, 'right')

    if 'layout' in options.keys():  # make figure according layout
        x, layoutList, optionsLayout = processBokehLayout(options["layout"], plotArray)
        pAll = gridplot(layoutList, **optionsLayout)
        handle = show(pAll, notebook_handle=isNotebook)
        return pAll, handle, source, plotArray

    nCols = 1
    if 'ncols' in options.keys():
        nCols = options['ncols']
    # nRows=len(plotArray)/ncols+1
    plotArray2D = []
    for i, plot in enumerate(plotArray):
        pRow = int(i / nCols)
        pCol = i % nCols
        if pCol == 0: plotArray2D.append([])
        plotArray2D[int(pRow)].append(plot)
    pAll = gridplot(plotArray2D)
    #    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    handle = show(pAll, notebook_handle=isNotebook)  # TODO make it OPTIONAL
    return pAll, handle, source, plotArray
