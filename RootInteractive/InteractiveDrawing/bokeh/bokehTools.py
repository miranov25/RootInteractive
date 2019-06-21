from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar, HoverTool
from bokeh.transform import *
from RootInteractive.Tools.aliTreePlayer import *
# from bokehTools import *
from bokeh.layouts import *
from bokeh.palettes import *
from bokeh.io import push_notebook
import logging
import pyparsing
from IPython import get_ipython
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import CustomJS, ColumnDataSource
from RootInteractive.Tools.pandaTools import *
import copy

# tuple of Bokeh markers
bokehMarkers = ["square", "circle", "triangle", "diamond", "squarecross", "circlecross", "diamondcross", "cross", "dash", "hex", "invertedtriangle", "asterisk", "squareX", "X"]


def makeJScallback(widgetDict, **kwargs):
    options = {
        "verbose": 0
    }
    options.update(kwargs)
    size = widgetDict['cdsOrig'].data["index"].size
    code = \
        """
    var dataOrig = cdsOrig.data;
    var dataSel = cdsSel.data;
    console.log('%f\t%f\t',dataOrig["index"].length, dataSel["index"].length);
    """
    for a in widgetDict['cdsOrig'].data:
        code += f"dataSel[\'{a}\']=[];\n"
    code += f"""var arraySize={size};\n"""
    code += """var nSelected=0;\n"""
    code += f"""for (var i = 0; i < {size}; i++)\n"""
    code += " {\n"
    code += """var isSelected=1;\n"""
    for key, value in widgetDict.items():
        if type(value).__name__ == "RangeSlider":
            dataName = key.replace("Range", "")
            code += f"      var {key}Value={key}.value;\n"
            code += f"      console.log(\"%s\t%f\t%f\t%f\",\"{key}\",{key}Value[0],{key}Value[1],dataOrig[\"{dataName}\"][i]);\n"
            code += f"      isSelected&=(dataOrig[\"{dataName}\"][i]>={key}Value[0])\n"
            code += f"      isSelected&=(dataOrig[\"{dataName}\"][i]<={key}Value[1])\n"
            # print(value)
    code += """      
        console.log(\"isSelected:%d\t%d\",i,isSelected);
        if (isSelected) nSelected++;
        if (isSelected){
    """
    for a in widgetDict['cdsOrig'].data:
        code += f"dataSel[\'{a}\'].push(dataOrig[\'{a}\'][i]);\n"
    code += """
        }
    }
    console.log(\"nSelected:%d\",nSelected); 
    cdsSel.change.emit();
    """
    if options["verbose"]>0:
        logging.info("makeJScallback:\n",code)
    callback = CustomJS(args=widgetDict, code=code)
    return callback


def __processBokehLayoutRow(layoutRow, figureList, layoutList, optionsMother, verbose=0):
    """
    :param layoutRow:
    :param figureList:
    :param layoutList:
    :param optionsMother:
    :param verbose:
    :return:
    """
    if verbose > 0: logging.info("Raw", layoutRow)
    array = []
    layoutList.append(array)
    option = __processBokehLayoutOption(layoutRow)
    if verbose > 0: logging.info("Option", option)
    for key in optionsMother:
        if not (key in option):
            option[key] = optionsMother[key]
    for idx, y in enumerate(layoutRow):
        if not y.isdigit(): continue
        try:
            fig = figureList[int(y)]
        except:
            logging.error("out of range index", y)
        array.append(fig)
        if type(fig).__name__ == 'DataTable':
            print("DataTable")
            continue
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
                    if verbose > 0: logging.info('Failed: to process option ' + option["commonX"])

        if (idx > 0) & ('y_visible' in option): fig.yaxis.visible = bool(option["y_visible"])
        if 'x_visible' in option:     fig.xaxis.visible = bool(option["x_visible"])
    nCols = len(array)
    for fig in array:
        if type(fig).__name__ == 'Figure':
            if 'plot_width' in option:
                fig.plot_width = int(option["plot_width"] / nCols)
            if 'plot_height' in option:
                fig.plot_height = int(option["plot_height"])
        if type(fig).__name__ == 'DataTable':
            if 'plot_width' in option:
                fig.width = int(option["plot_width"] / nCols)
            if 'plot_height' in option:
                fig.height = int(option["plot_height"])


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
    if verbose > 0: logging.info(res)
    options = __processBokehLayoutOption(res)
    if verbose > 0: logging.info(options)
    for x in res:
        if type(x) != str:
            __processBokehLayoutRow(x, figList, layoutList, options, verbose)
    for key in optionsParse:
        if key in options: del options[key]
    return res.asList(), layoutList, options


def gridplotRow(figList0, **options):
    """
    Make gridplot -resizing properly rows

    :param figList0: input array of figures
    :param options:
    :return:
    """
    figList = []
    for frow in figList0:
        figList.append([row(frow)])
    pAll = gridplot(figList, **options)
    return pAll


def makeBokehDataTable(dataFrame, source, **options):
    """
    Create widget for datatable

    :param dataFrame:
    input data frame
    :param source:
    :return:
    """
    columns = []
    for col in dataFrame.columns.values:
        title = dataFrame.metaData.get(col + ".OrigName", col);
        columns.append(TableColumn(field=col, title=title))
    data_table = DataTable(source=source, columns=columns, **options)
    return data_table


def drawColzArray(dataFrame, query, varX, varY, varColor, p, **kwargs):
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
    :param kwargs:
        optional drawing parameters
            * option           - ncols - number fo columns in drawing
            * option           - commonX=?,commonY=? - switch share axis
            * option           - size
            * option errX      - query for errors on X
            * option errY      - array of queries for errors on Y
            * option tooltip   - tooltip to show
    :return:
        figure, handle (for bokeh notebook), bokeh CDS
        drawing example - functionality like the tree->Draw( colz)
    :Example:
        https://github.com/miranov25/RootInteractiveTest/blob/master/JIRA/PWGPP-518/layoutPlay.ipynb
            >>>  df = pd.DataFrame(np.random.randint(0,100,size=(200, 4)), columns=list('ABCD'))
            >>>  drawColzArray(df,"A>0","A","A:B:C:D:A","C",None,ncols=2,plot_width=400,commonX=1, plot_height=200)
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
        'palette': Spectral6
    }
    if 'tooltip' in kwargs:  # bug fix - to be compatible with old interface (tooltip instead of tooltips)
        options['tooltips'] = kwargs['tooltip']
    options.update(kwargs)

    mapper = linear_cmap(field_name=varColor, palette=options['palette'], low=min(dfQuery[varColor]), high=max(dfQuery[varColor]))
    isNotebook = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    varYArray = varY.split(":")
    varXArray = varX.split(":")
    plotArray = []
    pFirst = None

    if len(options['errX']) > 1: varXerr = options['errX']
    if len(options['errY']) > 1:
        varYerrArray = options['errY'].split(":")
    else:
        varYerrArray = varYArray

    for idx, (yS, yErrorS) in enumerate(zip(varYArray, varYerrArray)):
        yArray = yS.strip('()').split(",")
        yArrayErr = yErrorS.strip('[]').split(",")
        p2 = figure(plot_width=options['plot_width'], plot_height=options['plot_height'], title=yS + " vs " + varX + "  Color=" + varColor,
                    tools=options['tools'], tooltips=options['tooltips'], x_axis_type=options['x_axis_type'], y_axis_type=options['y_axis_type'])
        fIndex = 0
        varX = varXArray[min(idx, len(varXArray) - 1)]

        for y, yError in zip(yArray, yArrayErr):
            if 'varXerr' in locals():
                err_x_x = []
                err_x_y = []
                for coord_x, coord_y, x_err in zip(source.data[varX], source.data[y], source.data[varXerr]):
                    err_x_y.append((coord_y, coord_y))
                    err_x_x.append((coord_x - x_err, coord_x + x_err))
                p2.multi_line(err_x_x, err_x_y)
            if 'errY' in kwargs.keys():
                err_y_x = []
                err_y_y = []
                for coord_x, coord_y, y_err in zip(source.data[varX], source.data[y], source.data[yError]):
                    err_y_x.append((coord_x, coord_x))
                    err_y_y.append((coord_y - y_err, coord_y + y_err))
                p2.multi_line(err_y_x, err_y_y)
            p2.scatter(x=varX, y=y, line_color=mapper, color=mapper, fill_alpha=1, source=source, size=options['size'], marker=bokehMarkers[fIndex % 4], legend=varX + y)
            if options['line'] > 0: p2.line(x=varX, y=y, source=source)
            p2.legend.click_policy = "hide"
            fIndex += 1

        if pFirst:  # set common X resp Y if specified. NOTE usage of layout options is more flexible
            if options['commonX'] > 0: p2.x_range = pFirst.x_range
            if options['commonY'] > 0: p2.y_range = pFirst.y_range
        else:
            pFirst = p2
        plotArray.append(p2)
        color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0, 0))
        p2.add_layout(color_bar, 'right')

    if len(options['layout']) > 0:  # make figure according layout
        x, layoutList, optionsLayout = processBokehLayout(options["layout"], plotArray)
        pAll = gridplotRow(layoutList, **optionsLayout)
        #handle = show(pAll, notebook_handle=isNotebook)
        return pAll, source, layoutList

    plotArray2D = []
    for i, plot in enumerate(plotArray):
        pRow = int(i / options['ncols'])
        pCol = i % options['ncols']
        if pCol == 0: plotArray2D.append([])
        plotArray2D[int(pRow)].append(plot)
    pAll = gridplot(plotArray2D)
    #    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    #handle = show(pAll, notebook_handle=isNotebook)  # set handle in case drawing is in notebook
    return pAll, source, plotArray


def parseWidgetString(widgetString):
    r'''
    Parse widget string and convert it to  nested lists
    :param widgetString:

    Example:
        https://github.com/miranov25/RootInteractiveTest/blob/master/JIRA/ADQT-3/tpcQADemoWithStatus.ipynb
            >>> from InteractiveDrawing.bokeh.bokehTools import *
            >>> widgets="tab.sliders(slider.meanMIP(45,55,0.1,45,55),slider.meanMIPele(50,80,0.2,50,80), slider.resolutionMIP(0,0.15,0.01,0,0.15)),"
            >>> widgets+="tab.checkboxGlobal(slider.global_Warning(0,1,1,0,1),checkbox.global_Outlier(0)),"
            >>> widgets+="tab.checkboxMIP(slider.MIPquality_Warning(0,1,1,0,1),checkbox.MIPquality_Outlier(0), checkbox.MIPquality_PhysAcc(1))"
            >>> print(parseWidgetString(widgets))
            >>> ['tab.sliders', ['slider.meanMIP', ['45', '55', '0.1', '45', '55'], 'slider.meanMIPele', ['50', '80', '0.2', '50', '80'], ....]

    :return:
        Nested lists of strings to create widgets
    '''
    toParse = "(" + widgetString + ")"
    theContent = pyparsing.Word(pyparsing.alphanums + ".+-_") | '#' | pyparsing.Suppress(',') | pyparsing.Suppress(':')
    widgetParser = pyparsing.nestedExpr('(', ')', content=theContent)
    widgetList = widgetParser.parseString(toParse)[0]
    return widgetList


def tree2Panda(tree, variables, selection, nEntries, firstEntry, columnMask):
    """
    :param tree:
    :param variables:
    :param selection:
    :param nEntries:
    :param firstEntry:
    :param columnMask:
    :return:
    """
    entries = tree.Draw(str(variables), selection, "goffpara", nEntries, firstEntry)  # query data
    columns = variables.split(":")
    # replace column names
    #    1.) pandas does not allow dots in names
    #    2.) user can specified own mask
    for i, iColumn in enumerate(columns):
        if columnMask == 'default':
            iColumn = iColumn.replace(".fElements", "").replace(".fX$", "X").replace(".fY$", "Y")
        else:
            masks = columnMask.split(":")
            for mask in masks:
                iColumn = iColumn.replace(mask, "")
        columns[i] = iColumn.replace(".", "_")

    ex_dict = {}
    for i, a in enumerate(columns):
        val = tree.GetVal(i)
        ex_dict[a] = np.frombuffer(val, dtype=float, count=entries)
    df = pd.DataFrame(ex_dict, columns=columns)
    for i, a in enumerate(columns):  # change type to time format if specified
        if (ROOT.TStatToolkit.GetMetadata(tree, a + ".isTime")):
            df[a] = pd.to_datetime(df[a], unit='s')
    return df


def bokehDrawArray(dataFrame, query, figureArray, **kwargs):
    """
    Wrapper bokeh draw array of figures

    :param dataFrame:         - input data frame
    :param query:             - query
    :param figureArray:       - figure array
    :param kwargs:
    :return:
        variable list:
            * pAll
            * handle
            * source
            * plotArray

    See example test:
        RootInteractive/InteractiveDrawing/bokeh/test_bokehDrawSA.py
    """
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
        "markers": bokehMarkers,
        "color": "#000000",
        "colors": 'Category10',
        "colorZvar":''
    }
    options.update(kwargs)
    dfQuery = dataFrame.query(query)
    dfQuery.metaData = dataFrame.metaData
    logging.info(dfQuery.metaData)
    # Check/resp. load derived variables
    i: int
    for i, variables in enumerate(figureArray):
        if len(variables) > 1:
            lengthX = len(variables[0])
            lengthY = len(variables[1])
            length = max(len(variables[0]), len(variables[1]))
            for j in range(0, length):
                dfQuery, varName = pandaGetOrMakeColumn(dfQuery, variables[0][j % lengthX])
                dfQuery, varName = pandaGetOrMakeColumn(dfQuery, variables[1][j % lengthY])

    try:
        source = ColumnDataSource(dfQuery)
    except:
        logging.error("Invalid source:", source)
    # define default options

    plotArray = []
    colorAll = all_palettes[options['colors']]
    for i, variables in enumerate(figureArray):
        logging.info(i, variables)
        if variables[0] == 'table':
            plotArray.append(makeBokehDataTable(dfQuery, source))
            continue
        figureI = figure(plot_width=options['plot_width'], plot_height=options['plot_height'], title="xxx",
                         tools=options['tools'], tooltips=options['tooltips'], x_axis_type=options['x_axis_type'], y_axis_type=options['y_axis_type'])

        # graphArray=drawGraphArray(df, variables)
        lengthX = len(variables[0])
        lengthY = len(variables[1])
        length = max(len(variables[0]), len(variables[1]))
        color_bar=None
        mapperC=None
        for i in range(0, length):
            dfQuery, varNameY = pandaGetOrMakeColumn(dfQuery, variables[1][i % lengthY])
            optionLocal = copy.copy(options)
            optionLocal['color'] = colorAll[max(length, 4)][i]
            optionLocal['marker']=optionLocal['markers'][i]
            if len(variables) > 2:
                logging.info("Option", variables[2])
                optionLocal.update(variables[2])
            varX = variables[0][i % lengthX]
            varY = variables[1][i % lengthY]
            if (len(optionLocal["colorZvar"])>0):
                logging.info(optionLocal["colorZvar"])
                varColor=optionLocal["colorZvar"]
                mapperC = linear_cmap(field_name=varColor, palette=options['palette'], low=min(dfQuery[varColor]), high=max(dfQuery[varColor]))
                optionLocal["color"]=mapperC
                color_bar = ColorBar(color_mapper=mapperC['transform'], width=8, location=(0, 0))

            figureI.scatter(x=varX, y=varNameY, fill_alpha=1, source=source, size=optionLocal['size'], color=optionLocal["color"],
                            marker=optionLocal["marker"], legend=varY + " vs " + variables[0][i % lengthX]);
            figureI.xaxis.axis_label = dfQuery.metaData.get(varX + ".AxisTitle", varX)
            figureI.yaxis.axis_label = dfQuery.metaData.get(varY + ".AxisTitle", varY)
        if color_bar!=None:
            figureI.add_layout(color_bar, 'right')
        figureI.legend.click_policy = "hide"
        plotArray.append(figureI)
    if len(options['layout']) > 0:  # make figure according layout
        x, layoutList, optionsLayout = processBokehLayout(options["layout"], plotArray)
        pAll = gridplotRow(layoutList, **optionsLayout)

    return pAll, source, layoutList, dfQuery
