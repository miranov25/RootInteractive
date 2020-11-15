from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar, HoverTool, CDSView, GroupFilter, VBar, HBar
from bokeh.transform import *
from RootInteractive.Tools.aliTreePlayer import *
# from bokehTools import *
from bokeh.layouts import *
from bokeh.palettes import *
from bokeh.io import push_notebook
import logging
import pyparsing
from IPython import get_ipython
from bokeh.models.widgets import *
from bokeh.models import CustomJS, ColumnDataSource
from RootInteractive.Tools.pandaTools import *
from RootInteractive.InteractiveDrawing.bokeh.bokehVisJS3DGraph import BokehVisJSGraph3D
import copy

# tuple of Bokeh markers
bokehMarkers = ["square", "circle", "triangle", "diamond", "squarecross", "circlecross", "diamondcross", "cross",
                "dash", "hex", "invertedtriangle", "asterisk", "squareX", "X"]


def makeJScallback(widgetDict, **kwargs):
    options = {
        "verbose": 0,
        "varList": ['AAA', 'BBB'],
        "nPointRender": 100000
    }
    options.update(kwargs)

    size = widgetDict['cdsOrig'].data["index"].size
    code = \
        """
    let dataOrig = cdsOrig.data;
    let dataSel = cdsSel.data;
    console.log('%f\t%f\t',dataOrig.index.length, dataSel.index.length);
    """
    # for key in options['varList']:
    #    code += f"      var {key}K={key};\n"

    for a in widgetDict['cdsOrig'].data:
        code += f"dataSel[\'{a}\']=[];\n"

    code += f"""let arraySize={size};\n"""
    code += """let nSelected=0;\n"""
    code += f"""for (var i = 0; i < {size}; i++)\n"""
    code += " {\n"
    code += """let isSelected=1;\n"""
    code += """let idx=0;\n"""
    code += f"const nPointRender =  {options['nPointRender']};\n"
    code += f"const precision=0.000001;\n"
    for a in widgetDict['cdsOrig'].data:
        if a == "index":
            continue
        code += f"var v{a} =dataOrig[\"{a}\"][i];\n"
        # code += f"var {a} =dataOrig[\"{a}\"][i];\n"

    code += "let isOK = false;\n"
    for key, value in widgetDict.items():
        if isinstance(value, Slider):
            code += f"      var {key}Value={key}.value;\n"
            code += f"      var {key}Step={key}.step;\n"
            # code += f"     console.log(\"%s\t%f\t%f\t%f\",\"{key}\",{key}Value,{key}Step,dataOrig[\"{key}\"][i]);\n"
            code += f"      isSelected&=(dataOrig[\"{key}\"][i]>={key}Value-0.5*{key}Step)\n"
            code += f"      isSelected&=(dataOrig[\"{key}\"][i]<={key}Value+0.5*{key}Step)\n"
        elif isinstance(value, RangeSlider):
            code += f"      var {key}Value={key}.value;\n"
            # code += f"      console.log(\"%s\t%f\t%f\t%f\",\"{key}\",{key}Value[0],{key}Value[1],dataOrig[\"{key}\"][i]);\n"
            code += f"      isSelected&=(dataOrig[\"{key}\"][i]>={key}Value[0])\n"
            code += f"      isSelected&=(dataOrig[\"{key}\"][i]<={key}Value[1])\n"
        elif isinstance(value, TextInput):
            code += f"      var queryText={key}.value;\n"
            # code += f"      console.log(queryText, queryText.length);\n"
            code += "      if (queryText.length > 1)  {"
            code += f"      var queryString='';\n"
            code += f"      var varString='';\n"
            code += f"     eval(varString+ 'var result = ('+ queryText+')');\n"
            # code += f"      console.log(\"query\", {key}, {key}.value, \"Result=\", varString, queryText, result, vA);\n"
            code += f"      isSelected&=result;\n"
            code += "}\n"
        elif isinstance(value, Select):
            # check if entry is equat to selected within relitive precission
            code += f"      var {key}Value={key}.value;\n"
            # code += f"     console.log(\"%s\t%s\t%f\",\"{key}\", {key}Value, dataOrig[\"{key}\"][i]);\n"
            code += f"      isOK=Math.abs((dataOrig[\"{key}\"][i]-{key}Value))<={key}Value*precision;\n"
            code += f"      isSelected&=(dataOrig[\"{key}\"][i]=={key}Value)|isOK;\n"
        elif isinstance(value, MultiSelect):
            code += f"      var {key}Value={key}.value;\n"
            code += f"     console.log(\"%s\t%s\t%f\t%s\",\"{key}\",{key}Value.toString,dataOrig[\"{key}\"][i],({key}Value.includes(dataOrig[\"{key}\"][i].toString())));\n"
            code += f"      isSelected&=({key}Value.includes(dataOrig[\"{key}\"][i].toString()))\n"
        elif isinstance(value, CheckboxGroup):
            code += f"      var {key}Value=({key}.active.length>0);\n"
            code += f"      isOK=Math.abs((dataOrig[\"{key}\"][i]-{key}Value))<={key}Value*precision;\n"
            # code += f"     console.log(\"%s\t%f\t%f\t%f\",\"{key}\",{key}Value,dataOrig[\"{key}\"][i]);\n"
            code += f"      isSelected&=((dataOrig[\"{key}\"][i]=={key}Value))|isOK;\n"
    code += """
        //console.log(\"isSelected:%d\t%d\",i,isSelected);
        if (isSelected){
          if(nSelected < nPointRender){
            for (const key in dataSel){
                dataSel[key].push(dataOrig[key][i]);
            }
            } else {
                if(Math.random() < 1 / nSelected){
                    idx = Math.floor(Math.random()*nPointRender);
                    for (const key in dataSel){
                        dataSel[key][idx] = dataOrig[key][i];
                    }
                }
            }
            nSelected++;
        }
    }
    console.log(\"nSelected:%d\",nSelected);
    cdsSel.change.emit();
    """
    if options["verbose"] > 0:
        logging.info("makeJScallback:\n", code)
    # print(code)
    callback = CustomJS(args=widgetDict, code=code)
    return callback


def makeJScallbackOptimized(widgetDict, cdsOrig, cdsSel, **kwargs):
    options = {
        "verbose": 0,
        "nPointRender": 100000,
        "cmapDict": None
    }
    options.update(kwargs)

    code = \
        """
    const dataOrig = cdsOrig.data;
    let dataSel = cdsSel.data;
    // console.log('%f\t%f\t',dataOrig.index.length, dataSel.index.length);
    const nPointRender = options.nPointRender;
    let nSelected=0;
    for (const i in dataSel){
        dataSel[i] = [];
    }
    const precision = 0.000001;
    const size = dataOrig.index.length;
    let selectedPointsBuffer = new ArrayBuffer(size);
    let isSelected = new Uint8Array(selectedPointsBuffer);
    let permutationFilter = [];
    for(let i=0; i<size; i++){
        isSelected[i] = 1;
    }
    for (const key in widgetDict){
        const widget = widgetDict[key];
        const widgetType = widget.type;
        if(widgetType == "Slider"){
            const col = dataOrig[key];
            const widgetValue = widget.value;
            const widgetStep = widget.step;
            for(let i=0; i<size; i++){
                isSelected[i] &= (col[i] >= widgetValue-0.5*widgetStep);
                isSelected[i] &= (col[i] <= widgetValue+0.5*widgetStep);
            }
        }
        if(widgetType == "RangeSlider"){
            const col = dataOrig[key];
            const low = widget.value[0];
            const high = widget.value[1];
            for(let i=0; i<size; i++){
                isSelected[i] &= (col[i] >= low);
                isSelected[i] &= (col[i] <= high);
            }
        }
        if(widgetType == "Select"){
            const col = dataOrig[key];
            const widgetValue = widget.value;
            for(let i=0; i<size; i++){
                let isOK = Math.abs(col[i] - widgetValue) <= widgetValue * precision;
                isSelected[i] &= (col[i] == widgetValue) | isOK;
            }
        }
        if(widgetType == "MultiSelect"){
            const col = dataOrig[key];
            const widgetValue = widget.value;
            for(let i=0; i<size; i++){
                isSelected[i] &= (widgetValue.includes(col[i].toString()));
            }
        }
        if(widgetType == "CheckboxGroup"){
            const col = dataOrig[key];
            const widgetValue = widget.value;
            for(let i=0; i<size; i++){
                isOK = Math.abs(col[i] - widgetValue) <= widgetValue * precision;
                isSelected &= (col[i] == widgetValue) | isOK;
            }
        }
        // This is broken, to be fixed later.
/*        if(widgetType == "TextInput"){
            const widgetValue = widget.value;
             if (queryText.length > 1)  {
                let queryString='';
                let varString='';
                eval(varString+ 'var result = ('+ queryText+')');
                for(let i=0; i<size; i++){
                    isSelected[i] &= result[i];
                }
             }
        }*/
    }
    for (let i = 0; i < size; i++){
        let randomIndex = 0;
        if (isSelected[i]){
            if(nSelected < nPointRender){
                permutationFilter.push(i);
            } else if(Math.random() < 1 / nSelected) {
                randomIndex = Math.floor(Math.random()*nPointRender);
                permutationFilter[randomIndex] = i;
            }
            nSelected++;
        }
    }
    nSelected = Math.min(nSelected, nPointRender);
    for (const key in dataSel){
        const colSel = dataSel[key];
        const colOrig = dataOrig[key];
        for(let i=0; i<nSelected; i++){
            colSel[i] = colOrig[permutationFilter[i]];
        }
    }
    const cmapDict = options.cmapDict;
    if (cmapDict !== undefined && nSelected !== 0){
        for(const key in cmapDict){
            const cmapList = cmapDict[key];
            const col = dataSel[key];
            const low = col.reduce((acc, cur)=>Math.min(acc,cur),col[0]);
            const high = col.reduce((acc, cur)=>Math.max(acc,cur),col[0]);
            for(let i=0; i<cmapList.length; i++){
                cmapList[i].transform.high = high;
                cmapList[i].transform.low = low;
                cmapList[i].transform.change.emit();
            }
        }
    }
    console.log(\"nSelected:%d\",nSelected);
    cdsSel.change.emit();
    """
    if options["verbose"] > 0:
        logging.info("makeJScallback:\n", code)
    # print(code)
    callback = CustomJS(args={'widgetDict': widgetDict, 'cdsOrig': cdsOrig, 'cdsSel': cdsSel, 'options': options},
                        code=code)
    return callback


def makeJSCallbackVisible(widgetDict, **kwargs):
    """
    make callback function to change of figure elements visible
    elements:
        * legend       visibility
        * axis title   visibility
        * legend size , axis size - should be similar to other function
    :param widgetDict:
    :param kwargs:
    :return:
    """
    options = {
        "verbose": 0,
        "element":"legend",
        "keyCheck":""
    }
    options.update(kwargs)
    code="console.log('Event occurred at x-position: ' + cb_obj.x)"
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
            continue
        if 'commonY' in option:
            if option["commonY"] >= 0:
                try:
                    fig.y_range = figureList[int(option["commonY"])].y_range
                except ValueError:
                    logging.info('Failed: to process option ' + option["commonY"])
                    continue
                except AttributeError:
                    logging.info('Failed: to process option ' + option["commonY"])
                    continue
        if 'commonX' in option:
            if option["commonX"] >= 0:
                try:
                    fig.x_range = figureList[int(option["commonX"])].x_range
                except ValueError:
                    if verbose > 0: logging.info('Failed: to process option ' + option["commonX"])
                    continue
                except AttributeError:
                    logging.info('Failed: to process option ')
                    continue

        if (idx > 0) & ('y_visible' in option):
            fig.yaxis.visible = bool(option["y_visible"]==1)
        if (idx == 0) & ('y_visible' in option):
            fig.yaxis.visible = bool(option["y_visible"]!=0)
        if 'x_visible' in option:
            fig.xaxis.visible = bool(option["x_visible"]==1)
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
        if type(fig).__name__ == 'BokehVisJSGraph3D':
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


def processBokehLayoutArray(widgetLayoutDesc, widgetArray):
    """
    apply layout on plain array of bokeh figures, resp. interactive widgets
    :param widgetLayoutDesc: array desciption of layout
    :param widgetArray: input plain array of widgets/figures
    :return: combined figure
    Example:  in tutorial/bokehDraw/makePandaWidgets.ipynb
    widgetLayoutDesc=[
        [0,1,2],
        [3,4,5],
        [6,7],
        {'width':10,'sizing_mode':'scale_width'}
    ]
    figureLayoutDesc=[
        [0,1,2, {'commonX':1,'y_visible':2, 'plot_height':300}],
        [2, {'commonX':1, 'y_visible':0}],
        {'width':10,'plot_height':200, 'sizing_mode':'scale_width'}
    ]
    """
    options = {
        'commonX': -1, 'commonY': -1,
        'x_visible': 1, 'y_visible': 1,
        'plot_width': -1, 'plot_height': -1,
        'sizing_mode': 'scale_width',
        'legend_visible': True
    }

    widgetRows = []
    nRows = len(widgetArray)
    # get/apply global options if exist
    if isinstance(widgetLayoutDesc[-1], dict):
        nRows -= 1
        options.update(widgetLayoutDesc[-1])
        widgetLayoutDesc = widgetLayoutDesc[0:-1]

    for rowWidget in widgetLayoutDesc:
        rowOptions = {}
        rowOptions.update(options)
        # patch local option
        if isinstance(rowWidget[-1], dict):
            rowOptions.update(rowWidget[-1])
            rowWidget = rowWidget[0:-1]
        rowWidgetArray0 = []
        for i, iWidget in enumerate(rowWidget):
            figure = widgetArray[iWidget]
            rowWidgetArray0.append(figure)
            if hasattr(figure, 'x_range'):
                if rowOptions['commonX'] >= 0:
                    figure.x_range = widgetArray[int(rowOptions["commonX"])].x_range
                if rowOptions['commonY'] >= 0:
                    figure.y_range = widgetArray[int(rowOptions["commonY"])].y_range
                if rowOptions['x_visible'] == 0:
                    figure.xaxis.visible = False
                else:
                     figure.xaxis.visible = True
                #figure.xaxis.visible = bool(rowOptions["x_visible"])
                if rowOptions['y_visible'] == 0:
                    figure.yaxis.visible = False
                if rowOptions['y_visible'] == 2:
                    if i > 0: figure.yaxis.visible = False
            if hasattr(figure, 'plot_width'):
                if rowOptions["plot_width"] > 0:
                    plot_width = int(rowOptions["plot_width"] / len(rowWidget))
                    figure.plot_width = plot_width
                if rowOptions["plot_height"] > 0:
                    figure.plot_height = rowOptions["plot_height"]
                figure.legend.visible = rowOptions["legend_visible"]
            if type(figure).__name__ == "DataTable":
                figure.height = int(rowOptions["plot_height"])
            if type(figure).__name__ == "BokehVisJSGraph3D":
                if rowOptions["plot_width"] > 0:
                    plot_width = int(rowOptions["plot_width"] / len(rowWidget))
                    figure.width = plot_width
                if rowOptions["plot_height"] > 0:
                    figure.height = rowOptions["plot_height"]

        rowWidgetArray = row(rowWidgetArray0, sizing_mode=rowOptions['sizing_mode'])
        widgetRows.append(rowWidgetArray)
    return column(widgetRows, sizing_mode=options['sizing_mode'])


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


def makeBokehDataTable(dataFrame, source, include, exclude, **kwargs):
    """
    Create widget for datatable

    :param dataFrame:
    input data frame
    :param source:
    :return:
    """
    columns = []
    for col in dataFrame.columns.values:
        isOK = True
        if hasattr(dataFrame, "meta"):
            title = dataFrame.meta.metaData.get(col + ".OrigName", col);
        else:
            title = col
        if include:
            isOK = False
            if re.match(include, col):
                isOK = True
        if exclude:
            if re.match(exclude, col):
                isOK = False
        if isOK:
            columns.append(TableColumn(field=col, title=title))
    data_table = DataTable(source=source, columns=columns, **kwargs)
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
        'palette': Spectral6,
        'doDraw': 0
    }
    options.update(kwargs)
    if 'tooltip' in kwargs:  # bug fix - to be compatible with old interface (tooltip instead of tooltips)
        options['tooltips'] = kwargs['tooltip']
        options['tooltip'] = kwargs['tooltip']

    mapper = linear_cmap(field_name=varColor, palette=options['palette'], low=min(dfQuery[varColor]),
                         high=max(dfQuery[varColor]))
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
        p2 = figure(plot_width=options['plot_width'], plot_height=options['plot_height'],
                    title=yS + " vs " + varX + "  Color=" + varColor,
                    tools=options['tools'], tooltips=options['tooltips'], x_axis_type=options['x_axis_type'],
                    y_axis_type=options['y_axis_type'])
        fIndex = 0
        varX = varXArray[min(idx, len(varXArray) - 1)]
        p2.xaxis.axis_label= varX
        #figureI.yaxis.axis_label = yAxisTitle
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
            p2.scatter(x=varX, y=y, line_color=mapper, color=mapper, fill_alpha=1, source=source, size=options['size'],
                       marker=bokehMarkers[fIndex % 4], legend_label=varX + y)
            if options['line'] > 0: p2.line(x=varX, y=y, source=source)
            p2.legend.click_policy = "hide"
            p2.yaxis.axis_label= y
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
        # handle = show(pAll, notebook_handle=isNotebook)
        if options['doDraw'] > 0:
            show(pAll)
        return pAll, source, layoutList

    plotArray2D = []
    for i, plot in enumerate(plotArray):
        pRow = int(i / options['ncols'])
        pCol = i % options['ncols']
        if pCol == 0: plotArray2D.append([])
        plotArray2D[int(pRow)].append(plot)
    pAll = gridplot(plotArray2D)
    if options['doDraw'] > 0:
        show(pAll)
    #    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    # handle = show(pAll, notebook_handle=isNotebook)  # set handle in case drawing is in notebook
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
    theContent = pyparsing.Word(pyparsing.alphanums + ".+-_[]{}") | '#' | pyparsing.Suppress(',') | pyparsing.Suppress(
        ':')
    widgetParser = pyparsing.nestedExpr('(', ')', content=theContent)
    widgetList = widgetParser.parseString(toParse)[0]
    return widgetList


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
        'tools': 'pan,box_zoom, wheel_zoom,box_select,lasso_select,reset,save',
        'tooltips': [],
        'y_axis_type': 'auto',
        'x_axis_type': 'auto',
        'plot_width': 600,
        'plot_height': 400,
        'errX': '',
        'errY': '',
        'commonX': 0,
        'commonY': 0,
        'ncols': -1,
        'layout': '',
        'widgetLayout': '',
        'palette': Spectral6,
        "marker": "square",
        "markers": bokehMarkers,
        "color": "#000000",
        "colors": 'Category10',
        "colorZvar": '',
        "rescaleColorMapper": False,
        "filter": '',
        'doDraw': 0,
        'nPointRender': 100000
    }
    options.update(kwargs)
    dfQuery = dataFrame.query(query)
    if hasattr(dataFrame, 'metaData'):
        dfQuery.metaData = dataFrame.metaData
        logging.info(dfQuery.metaData)
    # Check/resp. load derived variables
    i: int
    for i, variables in enumerate(figureArray):
        if len(variables) > 1 and variables[0] is not "table":
            lengthX = len(variables[0])
            lengthY = len(variables[1])
            length = max(len(variables[0]), len(variables[1]))
            if len(variables) > 2:
                optionLocal = options.copy()
                optionLocal.update(variables[2])
            else:
                optionLocal = options
            for j in range(0, length):
                dfQuery, varNameX = pandaGetOrMakeColumn(dfQuery, variables[0][j % lengthX])
                dfQuery, varNameY = pandaGetOrMakeColumn(dfQuery, variables[1][j % lengthY])
                if ('errY' in optionLocal.keys()) & (optionLocal['errY'] !=''):
                    seriesErrY = dfQuery.eval(optionLocal['errY'])
                    if varNameY+'_lower' not in dfQuery.columns:
                        seriesLower = dfQuery[varNameY]-seriesErrY
                        dfQuery[varNameY+'_lower'] = seriesLower
                    if varNameY+'_upper' not in dfQuery.columns:
                        seriesUpper = dfQuery[varNameY]+seriesErrY
                        dfQuery[varNameY+'_upper'] = seriesUpper
                if ('errX' in optionLocal.keys()) & (optionLocal['errX'] !=''):
                    seriesErrX = dfQuery.eval(optionLocal['errX'])
                    if varNameX+'_lower' not in dfQuery.columns:
                        seriesLower = dfQuery[varNameX]-seriesErrX
                        dfQuery[varNameX+'_lower'] = seriesLower
                    if varNameX+'_upper' not in dfQuery.columns:
                        seriesUpper = dfQuery[varNameX]+seriesErrX
                        dfQuery[varNameX+'_upper'] = seriesUpper

    try:
        #source = ColumnDataSource(dfQuery)
        source  = ColumnDataSource(dfQuery.sample(min(dfQuery.shape[0],options['nPointRender'])))
    except:
        logging.error("Invalid source:", source)
    # define default options

    plotArray = []
    colorAll = all_palettes[options['colors']]
    colorMapperDict = {}
    if isinstance(figureArray[-1], dict):
        options.update(figureArray[-1])
    for i, variables in enumerate(figureArray):
        logging.info("%d\t%s",i, variables)
        if isinstance(variables, dict): continue
        if variables[0] == 'table':
            TOptions = {
                'include': '',
                'exclude': ''
            }
            if len(variables) > 1:
                TOptions.update(variables[1])
            plotArray.append(makeBokehDataTable(dfQuery, source, TOptions['include'], TOptions['exclude']))
            continue
        xAxisTitle = ""
        yAxisTitle = ""
        # zAxisTitle = ""
        plotTitle = ""
        for varY in variables[1]:
            if hasattr(dfQuery, "meta"):
                yAxisTitle += dfQuery.meta.metaData.get(varY + ".AxisTitle", varY)
            else:
                yAxisTitle += varY
            yAxisTitle += ','
        for varX in variables[0]:
            if hasattr(dfQuery, "meta"):
                xAxisTitle += dfQuery.meta.metaData.get(varX + ".AxisTitle", varX)
            else:
                xAxisTitle += varX
            xAxisTitle += ','
        xAxisTitle = xAxisTitle[:-1]
        yAxisTitle = yAxisTitle[:-1]
        plotTitle += yAxisTitle + " vs " + xAxisTitle

        optionLocal = copy.copy(options)
        if len(variables) > 2:
            logging.info("Option %s", variables[2])
            optionLocal.update(variables[2])
        if 'varZ' in optionLocal.keys():
            dfQuery, varNameY = pandaGetOrMakeColumn(dfQuery, variables[1][0])
            _, varNameX = pandaGetOrMakeColumn(dfQuery, variables[0][0])
            _, varNameZ = pandaGetOrMakeColumn(dfQuery, optionLocal['varZ'])
            _, varNameColor = pandaGetOrMakeColumn(dfQuery, optionLocal['colorZvar'])
            options3D = {"width": "99%", "height": "99%"}
            plotI = BokehVisJSGraph3D(width=options['plot_width'], height=options['plot_height'],
                                      data_source=source, x=varNameX, y=varNameY, z=varNameZ, style=varNameColor, options3D=options3D)
            plotArray.append(plotI)
            continue
        else:
            figureI = figure(plot_width=options['plot_width'], plot_height=options['plot_height'], title=plotTitle,
                             tools=options['tools'], tooltips=options['tooltips'], x_axis_type=options['x_axis_type'],
                             y_axis_type=options['y_axis_type'])

        figureI.xaxis.axis_label = xAxisTitle
        figureI.yaxis.axis_label = yAxisTitle

        # graphArray=drawGraphArray(df, variables)
        lengthX = len(variables[0])
        lengthY = len(variables[1])
        length = max(len(variables[0]), len(variables[1]))
        color_bar = None
        mapperC = None
        for i in range(0, length):
            dfQuery, varNameY = pandaGetOrMakeColumn(dfQuery, variables[1][i % lengthY])
            dummy, varNameX = pandaGetOrMakeColumn(dfQuery, variables[0][i % lengthX])
            optionLocal = copy.copy(options)
            optionLocal['color'] = colorAll[max(length, 4)][i]
            optionLocal['marker'] = optionLocal['markers'][i]
            if len(variables) > 2:
                logging.info("Option %s", variables[2])
                optionLocal.update(variables[2])
            varX = variables[0][i % lengthX]
            varY = variables[1][i % lengthY]
            if (len(optionLocal["colorZvar"]) > 0):
                logging.info("%s",optionLocal["colorZvar"])
                varColor = optionLocal["colorZvar"]
                mapperC = linear_cmap(field_name=varColor, palette=options['palette'], low=min(dfQuery[varColor]),
                                      high=max(dfQuery[varColor]))
                optionLocal["color"] = mapperC
                if optionLocal["rescaleColorMapper"]:
                    if optionLocal["colorZvar"] in colorMapperDict:
                        colorMapperDict[optionLocal["colorZvar"]] += [mapperC]
                    else:
                        colorMapperDict[optionLocal["colorZvar"]] = [mapperC]
                color_bar = ColorBar(color_mapper=mapperC['transform'], width=8, location=(0, 0), title=varColor)
            #                zAxisTitle +=varColor + ","
            #            view = CDSView(source=source, filters=[GroupFilter(column_name=optionLocal['filter'], group=True)])
            figureI.scatter(x=varNameX, y=varNameY, fill_alpha=1, source=source, size=optionLocal['size'],
                            color=optionLocal["color"],
                            marker=optionLocal["marker"], legend_label=varY + " vs " + varX)
            if ('errX' in optionLocal.keys()) & (optionLocal['errX'] != ''):
                errorX = HBar(y=varNameY, height=0, left=varNameX+"_lower", right=varNameX+"_upper", line_color=optionLocal["color"])
                figureI.add_glyph(source, errorX)
            if ('errY' in optionLocal.keys()) & (optionLocal['errY'] != ''):
                errorY = VBar(x=varNameX, width=0, bottom=varNameY+"_lower", top=varNameY+"_upper", line_color=optionLocal["color"])
                figureI.add_glyph(source, errorY)
            #    errors = Band(base=varNameX, lower=varNameY+"_lower", upper=varNameY+"_upper",source=source)
            #    figureI.add_layout(errors)

        if color_bar != None:
            figureI.add_layout(color_bar, 'right')
        figureI.legend.click_policy = "hide"
        #        zAxisTitle=zAxisTitle[:-1]
        #        if(len(zAxisTitle)>0):
        #            plotTitle += " Color:" + zAxisTitle
        #        figureI.title = plotTitle
        plotArray.append(figureI)
    if isinstance(options['layout'], list):
        pAll = processBokehLayoutArray(options['layout'], plotArray)
        layoutList = [pAll]
    else:
        if len(options['layout']) > 0:  # make figure according layout
            x, layoutList, optionsLayout = processBokehLayout(options["layout"], plotArray)
            pAll = gridplotRow(layoutList, **optionsLayout)
    if options['doDraw'] > 0:
        show(pAll)
    return pAll, source, layoutList, dfQuery, colorMapperDict


def makeBokehSliderWidget(df, isRange, params, **kwargs):
    options = {
        'type': 'auto',
        'bins': 30,
        'sigma': 4,
        'limits': (0.05, 0.05),
        'title': '',
    }
    options.update(kwargs)
    name = params[0]
    title = params[0]
    if len(options['title']) > 0:
        title = options['title']
    start = 0
    end = 0
    step = 0
    if options['type'] == 'user':
        start = params[1], end = params[2], step = params[3], value = (params[4], params[5])
    elif (options['type'] == 'auto') | (options['type'] == 'minmax'):
        start = df[name].min()
        end = df[name].max()
        step = (end - start) / options['bins']
    elif (options['type'] == 'unique'):
        start = df[name].min()
        end = df[name].max()
        nbins=df[name].unique().size-1
        step = (end - start) / float(nbins)
    elif options['type'] == 'sigma':
        mean = df[name].mean()
        sigma = df[name].std()
        start = mean - options['sigma'] * sigma
        end = mean + options['sigma'] * sigma
        step = (end - start) / options['bins']
    elif options['type'] == 'sigmaMed':
        mean = df[name].median()
        sigma = df[name].std()
        start = mean - options['sigma'] * sigma
        end = mean + options['sigma'] * sigma
        step = (end - start) / options['bins']
    elif options['type'] == 'sigmaTM':
        mean = df[name].trimmed_mean(options['limits'])
        sigma = df[name].trimmed_std(options['limits'])
        start = mean - options['sigma'] * sigma
        end = mean + options['sigma'] * sigma
        step = (end - start) / options['bins']
    if isRange:
        value = (start, end)
        slider = RangeSlider(title=title, start=start, end=end, step=step, value=value)
    else:
        value = (start + end) * 0.5
        slider = Slider(title=title, start=start, end=end, step=step, value=value)
    return slider


def makeBokehSelectWidget(df, params, **kwargs):
    options = {'default': 0, 'size': 10}
    options.update(kwargs)
    # optionsPlot = []
    if len(params) == 1:
        optionsPlot = np.sort(df[params[0]].unique()).tolist()
    else:
        optionsPlot = params[1:]
    for i, val in enumerate(optionsPlot):
        optionsPlot[i] = str((val))
    return Select(title=params[0], value=optionsPlot[options['default']], options=optionsPlot)


def makeBokehMultiSelectWidget(df, params, **kwargs):
    # print("makeBokehMultiSelectWidget",params,kwargs)
    options = {'default': 0, 'size': 4}
    options.update(kwargs)
    # optionsPlot = []
    if len(params) == 1:
        optionsPlot = np.sort(df[params[0]].unique()).tolist()
    else:
        optionsPlot = params[1:]
    for i, val in enumerate(optionsPlot):
        optionsPlot[i] = str((val))
    # print(optionsPlot)
    return MultiSelect(title=params[0], value=optionsPlot, options=optionsPlot, size=options['size'])


def makeBokehCheckboxWidget(df, params, **kwargs):
    options = {'default': 0, 'size': 10}
    options.update(kwargs)
    # optionsPlot = []
    if len(params) == 1:
        optionsPlot = np.sort(df[params[0]].unique()).tolist()
    else:
        optionsPlot = params[1:]
    for i, val in enumerate(optionsPlot):
        optionsPlot[i] = str(val)
    return CheckboxGroup(labels=optionsPlot, active=[])


def makeBokehWidgets(df, widgetParams, cdsOrig, cdsSel, cmapDict=None, nPointRender=10000):
    widgetArray = []
    widgetDict = {}
    for widget in widgetParams:
        type = widget[0]
        params = widget[1]
        options = {}
        localWidget = None
        if len(widget) == 3:
            options = widget[2]
        if type == 'range':
            localWidget = makeBokehSliderWidget(df, True, params, **options)
        if type == 'slider':
            localWidget = makeBokehSliderWidget(df, False, params, **options)
        if type == 'select':
            localWidget = makeBokehSelectWidget(df, params, **options)
        if type == 'multiSelect':
            localWidget = makeBokehMultiSelectWidget(df, params, **options)
        # if type=='checkbox':
        #    localWidget=makeBokehCheckboxWidget(df,params,**options)
        if localWidget:
            widgetArray.append(localWidget)
        widgetDict[params[0]] = localWidget
    # callback = makeJScallback(widgetDict, nPointRender=nPointRender)
    callback = makeJScallbackOptimized(widgetDict, cdsOrig, cdsSel, cmapDict=cmapDict, nPointRender=nPointRender)
    for iWidget in widgetArray:
        if isinstance(iWidget, CheckboxGroup):
            iWidget.js_on_click(callback)
        elif isinstance(iWidget, Slider) or isinstance(iWidget, RangeSlider):
            iWidget.js_on_change("value", callback)
        else:
            iWidget.js_on_change("value", callback)
        iWidget.js_on_event("value", callback)
    return widgetArray
