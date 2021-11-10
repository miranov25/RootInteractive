from io import UnsupportedOperation
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar, HoverTool, VBar, HBar, Quad
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.widgets.tables import ScientificFormatter, DataTable
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
from RootInteractive.Tools.pandaTools import pandaGetOrMakeColumn
from RootInteractive.InteractiveDrawing.bokeh.bokehVisJS3DGraph import BokehVisJSGraph3D
from RootInteractive.InteractiveDrawing.bokeh.HistogramCDS import HistogramCDS
from RootInteractive.InteractiveDrawing.bokeh.HistoNdCDS import HistoNdCDS
import copy
from RootInteractive.Tools.compressArray import compressCDSPipe
from RootInteractive.InteractiveDrawing.bokeh.CDSCompress import CDSCompress
from RootInteractive.InteractiveDrawing.bokeh.HistoStatsCDS import HistoStatsCDS
from RootInteractive.InteractiveDrawing.bokeh.HistoNdProfile import HistoNdProfile
from RootInteractive.InteractiveDrawing.bokeh.DownsamplerCDS import DownsamplerCDS
import re

# tuple of Bokeh markers
bokehMarkers = ["square", "circle", "triangle", "diamond", "square_cross", "circle_cross", "diamond_cross", "cross",
                "dash", "hex", "invertedtriangle", "asterisk", "square_x", "x"]

# default tooltips for 1D and 2D histograms
defaultHistoTooltips = [
    ("range", "[@{bin_left}, @{bin_right}]"),
    ("count", "@bin_count")
]

defaultHisto2DTooltips = [
    ("range X", "[@{bin_bottom_0}, @{bin_top_0}]"),
    ("range Y", "[@{bin_bottom_1}, @{bin_top_1}]"),
    ("count", "@bin_count")
]


def makeJScallbackOptimized(widgetDict, cdsOrig, cdsSel, **kwargs):
    options = {
        "verbose": 0,
        "nPointRender": 10000,
        "cmapDict": None,
        "histogramList": []
    }
    options.update(kwargs)

    code = \
        """
    const t0 = performance.now();
    const dataOrig = cdsOrig.data;
    const nPointRender = options.nPointRender;
    let nSelected=0;
    const precision = 0.000001;
    const size = dataOrig.index.length;
    let isSelected = new Array(size);
    for(let i=0; i<size; ++i){
        isSelected[i] = true;
    }
    let permutationFilter = [];
    let indicesAll = [];
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
            let widgetValue = widget.value;
            widgetValue = widgetValue === "True" ? true : widgetValue;
            widgetValue = widgetValue === "False" ? false : widgetValue;
            for(let i=0; i<size; i++){
                let isOK = Math.abs(col[i] - widgetValue) <= widgetValue * precision;
                isOK|=(col[i] == widgetValue)
                isSelected[i] &= (col[i] == widgetValue) | isOK;
            }
        }
        if(widgetType == "MultiSelect"){
            const col = dataOrig[key];
            const widgetValue = widget.value.map((val)=>{
                if(val === "True") return true;
                if(val === "False") return false;
                if(!isNaN(val)) return Number(val);
                return val;
            });
            for(let i=0; i<size; i++){
                let isOK = widgetValue.reduce((acc,cur)=>acc|Math.abs(cur-col[i])<precision,0);
                if (!isOK){
                    isOK = widgetValue.reduce((acc,cur)=>acc|cur===col[i],0);
                }
                isSelected[i] &= isOK;
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
   
    const t1 = performance.now();
    console.log(`Filtering took ${t1 - t0} milliseconds.`);
    const view = options.view;
    const histogramList = options.histogramList
    if(histogramList != []){
        for (let i = 0; i < size; i++){
            if (isSelected[i]){
                indicesAll.push(i);
            }
        }
        for (const histo of histogramList){
            histo.view = indicesAll;
            histo.update_range();
        }
    }
    const t2 = performance.now();
    console.log(`Histogramming took ${t2 - t1} milliseconds.`);
    if(nPointRender > 0 && cdsSel != null){
        cdsSel.booleans = isSelected
        cdsSel.update()
        const t3 = performance.now();
        console.log(`Updating cds took ${t3 - t2} milliseconds.`);
    }
    if(options.cdsHistoSummary !== null){
        options.cdsHistoSummary.update();
    }
    console.log(\"nSelected:%d\",nSelected);
    """
    if options["verbose"] > 0:
        logging.info("makeJScallback:\n", code)
    # print(code)
    callback = CustomJS(args={'widgetDict': widgetDict, 'cdsOrig': cdsOrig, 'cdsSel': cdsSel, 'options': options},
                        code=code)
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
    if isinstance(widgetLayoutDesc, dict):
        tabs = []
        for i, iPanel in widgetLayoutDesc.items():
            tabs.append(Panel(child=processBokehLayoutArray(iPanel, widgetArray), title=i))
        return Tabs(tabs=tabs)
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
                if figure.legend:
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


def makeBokehHistoTable(histoDict, rowwise=False, **kwargs):
    histo_names = []
    histo_columns = []
    bin_centers = []
    edges_left = []
    edges_right = []
    sources = []
    quantiles = []
    compute_quantile = []
    sum_range = []

    if "formatter" in kwargs:
        formatter = kwargs["formatter"]
    else:
        formatter = ScientificFormatter(precision=3)

    for iHisto in histoDict:
        if histoDict[iHisto]["type"] == "histogram":
            histo_names.append(histoDict[iHisto]["name"])
            histo_columns.append("bin_count")
            bin_centers.append("bin_center")
            edges_left.append("bin_left")
            edges_right.append("bin_right")
            sources.append(histoDict[iHisto]["cds"])
            compute_quantile.append(True)
            if "quantiles" in histoDict[iHisto]:
                quantiles += histoDict[iHisto]["quantiles"]
            if "sum_range" in histoDict[iHisto]:
                sum_range += histoDict[iHisto]["sum_range"]
        elif histoDict[iHisto]["type"] in ["histo2d", "histoNd"]:
            for i in range(len(histoDict[iHisto]["variables"])):
                histo_names.append(histoDict[iHisto]["name"]+"_"+str(i))
                histo_columns.append("bin_count")
                bin_centers.append("bin_center_"+str(i))
                edges_left.append("bin_bottom_"+str(i))
                edges_right.append("bin_top_"+str(i))
                sources.append(histoDict[iHisto]["cds"])
                compute_quantile.append(False)

    quantiles = [*{*quantiles}]
    sum_range_uniq = []
    for i in sum_range:
        if i not in sum_range_uniq:
            sum_range_uniq.append(i)
    stats_cds = HistoStatsCDS(sources=sources, names=histo_names, bincount_columns=histo_columns, bin_centers=bin_centers,
                              quantiles=quantiles, compute_quantile=compute_quantile, rowwise=rowwise,
                              edges_left=edges_left, edges_right=edges_right, sum_range=sum_range_uniq)
    if rowwise:
        columns = [TableColumn(field="description")]
        for i in histo_names:
            columns.append(TableColumn(field=i, formatter=formatter))
        data_table = DataTable(source=stats_cds, columns=columns, **kwargs)
    else:
        columns = [TableColumn(field="name"), TableColumn(field="mean", formatter=formatter),
                   TableColumn(field="std", formatter=formatter), TableColumn(field="entries", formatter=formatter)]
        for (i, iQuantile) in enumerate(quantiles):
            columns.append(TableColumn(field="quantile_"+format(i), title="Quantile "+format(iQuantile),
                                       formatter=formatter))
        for (i, iBox) in enumerate(sum_range_uniq):
            columns.append(TableColumn(field="sum_"+format(i), title="Σ("+format(iBox[0])+","+format(iBox[1])+")",
                                       formatter=formatter))
            columns.append(TableColumn(field="sum_normed_"+format(i), title="Σ_normed("+format(iBox[0])+","+format(iBox[1])+")",
                                       formatter=formatter))
        data_table = DataTable(source=stats_cds, columns=columns, **kwargs)
    return stats_cds, data_table


def bokehDrawArray(dataFrame, query, figureArray, histogramArray=[], parameterArray=[], **kwargs):
    """
    Wrapper bokeh draw array of figures

    :param dataFrame:         - input data frame
    :param query:             - query
    :param figureArray:       - figure array
    :param histogramArray:    - (optional) histogram array
    :param parameterArray:    - (optional) parameter array for parameters controllable on client
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
        'histoTooltips': defaultHistoTooltips,
        'histo2dTooltips': defaultHisto2DTooltips,
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
        "color": None,
        "colors": 'Category10',
        "colorZvar": '',
        "rescaleColorMapper": False,
        "filter": '',
        'doDraw': 0,
        "legend_field": None,
        "legendTitle": None,
        'nPointRender': 10000,
        "nbins": 10,
        "weights": None,
        "histo2d": False,
        "range": None,
        "flip_histogram_axes": False,
        "show_histogram_error": False,
        "arrayCompression": None,
        "removeExtraColumns": True,
        "cdsDict": {},
        "cmapLow": None,
        "cmalHigh": None,
        "xAxisTitle": None,
        "yAxisTitle": None,
        "plotTitle": None
    }
    options.update(kwargs)
    if query is not None:
        dfQuery = dataFrame.query(query)
        if hasattr(dataFrame, 'metaData'):
            dfQuery.metaData = dataFrame.metaData
            logging.info(dfQuery.metaData)
    else:
        dfQuery = dataFrame.copy()
    # Check/resp. load derived variables
    i: int
    dfQuery, histogramDict, downsamplerColumns, \
    columnNameDict, parameterDict = makeDerivedColumns(dfQuery, figureArray, histogramArray=histogramArray,
                                                       parameterArray=parameterArray, options=options)

    plotArray = []
    colorAll = all_palettes[options['colors']]
    colorMapperDict = {}
    cdsHistoSummary = None

    cdsFull = None
    if options['arrayCompression'] is not None:
        print("compressCDSPipe")
        cdsCompress0, sizeMap= compressCDSPipe(dfQuery,options["arrayCompression"],1)
        cdsCompress=CDSCompress(inputData=cdsCompress0, sizeMap=sizeMap)
        cdsFull=cdsCompress
    else:
        try:
            cdsFull = ColumnDataSource(dfQuery)
        except:
            logging.error("Invalid source:", source)

    if downsamplerColumns:
        dummy_data = {}
        for i in downsamplerColumns:
            dummy_data[i] = []
        source = DownsamplerCDS(source=cdsFull, nPoints=options['nPointRender'], selectedColumns=downsamplerColumns, data=dummy_data)
    else:
        source = None

    histogramDict = bokehMakeHistogramCDS(dfQuery, cdsFull, histogramArray, histogramDict)
    cdsDict = options["cdsDict"]

    histoList = []
    profileList = []
    for i in histogramDict:
        if i not in cdsDict:
            cdsDict[i] = histogramDict[i]["cds"]
        if histogramDict[i]["type"] == "profile":
            profileList.append(histogramDict[i]["cds"])
        else:
            histoList.append(histogramDict[i]["cds"])

    paramDict = bokehMakeParameters(parameterArray, histogramArray, figureArray, variableList=list(columnNameDict))

    if isinstance(figureArray[-1], dict):
        options.update(figureArray[-1])
    for i, variables in enumerate(figureArray):
        logging.info("%d\t%s", i, variables)
        if isinstance(variables, dict):
            continue
        if variables[0] == 'table':
            TOptions = {
                'include': '',
                'exclude': ''
            }
            if len(variables) > 1:
                TOptions.update(variables[1])
            plotArray.append(makeBokehDataTable(dfQuery, source, TOptions['include'], TOptions['exclude']))
            continue
        if variables[0] == 'tableHisto':
            TOptions = {'rowwise': False}
            if len(variables) > 1:
                TOptions.update(variables[1])
            cdsHistoSummary, tableHisto = makeBokehHistoTable(histogramDict, rowwise=TOptions["rowwise"])
            plotArray.append(tableHisto)
            continue
        xAxisTitle = ""
        yAxisTitle = ""
        # zAxisTitle = ""
        plotTitle = ""

        for varY in variables[1]:
            if hasattr(dfQuery, "meta") and '.' not in varY:
                yAxisTitle += dfQuery.meta.metaData.get(varY + ".AxisTitle", varY)
            else:
                dfQuery, varNameY, cds_name = getOrMakeColumn(dfQuery, varY, None)
                yAxisTitle += getHistogramAxisTitle(histogramDict, varNameY, cds_name, False)
            yAxisTitle += ','
        for varX in variables[0]:
            if hasattr(dfQuery, "meta") and '.' not in varX:
                xAxisTitle += dfQuery.meta.metaData.get(varX + ".AxisTitle", varX)
            else:
                dfQuery, varNameX, cds_name = getOrMakeColumn(dfQuery, varX, None)
                xAxisTitle += getHistogramAxisTitle(histogramDict, varNameX, cds_name, False)
            xAxisTitle += ','
        xAxisTitle = xAxisTitle[:-1]
        yAxisTitle = yAxisTitle[:-1]

        optionLocal = copy.copy(options)
        if len(variables) > 2:
            logging.info("Option %s", variables[2])
            optionLocal.update(variables[2])

        if optionLocal["xAxisTitle"] is not None:
            xAxisTitle = optionLocal["xAxisTitle"]
        if optionLocal["yAxisTitle"] is not None:
            yAxisTitle = optionLocal["yAxisTitle"]
        plotTitle += yAxisTitle + " vs " + xAxisTitle
        if optionLocal["plotTitle"] is not None:
            plotTitle = optionLocal["plotTitle"]

        if 'varZ' in optionLocal.keys():
            dfQuery, varNameY, cds_name = getOrMakeColumn(dfQuery, variables[1][0], None)
            _, varNameX, cds_name = getOrMakeColumn(dfQuery, variables[0][0], cds_name)
            _, varNameZ, cds_name = getOrMakeColumn(dfQuery, optionLocal['varZ'], cds_name)
            _, varNameColor, cds_name = getOrMakeColumn(dfQuery, optionLocal['colorZvar'], cds_name)
            options3D = {"width": "99%", "height": "99%"}
            cds_used = source
            if cds_name is not None:
                cds_used = cdsDict[cds_name]
            plotI = BokehVisJSGraph3D(width=options['plot_width'], height=options['plot_height'],
                                      data_source=cds_used, x=varNameX, y=varNameY, z=varNameZ, style=varNameColor,
                                      options3D=options3D)
            plotArray.append(plotI)
            continue
        else:
            figureI = figure(plot_width=options['plot_width'], plot_height=options['plot_height'], title=plotTitle,
                             tools=options['tools'], x_axis_type=options['x_axis_type'],
                             y_axis_type=options['y_axis_type'])

        figureI.xaxis.axis_label = xAxisTitle
        figureI.yaxis.axis_label = yAxisTitle

        # graphArray=drawGraphArray(df, variables)
        lengthX = len(variables[0])
        lengthY = len(variables[1])
        length = max(len(variables[0]), len(variables[1]))
        color_bar = None
        mapperC = None
        cmap_cds_name = None
        if (len(optionLocal["colorZvar"]) > 0):
            #TODO: Support multiple color mappers, add more options, possibly use custom color mapper to improve performance
            #So far, parametrized colZ is only supported for the main CDS
            logging.info("%s", optionLocal["colorZvar"])
            colorZVar = optionLocal['colorZvar']
            if colorZVar in paramDict:
                colorZVar = paramDict[colorZVar]['value']
            _, varColor, cmap_cds_name = getOrMakeColumn(dfQuery, colorZVar, None)
            low = 0
            high = 1
            if cmap_cds_name is None:
                low = min(dfQuery[varColor])
                high=max(dfQuery[varColor])
            if "cmapLow" in optionLocal:
                low = optionLocal["cmapLow"]
            if "cmapHigh" in optionLocal:
                high = optionLocal["cmapHigh"]   
            if optionLocal["rescaleColorMapper"] or optionLocal["colorZvar"] in paramDict:
                if optionLocal["colorZvar"] in colorMapperDict:
                    mapperC = colorMapperDict[optionLocal["colorZvar"]]
                else:
                    mapperC = {"field": varColor, "transform": LinearColorMapper(palette=optionLocal['palette'])}
                    colorMapperDict[optionLocal["colorZvar"]] = mapperC
            else:
                mapperC = linear_cmap(field_name=varColor, palette=optionLocal['palette'], low=low, high=high)
            cds_used = source
            if cmap_cds_name is not None:
                cds_used = cdsDict[cmap_cds_name]
                # This is really hacky, will probably be removed when ND histogram joins start working
                if cmap_cds_name in histogramDict and histogramDict[cmap_cds_name]['type'] == 'profile' and varColor.split('_')[0] == 'bin':
                    histogramDict[cmap_cds_name]['cds'].js_on_change('change', CustomJS(code="""
                    const col = this.data[field]
                    const isOK = this.data.isOK
                    const low = col.map((x,i) => isOK[i] ? col[i] : Infinity).reduce((acc, cur)=>Math.min(acc,cur), Infinity);
                    const high = col.map((x,i) => isOK[i] ? col[i] : -Infinity).reduce((acc, cur)=>Math.max(acc,cur), -Infinity);
                    cmap.high = high;
                    cmap.low = low;
                    """, args={"field": mapperC["field"], "cmap": mapperC["transform"]})) 
            axis_title = getHistogramAxisTitle(histogramDict, varColor, cmap_cds_name)
            color_bar = ColorBar(color_mapper=mapperC['transform'], width=8, location=(0, 0), title=axis_title)
            if optionLocal['colorZvar'] in paramDict:
                paramDict[optionLocal['colorZvar']]["subscribed_events"].append(["value", color_bar, "title"])

        hover_tool_renderers = {}

        figure_cds_name = None

        for i in range(0, length):
            cds_name = None
            if variables[1][i % lengthY] in histogramDict:
                iHisto = histogramDict[variables[1][i % lengthY]]
                if iHisto["type"] == "histogram":
                    dfQuery, varNameY = pandaGetOrMakeColumn(dfQuery, iHisto["variables"][0])
                elif iHisto["type"] == "histo2d":
                    dfQuery, varNameX = pandaGetOrMakeColumn(dfQuery, iHisto["variables"][0])
                    dfQuery, varNameY = pandaGetOrMakeColumn(dfQuery, iHisto["variables"][1])
            else:
                dfQuery, varNameX, cds_name = getOrMakeColumn(dfQuery, variables[0][i % lengthX], cds_name)
                dfQuery, varNameY, cds_name = getOrMakeColumn(dfQuery, variables[1][i % lengthY], cds_name)
            if mapperC is not None and cds_name == cmap_cds_name:
                color = mapperC
            else:
                color = colorAll[max(length, 4)][i]
            if optionLocal['color'] is not None:
                color=optionLocal['color']
            try:
                marker = optionLocal['markers'][i]
            except:
                marker = optionLocal['markers']
            markerSize = optionLocal['size']
            if markerSize in paramDict:
                markerSize = paramDict[markerSize]['value']
            if len(variables) > 2:
                logging.info("Option %s", variables[2])
                optionLocal.update(variables[2])
            varX = variables[0][i % lengthX]
            varY = variables[1][i % lengthY]
            cds_used = source
            if cds_name is not None:
                cds_used = cdsDict[cds_name]

            if varY in histogramDict:
                histoHandle = histogramDict[varY]
                if histoHandle["type"] == "histogram":
                    colorHisto = colorAll[max(length, 4)][i]
                    addHistogramGlyph(figureI, histoHandle, marker, colorHisto, markerSize, optionLocal)
                elif histoHandle["type"] == "histo2d":
                    addHisto2dGlyph(figureI, varNameX, varNameY, histoHandle, colorMapperDict, color, marker, dfQuery,
                                    optionLocal)
            else:
                #                zAxisTitle +=varColor + ","
                #            view = CDSView(source=source, filters=[GroupFilter(column_name=optionLocal['filter'], group=True)])
                drawnGlyph = None
                colorMapperCallback = """
                glyph.fill_color={...glyph.fill_color, field:this.value}
                glyph.line_color={...glyph.line_color, field:this.value}
                """
                if optionLocal["legend_field"] is None:
                    x_label = getHistogramAxisTitle(histogramDict, varNameX, cds_name)
                    y_label = getHistogramAxisTitle(histogramDict, varNameY, cds_name)
                    drawnGlyph = figureI.scatter(x=varNameX, y=varNameY, fill_alpha=1, source=cds_used, size=markerSize,
                                color=color, marker=marker, legend_label=y_label + " vs " + x_label)
                else:
                    drawnGlyph = figureI.scatter(x=varNameX, y=varNameY, fill_alpha=1, source=cds_used, size=markerSize,
                                color=color, marker=marker, legend_field=optionLocal["legend_field"])
                if optionLocal["colorZvar"] in paramDict:
                    if len(color["transform"].domain) == 0:
                        color["transform"].domain = [(drawnGlyph, color["field"])]
                        # HACK: This changes the color mapper's domain, which only consists of one field. 
                        paramDict[optionLocal['colorZvar']]["subscribed_events"].append(["value", CustomJS(args={"transform": color["transform"]}, code="""
                            transform.domain[0] = [transform.domain[0][0], this.value]
                            transform.change.emit()
                        """)])
                    paramDict[optionLocal['colorZvar']]["subscribed_events"].append(["value", CustomJS(args={"glyph": drawnGlyph.glyph}, code=colorMapperCallback)])
                if optionLocal['size'] in paramDict:
                    paramDict[optionLocal['size']]["subscribed_events"].append(["value", drawnGlyph.glyph, "size"])
                if cds_name is None:
                    if "" not in hover_tool_renderers:
                        hover_tool_renderers[""] = []
                    hover_tool_renderers[""].append(drawnGlyph)
                elif cds_name in histogramDict:
                    if histogramDict[cds_name]["type"] == "profile":
                        if cds_name not in hover_tool_renderers:
                            hover_tool_renderers[cds_name] = []
                        hover_tool_renderers[cds_name].append(drawnGlyph)
                if ('errX' in optionLocal.keys()) and (optionLocal['errX'] != '') and (cds_name is None):
                    errorX = HBar(y=varNameY, height=0, left=varNameX+"_lower", right=varNameX+"_upper", line_color=color)
                    if optionLocal["colorZvar"] in paramDict:
                        paramDict[optionLocal['colorZvar']]["subscribed_events"].append(["value", CustomJS(args={"glyph": errorX}, code=colorMapperCallback)])
                    figureI.add_glyph(source, errorX)
                if ('errY' in optionLocal.keys()) and (optionLocal['errY'] != '') and (cds_name is None):
                    errorY = VBar(x=varNameX, width=0, bottom=varNameY+"_lower", top=varNameY+"_upper", line_color=color)
                    if optionLocal["colorZvar"] in paramDict:
                        paramDict[optionLocal['colorZvar']]["subscribed_events"].append(["value", CustomJS(args={"glyph": errorY}, code=colorMapperCallback)])
                    figureI.add_glyph(source, errorY)
                #    errors = Band(base=varNameX, lower=varNameY+"_lower", upper=varNameY+"_upper",source=source)
                #    figureI.add_layout(errors)
            if figure_cds_name is None:
                figure_cds_name = cds_name
            elif figure_cds_name != cds_name:
                figure_cds_name = ""

        if color_bar != None:
            figureI.add_layout(color_bar, 'right')
        for iCds, iRenderers in hover_tool_renderers.items():
            if iCds == "":
                tooltips = optionLocal["tooltips"]
            elif iCds in histogramDict and histogramDict[iCds]["type"] == "profile":
                profile_description = histogramDict[iCds]
                tooltips = defaultNDProfileTooltips(profile_description["variables"], profile_description["axis"],
                                                    profile_description["quantiles"],  profile_description["sum_range"])
            figureI.add_tools(HoverTool(tooltips=tooltips, renderers=iRenderers))
        if figureI.legend:
            figureI.legend.click_policy = "hide"
            if optionLocal["legendTitle"] is not None:
                logging.warn("legendTitle is deprecated, please use the 'title' field in 'legend_options'")
                figureI.legend.title = optionLocal["legendTitle"]
            elif figure_cds_name != "":
                figureI.legend.title = figure_cds_name
            if 'legend_options' in optionLocal:
                legend_options = optionLocal['legend_options'].copy()
                legend_options_parameters = {}
                for i, iOption in legend_options.items():
                    if iOption in parameterDict:
                        legend_options_parameters[i] = paramDict[iOption]
                for i, iOption in legend_options_parameters.items():
                    legend_options[i] = iOption['value']
                figureI.legend.update(**legend_options)
                for i, iOption in legend_options_parameters.items():
                    iOption["subscribed_events"].append(["value", figureI.legend[0], i])        
        #        zAxisTitle=zAxisTitle[:-1]
        #        if(len(zAxisTitle)>0):
        #            plotTitle += " Color:" + zAxisTitle
        #        figureI.title = plotTitle
        plotArray.append(figureI)
    if isinstance(options['layout'], list) or isinstance(options['layout'], dict):
        pAll = processBokehLayoutArray(options['layout'], plotArray)
        layoutList = [pAll]
    if options['doDraw'] > 0:
        show(pAll)
    return pAll, source, layoutList, dfQuery, colorMapperDict, cdsFull, histoList, cdsHistoSummary, profileList, paramDict


def addHisto2dGlyph(fig, x, y, histoHandle, colorMapperDict, color, marker, dfQuery, options):
    visualization_type = "heatmap"
    if "visualization_type" in options:
        visualization_type = options["visualization_type"]
    cdsHisto = histoHandle["cds"]

    tooltips = None
    if "tooltips" in histoHandle:
        tooltips = histoHandle["tooltips"]
    elif "tooltips" in options:
        tooltips = options["histo2dTooltips"]

    if visualization_type == "heatmap":
        # Flipping histogram axes probably doesn't make sense in this case.
        mapperC = {"field": "bin_count", "transform": LinearColorMapper(palette=options['palette'])}
        color_bar = ColorBar(color_mapper=mapperC['transform'], width=8, location=(0, 0),
                             title="Count")
        histoGlyph = Quad(left="bin_bottom_0", right="bin_top_0", bottom="bin_bottom_1", top="bin_top_1",
                          fill_color=mapperC)
        histoGlyphRenderer = fig.add_glyph(cdsHisto, histoGlyph)
        fig.add_layout(color_bar, 'right')
    elif visualization_type == "colZ":
        mapperC = {"field": "bin_count", "transform": LinearColorMapper(palette=options['palette'])}
        color_bar = ColorBar(color_mapper=mapperC['transform'], width=8, location=(0, 0),
                             title=y)
        if options["legend_field"] is None:
            histoGlyphRenderer = fig.scatter(x="bin_center_0", y="bin_count", fill_alpha=1, source=cdsHisto, size=options['size'],
                            color=mapperC, marker=marker, legend_label="Histogram of " + x)
        else:
            histoGlyphRenderer = fig.scatter(x="bin_center_0", y="bin_count", fill_alpha=1, source=cdsHisto, size=options['size'],
                            color=mapperC, marker=marker, legend_field=options["legend_field"])
        if "show_histogram_error" in options:
            errorbar = VBar(x="bin_center_0", width=0, top="errorbar_high", bottom="errorbar_low", line_color=mapperC)
            fig.add_glyph(cdsHisto, errorbar)
        fig.add_layout(color_bar, 'right')
    if tooltips is not None:
        fig.add_tools(HoverTool(renderers=[histoGlyphRenderer], tooltips=tooltips))


def addHistogramGlyph(fig, histoHandle, marker, colorHisto, size, options):
    cdsHisto = histoHandle["cds"]
    if options['color'] is not None:
        colorHisto = options['color']
    tooltips = None
    if "tooltips" in histoHandle:
        tooltips = histoHandle["tooltips"]
    elif "tooltips" in options:
        tooltips = options["histoTooltips"]
    visualization_type = "points"
    histoGlyphRenderer = None
    if "visualization_type" in options:
        visualization_type = options["visualization_type"]
    if visualization_type == "bars":
        if options['flip_histogram_axes']:
            histoGlyph = Quad(left=0, right="bin_count", bottom="bin_left", top="bin_right", fill_color=colorHisto)
        else:
            histoGlyph = Quad(left="bin_left", right="bin_right", bottom=0, top="bin_count", fill_color=colorHisto)
        histoGlyphRenderer = fig.add_glyph(cdsHisto, histoGlyph)
    elif visualization_type == "points":
        if options['flip_histogram_axes']:
            histoGlyphRenderer = fig.scatter(y="bin_center", x="bin_count", color=colorHisto, marker=marker, source=cdsHisto, size=size,
                        legend_label=histoHandle["variables"][0])
            if "show_histogram_error" in options:
                errorbar = HBar(y="bin_center", height=0, left="errorbar_low", right="errorbar_high", line_color=colorHisto)
                fig.add_glyph(cdsHisto, errorbar)
        else:
            histoGlyphRenderer = fig.scatter(x="bin_center", y="bin_count", color=colorHisto, marker=marker, source=cdsHisto, size=size,
                        legend_label=histoHandle["variables"][0])
            if "show_histogram_error" in options:
                errorbar = VBar(x="bin_center", width=0, top="errorbar_high", bottom="errorbar_low", line_color=colorHisto)
                fig.add_glyph(cdsHisto, errorbar)
    if tooltips is not None:
        fig.add_tools(HoverTool(renderers=[histoGlyphRenderer], tooltips=tooltips))

def makeBokehSliderWidget(df, isRange, params, paramDict, **kwargs):
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
    value=None
    if options['callback'] == 'parameter':
        if options['type'] == 'user':
            start = params[1], end = params[2], step = params[3], value = (params[4], params[5])
        else:
            param = paramDict[params[0]]
            start = param['range'][0]
            end = param['range'][1]
            bins = options['bins']   
            if 'bins' in param:
                bins = param['bins']
            if 'step' in param:
                step = param['step']
            else:
                step = (end - start) / bins
            value = paramDict[params[0]]["value"]
    else:
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
        if (start==end):
            start-=1
            end+=1
        if value is None:
            value = (start, end)
        slider = RangeSlider(title=title, start=start, end=end, step=step, value=value)
    else:
        if value is None:
            value = (start + end) * 0.5
        slider = Slider(title=title, start=start, end=end, step=step, value=value)
    return slider


def makeBokehSelectWidget(df, params, paramDict, **kwargs):
    options = {'default': 0, 'size': 10}
    options.update(kwargs)
    # optionsPlot = []
    if len(params) == 1:
        if options['callback'] == 'parameter':
            optionsPlot = paramDict[params[0]]["options"]
        else:
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
        try:
            optionsPlot = np.sort(df[params[0]].unique()).tolist()
        except:
            optionsPlot = sorted(df[params[0]].unique().tolist())
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


def makeBokehWidgets(df, widgetParams, cdsOrig, cdsSel, histogramList=[], cmapDict=None, cdsHistoSummary=None, profileList=None, paramDict={}, nPointRender=10000,cdsCompress=None):
    widgetArray = []
    widgetDict = {}
    options = {
        "callback": "selection"
    }
    for widget in widgetParams:
        type = widget[0]
        params = widget[1]
        optionLocal = options.copy()
        localWidget = None
        if len(widget) == 3:
            optionLocal.update(widget[2])
        if type == 'range':
            localWidget = makeBokehSliderWidget(df, True, params, paramDict, **optionLocal)
        if type == 'slider':
            localWidget = makeBokehSliderWidget(df, False, params, paramDict, **optionLocal)
        if type == 'select':
            localWidget = makeBokehSelectWidget(df, params, paramDict, **optionLocal)
        if type == 'multiSelect':
            localWidget = makeBokehMultiSelectWidget(df, params, **optionLocal)
        # if type=='checkbox':
        #    localWidget=makeBokehCheckboxWidget(df,params,**options)
        if localWidget:
            widgetArray.append(localWidget)
        if optionLocal["callback"] == "selection":
            widgetDict[params[0]] = localWidget
    callbackSel = makeJScallbackOptimized(widgetDict, cdsOrig, cdsSel, histogramList=histogramList,
                                       cmapDict=cmapDict, nPointRender=nPointRender,
                                       cdsHistoSummary=cdsHistoSummary, profileList=profileList, cdsCompress=cdsCompress)
    #callback = makeJScallbackOptimized(widgetDict, cdsOrig, cdsSel, histogramList=histogramList, cmapDict=cmapDict, nPointRender=nPointRender)
    for iDesc, iWidget in zip(widgetParams, widgetArray):
        optionLocal = options.copy()
        localWidget = None
        if len(iDesc) == 3:
            optionLocal.update(iDesc[2])       
        if optionLocal["callback"] == "selection":
            callback = callbackSel
        elif optionLocal["callback"] == "parameter":
            paramControlled = paramDict[iDesc[1][0]]
            for iEvent in paramControlled["subscribed_events"]:
                if len(iEvent) == 2:
                    iWidget.js_on_change(*iEvent)
                else:
                    iWidget.js_link(*iEvent)
            continue
        else:
            # TODO: Change this to custom JS callback
            callback = None
        if isinstance(iWidget, CheckboxGroup):
            iWidget.js_on_click(callback)
        elif isinstance(iWidget, Slider) or isinstance(iWidget, RangeSlider):
            iWidget.js_on_change("value", callback)
        else:
            iWidget.js_on_change("value", callback)
        iWidget.js_on_event("value", callback)
    return widgetArray


def bokehMakeHistogramCDS(dfQuery, cdsFull, histogramArray=[], histogramDict=None, parameterDict={}, **kwargs):
    options = {"range": None,
               "nbins": 10,
               "weights": None,
               "quantiles": [],
               "sum_range": []
               }
    histoDict = {}
    for iHisto in histogramArray:
        sampleVars = iHisto["variables"]
        histoName = iHisto["name"]
        if histogramDict is not None and not histogramDict[histoName]:
            continue
        optionLocal = copy.copy(options)
        optionLocal.update(iHisto)
        weights = None
        if optionLocal["weights"] is not None:
            _, weights = pandaGetOrMakeColumn(dfQuery, optionLocal["weights"])
        if len(sampleVars) == 1:
            _, varNameX = pandaGetOrMakeColumn(dfQuery, sampleVars[0])
            cdsHisto = HistogramCDS(source=cdsFull, nbins=optionLocal["nbins"],
                                    range=optionLocal["range"], sample=varNameX, weights=weights)
            histoDict[histoName] = iHisto.copy()
            histoDict[histoName].update({"cds": cdsHisto, "type": "histogram"})
        elif len(sampleVars) == 2:
            sampleVarNames = []
            for i in sampleVars:
                _, varName = pandaGetOrMakeColumn(dfQuery, i)
                sampleVarNames.append(varName)
            cdsHisto = HistoNdCDS(source=cdsFull, nbins=optionLocal["nbins"],
                                    range=optionLocal["range"], sample_variables=sampleVarNames, weights=weights)
            histoDict[histoName] = {"cds": cdsHisto, "type": "histo2d", "name": histoName,
                                    "variables": sampleVars}
            if "axis" in iHisto:
                axisIndices = iHisto["axis"]
                profilesDict = {}
                for i in axisIndices:
                    cdsProfile = HistoNdProfile(source=cdsHisto, axis_idx=i, quantiles=optionLocal["quantiles"],
                                                sum_range=optionLocal["sum_range"])
                    profilesDict[i] = cdsProfile
                    histoDict[histoName+"_"+str(i)] = {"cds": cdsProfile, "type": "profile", "name": histoName+"_"+str(i), "variables": sampleVars,
                    "quantiles": optionLocal["quantiles"], "sum_range": optionLocal["sum_range"], "axis": i}
                histoDict[histoName]["profiles"] = profilesDict
        else:
            sampleVarNames = []
            for i in sampleVars:
                _, varName = pandaGetOrMakeColumn(dfQuery, i)
                sampleVarNames.append(varName)
            cdsHisto = HistoNdCDS(source=cdsFull, nbins=optionLocal["nbins"],
                                    range=optionLocal["range"], sample_variables=sampleVarNames,
                                    weights=weights)
            histoDict[histoName] = {"cds": cdsHisto, "type": "histoNd", "name": histoName,
                                    "variables": sampleVars}
            if "axis" in iHisto:
                axisIndices = iHisto["axis"]
                profilesDict = {}
                for i in axisIndices:
                    cdsProfile = HistoNdProfile(source=cdsHisto, axis_idx=i, quantiles=optionLocal["quantiles"],
                                                sum_range=optionLocal["sum_range"])
                    profilesDict[i] = cdsProfile
                    histoDict[histoName+"_"+str(i)] = {"cds": cdsProfile, "type": "profile", "name": histoName+"_"+str(i), "variables": sampleVars,
                    "quantiles": optionLocal["quantiles"], "sum_range": optionLocal["sum_range"], "axis": i} 
                histoDict[histoName]["profiles"] = profilesDict

    return histoDict


def makeDerivedColumns(dfQuery, figureArray=None, histogramArray=None, parameterArray=None, widgetArray=None, options={}):
    histogramDict = {}
    columnNameDict = {}
    paramDict = {}
    downsamplerColumns = {}
    if histogramArray is not None:
        for i, histo in enumerate(histogramArray):
            histogramDict[histo["name"]] = True

    if parameterArray is not None:
          for i, param in enumerate(parameterArray):
            paramDict[param["name"]] = param

    if figureArray is not None:
        for i, variables in enumerate(figureArray):
            if len(variables) > 1 and variables[0] != "table" and variables[0] != "tableHisto":
                lengthX = len(variables[0])
                lengthY = len(variables[1])
                length = max(len(variables[0]), len(variables[1]))
                if len(variables) > 2:
                    optionLocal = options.copy()
                    optionLocal.update(variables[2])
                else:
                    optionLocal = options
                for j in range(0, length):
                    if variables[1][j % lengthY] not in histogramDict:
                        if '.' not in variables[0][j % lengthX]:
                            dfQuery, varNameX = pandaGetOrMakeColumn(dfQuery, variables[0][j % lengthX])
                            columnNameDict[varNameX] = True
                            downsamplerColumns[varNameX] = True
                        if '.' not in variables[1][j % lengthY]:
                            dfQuery, varNameY = pandaGetOrMakeColumn(dfQuery, variables[1][j % lengthY])
                            columnNameDict[varNameY] = True
                            downsamplerColumns[varNameY] = True
                        if ('colorZvar' in optionLocal) and (optionLocal['colorZvar'] != '') and ('.' not in optionLocal['colorZvar']):
                            if optionLocal['colorZvar'] in paramDict:
                                parameter = paramDict[optionLocal['colorZvar']]
                                if 'options' in parameter:
                                    for i in parameter['options']:
                                        dfQuery, varNameZ = pandaGetOrMakeColumn(dfQuery, i)
                                        columnNameDict[varNameZ] = True
                                        downsamplerColumns[varNameZ] = True
                            else:
                                dfQuery, varNameZ = pandaGetOrMakeColumn(dfQuery, optionLocal['colorZvar'])
                                columnNameDict[varNameZ] = True
                                downsamplerColumns[varNameZ] = True
                        if 'varZ' in optionLocal:
                            dfQuery, varNameZ = pandaGetOrMakeColumn(dfQuery, optionLocal['varZ'])
                            columnNameDict[varNameZ] = True
                            downsamplerColumns[varNameZ] = True                            
                        # TODO: Make error bars client side to get rid of this mess. At least ND histogram does support them.
                        if ('errY' in optionLocal) and (optionLocal['errY'] != ''):
                            dfQuery, varNameErrY = pandaGetOrMakeColumn(dfQuery, optionLocal['errY'])
                            seriesErrY = dfQuery[varNameErrY]
                            columnNameDict[varNameErrY] = True
                            if varNameY+'_lower' not in dfQuery.columns:
                                seriesLower = dfQuery[varNameY]-seriesErrY
                                dfQuery[varNameY+'_lower'] = seriesLower
                            columnNameDict[varNameY+'_lower'] = True
                            downsamplerColumns[varNameY+'_lower'] = True
                            if varNameY+'_upper' not in dfQuery.columns:
                                seriesUpper = dfQuery[varNameY]+seriesErrY
                                dfQuery[varNameY+'_upper'] = seriesUpper
                            columnNameDict[varNameY+'_upper'] = True
                            downsamplerColumns[varNameY+'_upper'] = True
                        if ('errX' in optionLocal) and (optionLocal['errX'] != ''):
                            dfQuery, varNameErrX = pandaGetOrMakeColumn(dfQuery, optionLocal['errX'])
                            seriesErrX = dfQuery[varNameErrX]
                            columnNameDict[varNameErrX] = True
                            if varNameX+'_lower' not in dfQuery.columns:
                                seriesLower = dfQuery[varNameX]-seriesErrX
                                dfQuery[varNameX+'_lower'] = seriesLower
                            columnNameDict[varNameX+'_lower'] = True
                            downsamplerColumns[varNameX+'_lower'] = True
                            if varNameX+'_upper' not in dfQuery.columns:
                                seriesUpper = dfQuery[varNameX]+seriesErrX
                                dfQuery[varNameX+'_upper'] = seriesUpper
                            columnNameDict[varNameX+'_upper'] = True
                            downsamplerColumns[varNameX+'_upper'] = True
                        if 'tooltips' in optionLocal:
                            tooltipColumns = getTooltipColumns(optionLocal['tooltips'])
                            columnNameDict.update(tooltipColumns)
                            downsamplerColumns.update(tooltipColumns)
                    else:
                        histogramDict[variables[1][j % lengthY]] = True

    if histogramArray is not None:
        for i, histo in enumerate(histogramArray):
            if histogramDict[histo["name"]]:
                for j, variable in enumerate(histo["variables"]):
                    dfQuery, varName = pandaGetOrMakeColumn(dfQuery, variable)
                    columnNameDict[varName] = True
                if "weights" in histo:
                    dfQuery, varName = pandaGetOrMakeColumn(dfQuery, histo["weights"])
                    columnNameDict[varName] = True

    if widgetArray is not None:
        for iWidget in widgetArray:
            if len(iWidget) < 3 or 'callback' not in iWidget[2] or iWidget[2]['callback'] == 'selection':
                dfQuery, varNameX = pandaGetOrMakeColumn(dfQuery, iWidget[1][0])
                columnNameDict[varNameX] = True

    if "removeExtraColumns" in options and options["removeExtraColumns"]:
        dfQuery = dfQuery[columnNameDict]

    return dfQuery, histogramDict, list(downsamplerColumns.keys() & columnNameDict.keys()), columnNameDict, paramDict

def bokehMakeParameters(parameterArray, histogramArray, figureArray, variableList, options={}):
    parameterDict = {}
    if parameterArray is not None:
        for param in parameterArray:
            parameterDict[param["name"]] = param.copy()
            parameterDict[param["name"]]["subscribed_events"] = []
    if histogramArray is not None:
        for iHisto in histogramArray: 
            pass
    if figureArray is not None:
        for i, variables in enumerate(figureArray):
            if len(variables) > 1 and variables[0] != "table" and variables[0] != "tableHisto":
                if len(variables) > 2:
                    optionLocal = options.copy()
                    optionLocal.update(variables[-1])
                else:
                    optionLocal = options      
                if 'colorZvar' in optionLocal:
                    varColor = optionLocal['colorZvar']
                    if varColor in parameterDict:
                        paramColor = parameterDict[varColor]
                        # Possibly also allow custom color mappers?
                        if "type" not in paramColor:
                            paramColor["type"] = "varName"
                        if "options" not in paramColor:
                            paramColor["options"] = variableList
                            # XXX: Add autofill for cases of histograms and main CDS
                if 'size' in optionLocal:
                    varSize = optionLocal['size']
                    if varSize in parameterDict:
                        paramSize = parameterDict[varSize]
                        if "type" not in paramSize:
                            t = type(paramSize['value'])
                            if t == str:
                                paramSize["type"] = "varName"
                            else:
                                paramSize["type"] = "scalar"
                        if paramSize["type"] == "varName":
                            if "options" not in paramColor:
                                paramSize["options"] = variableList
                        elif paramSize["type"] == "scalar":
                            if "range" not in paramSize:
                                raise ValueError("Missing range for parameter: ", paramSize["name"])
    return parameterDict

def defaultNDProfileTooltips(varNames, axis_idx, quantiles, sumRanges):
    tooltips = []
    for i, iAxis in enumerate(varNames):
        if i != axis_idx:
            tooltips.append((iAxis, "[@{bin_bottom_" + str(i) + "}, @{bin_top_" + str(i) + "}]"))
    tooltips.append(("Mean " + varNames[axis_idx], "@mean"))
    tooltips.append(("Std. " + varNames[axis_idx], "@std"))
    for i, iQuantile in enumerate(quantiles):
        tooltips.append((f"Quantile {iQuantile} {varNames[axis_idx]}", "@quantile_" + str(i)))
    for i, iRange in enumerate(sumRanges):
        tooltips.append((f"Sum_normed {varNames[axis_idx]} in [{iRange[0]}, {iRange[1]}]", "@sum_normed_" + str(i)))
    return tooltips

def getOrMakeColumn(dfQuery, column, cdsName):
    if '.' in column:
        c = column.split('.')
        if cdsName is None or cdsName == c[0]:
            return [dfQuery, c[1], c[0]]
        else:
            raise ValueError("Inconsistent CDS")
    else:
        dfQuery, column = pandaGetOrMakeColumn(dfQuery, column)
        return [dfQuery, column, None]

def getTooltipColumns(tooltips):
    if isinstance(tooltips, str):
        return {}
    result = {}
    tooltip_regex = re.compile(r'@(?:\w+|\{[^\}]*\})')
    for iTooltip in tooltips:
        for iField in tooltip_regex.findall(iTooltip[1]):
            if iField[1] == '{':
                result[iField[2:-1]] = True
            else:
                result[iField[1:]] = True
    return result

def getHistogramAxisTitle(histoDict, varName, cdsName, removeCdsName=True):
    if cdsName is None:
        return varName
    if cdsName in histoDict:
        if '_' in varName:
            if varName == "bin_count":
                # Maybe do something else
                return "entries"
            x = varName.split("_")
            if x[0] == "bin":
                if len(x) == 2:
                    return histoDict[cdsName]["variables"][0]
                return histoDict[cdsName]["variables"][int(x[-1])]
            if x[0] == "quantile":
                quantile = histoDict[cdsName]["quantiles"][int(x[-1])]
                if '_' in cdsName:
                    histoName, projectionIdx = cdsName.split("_")
                    return "quantile " + str(quantile) + " " + histoDict[histoName]["variables"][int(projectionIdx)]
                return "quantile " + str(quantile)
            if x[0] == "sum":
                range = histoDict[cdsName]["sum_range"][int(x[-1])]
                if len(x) == 2:
                    if '_' in cdsName:
                        histoName, projectionIdx = cdsName.split("_")
                        return "sum " + histoDict[histoName]["variables"][int(projectionIdx)] + " in [" + str(range[0]) + ", " + str(range[1]) + "]"
                    return "sum in [" + str(range[0]) + ", " + str(range[1]) + "]"
                else:
                    if '_' in cdsName:
                        histoName, projectionIdx = cdsName.split("_")
                        return "p " + histoDict[histoName]["variables"][int(projectionIdx)] + " in [" + str(range[0]) + ", " + str(range[1]) + "]"
                    return "p in ["+ str(range[0]) + ", " + str(range[1]) + "]"
        else:
            if '_' in cdsName:
                histoName, projectionIdx = cdsName.split("_")
                return varName + " " + histoDict[histoName]["variables"][int(projectionIdx)]
    if not removeCdsName:
        return cdsName+"."+varName
    return varName