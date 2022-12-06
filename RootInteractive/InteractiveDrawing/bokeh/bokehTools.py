from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar, HoverTool, VBar, HBar, Quad
from bokeh.models.transforms import CustomJSTransform
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.widgets.tables import ScientificFormatter, DataTable
from bokeh.models.plots import Plot
from bokeh.transform import *
from RootInteractive.InteractiveDrawing.bokeh.ConcatenatedString import ConcatenatedString
from RootInteractive.InteractiveDrawing.bokeh.compileVarName import getOrMakeColumns
from RootInteractive.Tools.aliTreePlayer import *
from bokeh.layouts import *
import logging
from IPython import get_ipython
from bokeh.models.widgets import DataTable, Select, Slider, RangeSlider, MultiSelect, Panel, TableColumn, TextAreaInput, Toggle, Spinner
from bokeh.models import CustomJS, ColumnDataSource
from RootInteractive.InteractiveDrawing.bokeh.bokehVisJS3DGraph import BokehVisJSGraph3D
from RootInteractive.InteractiveDrawing.bokeh.HistogramCDS import HistogramCDS
from RootInteractive.InteractiveDrawing.bokeh.HistoNdCDS import HistoNdCDS
from RootInteractive.Tools.compressArray import compressCDSPipe, removeInt64
from RootInteractive.InteractiveDrawing.bokeh.CDSCompress import CDSCompress
from RootInteractive.InteractiveDrawing.bokeh.HistoStatsCDS import HistoStatsCDS
from RootInteractive.InteractiveDrawing.bokeh.HistoNdProfile import HistoNdProfile
from RootInteractive.InteractiveDrawing.bokeh.DownsamplerCDS import DownsamplerCDS
from RootInteractive.InteractiveDrawing.bokeh.CDSAlias import CDSAlias
from RootInteractive.InteractiveDrawing.bokeh.CustomJSNAryFunction import CustomJSNAryFunction
from RootInteractive.InteractiveDrawing.bokeh.CDSJoin import CDSJoin
from RootInteractive.InteractiveDrawing.bokeh.MultiSelectFilter import MultiSelectFilter
from RootInteractive.InteractiveDrawing.bokeh.LazyTabs import LazyTabs
from RootInteractive.InteractiveDrawing.bokeh.RangeFilter import RangeFilter
import numpy as np
import pandas as pd
import re
import ast
from RootInteractive.InteractiveDrawing.bokeh.compileVarName import ColumnEvaluator
from bokeh.palettes import all_palettes
from RootInteractive.InteractiveDrawing.bokeh.palette import kBird256
import base64

# tuple of Bokeh markers
bokehMarkers = ["square", "circle", "triangle", "diamond", "square_cross", "circle_cross", "diamond_cross", "cross",
                "dash", "hex", "invertedtriangle", "asterisk", "square_x", "x"]

# default tooltips for 1D and 2D histograms
defaultHistoTooltips = [
    ("range", "[@{bin_bottom}, @{bin_top}]"),
    ("count", "@bin_count")
]

defaultHisto2DTooltips = [
    ("range X", "[@{bin_bottom_0}, @{bin_top_0}]"),
    ("range Y", "[@{bin_bottom_1}, @{bin_top_1}]"),
    ("count", "@bin_count")
]

BOKEH_DRAW_ARRAY_VAR_NAMES = ["X", "Y", "varZ", "colorZvar", "marker_field", "legend_field", "errX", "errY"]

ALLOWED_WIDGET_TYPES = ["slider", "range", "select", "multiSelect", "toggle", "multiSelectBitmask", "spinner", "spinnerRange"]

RE_CURLY_BRACE = re.compile(r"\{(.*?)\}")

def makeJScallback(widgetList, cdsOrig, cdsSel, **kwargs):
    options = {
        "verbose": 0,
        "histogramList": []
    }
    options.update(kwargs)

    code = \
        """
    const t0 = performance.now();
    let nSelected=0;
    const precision = 0.000001;
    const size = cdsOrig.length;

    let first = 0;
    let last = size;

    let isSelected = new Array(size);
    for(let i=0; i<size; ++i){
        isSelected[i] = false;
    }
    let indicesAll = [];
    if(index != null){
        const widget = index.widget;
        const widgetType = index.type;
        const col = index.key && cdsOrig.get_column(index.key);
        // TODO: Add more widgets for index, not only range slider
        if(widgetType == "range"){
            const low = widget.value[0];
            const high = widget.value[1];
            for(let i=0; i<size; i++){
                if(col[i] >= low){
                    first = i;
                    break;
                }
            }
            for(let i=first; i<size; i++){
                if(col[i] >= high){
                    last = i;
                    break;
                }
            }
        }
    }
    for(let i=first; i<last; ++i){
        isSelected[i] = true;
    }

    const t1 = performance.now();
    console.log(`Using index took ${t1 - t0} milliseconds.`);
    for (const iWidget of widgetList){
        if(iWidget.widget.disabled) continue;
        if(iWidget.filter != null){
            const widgetFilter = iWidget.filter.v_compute();
            for(let i=first; i<last; i++){
                isSelected[i] &= widgetFilter[i];
            }
            continue;
        }
        const widget = iWidget.widget;
        const widgetType = iWidget.type;
        const col = iWidget.key && cdsOrig.get_column(iWidget.key);
        if(widgetType == "textQuery"){
            const queryText = widget.value;
            if(queryText == ""){
                continue;
            }
            const pattern = /^[a-zA-Z_$][0-9a-zA-Z_$]*$/;
            const variablesLocal = cdsOrig.columns().filter((x) => pattern.test(x))
            const dataOrigArray = variablesLocal.map((x) => cdsOrig.get_column(x))
            const f = new Function(...variablesLocal, "\\"use strict\\"\\n" + queryText);
            for(let i=first; i<last; i++){
                const result = f(...dataOrigArray.map(x => x[i]));
                isSelected[i] &= result;
            }
        }
    }
   
    const t2 = performance.now();
    console.log(`Filtering took ${t2 - t1} milliseconds.`);
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
            histo.change_selection();
        }
    }
    const t3 = performance.now();
    console.log(`Histogramming took ${t3 - t2} milliseconds.`);
    if(cdsSel != null){
        console.log(isSelected.reduce((a,b)=>a+b, 0));
        cdsSel.booleans = isSelected
        cdsSel.invalidate()
        console.log(cdsSel._downsampled_indices.length);
        console.log(cdsSel.nPoints)
        const t4 = performance.now();
        console.log(`Updating cds took ${t4 - t3} milliseconds.`);
    }
    if(options.cdsHistoSummary !== null){
        options.cdsHistoSummary.update();
    }
    console.log(\"nSelected:%d\",nSelected);
    """
    if options["verbose"] > 0:
        logging.info("makeJScallback:\n", code)
    callback = CustomJS(args={'widgetList': widgetList, 'cdsOrig': cdsOrig, 'cdsSel': cdsSel, 'options': options, 'index':options["index"]},
                        code=code)
    return callback


def processBokehLayoutArray(widgetLayoutDesc, widgetArray: list, widgetDict: dict={}, isHorizontal: bool=False, options: dict=None):
    """
    apply layout on plain array of bokeh figures, resp. interactive widgets
    :param widgetLayoutDesc: array or dict desciption of layout
    :param widgetArray: input plain array of widgets/figures
    :param widgetDict: input dict of widgets/figures, used for accessing them by name
    :param isHorizontal: whether to create a row or column
    :param options: options to use - can also be specified as the last element of widgetArray
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
    return processBokehLayoutArrayRenderers(widgetLayoutDesc, widgetArray, widgetDict, isHorizontal, options)[0]

def processBokehLayoutArrayRenderers(widgetLayoutDesc, widgetArray: list, widgetDict: dict={}, isHorizontal: bool=False, options: dict=None):
    if isinstance(widgetLayoutDesc, dict):
        tabsModel = LazyTabs()
        tabs = []
        renderers = []
        for i, iPanel in widgetLayoutDesc.items():
            child, childRenderers = processBokehLayoutArrayRenderers(iPanel, widgetArray, widgetDict)
            tabs.append(Panel(child=child, title=i))
            renderers.append(childRenderers)
        tabsModel.tabs = tabs
        tabsModel.renderers = renderers
        return tabsModel, [tabsModel]
    if options is None:
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
        optionLocal = options.copy()
        optionLocal.update(widgetLayoutDesc[-1])
        widgetLayoutDesc = widgetLayoutDesc[0:-1]
    else:
        optionLocal = options

    renderers = []

    for i, iWidget in enumerate(widgetLayoutDesc):
        if isinstance(iWidget, dict):
            child, childRenderers = processBokehLayoutArrayRenderers(iWidget, widgetArray, widgetDict, isHorizontal=False, options=optionLocal)
            widgetRows.append(child)
            renderers += childRenderers
            continue
        if isinstance(iWidget, list):
            child, childRenderers = processBokehLayoutArrayRenderers(iWidget, widgetArray, widgetDict, isHorizontal=not isHorizontal, options=optionLocal)
            widgetRows.append(child)
            renderers += childRenderers
            continue

        if isinstance(iWidget, int) and iWidget < len(widgetArray):
            figure = widgetArray[iWidget]
        else:
            figure = widgetDict[iWidget]
        widgetRows.append(figure)
        if hasattr(figure, 'x_range'):
            if optionLocal['commonX'] >= 0:
                figure.x_range = widgetArray[int(optionLocal["commonX"])].x_range
            if optionLocal['commonY'] >= 0:
                figure.y_range = widgetArray[int(optionLocal["commonY"])].y_range
            if optionLocal['x_visible'] == 0:
                figure.xaxis.visible = False
            else:
                figure.xaxis.visible = True
            if optionLocal['y_visible'] == 0:
                figure.yaxis.visible = False
            if optionLocal['y_visible'] == 2:
                if i > 0: figure.yaxis.visible = False
        if hasattr(figure, 'plot_width'):
            if optionLocal["plot_width"] > 0:
                plot_width = int(optionLocal["plot_width"] / nRows)
                figure.plot_width = plot_width
            if optionLocal["plot_height"] > 0:
                figure.plot_height = optionLocal["plot_height"]
            if figure.legend:
                figure.legend.visible = optionLocal["legend_visible"]
        if type(figure).__name__ == "DataTable":
            figure.height = int(optionLocal["plot_height"])
        if type(figure).__name__ == "BokehVisJSGraph3D":
            if optionLocal["plot_width"] > 0:
                plot_width = int(optionLocal["plot_width"] / nRows)
                figure.width = plot_width
            if optionLocal["plot_height"] > 0:
                figure.height = optionLocal["plot_height"]
        if isinstance(figure, Plot):
            renderers += [i.data_source for i in figure.renderers if isinstance(i.data_source, DownsamplerCDS)]

    if isHorizontal:
        return row(widgetRows, sizing_mode=options['sizing_mode']), renderers
    return column(widgetRows, sizing_mode=options['sizing_mode']), renderers  


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
            title = dataFrame.meta.metaData.get(col + ".OrigName", col)
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


def makeBokehHistoTable(histoDict: dict, include: str, exclude: str, rowwise=False, **kwargs):
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
        # We are only interested in histograms so we filter the dict for histograms
        isOK = True
        if include:
            isOK = False
            if re.match(include, histoDict[iHisto]["name"]):
                isOK = True
        if exclude:
            if re.match(exclude, histoDict[iHisto]["name"]):
                isOK = False
        if isOK:
            if histoDict[iHisto]["type"] == "histogram":
                histo_names.append(histoDict[iHisto]["name"])
                histo_columns.append("bin_count")
                bin_centers.append("bin_center")
                edges_left.append("bin_bottom")
                edges_right.append("bin_top")
                sources.append(histoDict[iHisto]["cdsOrig"])
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


def bokehDrawArray(dataFrame, query, figureArray, histogramArray=[], parameterArray=[], jsFunctionArray=[], aliasArray=[], sourceArray=None, **kwargs):
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
        'commonX': 0,
        'commonY': 0,
        'ncols': -1,
        'layout': '',
        'widgetLayout': '',
        'palette': kBird256,
        "markers": bokehMarkers,
        "colors": 'Category10',
        "rescaleColorMapper": False,
        "filter": '',
        'doDraw': False,
        "legend_field": None,
        "legendTitle": None,
        'nPointRender': 10000,
        "nbins": 10,
        "range": None,
        "flip_histogram_axes": False,
        "show_histogram_error": False,
        "arrayCompression": None,
        "removeExtraColumns": False
    }
    options.update(kwargs)
    if query is not None:
        dfQuery = dataFrame.query(query).copy()
        if hasattr(dataFrame, 'metaData'):
            dfQuery.metaData = dataFrame.metaData
            logging.info(dfQuery.metaData)
    else:
        dfQuery = dataFrame.copy()
        initMetadata(dfQuery)
    # Check/resp. load derived variables

    if isinstance(figureArray[-1], dict):
        options.update(figureArray[-1])

    if sourceArray is not None:
        sourceArray = histogramArray + sourceArray
    
    else:
        sourceArray = histogramArray

    sourceArray = [{"data": dfQuery, "arrayCompression": options["arrayCompression"], "name":None, "tooltips": options["tooltips"]}] + sourceArray

    paramDict = {}
    for param in parameterArray:
        paramDict[param["name"]] = param.copy()
        paramDict[param["name"]]["subscribed_events"] = []

    widgetParams = []
    widgetArray = []
    widgetDict = {}

    cdsDict = makeCDSDict(sourceArray, paramDict, options={"nPointRender":options["nPointRender"]})

    jsFunctionDict = {}
    for i in jsFunctionArray:
        customJsArgList = {}
        if isinstance(i["parameters"], list):
            for j in i["parameters"]:
                customJsArgList[j] = paramDict[j]["value"]
        if "v_func" in i:
            jsFunctionDict[i["name"]] = CustomJSNAryFunction(parameters=customJsArgList, fields=i["fields"], v_func=i["v_func"])
        else:
            jsFunctionDict[i["name"]] = CustomJSNAryFunction(parameters=customJsArgList, fields=i["fields"], func=i["func"])
        if isinstance(i["parameters"], list):
            for j in i["parameters"]:
                paramDict[j]["subscribed_events"].append(["value", CustomJS(args={"mapper":jsFunctionDict[i["name"]], "param":j}, code="""
        mapper.parameters[param] = this.value
        mapper.update_args()
                """)])

    aliasDict = {None:{}}
    aliasSet = set()
    for i in aliasArray:
        customJsArgList = {}
        transform = None
        if not isinstance(i, dict):
            if len(i) == 2:
                i = {"name":i[0], "expr":i[1]}
            if len(i) == 3:
                i = {"name":i[0], "expr":i[1], "context": i[2]}
        aliasSet.add(i["name"])
        if "transform" in i:
            if i["transform"] in jsFunctionDict:
                if "context" in i:
                    if i["context"] not in aliasDict:
                        aliasDict[i["context"]] = {}
                    aliasDict[i["context"]][i["name"]] = {"fields": i["variables"], "transform": jsFunctionDict[i["transform"]]}
                else:
                    aliasDict[None][i["name"]] = {"fields": i["variables"], "transform": jsFunctionDict[i["transform"]]}
        else:
            if "parameters" in i:
                for j in i["parameters"]:
                    customJsArgList[j] = paramDict[j]["value"]
            if "v_func" in i:
                transform = CustomJSNAryFunction(parameters=customJsArgList, fields=i["variables"], v_func=i["v_func"])
            elif "func" in i:
                if i["func"] in paramDict:
                    transform = CustomJSNAryFunction(parameters=customJsArgList, fields=i["variables"], func=paramDict[i["func"]]["value"])
                    paramDict[i["func"]]["subscribed_events"].append(["value", CustomJS(args={"mapper":transform}, code="""
            mapper.func = this.value
            mapper.update_func()
                    """)])
                else:
                    transform = CustomJSNAryFunction(parameters=customJsArgList, fields=i["variables"], func=i["func"])
            if "expr" in i:
                exprTree = ast.parse(i["expr"], filename="<unknown>", mode="eval")
                context = i.get("context", None)
                evaluator = ColumnEvaluator(context, cdsDict, paramDict, jsFunctionDict, i["expr"], aliasDict)
                result = evaluator.visit(exprTree.body)
                if result["type"] == "javascript":
                    func = "return "+result["implementation"]
                    fields = list(evaluator.aliasDependencies)
                    parameters = list(evaluator.paramDependencies)
                    parameters = [i for i in evaluator.paramDependencies if "options" not in paramDict[i]]
                    variablesParam = [i for i in evaluator.paramDependencies if "options" in paramDict[i]]
                    customJsArgList = {i:paramDict[i]["value"] for i in evaluator.paramDependencies}
                    nvars_local = len(fields)
                    variablesAlias = fields.copy()
                    for j in variablesParam:
                        if "subscribed_events" not in paramDict[j]:
                            paramDict[j]["subscribed_events"] = []
                        paramDict[j]["subscribed_events"].append(["value", CustomJS(args={"idx":nvars_local, "column_name":i["name"], "table":cdsDict[context]["cdsFull"]}, code="""
                                            table.mapping[column_name].fields[idx] = this.value
                                            table.invalidate_column(column_name)
                                                    """)])
                        variablesAlias.append(paramDict[j]["value"])
                        fields.append(j)
                        nvars_local = nvars_local+1
                    transform = CustomJSNAryFunction(parameters=customJsArgList, fields=fields.copy(), func=func)
                    fields = variablesAlias
                else:
                    aliasDict[i["name"]] = result["name"]
                    break
            else:
                parameters = i.get("parameters", None)
                fields = i["variables"]
            if "context" in i:
                if i["context"] not in aliasDict:
                    aliasDict[i["context"]] = {}
                aliasDict[i["context"]][i["name"]] = {"fields": fields, "transform": transform}
            else:
                aliasDict[None][i["name"]] = {"fields": fields, "transform": transform}
            if parameters is not None:
                for j in parameters:
                    paramDict[j]["subscribed_events"].append(["value", CustomJS(args={"mapper":transform, "param":j}, code="""
            mapper.parameters[param] = this.value
            mapper.update_args()
                    """)])


    plotArray = []
    colorAll = all_palettes[options['colors']] if isinstance(options['colors'], str) else options['colors']
    colorMapperDict = {}
    cdsHistoSummary = None

    memoized_columns = {}
    sources = set()

    meta = dfQuery.meta.metaData.copy()

    profileList = []

    optionsChangeList = []
    for i, variables in enumerate(figureArray):
        if isinstance(variables, dict):
            optionsChangeList.append(i)

    plotDict = {}

    optionsIndex = 0
    if len(optionsChangeList) != 0:
        optionGroup = options.copy()
        optionGroup.update(figureArray[optionsChangeList[0]])
    else:
        optionGroup = options
    for i, variables in enumerate(figureArray):
        logging.info("%d\t%s", i, variables)
        if isinstance(variables, dict):
            optionsIndex = optionsIndex + 1
            if optionsIndex < len(optionsChangeList):
                optionGroup = options.copy()
                optionGroup.update(figureArray[optionsChangeList[optionsIndex]])
            else:
                optionGroup = options
            continue
        if variables[0] == 'table':
            TOptions = {
                'include': '',
                'exclude': ''
            }
            if len(variables) > 1:
                TOptions.update(variables[1])
            # TODO: This is broken if compression is used because CDSCompress isn't a ColumnDataSource
            dataTable = makeBokehDataTable(dfQuery, cdsDict[None]["cdsOrig"], TOptions['include'], TOptions['exclude'])
            plotArray.append(dataTable)
            if "name" in optionLocal:
                plotDict[optionLocal["name"]] = dataTable
            continue
        if variables[0] == 'tableHisto':
            histoListLocal = []
            TOptions = {
                'include': '',
                'exclude': '',
                'rowwise': False
            }
            for key, value in cdsDict.items():
                if value["type"] in ["histogram", "histo2d", "histoNd"]:
                    histoListLocal.append(key)
            # We just want to add them to the dependency tree
            _, _, memoized_columns, used_names_local = getOrMakeColumns("bin_count", histoListLocal, cdsDict, paramDict, jsFunctionDict, memoized_columns)
            sources.update(used_names_local)
            if len(variables) > 1:
                TOptions.update(variables[1])
            histoDict = {i: cdsDict[i] for i in histoListLocal}
            cdsHistoSummary, tableHisto = makeBokehHistoTable(histoDict, include=TOptions["include"], exclude=TOptions["exclude"], rowwise=TOptions["rowwise"])
            plotArray.append(tableHisto)
            if "name" in optionLocal:
                plotDict[optionLocal["name"]] = tableHisto
            continue
        if variables[0] in ALLOWED_WIDGET_TYPES:
            optionWidget = {}
            if len(variables) == 3:
                optionWidget = variables[2].copy()
            fakeDf = None
            widgetFilter = None
            if "callback" not in optionLocal:
                if variables[1][0] in paramDict:
                    optionWidget["callback"] = "parameter"
                    varName = variables[1]
                else:
                    optionWidget["callback"] = "selection"
                    column, cds_names, memoized_columns, used_names_local = getOrMakeColumns(variables[1][0], None, cdsDict, paramDict, jsFunctionDict, memoized_columns, aliasDict)
                    varName = column[0]["name"]
                    if column[0]["type"] == "column":
                        fakeDf = {varName: dfQuery[varName]}
                    elif column[0]["type"] == "server_derived_column":
                        fakeDf = {varName: column[0]["value"]}
                    if variables[0] != 'multiSelect':
                        sources.update(used_names_local)
            if variables[0] == 'slider':
                localWidget = makeBokehSliderWidget(fakeDf, False, variables[1], paramDict, **optionWidget)
                if optionWidget["callback"] == "selection":
                    widgetFilter = RangeFilter(range=(localWidget.value-.5*localWidget.step, localWidget.value+.5*localWidget.step), field=variables[1][0], name=variables[1][0])
                    localWidget.js_on_change("value", CustomJS(args={"target":widgetFilter}, code="""
                        target.range = [this.value-.5*this.step, this.value+.5*this.step]
                    """
                    ))
                widgetFull = localWidget
            if variables[0] == 'range':
                localWidget = makeBokehSliderWidget(fakeDf, True, variables[1], paramDict, **optionWidget)
                if optionWidget["callback"] == "selection" and "index" not in optionWidget:
                    widgetFilter = RangeFilter(range=localWidget.value, field=variables[1][0], name=variables[1][0])
                    localWidget.js_link("value", widgetFilter, "range")
                widgetFull = localWidget
            if variables[0] == 'select':
                localWidget, widgetFilter, newColumn = makeBokehSelectWidget(fakeDf, variables[1], paramDict, **optionWidget)
                if newColumn is not None:
                    memoized_columns[None][newColumn["name"]] = newColumn
                    sources.add((None, newColumn["name"]))
                widgetFull = localWidget
            if variables[0] == 'multiSelect':
                localWidget, widgetFilter, newColumn = makeBokehMultiSelectWidget(fakeDf, variables[1], paramDict, **optionWidget)
                if newColumn is not None:
                    memoized_columns[None][newColumn["name"]] = newColumn
                    sources.add((None, newColumn["name"]))
                widgetFull = localWidget
            if variables[0] == 'multiSelectBitmask':
                localWidget, widgetFilter = makeBokehMultiSelectBitmaskWidget(column[0], **optionWidget)
                widgetFull = localWidget
            if variables[0] == 'toggle':
                label = variables[1][0]
                if "label" in optionWidget:
                    label = optionWidget["label"]
                active = False
                if optionWidget["callback"] == "parameter":
                    active = paramDict[variables[1][0]]["value"]
                localWidget = Toggle(label=label, active=active)
                widgetFull = localWidget
            if variables[0] == 'spinner':
                label = variables[1][0]
                if "label" in optionWidget:
                    label = optionWidget["label"]
                value = 1
                if optionWidget["callback"] == "parameter":
                    value = paramDict[variables[1][0]]["value"]
                formatter = optionWidget.get("format", "0.[0000]")
                localWidget = Spinner(title=label, value=value, format=formatter)
                widgetFull = localWidget
            if variables[0] == 'spinnerRange':
                # TODO: Make a spinner pair custom widget, or something similar
                label = variables[1][0]
                start, end, step = makeSliderParameters(fakeDf, variables[1], **optionWidget)
                formatter = optionWidget.get("format", "0.[0000]")
                relativeStep = optionWidget.get("relativeStep", .05)
                localWidgetMin = Spinner(title=f"min({label})", value=start, step=step, format=formatter)
                localWidgetMax = Spinner(title=f"max({label})", value=end, step=step, format=formatter)
                if optionWidget["callback"] == "parameter":
                    pass
                else:
                    widgetFilter = RangeFilter(range=[start, end], field=variables[1][0], name=variables[1][0])
                    localWidgetMin.js_on_change("value", CustomJS(args={"other":localWidgetMax, "filter": widgetFilter, "relative_step":relativeStep}, code="""
                        other.step = this.step = (other.value - this.value) * relative_step
                        filter.range[0] = this.value
                        filter.properties.range.change.emit()
                        filter.change.emit()
                        """))
                    localWidgetMax.js_on_change("value", CustomJS(args={"other":localWidgetMin, "filter": widgetFilter, "relative_step":relativeStep}, code="""
                        other.step = this.step = (this.value - other.value) * relative_step
                        filter.range[1] = this.value
                        filter.properties.range.change.emit()
                        filter.change.emit()
                        """))
                widgetFull=localWidget=row([localWidgetMin, localWidgetMax])
            if "toggleable" in optionWidget:
                widgetToggle = Toggle(label="disable", active=True, width=70)
                if variables[0] == 'spinnerRange':
                    widgetToggle.js_on_change("active", CustomJS(args={"widgetMin":localWidgetMin, "widgetMax": localWidgetMax, "filter": widgetFilter}, code="""
                    widgetMin.disabled = this.active
                    widgetMax.disabled = this.active
                    filter.change.emit()
                    """))
                else:
                    localWidget.disabled=True
                    widgetToggle.js_on_change("active", CustomJS(args={"widget":localWidget}, code="""
                    widget.disabled = this.active
                    widget.properties.value.change.emit()
                    """))
                if widgetFilter is not None:
                    widgetToggle.js_on_change("active", CustomJS(args={"filter": widgetFilter}, code="""
                    filter.change.emit()
                    """))
                widgetFull = row([widgetFull, widgetToggle])
            plotArray.append(widgetFull)
            if "name" in optionWidget:
                plotDict[optionWidget["name"]] = widgetFull
            if optionWidget["callback"] != "selection" and localWidget:
                widgetArray.append(localWidget)
                widgetParams.append(variables)
            if optionWidget["callback"] == "selection":
                cds_used = cds_names[0]
                if cds_used not in widgetDict:
                    widgetDict[cds_used] = {"widgetList":[]}
                widgetDictLocal = {"widget": localWidget, "type": variables[0], "key": varName}
                if widgetFilter is not None:
                    widgetDictLocal["filter"] = widgetFilter
                if "index" in optionWidget and optionWidget["index"]:
                    widgetDict[cds_used]["index"] = widgetDictLocal
                else:
                    widgetDict[cds_used]["widgetList"].append(widgetDictLocal)
            continue
        if variables[0] == "textQuery":
            optionWidget = {}
            if len(variables) >= 2:
                optionWidget.update(variables[-1])
            cds_used = None
            # By default, uses all named variables from the data source - but they can't be known at this point yet
            localWidget = TextAreaInput(**optionWidget)
            plotArray.append(localWidget)
            if "name" in optionWidget:
                plotDict[optionWidget["name"]] = localWidget
            if cds_used not in widgetDict:
                widgetDict[cds_used] = {"widgetList":[]}
            widgetDict[cds_used]["widgetList"].append({"widget": localWidget, "type": variables[0], "key": None})
            continue
        if variables[0] == "text":
            optionWidget = {"title": variables[1][0]}
            if len(variables) > 2:
                optionWidget.update(variables[-1])
            value = paramDict[variables[1][0]]["value"]
            localWidget = TextAreaInput(value=value, **optionWidget)
            plotArray.append(localWidget)
            if "name" in optionWidget:
                plotDict[optionWidget["name"]] = localWidget
            widgetArray.append(localWidget)
            widgetParams.append(variables)
            continue

        optionLocal = optionGroup.copy()
        nvars = len(variables)
        if isinstance(variables[-1], dict):
            logging.info("Option %s", variables[-1])
            optionLocal.update(variables[-1])
            nvars -= 1

        x_transform = optionLocal.get("x_transform", None)
        x_transform_parsed, x_transform_customjs = make_transform(x_transform, paramDict, aliasDict, cdsDict, jsFunctionDict)
        y_transform = optionLocal.get("y_transform", None)
        y_transform_parsed, y_transform_customjs = make_transform(y_transform, paramDict, aliasDict, cdsDict, jsFunctionDict, orientation=1)

        variablesLocal = [None]*len(BOKEH_DRAW_ARRAY_VAR_NAMES)
        for axis_index, axis_name  in enumerate(BOKEH_DRAW_ARRAY_VAR_NAMES):
            if axis_index < nvars:
                variablesLocal[axis_index] = variables[axis_index].copy()
            elif axis_name in optionLocal:
                variablesLocal[axis_index] = optionLocal[axis_name]
            if variablesLocal[axis_index] is not None and not isinstance(variablesLocal[axis_index], list):
                variablesLocal[axis_index] = [variablesLocal[axis_index]]
        length = max(j is not None and len(j) for j in variablesLocal)
        cds_names = [None]*length
        if "source" in optionLocal:
            cds_names = optionLocal["source"]
        for i, iY in enumerate(variablesLocal[1]):
            if iY in cdsDict and cdsDict[iY]["type"] in ["histogram", "histo2d"]:
                cds_names[i] = "$IGNORE"
                _, _, memoized_columns, used_names_local = getOrMakeColumns("bin_count", variablesLocal[1][i], cdsDict, paramDict, jsFunctionDict, memoized_columns, aliasDict)
                sources.update(used_names_local)

        for axis_index, axis_name  in enumerate(BOKEH_DRAW_ARRAY_VAR_NAMES):
            variablesLocal[axis_index], cds_names, memoized_columns, used_names_local = getOrMakeColumns(variablesLocal[axis_index], cds_names, cdsDict, paramDict, jsFunctionDict, memoized_columns, aliasDict)
            sources.update(used_names_local)

        # varZ - if 3D use 3D
        if variablesLocal[2] is not None:
            cds_name = cds_names[0]
            varNameX = variablesLocal[0][0]["name"]
            varNameY = variablesLocal[1][0]["name"]
            varNameZ = variablesLocal[2][0]["name"]
            if variablesLocal[3] is not None:
                varNameColor = variablesLocal[3][0]["name"]
            else:
                varNameColor = None
            options3D = {"width": "99%", "height": "99%"}
            cds_used = cdsDict[cds_name]["cds"]
            plotI = BokehVisJSGraph3D(width=options['plot_width'], height=options['plot_height'],
                                      data_source=cds_used, x=varNameX, y=varNameY, z=varNameZ, style=varNameColor,
                                      options3D=options3D)
            plotArray.append(plotI)
            if "name" in optionLocal:
                plotDict[optionLocal["name"]] = plotI
            continue
        else:
            figureI = figure(plot_width=options['plot_width'], plot_height=options['plot_height'], 
                             tools=options['tools'], x_axis_type=options['x_axis_type'],
                             y_axis_type=options['y_axis_type'])

        lengthX = len(variables[0])
        lengthY = len(variables[1])
        length = max(len(variables[0]), len(variables[1]))
        color_bar = None

        hover_tool_renderers = {}

        figure_cds_name = None
        mapperC = None

        xAxisTitleBuilder = []
        yAxisTitleBuilder = []
        color_axis_title = None

        for i in range(length):
            variables_dict = {}
            for axis_index, axis_name  in enumerate(BOKEH_DRAW_ARRAY_VAR_NAMES):
                variables_dict[axis_name] = variablesLocal[axis_index]
                if isinstance(variables_dict[axis_name], list):
                    variables_dict[axis_name] = variables_dict[axis_name][i % len(variables_dict[axis_name])]
            cds_name = cds_names[i]
            varColor = variables_dict["colorZvar"]
            if varColor is not None:
                if mapperC is not None:
                    color = mapperC
                else:
                    palette = optionLocal['palette']
                    if isinstance(palette, str):
                        palette = all_palettes[palette]
                    rescaleColorMapper = optionLocal["rescaleColorMapper"] or varColor["type"] == "parameter" or cdsDict[cds_name]["type"] in ["histogram", "histo2d", "histoNd"]
                    if not rescaleColorMapper and cdsDict[cds_name]["type"] == "source":
                        low = np.nanmin(cdsDict[cds_name]["data"][varColor["name"]])
                        high= np.nanmax(cdsDict[cds_name]["data"][varColor["name"]])
                        mapperC = linear_cmap(field_name=varColor["name"], palette=palette, low=low, high=high)
                    else:
                        if varColor["name"] in colorMapperDict:
                            mapperC = colorMapperDict[varColor["name"]]
                        else:
                            mapperC = {"field": varColor["name"], "transform": LinearColorMapper(palette=palette)}
                            colorMapperDict[varColor["name"]] = mapperC
                    # HACK for projections - should probably just remove the rows as there's no issue with joins at all
                    if cdsDict[cds_name]["type"] == "projection" and not rescaleColorMapper and varColor["name"].split('_')[0] == 'bin':
                        cdsDict[cds_name]['cds'].js_on_change('change', CustomJS(code="""
                        const col = this.get_column(field)
                        const isOK = this.get_column("isOK")
                        const low = col.map((x,i) => isOK[i] ? col[i] : Infinity).reduce((acc, cur)=>Math.min(acc,cur), Infinity);
                        const high = col.map((x,i) => isOK[i] ? col[i] : -Infinity).reduce((acc, cur)=>Math.max(acc,cur), -Infinity);
                        cmap.high = high;
                        cmap.low = low;
                        """, args={"field": mapperC["field"], "cmap": mapperC["transform"]})) 
                    color = mapperC
                    # Also add the color bar
                    if "colorAxisTitle" in optionLocal:
                        color_axis_title = optionLocal["colorAxisTitle"]
                    else:
                        color_axis_title = getHistogramAxisTitle(cdsDict, varColor["name"], cds_name)
                    color_bar = ColorBar(color_mapper=mapperC['transform'], width=8, location=(0, 0))
                    color_axis_title_model = makeAxisLabelFromTemplate(color_axis_title, paramDict, meta)
                    applyParametricAxisLabel(color_axis_title_model, color_bar, "title")
            elif 'color' in optionLocal:
                color=optionLocal['color']
            else:
                color = colorAll[max(length, 4)][i]
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
            varY = variables[1][i % lengthY]
            cds_used = None
            if cds_name != "$IGNORE":
                cds_used = cdsDict[cds_name]["cds"]

            if isinstance(varY, str) and varY in cdsDict and cdsDict[varY]["type"] in ["histogram", "histo2d"]:
                histoHandle = cdsDict[varY]
                if histoHandle["type"] == "histogram":
                    colorHisto = colorAll[max(length, 4)][i]
                    x_label = f"{{{histoHandle['variables'][0]}}}"
                    y_label = "entries"
                    addHistogramGlyph(figureI, histoHandle, marker, colorHisto, markerSize, optionLocal)
                elif histoHandle["type"] == "histo2d":
                    x_label = f"{{{histoHandle['variables'][0]}}}"
                    y_label = f"{{{histoHandle['variables'][1]}}}"
                    addHisto2dGlyph(figureI, histoHandle, marker, optionLocal)
            else:
                drawnGlyph = None
                colorMapperCallback = """
                glyph.fill_color={...glyph.fill_color, field:this.value}
                glyph.line_color={...glyph.line_color, field:this.value}
                """
                visualization_type = optionLocal.get("visualization_type")
                if visualization_type is None:
                    if isinstance(variables_dict["X"], tuple):
                        visualization_type = "heatmap"
                    else:
                        visualization_type = "scatter"
                if visualization_type == "heatmap":
                    left = variables_dict["X"][0]["name"]
                    right = variables_dict["X"][1]["name"]
                    bottom = variables_dict["Y"][0]["name"]
                    top = variables_dict["Y"][1]["name"]
                    left_transform_modified = modify_2d_transform(x_transform_parsed, x_transform_customjs, variables_dict["Y"][0]["name"], cds_used) if x_transform else None
                    right_transform_modified = modify_2d_transform(x_transform_parsed, x_transform_customjs, variables_dict["Y"][1]["name"], cds_used) if x_transform else None
                    bottom_transform_modified = modify_2d_transform(y_transform_parsed, y_transform_customjs, variables_dict["X"][0]["name"], cds_used) if y_transform else None
                    top_transform_modified = modify_2d_transform(y_transform_parsed, y_transform_customjs, variables_dict["X"][1]["name"], cds_used) if y_transform else None
                    dataSpecBottom = {"field":bottom, "transform":bottom_transform_modified} if y_transform else bottom
                    dataSpecTop = {"field":top, "transform":top_transform_modified} if y_transform else top
                    dataSpecLeft = {"field":left, "transform":left_transform_modified} if x_transform else left
                    dataSpecRight = {"field":right, "transform":right_transform_modified} if x_transform else right
                    x_label = getHistogramAxisTitle(cdsDict, left, cds_name)
                    if x_transform:
                        x_label = f"{{{x_transform}}} {x_label}"
                    y_label = getHistogramAxisTitle(cdsDict, bottom, cds_name)
                    if y_transform:
                        y_label = f"{{{y_transform}}} {y_label}"
                    drawnGlyph = figureI.quad(top=dataSpecTop, bottom=dataSpecBottom, left=dataSpecLeft, right=dataSpecRight,
                    fill_alpha=1, source=cds_used, color=color, legend_label=y_label + " vs " + x_label)
                elif visualization_type == "scatter":
                    varNameX = variables_dict["X"]["name"]
                    varNameY = variables_dict["Y"]["name"]
                    x_transform_modified = modify_2d_transform(x_transform_parsed, x_transform_customjs, variables_dict["Y"]["name"], cds_used) if x_transform else None
                    y_transform_modified = modify_2d_transform(y_transform_parsed, y_transform_customjs, variables_dict["X"]["name"], cds_used) if y_transform else None
                    dataSpecX = {"field":varNameX, "transform":x_transform_modified} if x_transform else varNameX
                    dataSpecY = {"field":varNameY, "transform":y_transform_modified} if y_transform else varNameY
                    if optionLocal["legend_field"] is None:
                        x_label = getHistogramAxisTitle(cdsDict, varNameX, cds_name)
                        if x_transform:
                            x_label = f"{{{x_transform}}} {x_label}"
                        y_label = getHistogramAxisTitle(cdsDict, varNameY, cds_name)
                        if y_transform:
                            y_label = f"{{{y_transform}}} {y_label}"
                        legend_label = makeAxisLabelFromTemplate(f"{y_label} vs {x_label}", paramDict, meta)
                        if isinstance(legend_label, str):
                            drawnGlyph = figureI.scatter(x=dataSpecX, y=dataSpecY, fill_alpha=1, source=cds_used, size=markerSize,
                                    color=color, marker=marker, legend_label=legend_label)
                        else:
                            drawnGlyph = figureI.scatter(x=dataSpecX, y=dataSpecY, fill_alpha=1, source=cds_used, size=markerSize,
                                color=color, marker=marker, legend_label=''.join(legend_label.components))
                    else:
                        drawnGlyph = figureI.scatter(x=dataSpecX, y=dataSpecY, fill_alpha=1, source=cds_used, size=markerSize,
                                    color=color, marker=marker, legend_field=optionLocal["legend_field"])
                    if optionLocal['size'] in paramDict:
                        paramDict[optionLocal['size']]["subscribed_events"].append(["value", drawnGlyph.glyph, "size"])
                else:
                    raise NotImplementedError(f"Visualization type not suppoerted: {visualization_type}")
                if varColor is not None and varColor["name"] in paramDict:
                    if len(color["transform"].domain) == 0:
                        color["transform"].domain = [(drawnGlyph, color["field"])]
                        # HACK: This changes the color mapper's domain, which only consists of one field. 
                        paramDict[varColor["name"]]["subscribed_events"].append(["value", CustomJS(args={"transform": color["transform"]}, code="""
                            transform.domain[0] = [transform.domain[0][0], this.value]
                            transform.change.emit()
                        """)])
                    paramDict[varColor["name"]]["subscribed_events"].append(["value", CustomJS(args={"glyph": drawnGlyph.glyph}, code=colorMapperCallback)])
                if cds_name not in hover_tool_renderers:
                    hover_tool_renderers[cds_name] = []
                hover_tool_renderers[cds_name].append(drawnGlyph)
                if variables_dict['errX'] is not None:
                    if isinstance(variables_dict['errX'], dict):
                        if x_transform:
                            barLower, barUpper = errorBarWidthAsymmetric((variables_dict['errX'],variables_dict['errX']), variables_dict['X'], cds_used, x_transform_parsed, paramDict)
                            errorX = Quad(top=dataSpecY, bottom=dataSpecY, left=barLower, right=barUpper, line_color=color)                          
                        else:
                            errWidthX = errorBarWidthTwoSided(variables_dict['errX'], paramDict)
                            errorX = VBar(top=dataSpecY, bottom=dataSpecY, width=errWidthX, x=varNameX, line_color=color)
                    elif isinstance(variables_dict['errX'], tuple):
                        barLower, barUpper = errorBarWidthAsymmetric(variables_dict['errX'], variables_dict['X'], cds_used)
                        errorX = Quad(top=dataSpecY, bottom=dataSpecY, left=barLower, right=barUpper, line_color=color)                        
                    figureI.add_glyph(cds_used, errorX)
                if variables_dict['errY'] is not None:
                    if isinstance(variables_dict['errY'], dict):
                        if y_transform:
                            barLower, barUpper = errorBarWidthAsymmetric((variables_dict['errY'],variables_dict['errY']), variables_dict['Y'], cds_used, y_transform_parsed, paramDict)
                            errorY = Quad(top=barUpper, bottom=barLower, left=dataSpecX, right=dataSpecX, line_color=color)
                        else:
                            errWidthY = errorBarWidthTwoSided(variables_dict['errY'], paramDict)
                            errorY = HBar(left=dataSpecX, right=dataSpecX, height=errWidthY, y=varNameY, line_color=color)
                    elif isinstance(variables_dict['errY'], tuple):
                        barLower, barUpper = errorBarWidthAsymmetric(variables_dict['errY'], variables_dict['Y'], cds_used, y_transform_parsed, paramDict)
                        errorY = Quad(top=barUpper, bottom=barLower, left=dataSpecX, right=dataSpecX, line_color=color)
                    figureI.add_glyph(cds_used, errorY)
                if 'tooltips' in optionLocal and cds_names[i] is None:
                    tooltipColumns = getTooltipColumns(optionLocal['tooltips'])
                else:
                    tooltipColumns = getTooltipColumns(cdsDict[cds_name]["tooltips"])
                _, _, memoized_columns, tooltip_sources = getOrMakeColumns(list(tooltipColumns), cds_names[i], cdsDict, paramDict, jsFunctionDict, memoized_columns, aliasDict)
                sources.update(tooltip_sources)
            if cds_name == "$IGNORE":
                cds_name = varY
            if figure_cds_name is None and cds_name != "$IGNORE":
                figure_cds_name = cds_name
            elif figure_cds_name != cds_name:
                figure_cds_name = ""
            if len(variables[0]) > len(xAxisTitleBuilder):
                xAxisTitleBuilder.append(x_label)
            if len(variables[1]) > len(yAxisTitleBuilder):
                yAxisTitleBuilder.append(y_label)

        xAxisTitle = ", ".join(xAxisTitleBuilder)
        yAxisTitle = ", ".join(yAxisTitleBuilder)
        plotTitle = yAxisTitle + " vs " + xAxisTitle
        if color_axis_title is not None:
            plotTitle = f"{plotTitle} vs {color_axis_title}"

        xAxisTitle = optionLocal.get("xAxisTitle", xAxisTitle)
        yAxisTitle = optionLocal.get("yAxisTitle", yAxisTitle)
        plotTitle = optionLocal.get("plotTitle", plotTitle)

        xAxisTitleModel = makeAxisLabelFromTemplate(xAxisTitle, paramDict, meta)
        applyParametricAxisLabel(xAxisTitleModel, figureI.xaxis[0], "axis_label")
        yAxisTitleModel = makeAxisLabelFromTemplate(yAxisTitle, paramDict, meta)
        applyParametricAxisLabel(yAxisTitleModel, figureI.yaxis[0], "axis_label")
        plotTitleModel = makeAxisLabelFromTemplate(plotTitle, paramDict, meta)
        applyParametricAxisLabel(plotTitleModel, figureI.title, "text")

        if color_bar is not None:
            figureI.add_layout(color_bar, 'right')
        for iCds, iRenderers in hover_tool_renderers.items():
            figureI.add_tools(HoverTool(tooltips=cdsDict[iCds]["tooltips"], renderers=iRenderers))
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
                    if iOption in paramDict:
                        legend_options_parameters[i] = paramDict[iOption]
                for i, iOption in legend_options_parameters.items():
                    legend_options[i] = iOption['value']
                figureI.legend.update(**legend_options)
                for i, iOption in legend_options_parameters.items():
                    iOption["subscribed_events"].append(["value", figureI.legend[0], i])        
        plotArray.append(figureI)
        if "name" in optionLocal:
            plotDict[optionLocal["name"]] = figureI
    histoList = []
    for cdsKey, cdsValue in cdsDict.items():
        # Ignore unused data sources
        if cdsKey not in memoized_columns:
            continue
        # Populate the data sources - original columns
        if cdsValue["type"] == "source":
            sent_data = {}
            for key, value in memoized_columns[cdsKey].items():
                if (cdsKey, key) in sources:
                    if value["type"] == "server_derived_column":
                        sent_data[key] = value["value"]
                    elif value["type"] == "column":
                        sent_data[key] = cdsValue["data"][key]
            cdsOrig = cdsValue["cdsOrig"]
            if cdsValue['arrayCompression'] is not None:
                print("compressCDSPipe")
                sent_data = {i:removeInt64(iColumn) for i, iColumn in sent_data.items()}
                cdsCompress0, sizeMap= compressCDSPipe(sent_data, options["arrayCompression"],1)
                for keyCompressed, valueCompressed in cdsCompress0.items():
                    if isinstance(valueCompressed["array"], bytes):
                        cdsCompress0[keyCompressed]["array"] = base64.b64encode(valueCompressed["array"]).decode("utf-8")
                        cdsCompress0[keyCompressed]["actionArray"].append("base64")
                cdsOrig.inputData = cdsCompress0
                cdsOrig.sizeMap = sizeMap
            else:
                cdsOrig.data = sent_data
        elif cdsValue["type"] in ["histogram", "histo2d", "histoNd"]:
            cdsOrig = cdsValue["cdsOrig"]
            if "histograms" in cdsValue:
                for key, value in memoized_columns[cdsKey].items():
                    if key in cdsValue["histograms"]:
                        if value["type"] == "column":
                            cdsOrig.histograms[key] = cdsValue["histograms"][key]
        # Nothing needed for projections, they already come populated on initialization because of a quirk in the interface
        # In future an option can be added for creating them from array

        # Add aliases
        for key, value in memoized_columns[cdsKey].items():
            columnKey = value["name"]
            if (cdsKey, columnKey) not in sources:
                cdsFull = cdsValue["cdsFull"]
                # User defined aliases
                if value["type"] == "alias":
                    cdsFull.mapping[columnKey] = aliasDict[cdsKey][columnKey]
                # Columns directly controlled by parameter
                elif value["type"] == "parameter":
                    cdsFull.mapping[columnKey] = paramDict[value["name"]]["value"]
                    paramDict[value["name"]]["subscribed_events"].append(["value", CustomJS(args={"cdsAlias": cdsDict[cdsKey]["cdsFull"], "key": columnKey},
                                                                                            code="""
                                                                                                cdsAlias.mapping[key] = this.value;
                                                                                                cdsAlias.invalidate_column(key);
                                                                                            """)])

    for iCds, widgets in widgetDict.items():
        widgetList = widgets["widgetList"]
        index = None
        if "index" in widgets:
            index = widgets["index"]
        cdsOrig = cdsDict[iCds]["cdsOrig"]
        cdsFull = cdsDict[iCds]["cdsFull"]
        source = cdsDict[iCds]["cds"]
        histoList = []
        for cdsKey, cdsValue in cdsDict.items():
            if cdsKey not in memoized_columns:
                continue
            if cdsValue["type"] in ["histogram", "histo2d", "histoNd"] and cdsValue["source"] == iCds:
                histoList.append(cdsValue["cdsOrig"])
        # HACK: we need to add the aliasDict to the dependency tree somehow as bokeh can't find it - to be removed when dependency trees on client work with that
        callback = makeJScallback(widgetList, cdsFull, source, histogramList=histoList,
                                    cdsHistoSummary=cdsHistoSummary, profileList=profileList, aliasDict=list(aliasDict.values()), index=index)
        for iWidget in widgetList:
            if "filter" in iWidget:
                field = iWidget["filter"].field
                if memoized_columns[iCds][field]["type"] == "alias":
                    iWidget["filter"].source = cdsFull
                else:
                    iWidget["filter"].source = cdsOrig
                iWidget["filter"].js_on_change("change", callback)
            else:
                iWidget["widget"].js_on_change("value", callback)
        if index is not None:
            index["widget"].js_on_change("value", callback)
    connectWidgetCallbacks(widgetParams, widgetArray, paramDict, None)
    if isinstance(options['layout'], list) or isinstance(options['layout'], dict):
        pAll = processBokehLayoutArray(options['layout'], plotArray, plotDict)
    if options['doDraw']:
        show(pAll)
    return pAll, cdsDict[None]["cds"], plotArray, colorMapperDict, cdsDict[None]["cdsOrig"], histoList, cdsHistoSummary, profileList, paramDict, aliasDict, plotDict


def addHisto2dGlyph(fig, histoHandle, marker, options):
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
                          fill_color=mapperC, line_width=0)
        histoGlyphRenderer = fig.add_glyph(cdsHisto, histoGlyph)
        fig.add_layout(color_bar, 'right')
    elif visualization_type == "colZ":
        mapperC = {"field": "bin_count", "transform": LinearColorMapper(palette=options['palette'])}
        color_bar = ColorBar(color_mapper=mapperC['transform'], width=8, location=(0, 0),
                             title=histoHandle["variables"][1])
        if options["legend_field"] is None:
            histoGlyphRenderer = fig.scatter(x="bin_center_0", y="bin_count", fill_alpha=1, source=cdsHisto, size=options['size'],
                            color=mapperC, marker=marker, legend_label="Histogram of " + histoHandle["variables"][0])
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
    if 'color' in options:
        colorHisto = options['color']
    tooltips = None
    if "tooltips" in histoHandle:
        tooltips = histoHandle["tooltips"]
    elif "histoTooltips" in options:
        tooltips = options["histoTooltips"]
    visualization_type = "points"
    histoGlyphRenderer = None
    if "visualization_type" in options:
        visualization_type = options["visualization_type"]
    if visualization_type == "bars":
        if options['flip_histogram_axes']:
            histoGlyph = Quad(left=0, right="bin_count", bottom="bin_bottom", top="bin_top", fill_color=colorHisto)
        else:
            histoGlyph = Quad(left="bin_bottom", right="bin_top", bottom=0, top="bin_count", fill_color=colorHisto)
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

def makeBokehSliderWidget(df: pd.DataFrame, isRange: bool, params: list, paramDict: dict, **kwargs):
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
            start, end, step = params[1], params[2], params[3]
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
        start, end, step = makeSliderParameters(df, params, **kwargs)
    if isRange:
        if (start==end):
            start-=1
            end+=1
        if value is None:
            value = (start, end)
        slider = RangeSlider(title=title, start=start, end=end, step=step, value=value, name=name)
    else:
        if value is None:
            value = (start + end) * 0.5
        slider = Slider(title=title, start=start, end=end, step=step, value=value, name=name)
    return slider


def makeSliderParameters(df: pd.DataFrame, params: list, **kwargs):
    options = {
        'type': 'auto',
        'bins': 30,
        'sigma': 4,
        'limits': (0.05, 0.05),
        'title': '',
    }
    options.update(kwargs)
    name = params[0]
    start = 0
    end = 0
    step = 0
    #df[name].loc[ abs(df[name])==np.inf]=0
    try:
        if df[name].dtype=="float":                    #if type is float and has inf print error message and replace
            if (np.isinf(df[name])).sum()>0:
                print(f"makeBokehSliderWidget() - Invalid column {name} with infinity")
                raise
    except:
        pass
    if options['type'] == 'user':
        start, end, step = params[1], params[2], params[3]
    elif (options['type'] == 'auto') or (options['type'] == 'minmax'):
        if df is not None and name in df:
            start = np.nanmin(df[name])
            end = np.nanmax(df[name])
        else:
            start = 0
            end = 1
        step = (end - start) / options['bins']
    elif (options['type'] == 'unique'):
        start = np.nanmin(df[name])
        end = np.nanmax(df[name])
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
    return start, end, step    


def makeBokehSelectWidget(df: pd.DataFrame, params: list, paramDict: dict, default=None, **kwargs):
    options = {'size': 10}
    options.update(kwargs)
    # optionsPlot = []
    if options['callback'] == 'parameter':
        optionsPlot = paramDict[params[0]]["options"]
    else:
        if len(params) > 1:
            dfCategorical = df[params[0]].astype(pd.CategoricalDtype(ordered=True, categories=params[1:]))
        else:
            dfCategorical = df[params[0]]
        codes, optionsPlot = pd.factorize(dfCategorical, sort=True, na_sentinel=None)
        optionsPlot = optionsPlot.dropna().to_list()
    optionsPlot = [str(i) for i in optionsPlot]
    default_value = 0
    if isinstance(default, int):
        if 0 <= default < len(optionsPlot):
            default_value = default
        else:
            raise IndexError("Default value out of range for select widget.")
    elif default is None and options['callback'] == 'parameter':
        default_value = optionsPlot.index(str(paramDict[params[0]]["value"]))
    elif default is None:
        default_value = 0
    else:
        default_value = optionsPlot.index(str(default))
    widget_local = Select(title=params[0], value=optionsPlot[default_value], options=optionsPlot)
    filterLocal = None
    newColumn = None
    js_callback_code="""
        target.selected = [this.value]
    """
    if options['callback'] == 'parameter':
        return widget_local, filterLocal, newColumn

    mapping = {}
    for i, val in enumerate(optionsPlot):
        mapping[val] = i
    print(len(optionsPlot))
    filterLocal = MultiSelectFilter(selected=optionsPlot, field=params[0]+".factor()", how="whitelist", mapping=mapping)
    widget_local.js_on_change("value", CustomJS(args={"target":filterLocal}, code=js_callback_code))
    newColumn = {"name": params[0]+".factor()", "type": "server_derived_column", "value": codes.astype(np.int32)}
    return widget_local, filterLocal, newColumn


def makeBokehMultiSelectWidget(df: pd.DataFrame, params: list, paramDict: dict, **kwargs):
    # print("makeBokehMultiSelectWidget",params,kwargs)
    options = {'size': 4}
    options.update(kwargs)
    # optionsPlot = []
    if len(params) > 1:
        dfCategorical = df[params[0]].astype(pd.CategoricalDtype(ordered=True, categories=params[1:]))
    else:
        dfCategorical = df[params[0]]
    codes, optionsPlot = pd.factorize(dfCategorical, sort=True, na_sentinel=None)
    optionsPlot = optionsPlot.to_list()
    for i, val in enumerate(optionsPlot):
        optionsPlot[i] = str(val)
    widget_local = MultiSelect(title=params[0], value=optionsPlot, options=optionsPlot, size=options['size'])
    filterLocal = None
    newColumn = None
    if len(optionsPlot) < 31:
        mapping = {}
        for i, val in enumerate(optionsPlot):
            mapping[val] = 2**i
        # print(optionsPlot)
        filterLocal = MultiSelectFilter(selected=optionsPlot, field=params[0]+".factor()", how="any", mapping=mapping)
        widget_local.js_link("value", filterLocal, "selected")
        newColumn = {"name": params[0]+".factor()", "type": "server_derived_column", "value": (2**codes).astype(np.int32)}
    else:
        mapping = {}
        for i, val in enumerate(optionsPlot):
            mapping[val] = i
        print(len(optionsPlot))
        filterLocal = MultiSelectFilter(selected=optionsPlot, field=params[0]+".factor()", how="whitelist", mapping=mapping)
        widget_local.js_link("value", filterLocal, "selected")
        newColumn = {"name": params[0]+".factor()", "type": "server_derived_column", "value": codes.astype(np.int32)}
    return widget_local, filterLocal, newColumn


def makeBokehMultiSelectBitmaskWidget(column: dict, title: str, mapping: dict, **kwargs):
    options = {'size': 4, "how": "all"}
    options.update(kwargs)
    keys = list(mapping.keys())
    if options["how"] in ["any", "whitelist"]:
        multiselect_value = keys
    else:
        multiselect_value = []
    widget_local = MultiSelect(title=title, value=multiselect_value, options=keys, size=options["size"])
    filter_local = MultiSelectFilter(selected=multiselect_value, field=column["name"], how=options["how"], mapping=mapping)
    widget_local.js_link("value", filter_local, "selected")
    return widget_local, filter_local


def connectWidgetCallbacks(widgetParams: list, widgetArray: list, paramDict: dict, defaultCallback: CustomJS):
    for iDesc, iWidget in zip(widgetParams, widgetArray):
        optionLocal = {}
        params = iDesc[1]
        callback = None
        if len(iDesc) == 3:
            optionLocal = iDesc[2].copy()
        if "callback" not in optionLocal:
            if params[0] in paramDict:
                optionLocal["callback"] = "parameter"
            else:
                optionLocal["callback"] = "selection"
        if optionLocal["callback"] == "selection":
            callback = defaultCallback
        elif optionLocal["callback"] == "parameter":
            paramControlled = paramDict[iDesc[1][0]]
            for iEvent in paramControlled["subscribed_events"]:
                if len(iEvent) == 2:
                    iWidget.js_on_change(*iEvent)
                else:
                    if isinstance(iWidget, Toggle):
                        iEvent[0] = "active"
                    iWidget.js_link(*iEvent)
            continue
        if callback is not None:
            if isinstance(iWidget, Slider) or isinstance(iWidget, RangeSlider):
                iWidget.js_on_change("value", callback)
            else:
                iWidget.js_on_change("value", callback)
            iWidget.js_on_event("value", callback)


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

def getTooltipColumns(tooltips):
    if isinstance(tooltips, str):
        return {}
    result = set()
    tooltip_regex = re.compile(r'@(?:\w+|\{[^\}]*\})')
    for iTooltip in tooltips:
        for iField in tooltip_regex.findall(iTooltip[1]):
            if iField[1] == '{':
                result.add(iField[2:-1])
            else:
                result.add(iField[1:])
    return result

def errorBarWidthTwoSided(varError: dict, paramDict: dict, transform=None):
    if varError["type"] == "constant":
        return {"value": varError["value"]*2}
    if transform is None:
        transform = CustomJSTransform(v_func="return xs.map((x)=>2*x)")
    if varError["type"] == "parameter":
        return {"field": paramDict[varError["name"]]["value"], "transform": transform}
    return {"field": varError["name"], "transform": transform}

def errorBarWidthAsymmetric(varError: tuple, varX: dict, data_source, transform=None, paramDict = None):
    varNameX = varX["name"]
    # This JS callback can be optimized if needed
    if paramDict is None:
        paramDict = {}
    if transform is None:
        transform_lower = CustomJSTransform(args={"source":data_source, "key":varNameX}, v_func="""
            const column = [...source.get_column(key)]
            return column.map((x, i) => x-xs[i])
        """)
        transform_upper = CustomJSTransform(args={"source":data_source, "key":varNameX}, v_func="""
            const column = [...source.get_column(key)]
            return column.map((x, i) => x+xs[i])
        """)
    elif transform["type"] == "parameter":
        options_lower = {}
        options_upper = {}
        transform_upper = CustomJSTransform(args={"current":transform["default"]}, v_func="return options[current].v_compute(xs)")
        transform_lower = CustomJSTransform(args={"current":transform["default"]}, v_func="return options[current].v_compute(xs)")
        for i, iOption in transform["options"].items():
            if iOption is None:
                continue
            iParameters = iOption.get("parameters", {})
            transform_lower_i = CustomJSTransform(args={"source":data_source, "key":varNameX, **iParameters}, v_func=f"""
                const column = [...source.get_column(key)]
                return column.map((x, i) => {iOption["implementation"]}(x-xs[i]))
            """)
            options_lower[i] = transform_lower_i
            transform_upper_i = CustomJSTransform(args={"source":data_source, "key":varNameX, **iParameters}, v_func=f"""
                const column = [...source.get_column(key)]
                return column.map((x, i) => {iOption["implementation"]}(x+xs[i]))
            """)
            options_upper[i] = transform_upper_i
            for j in iParameters:
                paramDict[j]["subscribed_events"].append(["value", CustomJS(args={"mapper_lower":transform_lower_i, "mapper_upper":transform_upper_i,
                "parent_lower":transform_lower, "parent_upper": transform_upper, "param":j}, code="""
        mapper_lower.args[param] = this.value
        mapper_lower.change.emit()
        mapper_upper.args[param] = this.value
        mapper_upper.change.emit()
        parent_lower.change.emit()
        parent_upper.change.emit()
                """)])
        options_lower["null"] = options_lower["None"] = CustomJSTransform(args={"source":data_source, "key":varNameX}, v_func="""
            const column = [...source.get_column(key)]
            return column.map((x, i) => (x-xs[i]))
        """)
        options_upper["null"] = options_upper["None"] = CustomJSTransform(args={"source":data_source, "key":varNameX}, v_func="""
            const column = [...source.get_column(key)]
            return column.map((x, i) => (x+xs[i]))
        """)
        transform_lower.args["options"] = options_lower
        transform_upper.args["options"] = options_upper
        paramDict[transform["name"]]["subscribed_events"].append(["value", CustomJS(args={"mapper_lower":transform_lower, "mapper_upper":transform_upper}, code="""
            mapper_lower.args.current = this.value
            mapper_upper.args.current = this.value
            mapper_lower.change.emit()
            mapper_upper.change.emit()
        """)])
    else:
        args = transform["parameters"].copy()
        args.update({"source":data_source, "key":varNameX})
        transform_lower = CustomJSTransform(args=args, v_func=f"""
            const column = [...source.get_column(key)]
            return column.map((x, i) => {transform["implementation"]}(x-xs[i]))
        """)
        transform_upper = CustomJSTransform(args=args, v_func=f"""
            const column = [...source.get_column(key)]
            return column.map((x, i) => {transform["implementation"]}(x+xs[i]))
        """)        
        for j in transform["parameters"]:
            if j in paramDict:
                paramDict[j]["subscribed_events"].append(["value", CustomJS(args={"mapper":transform_lower, "param":j}, code="""
                    mapper.args[param] = this.value
                    mapper.change.emit()
                            """)])
                paramDict[j]["subscribed_events"].append(["value", CustomJS(args={"mapper":transform_upper, "param":j}, code="""
                    mapper.args[param] = this.value
                    mapper.change.emit()
                            """)])
            else:
                raise KeyError("Parameter "+j+" not found")
    return ({"field":varError[0]["name"], "transform":transform_lower}, {"field":varError[1]["name"], "transform":transform_upper})

def getHistogramAxisTitle(cdsDict, varName, cdsName, removeCdsName=True):
    if isinstance(varName, tuple):
        return getHistogramAxisTitle(cdsDict, varName[0], cdsName, removeCdsName)
    if cdsName is None:
        return f"{{{varName}}}"
    if cdsName in cdsDict:
        if cdsDict[cdsName]["type"] == "join":
            return f"{{{varName}}}"
        prefix = ""
        if not removeCdsName:
            prefix =  cdsName+"."
        if varName.startswith(cdsName+"."):
            varName = varName[len(cdsName)+1:]
        if '_' in varName:
            if varName == "bin_count":
                return "entries"
            x = varName.split("_")
            if x[0] == "bin":
                if "variables" in cdsDict[cdsName]:
                    variables = cdsDict[cdsName]["variables"]
                else:
                    variables = cdsDict[cdsName]["cdsOrig"].source.sample_variables
                if len(x) == 2:
                    return f"{{{variables[0]}}}"
                return f"{{{variables[int(x[2])]}}}"
            if x[0] == "quantile" and len(x) == 2:
                quantile = cdsDict[cdsName]["quantiles"][int(x[-1])]
                if cdsDict[cdsName]["type"] == "projection":
                    histogramOrig = cdsDict[cdsName]["cdsOrig"].source
                    projectionIdx = cdsDict[cdsName]["cdsOrig"].axis_idx
                    return f"quantile {quantile} {{{ histogramOrig.sample_variables[projectionIdx] }}}"
                return f"quantile {{{quantile}}}"
            if x[0] == "sum":
                range = cdsDict[cdsName]["sum_range"][int(x[-1])]
                if len(x) == 2:
                    if cdsDict[cdsName]["type"] == "projection":
                        histogramOrig = cdsDict[cdsName]["cdsOrig"].source
                        projectionIdx = cdsDict[cdsName]["cdsOrig"].axis_idx
                        return f"sum {{{histogramOrig.sample_variables[projectionIdx]}}} in [{range[0]}, {range[1]}]"
                    return f"sum in [{range[0]}, {range[1]}]"
                elif len(x) == 3:
                    if cdsDict[cdsName]["type"] == "projection":
                        histogramOrig = cdsDict[cdsName]["cdsOrig"].source
                        projectionIdx = cdsDict[cdsName]["cdsOrig"].axis_idx
                        return f"p {{{histogramOrig.sample_variables[projectionIdx]}}} in [{range[0]}, {range[1]}]"
                    return f"p in [{range[0]}, {range[1]}]"
        else:
            if cdsDict[cdsName]["type"] == "projection":
                histogramOrig = cdsDict[cdsName]["cdsOrig"].source
                projectionIdx = cdsDict[cdsName]["cdsOrig"].axis_idx
                return f"{varName} {{{histogramOrig.sample_variables[projectionIdx]}}}"
    return prefix+varName

def makeCDSDict(sourceArray, paramDict, options={}):
    # Create the cdsDict - identify types from user input and make empty ColumnDataSources
    cdsDict = {}
    for i, iSource in enumerate(sourceArray):
        iSource = iSource.copy()
        # Detect the name, error on collision
        cds_name = "anonymous_"+str(i)
        if "name" in iSource:
            cds_name = iSource["name"]
        else:
            raise ValueError("Column data sources other than the main one must have a name")
        if cds_name in cdsDict:
            raise ValueError("Column data source IDs must be unique. Multiple data sources with name: "+ str(cds_name)+ " detected.")
        cdsDict[cds_name] = iSource

        # Detect the type
        if "type" not in iSource:
            if "data" in iSource:
                cdsType = "source"
                if "arrayCompression" not in iSource:
                    iSource["arrayCompression"] = None
            elif "variables" in iSource:
                nvars = len(iSource["variables"])
                if nvars == 1:
                    cdsType = "histogram"
                elif nvars == 2:
                    cdsType = "histo2d"
                else:
                    cdsType = "histoNd"
            elif "left" in iSource or "right" in iSource:
                cdsType = "join"
            else:
                # Cannot determine type automatically
                raise ValueError(iSource)
            iSource["type"] = cdsType
        cdsType = iSource["type"]

        # Create the name for cdsOrig
        name_orig = "cdsOrig"
        if cds_name is not None:
            name_orig = cds_name+"_orig"

        # Create cdsOrig
        if cdsType == "source":
            if "arrayCompression" in iSource and iSource["arrayCompression"] is not None:
                iSource["cdsOrig"] = CDSCompress(name=name_orig)
            else:
                iSource["cdsOrig"] = ColumnDataSource(name=name_orig)
        elif cdsType == "histogram":
            weights = iSource.get("weights", None)
            iSource["cdsOrig"] = HistogramCDS(sample=iSource["variables"][0], weights=weights, name=name_orig)
            if "source" not in iSource:
                iSource["source"] = None
            if "tooltips" not in iSource:
                iSource["tooltips"] = defaultHistoTooltips
            nbins = 10
            if "nbins" in iSource:
                nbins = iSource["nbins"]
            if nbins in paramDict:
                paramDict[nbins]["subscribed_events"].append(["value", CustomJS(args={"histogram":iSource["cdsOrig"]}, code="""
                            histogram.nbins = this.value | 0;
                            histogram.change_selection();
                        """)])
                nbins = paramDict[nbins]["value"]
            histoRange = None
            if "range" in iSource:
                histoRange = iSource["range"]
            if isinstance(histoRange, str) and histoRange in paramDict:
                paramDict[histoRange]["subscribed_events"].append(["value", iSource["cdsOrig"], "range"])
                histoRange = paramDict[histoRange]["value"]
            iSource["cdsOrig"].update(nbins=nbins, range=histoRange)
        elif cdsType in ["histo2d", "histoNd"]:
            weights = iSource.get("weights", None)
            if "source" not in iSource:
                iSource["source"] = None
            iSource["cdsOrig"] = HistoNdCDS(sample_variables=iSource["variables"], weights=weights, name=name_orig)
            if "tooltips" not in iSource:
                iSource["tooltips"] = defaultHisto2DTooltips
            nbins = [10]*len(iSource["variables"])
            if "nbins" in iSource:
                nbins = iSource["nbins"].copy()
            for binsIdx, iBins in enumerate(nbins):
                if isinstance(iBins, str) and iBins in paramDict:
                    paramDict[iBins]["subscribed_events"].append(["value", CustomJS(args={"histogram":iSource["cdsOrig"], "i": binsIdx}, code="""
                        histogram.nbins[i] = this.value | 0;
                        histogram.update_nbins();
                        histogram.invalidate_cached_bins();
                        histogram.change_selection();
                    """)])
                    nbins[binsIdx] = paramDict[iBins]["value"]
            histoRange = None
            if "range" in iSource:
                histoRange = iSource["range"].copy()
            if histoRange is not None:
                for rangeIdx, iRange in enumerate(histoRange):
                    if isinstance(iRange, str) and iRange in paramDict:
                        paramDict[iRange]["subscribed_events"].append(["value", CustomJS(args={"histogram":iSource["cdsOrig"], "i": rangeIdx}, code="""
                            histogram.range[i] = this.value;
                            histogram.invalidate_cached_bins();
                            histogram.change_selection();
                        """)])
                        histoRange[rangeIdx] = paramDict[iRange]["value"]
            iSource["cdsOrig"].update(nbins=nbins, range=histoRange)
            if "axis" in iSource:
                axisIndices = iSource["axis"]
                projectionsLocal = {}
                sum_range = []
                if "sum_range" in iSource:
                    sum_range = iSource["sum_range"]
                quantiles = []
                if "quantiles" in iSource:
                    quantiles = iSource["quantiles"]
                unbinned = iSource.get("unbinned_projections", False)
                for j in axisIndices:
                    cdsProfile = HistoNdProfile(source=iSource["cdsOrig"], axis_idx=j, quantiles=quantiles, weights=weights,
                                                sum_range=sum_range, name=cds_name+"_"+str(j)+"_orig", unbinned=unbinned)
                    projectionsLocal[i] = cdsProfile
                    cdsDict[cds_name+"_"+str(j)] = {"cdsOrig": cdsProfile, "type": "projection", "name": cds_name+"_"+str(j), "variables": iSource["variables"],
                    "quantiles": quantiles, "sum_range": sum_range, "axis": j, "source": cds_name} 
                iSource["profiles"] = projectionsLocal
        elif cdsType == "join":
            left = None
            if "left" in iSource:
                left = iSource["left"]
            right = None
            if "left" in iSource:
                right = iSource["left"]
            if "left_on" in iSource:
                on_left = iSource["left_on"]
            on_right = []
            if "right_on" in iSource:
                on_right = iSource["right_on"]
            how  = iSource["how"] if "how" in iSource else "inner"
            iSource["cdsOrig"] = CDSJoin(prefix_left=left, prefix_right=right, on_left=on_left, on_right=on_right, how=how, name=name_orig)
        elif cdsType == "projection":
            axis_idx = iSource["axis_idx"][0] # Maybe support more than 1 projection axis
            quantiles = iSource["quantiles"] if "quantiles" in iSource else []
            sum_range = iSource["sum_range"] if "sum_range" in iSource else []
            unbinned = iSource["unbinned"] if "unbinned" in iSource else False
            iSource["cdsOrig"] = HistoNdProfile(axis_idx=axis_idx, quantiles=quantiles, sum_range=sum_range, name=cds_name, unbinned=unbinned)
        else:
            raise NotImplementedError("Unrecognized CDS type: " + cdsType)

    for cds_name, iSource in cdsDict.items():
        cdsOrig = iSource["cdsOrig"]
        if iSource["type"] in ["histogram", "histo2d", "histoNd"]:
            cdsOrig.source = cdsDict[iSource["source"]]["cdsFull"]
        elif iSource["type"] == "join":
            cdsOrig.left = cdsDict[iSource["left"]]["cdsFull"]
            cdsOrig.right = cdsDict[iSource["right"]]["cdsFull"]
        elif iSource["type"] == "projection":
            cdsOrig.source = cdsDict[iSource["source"]]["cdsOrig"]
            cdsOrig.weights = iSource.get("weights", cdsOrig.source.weights)
            if "tooltips" not in iSource:
                iSource["tooltips"] = defaultNDProfileTooltips(cdsOrig.source.sample_variables, cdsOrig.axis_idx, cdsOrig.quantiles, cdsOrig.sum_range)
        name_full = "cdsFull"
        if cds_name is not None:
            name_full = cds_name+"_full"
        # Add middleware for aliases
        iSource["cdsFull"] = CDSAlias(source=cdsOrig, mapping={}, name=name_full)

        # Add downsampler
        name_normal = "default source"
        if cds_name is not None:
            name_normal = cds_name
        nPoints = options["nPointRender"]
        if options["nPointRender"] in paramDict:
            nPoints = paramDict[options["nPointRender"]]["value"]
        iSource["cds"] = DownsamplerCDS(source=iSource["cdsFull"], nPoints=nPoints, name=name_normal)
        if options["nPointRender"] in paramDict:
            paramDict[options["nPointRender"]]["subscribed_events"].append(["value", CustomJS(args={"downsampler": iSource["cds"]}, code="""
                            downsampler.nPoints = this.value | 0
                            downsampler.update()
                        """)])

        if "tooltips" not in iSource:
            iSource["tooltips"] = []
    return cdsDict

def makeAxisLabelFromTemplate(template:str, paramDict:dict, meta: dict):
    components = re.split(RE_CURLY_BRACE, template)
    label = ConcatenatedString()
    for i in range(1, len(components), 2):
        if components[i] in paramDict:
            if "options" in paramDict[components[i]]:
                options = {str(j):meta.get(f"{j}.AxisTitle", str(j)) for j in paramDict[components[i]]["options"]}
                if 'None' in options:
                    options["None"] = ''
                paramDict[components[i]]["subscribed_events"].append(["change", CustomJS(args={"i":i, "label":label, "options":options}, code="""
                    label.components[i] = options[this.value];
                    label.properties.components.change.emit();
                    label.change.emit();
                """)])
            else:
                paramDict[components[i]]["subscribed_events"].append(["change", CustomJS(args={"i":i, "label":label}, code="""
                    label.components[i] = this.value;
                    label.properties.components.change.emit();
                    label.change.emit();
                """)])
            components[i] = paramDict[components[i]]["value"]
        components[i] = str(components[i]) if components[i] is not None else ''
        components[i] = meta.get(f"{components[i]}.AxisTitle", components[i])
    label.components = components
    return label

def applyParametricAxisLabel(label, target, attr):
    if isinstance(label, ConcatenatedString):
        label.js_on_change("change", CustomJS(args={"target":target, "attr":attr}, code="target[attr] = this.value"))
        target.update(**{attr:''.join(label.components)})
    elif isinstance(label, str):
        target.update(**{attr:label})

def make_transform(transform, paramDict, aliasDict, cdsDict, jsFunctionDict, parent=None, orientation=0):
    if isinstance(transform, str):
        exprTree = ast.parse(transform, filename="<unknown>", mode="eval")
        evaluator = ColumnEvaluator(None, cdsDict, paramDict, jsFunctionDict, transform, aliasDict)
        transform_parsed = evaluator.visit(exprTree.body)
        transform_parameters = {i:paramDict[i]["value"] for i in evaluator.paramDependencies}
        transform_parsed["parameters"] = transform_parameters
        if transform_parsed["type"] == "js_lambda":
            if transform_parsed["n_args"] == 1:
                transform_customjs = CustomJSTransform(args=transform_parameters, v_func=f"""
                    return xs.map({transform_parsed["implementation"]});
                """)
            elif transform_parsed["n_args"] == 2:
                transform_customjs = CustomJSTransform(args=transform_parameters, v_func=f"""
                    const ys = data_source.get_column(varY);
                    return xs.map((x, i) => ({transform_parsed["implementation"]}{"(x, ys[i])" if orientation==1 else "(ys[i],x)"}));
                """)
        elif transform_parsed["type"] == "parameter":
            if "options" not in paramDict[transform_parsed["name"]]:
                raise KeyError(transform_parsed["name"])
            js_transforms = {}
            parsed_options = {}
            default_func = paramDict[transform_parsed["name"]]["value"]
            if default_func is None:
                default_func = "None"
            transform_parsed["default"] = default_func
            transform_customjs = CustomJSTransform(args={"current":default_func}, v_func="return options[current].v_compute(xs)")
            for func_option in paramDict[transform_parsed["name"]]["options"]:
                if func_option is None:
                    option_customjs = CustomJSTransform(v_func="return xs")
                    js_transforms["None"] = option_customjs
                    parsed_options["None"] = None
                    continue
                option_parsed, option_customjs = make_transform(func_option, paramDict, aliasDict, cdsDict, jsFunctionDict, parent=transform_customjs, orientation=orientation)
                js_transforms[func_option] = option_customjs
                parsed_options[func_option] = option_parsed
            transform_parsed["options"] = parsed_options
            transform_customjs.args["options"] = js_transforms
            paramDict[transform_parsed["name"]]["subscribed_events"].append(["value", CustomJS(args={"mapper":transform_customjs}, code="""
        mapper.args.current = this.value
        mapper.change.emit()
                """)])
        if transform_parameters is not None:
            for j in transform_parameters:
                paramDict[j]["subscribed_events"].append(["value", CustomJS(args={"mapper":transform_customjs, "param":j}, code="""
        mapper.args[param] = this.value
        mapper.change.emit()
                """)])
        if parent is not None:
            transform_customjs.js_on_change("change", CustomJS(args={"parent":parent}, code="parent.change.emit()"))
        return (transform_parsed, transform_customjs)
    return (None, CustomJSTransform(v_func="return xs"))
        # Result has to be either lambda expression or parameter where all options are lambdas, signature must take 1 vector
        # later 2 vectors

def modify_2d_transform(transform_orig_parsed, transform_orig_js, varY, data_source):
    if transform_orig_parsed is None:
        return transform_orig_js
    if transform_orig_parsed["type"] == "js_lambda":
        if transform_orig_parsed["n_args"] == 1:
            return transform_orig_js
        # Reconstruct the callbacks for args
        transform_new = CustomJSTransform(args={"varY":varY, "data_source":data_source, **transform_orig_js.args}, v_func=transform_orig_js.v_func)
        transform_orig_js.js_on_change("change", CustomJS(args={"mapper_new": transform_new}, code="mapper_new.args = {...mapper_new.args, ...this.args}; mapper_new.change.emit()"))
        return transform_new
    if transform_orig_parsed["type"] == "parameter":
        transform_new = CustomJSTransform(args={"current":transform_orig_js.args["current"]}, v_func="return options[current].v_compute(xs)")
        options_new = {}
        for i, iOption in transform_orig_parsed["options"].items():
            options_new[i] = modify_2d_transform(iOption, transform_orig_js.args["options"][i], varY, data_source)
        transform_new.args["options"] = options_new
        transform_orig_js.js_on_change("change", CustomJS(args={"mapper_new": transform_new}, code="mapper_new.args.current = this.args.current; mapper_new.change.emit()"))
        return transform_new
