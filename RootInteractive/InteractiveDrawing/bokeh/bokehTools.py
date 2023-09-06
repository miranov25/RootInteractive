from bokeh.plotting import figure, show, output_file
from bokeh.models import ColorBar, HoverTool, VBar, HBar, Quad
from bokeh.models.sources import ColumnDataSource, CDSView
from bokeh.models.transforms import CustomJSTransform
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.widgets.tables import ScientificFormatter, DataTable
from bokeh.models.widgets.markups import Div
from bokeh.models.plots import Plot
from bokeh.transform import *
from RootInteractive.InteractiveDrawing.bokeh.ConcatenatedString import ConcatenatedString
from RootInteractive.InteractiveDrawing.bokeh.compileVarName import getOrMakeColumns
from RootInteractive.Tools.aliTreePlayer import *
from bokeh.layouts import *
import logging
from IPython import get_ipython
from bokeh.models.widgets import Select, Slider, RangeSlider, MultiSelect, Panel, TableColumn, TextAreaInput, Toggle, Spinner
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
from RootInteractive.InteractiveDrawing.bokeh.ColumnFilter import ColumnFilter
from RootInteractive.InteractiveDrawing.bokeh.LazyIntersectionFilter import LazyIntersectionFilter
from RootInteractive.InteractiveDrawing.bokeh.ClientLinearFitter import ClientLinearFitter
from RootInteractive.InteractiveDrawing.bokeh.CDSStack import CDSStack
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
                "dash", "hex", "inverted_triangle", "asterisk", "square_x", "x"]

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

RE_VALID_NAME = re.compile(r"^[a-zA-Z_$][0-9a-zA-Z_$]*$")

IS_PANDAS_1 = pd.__version__.split('.')[0] == '1'

def mergeFigureArrays(figureArrayOld, figureArrayNew):
    if len(figureArrayOld) == 0:
        return figureArrayNew
    if len(figureArrayNew) == 0:
        return figureArrayOld
    figureArrayMerged = figureArrayOld.copy()
    if isinstance(figureArrayMerged[-1], dict):
        optionsOld = figureArrayMerged.pop().copy()
        figureArrayMerged = figureArrayMerged + figureArrayNew
        if isinstance(figureArrayNew[-1], dict):
            optionsNew = figureArrayMerged.pop()
            optionsOld.update(optionsNew)
        figureArrayMerged.append(optionsOld)
        return figureArrayMerged
    return figureArrayMerged + figureArrayNew

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
            'sizing_mode': 'scale_width',
            'legend_visible': True
        }

    widgetRows = []
    nRows = len(widgetArray)
    # get/apply global options if exist
    if len(widgetLayoutDesc) > 0 and isinstance(widgetLayoutDesc[-1], dict):
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
            if "plot_width" in optionLocal:
                figure.plot_width = int(optionLocal["plot_width"] / nRows)
            if "plot_height" in optionLocal:
                figure.plot_height = optionLocal["plot_height"]
            if figure.legend:
                figure.legend.visible = optionLocal["legend_visible"]
        else:
            if "plot_width" in optionLocal:
                figure.width = int(optionLocal["plot_width"] / nRows)
            if "plot_height" in optionLocal:
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


def makeBokehHistoTable(histoDict: dict, include: str, exclude: str, rowwise=False, paramDict=None, **kwargs):
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

    if paramDict is None:
        paramDict = {}

    for iHisto in histoDict:
        # We are only interested in histograms so we filter the dict for histograms
        if histoDict[iHisto]["type"] not in ["histogram", "histo2d", "histoNd"]:
            continue
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
                sources.append(getOrMakeCdsOrig(histoDict, paramDict, iHisto))
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
                    sources.append(getOrMakeCdsOrig(histoDict, paramDict, iHisto))
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

    sourceArray = [{"data": dfQuery, "arrayCompression": options["arrayCompression"], "name":None, "tooltips": options["tooltips"], "meta":options.get("meta", None)}] + sourceArray

    paramDict = {}
    for param in parameterArray:
        paramDict[param["name"]] = param.copy()
        paramDict[param["name"]]["subscribed_events"] = []

    widgetParams = []
    widgetArray = []
    widgetDict = {}

    selectionTables = []

    cdsDict = makeCDSDict(sourceArray, paramDict, options=options)

    jsFunctionDict = {}
    for i in jsFunctionArray:
        customJsArgList = {}
        transformType = i.get("type", "customJS")
        if transformType == "linearFit":
            iFitter = jsFunctionDict[i["name"]] = ClientLinearFitter(varY=i["varY"], source=getOrMakeCdsFull(cdsDict, paramDict, i.get("source", None)), alpha=i.get("alpha", 0), weights=i.get("weights", None))
            if isinstance(i["varX"], str):
                if i["varX"] in paramDict:
                    varX = paramDict[i["varX"]]["value"]
                    paramDict[i["varX"]]["subscribed_events"].append(["value", CustomJS(args={"fitter":iFitter}, code="""
        fitter.varX = this.value
        fitter.onChange()
                """)])
                else:
                    varX = [i["varX"]]
            else:
                varX = i["varX"]
            iFitter.varX = varX
            break
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
        if isinstance(i, dict):
            variables = i.get("variables", [])
        customJsArgList = {}
        transform = None
        if not isinstance(i, dict):
            if len(i) == 2:
                i = {"name":i[0], "expr":i[1]}
            if len(i) == 3:
                i = {"name":i[0], "expr":i[1], "context": i[2]}
        aliasSet.add(i["name"])
        if "parameters" in i:
            for j in i["parameters"]:
                customJsArgList[j] = paramDict[j]["value"]
        if "transform" in i and i["transform"] in jsFunctionDict:
            transform = jsFunctionDict[i["transform"]]
        elif "v_func" in i:
            transform = CustomJSNAryFunction(parameters=customJsArgList, fields=variables, v_func=i["v_func"])
        elif "func" in i:
            if i["func"] in paramDict:
                transform = CustomJSNAryFunction(parameters=customJsArgList, fields=variables, func=paramDict[i["func"]]["value"])
                paramDict[i["func"]]["subscribed_events"].append(["value", CustomJS(args={"mapper":transform}, code="""
        mapper.func = this.value
        mapper.update_func()
                """)])
            else:
                transform = CustomJSNAryFunction(parameters=customJsArgList, fields=variables, func=i["func"])
        if "expr" in i:
            exprTree = ast.parse(i["expr"], filename="<unknown>", mode="eval")
            context = i.get("context", None)
            evaluator = ColumnEvaluator(context, cdsDict, paramDict, jsFunctionDict, i["expr"], aliasDict)
            result = evaluator.visit(exprTree.body)
            if result["type"] == "javascript":
                func = "return "+result["implementation"]
                fields = list(evaluator.aliasDependencies)
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
            fields = i.get("variables", None)
            if isinstance(fields, str):
                if fields in paramDict:
                    paramDict[fields]["subscribed_events"].append(["value", CustomJS(args={"column_name":i["name"], "table":cdsDict[context]["cdsFull"]}, code="""
                        table.mapping[column_name].fields = this.value
                        table.invalidate_column(column_name)
                        """)])
                    fields = paramDict[fields]["value"]
        source = i.get("context", None)
        if source not in aliasDict:
            aliasDict[source] = {}
        aliasDict[source][i["name"]] = {"fields": fields, "transform": transform}
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
        optionLocal = optionGroup.copy()
        nvars = len(variables)
        if isinstance(variables[-1], dict):
            logging.info("Option %s", variables[-1])
            optionLocal.update(variables[-1])
            nvars -= 1
        if variables[0] == 'selectionTable':
            columns = [
                TableColumn(title="Affected source", field="cdsName"),
                TableColumn(title="Field", field="field"),
                TableColumn(title="Widget type", field="type"),
                TableColumn(title="Value", field="value"),
                TableColumn(title="Is active", field="active"),
            ]
            widget = DataTable(columns=columns)
            selectionTables.append(widget)
            plotArray.append(widget)
            if "name" in optionLocal:
                plotDict[optionLocal["name"]] = widget       
            continue
        elif variables[0] == 'div':
            text_content = optionLocal.get("text", variables[1])
            widget = Div(text=text_content)
            plotArray.append(widget)
            if "name" in optionLocal:
                plotDict[optionLocal["name"]] = widget           
            continue
        elif variables[0] == 'descriptionTable':
            data_source = optionLocal.get("source", None)
            meta_fields = optionLocal.get("meta_fields", ["AxisTitle", "Description"])
            if "variables" in optionLocal:
                used_variables = optionLocal["variables"]
            else:
                meta = cdsDict[data_source]["meta"]
                used_variables = [*{i.split(".")[0] for i in meta.keys()}]
            widget = makeDescriptionTable(cdsDict, data_source, used_variables, meta_fields)
            plotArray.append(widget)
            if "name" in optionLocal:
                plotDict[optionLocal["name"]] = widget           
            continue
        elif variables[0] == 'table':
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
        elif variables[0] == 'tableHisto':
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
            cdsHistoSummary, tableHisto = makeBokehHistoTable(cdsDict, include=TOptions["include"], exclude=TOptions["exclude"], rowwise=TOptions["rowwise"], paramDict=paramDict)
            plotArray.append(tableHisto)
            if "name" in optionLocal:
                plotDict[optionLocal["name"]] = tableHisto
            continue
        elif variables[0] in ALLOWED_WIDGET_TYPES:
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
                if optionWidget["callback"] == "selection":
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
                zero_step = False
                if step == 0:
                    zero_step = True
                    step = 1
                localWidgetMin = Spinner(title=f"min({label})", value=start, step=step, format=formatter)
                localWidgetMax = Spinner(title=f"max({label})", value=end, step=step, format=formatter)
                if zero_step:
                    localWidgetMin.disabled = True
                    localWidgetMax.disabled = True
                if optionWidget["callback"] == "parameter":
                    pass
                else:
                    widgetFilter = RangeFilter(range=[start, end], field=variables[1][0], name=variables[1][0])
                    if zero_step:
                        widgetFilter.active = False
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
                widgetFilter.active = False
                if variables[0] == 'spinnerRange':
                    localWidgetMin.disabled = True
                    localWidgetMax.disabled = True
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
                    filter.active = !this.active
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
                widgetDictLocal = {"widget": localWidget, "type": variables[0], "key": varName, "filter": widgetFilter, "isIndex": optionWidget.get("index", False)}
                widgetDict[cds_used]["widgetList"].append(widgetDictLocal)
            continue
        elif variables[0] == "textQuery":
            optionWidget = {}
            if len(variables) >= 2:
                optionWidget.update(variables[-1])
            cds_used = optionWidget.get("source", None)
            # By default, uses all named variables from the data source - but they can't be known at this point yet
            localWidget = TextAreaInput(**optionWidget)
            widgetName = optionWidget.get("name", f"$widget_{len(plotArray)}")
            plotArray.append(localWidget)
            plotDict[widgetName] = localWidget
            transform = CustomJSNAryFunction(parameters=customJsArgList, fields=[], func=optionWidget.get("value", "return true"))
            localWidget.js_on_change("value", CustomJS(args={"mapper":transform}, code="""
                mapper.func = this.value
                mapper.update_func()
                        """))
            aliasDict[cds_used][widgetName] = {"fields": None, "transform": transform}
            memoized_columns[cds_used][widgetName] = {"type":"alias", "name":widgetName}
            widgetFilter = ColumnFilter(field=widgetName, name=widgetName)
            if cds_used not in widgetDict:
                widgetDict[cds_used] = {"widgetList":[]}
            widgetDict[cds_used]["widgetList"].append({"widget": localWidget, "type": variables[0], "key": widgetName, "filter": widgetFilter})
            continue
        elif variables[0] == "text":
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
        elif isinstance(variables[0], str):
            raise NotImplementedError(f"Widget not supported: {variables[0]}")

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
            cds_used = makeCdsSel(cdsDict, paramDict, cds_name)
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
            cds_used = None
            if cds_name != "$IGNORE":
                cds_used = makeCdsSel(cdsDict, paramDict, cds_name)
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
                        makeCdsSel(cdsDict, paramDict, cds_name)
                        cdsDict[cds_name]["cdsSel"].js_on_change('change', CustomJS(code="""
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
                    if optionLocal.get("showColorAxisTitle", False):
                        color_axis_title_model = makeAxisLabelFromTemplate(color_axis_title, paramDict, meta)
                        applyParametricAxisLabel(color_axis_title_model, color_bar, "title")
            elif 'color' in optionLocal:
                color=optionLocal['color']
            elif cds_name in cdsDict and isinstance(cds_used.source.source, CDSStack):
                color = factor_cmap("_source_index", colorAll[10], cdsDict[cds_name]["cdsOrig"].activeSources)
                stack_sources = cdsDict[cds_name]["sources"]
                if isinstance(stack_sources, str):
                    if stack_sources in paramDict:
                        paramDict[stack_sources]["subscribed_events"].append(("value", color["transform"], "factors"))
            elif cds_name in cdsDict and isinstance(cds_used.source.source, HistoNdProfile) and isinstance(cds_used.source.source.weights, list):
                weights = cdsDict[cds_name]["weights"]
                color = factor_cmap("weights", colorAll[10], [str(i) for i in paramDict["weights"]["value"]])
                paramDict[weights]["subscribed_events"].append(("value", color["transform"], "factors"))    
            else:
                color = colorAll[max(length, 4)][i]
            markerSize = optionLocal['size']
            if markerSize in paramDict:
                markerSize = paramDict[markerSize]['value']
            if len(variables) > 2:
                logging.info("Option %s", variables[2])
                optionLocal.update(variables[2])
            varY = variables[1][i % lengthY]
            markersAll = optionLocal["markers"]
            markerField = variables_dict.get("marker_field", None)
            if markerField is not None:
                # TODO: Also support marker fields created on the client
                if markerField["type"] == "column":
                    uniq_values = np.unique(dfQuery[varName])
                elif markerField["type"] == "server_derived_column":
                    uniq_values = np.unique(column[0]["value"])
                else:
                    raise NotImplementedError("Marker field not implemented for aliases yet")
                marker = factor_mark(markerField["name"], markersAll, uniq_values)
            elif cds_name in cdsDict and isinstance(cds_used.source.source, CDSStack):
                marker = factor_mark("_source_index", markersAll, cds_used.source.source.activeSources)
                stack_sources = cdsDict[cds_name]["sources"]
                if isinstance(stack_sources, str):
                    if stack_sources in paramDict:
                        paramDict[stack_sources]["subscribed_events"].append(("value", marker["transform"], "factors"))
            elif cds_name in cdsDict and isinstance(cds_used.source.source, HistoNdProfile) and isinstance(cds_used.source.source.weights, list):
                weights = cdsDict[cds_name]["weights"]
                marker = factor_mark("weights", markersAll, [str(i) for i in paramDict["weights"]["value"]])
                paramDict[weights]["subscribed_events"].append(("value", marker["transform"], "factors"))            
            else:
                try:
                    marker = optionLocal['markers'][i]
                except:
                    marker = optionLocal['markers']

            if isinstance(varY, str) and varY in cdsDict and cdsDict[varY]["type"] in ["histogram", "histo2d"]:
                histoHandle = cdsDict[varY]
                makeCdsSel(cdsDict, paramDict, varY)
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
                    if optionLocal["legend_field"] is not None:
                        drawnGlyph = figureI.scatter(x=dataSpecX, y=dataSpecY, fill_alpha=1, source=cds_used, size=markerSize,
                                    color=color, marker=marker, legend_field=optionLocal["legend_field"])
                    elif isinstance(cdsDict[cds_name]["cdsOrig"], CDSStack):
                        #TODO: Use more reasonable axis labels
                        x_label = getHistogramAxisTitle(cdsDict, varNameX, cds_name)
                        if x_transform:
                            x_label = f"{{{x_transform}}} {x_label}"
                        y_label = getHistogramAxisTitle(cdsDict, varNameY, cds_name)
                        if y_transform:
                            y_label = f"{{{y_transform}}} {y_label}"
                        drawnGlyph = figureI.scatter(x=dataSpecX, y=dataSpecY, fill_alpha=1, source=cds_used, size=markerSize,
                                    color=color, marker=marker, legend_field="_source_index")
                    elif isinstance(cdsDict[cds_name]["cdsOrig"], HistoNdProfile) and isinstance(cdsDict[cds_name]["cdsOrig"].weights, list):
                        x_label = getHistogramAxisTitle(cdsDict, varNameX, cds_name)
                        if x_transform:
                            x_label = f"{{{x_transform}}} {x_label}"
                        y_label = getHistogramAxisTitle(cdsDict, varNameY, cds_name)
                        if y_transform:
                            y_label = f"{{{y_transform}}} {y_label}"
                        drawnGlyph = figureI.scatter(x=dataSpecX, y=dataSpecY, fill_alpha=1, source=cds_used, size=markerSize,
                                    color=color, marker=marker, legend_field="weights")                                        
                    else:
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
                    tooltipColumns = getTooltipColumns(cdsDict[cds_name].get("tooltips", []))
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

        xAxisTitle = optionLocal.get("xAxisTitle", xAxisTitle)
        yAxisTitle = optionLocal.get("yAxisTitle", yAxisTitle)
        plotTitle = yAxisTitle + " vs " + xAxisTitle
        if color_axis_title is not None:
            plotTitle = f"{plotTitle} vs {color_axis_title}"
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
            if cdsDict[iCds].get("tooltips", None) is not None:
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
        if cdsKey not in memoized_columns or "cdsOrig" not in cdsValue:
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
                        cdsCompress0[keyCompressed]["history"].append("base64_decode")
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
        if "cdsFull" not in cdsValue:
            continue
        aliasArrayLocal = set()
        weakAll = []
        for key, value in memoized_columns[cdsKey].items():
            columnKey = value["name"]
            if re.match(RE_VALID_NAME, columnKey):
                if (cdsKey, columnKey) in sources:
                    weakAll.append(columnKey)
                elif cdsKey in aliasDict and columnKey in aliasDict[cdsKey] and aliasDict[cdsKey][columnKey].get("fields", []) is not None:
                    weakAll.append(columnKey)
        cdsFull = cdsValue["cdsFull"]
        for key, value in memoized_columns[cdsKey].items():
            columnKey = value["name"]
            if (cdsKey, columnKey) not in sources:
                # User defined aliases
                if value["type"] in ["alias", "expr"]:
                    cdsFull.mapping[columnKey] = aliasDict[cdsKey][columnKey]
                    if "transform" in aliasDict[cdsKey][columnKey]:
                        aliasArrayLocal.add(aliasDict[cdsKey][columnKey]["transform"])
                    if "fields" in aliasDict[cdsKey][columnKey] and aliasDict[cdsKey][columnKey]["fields"] is None:
                        aliasDict[cdsKey][columnKey]["fields"] = weakAll
                        aliasDict[cdsKey][columnKey]["transform"].fields = weakAll
                # Columns directly controlled by parameter
                elif value["type"] == "parameter":
                    cdsFull.mapping[columnKey] = paramDict[value["name"]]["value"]
                    paramDict[value["name"]]["subscribed_events"].append(["value", CustomJS(args={"cdsAlias": cdsDict[cdsKey]["cdsFull"], "key": columnKey},
                                                                                            code="""
                                                                                                cdsAlias.mapping[key] = this.value;
                                                                                                cdsAlias.invalidate_column(key);
                                                                                            """)])
        cdsFull.columnDependencies = list(aliasArrayLocal)

    for iCds, widgets in widgetDict.items():
        widgetList = widgets["widgetList"]
        cdsOrig = cdsDict[iCds]["cdsOrig"]
        cdsFull = cdsDict[iCds]["cdsFull"]
        cdsSel = cdsDict[iCds].get("cdsSel", None)
        histoList = []
        for cdsKey, cdsValue in cdsDict.items():
            if cdsKey not in memoized_columns or "cdsOrig" not in cdsValue:
                continue
            if cdsValue["type"] in ["histogram", "histo2d", "histoNd"] and cdsValue.get("source", None) == iCds:
                if isinstance(cdsValue["cdsOrig"], CDSStack):
                    for i in cdsValue["cdsOrig"].sources:
                        histoList.append(i)
                else:
                    histoList.append(cdsValue["cdsOrig"])
        intersectionFilter = LazyIntersectionFilter(filters=[])
        for iWidget in widgetList:
            if "filter" in iWidget:
                field = iWidget["filter"].field
                if memoized_columns[iCds][field]["type"] in ["alias", "expr"]:
                    iWidget["filter"].source = cdsFull
                else:
                    iWidget["filter"].source = cdsOrig    
                intersectionFilter.filters.append(iWidget["filter"])
        if cdsSel is not None and len(intersectionFilter.filters)>0:
            cdsSel.filter = intersectionFilter
        if len(intersectionFilter.filters)>0:
            for i in histoList:
                i.filter = intersectionFilter
    connectWidgetCallbacks(widgetParams, widgetArray, paramDict, None)
    if selectionTables:
        widgetNames = []
        widgetFields = []
        widgetTypes = []
        widgetActive = []
        widgetValues = []
        i=0
        selectionCDS = ColumnDataSource()
        for iCds, widgets in widgetDict.items():
            widgetList = widgets["widgetList"]
            for iWidget in widgetList:
                widgetActive.append(iWidget["filter"].active)
                widgetNames.append(iCds if iCds is not None else "")
                widgetFields.append(iWidget["filter"].field)
                iFilter = iWidget["filter"]
                if isinstance(iFilter, RangeFilter):
                    widgetTypes.append("range")
                    widgetValues.append(f"[{iFilter.range[0]}, {iFilter.range[1]}]")
                    iFilter.js_on_change("change", CustomJS(args={"target":selectionCDS, "i":i}, code="""
                        target.patch({"value": [[i, "["+this.range.join(', ')+ "]" ]]},null);
                    """))
                elif isinstance(iWidget["filter"], MultiSelectFilter):
                    widgetTypes.append(f"multiselect ({iFilter.how})")
                    widgetValues.append(f"{{{', '.join(iFilter.selected)}}}")
                    iFilter.js_on_change("change", CustomJS(args={"target":selectionCDS, "i":i}, code="""
                        target.patch({"value": [[i, '{'+this.selected.join(', ')+'}' ]]},null);
                    """))
                elif isinstance(iWidget["filter"], ColumnFilter):
                    widgetTypes.append("expression")
                    if iFilter.field in aliasDict[iCds]:
                        widgetValues.append(aliasDict[iCds][iFilter.field]["transform"].func)
                        aliasDict[iCds][iFilter.field]["transform"].js_on_change("change", CustomJS(args={"target":selectionCDS, "i":i}, code="""
                            target.patch({"value": [[i, this.func ]]},null);
                        """))
                    else:
                        widgetValues.append("true")
                i += 1
        selectionTableData={"type":widgetTypes, "cdsName":widgetNames, "field":widgetFields, "value":widgetValues, "active":widgetActive}
        selectionCDS.data = selectionTableData
    for i in selectionTables:
        i.source = selectionCDS
        i.view = CDSView(source=selectionCDS)
    if isinstance(options['layout'], list) or isinstance(options['layout'], dict):
        pAll = processBokehLayoutArray(options['layout'], plotArray, plotDict)
    if options['doDraw']:
        show(pAll)
    return pAll, makeCdsSel(cdsDict, paramDict, cds_name), plotArray, colorMapperDict, cdsDict[None]["cdsOrig"], histoList, cdsHistoSummary, [], paramDict, aliasDict, plotDict


def addHisto2dGlyph(fig, histoHandle, marker, options):
    visualization_type = "heatmap"
    if "visualization_type" in options:
        visualization_type = options["visualization_type"]
    cdsHisto = histoHandle["cdsSel"]

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
    cdsHisto = histoHandle["cdsSel"]
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
        if IS_PANDAS_1:
            codes, optionsPlot = pd.factorize(dfCategorical, sort=True, na_sentinel=None)
        else:
            codes, optionsPlot = pd.factorize(dfCategorical, sort=True, use_na_sentinel=False)
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
        target.change.emit()
    """
    if options['callback'] == 'parameter':
        return widget_local, filterLocal, newColumn

    mapping = {}
    for i, val in enumerate(optionsPlot):
        mapping[val] = i
    print(len(optionsPlot))
    filterLocal = MultiSelectFilter(selected=[optionsPlot[default_value]], field=params[0]+".factor()", how="whitelist", mapping=mapping)
    widget_local.js_on_change("value", CustomJS(args={"target":filterLocal}, code=js_callback_code))
    newColumn = {"name": params[0]+".factor()", "type": "server_derived_column", "value": codes.astype(np.int32)}
    return widget_local, filterLocal, newColumn


def makeBokehMultiSelectWidget(df: pd.DataFrame, params: list, paramDict: dict, **kwargs):
    # print("makeBokehMultiSelectWidget",params,kwargs)
    options = {'size': 4}
    options.update(kwargs)
    # optionsPlot = []
    if options['callback'] == 'parameter':
        optionsPlot = [str(i) for i in paramDict[params[0]]["options"]]
        optionsSelected = [str(i) for i in paramDict[params[0]]["value"]]
    else:
        if len(params) > 1:
            dfCategorical = df[params[0]].astype(pd.CategoricalDtype(ordered=True, categories=params[1:]))
        else:
            dfCategorical = df[params[0]]
        if IS_PANDAS_1:
            codes, optionsPlot = pd.factorize(dfCategorical, sort=True, na_sentinel=None)
        else:
            codes, optionsPlot = pd.factorize(dfCategorical, sort=True, use_na_sentinel=False)
        optionsPlot = optionsPlot.to_list()
        for i, val in enumerate(optionsPlot):
            optionsPlot[i] = str(val)
        optionsSelected = optionsPlot
    widget_local = MultiSelect(title=params[0], value=optionsSelected, options=optionsPlot, size=options['size'])
    if options['callback'] == 'parameter':
        return widget_local, None, None
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


def defaultNDProfileTooltips(profDescription, varNames):
    axis_idx = profDescription.get("axis", profDescription.get("axis_idx", 0))
    quantiles = profDescription.get("quantiles", [])
    sumRanges = profDescription.get("sumRanges", [])
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
        if cdsDict[cdsName]["type"] in ["join", "stack"]:
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
                    variables = cdsDict[cdsDict[cdsName]["source"]]["variables"]
                if len(x) == 2:
                    return f"{{{variables[0]}}}"
                return f"{{{variables[int(x[2])]}}}"
            if x[0] == "quantile" and len(x) == 2:
                quantile = cdsDict[cdsName]["quantiles"][int(x[-1])]
                if cdsDict[cdsName]["type"] == "projection":
                    cdsOrig = cdsDict[cdsName]["cdsOrig"]
                    cdsOrig = cdsOrig if isinstance(cdsOrig, HistoNdProfile) else cdsOrig.sources[0]
                    projectionIdx = cdsOrig.axis_idx
                    return f"quantile {quantile} {{{cdsDict[cdsDict[cdsName]['source']]['variables'][projectionIdx] }}}"
                return f"quantile {{{quantile}}}"
            if x[0] == "sum":
                range = cdsDict[cdsName]["sum_range"][int(x[-1])]
                if len(x) == 2:
                    if cdsDict[cdsName]["type"] == "projection":
                        cdsOrig = cdsDict[cdsName]["cdsOrig"]
                        cdsOrig = cdsOrig if isinstance(cdsOrig, HistoNdProfile) else cdsOrig.sources[0]
                        projectionIdx = cdsOrig.axis_idx
                        return f"sum {{{cdsDict[cdsDict[cdsName]['source']]['variables'][projectionIdx]}}} in [{range[0]}, {range[1]}]"
                    return f"sum in [{range[0]}, {range[1]}]"
                elif len(x) == 3:
                    if cdsDict[cdsName]["type"] == "projection":
                        cdsOrig = cdsDict[cdsName]["cdsOrig"]
                        cdsOrig = cdsOrig if isinstance(cdsOrig, HistoNdProfile) else cdsOrig.sources[0]
                        projectionIdx = cdsOrig.axis_idx
                        return f"p {{{cdsDict[cdsDict[cdsName]['source']]['variables'][projectionIdx]}}} in [{range[0]}, {range[1]}]"
                    return f"p in [{range[0]}, {range[1]}]"
        else:
            if cdsDict[cdsName]["type"] == "projection":
                cdsOrig = cdsDict[cdsName]["cdsOrig"]
                cdsOrig = cdsOrig if isinstance(cdsOrig, HistoNdProfile) else cdsOrig.sources[0]
                projectionIdx = cdsOrig.axis_idx
                return f"{varName} {{{cdsDict[cdsDict[cdsName]['source']]['variables'][projectionIdx]}}}"
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
        iSource["meta"] = iSource.get("meta", None)

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
            elif "axis_idx" in iSource:
                cdsType = "projection"
            elif "left" in iSource or "right" in iSource:
                cdsType = "join"
            elif "sources" in iSource:
                cdsType = "stack"
            else:
                # Cannot determine type automatically
                raise ValueError(iSource)
            iSource["type"] = cdsType
        cdsType = iSource["type"]

        # Create cdsOrig
        if cdsType == "source":
            if iSource["meta"] is None:
                try:
                    iSource["meta"] = iSource["data"].meta.metaData.copy()
                except AttributeError:
                    iSource["meta"] = {}
        elif cdsType == "stack":
            sources = iSource["sources"]
            if isinstance(sources, str):
                if sources in paramDict:
                    if "mapping" in iSource:
                        iSource["sources_all"] = list(iSource["mapping"].values())
                        iSource["active"] = paramDict[sources]["value"]
                    else:
                        iSource["sources_all"] = paramDict[sources]["options"]
                        iSource["active"] = paramDict[sources]["value"]
            else:
                iSource["sources_all"] = sources
        if cdsType in ["histo2d", "histoNd"]:
            sample_variables = iSource["variables"]
            for j in range(len(sample_variables)):
                if f"{cds_name}_{str(j)}" not in cdsDict:
                    cdsDict[f"{cds_name}_{str(j)}"] = {"type": "projection", "name": f"{cds_name}_{str(j)}", "source":cds_name, "weights":iSource.get("weights", None), 
                            "quantiles": iSource.get("quantiles", []), "sum_range": iSource.get("sum_range", []), "axis":j,
                            "sources":iSource.get("sources", None), "unbinned":iSource.get("unbinned_projections", False)}

    for cds_name, iSource in cdsDict.items():
        if iSource["type"] == "source":
            iSource["nPointRender"] = iSource.get("nPointRender", options.get("nPointRender", 1000))
        else:
            iSource["nPointRender"] = iSource.get("nPointRender", -1)
        if "tooltips" not in iSource:
            if iSource["type"] == "projection":
                iSource["tooltips"] = defaultNDProfileTooltips(iSource, cdsDict[iSource["source"]]["variables"])
            elif iSource["type"] == "histogram":
                iSource["tooltips"] = defaultHistoTooltips
            elif iSource["type"] in ["histo2d", "histoNd"]:
                iSource["tooltips"] = defaultHisto2DTooltips
    return cdsDict

def getOrMakeCdsOrig(cdsDict: dict, paramDict: dict, key: str):
    if key in cdsDict:
        iCds = cdsDict[key]
        if "cdsOrig" in iCds:
            return iCds["cdsOrig"]
        cdsName = iCds.get("name", key)
        if cdsName is None:
            cdsName = "cdsOrig"
        cdsType = iCds["type"]
        if cdsType == "source":
            if iCds.get("arrayCompression", None) is not None:
                iCds["cdsOrig"] = CDSCompress(name=cdsName)
            else:
                iCds["cdsOrig"] = ColumnDataSource(name=cdsName)
        elif cdsType == "projection":
            source = iCds.get("source", None)
            weightsOrig = cdsDict[source].get("weights", None)
            weightsNew = iCds.get("weights", None)
            unbinned = iCds.get("unbinned", False)
            quantiles = iCds.get("quantiles", [])
            sum_range = iCds.get("sum_range", [])
            axis_idx = iCds.get("axis", iCds.get("axis_idx", 0))
            if isinstance(axis_idx, list):
                axis_idx = axis_idx[0]
            cdsOrig = getOrMakeCdsOrig(cdsDict, paramDict, source)
            if isinstance(cdsOrig, CDSStack):
                projections = []
                # Query optimizer - eliminate join if not needed
                if weightsOrig in paramDict and isinstance(paramDict[weightsOrig]["value"], list):
                    weightsValue = paramDict[weightsOrig]["value"]
                    cdsProfile = HistoNdProfile(source=cdsOrig.sources[0], axis_idx=axis_idx, quantiles=quantiles, weights=weightsValue,
                            sum_range=sum_range, name=cdsName, unbinned=unbinned)
                    paramDict[weightsOrig]["subscribed_events"].append(["value", CustomJS(args={"cds":cdsProfile}, code="""
                        cds.weights=this.value.map(x => x === "None" ? null : x)
                                                                                          """)])
                    iCds["cdsOrig"] = cdsProfile
                    iCds["legend_field"] = "weights"
                    iCds["marker"] = "weights"
                    iCds["colorZvar"] = "weights"
                else:
                    for i in cdsOrig.sources:
                        weights = i.weights if weightsOrig == weightsNew else weightsNew
                        cdsProfile = HistoNdProfile(source=i, axis_idx=axis_idx, quantiles=quantiles, weights=weights,
                                                    sum_range=sum_range, name=f"{cdsName}_{i.name}", unbinned=unbinned)
                        if weightsOrig == weightsNew:
                            i.js_link("weights", cdsProfile, "weights")
                        projections.append(cdsProfile)
                    cdsMultiProfile = CDSStack(sources=projections, mapping=cdsOrig.mapping, activeSources=cdsOrig.activeSources)
                    cdsOrig.js_link("activeSources", cdsMultiProfile, "activeSources")
                    iCds["cdsOrig"] = cdsMultiProfile
                    iCds["sources"] = cdsDict[source]["sources"]
                    iCds["legend_field"] = "_source_index"
                    iCds["marker"] = "weights"
                    iCds["colorZvar"] = "weights"
            else:
                weightsValue = weightsNew if weightsNew not in paramDict else paramDict[weightsNew]["value"]
                cdsProfile = HistoNdProfile(source=cdsOrig, axis_idx=axis_idx, quantiles=quantiles, weights=weightsValue,
                                            sum_range=sum_range, name=cdsName, unbinned=unbinned)
                if weightsNew in paramDict:
                    paramDict[weightsNew]["subscribed_events"].append(("value", cdsProfile, "weights"))
                iCds["cdsOrig"] = cdsProfile
        elif cdsType in ["histo2d", "histoNd"]:
            multi_axis = None
            weights = iCds.get("weights", None)
            sample_variables = iCds["variables"]
            source = getOrMakeCdsFull(cdsDict, paramDict, iCds.get("source", None))
            if "source" not in iCds:
                iCds["source"] = None
            if "tooltips" not in iCds:
                iCds["tooltips"] = defaultHisto2DTooltips
            nbins = iCds.get("nbins", 10)
            if isinstance(nbins, int):
                nbins = [nbins]*len(sample_variables)
            if isinstance(nbins, str):
                raise NotImplemented("Using ND binning as one parameter is not supported yet, please provide an array of parameters instead")
            nbins_value = [paramDict[i]["value"] if i in paramDict else i for i in nbins]
            histoRange = iCds.get("range", None)
            range_value = [paramDict[i]["value"] if isinstance(i, str) and i in paramDict else i for i in histoRange] if histoRange is not None else None
            sample_value = [sample if sample not in paramDict else paramDict[sample]["value"] for sample in sample_variables]
            weights_value = weights if weights is None or weights not in paramDict else paramDict[weights]["value"]
            if weights in paramDict:
                if isinstance(paramDict[weights]["value"], list):
                    if multi_axis is not None:
                        raise NotImplementedError("Multiple multiselect axes for histogram not supported yet")
                    else:
                       multi_axis = ("weights",)
            for i, iVar in enumerate(sample_variables):
                if iVar in paramDict:
                    if isinstance(paramDict[iVar]["value"], list):
                        if multi_axis is not None:
                            raise NotImplementedError("Multiple multiselect axes for histogram not supported yet")
                        else:
                            multi_axis = ("variables", i)
            if multi_axis is None:
                cdsOrig = HistoNdCDS(source=source, sample_variables=sample_value, weights=weights_value, name=cdsName, nbins=nbins_value, range=range_value)
                iCds["cdsOrig"] = cdsOrig
                histogramsLocal = [cdsOrig]
            else:
                histogramsLocal = []
                acc = iCds
                for i in multi_axis:
                    acc = acc[i]
                iCds["sources"] = acc
                histoOptions = paramDict[acc]["options"]
                for i in histoOptions:
                    if multi_axis[0] == "weights":
                        weights_value = i
                    else:
                        sample_value[multi_axis[1]] = i
                    cdsOrig = HistoNdCDS(source=source, sample_variables=sample_value.copy(), weights=weights_value, name=f"{cdsName}[{i}]", nbins=nbins_value, range=range_value)
                    histogramsLocal.append(cdsOrig)
                cdsOrig = CDSStack(sources=histogramsLocal, activeSources=[str(i) for i in paramDict[acc]["value"]], mapping={str(value):i for (i, value) in enumerate(histoOptions)})
                iCds["cdsOrig"] = cdsOrig
                paramDict[acc]["subscribed_events"].append(["value", cdsOrig, "activeSources"])
            for binsIdx, iBins in enumerate(nbins):
                if isinstance(iBins, str) and iBins in paramDict:
                    paramDict[iBins]["subscribed_events"].append(["value", CustomJS(args={"histograms":histogramsLocal, "i": binsIdx}, code="""
                    for (const histogram of histograms){
                        histogram.nbins[i] = this.value | 0;
                        histogram.update_nbins();
                        histogram.invalidate_cached_bins();
                        histogram.change_selection();
                        }
                    """)])
            if histoRange is not None:
                for rangeIdx, iRange in enumerate(histoRange):
                    if isinstance(iRange, str) and iRange in paramDict:
                        paramDict[iRange]["subscribed_events"].append(["value", CustomJS(args={"histograms":histogramsLocal, "i": rangeIdx}, code="""
                        for (const histogram of histograms){
                            histogram.range[i] = this.value;
                            histogram.invalidate_cached_bins();
                            histogram.change_selection();
                            }
                        """)])
            if multi_axis != ("weights",) and weights in paramDict:
                for i in histogramsLocal:
                    paramDict[weights]["subscribed_events"].append(["value", i, "weights"])
            for i, sample in enumerate(sample_variables):
                if sample in paramDict and multi_axis != ("variables", i):
                    paramDict[sample]["subscribed_events"].append(["value", CustomJS(args={"histograms":histogramsLocal, "i":i}, code="""
                    for (const histogram of histograms){
                        histogram.sample_variables[i] = this.value;
                        histogram.invalidate_cached_bins();
                        histogram.change_selection();
                        }
                        """)])
        elif cdsType == "stack":
            iSource = iCds
            sources = iSource["sources"]
            sourcesObjs = [getOrMakeCdsFull(cdsDict, paramDict, i) for i in iSource["sources_all"]]
            mapping = iSource.get("mapping", None)
            if mapping is None:
                mappingNew = {value:i for (i, value) in enumerate(iSource["sources_all"])}
            elif isinstance(mapping, dict):
                sourcesInverse = {value:i for i,value in enumerate(iSource["sources_all"])}
                mappingNew = {i:sourcesInverse[value] for (i, value) in mapping.items()}
            iSource["cdsOrig"] = CDSStack(sources=sourcesObjs, mapping=mappingNew, activeSources=iSource["active"])
            if "tooltips" not in iSource:
                tooltipsOrig = cdsDict[iSource["sources_all"][0]].get("tooltips", None)
                if tooltipsOrig is not None:
                    iSource["tooltips"] = tooltipsOrig
            if sources in paramDict:
                paramDict[sources]["subscribed_events"].append(("value", iSource["cdsOrig"], "activeSources"))
        elif cdsType == "join":
            iSource = iCds
            left = iSource.get("left", None)
            right = iSource.get("right", None)
            on_left = iSource.get("left_on", [])
            on_right = iSource.get("right_on", [])
            how = iSource.get("how", "inner")
            sourceLeft = getOrMakeCdsFull(cdsDict, paramDict, left)
            sourceRight = getOrMakeCdsFull(cdsDict, paramDict, right)
            iSource["cdsOrig"] = CDSJoin(left=sourceLeft, right=sourceRight, prefix_left=left or "cdsFull", prefix_right=right or "csFull", on_left=on_left, on_right=on_right, how=how, name=cdsName)
        elif cdsType == "histogram":
            iSource = iCds
            weights = iSource.get("weights", None)
            sample = iSource["variables"][0]
            multi_axis = None 
            sample_value = sample if sample not in paramDict else paramDict[sample]["value"]
            weights_value = weights if weights is None or weights not in paramDict else paramDict[weights]["value"]
            nbins = iSource.get("nbins", 10)
            nbins_value = nbins if isinstance(nbins, int) else paramDict[nbins]["value"] 
            histoRange = iSource.get("range", None)
            range_value = histoRange if not isinstance(histoRange, str) else paramDict[histoRange]["value"]
            source = getOrMakeCdsFull(cdsDict, paramDict, iSource.get("source", None))
            if weights in paramDict:
                if isinstance(paramDict[weights]["value"], list):
                    if multi_axis is not None:
                        raise NotImplementedError("Multiple multiselect axes for histogram not supported yet")
                    else:
                       multi_axis = ("weights",)
            if sample in paramDict:
                if isinstance(paramDict[sample]["value"], list):
                    if multi_axis is not None:
                        raise NotImplementedError("Multiple multiselect axes for histogram not supported yet")
                    else:
                        multi_axis = ("variables",0)
            if multi_axis is None:
                cdsOrig = HistogramCDS(source=source, sample=sample_value, weights=weights_value, name=cdsName, nbins=nbins_value, range=range_value)
                iSource["cdsOrig"] =cdsOrig 
                histogramsLocal = [cdsOrig]
            else:
                histogramsLocal = []
                acc = iSource
                for i in multi_axis:
                    acc = acc[i]
                iSource["sources"] = acc
                histoOptions = paramDict[acc]["options"]
                for i in histoOptions:
                    if multi_axis[0] == "weights":
                        weights_value = i
                    else:
                        sample_value = i
                    cdsOrig = HistogramCDS(source=source, sample=sample_value, weights=weights_value, name=f"{cdsName}_{i}", nbins=nbins_value, range=range_value)
                    histogramsLocal.append(cdsOrig)
                cdsOrig = CDSStack(sources=histogramsLocal, activeSources=paramDict[acc]["value"], mapping={value:i for (i, value) in enumerate(histoOptions)})
                paramDict[acc]["subscribed_events"].append(["value", cdsOrig, "activeSources"])
                iSource["cdsOrig"] = cdsOrig
            if "source" not in iSource:
                iSource["source"] = None
            if "tooltips" not in iSource:
                iSource["tooltips"] = defaultHistoTooltips
            if nbins in paramDict:
                paramDict[nbins]["subscribed_events"].append(["value", CustomJS(args={"histograms":histogramsLocal}, code="""
                        for (const histogram of histograms){
                            histogram.nbins = this.value | 0;
                            histogram.change_selection();
                            }
                        """)])
            if isinstance(histoRange, str) and histoRange in paramDict:
                for i in histogramsLocal:
                    paramDict[histoRange]["subscribed_events"].append(["value", i, "range"])
            if multi_axis != ("weights",) and weights in paramDict:
                for i in histogramsLocal:
                    paramDict[weights]["subscribed_events"].append(["value", i, "weights"])
            if sample in paramDict and multi_axis != ("variables", 0):
                for i in histogramsLocal:
                    paramDict[sample]["subscribed_events"].append(["value", i, "sample"])
        return iCds["cdsOrig"]

def getOrMakeCdsFull(cdsDict: dict, paramDict: dict, key: str):
    if key in cdsDict:
        if "cdsFull" in cdsDict[key]:
            return cdsDict[key]["cdsFull"]
        name_full = "cdsFull"
        if key is not None:
            name_full = f"{key}_full"
        cdsDict[key]["cdsFull"] = CDSAlias(source=getOrMakeCdsOrig(cdsDict, paramDict, key), mapping={}, name=name_full)
        return cdsDict[key]["cdsFull"]

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

def makeCdsSel(cdsDict, paramDict, key):
    cds_used = cdsDict[key]
    if "cdsSel" in cds_used:
        return cds_used["cdsSel"]
    cds_name = key if key is not None else "default cds"
    nPoints = cds_used.get("nPointRender",-1)
    if nPoints in paramDict:
        nPoints = paramDict[cds_used["nPointRender"]]["value"]
    cdsSel = DownsamplerCDS(source=getOrMakeCdsFull(cdsDict, paramDict, key), nPoints=nPoints, name=cds_name)
    if cds_used["nPointRender"] in paramDict:
        paramDict[cds_used["nPointRender"]]["subscribed_events"].append(["value", CustomJS(args={"downsampler": cdsSel}, code="""
                        downsampler.nPoints = this.value | 0
                        downsampler.update()
                    """)])
    cds_used["cdsSel"] = cdsSel
    return cdsSel

def makeDescriptionTable(cdsDict, cdsName, fields, meta_fields):
    cds = ColumnDataSource()
    new_dict = {}
    columns = []
    new_dict["name"] = [] 
    for j in fields:
        new_dict["name"].append(str(j))
    columns.append(TableColumn(field="name", title="name"))
    for i in meta_fields:
        column = []
        for j in fields:
            value = cdsDict[cdsName]["meta"].get(f"{j}.{i}", "")
            column.append(value)
        new_dict[i] = column
        columns.append(TableColumn(field=i, title=i))
    cds.data = new_dict
    return DataTable(source=cds, columns=columns)
