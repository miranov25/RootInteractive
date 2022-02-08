from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar, HoverTool, VBar, HBar, Quad
from bokeh.models.transforms import CustomJSTransform
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.widgets.tables import ScientificFormatter, DataTable
from bokeh.transform import *
from RootInteractive.InteractiveDrawing.bokeh.compileVarName import getOrMakeColumns
from RootInteractive.Tools.aliTreePlayer import *
from bokeh.layouts import *
from bokeh.palettes import *
from bokeh.io import push_notebook, curdoc
import logging
from IPython import get_ipython
from bokeh.models.widgets import DataTable, Select, Slider, RangeSlider, MultiSelect, CheckboxGroup, Panel, Tabs, TableColumn
from bokeh.models import CustomJS, ColumnDataSource
from RootInteractive.Tools.pandaTools import pandaGetOrMakeColumn
from RootInteractive.InteractiveDrawing.bokeh.bokehVisJS3DGraph import BokehVisJSGraph3D
from RootInteractive.InteractiveDrawing.bokeh.HistogramCDS import HistogramCDS
from RootInteractive.InteractiveDrawing.bokeh.HistoNdCDS import HistoNdCDS
from RootInteractive.Tools.compressArray import compressCDSPipe
from RootInteractive.InteractiveDrawing.bokeh.CDSCompress import CDSCompress
from RootInteractive.InteractiveDrawing.bokeh.HistoStatsCDS import HistoStatsCDS
from RootInteractive.InteractiveDrawing.bokeh.HistoNdProfile import HistoNdProfile
from RootInteractive.InteractiveDrawing.bokeh.DownsamplerCDS import DownsamplerCDS
from RootInteractive.InteractiveDrawing.bokeh.CDSAlias import CDSAlias
from RootInteractive.InteractiveDrawing.bokeh.CustomJSNAryFunction import CustomJSNAryFunction
from RootInteractive.InteractiveDrawing.bokeh.CDSJoin import CDSJoin
import numpy as np
import pandas as pd
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

BOKEH_DRAW_ARRAY_VAR_NAMES = ["X", "Y", "varZ", "colorZvar", "marker_field", "legend_field", "errX", "errY"]

ALLOWED_WIDGET_TYPES = ["slider", "range", "select", "multiSelect"]

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
    const size = cdsOrig.length;
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
            const col = cdsOrig.get_column(key);
            const widgetValue = widget.value;
            const widgetStep = widget.step;
            for(let i=0; i<size; i++){
                isSelected[i] &= (col[i] >= widgetValue-0.5*widgetStep);
                isSelected[i] &= (col[i] <= widgetValue+0.5*widgetStep);
            }
        }
        if(widgetType == "RangeSlider"){
            const col = cdsOrig.get_column(key);
            const low = widget.value[0];
            const high = widget.value[1];
            for(let i=0; i<size; i++){
                isSelected[i] &= (col[i] >= low);
                isSelected[i] &= (col[i] <= high);
            }
        }
        if(widgetType == "Select"){
            const col = cdsOrig.get_column(key);
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
            const col = cdsOrig.get_column(key);
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
            const col = cdsOrig.get_column(key);
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
        console.log(isSelected.reduce((a,b)=>a+b, 0));
        cdsSel.booleans = isSelected
        cdsSel.update()
        console.log(cdsSel._downsampled_indices.length);
        console.log(cdsSel.nPoints)
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


def processBokehLayoutArray(widgetLayoutDesc, widgetArray: list, isHorizontal: bool=False, options: dict=None):
    """
    apply layout on plain array of bokeh figures, resp. interactive widgets
    :param widgetLayoutDesc: array or dict desciption of layout
    :param widgetArray: input plain array of widgets/figures
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
    if isinstance(widgetLayoutDesc, dict):
        tabs = []
        for i, iPanel in widgetLayoutDesc.items():
            tabs.append(Panel(child=processBokehLayoutArray(iPanel, widgetArray), title=i))
        return Tabs(tabs=tabs)
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

    for i, iWidget in enumerate(widgetLayoutDesc):
        if isinstance(iWidget, dict):
            widgetRows.append(processBokehLayoutArray(iWidget, widgetArray, isHorizontal=False, options=optionLocal))
            continue
        if isinstance(iWidget, list):
            widgetRows.append(processBokehLayoutArray(iWidget, widgetArray, isHorizontal=not isHorizontal, options=optionLocal))
            continue

        figure = widgetArray[iWidget]
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

    if isHorizontal:
        return row(widgetRows, sizing_mode=options['sizing_mode'])
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
        # We are only interested in histograms so we filter the dict for histograms
        if histoDict[iHisto]["type"] == "histogram":
            histo_names.append(histoDict[iHisto]["name"])
            histo_columns.append("bin_count")
            bin_centers.append("bin_center")
            edges_left.append("bin_bottom")
            edges_right.append("bin_top")
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
        'palette': Spectral6,
        "markers": bokehMarkers,
        "colors": 'Category10',
        "rescaleColorMapper": False,
        "filter": '',
        'doDraw': False,
        "legend_field": None,
        "legendTitle": None,
        'nPointRender': 10000,
        "nbins": 10,
        "weights": None,
        "range": None,
        "flip_histogram_axes": False,
        "show_histogram_error": False,
        "arrayCompression": None,
        "removeExtraColumns": True,
        "xAxisTitle": None,
        "yAxisTitle": None,
        "plotTitle": None,
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
            else:
                transform = CustomJSNAryFunction(parameters=customJsArgList, fields=i["variables"], func=i["func"])
            if "context" in i:
                if i["context"] not in aliasDict:
                    aliasDict[i["context"]] = {}
                aliasDict[i["context"]][i["name"]] = {"fields": i["variables"], "transform": transform}
            else:
                aliasDict[None][i["name"]] = {"fields": i["variables"], "transform": transform}
            if "parameters" in i:
                for j in i["parameters"]:
                    paramDict[j]["subscribed_events"].append(["value", CustomJS(args={"mapper":transform, "param":j}, code="""
            mapper.parameters[param] = this.value
            mapper.update_args()
                    """)])


    plotArray = []
    colorAll = all_palettes[options['colors']]
    colorMapperDict = {}
    cdsHistoSummary = None

    memoized_columns = {}
    sources = set()

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

        # Create cdsOrig
        if cdsType == "source":
            if "arrayCompression" in iSource and iSource["arrayCompression"] is not None:
                iSource["cdsOrig"] = CDSCompress()
            else:
                iSource["cdsOrig"] = ColumnDataSource()
        elif cdsType == "histogram":
            nbins = 10
            if "nbins" in iSource:
                nbins = iSource["nbins"]
            weights = None
            if "weights" in iSource:
                weights = iSource["weights"]
            histoRange = None
            if "range" in iSource:
                histoRange = iSource["range"]
            if "source" not in iSource:
                iSource["source"] = None
            iSource["cdsOrig"] = HistogramCDS(nbins=nbins, sample=iSource["variables"][0], weights=weights, range=histoRange)
            if "tooltips" not in iSource:
                iSource["tooltips"] = defaultHistoTooltips
        elif cdsType in ["histo2d", "histoNd"]:
            nbins = [10]*len(iSource["variables"])
            if "nbins" in iSource:
                nbins = iSource["nbins"]
            weights = None
            if "weights" in iSource:
                weights = iSource["weights"]
            histoRange = None
            if "range" in iSource:
                histoRange = iSource["range"]
            if "source" not in iSource:
                iSource["source"] = None
            iSource["cdsOrig"] = HistoNdCDS(nbins=nbins, sample_variables=iSource["variables"], weights=weights, range=histoRange)
            if "tooltips" not in iSource:
                iSource["tooltips"] = defaultHisto2DTooltips
            #TODO: Add projections
            if "axis" in iSource:
                axisIndices = iSource["axis"]
                projectionsLocal = {}
                sum_range = []
                if "sum_range" in iSource:
                    sum_range = iSource["sum_range"]
                quantiles = []
                if "quantiles" in iSource:
                    quantiles = iSource["quantiles"]
                for j in axisIndices:
                    cdsProfile = HistoNdProfile(source=iSource["cdsOrig"], axis_idx=j, quantiles=quantiles,
                                                sum_range=sum_range, name=cds_name+"_"+str(j))
                    projectionsLocal[i] = cdsProfile
                    tooltips = defaultNDProfileTooltips(iSource["variables"], j, quantiles, sum_range)
                    cdsDict[cds_name+"_"+str(j)] = {"cdsOrig": cdsProfile, "type": "projection", "name": cds_name+"_"+str(j), "variables": iSource["variables"],
                    "quantiles": quantiles, "sum_range": sum_range, "axis": j, "tooltips": tooltips, "source": cds_name} 
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
            how  = "inner"
            if "how" in iSource:
                how = iSource["how"]
            iSource["cdsOrig"] = CDSJoin(prefix_left=left, prefix_right=right, on_left=on_left, on_right=on_right, how=how)
        else:
            raise NotImplementedError("Unrecognized CDS type: " + cdsType)
            
    for iSource in cdsDict.values():
        # Add middleware for aliases
        iSource["cdsFull"] = CDSAlias(source=iSource["cdsOrig"], mapping={})

        # Add downsampler
        iSource["cds"] = DownsamplerCDS(source=iSource["cdsFull"], nPoints=options["nPointRender"])

        if "tooltips" not in iSource:
            iSource["tooltips"] = []

    profileList = []

    optionsChangeList = []
    for i, variables in enumerate(figureArray):
        if isinstance(variables, dict):
            optionsChangeList.append(i)

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
            plotArray.append(makeBokehDataTable(dfQuery, cdsDict[None]["cdsOrig"], TOptions['include'], TOptions['exclude']))
            continue
        if variables[0] == 'tableHisto':
            histoListLocal = []
            for key, value in cdsDict.items():
                if value["type"] in ["histogram", "histo2d", "histoNd"]:
                    histoListLocal.append(key)
            # We just want to add them to the dependency tree
            _, _, memoized_columns, used_names_local = getOrMakeColumns("bin_count", histoListLocal, cdsDict, paramDict, jsFunctionDict, memoized_columns)
            sources.update(used_names_local)
            TOptions = {'rowwise': False}
            if len(variables) > 1:
                TOptions.update(variables[1])
            cdsHistoSummary, tableHisto = makeBokehHistoTable(cdsDict, rowwise=TOptions["rowwise"])
            plotArray.append(tableHisto)
            continue
        if variables[0] in ALLOWED_WIDGET_TYPES:
            optionWidget = {}
            if len(variables) == 3:
                optionWidget = variables[2].copy()
            fakeDf = None
            if "callback" not in optionLocal:
                if variables[1][0] in paramDict:
                    optionWidget["callback"] = "parameter"
                    varName = variables[1]
                else:
                    optionWidget["callback"] = "selection"
                    column, cds_names, memoized_columns, used_names_local = getOrMakeColumns(variables[1][0], None, cdsDict, paramDict, jsFunctionDict, memoized_columns)
                    varName = column[0]["name"]
                    if column[0]["type"] == "column":
                        fakeDf = {varName: dfQuery[varName]}
                    elif column[0]["type"] == "server_derived_column":
                        fakeDf = {varName: column[0]["value"]}
                    sources.update(used_names_local)
            if variables[0] == 'slider':
                localWidget = makeBokehSliderWidget(fakeDf, False, variables[1], paramDict, **optionWidget)
            if variables[0] == 'range':
                localWidget = makeBokehSliderWidget(fakeDf, True, variables[1], paramDict, **optionWidget)
            if variables[0] == 'select':
                localWidget = makeBokehSelectWidget(fakeDf, variables[1], paramDict, **optionWidget)
            if variables[0] == 'multiSelect':
                localWidget = makeBokehMultiSelectWidget(fakeDf, variables[1], paramDict, **optionWidget)
            plotArray.append(localWidget)
            if localWidget:
                widgetArray.append(localWidget)
                widgetParams.append(variables)
            if optionWidget["callback"] == "selection":
                widgetDict[variables[1][0]] = localWidget
            continue
        xAxisTitle = ""
        yAxisTitle = ""
        plotTitle = ""

        for varY in variables[1]:
            if hasattr(dfQuery, "meta") and '.' not in varY:
                yAxisTitle += dfQuery.meta.metaData.get(varY + ".AxisTitle", varY)
            else:
                yAxisTitle += getHistogramAxisTitle(cdsDict, varY, cds_name, False)
            yAxisTitle += ','
        for varX in variables[0]:
            if hasattr(dfQuery, "meta") and '.' not in varX:
                xAxisTitle += dfQuery.meta.metaData.get(varX + ".AxisTitle", varX)
            else:
                xAxisTitle += getHistogramAxisTitle(cdsDict, varX, cds_name, False)
            xAxisTitle += ','
        xAxisTitle = xAxisTitle[:-1]
        yAxisTitle = yAxisTitle[:-1]

        optionLocal = optionGroup.copy()
        nvars = len(variables)
        if isinstance(variables[-1], dict):
            logging.info("Option %s", variables[-1])
            optionLocal.update(variables[-1])
            nvars -= 1

        if optionLocal["xAxisTitle"] is not None:
            xAxisTitle = optionLocal["xAxisTitle"]
        if optionLocal["yAxisTitle"] is not None:
            yAxisTitle = optionLocal["yAxisTitle"]
        plotTitle += yAxisTitle + " vs " + xAxisTitle
        if optionLocal["plotTitle"] is not None:
            plotTitle = optionLocal["plotTitle"]

        variablesLocal = [None]*len(BOKEH_DRAW_ARRAY_VAR_NAMES)
        for axis_index, axis_name  in enumerate(BOKEH_DRAW_ARRAY_VAR_NAMES):
            if axis_index < nvars:
                variablesLocal[axis_index] = variables[axis_index].copy()
            elif axis_name in optionLocal:
                variablesLocal[axis_index] = optionLocal[axis_name]
            if variablesLocal[axis_index] is not None and not isinstance(variablesLocal[axis_index], list):
                variablesLocal[axis_index] = [variablesLocal[axis_index]]
        lengthX = len(variables[0])
        lengthY = len(variables[1])
        length = max(j is not None and len(j) for j in variablesLocal)
        cds_names = None
        if "source" in optionLocal:
            cds_names = optionLocal["source"]
        for i, iY in enumerate(variablesLocal[1]):
            if iY in cdsDict and cdsDict[iY]["type"] in ["histogram", "histo2d"]:
                variablesLocal[1][i] += ".bin_count"
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
            continue
        else:
            figureI = figure(plot_width=options['plot_width'], plot_height=options['plot_height'], title=plotTitle,
                             tools=options['tools'], x_axis_type=options['x_axis_type'],
                             y_axis_type=options['y_axis_type'])

        figureI.xaxis.axis_label = xAxisTitle
        figureI.yaxis.axis_label = yAxisTitle

        lengthX = len(variables[0])
        lengthY = len(variables[1])
        length = max(len(variables[0]), len(variables[1]))
        color_bar = None
        mapperC = None
        cmap_cds_name = None

        hover_tool_renderers = {}

        figure_cds_name = None
        mapperC = None

        for i in range(length):
            variables_dict = {}
            for axis_index, axis_name  in enumerate(BOKEH_DRAW_ARRAY_VAR_NAMES):
                variables_dict[axis_name] = variablesLocal[axis_index]
                if isinstance(variables_dict[axis_name], list):
                    variables_dict[axis_name] = variables_dict[axis_name][i % len(variables_dict[axis_name])]
            cds_name = cds_names[i]
            varNameX = variables_dict["X"]["name"]
            varNameY = variables_dict["Y"]["name"]
            varColor = variables_dict["colorZvar"]
            if varColor is not None:
                if mapperC is not None:
                    color = mapperC
                else:
                    rescaleColorMapper = optionLocal["rescaleColorMapper"] or varColor["type"] == "parameter" or cdsDict[cds_name]["type"] in ["histogram", "histo2d", "histoNd"]
                    if not rescaleColorMapper and cdsDict[cds_name]["type"] == "source":
                        low = np.nanmin(cdsDict[cds_name]["data"][varColor["name"]])
                        high= np.nanmax(cdsDict[cds_name]["data"][varColor["name"]])
                        mapperC = linear_cmap(field_name=varColor["name"], palette=optionLocal['palette'], low=low, high=high)
                    else:
                        if varColor["name"] in colorMapperDict:
                            mapperC = colorMapperDict[varColor["name"]]
                        else:
                            mapperC = {"field": varColor["name"], "transform": LinearColorMapper(palette=optionLocal['palette'])}
                            colorMapperDict[varColor["name"]] = mapperC
                    # HACK for projections - should probably just remove the rows as there's no issue with joins at all
                    if cdsDict[cds_name]["type"] == "projection" and not rescaleColorMapper and varColor["name"].split('_')[0] == 'bin':
                        cdsDict[cds_name]['cdsOrig'].js_on_change('change', CustomJS(code="""
                        const col = this.data[field]
                        const isOK = this.data.isOK
                        const low = col.map((x,i) => isOK[i] ? col[i] : Infinity).reduce((acc, cur)=>Math.min(acc,cur), Infinity);
                        const high = col.map((x,i) => isOK[i] ? col[i] : -Infinity).reduce((acc, cur)=>Math.max(acc,cur), -Infinity);
                        cmap.high = high;
                        cmap.low = low;
                        """, args={"field": mapperC["field"], "cmap": mapperC["transform"]})) 
                    color = mapperC
                    # Also add the color bar
                    if varColor["type"] == "parameter":
                        axis_title = paramDict[varColor["name"]]["value"]
                    else:
                        axis_title = getHistogramAxisTitle(cdsDict, varColor["name"], cmap_cds_name)
                    color_bar = ColorBar(color_mapper=mapperC['transform'], width=8, location=(0, 0), title=axis_title)
                    if varColor["type"] == "parameter":
                        paramDict[varColor["name"]]["subscribed_events"].append(["value", color_bar, "title"])
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
            varX = variables[0][i % lengthX]
            varY = variables[1][i % lengthY]
            cds_used = cdsDict[cds_name]["cds"]

            if varY in cdsDict and cdsDict[varY]["type"] in ["histogram", "histo2d"]:
                histoHandle = cdsDict[varY]
                if histoHandle["type"] == "histogram":
                    colorHisto = colorAll[max(length, 4)][i]
                    addHistogramGlyph(figureI, histoHandle, marker, colorHisto, markerSize, optionLocal)
                elif histoHandle["type"] == "histo2d":
                    addHisto2dGlyph(figureI, histoHandle, marker, optionLocal)
            else:
                drawnGlyph = None
                colorMapperCallback = """
                glyph.fill_color={...glyph.fill_color, field:this.value}
                glyph.line_color={...glyph.line_color, field:this.value}
                """
                if optionLocal["legend_field"] is None:
                    x_label = getHistogramAxisTitle(cdsDict, varNameX, cds_name)
                    y_label = getHistogramAxisTitle(cdsDict, varNameY, cds_name)
                    drawnGlyph = figureI.scatter(x=varNameX, y=varNameY, fill_alpha=1, source=cds_used, size=markerSize,
                                color=color, marker=marker, legend_label=y_label + " vs " + x_label)
                else:
                    drawnGlyph = figureI.scatter(x=varNameX, y=varNameY, fill_alpha=1, source=cds_used, size=markerSize,
                                color=color, marker=marker, legend_field=optionLocal["legend_field"])
                if "colorZvar" in optionLocal and optionLocal["colorZvar"] in paramDict:
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
                    if cds_name not in hover_tool_renderers:
                        hover_tool_renderers[cds_name] = []
                    hover_tool_renderers[cds_name].append(drawnGlyph)
                if variables_dict['errX'] is not None:
                    errWidthX = errorBarWidthTwoSided(variables_dict['errX'], paramDict)
                    errorX = VBar(top=varNameY, bottom=varNameY, width=errWidthX, x=varNameX, line_color=color)
                    figureI.add_glyph(cds_used, errorX)
                if variables_dict['errY'] is not None:
                    errWidthY = errorBarWidthTwoSided(variables_dict['errY'], paramDict)
                    errorY = HBar(left=varNameX, right=varNameX, height=errWidthY, y=varNameY, line_color=color)
                    figureI.add_glyph(cds_used, errorY)
                if 'tooltips' in optionLocal and cds_names[i] is None:
                    tooltipColumns = getTooltipColumns(optionLocal['tooltips'])
                else:
                    tooltipColumns = getTooltipColumns(cdsDict[cds_name]["tooltips"])
                _, _, memoized_columns, tooltip_sources = getOrMakeColumns(list(tooltipColumns), cds_names[i], cdsDict, paramDict, jsFunctionDict, memoized_columns, aliasDict)
                sources.update(tooltip_sources)
            if figure_cds_name is None:
                figure_cds_name = cds_name
            elif figure_cds_name != cds_name:
                figure_cds_name = ""

        if color_bar != None:
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
                cdsCompress0, sizeMap= compressCDSPipe(sent_data, options["arrayCompression"],1)
                cdsOrig.inputData = cdsCompress0
                cdsOrig.sizeMap = sizeMap
            else:
                cdsOrig.data = sent_data
        elif cdsValue["type"] in ["histogram", "histo2d", "histoNd"]:
            cdsOrig = cdsValue["cdsOrig"]
            cdsOrig.source = cdsDict[cdsValue["source"]]["cdsFull"]
            if cdsValue["source"] is None:
                histoList.append(cdsOrig)
            if "histograms" in cdsValue:
                for key, value in memoized_columns[cdsKey].items():
                    if key in cdsValue["histograms"]:
                        if value["type"] == "column":
                            cdsOrig.histograms[key] = cdsValue["histograms"][key]  
        elif cdsValue["type"] == "join":
            cdsOrig = cdsValue["cdsOrig"]
            cdsOrig.left = cdsDict[cdsValue["left"]]["cdsFull"]
            cdsOrig.right = cdsDict[cdsValue["right"]]["cdsFull"] 
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
                                                                                                cdsAlias.compute_function(key);
                                                                                                cdsAlias.change.emit();
                                                                                            """)])

    cdsFull = cdsDict[None]["cdsFull"]
    source = cdsDict[None]["cds"]
    callbackSel = makeJScallbackOptimized(widgetDict, cdsFull, source, histogramList=histoList,
                                          cdsHistoSummary=cdsHistoSummary, profileList=profileList, aliasDict=list(aliasDict.values()))
    connectWidgetCallbacks(widgetParams, widgetArray, paramDict, callbackSel)
    if isinstance(options['layout'], list) or isinstance(options['layout'], dict):
        pAll = processBokehLayoutArray(options['layout'], plotArray)
    if options['doDraw']:
        show(pAll)
    return pAll, cdsDict[None]["cds"], plotArray, colorMapperDict, cdsDict[None]["cdsOrig"], histoList, cdsHistoSummary, profileList, paramDict, aliasDict


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
                          fill_color=mapperC)
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
    #df[name].loc[ abs(df[name])==np.inf]=0
    try:
        if df[name].dtype=="float":                    #if type is float and has inf print error message and replace
            if (np.isinf(df[name])).sum()>0:
                print(f"makeBokehSliderWidget() - Invalid column {name} with infinity")
                raise
    except:
        pass
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
            start = np.nanmin(df[name])
            end = np.nanmax(df[name])
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


def makeBokehSelectWidget(df: pd.DataFrame, params: list, paramDict: dict, default=None, **kwargs):
    options = {'size': 10}
    options.update(kwargs)
    # optionsPlot = []
    if len(params) == 1:
        if options['callback'] == 'parameter':
            optionsPlot = paramDict[params[0]]["options"]
        else:
            optionsPlot = np.sort(df[params[0]].dropna().unique()).tolist()
    else:
        optionsPlot = params[1:]
    for i, val in enumerate(optionsPlot):
        optionsPlot[i] = str((val))
    default_value = 0
    if isinstance(default, int):
        if 0 <= default < len(optionsPlot):
            default_value = default
        else:
            raise IndexError("Default value out of range for select widget.")
    elif default is None:
        if options['callback'] == 'parameter':
            default_value = optionsPlot.index(paramDict[params[0]]["value"])
    else:
        default_value = optionsPlot.index(paramDict[params[0]]["value"])
    return Select(title=params[0], value=optionsPlot[default_value], options=optionsPlot)


def makeBokehMultiSelectWidget(df: pd.DataFrame, params: list, paramDict: dict, **kwargs):
    # print("makeBokehMultiSelectWidget",params,kwargs)
    options = {'default': 0, 'size': 4}
    options.update(kwargs)
    # optionsPlot = []
    if len(params) == 1:
        try:
            optionsPlot = np.sort(df[params[0]].unique()).tolist()
        except:
            optionsPlot = sorted(df[params[0]].dropna().unique().tolist())
    else:
        optionsPlot = params[1:]
    for i, val in enumerate(optionsPlot):
        optionsPlot[i] = str((val))
    # print(optionsPlot)
    return MultiSelect(title=params[0], value=optionsPlot, options=optionsPlot, size=options['size'])


def makeBokehCheckboxWidget(df: pd.DataFrame, params: list, paramDict: dict, **kwargs):
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


def connectWidgetCallbacks(widgetParams: list, widgetArray: list, paramDict: dict, defaultCallback: CustomJS):
    for iDesc, iWidget in zip(widgetParams, widgetArray):
        optionLocal = {}
        params = iDesc[1]
        callback = None
        if len(iDesc) == 3:
            optionLocal = iDesc[2]
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
                    iWidget.js_link(*iEvent)
            continue
        if callback is not None:
            if isinstance(iWidget, CheckboxGroup):
                iWidget.js_on_click(callback)
            elif isinstance(iWidget, Slider) or isinstance(iWidget, RangeSlider):
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

def getOrMakeColumn(dfQuery, column, cdsName, ignoreDict={}):
    if '.' in column:
        c = column.split('.')
        if cdsName is None or cdsName == c[0]:
            return [dfQuery, c[1], c[0]]
        else:
            raise ValueError("Inconsistent CDS")
    else:
        if column in ignoreDict:
            return [dfQuery, column, cdsName]
        dfQuery, column = pandaGetOrMakeColumn(dfQuery, column)
        return [dfQuery, column, None]

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

def makeBokehDataSpec(thing: dict, paramDict: dict):
    if thing["type"] == "constant":
        return {"value": thing["value"]}
    return {"field": thing["name"]}

def errorBarWidthTwoSided(varError: dict, paramDict: dict, transform=None):
    if varError["type"] == "constant":
        return {"value": varError["value"]*2}
    if transform is None:
        transform = CustomJSTransform(v_func="return xs.map((x)=>2*x)")
    if varError["type"] == "parameter":
        return {"field": paramDict[varError["name"]]["value"], "transform": transform}
    return {"field": varError["name"], "transform": transform}

def getHistogramAxisTitle(cdsDict, varName, cdsName, removeCdsName=True):
    if cdsName is None:
        return varName
    if cdsName in cdsDict:
        if "variables" not in cdsDict[cdsName]:
            return varName
        if '_' in varName:
            if varName == "bin_count":
                return "entries"
            x = varName.split("_")
            if x[0] == "bin":
                if len(x) == 2:
                    return cdsDict[cdsName]["variables"][0]
                return cdsDict[cdsName]["variables"][int(x[-1])]
            if x[0] == "quantile":
                quantile = cdsDict[cdsName]["quantiles"][int(x[-1])]
                if '_' in cdsName:
                    histoName, projectionIdx = cdsName.split("_")
                    return "quantile " + str(quantile) + " " + cdsDict[histoName]["variables"][int(projectionIdx)]
                return "quantile " + str(quantile)
            if x[0] == "sum":
                range = cdsDict[cdsName]["sum_range"][int(x[-1])]
                if len(x) == 2:
                    if '_' in cdsName:
                        histoName, projectionIdx = cdsName.split("_")
                        return "sum " + cdsDict[histoName]["variables"][int(projectionIdx)] + " in [" + str(range[0]) + ", " + str(range[1]) + "]"
                    return "sum in [" + str(range[0]) + ", " + str(range[1]) + "]"
                else:
                    if '_' in cdsName:
                        histoName, projectionIdx = cdsName.split("_")
                        return "p " + cdsDict[histoName]["variables"][int(projectionIdx)] + " in [" + str(range[0]) + ", " + str(range[1]) + "]"
                    return "p in ["+ str(range[0]) + ", " + str(range[1]) + "]"
        else:
            if '_' in cdsName:
                histoName, projectionIdx = cdsName.split("_")
                return varName + " " + cdsDict[histoName]["variables"][int(projectionIdx)]
    if not removeCdsName:
        return cdsName+"."+varName
    return varName