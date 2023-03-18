# from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
# from RootInteractive.Tools.compressArray import arrayCompressionRelative16, arrayCompressionRelative8
# from bokeh.plotting import output_file
from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveParameters import *
def getDefaultVarsDiff():
    """
    TODO make diff parameterizable
    Function to get default RootInteractive variables for the simulated complex data.
    :return: aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc
    """
    # defining custom java script function to query  (used later in variable list)
    variables=["funCustom0","funCustom1","funCustom1"]

    aliasArray=[
        {"name": "funCustom0",  "func":"funCustomForm0",},
        {"name": "funCustom1",  "func":"funCustomForm1",},
        {"name": "funCustom2",  "func":"funcCustomForm2",},
    ]

    parameterArray = [
        # histo vars
        {"name": "nbinsX", "value":100, "range":[10, 200]},
        {"name": "nbinsY", "value":120, "range":[10, 200]},
        {"name": "nbinsZ", "value":5, "range":[1,10]},
        # transformation
        {"name": "exponentX", "value":1, "range":[-5, 5]},
        {'name': "xAxisTransform", "value":None, "options":[None, "sqrt", "lambda x: log(1+x)","lambda x: 1/sqrt(x)", "lambda x: x**exponentX","lambda x,y: x/y" ]},
        {'name': "yAxisTransform", "value":None, "options":[None, "sqrt", "lambda x: log(1+x)","lambda x: 1/sqrt(x)", "lambda x: x**exponentX","lambda x,y: y/x" ]},
        {'name': "zAxisTransform", "value":None, "options":[None, "sqrt", "lambda x: log(1+x)","lambda x: 1/sqrt(x)", "lambda x: x**exponentX" ]},
        # custom selection
        {'name': 'funCustomForm0', "value":"return 1"},
        {'name': 'funCustomForm1', "value":"return 1"},
        {'name': 'funCustomForm2', "value":"return 1"},
        #
        {"name": "sigmaNRel", "value":3.35, "range":[1,5]},
    ]

    parameterArray.extend(figureParameters["legend"]['parameterArray'])
    parameterArray.extend(figureParameters["markers"]['parameterArray'])

    widgetParams=[
        # custom selection
        ['textQuery', {"name": "customSelect0","value":"return 1"}],
        ['textQuery', {"name": "customSelect1","value":"return 1"}],
        ['textQuery', {"name": "customSelect2","value":"return 1"}],
        ['text', ['funCustomForm0'], {"name": "funCustomForm0"}],
        ['text', ['funCustomForm1'], {"name": "funCustomForm1"}],
        ['text', ['funCustomForm2'], {"name": "funCustomForm2"}],
        # histogram selection
        ['select', ['varX'], {"name": "varX"}],
        ['select', ['varY'], {"name": "varY"}],
        ['select', ['varYNorm'], {"name": "varYNorm"}],
        ['select', ['varZ'], {"name": "varZ"}],
        ['select', ['varZNorm'], {"name": "varZNorm"}],
        ['spinner', ['nbinsY'], {"name": "nbinsY"}],
        ['spinner', ['nbinsX'], {"name": "nbinsX"}],
        ['spinner', ['nbinsZ'], {"name": "nbinsZ"}],
        # transformation
        ['spinner', ['exponentX'],{"name": "exponentX"}],
        ['spinner', ['sigmaNRel'],{"name": "sigmaNRel"}],
        ['select', ['yAxisTransform'], {"name": "yAxisTransform"}],
        ['select', ['xAxisTransform'], {"name": "xAxisTransform"}],
        ['select', ['zAxisTransform'], {"name": "zAxisTransform"}],
    ]

    widgetParams.extend(figureParameters["legend"]["widgets"])
    widgetParams.extend(figureParameters["markers"]["widgets"])

    widgetLayoutDesc={
        "Select": [],
        "Custom":[["customSelect0","customSelect1","customSelect2"],["funCustomForm0","funCustomForm1","funCustomForm2"]],
        "Histograms":[["nbinsX","nbinsY", "nbinsZ", "varX","varY","varYNorm","varZ","varZNorm"], {'sizing_mode': 'scale_width'}],
        "Transform":[["exponentX","xAxisTransform", "yAxisTransform","zAxisTransform"],{'sizing_mode': 'scale_width'}],
        "Legend": figureParameters['legend']['widgetLayout'],
        "Markers":["markerSize"]
    }

    figureGlobalOption={}
    figureGlobalOption=figureParameters["legend"]["figureOptions"]
    figureGlobalOption["size"]="markerSize"
    figureGlobalOption["x_transform"]="xAxisTransform"
    figureGlobalOption["y_transform"]="yAxisTransform"
    figureGlobalOption["z_transform"]="zAxisTransform"

    histoArray=[
        {
            "name": "histoXYData",
            "variables": ["varX","varY"],
            "nbins":["nbinsX","nbinsY"], "axis":[1],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYNormData",
            "variables": ["varX","varY-varYNorm"],
            "nbins":["nbinsX","nbinsY"], "axis":[1],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYZData",
            "variables": ["varX","varY","varZ"],
            "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYNormZData",
            "variables": ["varX","varY-varYNorm","varZ"],
            "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYZNormData",
            "variables": ["varX","varY","varZ-varZNorm"],
            "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
    ]

    figureArray=[
        # histo XY
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "bin_count", "source":"histoXYData"}],
        [["bin_center_1"], ["bin_count"], { "source":"histoXYData", "colorZvar": "bin_center_0"}],
        [["bin_center_0"], ["mean","quantile_1",], { "source":"histoXYData_1","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYData_1","errY":"std/sqrt(entries)"}],
        # histoXYNorm
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "bin_count", "source":"histoXYNormData"}],
        [["bin_center_1"], ["bin_count"], { "source":"histoXYNormData", "colorZvar": "bin_center_0"}],
        [["bin_center_0"], ["mean","quantile_1",], { "source":"histoXYNormData_1","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYNormData_1","errY":"std/sqrt(entries)"}],
        # histoXYZ
        [["bin_center_0"], ["mean"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["bin_count"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"2*std/sqrt(entries)"}],
        [["bin_center_0"], ["quantile_1"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"3*std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)"}],
        # histoXYNormZ
        [["bin_center_0"], ["mean"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)","yAxisTitle":"{varY}-{varYNorm}"}],
        [["bin_center_0"], ["bin_count"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"2*std/sqrt(entries)","yAxisTitle":"{varY}-{varYNorm}"}],
        [["bin_center_0"], ["quantile_1"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"3*std/sqrt(entries)","yAxisTitle":"{varY}-{varYNorm}"}],
        [["bin_center_0"], ["std"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)","yAxisTitle":"{varY}-{varYNorm}"}],
        # histoXYNormZMedian
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "quantile_1", "source":"histoXYZData_2"}],
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "quantile_1", "source":"histoXYZNormData_2"}],
        # histoXYNormZMean
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "mean", "source":"histoXYZData_2"}],
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "mean", "source":"histoXYZNormData_2"}],
        #
        ['descriptionTable', {"name":"description"}],
        ['selectionTable', {"name":"selection"}],
        figureGlobalOption
    ]
    figureLayoutDesc={
        "histoXY":[[0,1],[2,3],{"plot_height":200}],
        "histoXYNorm":[[4,5],[6,7],{"plot_height":200}],
        "histoXYZ":[[8,9],[10,11],{"plot_height":200}],
        "histoXYNormZ":[[12,13],[14,15],{"plot_height":200}],
        "histoXYNormZMedian":[[16,17],{"plot_height":350}],
        "histoXYNormZMean":[[18,19],{"plot_height":350}],
    }
    figureLayoutDesc["selection"] = ["selection", {'plot_height': 100, 'sizing_mode': 'scale_width'}]
    figureLayoutDesc["description"] = ["description", {'plot_height': 100, 'sizing_mode': 'scale_width'}]

    print("Default RootInteractive variables are defined.")
    return aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc

def getDefaultVarsRatio():
    """
    TODO make diff parameterizable
    Function to get default RootInteractive variables for the simulated complex data.
    :return: aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc
    """
    # defining custom java script function to query  (used later in variable list)
    variables=["funCustom0","funCustom1","funCustom1"]

    aliasArray=[
        {"name": "funCustom0",  "func":"funCustomForm0",},
        {"name": "funCustom1",  "func":"funCustomForm1",},
        {"name": "funCustom2",  "func":"funcCustomForm2",},
    ]

    parameterArray = [
        # histo vars
        {"name": "nbinsX", "value":100, "range":[10, 200]},
        {"name": "nbinsY", "value":120, "range":[10, 200]},
        {"name": "nbinsZ", "value":5, "range":[1,10]},
        # transformation
        {"name": "exponentX", "value":1, "range":[-5, 5]},
        {'name': "xAxisTransform", "value":None, "options":[None, "sqrt", "lambda x: log(1+x)","lambda x: 1/sqrt(x)", "lambda x: x**exponentX","lambda x,y: x/y" ]},
        {'name': "yAxisTransform", "value":None, "options":[None, "sqrt", "lambda x: log(1+x)","lambda x: 1/sqrt(x)", "lambda x: x**exponentX","lambda x,y: y/x" ]},
        {'name': "zAxisTransform", "value":None, "options":[None, "sqrt", "lambda x: log(1+x)","lambda x: 1/sqrt(x)", "lambda x: x**exponentX" ]},
        # custom selection
        {'name': 'funCustomForm0', "value":"return 1"},
        {'name': 'funCustomForm1', "value":"return 1"},
        {'name': 'funCustomForm2', "value":"return 1"},
        #
        {"name": "sigmaNRel", "value":3.35, "range":[1,5]},
    ]

    parameterArray.extend(figureParameters["legend"]['parameterArray'])
    parameterArray.extend(figureParameters["markers"]['parameterArray'])

    widgetParams=[
        # custom selection
        ['textQuery', {"name": "customSelect0","value":"return 1"}],
        ['textQuery', {"name": "customSelect1","value":"return 1"}],
        ['textQuery', {"name": "customSelect2","value":"return 1"}],
        ['text', ['funCustomForm0'], {"name": "funCustomForm0"}],
        ['text', ['funCustomForm1'], {"name": "funCustomForm1"}],
        ['text', ['funCustomForm2'], {"name": "funCustomForm2"}],
        # histogram selection
        ['select', ['varX'], {"name": "varX"}],
        ['select', ['varY'], {"name": "varY"}],
        ['select', ['varYNorm'], {"name": "varYNorm"}],
        ['select', ['varZ'], {"name": "varZ"}],
        ['select', ['varZNorm'], {"name": "varZNorm"}],
        ['spinner', ['nbinsY'], {"name": "nbinsY"}],
        ['spinner', ['nbinsX'], {"name": "nbinsX"}],
        ['spinner', ['nbinsZ'], {"name": "nbinsZ"}],
        # transformation
        ['spinner', ['exponentX'],{"name": "exponentX"}],
        ['spinner', ['sigmaNRel'],{"name": "sigmaNRel"}],
        ['select', ['yAxisTransform'], {"name": "yAxisTransform"}],
        ['select', ['xAxisTransform'], {"name": "xAxisTransform"}],
        ['select', ['zAxisTransform'], {"name": "zAxisTransform"}],
    ]

    widgetParams.extend(figureParameters["legend"]["widgets"])
    widgetParams.extend(figureParameters["markers"]["widgets"])

    widgetLayoutDesc={
        "Select": [],
        "Custom":[["customSelect0","customSelect1","customSelect2"],["funCustomForm0","funCustomForm1","funCustomForm2"]],
        "Histograms":[["nbinsX","nbinsY", "nbinsZ", "varX","varY","varYNorm","varZ","varZNorm"], {'sizing_mode': 'scale_width'}],
        "Transform":[["exponentX","xAxisTransform", "yAxisTransform","zAxisTransform"],{'sizing_mode': 'scale_width'}],
        "Legend": figureParameters['legend']['widgetLayout'],
        "Markers":["markerSize"]
    }

    figureGlobalOption={}
    figureGlobalOption=figureParameters["legend"]["figureOptions"]
    figureGlobalOption["size"]="markerSize"
    figureGlobalOption["x_transform"]="xAxisTransform"
    figureGlobalOption["y_transform"]="yAxisTransform"
    figureGlobalOption["z_transform"]="zAxisTransform"

    histoArray=[
        {
            "name": "histoXYData",
            "variables": ["varX","varY"],
            "nbins":["nbinsX","nbinsY"], "axis":[1],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYNormData",
            "variables": ["varX","varY-varYNorm"],
            "nbins":["nbinsX","nbinsY"], "axis":[1],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYZData",
            "variables": ["varX","varY","varZ"],
            "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYNormZData",
            "variables": ["varX","varY-varYNorm","varZ"],
            "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYZNormData",
            "variables": ["varX","varY","varZ-varZNorm"],
            "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
    ]

    figureArray=[
        # histo XY
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "bin_count", "source":"histoXYData"}],
        [["bin_center_1"], ["bin_count"], { "source":"histoXYData", "colorZvar": "bin_center_0"}],
        [["bin_center_0"], ["mean","quantile_1",], { "source":"histoXYData_1","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYData_1","errY":"std/sqrt(entries)"}],
        # histoXYNorm
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "bin_count", "source":"histoXYNormData"}],
        [["bin_center_1"], ["bin_count"], { "source":"histoXYNormData", "colorZvar": "bin_center_0"}],
        [["bin_center_0"], ["mean","quantile_1",], { "source":"histoXYNormData_1","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYNormData_1","errY":"std/sqrt(entries)"}],
        # histoXYZ
        [["bin_center_0"], ["mean"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["bin_count"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"2*std/sqrt(entries)"}],
        [["bin_center_0"], ["quantile_1"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"3*std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)"}],
        # histoXYNormZ
        [["bin_center_0"], ["mean"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)","yAxisTitle":"{varY}-{varYNorm}"}],
        [["bin_center_0"], ["bin_count"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"2*std/sqrt(entries)","yAxisTitle":"{varY}-{varYNorm}"}],
        [["bin_center_0"], ["quantile_1"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"3*std/sqrt(entries)","yAxisTitle":"{varY}-{varYNorm}"}],
        [["bin_center_0"], ["std"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)","yAxisTitle":"{varY}-{varYNorm}"}],
        # histoXYNormZMedian
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "quantile_1", "source":"histoXYZData_2"}],
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "quantile_1", "source":"histoXYZNormData_2"}],
        # histoXYNormZMean
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "mean", "source":"histoXYZData_2"}],
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "mean", "source":"histoXYZNormData_2"}],
        #
        ['descriptionTable', {"name":"description"}],
        ['selectionTable', {"name":"selection"}],
        figureGlobalOption
    ]
    figureLayoutDesc={
        "histoXY":[[0,1],[2,3],{"plot_height":200}],
        "histoXYNorm":[[4,5],[6,7],{"plot_height":200}],
        "histoXYZ":[[8,9],[10,11],{"plot_height":200}],
        "histoXYNormZ":[[12,13],[14,15],{"plot_height":200}],
        "histoXYNormZMedian":[[16,17],{"plot_height":350}],
        "histoXYNormZMean":[[18,19],{"plot_height":350}],
    }

    figureLayoutDesc["selection"] = ["selection", {'plot_height': 100, 'sizing_mode': 'scale_width'}]
    figureLayoutDesc["description"] = ["description", {'plot_height': 100, 'sizing_mode': 'scale_width'}]

    print("Default RootInteractive variables are defined.")
    return aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc

