# from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
# from RootInteractive.Tools.compressArray import arrayCompressionRelative16, arrayCompressionRelative8
# from bokeh.plotting import output_file
from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveParameters import figureParameters 

def getDefaultVars(normalization=None, variables=None, defaultVariables={}, weights=None, multiAxis=None):
    """
    Function to get default RootInteractive variables for the simulated complex data.
    :return: aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc
    """
    # defining custom java script function to query  (used later in variable list)
    if variables is None:
        variables = []
    variables.extend(["funCustom0","funCustom1","funCustom2"])
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
        {"name": "epsilonLog", "value": 0.1, "range": [0.00, 100000]},
        {'name': "xAxisTransform", "value":None, "options":[None, "sqrt","log", "lambda x: log(1+x)","lambda x: log(1+epsilonLog*x)/epsilonLog","lambda x: 1/sqrt(x)", "lambda x: x**exponentX","lambda x,y: x/y" ]},
        {'name': "yAxisTransform", "value":None, "options":[None, "sqrt", "log", "lambda x: log(1+x)","lambda x: log(1+epsilonLog*x)/epsilonLog","lambda x: 1/sqrt(x)", "lambda x: x**exponentX","lambda x,y: y/x" ]},
        {'name': "zAxisTransform", "value":None, "options":[None, "sqrt", "log","lambda x: log(1+x)","lambda x: log(1+epsilonLog*x)/epsilonLog","lambda x: 1/sqrt(x)", "lambda x: x**exponentX" ]},
        # custom selection
        {'name': 'funCustomForm0', "value":"return 1"},
        {'name': 'funCustomForm1', "value":"return 1"},
        {'name': 'funCustomForm2', "value":"return 1"},
        #
        {"name": "sigmaNRel", "value":3.35, "range":[1,5]},
    ]

    parameterArray.extend(figureParameters["legend"]['parameterArray'])
    parameterArray.extend(figureParameters["markers"]['parameterArray'])
    parameterVars = ["varX", "varY", "varZ"] if normalization is None else ["varX", "varY", "varYNorm", "varZ", "varZNorm"]
    if isinstance(weights, list):
        parameterVars.append("weights")

    widgetParams=[
        # custom selection
        ['textQuery', {"name": "customSelect0","value":"return 1"}],
        ['textQuery', {"name": "customSelect1","value":"return 1"}],
        ['textQuery', {"name": "customSelect2","value":"return 1"}],
        ['text', ['funCustomForm0'], {"name": "funCustomForm0"}],
        ['text', ['funCustomForm1'], {"name": "funCustomForm1"}],
        ['text', ['funCustomForm2'], {"name": "funCustomForm2"}],
        # histogram bins 
        ['spinner', ['nbinsY'], {"name": "nbinsY"}],
        ['spinner', ['nbinsX'], {"name": "nbinsX"}],
        ['spinner', ['nbinsZ'], {"name": "nbinsZ"}],
        # transformation
        ['spinner', ['exponentX'],{"name": "exponentX"}],
        ['spinner', ['epsilonLog'], {"name": "epsilonLog"}],
        ['spinner', ['sigmaNRel'],{"name": "sigmaNRel"}],
        ['select', ['yAxisTransform'], {"name": "yAxisTransform"}],
        ['select', ['xAxisTransform'], {"name": "xAxisTransform"}],
        ['select', ['zAxisTransform'], {"name": "zAxisTransform"}],
    ]

    # histogram selection
    for i, iVar in enumerate(parameterVars):
        defaultValue = defaultVariables.get(iVar, variables[i % len(variables)])
        if iVar == "weights":
            defaultValue = defaultVariables.get("weights", weights[0])
        if isinstance(defaultValue, list):
            if multiAxis is None:
                multiAxis = iVar
            else:
                raise NotImplementedError("Multiple multiselect axes not implemented")
        if iVar == multiAxis and not isinstance(defaultValue, list):
            defaultValue = [defaultValue]
        parameterArray.append({"name": iVar, "value": defaultValue, "options":weights if iVar == "weights" else variables}) 
        widgetParams.append(['multiSelect' if iVar == multiAxis else 'select', [iVar], {"name":iVar}])

    defaultWeights=None
    if isinstance(weights, list):
        defaultWeights = defaultVariables.get("weights", weights[0])

    widgetParams.extend(figureParameters["legend"]["widgets"])
    widgetParams.extend(figureParameters["markers"]["widgets"])

    widgetLayoutDesc={
        "Select": [],
        "Custom":[["customSelect0","customSelect1","customSelect2"],["funCustomForm0","funCustomForm1","funCustomForm2"]],
        "Histograms":[["nbinsX","nbinsY", "nbinsZ"], parameterVars, {'sizing_mode': 'scale_width'}],
        "Transform":[["exponentX","epsilonLog","xAxisTransform", "yAxisTransform","zAxisTransform"],{'sizing_mode': 'scale_width'}],
        "Legend": figureParameters['legend']['widgetLayout'],
        "Markers":["markerSize"]
    }

    figureGlobalOption={}
    figureGlobalOption=figureParameters["legend"]["figureOptions"]
    figureGlobalOption["size"]="markerSize"
    figureGlobalOption["x_transform"]="xAxisTransform"
    figureGlobalOption["y_transform"]="yAxisTransform"
    figureGlobalOption["z_transform"]="zAxisTransform"

    formulaYNorm = {
            "diff":"varY-varYNorm",
            "ratio":"varY/varYNorm",
            "logRatio":"log(varY)/log(varYNorm)"
            }[normalization]
    
    formulaZNorm = {
            "diff":"varZ-varZNorm",
            "ratio":"varZ/varZNorm",
            "logRatio":"log(varZ)/log(varZNorm)"
            }[normalization]

    histoArray=[
        {
            "name": "histoXYData",
            "variables": ["varX","varY"],
            "weights": "weights" if weights else defaultWeights,
            "nbins":["nbinsX","nbinsY"], "quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYZData",
            "variables": ["varX","varY","varZ"],
            "weights": "weights" if weights else defaultWeights,
            "nbins":["nbinsX","nbinsY","nbinsZ"], "quantiles": [0.35,0.5],"unbinned_projections":True,
        },
    ]

    if normalization is not None:
        if multiAxis in ["varY", "varYNorm"]:
            histoXYNames = {i:f"histoXYNormData[{i}]" for i in variables}
            histoXY1Names = {i:f"histoXYNormData_1[{i}]" for i in variables}
            histoXYZNames = {i:f"histoXYNormZData[{i}]" for i in variables}
            histoXYZ1Names = {i:f"histoXYNormZData_1[{i}]" for i in variables}
            histoXYZ2Names = {i:f"histoXYNormZData_1[{i}]" for i in variables}
            varNorm = {
                "diff": lambda x,y: f"{x}-{y}",
                "ratio": lambda x,y: f"{x}/{y}",
                "logRatio":lambda x,y: f"log({x})/log({y})"
            }[normalization]
            histoArray.extend([{
                "name": f"histoXYNormData[{i}]",
                "variables": ["varX", varNorm(i, "varYNorm") if multiAxis == "varY" else varNorm("varY", i)],
                "weights": "weights" if weights else defaultWeights,   
                "nbins": ["nbinsX", "nbinsY"]        
            } for i in variables])
            histoArray.extend([{
                "name": f"histoXYNormData_1[{i}]",
                "type": "projection",
                "source": f"histoXYNormData[{i}]",
                "axis_idx": 1,
                "weights": "weights" if weights else defaultWeights,
                "quantiles": [0.35,0.5],"unbinned":True,                
            } for i in variables])
            histoArray.extend([{
                "name": f"histoXYNormZData[{i}]",
                "variables": ["varX", varNorm(i, "varYNorm") if multiAxis == "varY" else varNorm("varY", i), "varZ"],
                "weights": "weights" if weights else defaultWeights,
                "nbins":["nbinsX","nbinsY", "nbinsZ"],                
            } for i in variables])
            histoArray.extend([{
                "name": f"histoXYNormZData_1[{i}]",
                "type": "projection",
                "source": f"histoXYNormZData[{i}]",
                "axis_idx": 1,
                "weights": "weights" if weights else defaultWeights,
                "quantiles": [0.35,0.5],"unbinned":True,                
            } for i in variables])
            histoArray.extend([{
                "name": f"histoXYNormZData_2[{i}]",
                "type": "projection",
                "source": f"histoXYNormZData[{i}]",
                "axis_idx": 2,
                "weights": "weights" if weights else defaultWeights,
                "quantiles": [0.35,0.5],"unbinned":True,                
            } for i in variables])
            histoArray.extend([
            {
                "name": "histoXYNormData",
                "sources": multiAxis,
                "mapping": histoXYNames
            },
            {
                "name": "histoXYNormData_1",
                "sources": multiAxis,
                "mapping": histoXY1Names
            },
            {
                "name": "histoXYNormZData",
                "sources": multiAxis,
                "mapping": histoXYZNames
            },
            {
                "name": "histoXYNormZData_1",
                "sources": multiAxis,
                "mapping": histoXYZ1Names
            },
            {
                "name": "histoXYNormZData_2",
                "sources": multiAxis,
                "mapping": histoXYZ2Names
            }
            ])
        else:
            histoArray.extend([
            {
                "name": "histoXYNormData",
                "variables": ["varX",formulaYNorm],
                "weights": "weights" if weights else defaultWeights,
                "nbins":["nbinsX","nbinsY"], "axis":[1],"quantiles": [0.35,0.5],"unbinned_projections":True,
            },
            {
                "name": "histoXYNormZData",
                "variables": ["varX",formulaYNorm,"varZ"],
                "weights": "weights" if weights else defaultWeights,
                "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
            },
            ])
        if multiAxis in ["varZ", "varZNorm"]:
            pass
        else:
            histoArray.append(            
                {
                    "name": "histoXYZNormData",
                    "variables": ["varX","varY",formulaZNorm],
                    "weights": "weights" if weights else defaultWeights,
                    "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
                })

    yAxisTitleNorm = {
            "diff":"{varY}-{varYNorm}",
            "ratio":"{varY}/{varYNorm}",
            "logRatio":"log({varY})/log({varYNorm})"
            }[normalization]

    figureArray=[
        # histo XY
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "bin_count", "source":"histoXYData"}],
        [["bin_center_1"], ["bin_count"], { "source":"histoXYData", "colorZvar": "bin_center_0"}],
        [["bin_center_0"], ["mean","quantile_1",], { "source":"histoXYData_1","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYData_1","errY":"std/sqrt(entries)"}],
        # histoXYNorm
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "bin_count", "source":"histoXYNormData","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_1"], ["bin_count"], { "source":"histoXYNormData", "colorZvar": "bin_center_0","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_0"], ["mean","quantile_1",], { "source":"histoXYNormData_1","errY":"std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_0"], ["std"], { "source":"histoXYNormData_1","errY":"std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        # histoXYZ
        [["bin_center_0"], ["mean"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["entries"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"2*std/sqrt(entries)"}],
        [["bin_center_0"], ["quantile_1"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"3*std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)"}],
        # histoXYNormZ
        [["bin_center_0"], ["mean"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_0"], ["entries"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"2*std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_0"], ["quantile_1"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"3*std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_0"], ["std"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        # histoXYZNormMedian
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "quantile_1", "source":"histoXYZData_2"}],
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "quantile_1", "source":"histoXYZNormData_2"}],
        # histoXYZNormMean
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "mean", "source":"histoXYZData_2"}],
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "mean", "source":"histoXYZNormData_2"}],
        #
        ['descriptionTable', {"name":"description"}],
        ['selectionTable', {"name":"selection"}],
        figureGlobalOption
    ]
    figureLayoutDesc={
        "histoXY":[[0,1],[2,3],{"plot_height":250}],
        "histoXYNorm":[[4,5],[6,7],{"plot_height":250}],
        "histoXYZ":[[8,9],[10,11],{"plot_height":250}],
        "histoXYNormZ":[[12,13],[14,15],{"plot_height":250}],
        "histoXYNormZMedian":[[16,17],{"plot_height":350}],
        "histoXYNormZMean":[[18,19],{"plot_height":350}],
    }
    figureLayoutDesc["selection"] = ["selection", {'plot_height': 200, 'sizing_mode': 'scale_width'}]
    figureLayoutDesc["description"] = ["description", {'plot_height': 200, 'sizing_mode': 'scale_width'}]

    print("Default RootInteractive variables are defined.")
    return aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc

def getDefaultVarsNormAll(variables=None, defaultVariables={}, weights=None, multiAxis=None):
    """
    Function to get default RootInteractive variables for the simulated complex data.
    :return: aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc
    """
    # defining custom javascript function to query  (used later in variable list)
    if variables is None:
        variables = []
    variablesCopy = variables.copy()
    aliasArray=[
        {"name": "funCustom0",  "func":"funCustomForm0", "variables":variablesCopy},
        {"name": "funCustom1",  "func":"funCustomForm1", "variables":variablesCopy},
        {"name": "funCustom2",  "func":"funCustomForm2", "variables":variablesCopy},
    ]
    variables.extend(["funCustom0","funCustom1","funCustom2"])

    parameterArray = [
        # histo vars
        {"name": "nbinsX", "value":100, "range":[10, 200]},
        {"name": "nbinsY", "value":120, "range":[10, 200]},
        {"name": "nbinsZ", "value":5, "range":[1,10]},
        {"name": "diffFuncTransform", "value":"diff", "options":["diff", "ratio", "logRatio"]},
        # transformation
        {"name": "exponentX", "value":1, "range":[-5, 5]},
        {"name": "epsilonLog", "value": 0.1, "range": [0.00, 100000]},
        {'name': "xAxisTransform", "value":None, "options":[None, "sqrt","log", "lambda x: log(1+x)","lambda x: log(1+epsilonLog*x)/epsilonLog","lambda x: 1/sqrt(x)", "lambda x: x**exponentX","lambda x,y: x/y" ]},
        {'name': "yAxisTransform", "value":None, "options":[None, "sqrt", "log", "lambda x: log(1+x)","lambda x: log(1+epsilonLog*x)/epsilonLog","lambda x: 1/sqrt(x)", "lambda x: x**exponentX","lambda x,y: y/x" ]},
        {'name': "zAxisTransform", "value":None, "options":[None, "sqrt", "log","lambda x: log(1+x)","lambda x: log(1+epsilonLog*x)/epsilonLog","lambda x: 1/sqrt(x)", "lambda x: x**exponentX" ]},
        # custom selection
        {'name': 'funCustomForm0', "value":"return 1"},
        {'name': 'funCustomForm1', "value":"return 1"},
        {'name': 'funCustomForm2', "value":"return 1"},
        #
        {"name": "sigmaNRel", "value":3.35, "range":[1,5]},
    ]

    transformArray = [
            {"name":"diffFunc",
              "v_func":"""
                if($output == null || $output.length !== varY.length){
                    $output = new Float64Array(varY.length)
                }
                if(diffFuncTransform=='diff'){
                    for(let i=0; i<$output.length; i++){
                        $output[i] = varY[i]-varYNorm[i]
                    }
                }
                else if(diffFuncTransform=='ratio'){
                    for(let i=0; i<$output.length; i++){
                        $output[i] = varY[i]/varYNorm[i]
                    }                
                }
                else if(diffFuncTransform=='logRatio'){
                    for(let i=0; i<$output.length; i++){
                        $output[i] = Math.log(varY[i])/Math.log(varYNorm[i])
                    }                
                }
                return $output
              """, "parameters":["diffFuncTransform"], "fields":["varY", "varYNorm"]}
    ]

    parameterArray.extend(figureParameters["legend"]['parameterArray'])
    parameterArray.extend(figureParameters["markers"]['parameterArray'])
    parameterVars = ["varX", "varY", "varYNorm", "varZ", "varZNorm"]
    if isinstance(weights, list):
        parameterVars.append("weights")

    widgetParams=[
        # custom selection
        ['textQuery', {"name": "customSelect0","value":"return 1"}],
        ['textQuery', {"name": "customSelect1","value":"return 1"}],
        ['textQuery', {"name": "customSelect2","value":"return 1"}],
        ['text', ['funCustomForm0'], {"name": "funCustomForm0"}],
        ['text', ['funCustomForm1'], {"name": "funCustomForm1"}],
        ['text', ['funCustomForm2'], {"name": "funCustomForm2"}],
        # histogram bins 
        ['spinner', ['nbinsY'], {"name": "nbinsY"}],
        ['spinner', ['nbinsX'], {"name": "nbinsX"}],
        ['spinner', ['nbinsZ'], {"name": "nbinsZ"}],
        ['select', ['diffFuncTransform'], {"name": "diffFunc"}],
        # transformation
        ['spinner', ['exponentX'],{"name": "exponentX"}],
        ['spinner', ['epsilonLog'], {"name": "epsilonLog"}],
        ['spinner', ['sigmaNRel'],{"name": "sigmaNRel"}],
        ['select', ['yAxisTransform'], {"name": "yAxisTransform"}],
        ['select', ['xAxisTransform'], {"name": "xAxisTransform"}],
        ['select', ['zAxisTransform'], {"name": "zAxisTransform"}],
    ]

    # histogram selection
    for i, iVar in enumerate(parameterVars):
        defaultValue = defaultVariables.get(iVar, variables[i % len(variables)])
        if iVar == "weights":
            defaultValue = defaultVariables.get("weights", weights[0])
        if isinstance(defaultValue, list):
            if multiAxis is None:
                multiAxis = iVar
            else:
                raise NotImplementedError("Multiple multiselect axes not implemented")
        if iVar == multiAxis and not isinstance(defaultValue, list):
            defaultValue = [defaultValue]
        parameterArray.append({"name": iVar, "value": defaultValue, "options":weights if iVar == "weights" else variables}) 
        widgetParams.append(['multiSelect' if iVar == multiAxis else 'select', [iVar], {"name":iVar}])

    defaultWeights=None
    if isinstance(weights, list):
        defaultWeights = defaultVariables.get("weights", weights[0])

    widgetParams.extend(figureParameters["legend"]["widgets"])
    widgetParams.extend(figureParameters["markers"]["widgets"])

    widgetLayoutDesc={
        "Select": [],
        "Custom":[["customSelect0","customSelect1","customSelect2"],["funCustomForm0","funCustomForm1","funCustomForm2"]],
        "Histograms":[["nbinsX","nbinsY", "nbinsZ", "diffFunc"], parameterVars, {'sizing_mode': 'scale_width'}],
        "Transform":[["exponentX","epsilonLog","xAxisTransform", "yAxisTransform","zAxisTransform"],{'sizing_mode': 'scale_width'}],
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
            "weights": "weights" if weights else defaultWeights,
            "nbins":["nbinsX","nbinsY"], "quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYZData",
            "variables": ["varX","varY","varZ"],
            "weights": "weights" if weights else defaultWeights,
            "nbins":["nbinsX","nbinsY","nbinsZ"], "quantiles": [0.35,0.5],"unbinned_projections":True,
        },
    ]

    if multiAxis in ["varY", "varYNorm"]:
            histoXYNames = {i:f"histoXYNormData[{i}]" for i in variables}
            histoXY1Names = {i:f"histoXYNormData_1[{i}]" for i in variables}
            histoXYZNames = {i:f"histoXYNormZData[{i}]" for i in variables}
            histoXYZ1Names = {i:f"histoXYNormZData_1[{i}]" for i in variables}
            histoXYZ2Names = {i:f"histoXYNormZData_1[{i}]" for i in variables}
            varNorm = lambda x: f"diff_{x}"
            aliasArray.extend([{
                    "name": varNorm(i),
                    "transform": "diffFunc",
                    "variables": [iVar, "varYNorm"] if multiAxis == "varY" else ["varY", iVar]
                } for i, iVar in enumerate(variables)])
            histoArray.extend([{
                "name": f"histoXYNormData[{iVar}]",
                "variables": ["varX", varNorm(i)],
                "weights": "weights" if weights else defaultWeights,   
                "nbins": ["nbinsX", "nbinsY"]        
            } for i, iVar in enumerate(variables)])
            histoArray.extend([{
                "name": f"histoXYNormData_1[{i}]",
                "type": "projection",
                "source": f"histoXYNormData[{i}]",
                "axis_idx": 1,
                "weights": "weights" if weights else defaultWeights,
                "quantiles": [0.35,0.5],"unbinned":True,                
            } for i in variables])
            histoArray.extend([{
                "name": f"histoXYNormZData[{iVar}]",
                "variables": ["varX", varNorm(i), "varZ"],
                "weights": "weights" if weights else defaultWeights,
                "nbins":["nbinsX","nbinsY", "nbinsZ"],                
            } for i, iVar in enumerate(variables)])
            histoArray.extend([{
                "name": f"histoXYNormZData_1[{i}]",
                "type": "projection",
                "source": f"histoXYNormZData[{i}]",
                "axis_idx": 1,
                "weights": "weights" if weights else defaultWeights,
                "quantiles": [0.35,0.5],"unbinned":True,                
            } for i in variables])
            histoArray.extend([{
                "name": f"histoXYNormZData_2[{i}]",
                "type": "projection",
                "source": f"histoXYNormZData[{i}]",
                "axis_idx": 2,
                "weights": "weights" if weights else defaultWeights,
                "quantiles": [0.35,0.5],"unbinned":True,                
            } for i in variables])
            histoArray.extend([
            {
                "name": "histoXYNormData",
                "sources": multiAxis,
                "mapping": histoXYNames
            },
            {
                "name": "histoXYNormData_1",
                "sources": multiAxis,
                "mapping": histoXY1Names
            },
            {
                "name": "histoXYNormZData",
                "sources": multiAxis,
                "mapping": histoXYZNames
            },
            {
                "name": "histoXYNormZData_1",
                "sources": multiAxis,
                "mapping": histoXYZ1Names
            },
            {
                "name": "histoXYNormZData_2",
                "sources": multiAxis,
                "mapping": histoXYZ2Names
            }
            ])
    else:
        aliasArray.append({
            "name":"diffY",
            "transform":"diffFunc",
            "variables":["varY","varYNorm"]
            })
        histoArray.extend([
        {
            "name": "histoXYNormData",
            "variables": ["varX","diffY"],
            "weights": "weights" if weights else defaultWeights,
            "nbins":["nbinsX","nbinsY"], "axis":[1],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        {
            "name": "histoXYNormZData",
            "variables": ["varX","diffY","varZ"],
            "weights": "weights" if weights else defaultWeights,
            "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
        },
        ])
    if multiAxis in ["varZ", "varZNorm"]:
            pass
    else:
            aliasArray.append({
                "name":"diffZ",
                "transform":"diffFunc",
                "variables":["varZ", "varZNorm"]
                })
            histoArray.append(            
                {
                    "name": "histoXYZNormData",
                    "variables": ["varX","varY","diffZ"],
                    "weights": "weights" if weights else defaultWeights,
                    "nbins":["nbinsX","nbinsY","nbinsZ"], "axis":[1,2],"quantiles": [0.35,0.5],"unbinned_projections":True,
                })

    yAxisTitleNorm = "{varY}-{varYNorm} ({diffFuncTransform})"

    figureArray=[
        # histo XY
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "bin_count", "source":"histoXYData"}],
        [["bin_center_1"], ["bin_count"], { "source":"histoXYData", "colorZvar": "bin_center_0"}],
        [["bin_center_0"], ["mean","quantile_1",], { "source":"histoXYData_1","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYData_1","errY":"std/sqrt(entries)"}],
        # histoXYNorm
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "bin_count", "source":"histoXYNormData","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_1"], ["bin_count"], { "source":"histoXYNormData", "colorZvar": "bin_center_0","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_0"], ["mean","quantile_1",], { "source":"histoXYNormData_1","errY":"std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_0"], ["std"], { "source":"histoXYNormData_1","errY":"std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        # histoXYZ
        [["bin_center_0"], ["mean"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)"}],
        [["bin_center_0"], ["entries"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"2*std/sqrt(entries)"}],
        [["bin_center_0"], ["quantile_1"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"3*std/sqrt(entries)"}],
        [["bin_center_0"], ["std"], { "source":"histoXYZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)"}],
        # histoXYNormZ
        [["bin_center_0"], ["mean"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_0"], ["entries"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"2*std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_0"], ["quantile_1"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"3*std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        [["bin_center_0"], ["std"], { "source":"histoXYNormZData_1","colorZvar":"bin_center_2","errY":"std/sqrt(entries)","yAxisTitle":yAxisTitleNorm}],
        # histoXYZNormMedian
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "quantile_1", "source":"histoXYZData_2"}],
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "quantile_1", "source":"histoXYZNormData_2"}],
        # histoXYZNormMean
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "mean", "source":"histoXYZData_2"}],
        [[("bin_bottom_0", "bin_top_0")], [("bin_bottom_1", "bin_top_1")], {"colorZvar": "mean", "source":"histoXYZNormData_2"}],
        #
        ['descriptionTable', {"name":"description"}],
        ['selectionTable', {"name":"selection"}],
        figureGlobalOption
    ]
    figureLayoutDesc={
        "histoXY":[[0,1],[2,3],{"plot_height":250}],
        "histoXYNorm":[[4,5],[6,7],{"plot_height":250}],
        "histoXYZ":[[8,9],[10,11],{"plot_height":250}],
        "histoXYNormZ":[[12,13],[14,15],{"plot_height":250}],
        "histoXYNormZMedian":[[16,17],{"plot_height":350}],
        "histoXYNormZMean":[[18,19],{"plot_height":350}],
    }
    figureLayoutDesc["selection"] = ["selection", {'plot_height': 200, 'sizing_mode': 'scale_width'}]
    figureLayoutDesc["description"] = ["description", {'plot_height': 200, 'sizing_mode': 'scale_width'}]

    print("Default RootInteractive variables are defined.")
    return aliasArray, transformArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc


def getDefaultVarsDiff(*args, **kwargs):
    return getDefaultVars("diff", *args, **kwargs)

def getDefaultVarsRatio(*args, **kwargs):
    return getDefaultVars("ratio", *args, **kwargs)

