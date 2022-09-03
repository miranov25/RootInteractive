## README for bokeh figure array

The figure array declaration is used as an argument in the bokehDrawSA to define derived variables on the client side.

The declarative programming used in bokehDrawSA is a type of coding where developers express the computational 
logic without having to programme the control flow of each process. This can help simplify coding, as developers 
only need to describe what they want the programme to achieve, rather than explicitly prescribing the steps or 
commands required to achieve the desired result.

`
bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300,
                           aliasArray=aliasArray, histogramArray=histoArray)`

For example usage see also Jupyter notebook tutorial file:
* https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/standAlone.ipynb

##  Parameters of figure array

* [['x-axis variable'], ['one or more y-axis variables'], {figure options}]
* {figure options} (can be defined for each figure separately or globally)
  *  following bokeh convention if exist
  * predefined options could be find in 
    * https://github.com/miranov25/RootInteractive/blob/b523a66480e0151a90cd7a3d51c08bb19bed2163/RootInteractive/InteractiveDrawing/bokeh/bokehInteractiveParameters.py#L21-L35
  * "size": integer to define marker size
  * "colorZvar": variable to be displayed on the color (z) axis
  * "errY", "errX": variables to be used as the error bars in y, x
  * "xAxisTitle","yAxisTitle"
  * "rescaleColorMapper": if True, color (z) axis will automatically rescale when making the data selection with widgets
  * "legend_options": e.g
    * {"legend_options": {"label_text_font_size": "legendFontSize", "visible": "legendVisible","location":"legendLocation"}}
  * "source" - optional in case non default source used . e..g histogram data source, histogram projection or join
    * string  - in case only one data source
    * array of string - in case separate source for each array X, array Y variable
  * "name" - used later in the layout configuration
*  TODO options to be implemented:
  * plot_title - format string as a parameter similar to the hint
    *  {y-axis}:{x-axis} {colorAxis}

## Figure array modification on client

The parameters of the figure defined on the server (in the Python code) can be changed later with parameters/widget on the client.
In the https://github.com/miranov25/RootInteractive/blob/b523a66480e0151a90cd7a3d51c08bb19bed2163/RootInteractive/InteractiveDrawing/bokeh/bokehInteractiveParameters.py you will find an example of parameterisation

## TODO See more details in the READMEparameters.py

## Example figure array declaration:
* https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/standAlone.ipynb
```figureArray = [
    [['A'], ['A*A-C*C'], {"size": 2, "colorZvar": "A", "errY": "errY", "errX":"0.01"}],
    [['A'], ['C+A', 'C-A', 'A/A']],
    [['B'], ['C+B', 'C-B'], { "colorZvar": "colorZ", "errY": "errY", "rescaleColorMapper": True}],
    [['D'], ['(A+B+C)*D'], {"colorZvar": "colorZ", "size": 10, "errY": "errY"} ],
    [['D'], ['D*10'], {"errY": "errY"}],
    {"size":"size", "legend_options": {"label_text_font_size": "legendFontSize"}}
]
```
* https://github.com/miranov25/RootInteractive/blob/b523a66480e0151a90cd7a3d51c08bb19bed2163/RootInteractive/InteractiveDrawing/bokeh/test_bokehDrawSA.py#L61-L71

```
figureArray = [
#   ['A'], ['C-A'], {"color": "red", "size": 7, "colorZvar":"C", "filter": "A<0.5"}],
    [['A'], ['A*A-C*C'], {"color": "red", "size": 2, "colorZvar": "A", "varZ": "C", "errY": "errY", "errX":"0.01"}],
    [['X'], ['C+A', 'C-A', 'A/A'], {"name": "fig1"}],
    [['B'], ['C+B', 'C-B'], { "colorZvar": "colorZ", "errY": "errY", "rescaleColorMapper": True}],
    [['D'], ['(A+B+C)*DD'], {"colorZvar": "colorZ", "size": 10, "errY": "errY"} ],
#    [['D'], ['D*10'], {"size": 10, "errY": "errY","markers":markerFactor, "color":colorFactor,"legend_field":"DDC"}],
    #marker color works only once - should be constructed in wrapper
    [['D'], ['D*10'], {"size": 10, "errY": "errY"}],
    {"size":"size", "legend_options": {"label_text_font_size": "legendFontSize", "visible": "legendVisible"}}
]
```

## Real ALICE use cases:
* https://gitlab.cern.ch/alice-tpc-offline/alice-tpc-notes/-/blob/7ab10e422686f2641b1a3fb92bb5db78a10fd3fb/JIRA/ATO-609/clusterDumpDraw.ipynb

```
figureArray=[
    [["bin_center"],["isMCTransformed","isDataTransformed"],{"source":"histoX","yAxisTitle":"N", "xAxisTitle":"varX", "errY": ["isMCErr","isDataErr"]}],
    [["bin_center"],["ratio_isData"],{"source":"histoX","yAxisTitle":"N", "xAxisTitle":"varX","errY": ["ratio_isDataErr"]}],
    #
    [["bin_center"],["isGoldTransformed","isSplitTransformed","isEdgeTransformed","isSingleTransformed","entriesTransformed"],{"source":"histoX","yAxisTitle":"N", "xAxisTitle":"varX", 
                        "errY": ["isGoldErr","isSplitErr","isEdgeErr","isSingleErr"]}],
    [["bin_center"],["eff_isGold","eff_isSplit","eff_isEdge","eff_isSingle"],{"source":"histoX","yAxisTitle":"N", "xAxisTitle":"varX","errY": ["eff_isGoldErr","eff_isSplitErr","eff_isEdgeErr","eff_isSingleErr"]}],
    #
    [["bin_center"],["inNoiseTransformed","inSaturTransformed","entriesTransformed"],{"source":"histoX","yAxisTitle":"N", "xAxisTitle":"varX", 
                        "errY": ["inNoiseErr","inSaturErr"]}],
    [["bin_center"],["eff_inNoise","eff_inSatur"],{"source":"histoX","yAxisTitle":"N", "xAxisTitle":"varX","errY": ["eff_inNoiseErr","eff_inSaturErr"]}],

    #
    [["bin_center_1"], ["mean","quantile_1"],{"source":"histoIRData_0"}],
    [["bin_center_1"], ["mean","quantile_1"],{"source":"histoIRMC_0"}],
     
        figureGlobalOption
]
```
* https://gitlab.cern.ch/alice-tpc-offline/alice-tpc-notes/-/blob/master/JIRA/ATO-609/trackClusterDumpDraw.ipynb

```python
figureArray=[
    [["bin_center_1"], ["mean","quantile_1"],{"source":"histoIRData_0"}],
    [["bin_center_1"], ["mean","quantile_1"],{"source":"histoIRMC_0"}],
    [["bin_center"],["isMCTransformed","isDataTransformed"],{"source":"histoX","yAxisTitle":"N", "xAxisTitle":"varX", "errY": ["isMCErr","isDataErr"]}],
    [["bin_center"],["ratio_isData"],{"source":"histoX","yAxisTitle":"N", "xAxisTitle":"varX","errY": ["ratio_isDataErr"]}],
        figureGlobalOption
]
```