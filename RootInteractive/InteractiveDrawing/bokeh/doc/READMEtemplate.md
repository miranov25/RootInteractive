## README for getDefaultVars

In bokehInteractiveTemplates there are multiple templates for dashboards, used to make creating dashboards easier by reducing the need for boilerplate code.

```python
    # Get default vars
    aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVars('diff', variables=  ["A", "B", "C", "D", "A*A", "A*A+B", "B/(1+C)"], defaultVariables={"varY":["B", "A*A+B"]})
    # Add widgets for selection
    widgetsSelect = [
        ['range', ['A'], {"name":"A"}],
        ['range', ['B'], {"name":"B"}],
        ['range', ['C'], {"name":"C"}],
        ['range', ['D'], {"name":"D"}],
        ]
    widgetParams = mergeFigureArrays(widgetParams, widgetsSelect)
    widgetLayoutDesc["Select"] = [["A","B"],["C","D"]]
    # Make the dashboard
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray, aliasArray=aliasArray, arrayCompression=arrayCompressionRelative16)
```


Bokeh parameters are declared in parameterArray. The widgets that contain parameters must be added to the widget array and the widget layout. 
To simplify the creation and use of the parameter array, a dictionary of predefined parameters has been created.

## Templates:
`aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVars('diff', variables=  ["A", "B", "C", "D", "A*A", "A*A+B", "B/(1+C)"], defaultVariables={"varY":["B", "A*A+B"]})`
This one is used to create a template for a dashboard with multiple tabs
`aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVars('diff', variables=  ["A", "B", "C", "D", "A*A", "A*A+B", "B/(1+C)"], defaultVariables={"varY":["B", "A*A+B"]})`
Similar as previous template, the only difference being the option to toggle the diff function on the client

## parameterArray options:
`[name,value,options]`
* _name_ - the name with which it is indexed in figureArray / aliasArray.
* _value_ - the initial value - the option "default" must be specified, otherwise it will be initialised with the first value in the option list
* _{options}_ - the options that the parameter can have as a value.

Options:
* _range_ - special option if controlled by a slider, the range that the variable can occupy.
* options - options for select e.g list of variables to draw

## Controllable by parameterArray:
* parameters of figure array
  * varX, varY, varZ
* paramters for histograms:
  * axis variables
  * nBins
  * ranges
* parametrizible functions in aliasArray  
* Graphics parameters:
  * marker size
  * legend options - in this example we set the legend font size
* functions in aliasArray

## Predefined parameters
 https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/InteractiveDrawing/bokeh/bokehInteractiveParameters.py#L1-L7

 * Figure options could be parameterized on client. Following parameters to be  provided:
   * parameterArray, widgets, widgetLayout, figureOptions ...
 * The naming convention is following naming in the bokehDraw:
    *  parameterArray - array of parameters  to be added to the parameterArray
    *  widgets        - array of widgets controlling parameters to be added to wifgets
    *  widgetLayout   - array of widgets ID to be added to the widget
    *  figureOptions  - map of the options to be added to the figureArray

* Example predefined parameters for the legend layout parameterization:
  * https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/InteractiveDrawing/bokeh/bokehInteractiveParameters.py#L1-L7
* Example predefined parameters for statistic parameterization
  * https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/InteractiveDrawing/bokeh/bokehInteractiveParameters.py#L42-L53
* Example predefined parameters for the markers
  * https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/InteractiveDrawing/bokeh/bokehInteractiveParameters.py#L15-L20

## Example usage
  
#### test_bokehClientHistogram.py
* Template with multiple weights
  * https://github.com/miranov25/RootInteractive/blob/97885d5967b18c1a432e7fb49806d6f946b6df6a/RootInteractive/InteractiveDrawing/bokeh/test_bokehClientHistogram.py#L32-L36
* Usage with switchable diff function
  * https://github.com/miranov25/RootInteractive/blob/97885d5967b18c1a432e7fb49806d6f946b6df6a/RootInteractive/InteractiveDrawing/bokeh/test_bokehClientHistogram.py#L57-L62
