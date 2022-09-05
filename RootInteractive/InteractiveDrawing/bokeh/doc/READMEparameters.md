## README for parameterArray

The ParameterArray is used for the parameterisation 
of the figures, histograms, selection, weights and the graphic properties on the client.

The declarative programming used in bokehDrawSA is a type of coding where developers express the computational 
logic without having to programme the control flow of each process. This can help simplify coding, as developers 
only need to describe what they want the programme to achieve, rather than explicitly prescribing the steps or 
commands required to achieve the desired result.

```python
bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300,
                           aliasArray=aliasArray, histogramArray=histoArray,arrayCompression=arrayCompression)
```

For example usage see also Jupyter notebook tutorial file:
* https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/standAlone.ipynb

Bokeh parameters are declared in parameterArray. The widgets that contain parameters must be added to the widget array and the widget layout. 
To simplify the creation and use of the parameter array, a dictionary of predefined parameters has been created.


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

## Example parameter array


#### test_Alias.py

https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/InteractiveDrawing/bokeh/test_Alias.py
* Parameterisation of the graphic and user-parameterisable function, and user-defined Javascript function.
* Defining parameter array
  * create custom user defined function `CustomFunc` as text
  * define dynamic cut variable `C<C_cut`
  * define mutliplicative factor for alias function `paramX`
    * https://github.com/miranov25/RootInteractive/blob/a462dd629be518d6977c63fa0498d22a2f4755f6/RootInteractive/InteractiveDrawing/bokeh/test_Alias.py#L21-L27
* Adding control widgets for parameters into widget array 
  * https://github.com/miranov25/RootInteractive/blob/a462dd629be518d6977c63fa0498d22a2f4755f6/RootInteractive/InteractiveDrawing/bokeh/test_Alias.py#L99-L112
* Using parameters in the  client functions/aliases 
  * https://github.com/miranov25/RootInteractive/blob/a462dd629be518d6977c63fa0498d22a2f4755f6/RootInteractive/InteractiveDrawing/bokeh/test_Alias.py#L99-L112 

  
#### test_bokehClientHistogram.py
* Definition of histogram parameters in parameterArray
  * https://github.com/miranov25/RootInteractive/blob/97885d5967b18c1a432e7fb49806d6f946b6df6a/RootInteractive/InteractiveDrawing/bokeh/test_bokehClientHistogram.py#L32-L36
* Usage in the histogramArray
  * https://github.com/miranov25/RootInteractive/blob/97885d5967b18c1a432e7fb49806d6f946b6df6a/RootInteractive/InteractiveDrawing/bokeh/test_bokehClientHistogram.py#L57-L62
* Adding parameters to widgets:
  * https://github.com/miranov25/RootInteractive/blob/97885d5967b18c1a432e7fb49806d6f946b6df6a/RootInteractive/InteractiveDrawing/bokeh/test_bokehClientHistogram.py#L38-L50
