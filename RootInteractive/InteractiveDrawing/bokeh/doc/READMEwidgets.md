## README for bokeh widget array

The widget array declaration is used as an argument in bokehDrawSA to create an array of widgets interactivelly controlling figures/graphs/scatter plots/
Unbined or binned (histogram) bokeh data sources and derived variables and aggregated statistics.

The declarative programming used in bokehDrawSA is a type of coding where developers express the computational 
logic without having to programme the control flow of each process. This can help simplify coding, as developers 
only need to describe what they want the programme to achieve, rather than explicitly prescribing the steps or 
commands required to achieve the desired result.

```python
bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300,
                           aliasArray=aliasArray, histogramArray=histoArray,arrayCompression=arrayCompression)`
```

For example usage see also Jupyter notebook tutorial file:
* https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/standAlone.ipynb

##  Parameters of widget array

The layout of the widget array is as follows:

`[[widgetType,[variableName],{widgetOptions}], ...]`
* _widgetType_ - type of widget bokeh or composite.
* _variable_name_ - Name of the variable in the CDS or name of the parameter to be controlled
* _widgetOptions_ - dictionary of widget options needed for the bokeh widgets or derived RootInteractive widgets.

Bokeh standard widgets and RooInteractive composed widgets can be used to parameterzie dashboard behaviour. By default 
dictionary of options with name  as in standard bokeh used. Some extension, new variable to simplify the queries provided.
In the first version of the  RootInteractive some options could be specified as element of array, but that options is depreacated.

## Standard bokeh widget types supported
* _range_
* _slider_
* _spinner_
* _select_ 
* _multiSelect_
* _toggle_
* _text_

## RootInteractive special
* _textQuery_
* _multiSelectBitmask_
* _spinnerRange_

### Widget special options
* _name_
  * can be specified for all widgets, used to identify widget in widget layout
*_toggleable_    
  * switch to enable/disable usage of the widget


### Example usage test_bokehDrawSA.py
* defining parameters of dashboard
https://github.com/miranov25/RootInteractive/blob/97885d5967b18c1a432e7fb49806d6f946b6df6a/RootInteractive/InteractiveDrawing/bokeh/test_bokehDrawSA.py#L52-L59
* defining widgets controling data source selection and parameters
* Some example of RootInteractive custom widgets:
  * in example below some widgets are toglebal, their usage can be switched ON and OFF 
  * spinnerRange can be used for ubound range selection with exponential delta step
https://github.com/miranov25/RootInteractive/blob/97885d5967b18c1a432e7fb49806d6f946b6df6a/RootInteractive/InteractiveDrawing/bokeh/test_bokehDrawSA.py#L76-L96


### Example usage test_bokehClienHistogram.py
* defining parameter for visualization
  * number of  bins 
  * histogram ranges
https://github.com/miranov25/RootInteractive/blob/97885d5967b18c1a432e7fb49806d6f946b6df6a/RootInteractive/InteractiveDrawing/bokeh/test_bokehClientHistogram.py#L32-L36

* defining widgets for data source selection and parameter control
https://github.com/miranov25/RootInteractive/blob/97885d5967b18c1a432e7fb49806d6f946b6df6a/RootInteractive/InteractiveDrawing/bokeh/test_bokehClientHistogram.py#L38-L49