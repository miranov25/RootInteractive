## README for bokeh layout

The figure layout array declaration is used as an argument in bokehDrawSA to create figure layout

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
The figure layout can be defined as an array for a simple layout or as a dictionary for tabbed layouts, 
where the last elements should be expressed as a row array. LayoutTab is a dictionary of tabID:layout

The terminal layout consists of n rows. Each row can have an independent number of columns. 
Numbers are placed in the same row and column. The figure ID can be specified by an index in figureArray or 
by the name of the figure if the attribute name of the figure is defined.

The layout can be parameterised and properties can be changed per line, per terminal layout or per tab layout. 
Local properties have priority.
The following layout properties can be specified:
* _commonX_, _commonY_ - common X and Y axis range respectively. * Parameter - with which figure the axis is shared. 
* _x\_visible_, _y\_visible_ - switches on the display of the X and Y axes
* _plot_height_ - plot height per line or for the whole layout * should be adjusted for different layouts.
* _sizing\_mode_ - scale_width to be default - with herarchical layouts sometimes problems



### Example:
* https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/standAlone.ipynb
```
layout = {
    "A": [
        [0, 1, 2, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
        ],
    "B": [
        [3, 4, {'commonX': 1, 'y_visible': 3, 'x_visible':1, 'plot_height': 100}],
        {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
        ]
}
```
