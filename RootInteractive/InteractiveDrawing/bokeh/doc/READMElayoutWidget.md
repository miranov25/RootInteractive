## README for bokeh _widgetLayout_

The widget layout (dictionary/array) declaration is used as an argument in bokehDrawSA to create figure layout

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

##  Parameters of  _widgetLayout_
The widget layout can be defined as an array for a simple layout or as a dictionary for tabbed layouts, 
where the last elements should be expressed as a row array. LayoutTab is a dictionary of tabID:layout

The terminal layout consists of n rows. Each row can have an independent number of columns. 
Numbers are placed in the same row and column. The figure ID can be specified by an index in figureArray or 
by the name of the figure if the attribute name of the figure is defined.


### Example usage

* https://gitlab.cern.ch/alice-tpc-offline/alice-tpc-notes/-/blob/master/JIRA/ATO-609/trackClusterDumpDraw.ipynb
  * Example snapshot  for widget declaration and following layout - track properties.
      ```python
      widgetParams=[
                    ['range', ['dca0'],{"name": "dca0"}],
                    ['range', ['tgl'],{"name": "tgl"}],
                    ['range', ['qPt'],{"name": "qPt"}],
                    ['range', ['ncl'],{"name": "ncl"}], 
                    ['range', ['dEdxtot'],{"name": "dEdxtot"}],
                    ['range', ['dEdxmax'],{"name": "dEdxmax"}],
                    ['range', ['mTime0'],{"name": "mTime0"}], 
                    ['multiSelect',["hasA"],{"name":"hasA"}],
                    ['multiSelect',["Run"],{"name":"Run"}],
                    ['multiSelect',["IR"],{"name":"IR"}],
                    ['multiSelect',["isMC"],{"name":"isMC"}],
                    #
                    ['select', ['varX'], {"name": "varX"}],
                    ['slider', ['nbinsY'], {"name": "nbinsY"}],
                    ['slider', ['nbinsX'], {"name": "nbinsX"}],
                    ['select', ['yAxisTransform'], {"name": "yAxisTransform"}],
      ]
    
      widgetLayoutKine=[
          ["dca0","tgl","qPt","ncl"],
          ["dEdxtot","dEdxmax","mTime0"], 
          ["hasA","Run","IR","isMC"], 
          {'sizing_mode': 'scale_width'}
      ]
    
      widgetLayoutDesc={
          "Select":widgetLayoutKine,
          "Histograms":[["nbinsX","nbinsY", "varX","yAxisTransform"], {'sizing_mode': 'scale_width'}],
          "Legend": figureParameters['legend']['widgetLayout'],
          "Markers":["markerSize"]
      }
      ```
