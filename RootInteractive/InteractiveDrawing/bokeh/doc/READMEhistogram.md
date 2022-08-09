## README for bokeh histograms
The histogram array is used as an optional argument in the bokehDrawSA to define histograms and histogram 
statistics on the client side.

The declarative programming used in bokehDrawSA is a type of coding where developers express the computational 
logic without having to programme the control flow of each process. This can help simplify coding, as developers 
only need to describe what they want the programme to achieve, rather than explicitly prescribing the steps or 
commands required to achieve the desired result.

`
bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300,
                           aliasArray=aliasArray, histogramArray=histoArray)`

For example usage see also Jupyter notebook tutorial file:
https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/test_bokehClientHistogram.ipynb


### Parameters each element of the histogram array takes

The RootInteractive histograms parameterization was inspired by the numpy histogram and Root histogram syntax.
* name   - histogram name 
* variables - array of variables used (1D or ND)
* nbins  - number for 1D or array for >1D e.g:
  * nbins=20 in 1D (see )
  * nbins=[20,20] in 2D
* range  - range for the histogram axis
  * for 2D eg. [[0,1],[0,1]]
  * None can be used to enable automatic range (min,max)
  * range can be parameterized using widget parameters (starting on RI v0-01-06)
* quantiles - quantile array to calculate - used for tableHisto 
* sum_range - integral in range - used for tableHisto output
* weight    - weigh used to fill histogram

### Example use case

https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/test_bokehClientHistogram.ipynb

### Example with table aggregation:

```python
    histoArray = [
        {"name": "histoA", "variables": ["A"], "nbins":20, "quantiles": [.05, .5, .95], "sum_range": [[.25, .75], [.4, .6]]},
        {"name": "histoB", "variables": ["B"], "nbins":20, "range": [0, 1]},
        {"name": "histoTransform", "variables": ["(A+B)/2"], "nbins": 20, "weights": "A*C"},
    ]
    figureArray = [
        [['A'], ['histoA']],
        [['B'], ['histoB'], {"show_histogram_error": True}],
        [['(A+B)/2'], ['histoTransform'], {"yAxisTitle": "sum A*C"}],
        ["tableHisto", {"rowwise": True}]
    ]
```
### Example with 2D histogram and projections
Showing histogram of A vs (A+B)/2, mean, median, RMS, quantiles of A as a function of (A+B)/2

```python
    histoArray = [
        {"name": "histoAB", "variables": ["A", "(A+B)/2"], "nbins": [20, 20], "weights": "D", "axis": [0, 1], "quantiles": [.1, .5, .9]}
    ]
    figureArray = [
        [['quantile_0', 'quantile_1', 'quantile_2'], ['bin_center_1'], {"source": "histoAB_0"}],
        [['quantile_1', 'mean'], ['bin_center_1'], {"source": "histoAB_0"}],
        [['A'], ['histoAB'], {"yAxisTitle": "(A+B)/2"}],
        [['std'], ['bin_center_1'], {"source": "histoAB_0"}],
        ["tableHisto", {"rowwise": False}]
    ]
```

### Example with projections of a 3D histogram into 2D

* Uniform distribution in varable A,B,C, weighted 3D histogram of (A+C)/2, B, C with D as weights
* showing mean and rms in (A+C)/2 as function of B with C bin center as a color

```python
histoArray = [
        {"name": "histoABC", "variables": ["(A+C)/2", "B", "C"], "nbins": [8, 10, 12], "weights": "D", "axis": [0], "sum_range": [[.25, .75]]},
    ]
    figureArray = [
        [['bin_center_1'], ['mean']],
        [['bin_center_1'], ['sum_0']],
        [['bin_center_1'], ['std']],
        {"source": "histoABC_0", "colorZvar": "bin_center_2", "size": 7}
    ]
```