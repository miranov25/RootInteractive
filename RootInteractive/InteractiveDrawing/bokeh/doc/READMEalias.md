## README for bokeh aliases



The alias array is used as an optional argument in the bokehDrawSA to define derived variables on the client side.

`
bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300,
                           aliasArray=aliasArray, histogramArray=histoArray)`

For example usage see also Jupyter notebook tutorial file:
https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/customJsColumns.ipynb



#### Parameters each element of the array takes
* _name_ - the name of alias
* _variables_ - the names of the columns used by the transform (defined as columns in data source) 
* _parameters_ - parameters controlled by widget
* _func_- the function to be computed on the client
* _context_ - the name of the input data source

#### Example alias array as used in the tutorial

Example of the definition of a derived parametric column, a user-defined parametric selection and an efficiency using the histogram ratio

```python
aliasArray = [
    # User-defined JS Columns for defining derived variables depending on widget parameter paramX
    {
        "name": "A_mul_paramX_plus_B",
        "variables": ["A", "B"],
        "parameters": ["paramX"],
        "func": "return paramX * A + B" 
    },
    # They can also be used as selection (boolen)  used e.g. for histogram weights
    {
        "name": "C_accepted",
        "variables": ["C"],
        "parameters": ["C_cut"],
        "func": "return C < C_cut"
    },
    # User-defined JS columns can also be created in histograms by specifying the context (CDS) parameter
    {
        "name": "efficiency_A",
        "variables": ["entries", "entries_C_cut"],
        "func": "return entries_C_cut / entries",
        "context": "histoA"
    },
    {
        "name": "efficiency_AC",
        "variables": ["entries", "entries_C_cut"],
        "func": "return entries_C_cut / entries",
        "context": "histoAC"
    }
]
```