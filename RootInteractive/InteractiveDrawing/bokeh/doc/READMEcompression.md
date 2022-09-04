## README for bokeh data source compression _compressionArray_

The dat source compression declaration _compressionArray_ is used as an argument in bokehDrawSA to create figure layout

The declarative programming used in bokehDrawSA is a type of coding where developers express the computational 
logic without having to programme the control flow of each process. This can help simplify coding, as developers 
only need to describe what they want the programme to achieve, rather than explicitly prescribing the steps or 
commands required to achieve the desired result.

`
bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300,
                           aliasArray=aliasArray, histogramArray=histoArray,arrayCompression=arrayCompression)`

## Compression array declaration
see https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/Tools/compressArray.py#L11-L13

The data is compressed on the master and later expanded on the client. 
Compression is per column and settings can be specified per column with regular expressions. Significant compression can be achieved by lossy compression (absolute or relative rounding) followed by entropy coding (zip).
The column is compressed based on the recipe in the first element in arrayCompression corresponding to the regular expression.



_arrayCompression_ 
* _variableMask_
  * regular expression to select variable 
* _compressionArray_ - optional filter
  * relative
  * absolute
  * code
  * zip
  * base64


For example usage see also Jupyter notebook tutorial file:
* https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/standAlone.ipynb

## Example declaration
* https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/standAlone.ipynb
* simple declaration all columns `(.* mask)` rounded to 8 bits (`"relatve",8)`
```python
arrayCompressionRelative8=[(".*",[("relative",8), ("code", 0), ("zip",0), ("base64",0)])]
```
## Column dependent declaration
* https://gitlab.cern.ch/alice-tpc-offline/alice-tpc-notes/-/blob/master/JIRA/ATO-560/fitFFT.ipynb
* different precision for the `(.*Sigma)`,  `(.*delta)`,  `(.*i2)` and the rest  `(.*)`
```python
arrayCompressionParam=[(".*conv.*Sigma.*",[("relative",7), ("code",0), ("zip",0), ("base64",0)]),
                           (".*delta.*",[("relative",10), ("code",0), ("zip",0), ("base64",0)]),
                           (".*i2.*",[("relative",7), ("code",0), ("zip",0), ("base64",0)]),
                           (".*",[("relative",8), ("code",0), ("zip",0), ("base64",0)])]
```
