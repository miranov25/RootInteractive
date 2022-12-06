# RootInteractive

Code for the interactive visualisation of multidimensional data in ROOT or native Python formats (Panda, numpy).

Support for ROOT data structures:
* TTree and TTreeFormula, aliases ... 
* TFormula, or any static Root/AliRoot functions.
* RDataFrame <-> awkward - work in progress


Root and PyRoot (AliRoot/O2) data structures could be used as input data sources. However, the code also works with pandas only, without the need to have the ROOT package installed. Internally, these data structures are converted into the Bokeh CDS (ColumnDataSource) or in our RootInteractive
CDS for the NDimensional histograms, projections and aggregated information.


## RootInteractive content:

* Interactive, easily configurable visualisation of non-binned and binned data.
* Interactive n-dimensional histogramming/projection and derived aggregated information extraction
* Client/server application Jupyter, Bokeh 
* Standalone client application - (Bokeh Standalone Dashboard)
*  Lossy and lossless data compression (server- > client)
  


### Interactive visualization, histogramming and data aggregation in N-dimensions on client
 

The figure array declaration is used as an argument in bokehDrawSA to create an array of figures/graphs/scatter plots/
Unbined or binned (Ndimension histogram and derived statistics/projection) bokeh data sources and derived variables and aggregated statistics can be used for drawing.

The declarative programming used in bokehDrawSA is a type of coding where developers express the computational 
logic without having to programme the control flow of each process. This can help simplify coding, as developers 
only need to describe what they want the programme to achieve, rather than explicitly prescribing the steps or 
commands required to achieve the desired result.

The interactive visualization is declared in the 6 arrays as ine the example bellow
```python
bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, parameterArray=parameterArray,
                          widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", nPointRender=300,
                           aliasArray=aliasArray, histogramArray=histoArray,arrayCompression=arrayCompression)
```
* [figureArray](RootInteractive/InteractiveDrawing/bokeh/doc/READMEfigure.md)
* [histogramArray](RootInteractive/InteractiveDrawing/bokeh/doc/READMEhistogram.md)
* [aliasArray](RootInteractive/InteractiveDrawing/bokeh/doc/READMEalias.md)
* [layout](RootInteractive/InteractiveDrawing/bokeh/doc/READMElayout.md)
* [widgetLayout](RootInteractive/InteractiveDrawing/bokeh/doc/READMElayoutWidget.md)
* [parameterArray](RootInteractive/InteractiveDrawing/bokeh/doc/READMEparameters.md)
* [arrayCompression](RootInteractive/InteractiveDrawing/bokeh/doc/READMEcompression.md)


#### _figureArrray_ - figure parameterization 
* see [READMEfigure](/RootInteractive/InteractiveDrawing/bokeh/doc/READMEfigure.md)
* Defining scatter/histogram/derived figures using input data source 
* Example declaration of the figure from data source with columns ABCD
  ```python
  figureArray = [
  [['A'], ['A*A-C*C'], {"size": 2, "colorZvar": "A", "errY": "errY", "errX":"0.01"}],
  [['A'], ['C+A', 'C-A', 'A/A']],
  [['B'], ['C+B', 'C-B'], { "colorZvar": "colorZ", "errY": "errY", "rescaleColorMapper": True}],
  [['D'], ['(A+B+C)*D'], {"colorZvar": "colorZ", "size": 10, "errY": "errY"} ],
  [['D'], ['D*10'], {"errY": "errY"}],
  {"size":"size", "legend_options": {"label_text_font_size": "legendFontSize"}}
  ]
  ```
#### _histogramAray_ - interactive histogramming parameterization and examples
* Defining interactive ND histogramsand derived statistics,  updated based on the user selection, resp. by parametriz 
* see [READMEhistogram](/RootInteractive/InteractiveDrawing/bokeh/doc/READMEhistogram.md) 
* Example of creating a 3D histogram showing mean, sum and standard in the projection with colour code in the second dimension
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

#### _aliasArray_   alias/client side function parameterization
* see [READMEaliase](/RootInteractive/InteractiveDrawing/bokeh/doc/READMEalias.md)
* javascrript function with which you can define derived variables on the client. Used e.g. to parameterise the selection,
 histogram weights, efficiencies
* newly created variables can be used in histogramArray, figureAray, aliasArray
* Dependency trees to ensure consistency of aliases and the correct order of evaluation of derived variables and use in visualisation.
* Example declaration:
  ```python
      aliasArray = [
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
          # Shorthand notation - only for scalar functions
          ("effC", "entries_C_cut / bin_count", "histoAC"),
      ]
  ```    


#### _widgetLayout_ - layout of the figures
  * [READMElayout](/RootInteractive/InteractiveDrawing/bokeh/doc/READMElayout.md)
  * Layout declared by and dictionary(tabs)/array of figure IDs (index or name ID)
  * Properties per row/simple layout/tab layout can be specified. More local properties have priority.
  * Example declaration:
    ```python
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

#### _layout_ - layout of the widgets
  * see [READMElayoutWidget](/RootInteractive/InteractiveDrawing/bokeh/doc/READMElayoutWidget.md)
  * Layout declared by and dictionary(tabs)/array of figure IDs (index or name ID)
  * Properties per row/simple layout/tab layout can be specified. More local properties have priority.    
  * Example declaration:
    * simple layout
       ```python 
        widgetLayoutKine=[
            ["dca0","tgl","qPt","ncl"],
            ["dEdxtot","dEdxmax","mTime0"], 
            ["hasA","Run","IR","isMC"], 
            {'sizing_mode': 'scale_width'}
        ]
      ```
    * composed layout:
        ```python 
        widgetLayoutDesc={
            "Select":widgetLayoutKine,
            "Histograms":[["nbinsX","nbinsY", "varX","yAxisTransform"], {'sizing_mode': 'scale_width'}],
            "Legend": figureParameters['legend']['widgetLayout'],
            "Markers":["markerSize"]
        }
        ```

#### _arrayCompresion_ - 
  * see [READMEcompression](/RootInteractive/InteractiveDrawing/bokeh/doc/READMEcompression.md) 
    * https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/Tools/compressArray.py#L141-L196
    * https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/compression.ipynb
  * Significant data compression of the data (up to O(5-10%)) for server- > client transmission and for storage in the html file. 
   * Compression depends heavily on the entropy of the data after lossy compression and on data repetition
   * Lossy and lossles compression expressed by regular expression per columns
   * In realistic use cases facto 10-100 compression achieved
   * further compression - using javascript aliases on client  instead data transfer 
  * Example declaration:
    ```python
    arrayCompressionParam=[(".*conv.*Sigma.*",[("relative",7), ("code",0), ("zip",0), ("base64",0)]),
                               (".*delta.*",[("relative",10), ("code",0), ("zip",0), ("base64",0)]),
                               (".*i2.*",[("relative",7), ("code",0), ("zip",0), ("base64",0)]),
                               (".*",[("relative",8), ("code",0), ("zip",0), ("base64",0)])]
    ```

### Machine learning part  -  work in progrees
  * Wrappers for decision trees and Neural Net
  * Provides interface for the reducible, irreducible errors, proability density function
  * Local linear forest, resp. local kernel regression
  


## RootInteractive Information

* RootInteractive github (source code)
  * https://github.com/miranov25/RootInteractive
  * JIRA: https://alice.its.cern.ch/jira/browse/PWGPP-485
* Documentation server at CERN (TODO -add reular update)
  * https://rootinteractive.web.cern.ch/RootInteractive/html/ 
  * Not yet regularly updated - TODO
  * /eos/user/r/rootinteractive/www/html/

## Tutorials
* 1.) Bokeh draw standalone (graphs,compression, down-sampling)
  * https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/standAlone.ipynb
* 2.) N dimensional histogramming on client (data aggregation)
  * https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/test_bokehClientHistogram.ipynb 
* 3.) Custom function on client:
  * https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/customJsColumns.ipynb 

## ALICE ROOTIntteractive tutorial
Sevearal ALICE use case (detector calibration, QA/QC)
* https://indico.cern.ch/event/1135398/


## Galery material in the ALICE agenda () and document server

* Support material for RCU note [N2]
  * [D1] Visualization of the common-mode effect dependencies using ROOT interactive ( 11 Dimensions)
    * https://gitlab.cern.ch/aliceeb/TPC/-/blob/master/SignalProcessing/commonModeFractionML.html   
  * [D2] Visualization of the ion-tail fit parameters and correction graphs using ROOT interactive (12 Dimensions)
     * https://gitlab.cern.ch/aliceeb/TPC/-/blob/master/SignalProcessing/ionTailFitParameters_sectorScan.html    
  * [D3] Visualization of the toy MC results using ROOT interactive (13 Dimensions)
    * https://gitlab.cern.ch/aliceeb/TPC/-/blob/master/simulationScan/toyMCParameterScan.html  

* Support material for V0 reconstruction studies [P1]
  * [D4] Interactive invariant mass histogram  dashboards (6+2 Dimensions)
    * https://indico.cern.ch/event/1088044/#sc-1-3-interactive-histograms
  * [D5] Pt and invariant mass performance maps dashboards
    * https://indico.cern.ch/event/1088044/#sc-1-2-gamma-dashboards
    * https://indico.cern.ch/event/1088044/#sc-1-4-k0-dashboards 

* QA and production preparation :
  * [D6] QA comparison of ongoing MC and raw data production (LHC18q,r, LHC18c,LHC16f,LHC17g..)   See interactive dashboards in agenda of calibration/tracking meeting:
    * https://indico.cern.ch/event/991449/ , https://indico.cern.ch/event/991450/  , https://indico.cern.ch/event/991451/ 
* PID
  * [D7] TPC PID calibration  and QA
    * https://indico.cern.ch/event/983778
      * https://alice.its.cern.ch/jira/secure/attachment/53371/qaPlotPion_test1.html 
      * https://indico.cern.ch/event/991451/contributions/4220782/attachments/2184007/3689893/qaPlotPion_Delta.html 
* Fast MCkalman and event display
  * [D8] Space charge distortion calibration (Run3) and performance optimization (Run2, Alice3) - [P9]
     * https://indico.cern.ch/event/1091510/contributions/4599999/attachments/2338476/3986580/residualTrackParam.html
     * https://indico.cern.ch/event/1087849/contributions/4577709/attachments/2331293/3973338/residual_track_parameter_Dist_GainIBF.html 
  * [D9] High dEdx (spallation product) reconstruction  and magnetic monopole tracking
     * https://indico.cern.ch/event/991452/contributions/4222204/attachments/2184856/3691411/seed1Display2.html   

* Space charge distortion calibration
  * [D10] digital current grouping and factorization studies 
    * https://indico.cern.ch/event/1091510/
    * https://indico.cern.ch/event/1087849/	