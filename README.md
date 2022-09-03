# RootInteractive

Code for interactive visualisation of multidimensional data in ROOT or native Python formats (Panda, numpy).
Support for ROOT data structures:
* TTree and TTreeFormula, aliases ... 
* TFormula, or any static Root/AliRoot functions.
* RDataFrame <-> awkward - work in progress

#### No python packages dependencies on AliRoot
* ROOT + libStat 



## RootInteractive content:

* **Visualisation Part Wrapper**. 
* Interactive, easily configurable visualisation of unbinned and binned data. 
  *  
* Interactive histogramming 
* Client/server application Jupyter, Bokeh 
* Client standalone application - (Bokeh Standalone)
  


* **Interactive   histogramming and data aggregation**  in many dimensions on client
  * figure parameterization
    * [READMEfigure](/RootInteractive/InteractiveDrawing/bokeh/doc/READMEfigure.md)
  * alias/client side function parameterization
    * [READMEaliase](/RootInteractive/InteractiveDrawing/bokeh/doc/READMEalias.md)
  * interactive histogramming parameterization and examples:
    * [READMEhistogram](/RootInteractive/InteractiveDrawing/bokeh/doc/READMEhistogram.md)
  * layout of the fugures and widgets
    * [READMElayout](/TODO)
* **Machine learning part ** work in progrees
  * Wrappers for decision trees and Neural Net
  * Provides interface for the reducible, irreducible errors, proability density function
  * Local linear forest, resp. local kernel regression
  


### RootInteractive Information

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


### Galery material in the ALICE agenda and document server

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