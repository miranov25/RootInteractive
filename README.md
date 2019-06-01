# RootInteractive

Code for interactive visualization of multidimensional data in ROOT or native Python formats (Panda, numpy)
Support for ROOT data structures:
* THn   - for multi-dimensional histograms
* TTree and TTreeFormula, Aliases .. 
* TFormula, resp. any static Root/AliRoot functions

#### No python packages dependencies
* ROOT
* aliBuild, AliRoot
  * only one library libSTAT needed  - to be written as standalone package ...



## RootInteractive content:

* **Visualization part wrappers**
  * client/server application (Jupyter, Bokeh, bqplot (in future))
  * client application - (Bokeh standalone)

* **Machine learning part (plans)**
  * We would like to provide set of wrappers which will simplify/and user analysis using tabular data (trees,csv, pandas ...)
error estimates, robust statistic, handling of extrapolation errors
set of predefined (parametrizable layouts - e.g autonecoders).
Similar approach was choosen recently by GOOGLE (announcement -09.04.2019) and Microsoft (announcement 02.05.2019)
* **Integration of the ML part with visualization tools**
  * Similar approach as was chosen by Ilastic project


### RootInteractive Information

* RootInteractive github (source code)
  * https://github.com/miranov25/RootInteractive
  * JIRA: https://alice.its.cern.ch/jira/browse/PWGPP-485
* RootIteractive tutorial github
  * https://github.com/miranov25/RootInteractiveTest
  * Mostly example Jupyter notebooks  using Alice data
  * JIRA https://alice.its.cern.ch/jira/browse/PWGPP-532
* Data server at CERN
  * https://rootinteractive.web.cern.ch/RootInteractive/data/ 
  * data on eos: 
    * /eos/user/r/rootinteractive/www/testData/
    * /eos/user/r/rootinteractive/www/data/
* Documentation server at CERN
  * https://rootinteractive.web.cern.ch/RootInteractive/html/ 
  * /eos/user/r/rootinteractive/www/html/


### Detailed description for ALICE user in JIRA
For ALICE CERN see detailed description in issue tracker in JIRA:
* Jupyter notebooks for interactive n-dimensional analysis. Ipywidgets + bokeh. RootInteractive
https://alice.its.cern.ch/jira/browse/PWGPP-485
* ROOT tabular metadata description
* ....