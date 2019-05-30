# RootInteractive

Code for interactive visualization of multidimensional data in ROOT or native Python formats (Panda, numpy)
Support for ROOT data structures:
* THn   - for multi-dimensional histograms
* TTree and TTreeFormula, Aliases .. 
* TFormula, resp. any static Root/AliRoot functions

### Additional dependencies
* ROOT
* aliBuild, AliRoot
  * only one library libSTAT needed  - to be written as standalone package ...



## RootInteractive content:


#### Visualization part wrappers
* client/server application (Jupyter+Bokeh+???)
* client application - (Bokeh standalone)

#### Machine learning part (plans)
We would like to provide set of wrappers which will simplify/and user analysis using tabular data (trees,csv, pandas ...)
error estimates, robust statistic, handling of extrapolation errors
set of predefined (parametrizable layouts - e.g autonecoders)

Similar approach was choosen recently by GOOGLE (announcement -09.04.2019) and Microsoft (announcement 02.05.2019)

#### Integrate the ML part with visualization tools
Similar approach as was choosen by Ilastic project



## Detailed description for ALICE user in JIRA
For ALICE CERN see detailed description in issue tracker in JIRA:
* Jupyter notebooks for interactive n-dimensional analysis. Ipywidgets + bokeh. RootInteractive
https://alice.its.cern.ch/jira/browse/PWGPP-485
* ROOT tabular metadata description
* ....