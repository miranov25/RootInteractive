## aliTreePlayer - tools for tree queries
* tree2Panda - main function
* _GetMetadata_, _addMetadata_
  * used to annotate columns  (AxisTitle)
  * more availble in the ALICE code but not ported to RootInteractive
* treeToAnyTree
  * used for queries of the tree varaibles using regular expressions 


## _tree2Panda_

https://github.com/miranov25/RootInteractive/blob/ed1914878e6f4df0083debd7906856f77d243b3f/RootInteractive/Tools/aliTreePlayer.py#L395-L412


## Example usage:
* https://gitlab.cern.ch/alice-tpc-offline/alice-tpc-notes/-/blob/master/JIRA/ATO-609/trackClusterDumpDraw.ipynb
```python
nEntries=12000000
varList=[
    "dca0",
    "hasA","hasC",
    "mTime0",
    ".*dumpVar_*",
    "dEdxtot","dEdxmax",
    "ncl",
    "tgl","qPt",
]
columnMask=[["dumpVar_",""]]
df = tree2Panda(ROOT.treeTr0, varList, "(rndm<0.3)||dumpVar_IR<1000|(rndm<0.6&&dumpVar_isMC)", exclude=["YYYY"], columnMask=columnMask,nEntries=nEntries)
```
* Momentum resolution example use case
  * using columnMaks to replace long cryptic names with shorte and also supress unsuported . in variable names
  * https://gitlab.cern.ch/alice-tpc-offline/alice-tpc-notes/-/blob/7ab10e422686f2641b1a3fb92bb5db78a10fd3fb/JIRA/PWGPP-695/momentumResolution.py#L158-168
  *
  ```python
         varList = [
                     ".*particle.fPdgCode.*",".*particleMother.fPdgCode.*", ".*logPtRes.*",         # MC
                     ".*Refit.*", ".*On.*",                                                         # status
                     ".*cls.*","ncrROCA","nclFractionROCA",                                         # ncl, ncr
                     "sqrtConstChi2",".*normChi2.*",                                                 # chi2s
                     "TPCCross","mdEdx",
                     "dSectorM", "alphaM", "tgl","qPt",
                     "mPtMC", "isPrim","isFromStrangessITS","isFromMaterialITS","isSec"]           # MC info
  
          columnMask=[["particle__particle_f","p_"],["particleMother__particleMother_f","pMother_"],["esdTrack__esdTrack_",""]]
          dfPt = tree2Panda(tree, varList, "select0", exclude=[], columnMask=columnMask,nEntries=nEntries)
  ```


### _GetMetadata_, _addMetadata_

Used to get/set metadata to the TTree User info - metaTable.
Variables in the metaTable are propagated to the Panada structures and are used further in the 
RootInteractive visualization
Until now only `AxisTitle` metadata used


## _treeToAnyTree_

Parse the contents of the tree, branches, aliases and friends and save them in the Python data 
structure anyTree. The anyTree structure will later be used for variable queries in 'tree2Panda' 
and for visualising the tree content.

```python
def treeToAnyTree(tree):
    r"""
    :param tree:  input TTree
    :return:  parent node of the anyTree object
```    

### _GetMetadata_, _addMetadata_

Used to get/set metadata to the TTree User info - metaTable.
Variables in the metaTable are propagated to the Panada structures and are used further in the 
RootInteractive visualization
Until now only `AxisTitle` metadata used
