import ROOT

def filterRDFColumns(rdf, selectList=[".*"], excludeList=[], selectTypeList=[".*"], excludeTypeList=[".*AliExternal.*"], verbose=0):
    """
    function to filter available columns in RDataFrame
    :param rdf:               - input RDataFrame
    :param selectList:        - columns to select (regExp)
    :param excludeList:       - columns to reject (regExp)
    :param selectTypeList     - types to accept   (regExp)
    :param excludeTypeList:   - types to reject   (regExp)
    :param verbose:           - verbosity 0x1 -print all status  0x2 - print selected  , 0x4 print rejected
    :return:                    filtered list of columns

    example:
    filterRDFColumns(rdf1, ["param.*","delta","covar"],["part.",".*Refit.*"],[".*"],[""], verbose=1)
    """
    inputList=rdf.GetColumnNames()
    import re
    selectListRe=[]
    excludeListRe=[]
    selectTypeListRe=[]
    excludeTypeListRe=[]
    for sel in selectList:
        selectListRe.append(re.compile(sel))
    for sel in excludeList:
        excludeListRe.append(re.compile(sel))
    for sel in selectTypeList:
        selectTypeListRe.append(re.compile(sel))
    for sel in excludeTypeList:
        excludeTypeListRe.append(re.compile(sel))

    selected=[]
    for column0 in inputList:
        column=str(column0)
        try:
            type=rdf.GetColumnType(column)
        except:
            type=""
            pass

        isOK=False
        isOKType=False
        if verbose&0x8 : print(column,type)
        # test columns
        for iSel, select in enumerate(selectList):
            if column == select:
                isOK=True
                break
            if selectListRe[iSel].fullmatch(column):
                isOK=True
                break
        for iRej, reject in enumerate(excludeList):
            if column == reject:
                isOK=False
                break
            if excludeListRe[iRej].fullmatch(column):
                isOK=False
                break
        #test type
        for iSel, select in enumerate(selectTypeList):
            if column == type:
                isOKType=True
                break
            if selectTypeListRe[iSel].fullmatch(type):
                isOKType=True
                break
        for iRej, reject in enumerate(excludeTypeList):
            if column == type:
                isOKType=False
                break
            if excludeTypeListRe[iRej].fullmatch(type):
                isOKType=False
                break

        if (verbose &0x1)>0:
            print(f"Column {isOK} {isOKType}", column,type)
        if ((verbose & 0x2)>0) & ((isOK & isOKType) is True) :
            print(f"Column {isOK} {isOKType}", column,type)
        if ((verbose & 0x4)>0) & ((isOK & isOKType) is False) :
            print(f"Column {isOK} {isOKType}", column,type)

        if isOK & isOKType:
            selected.append(column)
            #print(column,type)
    return selected
