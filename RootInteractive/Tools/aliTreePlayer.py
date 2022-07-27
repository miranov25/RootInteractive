import pandas as pd
import numpy as np
import urllib.request as urlopen
import pyparsing
import sys
import uproot
from RootInteractive.Tools.pandaTools import *
try:
    import ROOT
    ROOT.gSystem.Load("$ALICE_ROOT/lib/libSTAT.so")
except ImportError:
    pass

from anytree import *
#if "ROOT" in sys.modules:
#    from root_pandas import *

import re
import logging
from pandas import CategoricalDtype

def readDataFrameURL(fName, nrows=0):
    if 'http' in fName:
        line = urlopen(fName).readline().rstrip()  # get header - using root cvs convention
    else:
        line = open(fName).readline().rstrip()  # get header - using root cvs convention
    names = line.replace("/D", "").replace("/I", "").split(":")
    variables = []
    for a in names: variables.append(a.split('\t')[0])  #
    dataFrame = pd.read_csv(fName, sep='\t', index_col=False, names=variables, skiprows=1, nrows=nrows)
    return dataFrame


def SetAlias(data, column_name, formula):
    """
    :param data:            panda data frame
    :param column_name:     name of column for further query
    :param formula:         alias formula
    :return:                new panda data frame
    """
    newCol = data.eval(formula)
    out = data.assign(column=newCol)
    out = out.rename(columns={'column': column_name})
    return out


def treeToPanda(tree, variables, selection, nEntries, firstEntry, columnMask='default'):
    r"""
    Convert selected items from the tree into panda table
        TODO:
            * import fail in case of number of entries>2x10^6  (infinite loop) - to check the reason
            * use data source and aliases to enable characters forbidden in Pandas .[]() which are allowed in ROOT trees

    :param tree:            input tree
    :param variables:       ":" separated variable list
    :param selection:       tree selection ()
    :param nEntries:        number of entries to query
    :param firstEntry:      first entry to query
    :param columnMask:      mask - replace variable
    :return:                panda data frame
    """
    entries = tree.Draw(str(variables), selection, "goffpara", nEntries, firstEntry)  # query data
    columns = variables.split(":")
    # replace column names
    #    1.) pandas does not allow dots in names
    #    2.) user can specified own mask
    for i, column in enumerate(columns):
        if columnMask == 'default':
            column = column.replace(".fElements", "").replace(".fX$", "X").replace(".fY$", "Y")
        else:
            masks = columnMask.split(":")
            for mask in masks:
                column = column.replace(mask, "")
        columns[i] = column.replace(".", "_")

    ex_dict = {}
    for i, a in enumerate(columns):
        val = tree.GetVal(i)
        ex_dict[a] = np.frombuffer(val, dtype=float, count=entries)
    df = pd.DataFrame(ex_dict, columns=columns)
    initMetadata(df)
    metaData = tree.GetUserInfo().FindObject("metaTable")
    if metaData:
        for key in metaData:
            df.meta.metaData[key.GetName()] = key.GetTitle()
    return df


#  Dictionary processing

def aliasToDictionary(tree):
    """
    :param tree: input tree
    :return: dictionary of aliases
    """
    aliases = {}
    if tree.GetListOfAliases() is not None:
        for a in tree.GetListOfAliases():
            aliases[a.GetName()] = a.GetTitle()
    return aliases


def __processAnyTreeBranch(branch0, parent):
    nodeF = Node(branch0.GetName(), parent=parent, ttype="branch")
    for branch in branch0.GetListOfBranches():
        __processAnyTreeBranch(branch, nodeF)


def treeToAnyTree(tree):
    r"""
    :param tree:  input TTree
    :return:  parent node of the anyTree object

    Example usage:
        see test_aliTreePlayer.py::test_AnyTree()

            >>> branchTree=treeToAnyTree(treeQA)
            >>> print(findall(branchTree, filter_=lambda node: re.match("bz", node.name)))
            >>> print(findall(branchTree, filter_=lambda node: re.match("MIP.*Warning$", node.name)))
            ==>
            (Node('/tree/bz'),)
            (Node('/tree/MIPattachSlopeA_Warning'), Node('/tree/MIPattachSlopeC_Warning'), Node('/tree/MIPattachSlope_comb2_Warning'), Node('/tree/MIPquality_Warning'))
    """
    parent = Node("", ttype="base")
    for branch in tree.GetListOfBranches():
        __processAnyTreeBranch(branch, parent)
    if tree.GetListOfAliases():
        for alias in tree.GetListOfAliases():
            Node(alias.GetName(), parent=parent, ttype="alias")
    if tree.GetListOfFriends():
        for friend in tree.GetListOfFriends():
            treeF = tree.GetFriend(friend.GetName())
            nodeF = Node(friend.GetName(), parent=parent, ttype="branch")
            for branch in treeF.GetListOfBranches():
                __processAnyTreeBranch(branch, nodeF)
    tree.anyTree = parent
    return parent


def findSelectedBranch(anyTree, regexp, **findOption):
    """
    :param anyTree:
    :param regexp:
    :param findOption:
        * stop=None
        * maxlevel=None
        * mincount=None
        * maxcount=None
    :return:
        selected anyTree branches

    Example usage:
        >>>   test_aliTreePlayer.py::test_AnyTree()
        >>>   branchTree = treeToAnyTree(tree)
        >>> print(findSelectedBranch(branchTree, "MIP.*Warning"))
        ==> (Node('/MIPattachSlopeA_Warning'), Node('/MIPattachSlopeC_Warning'), Node('/MIPattachSlope_comb2_Warning'), Node('/MIPquality_Warning'))
    """
    options = {
        "inPath": 1,
        "inName": 1
    }
    options.update(findOption)
    array = []
    if options["inPath"] > 0:
        array += findall(anyTree, filter_=lambda nodeF: re.match(regexp, str(nodeF.path)), **findOption)
    if options["inName"] > 0:
        array += findall(anyTree, filter_=lambda nodeF: re.match(regexp, str(nodeF.name)), **findOption)
    return array


def findSelectedBranches(anyTree, include, exclude, **findOption):
    """
    :param anyTree:  anyTree or TTree
    :param include:  include array of regular expression
    :param exclude:  exclude array
    :return:  array of selected expression
    Example usage:
        >>> anyTree = treeToAnyTree(treeMap)
        >>> print("Search 0:",  findSelectedBranches(anyTree, [".*LHC15o.*Chi2.*meanG.*"], [".*ITS.*"]))
        >>> print("Search 1:",  findSelectedBranches(anyTree, [".*LHC15o.*Chi2.*meanG.*"], [".*TPC.*"]))
        >>>
        >>> Search 0 ['LHC15o_pass1.hnormChi2TPCMult_Tgl_mdEdxDist/meanG', 'LHC15o_pass1.hnormChi2TPCMult_Tgl_qPtDist/meanG']
        >>> Search 1 ['LHC15o_pass1.hnormChi2ITSMult_Tgl_mdEdxDist/meanG', 'LHC15o_pass1.hnormChi2ITSMult_Tgl_qPtDist/meanG']
    """
    if isinstance(anyTree, ROOT.TTree):
        anyTree = treeToAnyTree(anyTree)
    options = {}
    options.update(findOption)
    variablesTree = []
    for selection in include:
        for var in findall(anyTree, filter_=lambda node: re.match(selection, str(node.leaves[-1]))):
            path = str(var.leaves[-1]).split("'")[1].replace("//", "")
            isOK = 1
            if exclude:
                for varE in exclude:
                    if re.match(varE, path):
                        isOK = 0
                        break
            if isOK > 0:
                variablesTree.append(path)
    variablesTreeS=set(variablesTree)
    variablesTree=list(variablesTreeS)
    return variablesTree


def makeAliasAnyTree(key, aliases, parent=None):
    """
    Build recursive alias anytree

    :param key:          - start key
    :param parent:       - initial node
    :param aliases:      - alias dictionary
    :return:               anytree object
    """
    if parent is None:
        parent = Node(key)
    theContent = pyparsing.Word(pyparsing.alphanums + ".+-=_><") | pyparsing.Suppress(',') | pyparsing.Suppress(
        '||') | pyparsing.Suppress('&&') | pyparsing.Suppress('!')
    parents = pyparsing.nestedExpr('(', ')', content=theContent)
    res = parents.parseString("(" + aliases[key] + ")")[0]
    for subExpression in res:
        if len(subExpression) == 0:
            continue
        for a in subExpression:
            if a in aliases:
                newNode = Node(a, parent=parent, content=aliases[a])
                makeAliasAnyTree(a, aliases, newNode, )
            else:
                Node(a, parent=parent)
    return parent


def getAliasAnyTree(base, regexp, **findOption):
    """
    :param base:        base node
    :param regexp:      regular expression
    :param findOption:   see https://anytree.readthedocs.io/en/latest/api/anytree.search.html
    :return: liast of aliases fulfilling
    """
    return [a.name for a in findall(base, filter_=lambda node: re.match(regexp, node.name), **findOption)]


def getTreeInfo(tree):
    """
    GetTree information description

    :param tree:        input tree
    :return:
        * dictionary with tree information
            * friends
            * aliases
            * metaTable
    """
    treeInfo = {'aliases': aliasToDictionary(tree)}
    friends = treeInfo['friends'] = {}
    for a in tree.GetListOfFriends():
        friends[a.GetName()] = a.GetTitle()
    metaTable = treeInfo['metaTable'] = {}
    #    if ROOT.TStatToolkit.GetMetadata(tree):
    #        table = ROOT.TStatToolkit.GetMetadata(tree)
    #        for a in table: metaTable[a.GetName()] = a.GetTitle()
    if GetMetadata(tree):
        table = GetMetadata(tree)
        for a in table: metaTable[a.GetName()] = a.GetTitle()
    return treeInfo


def __parseVariableList(parserOut, varList):
    """
    :param parserOut:
    :param varList:
    :return:
    """
    for a in parserOut:
        if type(a) == pyparsing.ParseResults:
            __parseVariableList(a, varList)
            continue
        try:
            float(a)
            continue
        except ValueError:
            pass
        varList.append(a)
    return varList


def parseTreeVariables(expression, counts=None, verbose=0):
    r"""
    ParseTreeExpression and fill flat list with tree variable needed for evaluation
        * Used in  getAndTestVariableList

    :param verbose:     verbosity
    :param expression: expression to parse
    :param counts:
    :return: dictionary with pairs variable:count
        :type counts: dict

    Example usage:
        >>> parseVariables("x>1 & x>0 | y==1 |x+1>2| (x2<2) | (x1*2)<2| sin(x)<1")
        ==>
        {'sin': 1, 'x': 4, 'x1': 1, 'x2': 1, 'y': 1}
    """
    if verbose:
        logging.info("expression", expression)
    if counts is None:
        counts = dict()
    varList = []
    theContent = pyparsing.Word(pyparsing.alphanums + "._") | pyparsing.Suppress(',') | pyparsing.Suppress(
        '|') | pyparsing.Suppress('&') | pyparsing.Suppress('!') \
                 | pyparsing.Suppress('>') | pyparsing.Suppress('=') | pyparsing.Suppress('+') | pyparsing.Suppress(
        '-') | pyparsing.Suppress('<') | pyparsing.Suppress('*') \
                 | pyparsing.Suppress('*') | pyparsing.Suppress(':')
    parents = pyparsing.nestedExpr('(', ')', content=theContent)
    try:
        res = parents.parseString("(" + expression + ")")
        __parseVariableList(res, varList)
    except:
        logging.error("Oops!  That was no valid number.  Try again...", expression)
    for i in varList:
        counts[i] = counts.get(i, 0) + 1
    return counts


def getAndTestVariableList(expressions, toRemove=None, toReplace=None, tree=None, verbose=0):
    r"""
    :param verbose:
    :type toReplace: list
    :type toRemove: list
    :param expressions:      - list of expressions
    :param toRemove:         - list of regular expression to be ignored
    :param toReplace:        - list of regular expression to be replaced
    :param tree:             - tree
    :return:
        list of the trivial expression to export
            * getAndTest variable list
            * decompose expression and extract the list of variables/branches/aliases  which should be extracted from trees

    Example usage:
        * see also test_aliTreePlayer.py:test_TreeParsing()
            >>> selection="meanMIP>0&resolutionMIP>0"
            >>> varDraw="meanMIP:meanMIPele:resolutionMIP:xxx"
            >>> widgets="tab.sliders(slider.meanMIP(45,55,0.1,45,55),slider.meanMIPele(50,80,0.2,50,80), slider.resolutionMIP(0,0.15,0.01,0,0.15)),"
            >>> widgets+="tab.checkboxGlobal(slider.global_Warning(0,1,1,0,1),checkbox.global_Outlier(0)),"
            >>> widgets+="tab.checkboxMIP(slider.MIPquality_Warning(0,1,1,0,1),checkbox.MIPquality_Outlier(0), checkbox.MIPquality_PhysAcc(1))"
            >>> toRemove=["^tab\..*"]
            >>> toReplace=["^slider.","^checkbox."]
            >>>
            >>> getAndTestVariableList([selection,varDraw,widgets],toRemove,toReplace)
            ==>
            ('Not existing tree variable', 'xxx')
            {'meanMIP': 1, 'global_Warning': 1, 'MIPquality_Outlier': 1, 'resolutionMIP': 1, 'MIPquality_Warning': 1, 'global_Outlier': 1, 'time': 1, 'meanMIPele': 1, 'MIPquality_PhysAcc': 1}
    Usage:
        general - but it is used for the bokeDhraw from tree to export variables to bokeh format
    """
    if toReplace is None:
        toReplace = []
    if toRemove is None:
        toRemove = []
    counts = dict()
    for expression in expressions:
        if type(expression) == str:
            parseTreeVariables(expression, counts, verbose)
    pop_list = []
    for mask in toRemove:
        for a in counts.keys():
            if re.findall(mask, a):
                # del (counts[a])
                pop_list.append(a)
    for x in pop_list:
        counts.pop(x)

    for mask in toReplace:
        pop_list = []
        for key in counts.keys():
            if re.findall(mask, key):
                # newKey = re.sub(mask, "", key)
                pop_list.append(key)
                # counts[newKey] = counts.pop(key)
        for x in pop_list:
            newKey = re.sub(mask, "", x)
            counts[newKey] = counts.pop(x)
    pop_list = []
    if tree:
        dictionary = treeToAnyTree(tree)
        for key in counts.keys():
            if findSelectedBranch(dictionary, key + "$") == ():
                logging.info("Not existing tree variable", key)
                pop_list.append(key)
                # del (counts[key])
    for x in pop_list:
        counts.pop(x)
    return counts


def tree2Panda(tree, include, selection, **kwargs):
    r"""
    Convert selected items from the tree into panda table
    TODO:
        * to  consult with uproot
            * currently not able to work with friend trees
        * check the latest version of RDeatFrame (in AliRoot latest v16.16.00)
        * Add filter on metadata - e.g class of variables
    :param tree:            input tree
    :param include:         regular expresion array - processing Tree+Friends, branches, aliases
    :param selection:       tree selection ()
    :param kwargs:
        * exclude           exclude arrray
        * firstEntry        firt entry to enter
        * nEntries          number of entries to convert
        * column mask
    :return:                panda data frame
    """
    options = {
        "exclude": [],
        "firstEntry": 0,
        "nEntries": 100000000,
        "columnMask": [[".fX$", "_X"], [".fY$", "_y"], [".fElements", ""]],
        "category":0,
        "verbose": 0,
        "estimate":-1
    }
    options.update(kwargs)
    if not hasattr(tree, 'anyTree'):
        treeToAnyTree(tree)          # expand tree/aliases/variables - if not done before
    anyTree = tree.anyTree
    # check regular expressions in anyTree
    variablesTree = findSelectedBranches(anyTree, include, options["exclude"])
    variables = ""

    for var in variablesTree:
        # if var.length<2: continue
        var = var.replace("/", ".")
        variables += var + ":"
    # check if valid TTree formula
    for var in include:
        if ".*" in var:
            continue
        formula=    ROOT.TTreeFormula('test', var, tree)
        if (formula.GetNdim()>0):
            variables += var + ":"
    variables = variables[0:-1]
    if options["estimate"]>0 & options["estimate"] > tree.GetEstimate():
        tree.SetEstimate(options["estimate"])
    estimate0=tree.GetEstimate()
    entries = tree.Draw(str(variables), selection, "goff", options["nEntries"], options["firstEntry"])  # query data
    if entries>estimate0:
        tree.SetEstimate(entries)
        entries = tree.Draw(str(variables), selection, "goff", options["nEntries"], options["firstEntry"])  # query data

    columns = variables.split(":")
    for i, column in enumerate(columns):
        columns[i] = column.replace(".", "_")
    # replace column names
    #    1.) pandas does not allow dots in names
    #    2.) user can specified own column mask
    for i, column in enumerate(columns):
        for mask in options["columnMask"]:
            columns[i] = columns[i].replace(mask[0], mask[1])
    if options["verbose"]&0xF>0 :
        for i, a in enumerate(columns):
            print(f"{i}\t{a}\t{tree.GetVal(i)}\t{tree.GetVal(i)[0]}\t{tree.GetVal(i)[2]}")

    ex_dict = {}
    for i, a in enumerate(columns):
        val = tree.GetVal(i)
        ex_dict[a] = np.frombuffer(val, dtype=float, count=entries) # potential fix for the python3.8+root
        #ex_dict[a] = np.frombuffer(np.ascontiguousarray(val), dtype=float, count=entries)  # not orking for python mismatch  as suggested at https://github.com/almarklein/pyelastix/issues/14
        # TODO - conversion not needed  - proper type to be used here
    df = pd.DataFrame(ex_dict, columns=columns)
    df = df.loc[:, ~df.columns.duplicated()]
    for i, a in enumerate(columns):
        if (tree.GetLeaf(a)):
            try:
              if (tree.GetLeaf(a).ClassName() == 'TLeafC'): df[a]=df[a].astype(np.int8)
              if (tree.GetLeaf(a).ClassName() == 'TLeafS'): df[a]=df[a].astype(np.int16)
              if (tree.GetLeaf(a).ClassName() == 'TLeafI'): df[a]=df[a].astype(np.int32)
              if (tree.GetLeaf(a).ClassName() == 'TLeafL'): df[a]=df[a].astype(np.int64)
              if (tree.GetLeaf(a).ClassName() == 'TLeafB'): df[a] = df[a].astype(bool)
              if (tree.GetLeaf(a).ClassName() == 'TLeafF'): df[a] = df[a].astype(np.float32)
            except:
                print(f"invalid branch content branch: {a} branch type: {tree.GetLeaf(a).ClassName()}")
        if (options["category"]>0):
            dfUniq=df[a].unique()
            if dfUniq.shape[0]<=options["category"] :
                df[a]=df[a].astype(CategoricalDtype(ordered=True))



    initMetadata(df)
    metaData = tree.GetUserInfo().FindObject("metaTable")
    if metaData:
        for key in metaData:
            df.meta.metaData[key.GetName()] = key.GetTitle()
    return df


def AddMetadata(tree, varTagName, varTagValue):
    if not tree:
        return None
    metaData = tree.GetUserInfo().FindObject("metaTable")
    if not metaData:
        metaData = ROOT.THashList()
        metaData.SetName("metaTable")
        tree.GetUserInfo().AddLast(metaData)
    if not (varTagName is None or varTagValue is None):
        named = GetMetadata(tree, varTagName, None, True)
        if not named:
            metaData.AddLast(ROOT.TNamed(varTagName, varTagValue))
        else:
            named.SetTitle(varTagValue)
    return metaData


def GetMetadata(tree, *args):
    '''
        Usage:
            GetMetadata(TTree)
            GetMetadata(TTree, varTagName)
            GetMetadata(TTree, varTagName, prefix, fullMatch)
    '''
    if not tree:
        return None
    treeMeta = tree
    metaData = treeMeta.GetUserInfo().FindObject("metaTable")
    if not metaData:
        metaData = ROOT.THashList()
        metaData.SetName("metaTable")
        tree.GetUserInfo().AddLast(metaData)
        return 0
    if len(args) == 0:
        return metaData
    elif len(args) == 1:
        prefix = ""
        fullMatch = False
    elif len(args) == 3:
        prefix = args[1]
        fullMatch = args[2]
    else:
        raise TypeError("Invalid number of arguments")
    named = metaData.FindObject(args[0])
    if named or fullMatch:
        return named
    metaName = varTagName
    nDots = metaName.count('.')
    prefix = ""
    while nDots > 1:
        fList = tree.GetListOfFriends()
        if fList:
            for kf in range(fList.GetEntries()):
                iName = fList.At(kf).GetName()
                regFriend = "^" + iName + '.'
                if re.match(regFriend, metaName):
                    treeMeta = treeMeta.GetFriend(iName)
                    re.sub(regFriend, "", metaName)
            prefix += iName
        if (nDots == metaName.count('.')):
            break
        nDots = metaName.count('.')
    named = metaData.FindObject(metaName)
    return named


def LoadTrees(inputDataList, chRegExp, chNotReg, inputFileSelection, verbose):
    regExp = ROOT.TPRegexp(chRegExp)
    notReg = ROOT.TPRegexp(chNotReg)
    regExpArray = inputFileSelection.split(':')
    residualMapList = ROOT.gSystem.GetFromPipe(inputDataList).Tokenize('\n')
    nFiles = residualMapList.GetEntries()
    treeBase = ROOT.TTree()
    tagValue = {}
    nSelected = 0
    treeBaseList = []
    fileList = []
    #for iFile in range(nFiles):
    iFile=0
    while iFile<nFiles :
        tagValue.clear()
        tagValue["Title"] = ""
        #      read metadata #tag:value information - signed by # at the beginning of the line
        for jFile in range(iFile, nFiles):  # get rid of the loop using continue
            name = residualMapList.At(jFile).GetName()
            if name[0] is '#':
                first = name.find(":")
                tag = name[1:name.find(":")]
                value = name[name.find(":") + 1:len(name)]
                tagValue[tag] = value
            else:
                iFile = jFile
                break
        fileName = residualMapList.At(iFile).GetName()
        isSelected = True
        #         check if the file was selected
        if len(regExpArray) > 0:
            isSelected = False
            for entry in range(len(regExpArray)):
                reg = ROOT.TPRegexp(regExpArray[entry])
                if reg.Match(fileName):
                    isSelected = True
        if not isSelected:
            continue
        logging.info("<LoadTrees>: Load file\t%s", fileName)
        description = ""
        option = ""
        if "http" in fileName:
            option = "cacheread"
        finput = ROOT.TFile.Open(fileName, option)
        fileList.append(finput)
        if finput is None:
            logging.error("<MakeResidualDistortionReport>: Invalid file name {}".format(fileName))
            #            logging.error("<MakeResidualDistortionReport>: Invalid file name %s", fileName)
            return None,None,None
        try:
            keys = finput.GetListOfKeys()
        except:
            logging.error("<MakeResidualDistortionReport>: Invalid file name or access rights {}".format(fileName))
            return None,None,None
        isLegend = False
        for iKey in range(keys.GetEntries()):
            if regExp.Match(keys.At(iKey).GetName()) is 0:
                continue  # is selected
            if notReg.Match(keys.At(iKey).GetName()) is not 0:
                continue  # is rejected
            tree = finput.Get(keys.At(iKey).GetName())  # better to use dynamic cast
            treeBaseList.append(tree)
            if treeBase.GetEntries() is 0:
                finput2 = ROOT.TFile.Open(fileName, option)
                fileList.append(finput2)
                treeBase = finput2.Get(keys.At(iKey).GetName())
                treeBaseList.append(treeBase)
            fileTitle = tagValue["Title"]
            if len(fileTitle):
                treeBase.AddFriend(tree, "{}.{}".format(fileTitle, keys.At(iKey).GetName()))
            else:
                treeBase.AddFriend(tree, "{}".format(keys.At(iKey).GetName()))
            entriesF = tree.GetEntries()
            entriesB = treeBase.GetEntries()
            if entriesB == entriesF:
                if (verbose > 0):
                    logging.info("InitMapTree: %s %s.%s: %d\t%d", treeBase.GetName(), fileName, keys.At(iKey).GetName(),
                                 entriesB, entriesF)
            else:
                raise ValueError(
                    "InitMapTree: {} {} . {}:  {}  {}".format(treeBase.GetName(), fileName, keys.At(iKey).GetName(),
                                                              entriesB, entriesF))
    #                logging.error("InitMapTree", "%s\t%s.%s:\t%d\t%d", treeBase.GetName(), fileName, keys.At(iKey).GetName(), entriesB, entriesF)
        iFile+=1
    return treeBase, treeBaseList, fileList

def makeABCD(nPoints=10000):
    with uproot.recreate("ABDC.root") as f:
        f["ABCD"] = uproot.newtree({"A": "float32", "B": "float32", "C": "float32", "D": "float32"})
        for i in range(5):
            f["ABCD"].extend({"A": np.random.normal(0, 1, nPoints), "B":np.random.normal(0, 1, nPoints),
                           "C": np.random.normal(0, 1, nPoints), "D": np.random.normal(0, 1, nPoints)})
    f = ROOT.TFile("ABCD.root")
    tree = ROOT.TTree(f.Get("ABCD"))
    tree.SetAlias("bigA","A>0.5")
    tree.SetAlias("smallA","A<0.5")
    return tree, f

def pandaToTree(df,name=None):
    if name is None:
        name="my_ttree"
        if hasattr(df, "meta"):
            if hasattr(df.meta, "metaData"):            
                name=df.meta.metaData.get("Name",name)
    df.to_root("./."+name+".root",name)
    f = ROOT.TFile("./."+name+".root")
    tree = f.Get(name)
    return tree, f


