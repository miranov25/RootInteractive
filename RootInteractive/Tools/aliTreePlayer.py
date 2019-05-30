import pandas as pd
import numpy as np
import urllib.request as urlopen
import pyparsing
from anytree import *
import ROOT
import re


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
    """
    convert selected items from the tree into panda table
    TODO - import fail in case of number of entries>2x10^6  (infinite loop) - to check the reason
    :param tree:            input tree
    :param variables:
    :param selection:
    :param nEntries:
    :param firstEntry:
    :param columnMask:
    :return:
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
    #    print(i, column)
    # print(columns)
    ex_dict = {}
    for i, a in enumerate(columns):
        # print(i,a)
        val = tree.GetVal(i)
        ex_dict[a] = np.frombuffer(val, dtype=float, count=entries)
    df = pd.DataFrame(ex_dict, columns=columns)
    return df


#  Dictionary processing

def aliasToDictionary(tree):
    aliases = {}
    for a in tree.GetListOfAliases(): aliases[a.GetName()] = a.GetTitle()
    return aliases


def __processAnyTreeBranch(branch0, parent):
    for branch in branch0.GetListOfBranches():
        __processAnyTreeBranch(branch, parent)


def treeToAnyTree(tree):
    """
    treeToAntTree representation
    :param tree:  input TTree
    :return:  parent node of the anytree object
    Example usage: see test_aliTreePlayer.py::test_AnyTree()
        branchTree=treeToAnyTree(treeQA)
        print(findall(branchTree, filter_=lambda node: re.match("bz", node.name)))
        print(findall(branchTree, filter_=lambda node: re.match("MIP.*Warning$", node.name)))
        (Node('/tree/bz'),)
        (Node('/tree/MIPattachSlopeA_Warning'), Node('/tree/MIPattachSlopeC_Warning'), Node('/tree/MIPattachSlope_comb2_Warning'), Node('/tree/MIPquality_Warning'))
    """
    parent = Node("tree")
    for branch in tree.GetListOfBranches():
        branchT = Node(branch.GetName(), parent)
        __processAnyTreeBranch(branch, branchT)
    if tree.GetListOfAliases():
        for alias in tree.GetListOfAliases():
            Node(alias.GetName(), parent)
    for friend in tree.GetListOfFriends():
        treeF = tree.GetFriend(friend.GetName())
        nodeF = Node(friend.GetName(), parent)
        for branch in treeF.GetListOfBranches():
            __processAnyTreeBranch(branch, nodeF)
    return parent


def findSelectedBranch(anyTree, regexp, **findOption):
    """
    return array of selected branches
    :param anyTree:
    :param regexp:
    :param findOption   -   from list = stop=None, maxlevel=None, mincount=None, maxcount=None
    :return: selected anyTree branches
    Example usage: test_aliTreePlayer.py::test_AnyTree()
    branchTree = treeToAnyTree(tree)
    print(findSelectedBranch(branchTree, "MIP.*Warning"))
    ==> (Node('/tree/MIPattachSlopeA_Warning'), Node('/tree/MIPattachSlopeC_Warning'), Node('/tree/MIPattachSlope_comb2_Warning'), Node('/tree/MIPquality_Warning'))
    """
    return findall(anyTree, filter_=lambda nodeF: re.match(regexp, nodeF.name), **findOption)


def makeAliasAnyTree(key, aliases, parent=None):
    """
    build recursive alias anytree
    :param key:          - start key
    :param parent:       - initial node
    :param aliases:      - alias dictionary
    :return:               anytree object
    """
    if (parent == None):
        parent = Node(key)
    theContent = pyparsing.Word(pyparsing.alphanums + ".+-=_><") | pyparsing.Suppress(',') | pyparsing.Suppress('||') | pyparsing.Suppress('&&') | pyparsing.Suppress('!')
    parents = pyparsing.nestedExpr('(', ')', content=theContent)
    res = parents.parseString("(" + aliases[key] + ")")[0]
    for subExpression in res:
        if len(subExpression) == 0: continue
        for a in subExpression:
            if a in aliases:
                newNode = Node(a, parent=parent, content=aliases[a])
                makeAliasAnyTree(a, aliases, newNode,)
            else:
                Node(a, parent=parent)
    return parent


def getAliasAnyTreee(base, regexp, **findOption):
    return [a.name for a in findall(base, filter_=lambda node: re.match(regexp, node.name), **findOption)]


def getTreeInfo(tree):
    """
    GetTree information description
    :param tree: input tree
    :return: dictionary with all branches,
    """
    treeInfo = {'aliases': aliasToDictionary(tree)}
    friends = treeInfo['friends'] = {}
    for a in tree.GetListOfFriends(): friends[a.GetName()] = a.GetTitle()
    metaTable = treeInfo['metaTable'] = {}
    if ROOT.TStatToolkit.GetMetadata(tree):
        table = ROOT.TStatToolkit.GetMetadata(tree)
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
    """
        parseTreeExpression and fill flat list with tree variable needed for evaluation
    Used in  getAndTestVariableList
    :param verbose:     verbosity
    :param expression:  expression to parse e.g. expr="x>1 & x>0 | y==1 |x+1>2| (x2<2) | (x1*2)<2| sin(x)<1"
    :param counts:
    :return:
        :type counts: dict
    Example usage:
        :parseVariables("x>1 & x>0 | y==1 |x+1>2| (x2<2) | (x1*2)<2| sin(x)<1")
         ==>
        {'sin': 1, 'x': 4, 'x1': 1, 'x2': 1, 'y': 1}
    """
    if verbose: print("expression", expression)
    if counts is None:
        counts = dict()
    varList = []
    theContent = pyparsing.Word(pyparsing.alphanums + "._") | pyparsing.Suppress(',') | pyparsing.Suppress('|') | pyparsing.Suppress('&') | pyparsing.Suppress('!') \
                 | pyparsing.Suppress('>') | pyparsing.Suppress('=') | pyparsing.Suppress('+') | pyparsing.Suppress('-') | pyparsing.Suppress('<') | pyparsing.Suppress('*') \
                 | pyparsing.Suppress('*') | pyparsing.Suppress(':')
    parents = pyparsing.nestedExpr('(', ')', content=theContent)
    res = parents.parseString("(" + expression + ")")
    __parseVariableList(res, varList)
    for i in varList:
        counts[i] = counts.get(i, 0) + 1
    return counts


def getAndTestVariableList(expressions, toRemove=None, toReplace=None, tree=None, verbose=0):
    """
    getAndTest variable list - decompose expression and extract the list of variables/branches/aliases  which should be extracted from trees
    :param verbose:
    :type toReplace: list
    :type toRemove: list
    :param expressions:      - list of expressions
    :param toRemove:         - list of regular expression to be ignored
    :param toReplace:        - list of regular expression to be replaced
    :param tree:             - tree
    :return:                 - list of the trivial expression to export
    Example: - see also test_aliTreePlayer.py:test_TreeParsing():
        selection="meanMIP>0&resolutionMIP>0"
        varDraw="meanMIP:meanMIPele:resolutionMIP:xxx"
        widgets="tab.sliders(slider.meanMIP(45,55,0.1,45,55),slider.meanMIPele(50,80,0.2,50,80), slider.resolutionMIP(0,0.15,0.01,0,0.15)),"
        widgets+="tab.checkboxGlobal(slider.global_Warning(0,1,1,0,1),checkbox.global_Outlier(0)),"
        widgets+="tab.checkboxMIP(slider.MIPquality_Warning(0,1,1,0,1),checkbox.MIPquality_Outlier(0), checkbox.MIPquality_PhysAcc(1))"
        toRemove=["^tab\..*"]
        toReplace=["^slider.","^checkbox."]
        #
        getAndTestVariableList([selection,varDraw,widgets],toRemove,toReplace)
        ==>
        ('Not existing tree variable', 'xxx')
        {'meanMIP': 1, 'global_Warning': 1, 'MIPquality_Outlier': 1, 'resolutionMIP': 1, 'MIPquality_Warning': 1, 'global_Outlier': 1, 'time': 1, 'meanMIPele': 1, 'MIPquality_PhysAcc': 1}
    Usage: general - but it is used for the bokeDraw from tree to export varaibles to bokeh format

    """
    if toReplace is None:
        toReplace = []
    if toRemove is None:
        toRemove = []
    counts = dict()
    for expression in expressions:
        parseTreeVariables(expression, counts, verbose)
    pop_list = []
    for mask in toRemove:
        for a in counts.keys():
            if re.findall(mask, a):
                #del (counts[a])
                pop_list.append(a)
    for x in pop_list:
        counts.pop(x)
    
    for mask in toReplace:
        pop_list = []
        for key in counts.keys():
            if re.findall(mask, key):
                #newKey = re.sub(mask, "", key)
                pop_list.append(key)
                #counts[newKey] = counts.pop(key)
        for x in pop_list:
            newKey = re.sub(mask,"",x)
            counts[newKey] = counts.pop(x)
    pop_list = []
    if tree:
        dictionary = treeToAnyTree(tree)
        for key in counts.keys():
            if findSelectedBranch(dictionary, key + "$") == ():
                print("Not existing tree variable", key)
                pop_list.append(key)
                #del (counts[key])
    for x in pop_list:
        counts.pop(x)
    return counts
