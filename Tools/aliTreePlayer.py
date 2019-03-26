import pandas as pd
import numpy as np
import urllib2 as urllib2
import pyparsing
from anytree import *
import ROOT
import re


def readDataFrameURL(fName, nrows=0):
    if 'http' in fName:
        line = urllib2.urlopen(fName).readline().rstrip()  # get header - using root cvs convention
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


def aliasToDictionary(tree):
    aliases = {}
    for a in tree.GetListOfAliases(): aliases[a.GetName()] = a.GetTitle()
    return aliases


def processAnyTreeBranch(branch0, parent):
    for branch in branch0.GetListOfBranches():
        processAnyTreeBranch(branch, parent)


def treeToAnyTree(tree):
    """
    treeToAntTree representation
    :param tree:  input TTree
    :return:  parent node of the anytree object
    Example usage:
        branchTree=getListOfBranches(treeQA)
        print(findall(branchTree, filter_=lambda node: re.match("bz", node.name)))
        print(findall(branchTree, filter_=lambda node: re.match("MIP.*Warning$", node.name)))
        (Node('/tree/bz'),)
        (Node('/tree/MIPattachSlopeA_Warning'), Node('/tree/MIPattachSlopeC_Warning'), Node('/tree/MIPattachSlope_comb2_Warning'), Node('/tree/MIPquality_Warning'))
    """
    parent = Node("tree")
    for branch in tree.GetListOfBranches():
        branchT = Node(branch.GetName(), parent)
        processAnyTreeBranch(branch, branchT)
    for alias in tree.GetListOfAliases():
        Node(alias.GetName(), parent)
    for friend in tree.GetListOfFriends():
        treeF = tree.GetFriend(friend.GetName())
        nodeF = Node(friend.GetName(), parent)
        for branch in treeF.GetListOfBranches():
            processAnyTreeBranch(branch, nodeF)
    return parent


def findSelectedBranch(anyTree, regexp):
    """
    return array of selected branches
    :param anyTree:
    :param regexp:
    :return:
    """
    return findall(anyTree, filter_=lambda nodeF: re.match(regexp, nodeF.name))


def makeAliasAnyTree(key, parent, aliases):
    """
    build recursive alias anytree
    :param key:          - start key
    :param parent:       - initial node
    :param aliases:      - alias dictionary
    :return:               anytree object
    """
    theContent = pyparsing.Word(pyparsing.alphanums + ".+-=_><") | pyparsing.Suppress(',') | pyparsing.Suppress('||') | pyparsing.Suppress('&&') | pyparsing.Suppress('!')
    parents = pyparsing.nestedExpr('(', ')', content=theContent)
    res = parents.parseString("(" + aliases[key] + ")")[0]
    for subExpression in res:
        if len(subExpression) == 0: continue
        for a in subExpression:
            if a in aliases:
                newNode = Node(a, parent=parent, content=aliases[a])
                makeAliasAnyTree(a, newNode, aliases)
            else:
                Node(a, parent=parent)


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
