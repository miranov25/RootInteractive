import pandas as pd
import numpy as np
import urllib2 as urllib2


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
