import pandas as pd
import numpy as np


def treeToPanda(tree, variables, selection, nEntries, firstEntry, columnMask='default'):
    """
    convert selected items from the tree into panda table
    TODO - import fail (infinite loop) - to check the reason
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
