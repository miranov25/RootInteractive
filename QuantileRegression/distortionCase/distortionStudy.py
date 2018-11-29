import pandas as pd
from typing import List, Any, Union


def readDataFrame2(fName):
    """
    :param fName:
    :return:
    """
    line = open(fName).readline().rstrip()  # get header - using root cvs convention
    names = line.replace("/D", "").replace("/I", "").split(":")
    variables = []
    for a in names: variables.append(a.split('\t')[0])  #
    dataFrame = pd.read_csv(fName, sep='\t', index_col=False, names=variables, skiprows=1)
    return dataFrame


def splitDistortionFrame(df):
    """
    reshuffle distortion frame
    :param df:  input data frame with distortions
    :return:
    """
    sectors = [2, 4, 6, 7, 9, 16, 20, 30]
    dfAll = df.query("isec==2&isIROC").reset_index(drop=True)  # type: object
    pandaList = [dfAll]  # type: List[Union[object, pandaList]]
    for isector in sectors:
        global pandaList
        query = "isIROC&isec==" + str(isector)
        dfsec = df.query(query)[['drphiSmoothedQ95']].reset_index(drop=True)
        dfsec.columns = ["drphiSector" + str(isector)]
        pandaList.append(dfsec)
    pall = pd.concat(pandaList, axis=1)
    pall.dropna(inplace=True) # skip non full rows
    return pall
