import pandas as pd

def loadCSVTreePanda(input,separator='\t'):
    """
    :param input:       path to input root csv file
    :param separator:   separator - default is tabulator
    :return:            panda dataframe
    """
    with open(input) as f:
        csv_header = f.readline()
        csv_header = csv_header.replace("/D:",":").replace("/I:",":").split(":")
        df1 = pd.read_csv(input,sep=separator,names=csv_header,skiprows=1, index_col=0)
    return df1


