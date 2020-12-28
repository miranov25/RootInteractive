import zlib
import pickle


def getCompressionSize(inputObject):
    return len(pickle.dumps(zlib.compress(inputObject)))


def getSize(inputObject):
    return len(pickle.dumps(inputObject))


def codeMapDF(df, maxFraction=0.5):
    """
    Compress panda data frame using remapping to arrays
    :param df:              input data frame
    :param maxFraction:     maximal fraction of distinct points
    :return:
    """
    mapIndex = {}
    mapCodeI = {}
    for column in df.columns:
        values = df[column].unique()
        if values.size < maxFraction * df[column].size:
            dictValues = {}
            dictValuesI = {}
            for i, value in enumerate(values):
                dictValues[value] = i
                dictValuesI[i] = value
            mapCodeI[column] = dictValuesI
            if values.size < 255:
                mapIndex[column] = df[column].map(dictValues).astype("int8")
            else:
                mapIndex[column] = df[column].map(dictValues).astype("int16")
            dfSize = getCompressionSize(mapIndex[column].to_numpy())
            print(column, values.size, dfSize)
        else:
            mapIndex[column] = df[column]
    return mapIndex, mapCodeI
