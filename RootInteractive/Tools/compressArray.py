import base64
import pickle
import zlib

import numpy as np


def getCompressionSize(inputObject):
    return len(pickle.dumps(zlib.compress(inputObject)))


def getSize(inputObject):
    return len(pickle.dumps(inputObject))


def roundRelativeBinary(df, nBits):
    """ TODO - check more optimal implementation using shifts ... and pipes
    roundRelativeBinary     - round mantissa of float number in nBits, assuming better lossy compression later
    :param df:              - input array (for a moment only pandas or numpy)
    :param nBits:           - number of significant bits to round
    :return:                - rounded array
    """
    shiftN = 2 ** nBits
    mantissa, exp2 = np.frexp(df)
    mantissa = (mantissa * shiftN).astype("int")
    mantissa /= shiftN
    return mantissa * 2 ** exp2


def codeMapDF(df, maxFraction=0.5, doPrint=0):
    """
    Compress data frame using remapping to arrays (working for  panda and vaex)
    :param df:              input data frame
    :param maxFraction:     maximal fraction of distinct points
    :param doPrint:         print flag
    :return:                mapIndex and mapCodeI

    """
    mapIndex = {}
    mapCodeI = {}
    for column in df.columns:
        values = df[column].unique()
        if values.size < maxFraction * df[column].shape[0]:
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
            if doPrint:
                dfSizeC = getCompressionSize(mapIndex[column].to_numpy())
                dfSize0 = getSize(mapIndex[column].to_numpy())
                print(column, values.size, dfSize0, dfSizeC, dfSizeC / float(dfSize0))
        else:
            mapIndex[column] = df[column]
    return mapIndex, mapCodeI


def codeCDS(df, doZip=0, printSize=0):
    mapIndex, mapCodeI = codeMapDF(df, 0.5, printSize)
    data = {}
    if doZip:
        for key, value in mapIndex.items():
            mapIndex[key] = base64.b64encode(zlib.compress(value.to_numpy())).decode("utf-8")
    for key, value in mapIndex.items():
        data[key] = value
    data["mapCodeI"] = mapCodeI
    return data
