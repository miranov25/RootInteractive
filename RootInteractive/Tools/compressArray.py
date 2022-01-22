import base64
import pickle
import zlib
from bokeh.util.serialization import *
import numpy as np
import pandas as pdMap
import sys
import re
import collections

arrayCompressionRelative8=[(".*",[("relative",8), ("code",0), ("zip",0), ("base64",0)])]
arrayCompressionRelative16=[(".*",[("relative",16), ("code",0), ("zip",0), ("base64",0)])]
arrayCompressionRelative32=[(".*",[("relative",32), ("code",0), ("zip",0), ("base64",0)])]

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
    type=df.dtype
    # If dtype is not floating point number or int, skip this step
    if type.kind not in ['f', 'c', 'i', 'u']:
        return df
    shiftN = 2 ** nBits
    mantissa, exp2 = np.frexp(df)
    mantissa = np.rint(mantissa * shiftN)/shiftN
    result=(mantissa * 2 ** exp2.astype(float)).astype(type)
    return result


def roundAbsolute(df, delta):
    type=df.dtype
    if type.kind not in ['f', 'c', 'i', 'u']:
        return df
    result = np.rint(df / delta) * delta
    deltaMean = (df - result).mean()
    result -= deltaMean
    return result.astype(type)


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


def compressArray0(inputArray, maxFraction=0.5, doZip=True, doBase64=True, keepOrig=False, nBitsRelative=None,
                   deltaAbsolute=None):
    """
    compress array  - works for panda - to be tested for vaex
    :param inputArray:
    :param maxFraction:
    :param doZip:
    :param doBase64:
    :param keepOrig:
    :param nBitsRelative:  scalar or array specifying relative precision 5 bits  or np.round(1+np.log2(1+df["weight"])
    :return:
    """
    arrayC = {}
    if keepOrig:
        arrayC["inputArray"] = inputArray
    if nBitsRelative is not None:
        inputArray = roundRelativeBinary(inputArray, nBitsRelative)
    if deltaAbsolute is not None:
        inputArray = roundAbsolute(inputArray, nBitsRelative)
    values = inputArray.unique()
    arrayC["isZip"] = False
    arrayC["indexType"] = ""
    if values.size < maxFraction * inputArray.shape[0]:
        dictValues = {}
        dictValuesI = {}
        for i, value in enumerate(values):
            dictValues[value] = i
            dictValuesI[i] = value
            arrayC["values"] = dictValuesI
        if values.size < 255:
            arrayC["index"] = inputArray.map(dictValues).astype("int8")
            arrayC["indexType"] = "int8"
        else:
            arrayC["index"] = inputArray.map(dictValues).astype("int16")
            arrayC["indexType"] = "int16"
        if doZip:
            arrayC["indexC"] = zlib.compress(arrayC["index"].to_numpy())
            arrayC["isZip"] = True
        if doBase64:
            arrayC["indexC"] = base64.b64encode(arrayC["indexC"])
        if not keepOrig:
            arrayC.pop('index', None)

    return arrayC


def compressArray(inputArray, actionArray, keepValues=False):
    arrayInfo = {"actionArray": actionArray, "history": []}
    currentArray = inputArray
    counter=0
    for action, actionParam in actionArray:
       # try:
            if keepValues:
                arrayInfo["history"].append(currentArray)
            if action == "relative":
                currentArray = roundRelativeBinary(currentArray, actionParam)
            if action == "delta":
                currentArray = roundAbsolute(currentArray, actionParam)
            if action == "zip":
                arrayInfo["dtype"] = currentArray.to_numpy().dtype.name
                currentArray = zlib.compress(currentArray.to_numpy())
            if action == "unzip":
                currentArray = np.frombuffer(zlib.decompress(currentArray),dtype=arrayInfo["dtype"])
            if action == "base64":
                currentArray = base64.b64encode(currentArray).decode("utf-8")
            if action == "debase64":
                currentArray = base64.b64decode(currentArray)
            if action == "code":
                # Skip for normal number types, these can be unpacked in an easier way.
                # Do not send int64 arrays to the client, it will not work
                if currentArray.dtype.kind not in ['O', 'S', 'U']:
                    arrayInfo["skipCode"] = True
                    continue
                arrayInfo["skipCode"] = False
                values = currentArray.unique()
                dictValues = {}
                dictValuesI = {}
                for i, value in enumerate(values):
                    dictValues[value] = i
                    dictValuesI[i] = value
                    arrayInfo["valueCode"] = dictValuesI
                if values.size < 2 ** 8:
                    currentArray = currentArray.map(dictValues).astype("int8")
                elif values.size < 2 ** 16:
                    currentArray = currentArray.map(dictValues).astype("int16")
                else:
                    currentArray = currentArray.map(dictValues).astype("int32")
            if action == "decode":
                if not arrayInfo["skipCode"]:
                    arrayAsPanda=pdMap.Series(currentArray)      # TODO - numpy does not have map function better solution to fine
                    currentArray = arrayAsPanda.map(arrayInfo["valueCode"])
            counter+=1
       # except:
        #    print("compressArray - Unexpected error in ", action,  sys.exc_info()[0])
            #pass
    arrayInfo["byteorder"] = sys.byteorder
    arrayInfo["array"] = currentArray
    return arrayInfo


def compressCDSPipe(df, arrayCompression, verbosity, columnsSelect=None):
    """
    compress CDSPipe - based on the arrayCompression
    :param df:                   input map of arrays (DF)
    :param arrayCompression:     compression descriptions array of pairs  (regular expression, array compression description)
    :param verbosity
    :param  columnsSelect        columsn which will be used to compress - in None all colusms from df are used
    :return:                     map
    Example array description
    actionArrayDelta=[("delta",0.01), ("code",0), ("zip",0), ("base64",0)]
    actionArrayRel=[("relative",8), ("code",0), ("zip",0), ("base64",0)]
    actionArrayRel4=[("relative",4), ("code",0), ("zip",0), ("base64",0)]
    arrayCompression=[ (".*Center",actionArrayDelta), (".*MeanD",actionArrayRel4),(".*",actionArrayRel)]
    """
    outputMap = {}
    sizeMap = {}
    sizeInAll = 0
    sizeOutAll = 0
    counter = 0
    for col in df:
        counter += 1
        if columnsSelect is not None:
            if col not in columnsSelect:
                continue
        for action in arrayCompression:
            if re.match(action[0], col) == None:
                continue
            arrayC = compressArray(df[col], action[1], False)
            sizeIn = getSize(df[col])
            sizeOut = getSize(arrayC)
            sizeOutAll += sizeOut
            sizeInAll += sizeIn
            sizeMap[col] = [sizeOut, sizeIn, sizeOut / sizeIn, counter]
            if verbosity > 0:
                print("Compress", counter, col, action[0], action[1])
                print("Compress factor", sizeOut, sizeIn, sizeOut / sizeIn, counter, col)
            outputMap[col] = arrayC
            break
    sizeMap = {k: sizeMap[k] for k in sorted(sizeMap, key=sizeMap.get, reverse=True)}
    sizeMap["_all"] = [sizeOutAll, sizeInAll, sizeOutAll / sizeInAll, counter]

    if verbosity > 0:
        print("Compress", "_all", sizeOutAll, sizeInAll, sizeOutAll / sizeInAll, counter)
    return outputMap, sizeMap


def codeCDS(df, doZip=0, printSize=0):
    mapIndex, mapCodeI = codeMapDF(df, 0.5, printSize)
    data = {}
    data["indexMap0"] = mapIndex  # original index
    data["indexMap1"] = transform_column_source_data(mapIndex)  # transform index
    if doZip:
        for key, value in mapIndex.items():
            mapIndex[key] = base64.b64encode(zlib.compress(value.to_numpy().tobytes())).decode("utf-8")
    for key, value in mapIndex.items():
        data[key] = value
    data["mapCodeI"] = mapCodeI
    data["indexMap2"] = transform_column_source_data(mapIndex)  # compressed index
    return data


def decodeCDS(data):
    return 0
