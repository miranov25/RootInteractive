import base64
import pickle
import zlib
from bokeh.util.serialization import *
import numpy as np
import pandas as pd
import sys
import re

arrayCompressionRelative8=[(".*",[("relative",8), "code", "zip"])]
arrayCompressionRelative16=[(".*",[("relative",16), "code", "zip"])]
arrayCompressionRelative32=[(".*",[("relative",32), "code", "zip"])]

def getCompressionSize(inputObject):
    return len(pickle.dumps(zlib.compress(inputObject)))


def getSize(inputObject):
    return len(pickle.dumps(inputObject))


def roundRelativeBinary(df, nBits, eps=0., downgradeType = True):
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
    # If result can be represented by single precision float, use that, otherwise cast to double
    if downgradeType and type.kind == 'f' and nBits <= 23 and np.min(exp2) > -256 and np.max(exp2) < 255:
        return np.ldexp(mantissa.astype(np.float32), exp2).astype(np.float32)
    return np.ldexp(mantissa, exp2).astype(type)


def removeInt64(column):
    type = column.dtype
    if type.kind in ['i', 'u'] and type.itemsize == 8:
        maxvalue = column.max()
        minvalue = column.min()
        if maxvalue > 0x7FFFFFFF or minvalue < -0x80000000:
            return column.astype("float64")
        return column.astype("int32")
    return column


def roundAbsolute(df, delta, downgrade_type=True):
    # This should probably also downgrade the type if safe to do so instead of upgrading back
    type=df.dtype
    if delta == 0:
        raise ZeroDivisionError(df, delta)
    if type.kind not in ['f', 'c', 'i', 'u']:
        return df
    if type.kind in ['i','u'] and delta == 1:
        # delta == 1 for integer means no change
        return df
    quantized = np.rint(df / delta)
    result = quantized * delta
    deltaMean = (df - result).mean()
    if downgrade_type:
        dfMin = np.nanmin(quantized)
        quantized = np.where(np.isnan(df), -1, quantized-dfMin)
        rangeSize = np.max(quantized)
        if rangeSize <= 0x7f:
            return quantized.astype(np.int8), {"scale":delta, "origin":dfMin*delta-deltaMean}
        if rangeSize <= 0x7fff:
            return quantized.astype(np.int16), {"scale":delta, "origin":dfMin*delta-deltaMean}
        return quantized.astype(np.int32), {"scale":delta, "origin":dfMin*delta-deltaMean}
    result -= deltaMean
    return result.astype(type), None


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


def compressArray(inputArray, actionArray, keepValues=False, verbosity=0):
    arrayInfo = {"actionArray": actionArray.copy(), "history": []}
    currentArray = inputArray
    counter=0
    for actionTuple in actionArray:
        if isinstance(actionTuple, tuple):
            action = actionTuple[0]
            actionParams = actionTuple[1:] if len(actionTuple) > 1 else None
        else:
            action = actionTuple
            actionParams = []
        if keepValues:
            arrayInfo["history"].append(currentArray)
        if action == "relative":
            currentArray = roundRelativeBinary(currentArray, *actionParams)
        if action == "delta":
            currentArray, decode_transform = roundAbsolute(currentArray, *actionParams)
            if decode_transform is not None:
                arrayInfo["history"].append(("linear", decode_transform))
        if action == "zip":
            if isinstance(currentArray, pd.Series):
                currentArray = currentArray.to_numpy()
            arrayInfo["dtype"] = currentArray.dtype.name
            arrayInfo["history"].append(("array", currentArray.dtype.name))
            currentArray = zlib.compress(currentArray)
            arrayInfo["history"].append("inflate")
        if action == "unzip":
            currentArray = np.frombuffer(zlib.decompress(currentArray),dtype=arrayInfo["dtype"])
        if action == "removeInt64":
            currentArray = removeInt64(currentArray)
        if action == "base64":
            currentArray = base64.b64encode(currentArray).decode("utf-8")
            arrayInfo["history"].append("base64_decode")
        if action == "base64_decode":
            currentArray = base64.b64decode(currentArray)
        if action == "code":
            # Skip for normal number types, these can be unpacked in an easier way.
            # Do not send int64 arrays to the client, it will not work
            if currentArray.dtype.kind not in ['O', 'S', 'U']:
                arrayInfo["skipCode"] = True
                continue
            arrayInfo["history"].append("code")
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
                arrayAsPanda=pd.Series(currentArray)      # TODO - numpy does not have map function better solution to fine
                currentArray = arrayAsPanda.map(arrayInfo["valueCode"])
        if action == "astype":
            currentArray = currentArray.astype(actionParams[0])
        if verbosity & 2:
            print(actionTuple)
            print(len(currentArray))
        counter+=1
    arrayInfo["byteorder"] = sys.byteorder
    arrayInfo["array"] = currentArray
    return arrayInfo


def frombuffer(inputArray, step_info):
    #TODO: Add logic for numpy
    return inputArray


def tobytes(inputArray):
    if isinstance(inputArray, bytes):
        return (inputArray, None)
    #TODO: Add logic for numpy
    return (inputArray, None)

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
            arrayC = compressArray(df[col], action[1], False, verbosity)
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
