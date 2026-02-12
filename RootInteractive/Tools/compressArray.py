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


def fitsInRange(dfMin, dfMax, minval, maxval, useSentinels):
    return dfMin >= minval + 2*useSentinels and dfMax  <= maxval - useSentinels


def roundAbsolute(df, delta, downgrade_type=True):
    # This should probably also downgrade the type if safe to do so instead of upgrading back
    type=df.dtype
    if delta == 0:
        raise ZeroDivisionError(df, delta)
    if type.kind not in ['f', 'c']:
        return df, None
    if type.kind in ['i','u'] and delta == 1:
        # delta == 1 for integer means no change
        return df, None
    if not np.any(np.isfinite(df)):
        out = np.zeros(len(df), dtype=np.int8)
        out[np.isnan(df)] = 0
        out[np.isposinf(df)] = 1
        out[np.isneginf(df)] = -1
        return out, {"scale": 1, "origin": 0, "sentinels": {"nan":0, "neginf":-1, "posinf":1}}
    quantized = np.rint(df / delta)
    result = quantized * delta
    deltaMean = (df - result)[np.isfinite(df)].mean()
    quantized = np.rint((df-deltaMean) / delta)
    if downgrade_type:
        dfMin = np.nanmin(quantized[np.isfinite(quantized)])
        dfMax = np.nanmax(quantized[np.isfinite(quantized)])
        dfCenter = np.ceil(dfMin + (dfMax - dfMin) * .5)
        allFinite = np.all(np.isfinite(quantized))
        quantized = quantized-dfCenter
        useSentinels = 0 if allFinite else 1
        dfMin -= dfCenter
        dfMax -= dfCenter
        if fitsInRange(dfMin, dfMax, -0x80, 0x7f, useSentinels):
            quantized = np.where(np.isnan(quantized), -0x80, quantized)
            quantized = np.where(np.isposinf(quantized), 0x7f, quantized)
            quantized = np.where(np.isneginf(quantized), -0x7f, quantized)
            out = quantized.astype(np.int8)
            sentinels = {"nan": -0x80, "neginf": -0x7f, "posinf":0x7f}
        elif fitsInRange(dfMin, dfMax, -0x8000, 0x7fff, useSentinels):
            quantized = np.where(np.isnan(quantized), -0x8000, quantized)
            quantized = np.where(np.isposinf(quantized), 0x7fff, quantized)
            quantized = np.where(np.isneginf(quantized), -0x7fff, quantized)
            out = quantized.astype(np.int16)
            sentinels = {"nan": -0x8000, "neginf": -0x7fff, "posinf":0x7fff}
        else:
            quantized = np.where(np.isnan(quantized), -0x80000000, quantized)
            quantized = np.where(np.isposinf(quantized), 0x7fffffff, quantized)
            quantized = np.where(np.isneginf(quantized), -0x7fffffff, quantized)
            out = quantized.astype(np.int32)
            sentinels = {"nan": -0x80000000, "neginf": -0x7fffffff, "posinf":0x7fffffff}
        return out, {"scale":delta, "origin":dfCenter*delta+deltaMean, "sentinels": {} if allFinite else sentinels}
    result -= deltaMean
    return result.astype(type), None


def roundSqrtScaling(df, sigma0, sigma1, nBits=8):
    """
    roundSqrtScaling - round sqrt scaling of float number in nBits, assuming better lossy compression later
    :param df:              - input array (for a moment only pandas or numpy)
    :param sigma0:          - rounding factor
    :param sigma1:          - scaling factor
    :param nBits:           - number of significant bits to round
    :return:                - (rounded array, type of the array)
    """
    type = df.dtype
    if type.kind not in ['f', 'c']:
        return df, None
    if sigma0 <= 0 or sigma1 <= 0:
        raise ValueError("sigma0 and sigma1 must be positive")
    quantized = np.rint(np.arcsinh(df*sigma1/sigma0)/sigma0)
    neginf = -2**(nBits-1)+1
    posinf = 2**(nBits-1)-1
    quantized = np.clip(quantized, neginf, posinf)
    sentinel = -2**(nBits-1)
    quantized = np.nan_to_num(quantized, nan=sentinel)
    if nBits <= 8:
        return quantized.astype(np.int8), {"sigma0": sigma0, "sigma1": sigma1, "mu":0, "sentinels": {"nan": sentinel, "neginf": neginf, "posinf":posinf}}
    if nBits <= 16:
        return quantized.astype(np.int16), {"sigma0": sigma0, "sigma1": sigma1, "mu":0, "sentinels": {"nan": sentinel, "neginf": neginf, "posinf":posinf}}
    return quantized.astype(np.int32), {"sigma0": sigma0, "sigma1": sigma1, "mu":0, "sentinels": {"nan": sentinel, "neginf": neginf, "posinf":posinf}}


def decodeSinhScaling(df, sigma0, sigma1):
    """
    decodeSinhScaling - decode sqrt scaling of float number in nBits, assuming better lossy compression later
    :param df:              - input array (for a moment only pandas or numpy)
    :param sigma0:          - rounding factor
    :param sigma1:          - scaling factor
    :return:                - decoded array
    """
    type = df.dtype
    if type.kind not in ['i', 'u']:
        return df
    if sigma0 <= 0 or sigma1 <= 0:
        raise ValueError("sigma0 and sigma1 must be positive")
    return np.sinh(df * sigma0) * (sigma0 / sigma1)


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
    decodeProgram = []
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
                decodeProgram.append(("linear", decode_transform))
        if action == "sqrt_scaling":
            currentArray, decode_transform = roundSqrtScaling(currentArray, *actionParams)
            if decode_transform is not None:
                decodeProgram.append(("sinh", decode_transform))
        if action == "sqrt_scaling_decode":
            currentArray = decodeSinhScaling(currentArray, *actionParams)
            decodeProgram.append(("decodeSinhScaling", actionParams))
        if action == "zip":
            if not isinstance(currentArray, bytes):
                if isinstance(currentArray, pd.Series):
                    currentArray = currentArray.to_numpy()
                arrayInfo["dtype"] = currentArray.dtype.name
                decodeProgram.append(("array", currentArray.dtype.name))
            currentArray = zlib.compress(currentArray)
            decodeProgram.append("inflate")
        if action == "unzip":
            currentArray = zlib.decompress(currentArray)
            if actionParams:
                currentArray = np.frombuffer(currentArray, dtype=actionParams[0])
        if action == "removeInt64":
            currentArray = removeInt64(currentArray)
        if action == "base64":
            currentArray = base64.b64encode(currentArray).decode("utf-8")
            decodeProgram.append("base64_decode")
        if action == "base64_decode":
            currentArray = base64.b64decode(currentArray)
        if action == "code":
            # Do not send int64 arrays to the client, it will not work
            # Also using pandas for the unique() function
            maxFraction = actionParams[0] if actionParams else .5
            currentArray = pd.Series(currentArray)
            values = currentArray.unique()
            if values.size < maxFraction * currentArray.shape[0]:
                dictValues = {}
                for i, value in enumerate(values):
                    dictValues[value] = i
                if values.size < 2 ** 8:
                    currentArray = currentArray.map(dictValues).astype("int8")
                elif values.size < 2 ** 16:
                    currentArray = currentArray.map(dictValues).astype("int16")
                else:
                    currentArray = currentArray.map(dictValues).astype("int32")
                dtype0 = currentArray.dtype
                dtype1 = values.dtype
                if dtype1.kind == 'O':
                    decodeProgram.append(("code", {"values": values}))
                    arrayInfo["valueCode"] = values
                    continue
                bytes0 = currentArray.to_numpy().tobytes()
                bytes1 = values.tobytes()
                currentArray = bytes0 + bytes1
                decodeDope = {"dtype0":dtype0.name, "dtype1":dtype1.name, "len0":len(bytes0), "len1":len(bytes1), "version":1}
                arrayInfo["decodeDope"] = decodeDope
                decodeProgram.append(("code", decodeDope))
            else:
                arrayInfo["skipCode"] = True
        if action == "decode":
            if not arrayInfo.get("skipCode", False):
                if "valueCode" in arrayInfo:
                    currentArray = arrayInfo["valueCode"][currentArray]
                elif "decodeDope" in arrayInfo:
                    decodeDope = arrayInfo["decodeDope"]
                    len0 = decodeDope["len0"]
                    len1 = decodeDope["len1"]
                    bytes0 = currentArray[:len0]
                    bytes1 = currentArray[len0:len0+len1]
                    arrayCodes = np.frombuffer(bytes0, dtype=decodeDope["dtype0"])
                    arrayValues = np.frombuffer(bytes1, dtype=decodeDope["dtype1"])
                    currentArray = arrayValues[arrayCodes]
        if action == "astype":
            currentArray = currentArray.astype(actionParams[0])
        if action == "array":
            currentArray = np.frombuffer(currentArray, dtype=actionParams[0])
        if action == "snap_size":
            arrayInfo["snap_size"] = getSize((currentArray, decodeProgram))
        if verbosity & 2:
            print(actionTuple)
            print(len(currentArray))
        counter+=1
    arrayInfo["byteorder"] = sys.byteorder
    arrayInfo["array"] = currentArray
    arrayInfo["decodeProgram"] = decodeProgram[::-1]
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
