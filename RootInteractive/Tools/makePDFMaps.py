from RootInteractive.Tools.histoNDTools import *


def makePdfMaps(histo, slices, dimI):
    binList = []  # Prepare list of bin numbers which has same dimensions with original histogram
    for el in histo["axes"]:  # so same slicing can be applied on the bin list
        binList.append(np.arange(len(el) - 1))

    widthList = [item[3] for item in slices]  # extract the list of widths from slicing tuple
    localHist = histo["H"].copy()
    localAxes = histo["axes"].copy()

    for iDim, w in enumerate(widthList):  # get the original histogram and shifted histograms by +/- i,
        L = localHist.shape[iDim]

        iSlice = [slice(None)] * len(localHist.shape)
        iSlice[iDim] = slice(w, L - w)
        dummy = localHist[tuple(iSlice)]

        for i in range(1, w + 1):
            iSlice = [slice(None)] * len(localHist.shape)
            iSlice[iDim] = slice(w + i, L - w + i)
            dummy += localHist[tuple(iSlice)]

            iSlice = [slice(None)] * len(localHist.shape)
            iSlice[iDim] = slice(w - i, L - w - i)
            dummy += localHist[tuple(iSlice)]

        localHist = dummy
        binList[iDim] = binList[iDim][slice(w, L - w)]  # trim the edges
        localAxes[iDim] = localAxes[iDim][slice(w, L - w + 1)]

    newSliceList = []
    for iDim, iSlice in enumerate(slices):  # rearranging slice according to width
        jSlice = slice(iSlice[0] - iSlice[3], iSlice[1] - iSlice[3], iSlice[2])

        if jSlice.start < 0 or jSlice.stop > localHist.shape[iDim]:
            raise ValueError("Range and width of {}th dimension is not compatible: {}".format(iDim, iSlice))

        newSliceList.append(jSlice)

    npSlice = tuple(newSliceList)

    CenterList = []
    for arr in localAxes:
        arri = arr[:-1] + (arr[1] - arr[0]) / 2
        CenterList.append(arri)

    histogram = localHist[npSlice]
    localCenter = []
    localNumbers = []

    for i, (b, c) in enumerate(zip(binList, CenterList)):
        localNumbers.append(b[npSlice[i]])
        localCenter.append(c[npSlice[i]])

    centerI = localCenter.pop(dimI)
    localNumbers.pop(dimI)

    mesharrayNumbers = np.array(np.meshgrid(*localNumbers, indexing='ij'))
    binNumbers = []
    for el in mesharrayNumbers:
        binNumbers.append(el.flatten())

    mesharrayCenter = np.array(np.meshgrid(*localCenter, indexing='ij'))
    binCenters = []
    for el in mesharrayCenter:
        binCenters.append(el.flatten())

    newHistogram = np.rollaxis(histogram, dimI, 0)

    histoList = []
    for i in range(newHistogram.shape[0]):
        histoList.append(newHistogram[i].flatten())

    histoArray = np.array(histoList).transpose()
    means = []
    rms = []
    medians = []
    for iHisto in histoArray:
        means.append(np.average(centerI, weights=iHisto))
        rms.append(np.sqrt(np.average(centerI ** 2, weights=iHisto)))
        medians.append(np.median(np.repeat(centerI, iHisto.astype(int))))

    histogramMap = {}
    varNames = histo['varNames']
    varNames.pop(dimI)

    for iDim in range(len(varNames)):
        histogramMap[varNames[iDim] + "BinNumber"] = binNumbers[iDim]
        histogramMap[varNames[iDim] + "BinCenter"] = binCenters[iDim]

    entries = np.ndarray.flatten(np.sum(histogram, axis=0))

    histogramMap["means"] = means
    histogramMap["rms"] = rms
    histogramMap["medians"] = medians
    histogramMap["entries"] = entries

    out = pd.DataFrame(histogramMap)

    return out
