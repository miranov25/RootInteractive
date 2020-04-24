from RootInteractive.Tools.histoNDTools import *

def makePdfMaps(histo, slices, dimI, **kwargs):
    options = {
        'quantiles': []
    }
    options.update(kwargs)

    binList = []                        # Prepare list of bin numbers which has same dimensions with original histogram
    for  axis in histo["axes"]:            #        so same slicing can be applied on the bin list
        binList.append(np.arange(len(axis) - 1))

    widthList = [item[3] for item in slices]    # extract the list of widths from slicing tuple
    localHist = histo["H"]
    localAxes = histo["axes"]
    for iDim, w in enumerate(widthList):                        #  get the original histogram and shifted histograms by +/- i,
        L = localHist.shape[iDim]                               #  summing all of these will end a merged version of histograms
                                                                #
        iSlice = [slice(None)] * len(localHist.shape)           #   Ex:  Assume we have: [a b c d e] for an array and apply w=1
        iSlice[iDim] = slice(w, L - w)                          #
        dummy = localHist[tuple(iSlice)]                        #                   First we trim orginal array as: [b c d]
                                                                #    then produce two more arrays for w=-1 and w=1: [a b c]
        for i in range(1, w + 1):                               #                                                   [c d e]    +
            iSlice = [slice(None)] * len(localHist.shape)       #                                             ------------------
            iSlice[iDim] = slice(w + i, L - w + i)              #                                                [a+b+c   b+c+d   c+d+e]
            dummy += localHist[tuple(iSlice)]                   #
                                                                #
            iSlice = [slice(None)] * len(localHist.shape)       #  and every iteration it passed merge histogram to next iteretion to apply width to next dimension
            iSlice[iDim] = slice(w - i, L - w - i)
            dummy += localHist[tuple(iSlice)]

        localHist = dummy
        binList[iDim] = binList[iDim][slice(w, L - w)]          # trim the edges of list of bins
        localAxes[iDim] = localAxes[iDim][slice(w, L - w + 1)]  # trim the axes

    CenterList = []                                             # initialize list of bin Centers
    binEdgeList = []                                            # initialize list of bin Edges
    for axis in localAxes:
        axisCenter = axis[:-1] + (axis[1] - axis[0]) / 2
        CenterList.append(axisCenter)
        binEdgeList.append(axis[1:])

    newSliceList = []
    for iDim, iSlice in enumerate(slices):                                          # rearranging slice according to width
        jSlice = slice(iSlice[0] - iSlice[3], iSlice[1] - iSlice[3], iSlice[2])

        if jSlice.start < 0 or jSlice.stop > localHist.shape[iDim]:
            raise ValueError("Range and width of {}th dimension is not compatible: {}".format(iDim, iSlice))

        newSliceList.append(jSlice)

    npSlice = tuple(newSliceList)

    histogram = localHist[npSlice]                  # apply the slicing
    localCenter = []
    localNumbers = []
    localBinEdge = []

    for i, (iBin, iCenter, iEdge) in enumerate(zip(binList, CenterList, binEdgeList)):      # apply the slicing to binNumbers, binEdges and binCenters
        localNumbers.append(iBin[npSlice[i]])
        localCenter.append(iCenter[npSlice[i]])
        localBinEdge.append(iEdge[npSlice[i]])

    centerI = localCenter.pop(dimI)             # remove the dimension of interest from bin centers and bin numbers
    localNumbers.pop(dimI)                      # the bin centers for dimension of interest is stored at centerI to calculate mean etc.
    edgeI = localBinEdge.pop(dimI)

    mesharrayNumbers = np.array(np.meshgrid(*localNumbers, indexing='ij'))  #
    binNumbers = []                                                         #
    for el in mesharrayNumbers:                                             #
        binNumbers.append(el.flatten())                                     #
                                                                            # using mesharray, n-dimensional coordinate arrays are produced for both bbin number and bin centers
    mesharrayCenter = np.array(np.meshgrid(*localCenter, indexing='ij'))    # by flattening them thaey are compatible for panda dataframe
    binCenters = []                                                         #
    for el in mesharrayCenter:                                              #
        binCenters.append(el.flatten())                                     #

    newHistogram = np.rollaxis(histogram, dimI, 0)      # move dimension of interest to the dimension-0

    histoList = []
    for i in range(newHistogram.shape[0]):
        histoList.append(newHistogram[i].flatten())     # flatten all dimensions except the dimension of interest

    histoArray = np.array(histoList).transpose()        # by taking transpose we have a flattened array of histograms.
    means = []
    rmsd = []
    meansOK = []
    medians = []
    mediansOK = []
    quantiles = []
    quantilesOK = []
    for iQuantile in options['quantiles']:
        quantiles.append([])
        quantilesOK.append([])
    nBinsI = len(centerI)
    for iHisto in histoArray:                           # loop on array of histogram produced above and calculate mean, rmsd and median for each histogram
        cumsumHisto=np.cumsum(iHisto)
        sumHisto=cumsumHisto[-1]
        if sumHisto > 0:
            means.append(np.average(centerI, weights=iHisto))
            rmsd.append(np.sqrt(np.average((centerI - means[-1]) ** 2, weights=iHisto)))
            meansOK.append(1)
        else:
            means.append(0)
            rmsd.append(0)
            meansOK.append(0)

        halfSum = sumHisto/2
        iBin = np.searchsorted(cumsumHisto,halfSum,'right')-1
        if iBin == nBinsI - 1:
            medians.append(0)
            mediansOK.append(0)
        else:
            medians.append((halfSum - cumsumHisto[iBin]) * (edgeI[iBin + 1] - edgeI[iBin]) / (iHisto[iBin+1]) +edgeI[iBin])
            mediansOK.append(1)

        for i, iQuantile in enumerate(options['quantiles']):
            quantileLimit = sumHisto*iQuantile/100
            for iBin in range(nBinsI - 1):
                if iBin == nBinsI - 1:
                    quantiles[i].append(0)
                    quantilesOK[i].append(0)
                    break
                if iHisto[:iBin].sum() < quantileLimit <= iHisto[:iBin + 1].sum():
                    quantiles[i].append((quantileLimit - iHisto[:iBin].sum())*(edgeI[iBin+1]-edgeI[iBin])/(iHisto[:iBin + 1].sum()-iHisto[:iBin ].sum()) + edgeI[iBin])
                    #quantiles[i].append(centerI[iBin])
                    quantilesOK[i].append(1)
                    break
                quantiles[i].append(0)
                quantilesOK[i].append(0)



    histogramMap = {}
    varNames = histo['varNames']
    varNames.pop(dimI)

    for iDim in range(len(varNames)):
        histogramMap[varNames[iDim] + "BinNumber"] = binNumbers[iDim]
        histogramMap[varNames[iDim] + "BinCenter"] = binCenters[iDim]

    entries = np.ndarray.flatten(np.sum(histogram, axis=0))

    histogramMap["means"] = means
    histogramMap["rmsd"] = rmsd
    histogramMap["meansOK"] = meansOK
    histogramMap["medians"] = medians
    histogramMap["mediansOK"] = mediansOK
    histogramMap["entries"] = entries
    for i,iQuantile in enumerate(options['quantiles']):
        histogramMap["quantile_"+ str(iQuantile)] = quantiles[i]
        histogramMap["quantile_"+ str(iQuantile) + "OK"] = quantilesOK[i]

    out = pd.DataFrame(histogramMap)

    return out
