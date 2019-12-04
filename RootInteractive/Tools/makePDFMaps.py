from RootInteractive.Tools.histoNDTools import *
import time

def makePdfMaps(histo, slices, dimI):
    binList = []                        # Prepare list of bin numbers which has same dimensions with original histogram
    for  axis in histo["axes"]:            #        so same slicing can be applied on the bin list
        binList.append(np.arange(len(axis) - 1))

    widthList = [item[3] for item in slices]    # extract the list of widths from slicing tuple
    localHist = histo["H"].copy()
    localAxes = histo["axes"].copy()
    zeit=time.time()
    print(time.time()-zeit)
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
    
    print(time.time()-zeit)
    CenterList = []                                             # initialize list of bin Centers
    for axis in localAxes:
        axisCenter = axis[:-1] + (axis[1] - axis[0]) / 2
        CenterList.append(axisCenter)

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

    for i, (iBin, iCenter) in enumerate(zip(binList, CenterList)):      # apply the slicing to binNumbers and binCenters
        localNumbers.append(iBin[npSlice[i]])
        localCenter.append(iCenter[npSlice[i]])

    print(time.time()-zeit)
    centerI = localCenter.pop(dimI)             # remove the dimension of interest from bin centers and bin numbers
    localNumbers.pop(dimI)                      # the bin centers for dimension of interest is stored at centerI to calculate mean etc.

    mesharrayNumbers = np.array(np.meshgrid(*localNumbers, indexing='ij'))  # 
    binNumbers = []                                                         # 
    for el in mesharrayNumbers:                                             #
        binNumbers.append(el.flatten())                                     #        
                                                                            # using mesharray, n-dimensional coordinate arrays are produced for both bbin number and bin centers  
    mesharrayCenter = np.array(np.meshgrid(*localCenter, indexing='ij'))    # by flattening them thaey are compatible for panda dataframe
    binCenters = []                                                         # 
    for el in mesharrayCenter:                                              #
        binCenters.append(el.flatten())                                     # 
                                                                             
    print(time.time()-zeit)
    newHistogram = np.rollaxis(histogram, dimI, 0)      # move dimension of interest to the dimension-0

    histoList = []
    for i in range(newHistogram.shape[0]):
        histoList.append(newHistogram[i].flatten())     # flatten all dimensions except the dimension of interest

    histoArray = np.array(histoList).transpose()        # by taking transpose we have a flattened array of histograms. 
    means = []
    rms = []
    medians = []
    print(time.time()-zeit)
    for iHisto in histoArray:                           # loop on array of histogram produced above and calculate mean, rms and median for each histogram
        means.append(np.average(centerI, weights=iHisto))
        rms.append(np.sqrt(np.average(centerI ** 2, weights=iHisto)))
    #        medians.append(np.median(np.repeat(centerI, iHisto.astype(int))))   # repeat the values at axis propotional to bin values and find madian of outcome array
        halfSum =  iHisto.sum()/2
        for iBin in range(len(centerI)):
            if iHisto[:iBin].sum() < halfSum and iHisto[:iBin+1].sum() >= halfSum:
                medians.append(centerI[iBin])
                continue

    print(time.time()-zeit)

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
