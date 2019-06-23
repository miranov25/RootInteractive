import numpy as np
import pandas as pd
import logging
from bokeh.models import *
from bokeh.palettes import *
from root_numpy import *
from RootInteractive.InteractiveDrawing.bokeh.bokehTools import *
from bokeh.palettes import *

# from bokeh.io import push_notebook

# Standard
histoNDOptions = {
    "verbose": 0,
    "colors":  Category10
}


def makeHistogram(data, hisString, **kwargs):
    r"""
    Make N dimensional numpy histograms for given histogram string
    :param data:            input panda
    :param hisString:       histogram setup string
    :param kwargs           options
    :return:
        dictionary with histogram information
            * H              - ndarrray
            * axis           - axis description
            * varNames       - variable Names
            * name           - histogram name

    Example usage:
        >>> makeHistogram(MCdata,"TRD:pmeas:particle:#TRD>0>>hisTRDPP(50,0.5,3,20,0.3,5,5,0,5)",3)
    """
    options={}
    options.update(histoNDOptions)
    options.update(kwargs)
    verbose=options['verbose']
    if verbose & 0x1:
        logging.info("makeHistogram   :%s", hisString)
    varList = hisString.split(":")
    description = varList[-1].replace('#', '')
    description = description.split(">>")
    selection = description[0]
    histoInfo = description[1]
    del (varList[-1])
    # get input data as an np array
    df = data.query(selection)
    inputArray = df[varList]
    histoInfo = histoInfo.replace('(', ',').replace(')', ',').split(",")
    bins = []
    hRange = []
    for idx, a in enumerate(varList):
        bins.append(int(histoInfo[idx * 3 + 1]))
        hRange.append((float(histoInfo[idx * 3 + 2]), float(histoInfo[idx * 3 + 3])))
    if verbose & 0x2:
        logging.info("Histogram Name  :%s", histoInfo[0])
        logging.info("Variable list   :%s", varList)
        logging.info("Histogram bins  :%s", bins)
        logging.info("Histogram range :%s", hRange)
    H, axis = np.histogramdd(inputArray.values, bins=bins, range=hRange)
    histogram = {"H": H, "axes": axis, "name": histoInfo[0], "varNames": varList}
    return histogram

def thnToNumpyDD(his):
    tuple=hist2array(his,False,True,True)
    histogram={"H":tuple[0],  "axes":tuple[1], "name": his.GetName(), "title":his.GetTitle()}
    varNames=[]
    varTitles=[]
    for axis in range(his.GetNdimensions()):
        varNames.insert(axis,his.GetAxis(axis).GetName())
        varTitles.insert(axis,his.GetAxis(axis).GetTitle())
    histogram["varNames"]=varNames
    histogram["varTitles"]=varTitles
    return histogram


def makeHistogramPanda(data, hisString, **kwargs):
    """

    :param data:
    :param hisString:
    :param kwargs:
    :return:
    """
    options={}
    options.update(histoNDOptions)
    options.update(kwargs)
    verbose=options['verbose']
    histo = makeHistogram(data, hisString, **options)
    varList = histo["varNames"]
    varList.append("count")
    axes = histo["axes"]
    rows_list = []
    nVars = len(varList)
    for index, x in np.ndenumerate(histo["H"]):
        y = np.arange(nVars, dtype='f')
        for idx, i in enumerate(index):
            y[idx] = float(axes[idx][i])
            if verbose & 0x4:
                logging.info(idx, i, axes[idx][i])
            y[nVars - 1] = x
            # print(y)
            rows_list.append(y)
    df = pd.DataFrame(rows_list, columns=varList)
    return df

def makeHistogramArray(dataSource, histogramStringArray, **kwargs):
    """
    :param dataSource:
    :param histogramStringArray:
    :return:
    """
    options={}
    options.update(histoNDOptions)
    options.update(kwargs)
    histogramArray = []
    for histoString in histogramStringArray:
        histogram = makeHistogram(dataSource, histoString,**options)
        histogramArray.append(histogram)
    return histogramArray


def bokehDrawHistoSliceColz(histo, hSlice, axisX, axisColor, axisStep, figOption, graphOption):
    """
    Draw slices of histogram
        prototype usage in https://github.com/miranov25/RootInteractiveTest/blob/394a934ac206da8a7ae8da582fa9e0ae7efd686f/JIRA/PWGPP-517/pandaHistoPrototype.ipynb

    :param histo:                 - histogram dictionary - see description in makeHistogram
    :param hSlice:
        - slice to visualize (see numpy slice documentation)  e.g:
            >>> np.index_exp[:, 1:3,3:5]
    :param axisX:                 - variable index - projection to draw
    :param axisColor:             - variable index to scan
    :param figOption:             - options (python dictionary)  for figure
    :param graphOption:           - option (dictionary) for figure
    :return:                       histogram figure
    """
    #
    options={}
    options.update(histoNDOptions)
    options.update(figOption)
    sliceString = str(hSlice).replace("slice", "")
    TOOLTIPS = [
        ("index", "$index"),
        #("Slice", sliceString)
    ]
    start = hSlice[axisColor].start
    stop = hSlice[axisColor].stop
    step = 1
    hSliceList = list(hSlice)
    maxColor=len(options['colors'])
    color = options['colors'][min(stop - start + 2,maxColor)]
    data = {}
    x = histo["axes"][axisX][hSlice[axisX]]
    data['varX'] = x
    fIndex = 0
    axis=tuple([a for a in range(0,len(histo["axes"])) if a!=axisX])
    colorAxisLabel=None
    try:
        colorAxisLabel=histo["varTitles"][axisColor]
    except:
        colorAxisLabel=histo["varNames"][axisColor]
    for a in range(start, stop, axisStep):
        hSliceList[axisColor] = slice(a, a + 1, step)
        hSliceLocal = tuple(hSliceList)
        # print a, histo["axes"][axisColor][a]
        hLocal=histo["H"][hSliceLocal]
        y = np.sum(histo["H"][hSliceLocal], axis=axis)
        data["varY" + str(fIndex)] = y
        TOOLTIPS.append((colorAxisLabel+"["+str(a)+"]", "@varY" + str(fIndex)))
        fIndex += 1
    source = ColumnDataSource(data)
    p2 = figure(title=histo["name"], tooltips=TOOLTIPS, **figOption)
    fIndex = 0
    for a in range(start, stop, axisStep):
        xAxisLabel=None
        try:
            xAxisLabel=histo["varTitles"][axisX]
        except:
            xAxisLabel=histo["varNames"][axisX]
        p2.scatter("varX", "varY" + str(fIndex), source=source, color=color[fIndex%maxColor],
                   marker=bokehMarkers[fIndex % 4], legend=colorAxisLabel+"["+str(a)+"]", **graphOption)
        p2.xaxis.axis_label=xAxisLabel
        fIndex += 1
    p2.legend.click_policy = "hide"
    show(p2)
    return source
