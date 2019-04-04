import numpy as np
import pandas as pd
import logging
from bokeh.models import *
from bokeh.palettes import *
from bokeh.models import ColumnDataSource, ColorBar, HoverTool
# from bokeh.palettes import *
from bokeh.transform import *
from bokehTools import *
from bokeh.layouts import *
from bokeh.palettes import *
from bokeh.io import push_notebook


def makeHistogram(data, hisString, verbose):
    """
    Make N dimensional numpy histograms for given histogram string
    :param data:            input panda
    :param hisString:       histogram setup string
    :param verbose:         verbosity flag
    :return: dictionary with histogram information
            H              - ndarrray
            axis           - axis description
            varNames       - variable Names
            name           - histogram name
    Example usage:
        makeHistogram(MCdata,"TRD:pmeas:particle:#TRD>0>>hisTRDPP(50,0.5,3,20,0.3,5,5,0,5)",3)
    """
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
    histo = {"H": H, "axes": axis, "name": histoInfo[0], "varNames": varList}
    return histo


def makeHistogramPanda(data, hisString, verbose):
    histo = makeHistogram(data, hisString, verbose)
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


def bokehDrawHistoSliceColz(histo, hSlice, axisX, axisColor, figOption, graphOption):
    """
    Draw slices of histogram
    :param histo:                 - hestogram dictionary - see description in makeHistogram
    :param hSlice:                - slice to visualize (see numpy slice decumatiation)  e.g np.index_exp[:, 1:3,3:5]
    :param axisX:                 - variable index - projection to draw
    :param axisColor:             - variable index to scan
    :param figOption:             - options (python dictionary)  for figure
    :param graphOption:           - option (dictionary) for figure
    :return:
    """
    #
    sliceString = str(hSlice).replace("slice", "")
    TOOLTIPS = [
        ("index", "$index"),
        ("Slice", sliceString)
    ]
    start = hSlice[axisColor].start
    stop = hSlice[axisColor].stop
    step = 1
    hSliceList = list(hSlice)
    color = Category10[stop - start + 2]
    data = {}
    x = histo["axes"][axisX][hSlice[axisX]]
    data['varX'] = x
    fIndex = 0
    for a in xrange(start, stop, step):
        hSliceList[axisColor] = slice(a, a + 1, step)
        hSliceLocal = tuple(hSliceList)
        # print a, histo["axes"][axisColor][a]
        y = np.sum(histo["H"][hSliceLocal], axis=(1, 2))
        data["varY" + str(fIndex)] = y
        TOOLTIPS.append(("varY" + str(hSliceList[axisColor]).replace("slice", ""), "@varY" + str(fIndex)))
        fIndex += 1
    source = ColumnDataSource(data)
    p2 = figure(title=histo["name"], tooltips=TOOLTIPS, **figOption)
    fIndex = 0
    for a in xrange(start, stop, step):
        p2.scatter("varX", "varY" + str(fIndex), source=source, color=color[fIndex], marker=bokehMarkers[fIndex % 4], legend="Bin" + str(a), **graphOption)
        fIndex += 1
    p2.legend.click_policy = "hide"
    show(p2)
    return source


class bokehDrawHistoND(object):
     def __init__(self, source, query, varX, varY, varColor, widgetString, p, **options):
         """
         :param source:            panda data source  or tree
         :param query:             selection
         :param variableList
         :param widgetString:
         :param graphOption
         :param figOption
         """
         print(source)