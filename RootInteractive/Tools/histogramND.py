#import numpy as np
#import pandas as pd
#import logging
#from bokeh.models import *
#from bokeh.palettes import *
#from root_numpy import *
#from bokeh.transform import *
from RootInteractive.Tools.histoNDTools import *

# Standard
histogramNDOptions = {
    "verbose": 0,
    "colors": Category10,
    "plotLegendFormat": "%d",
    "use_pytorch": False
}


class histogramND(object):
    def __init__(self, HInput=None, HOptions=None):
        self.name = 'histogram'
        self.title = 'histogram'
        self.H = HInput
        self.axes = []
        self.varNames = []
        self.varTitles = []
        self.axisMetadata = []
        self.options = histogramNDOptions
        if HOptions: self.options.update(HOptions)
        self.verbose = 0

    def __str__(self):
        return str(self.__class__) + "\n" + str(self.__dict__)

    @classmethod
    def fromDictionary(cls, HInput):
        self = cls()
        self.H = HInput['H']
        self.axes = HInput['axes']
        self.varNames=HInput['varNames']
        if 'varTitles' in HInput: self.varTitles=HInput['varTitles']
        if 'histoSlice' in HInput: self.histogramSlice=HInput['histoSlice']
        return self

    @classmethod
    def fromTHn(cls, rootTHn):
        self = cls()
        self.name = rootTHn.GetName()
        self.title = rootTHn.GetTitle()
        hTuple = hist2array(rootTHn, False, True, True)
        self.H = hTuple[0]
        self.axes = hTuple[1]
        for axis in range(rootTHn.GetNdimensions()):
            self.varNames.insert(axis, rootTHn.GetAxis(axis).GetName())
            self.varTitles.insert(axis, rootTHn.GetAxis(axis).GetTitle())
        return self

    @classmethod
    def fromPanda(cls, data, hisString, **kwargs):
        r"""
        Make N dimensional numpy histograms for given histogram string
        :param data:            input panda
        :param hisString:       histogram setup string
        :param kwargs           options
        :return: histogram
        Example usage:
            >>> histogram=histogramND.fromPanda(data,"TRD:pmeas:particle:#TRD>0>>hisTRDPP(50,0.5,3,20,0.3,5,5,0,5)",3)
        """
        self = cls()
        self.options.update(kwargs)
        self.verbose = self.options['verbose']
        if self.verbose & 0x1:
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
        if self.verbose & 0x2:
            logging.info("Histogram Name  :%s", histoInfo[0])
            logging.info("Variable list   :%s", varList)
            logging.info("Histogram bins  :%s", bins)
            logging.info("Histogram range :%s", hRange)
        H, axes = np.histogramdd(inputArray.values, bins=bins, range=hRange)
        self.H = H
        self.axes = axes
        self.name = histoInfo[0]
        self.varNames = varList
        self.varTitles = varList
        return self

    @classmethod
    def arrayToMap(cls, histogramArray):
        histogramMap = {}
        for his in histogramArray:
            histogramMap[his.name] = his
        return histogramMap

    @classmethod
    def makeHistogramMap(cls, dataSource, histogramStringArray, **kwargs):
        """
        :param dataSource:
        :param histogramStringArray:
        :return:
        """
        options = {}
        options.update(histoNDOptions)
        options.update(kwargs)
        histogramMap = {}
        for histString in histogramStringArray:
            histogram = cls.fromPanda(dataSource, histString, **options)
            histogramMap[histogram.name] = histogram
        return histogramMap

    def bokehDrawColz(self, hSlice, axisX, axisColor, axisStep, figOption, graphOption):
        """
        Draw slices of histogram

        :param histogram:                 - histogram dictionary - see description in makeHistogram
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
        options = {}
        options.update(histoNDOptions)
        options.update(figOption)
        figOption.pop("plotLegendFormat", None)
        sliceString = str(hSlice).replace("slice", "")
        TOOLTIPS = [
            ("index", "$index"),
            # ("Slice", sliceString)
        ]
        start = hSlice[axisColor].start
        stop = hSlice[axisColor].stop
        step = 1
        hSliceList = list(hSlice)
        maxColor = len(options['colors'])
        color = options['colors'][min(stop - start + 3, maxColor)]
        data = {}
        x = self.axes[axisX][hSlice[axisX]]
        data['varX'] = x
        fIndex = 0
        axis = tuple([a for a in range(0, len(self.axes)) if a != axisX])
        colorAxisLabel = None
        try:
            colorAxisLabel = self.varTitles[axisColor]
        except:
            colorAxisLabel = self.varNames[axisColor]
        for a in range(start, stop, axisStep):
            hSliceList[axisColor] = slice(a, a + 1, step)
            hSliceLocal = tuple(hSliceList)
            # print a, self.axes"][axisColor][a]
            hLocal = self.H[hSliceLocal]
            y = np.sum(self.H[hSliceLocal], axis=axis)
            data["varY" + str(fIndex)] = y
            TOOLTIPS.append((colorAxisLabel + "[" + str(a) + "]", "@varY" + str(fIndex)))
            fIndex += 1
        source = ColumnDataSource(data)
        if "figure" in figOption:
            p2 = figOption["figure"]
        else:
            p2 = figure(title=self.name, tooltips=TOOLTIPS, **figOption)
        fIndex = 0
        for a in range(start, stop, axisStep):
            xAxisLabel = None
            try:
                xAxisLabel = self.varTitles[axisX]
            except:
                xAxisLabel = self.varNames[axisX]
            plotLegend = ""
            if options['plotLegendFormat'] == "%d":
                plotLegend = colorAxisLabel + "[" + str(a) + "]"
            else:
                if "f" in options['plotLegendFormat']:
                    plotLegendFormat = options['plotLegendFormat']
                    plotLegend = colorAxisLabel + "[" + (plotLegendFormat % self.axes[axisColor][a]) + ":" + (
                                plotLegendFormat % self.axes[axisColor][a + 1]) + "]"
            p2.scatter("varX", "varY" + str(fIndex), source=source, color=color[fIndex % maxColor],
                       marker=bokehMarkers[fIndex % 4], legend=plotLegend, **graphOption)
            p2.xaxis.axis_label = xAxisLabel
            fIndex += 1
        p2.legend.click_policy = "hide"
        return p2, source

    def bokehDraw1D(self, hSlice, indexX, figOption, graphOption):
        """
        Draw slices of histogram

        :param hSlice:
            - slice to visualize (see numpy slice documentation)  e.g:
                >>> np.index_exp[:, 1:3,3:5]
        :param indexX:                 - variable index
        :param figOption:             - options (python dictionary)  for figure
        :param graphOption:           - option (dictionary) for figure
        :return:                       histogram figure
        """
        #
        options = {}
        options.update(histoNDOptions)
        options.update(figOption)
        figOption.pop("plotLegendFormat", None)
        #sliceString = str(hSlice).replace("slice", "")
        TOOLTIPS = [
            (self.varNames[indexX], "$x"),
            ('value', "@image"),
            # ("Slice", sliceString)
        ]

        hLocal = self.H[hSlice]
        axisList = [index for index,i in enumerate(self.axes)]
        axisList.remove(indexX)
        hLocal=np.sum(hLocal, axis=tuple(axisList))
        axisX=self.axes[indexX]
        # produce an image of the 1d histogram
        p = figure(x_range=(min(axisX), max(axisX)), title='Image', tooltips=TOOLTIPS, x_axis_label=self.varNames[indexX])
        p.vbar(top=hLocal, x=axisX, width=axisX[1] - axisX[0])
        return p

    def bokehDraw2D(self, hSlice, indexX, indexY, figOption, graphOption):
        """
        Draw slices of histogram

        :param histogram:                 - histogram dictionary - see description in makeHistogram
        :param hSlice:
            - slice to visualize (see numpy slice documentation)  e.g:
                >>> np.index_exp[:, 1:3,3:5]
        :param axisX:                 - variable index X
        :param axisY:                 - variable index Y
        :param figOption:             - options (python dictionary)  for figure
        :param graphOption:           - option (dictionary) for figure
        :return:                       histogram figure
        """
        #
        options = {
            'palette': Spectral6
        }
        options.update(histoNDOptions)
        options.update(figOption)
        figOption.pop("plotLegendFormat", None)
        #sliceString = str(hSlice).replace("slice", "")
        TOOLTIPS = [
            (self.varNames[indexX], "$x"),
            (self.varNames[indexY], "$y"),
            ('value', "@image"),
            # ("Slice", sliceString)
        ]
        hLocal=self.H

        if slice!=None:
            hLocal = self.H[hSlice]
        axisList = [ index for index,i in enumerate(self.axes)]
        axisList.remove(indexX)
        axisList.remove(indexY)
        hLocal=np.sum(hLocal, axis=tuple(axisList))
        axisX=self.axes[indexX]
        axisY=self.axes[indexY]
        mapper = LinearColorMapper(palette=options['palette'], low=np.min(hLocal), high=np.max(hLocal))
        color_bar = ColorBar(color_mapper=mapper, width=8, location=(0, 0))
        #source = ColumnDataSource(data)
        # produce an image of the 2d histogram
        p = figure(x_range=(min(axisX), max(axisX)), y_range=(min(axisY), max(axisY)), title='Image', tooltips=TOOLTIPS, x_axis_label=self.varNames[indexX], y_axis_label=self.varNames[indexY])
        p.image(image=[hLocal], color_mapper=mapper, x=axisX[0], y=axisY[0], dw=axisX[-1] - axisX[0], color=mapper, dh=axisY[-1] - axisY[0])
        p.add_layout(color_bar, 'right')
        return p


class histogramNDProjection(object):
    def __init__(self):
        r"""
            Attributes:
                projectionDescription      description string
                histogramND                histogram expression
                histogramQuery             query to evaluate
                histogramSlice             slide describing projection
                histogramProjection        array of axes to project
                histogramOption            options dictionary
                histogramMap               array/map of histograms used for projection queries
                bokehDCS                   output data source for bokeh visualization
                controlArray               array of controls  - to specify projection ranges
                graphArray                 array of graphical objects related to projection (should be one?)
        """
        self.projectionDescription = ""
        self.histogramQuery = ''
        self.histogramSliceString = None
        self.histogramProjection = None
        self.histogramOption = None
        self.histogramMap = {}
        self.bokehCDS = {}
        self.controlArray = []
        self.bokehFigure = {}

    def __str__(self):
        return str(self.__class__) + "\n" + str(self.__dict__)

    @classmethod
    def fromMap(cls, projectionDescription, histogramMap):
        self = cls()
        self.projectionDesciption = projectionDescription
        self.histogramMap = histogramMap
        self.parseProjectionExpression(projectionDescription)
        return self

    def parseProjectionExpression(self, projectionExpression):
        """
        :param projectionExpression: (variableExpression)(sliceExpression)(projection)(stat)
        :return:
        example:
            >>> expression=parseExpression("( (TRD-TRD*0.5+ITS-TRD/2) (0:100,1:10,0:10:2) (0,1) () )")
            >>> print(expression)
            >>> [['(TRD-TRD*0.5+ITS-TRD/2)', ['0:100,1:10,0:10:2'], ['0,1'], []]]
        """
        self.projectionDescription = projectionExpression

        theContent = pyparsing.Word(pyparsing.alphanums + ":,;+/-*^.\/_")
        parens = pyparsing.nestedExpr("(", ")", content=theContent)
        if projectionExpression[0] != "(":
            projectionExpression = "(" + projectionExpression + ")"
        try:
            res = parens.parseString(projectionExpression)
        except:
            logging.error("Invalid projection expression", projectionExpression)
            return
        projection = res.asList()

        def buildStr(strToBeBuild):
            if isinstance(strToBeBuild, str):
                return strToBeBuild
            iString = ''
            iString += '('
            for sub in strToBeBuild:
                iString += buildStr(sub)
            iString += ')'
            return iString

        self.histogramQuery = buildStr(projection[0][0])
        self.histogramSliceString = projection[0][1][0]
        self.histogramOption = projection[0][3]
        try:
            self.histogramProjection = [int(i) for i in projection[0][2][0].split(",")]
        except:
            logging.error("Invalid syntax for projection slice", self)

    def evaluateHistogram(self):
        """

        :return: histogram
        """
        # expression  hisdY-hisdZ, abs(hisdY-hisdZ)
        histogram = {}
        axes = []
        varNames = []
        query = self.histogramQuery
        keys = list(set(re.findall(r"\w+", query)).intersection(list(self.histogramMap.keys())))
        func_list = set(re.findall(r"\w+\(", query))  # there still a parenthesis at the end

        for iKey in keys:
            query = query.replace(iKey, "self.histogramMap[\'" + iKey + "\'].H")
            for i, var in enumerate(self.histogramMap[iKey].varNames):
                try:
                    varNames[i].append(var)
                except:
                    varNames.append([var])
                varNames[i] = list(set(varNames[i]))

            axes.append(self.histogramMap[iKey].axes)
        if len(axes) == 0:
            print("Histogram {query} does not exist")
        tmp = axes[0]
        for axe in axes:
            for i in range(len(tmp)):
                if not (axe[i] == tmp[i]).all():
                    raise ValueError("histograms have incompatible axes.")
        axes = tmp

        for iFunc in func_list:
            if iFunc[:-1] in dir(np):
                query = query.replace(iFunc, "np." + iFunc)

        for i, var in enumerate(varNames):
            varNames[i] = ','.join(var)
        print(query)

        try:
            histogram['H'] = eval(query)
        except:
            raise ValueError("Invalid Histogram expression: {}".format(self.histogramQuery))

        histogram["name"] = self.histogramQuery
        print(varNames)
        histogram["varNames"] = varNames
        histogram["axes"] = axes
        self.histogramND=histogramND.fromDictionary(histogram)
        return self.histogramND

    def makeProjection(self, controlArray, projectionND=None,  sliceString=None, histogramND=None):
        """
        :param controlArray:      control array with bokeh slideres
        :param sliceString:
        :param histogramND:
        :return:
        """
        if histogramND==None:
            histogramND=self.histogramND
        if sliceString==None:
            sliceString=self.histogramSliceString
        if (projectionND==None):
            projectionND=self.histogramProjection
        controlMap={}
        for control in controlArray:
            controlMap[control.title]=control
        axes = histogramND.axes
        maxSize = len(axes)
        sliceList = sliceString.split(',')
        if len(sliceList) > maxSize:
            logging.error("Size bigger than expected")
            # raise exception
        # replace empty slice by full range slice
        for i in range(len(sliceList), maxSize):
            sliceList.append(":")
        for i, rslice in enumerate(sliceList):
            if len(rslice) == 0:
                sliceList[i] = ":"
        sliceString = ",".join(sliceList)
        npSlice = eval("np.index_exp[" + sliceString + "]")
        npSliceList=list(npSlice)
        for index, i in enumerate(npSlice):
            aName=histogramND.varNames[index]
            #npSliceList.replace(index,npSlice[index])
            if aName in controlMap:
                print(aName, controlMap[aName])
                if i.start==None:
                    npSliceList[index]=slice(controlMap[aName].value[0], controlMap[aName].value[1])
                else:
                    npSliceList[index]=i
        npSlice=tuple(npSliceList)
        histogram={}
        histogram["axes"]=[]
        histogram["varNames"]=[]
        hLocal=histogramND.H[npSlice]
        axisList = [index for index,i in enumerate(histogramND.axes)]

        for axisIndex in projectionND:
            axisList.remove(axisIndex)
            histogram["axes"].append(histogramND.axes[axisIndex])
            histogram["varNames"].append(histogramND.varNames[axisIndex])
            #histogram["varTitles"].append(histogramND.varTitles[axisIndex])

        hLocal=np.sum(hLocal, axis=tuple(axisList))
        histogram["H"]=hLocal
        histogramDraw=histogramND.fromDictionary(histogram)

        return npSlice,histogramDraw
