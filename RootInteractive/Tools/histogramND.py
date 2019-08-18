import numpy as np
import pandas as pd
import logging
from bokeh.models import *
from bokeh.palettes import *
from root_numpy import *
from RootInteractive.Tools.histoNDTools import *


# Standard
histogramNDOptions = {
    "verbose": 0,
    "colors": Category10,
    "plotLegendFormat": "%d"
}


class histogramND(object):
    def __init__(self, HInput=None, HOptions=None ):
        self.name='histogram'
        self.title='histogram'
        self.H=HInput
        self.axes=[]
        self.varNames=[]
        self.varTitles=[]
        self.axisMetadata=[]
        self.options=histogramNDOptions
        if HOptions: self.options.update(HOptions)
        self.verbose=0

    @classmethod
    def fromDictionary(cls, HInput):
        self = cls()
        self.H=HInput['H']
        self.axes=HInput['axes']

    @classmethod
    def fromTHn(cls, rootTHn):
        self = cls()
        self.name=rootTHn.GetName()
        self.title=rootTHn.GetTitle()
        hTuple = hist2array(rootTHn, False, True, True)
        self.H =hTuple[0]
        self.axes=hTuple[1]
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
        self=cls()
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
        self.H=H
        self.axes=axes
        self.name=histoInfo[0]
        self.varNames=varList
        self.varTitles=varList
        return self

    @classmethod
    def arrayToMap(cls, histogramArray):
        histogramMap={}
        for his in histogramArray:
            histogramMap[his.name]=his
        return histogramMap



class histogramNDProjection(object):
    def __init__(self):
        r"""
            Attributes:
                projectionDescription      description string
                histogramQuery             query to evaluate
                histogramSlice             slide describing projection
                histogramProjection        array of axes to project
                histogramOption            options dictionary
                histogramMap               array/map of histograms used for projection queries
                bokehCDS                   output data source for bokeh visualization
                controlArray               array of controls  - to specify projection ranges
                graphArray                 array of graphical objects related to projection (should be one?)
        """
        self.projectionDescription=""
        self.histogramQuery=''
        self.histogramSlice=None
        self.histogramProjection=None
        self.histogramOption=None
        self.histogramMap={}
        self.bokehCDS={}
        self.controlArray=[]
        self.bokehFigure= {}

    def parseProjectionExpression(self, projectionExpression):
        """
        :param projectionExpression: (variableExpression)(sliceExpression)(projection)(stat)
        :return:
        example:
            >>> expression=parseExpression("( (TRD-TRD*0.5+ITS-TRD/2) (0:100,1:10,0:10:2) (0,1) () )")
            >>> print(expression)
            >>> [['(TRD-TRD*0.5+ITS-TRD/2)', ['0:100,1:10,0:10:2'], ['0,1'], []]]
        """
        self.projectionDescription=projectionExpression

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
        self.histogramSlice=projection[0][1]
        self.histogramOption=projection[0][3]
        try:
            self.histogramProjection = [int(i) for i in projection[0][2][0].split(",")]
        except:
            logging.error("Invalid syntax for projection slice", self)


    def evalHistoExpression(self, expression, histogramArray):
        """
        :param expression:         histogram expression to evaluate
        :param histogramArray:
        :return: histogram
        """
        # expression  hisdY-hisdZ, abs(hisdY-hisdZ)
        print(expression)
        histogram = {}
        axes = []
        varNames = []
        query = self.histogramQuery
        keys = list(set(re.findall(r"\w+", query)).intersection(list(histogramArray.keys())))
        func_list = set(re.findall(r"\w+\(", query))  # there still a parenthesis at the end

        for iKey in keys:
            query = query.replace(iKey, "histogramArray[\'" + iKey + "\'][\'H\']")

            for i, var in enumerate(histogramArray[iKey]["varNames"]):
                try:
                    varNames[i].append(var)
                except:
                    varNames.append([var])
                varNames[i] = list(set(varNames[i]))

            axes.append(histogramArray[iKey]["axes"])
        if len(axes) == 0:
            print("Histogram {query} does not exist")
        tmp = axes[0]
        for axe in axes:
            for i in range(len(tmp)):
                if not (axe[i] == tmp[i]).all():
                    raise ValueError("histograms have incompatible axeses.")
        axes = tmp

        for iFunc in func_list:
            if iFunc[:-1] in dir(np):
                query = query.replace(iFunc, "np." + iFunc)

        for i, var in enumerate(varNames):
            varNames[i] = ','.join(var)
        print(query)

        #try:
        #    nSlice = len(eval("np.index_exp[" + str(expression[0][1][0]) + "]"))
        #except:
        #    raise SyntaxError("Invalid Slice: {}".format(str(expression[0][1][0])))

        #nAxes = len(axes)
        #if nSlice != nAxes:
        #    raise IndexError("Number of Slices should be equal to number of axes. {} slices requested but {} axes  exist".format(nSlice, nAxes))

        try:
            # histogram['H'] = eval(query + "[np.index_exp" + str(expression[0][1]).replace("'", "") + "]")
            histogram['H'] = eval(query)
        except:
            raise ValueError("Invalid Histogram expression: {}".format(expression[0][0]))

        histogram["name"] = expression[0][0][0:]
        print(varNames)
        histogram["varNames"] = varNames
        histogram["axes"] = axes
        if len(expression[0][1])==0:
            expression[0][1].append("")
        histoSlice=autoCompleteSlice(expression[0][1][0],histogram)
        histogram["sliceSlider"]= histoSlice
        return histogram

