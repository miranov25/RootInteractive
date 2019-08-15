from ipywidgets import *
from RootInteractive.Tools.histoNDTools import *
import ipywidgets as widgets
from IPython.display import *


class bokehDrawHisto(object):
    def __init__(self):
        """
        """
        self.histogramList = []
        self.projectionList = []
        self.sliderList = []
        self.imageList = []
        self.sliderWidgets = None
        self.figureOptions={}
        self.graphOptions={}
        self.figureOptions.update(self.classFigureOptions)
        self.graphOptions.update(self.classGraphOptions)

    @classmethod
    def fromTHnArray(cls, tHnArray, inputProjectionList, figureOptions, graphOptions):
        """
        initialize bokehDrawHisto from histogramArray and inputProjectionArray
        :param tHnArray:                 TContainer with histograms
        :param inputProjectionList:
        :param graphOptions:
        :param figureOptions:
        :return:
        """
        self: bokehDrawHisto = cls()
        self.figureOptions.update(figureOptions)
        self.graphOptions.update(graphOptions)
        self.projectionList = inputProjectionList
        self.histogramList = {}
        for his in tHnArray:
            try:
                # check if root THn - otherwise assume
                ddHis = thnToNumpyDD(his)
                logging.info("Processing histogram %s", his.GetName())
            except:
                #logging.error("non compatible object %s\t%s", his.GetName(), his.__class__)
                ddHis=his
            logging.info("Processing histogram %s", ddHis['name'])
            self.histogramList[ddHis['name']] = ddHis
        self.__processSliders()
        self.sliderWidgets = widgets.VBox(self.sliderList)
        self.__processProjections()
        isNotebook = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        #self.handle = show(self.sliderWidgets, notebook_handle=isNotebook)
        display(self.sliderWidgets)
        return self

    def __processSliders(self):
        """
        Create sliders for all (in future used) variables
        :return: None
        """
        axesArray = {}
        for hisKey in self.histogramList:
            his = self.histogramList[hisKey]
            for i, axisName in enumerate(his["varNames"]):
                # axisName=his["varNames"][axisKey]
                # print(axisName)
                nBins = len(his["axes"][i])
                xMin = his["axes"][i][0]
                xMax = his["axes"][i][nBins - 1]
                # print(his["name"],his["title"],axisName, nBins,xMin,xMax)
                if axisName not in axesArray:
                    axesArray[axisName] = [axisName, nBins, xMin, xMax]
        for axisKey in axesArray:
            print(axesArray[axisKey])
            axis = axesArray[axisKey]
            slider = widgets.FloatRangeSlider(description=axis[0], layout=Layout(width='66%'), min=axis[2], max=axis[3],
                                              step=(axis[3] - axis[2]) / axis[1], value=[axis[2], axis[3]])
            self.sliderList.append(slider)

    def __processProjections(self):
        r"""
        process projectionList and create array/list of figures and CDS
            * >>>  projectionList=[ figureDescription0, figureDecription1, ... options]
            * >>>  figureDescription=[[projection0,projection1, ...], options],
            * >>>  projection=(hisExpression)(slice)(projection)(statistic)
            Example projection:
                projection=['hisdY',[0,1], {options+dictionary}]
                projectionOptions:
                    projectSlice - per dimension
                        None   -  use sliders
                        ":"    -  use full range
                        "xxx"     -  anything else - user defined input
                to be added:
                    CDS  - per projection
        :return:
        """
        optionAll=self.figureOptions.copy()
        if isinstance(self.projectionList[-1], dict):
            optionAll.update(self.projectionList[-1])
        for figureDescription in self.projectionList:
            optionRow=optionAll.copy()
            if isinstance(figureDescription[-1], dict):
                optionRow.update(figureDescription[-1])
            figureRow = figure(title=optionRow['title'], tools=optionAll['tools'], background_fill_color=optionRow['background_fill_color'], y_axis_type=optionRow['y_axis_type'], x_axis_type=optionRow['x_axis_type'])
            for projection in figureDescription:
                self.__processProjection(projection,figureRow,{"figure":figureRow},optionRow)
            self.imageList.append(figureRow)
        return 0

    def __processProjection(self, projection, figureRow, figureOption, graphOption):
        """
        see comment above
        :param projection:
        :param figureRow:
        :return:
        """
        return
        expressionList = parseProjectionExpression(projection)
        histo = evalHistoExpression(expressionList, self.histogramList)
        if len(expressionList[0][2]) >1:
            p, d = bokehDrawHistoSliceColz(histo, eval("np.index_exp[" + str(expressionList[0][1][0]) + "]"), expressionList[0][2][0], expressionList[0][2][1], 1, figureOption, graphOption)
        return p, d



    classFigureOptions={
        "title":None,
        'tools': 'pan,box_zoom, wheel_zoom,box_select,lasso_select,reset',
        'tooltips': [],
        'y_axis_type': 'auto',
        'x_axis_type': 'auto',
        'background_fill_color':None,
    }
    classGraphOptions={
        "projectSlice":None
    }