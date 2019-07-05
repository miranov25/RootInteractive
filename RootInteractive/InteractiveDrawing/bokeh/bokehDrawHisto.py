from ipywidgets import *
from RootInteractive.Tools.histoNDTools import *
import ipywidgets as widgets
from IPython.core.display import display


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
                logging.info("Processing histogram %s", his.GetName())
                ddHis = thnToNumpyDD(his)
            except:
                logging.error("non compatible object %s\t%s", his.GetName(), his.__class__)
                continue
            self.histogramList[his.GetName()] = ddHis
        self.__processSliders()
        self.sliderWidgets = widgets.VBox(self.sliderList)
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
            * >>>  projection=[hisName,[axisShow],[axisProject]]
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
        optionAll.update(self.projectionList[-1])
        for figureDescription in self.projectionList:
            optionRow=optionAll.copy()
            optionRow.update(figureDescription[-1])
            figureRow = figure(title=optionRow['title'], tools=optionAll['tools'], background_fill_color=optionRow['background_fill_color'], y_axis_type=optionRow['y_axis_type'], x_axis_type=optionRow['x_axis_type'])
            for projection in figureDescription:
                self.processProjection(projection,figureRow)
            self.imageList.append(figureRow)
        return 0

    def __processProjection(self, projection, figureRow):
        """
        see comment above
        :param projection:
        :param figureRow:
        :return:
        """
        option=self.classGraphOptions.copy()
        if len(projection)>1:
            option.update(projection[2])
        ## option colz
        hisName=projection[0]
        histogram=self.histogramList[hisName]
        if option['projectSlice']!=None:
            npIndex=copy(option['projectSlice'])
        else:
            npList=[]
            for i, axisName in enumerate(his["varNames"]):
                nBins = len(histogram["axes"][i])
                npList.append(0,nBins,1)
            npIndex=tuple(npList)

        figdY,sourcedY=bokehDrawHistoSliceColz(self.histogramList[hisName],npIndex, 0,3,1, {'plot_width':800, 'plot_height':300},{'size':5})
        return 0


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