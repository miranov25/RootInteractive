from bokeh.models import *
from RootInteractive.InteractiveDrawing.bokeh.bokehTools import *
import logging


class bokehDrawSA(object):

    def __init__(self, source, query, varX, varY, varColor, widgetsDescription, p, **kwargs):
        """
        :param source:           input data frame
        :param query:            query string
        :param varX:             X variable name
        :param varY:             : separated list of the Y variables
        :param varColor:         color map variable name
        :param widgetsDescription:     :  separated string - list of widgets separated by ','
                                  slider:
                                    Requires 4 or 5 numbers as parameters
                                    for single valued sliders: slider.name(min,max,step,initial value)
                                      Ex: slider.commonF(0,15,5,0)
                                    for Ranged sliders: slider.name(min,max,step,initial start value, initial end value)
                                      Ex: slider.commonF(0,15,5,0,10)


        :param p:                template figure
        :param options:          optional drawing parameters
                                 - ncols - number fo columns in drawing
                                 - commonX=?,commonY=? - switch share axis
                                 - size
                                 - errX=?  - query for errors on X-axis
                                 - errY=?  - array of queries for errors on Y
                                 Tree options:
                                 - variables     - List of variables which will extract from ROOT File
                                 - nEntries      - number of entries which will extract from ROOT File
                                 - firstEntry    - Starting entry number
                                 - mask          - mask for variable names
                                 - verbosity     - first bit: verbosity for query for every update
                                                 - second bit: verbosity for source file.
        Example usage in:
            test_bokehDrawArray.py
        """

        # define default options
        options = {
            'nCols': 2,
            'tools': 'pan,box_zoom, wheel_zoom,box_select,lasso_select,reset',
            'tooltips': [],
            'tooltip':[],
            'y_axis_type': 'auto',
            'x_axis_type': 'auto',
            'plot_width': 400,
            'plot_height': 400,
            'bg_color': '#fafafa',
            'color': "navy",
            'line_color': "white",
            'widgetLayout':'',
            'sizing_mode': None,
            'nPointRender': 10000
        }
        options.update(kwargs)
        self.options=options
        self.widgetLayout=options['widgetLayout']
        self.isNotebook = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        if isinstance(source, pd.DataFrame):
            if (self.verbosity >> 1) & 1:
                logging.info("Panda DataFrame is parsing...")
            df = source
        else:
            if (self.verbosity >> 1) & 1:
                logging.info('source is not a Panda DataFrame, assuming it is ROOT::TTree')
            treeoptions = {
                'nEntries': source.GetEntries(),
                'firstEntry': 0,
                'columnMask': 'default'
            }
            treeoptions.update(kwargs)

            variableList = constructVariables(query, varX, varY, varColor, widgetsDescription, self.verbosity, **kwargs)
            df = treeToPanda(source, variableList, query, nEntries=treeoptions['nEntries'], firstEntry=treeoptions['firstEntry'], columnMask=treeoptions['columnMask'])

        if query is None:
            self.dataSource = df.copy()
        else:
            self.dataSource = df.query(query).copy()
        if hasattr(df, 'metaData'):
            self.dataSource.metaData = df.metaData
        self.cdsOrig = ColumnDataSource(self.dataSource)
        if len(varX) == 0:
            return
        if ":" not in varX:
            self.dataSource.sort_values(varX, inplace=True)
        self.figure, self.cdsSel, self.plotArray = drawColzArray(df, query, varX, varY, varColor, p, **options)
        self.plotArray.append(self.initWidgets(widgetsDescription))
        self.pAll=gridplotRow(self.plotArray)
        self.handle=show(self.pAll,notebook_handle=self.isNotebook)
        self.cmapList = None
        self.histoList = None
        self.cdsHistoSummary = None
        self.profileList = None
        self.paramDict = None
        self.aliasDict = None

    @classmethod
    def fromArray(cls, dataFrame, query, figureArray, widgetsDescription, **kwargs):
        """
        * Constructor of  interactive standalone figure array
        * Example usage in
            * test_bokehDrawArray.py
            * for list opf options see tutorials
               * https://github.com/miranov25/RootInteractive/blob/master/RootInteractive/tutorial/bokehDraw/test_bokehClinetHistogram.ipynb

        :param widgetsDescription:
        :param dataFrame:      input data frame currently it is panda plan intefrace others (vaex)
        :param query:          panda (later vaex) query to select subset of the data
        :param figureArray:    list of figure - [[varX ] [ varY], {figure options}]
        :param kwargs:         widgetArray, layout=figureLayout, tooltips=tooltips,widgetLayout=widgetLayout,sizing_mode

            example
                >>> figureArray = [
                >>>    [['A'], ['histoA']],
                >>>    [['A'], ['histoAB'], {"visualization_type": "colZ", "show_histogram_error": True}],
                >>>    [['A'], ['histoAB']],
                >>>    [['B'], ['histoTransform'], {"flip_histogram_axes": True}],
                >>>    ["tableHisto", {"rowwise": False}]
                >>> ]
                >>> figureLayoutDesc=[
                >>>    [0, 1,  {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
                >>>    [2, 3, {'y_visible': 1, 'x_visible':1, 'plot_height': 200}],
                >>>    [4, {'plot_height': 40}],
                >>>    {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2, "size": 5}
                >>> ]
                >>> xxx = bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips,
                >>>                        widgetLayout=widgetLayoutDesc, sizing_mode="scale_width", histogramArray=histoArray)

        :return:
        """
        options={
            'nPointRender': 10000,
            "histogramArray": [],
            'parameterArray': [],
            "aliasArray": [],
            "sourceArray": []
        }
        options.update(kwargs)
        kwargs=options
        tmp=""
        optionList=[]
        nFigures = 0
        for fig in figureArray:
            if isinstance(fig,dict):
                optionList.append(fig)
                continue
            nFigures = nFigures + 1
            if fig[0] == 'table':    continue
            for entry in fig:
                if isinstance(entry, dict):
                    optionList.append(entry)
                    continue
                for word in entry:
                    tmp+=word+":"
        varList=""
        for word in re.split('[^a-zA-Z0-9]', tmp[:-1]):
            if not word.isnumeric():
                varList += word + ":"
        if type(widgetsDescription)==str:
            varList += widgetsDescription
        elif type(widgetsDescription)==list:
            for w in widgetsDescription:
                varList+=w[1][0]+":"
        kwargs["optionList"]=optionList
        self = cls(dataFrame, query, "", "", "", "", None, variables=varList, **kwargs)
        if "arrayCompression" in options:
            self.isNotebook = False
        mergedFigureArray = figureArray + widgetsDescription
        self.figure, self.cdsSel, self.plotArray, self.cmapList, self.cdsOrig, self.histoList,\
            self.cdsHistoSummary, self.profileList, self.paramDict, self.aliasDict = bokehDrawArray(self.dataSource, None, mergedFigureArray, **kwargs)
        # self.cdsOrig=ColumnDataSource(dataFrameOrig)
        #self.Widgets = self.initWidgets(widgetString)
        if isinstance(self.widgetLayout, list) or isinstance(self.widgetLayout, dict):
            widgetList=processBokehLayoutArray(self.widgetLayout, self.plotArray[nFigures:])
        self.pAll=gridplotRow([self.figure, widgetList], sizing_mode=self.options['sizing_mode'])
        self.widgetList=widgetList
        #self.pAll=column([self.figure,widgetList],sizing_mode=self.options['sizing_mode'])
        self.handle=show(self.pAll,notebook_handle=self.isNotebook)
        return self

    def initWidgets(self, widgetsDescription):
        """
        Initialize widgets

        :param widgetsDescription:
            example
                >>>  widgetParams=[['range', ['A']], ['slider', ['AA'], {'bins': 10}],  ['select',["Bool"]]]
        :return: VBox includes all widgets
        """
        if type(widgetsDescription)==list:
            widgetList= makeBokehWidgets(self.dataSource, widgetsDescription, self.cdsOrig, self.cdsSel, self.histoList,
                                         self.cmapList, self.cdsHistoSummary, self.profileList, self.paramDict, self.aliasDict, nPointRender = self.options['nPointRender'])
            if isinstance(self.widgetLayout, list) or isinstance(self.widgetLayout, dict):
                widgetList=processBokehLayoutArray(self.widgetLayout, widgetList)
            else:
                widgetList=column(widgetList)
            return widgetList
        raise RuntimeError("String based interface for widgets no longer supported")

    verbosity = 0

def constructVariables(query, varX, varY, varColor, widgetString, verbosity, **kwargs):
    varList = []
    optionParse=["variables","errY","errX","size","markers","colorZvar"]
    varSource = [varColor, varX, varY, widgetString, query]
    for option in optionParse:
            if option in kwargs.keys():
                 varSource.append(kwargs[option])
    if 'tooltips' in kwargs.keys():
        for tip in kwargs["tooltips"]:
            varSource.append(tip[1].replace("@", ""))

    for figOption in  kwargs['optionList']:
        for option in optionParse:
            if option in figOption.keys():
                varSource.append(figOption[option])
    toRemove = [r"^tab.*", r"^query.*", r"^accordion.*", "^False", "^True", "^false", "^true"]
    toReplace = ["^slider.", "^checkbox.", "^dropdown.","^multiselect."]
    varList += getAndTestVariableList(varSource, toRemove, toReplace, verbosity)
    if 'tooltip' in kwargs.keys():
        tool = str([str(a[1]) for a in kwargs["tooltip"]])
        varList += filter(None, re.split('[^a-zA-Z0-9_]', tool))
    variableList = ""
    for var in set(varList):
        if len(variableList) > 0: variableList += ":"
        variableList += var
    return variableList
