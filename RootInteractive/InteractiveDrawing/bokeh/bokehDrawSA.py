from bokeh.models import *
from .bokehTools import *
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
            'nPointRender': 100000
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

        self.dataSource = df.query(query)
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
        self.cmapDist = None

    @classmethod
    def fromArray(cls, dataFrame, query, figureArray, widgetsDescription, **kwargs):
        r"""
        * Constructor of  interactive standalone figure array
        * :Example usage in:
            * test_bokehDrawArray.py

        :param widgetsDescription:
        :param dataFrame:
        :param query:
        :param figureArray:
        :param kwargs:
        :return:
        """
        options={
            'nPointRender': 100000
        }
        options.update(kwargs)
        kwargs=options
        tmp=""
        optionList=[]
        for fig in figureArray:
            if isinstance(fig,dict):
                optionList.append(fig)
                continue
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
        self.figure, self.cdsSel, self.plotArray, dataFrameOrig, self.cmapDict = bokehDrawArray(self.dataSource, query,
                                                                                                figureArray, **kwargs)
        self.cdsOrig=ColumnDataSource(dataFrameOrig)
        #self.Widgets = self.initWidgets(widgetString)
        widgetList=self.initWidgets(widgetsDescription)
        self.plotArray.append(widgetList)
        self.pAll=gridplotRow(self.plotArray,sizing_mode=self.options['sizing_mode'])
        self.widgetList=widgetList
        #self.pAll=column([self.figure,widgetList],sizing_mode=self.options['sizing_mode'])
        self.handle=show(self.pAll,notebook_handle=self.isNotebook)
        return self

    def initWidgets(self, widgetsDescription):
        r"""
        Initialize widgets

        :param widgetString:
            example string
                >>>  widgets="slider.A(0,100,0.5,0,100),slider.B(0,100,5,0,100),slider.C(0,100,1,1,100):slider.D(0,100,1,1,100)"
        :return: VBox includes all widgets
        """
        if type(widgetsDescription)==list:
            widgetList= makeBokehWidgets(self.dataSource, widgetsDescription, self.cdsOrig, self.cdsSel, self.cmapDict, nPointRender = self.options['nPointRender'])
            if isinstance(self.widgetLayout,list):
                widgetList=processBokehLayoutArray(self.widgetLayout, widgetList)
            else:
                if len(self.widgetLayout)>0:
                    x, layoutList, optionsLayout = processBokehLayout(self.widgetLayout,  widgetList)
                    widgetList=gridplotRow(layoutList)
                else:
                    widgetList=column(widgetList)
            return widgetList

        widgetList = []
        try:
            widgetList = parseWidgetString(widgetsDescription)
        except:
            logging.error("Invalid widget string", widgetsDescription)
        widgetDict = {"cdsOrig":self.cdsOrig, "cdsSel":self.cdsSel}
        widgetList=self.createWidgets(widgetList)
        for iWidget in widgetList:
            try:
                widgetDict[iWidget.title]=iWidget
            except AttributeError:
                widgetDict[iWidget.labels[0]] = iWidget
        callback=makeJScallback(widgetDict, nPointRender=self.options['nPointRender'])
#        callback.code = callback.code.replace("A","PA")     # Contemporary correction until makeJScallback is fixed
        for iWidget in widgetList:
            if isinstance(iWidget,CheckboxGroup):
                iWidget.js_on_click(callback)
            else:
                iWidget.js_on_change("value", callback)
            iWidget.js_on_event("value", callback)
        #display(callback.code)

        if len(self.widgetLayout)>0:
            x, layoutList, optionsLayout = processBokehLayout(self.widgetLayout,  widgetList)
            widgetList=gridplotRow(layoutList)
        else:
            widgetList=column(widgetList)
        return widgetList

    def createWidgets(self, widgetList0):
        r'''
        Build widgets and connect observe function of the bokehDraw object

        :param widgetString:
            Example:
                RootInteractive/InteractiveDrawing/bokeh/test_bokehDrawSA.py
            >>> widgets="slider.A(0,100,0.5,0,100),slider.B(0,100,5,0,100),slider.C(0,100,1,1,100):slider.D(0,100,1,1,100)"

        :return:
            fill widgets arrays  to be shown in the Notebook
                * widgetArray
                * accordArray
                * tabArray

        Algorithm:
            * make a tree representation of widgets (recursive list of lists)
            * create an recursive widget structure
            * assign
        '''
        widgetSubList = []
        for widgetTitle, subList in zip(*[iter(widgetList0)] * 2):
            name = widgetTitle.split('.')
            if name[0] == "dropdown":
                values = list(subList)
                if len(values) == 0:
                    raise ValueError("dropdown menu quires at least 1 option. The dropdown menu {} has no options",
                                     format(name[1]))
                iWidget = widgets.Select(title=name[1], value=values[0], options=values)
            elif name[0] == "multiselect":
                value = list(subList)
                if len(value) == 0:
                    raise ValueError("Multiselect menu quires at least 1 option. The multiselect menu {} has no options",
                                     format(name[1]))
                iWidget = widgets.MultiSelect(title=name[1], value=value, options=value)
            elif name[0] == "checkbox":
                if len(subList) == 0:
                    active = []
                elif len(subList) == 1:
                    if subList[0] in ['True', 'true', '1']:
                        active = [0]
                    elif subList[0] in ['False', 'false', '0']:
                        active = []
                    else:
                        raise ValueError("The parameters for checkbox can only be \"True\", \"False\", \"0\" or \"1\". "
                                         "The parameter for the checkbox {} was:{}".format(name[1], subList[0]))
                else:
                    raise SyntaxError("The number of parameters for Checkbox can be 1 or 0."
                                      "Checkbox {} has {} parameters.".format(name[1], len(subList)))
                iWidget = widgets.CheckboxGroup(labels=[name[1]], active=active)
#                iWidget.title=name[1]
            elif name[0] == "query":
                iWidget = widgets.TextInput(value="", placeholder="Type a query", title="Query")
            elif name[0] == "slider":
                if len(subList) == 4:
                    iWidget = widgets.Slider(title=name[1], start=float(subList[0]), end=float(subList[1]),
                                             step=float(subList[2]), value=float(subList[3]))
                elif len(subList) == 5:
                    iWidget = widgets.RangeSlider(title=name[1], start=float(subList[0]), end=float(subList[1]),
                                                  step=float(subList[2]), value=(float(subList[3]), float(subList[4])))
                else:
                    raise SyntaxError(
                        "The number of parameters for Sliders can be 4 for Single value sliders and 5 for ranged sliders. "
                        "Slider {} has {} parameters.".format(name[1], len(subList)))
            else:
                if (self.verbosity >> 1) & 1:
                    logging.info("type of the widget\"" + name[0] + "\" is not specified. Assuming it is a slider.")
                iWidget = self.createWidgets([["slider." + name[0], subList]])  # For backward compatibility
            widgetSubList.append(iWidget)
#        self.allWidgets += widgetSubList
        return widgetSubList
    verbosity=0


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
