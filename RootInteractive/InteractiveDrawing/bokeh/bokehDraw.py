import logging

from IPython import get_ipython
from IPython.display import display
from bokeh.models import *
from ipywidgets import *

from RootInteractive.Tools.aliTreePlayer import *
from .bokehTools import *


# from bokeh.models import ColumnDataSource


class bokehDraw(object):

    def __init__(self, source, query, varX, varY, varColor, widgetString, p, **kwargs):
        """
        :param source:           input data frame (Panda DataFrame or ROOT:TTree)
        :param query:            query string
        :param varX:             X variable name
        :param varY:             : separated list of the Y variables
        :param varColor:         color map variable name
        :param widgetString:     :  separated string - list of widgets separated by ','
                                 widget options: dropdown, checkbox, slider
                                  slider:
                                    Requires 4 or 5 numbers as parameters
                                    for single valued sliders: slider.name(min,max,step,initial value)
                                      Ex: slider.commonF(0,15,5,0)
                                    for Ranged sliders: slider.name(min,max,step,initial start value, initial end value)
                                      Ex: slider.commonF(0,15,5,0,10)
                                  checkbox:
                                    Requires 1 or none parameters. Allowed parameters are: 0,1,True,False
                                    checkbox.name(initial value default=False)
                                      Ex: checkbox.isMax(True)
                                  dropdown menu:
                                    Requires 1 or more parameters.
                                    dropdown.name(option1,option2,....)
                                      Ex: dropdown.MB(0,0.5,1)

                                 to group widget you can use accordion or tab:
                                   Ex:
                                     accordion.group1(widget1,widget2...), accordion.group2(widget1,widget2...)
                                     tab.group1(widget1,widget2...), tab.group2(widget1,widget2...)

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
            'nPointRender': 100000
        }
        options.update(kwargs)

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

            variableList = constructVariables(query, varX, varY, varColor, widgetString, self.verbosity, **kwargs)
            df = treeToPanda(source, variableList, query, nEntries=treeoptions['nEntries'], firstEntry=treeoptions['firstEntry'], columnMask=treeoptions['columnMask'])

        self.dataSource = df.query(query)
        if hasattr(df, 'metaData'):
            self.dataSource.metaData = df.metaData

        if len(varX) == 0:
            return
        if ":" not in varX:
            self.dataSource.sort_values(varX, inplace=True)
        self.Widgets = self.initWidgets(widgetString)
        self.figure, self.bokehSource, self.plotArray = drawColzArray(df, query, varX, varY, varColor, p, **options)
        self.handle = show(self.figure, notebook_handle=True)
        self.updateInteractive("")
        display(self.Widgets)

    @classmethod
    def fromArray(cls, dataFrame, query, figureArray, widgetString, **kwargs):
        """
        Constructor of  figure array
        :param dataFrame:           input data frame (Panda DataFrame or ROOT:TTree)
        :param query:               query string
        :param figureArray:         List of list indicates what to plot
                            Every element of FigureArray is a List of information about different figure. Only last
                        element may be a dictionary for global option for all figures.

                        If the Figure is a plot, the sublist has three elements: the first element is list of variables
                        for X axis, the second element is a list of variables for Y axis, the third element is a dictionary
                        for options of this particular figure. Note that this dictionary overwrites the global options.

                        If the Figure is a table, the sublist has two elements: the first element is the string: "table"
                        and the second element is a dictionary of which columns are needed to include or exclude.
                    Example:
                        figureArray = [
                            [['A','B'], ['C-A'], {"color": "red", "size": 2, "colorZvar":"C"}],
                            [['A'], ['C+A', 'C-A']],
                            [['B'], ['C+B', 'C-B'],{"color": "red", "size": 7, "colorZvar":"C"}],
                            [['D'], ['(A+B+C)*D'], {"size": 10}],
                            ['table', {'include':'.*A.*'}],
                            {"size":10}
                         ]

                    Possible figure options:
                    size:           the size of markers
                    colorZvar:      Z variable which determines the marker colors for 2D plots. (Overwrites the color option)
                    color:          the color of markers
                    marker:         marker type





        :param kwargs:
        :return:
        """
        options={
            'nPointRender': 100000
        }
        options.update(kwargs)
        kwargs=options
        tmp=""
        for fig in figureArray:
            for entry in fig[0:2]:
                if entry == 'table':    continue
                for word in entry:
                    tmp+=word+":"
        varList=""
        for word in re.split('[^a-zA-Z0-9]', tmp[:-1]):
            if not word.isnumeric():
                varList += word + ":"
        varList += widgetString
        self = cls(dataFrame, query, "", "", "", "", None, variables=varList, **kwargs)
        self.Widgets = self.initWidgets(widgetString)
        self.figure, self.bokehSource, self.plotArray, self.dataSource, _ = bokehDrawArray(self.dataSource, query,
                                                                                        figureArray, **kwargs)
        isNotebook = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        self.handle = show(self.figure, notebook_handle=isNotebook)
        self.updateInteractive("")
        display(self.Widgets)
        return self

    def initWidgets(self, widgetString):
        r"""
        Initialize widgets

        :param widgetString:
            example string
                >>>  slider.name0(min,max,step,valMin,valMax),tab.tabName(checkbox.name1())
        :return: VBox includes all widgets
        """
        self.allWidgets = []
        try:
            widgetList = parseWidgetString(widgetString)
        except:
            logging.error("Invalid widget string", widgetString)
        return widgets.VBox(self.createWidgets(widgetList), layout=Layout(width='66%'))

    def createWidgets(self, widgetList0):
        r'''
        Build widgets and connect observe function of the bokehDraw object

        :param widgetString:
            Example:  https://github.com/miranov25/RootInteractiveTest/blob/master/JIRA/ADQT-3/tpcQADemoWithStatus.ipynb
                >>> widgets="tab.sliders(slider.meanMIP(45,55,0.1,45,55),slider.meanMIPele(50,80,0.2,50,80), slider.resolutionMIP(0,0.15,0.01,0,0.15)),"
                >>> widgets+="tab.checkboxGlobal(slider.global_Warning(0,1,1,0,1),checkbox.global_Outlier(0)),"
                >>> widgets+="tab.checkboxMIP(slider.MIPquality_Warning(0,1,1,0,1),checkbox.MIPquality_Outlier(0), checkbox.MIPquality_PhysAcc(1))"
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
        accordion = widgets.Accordion()
        tab = widgets.Tab()
        for widgetTitle, subList in zip(*[iter(widgetList0)] * 2):
            name = widgetTitle.split('.')
            if name[0] == 'accordion':
                accordion.children = accordion.children + (widgets.VBox(self.createWidgets(subList)),)
                accordion.set_title(len(accordion.children) - 1, name[1])
                continue
            elif name[0] == 'tab':
                tab.children = tab.children + (widgets.VBox(self.createWidgets(subList)),)
                tab.set_title(len(tab.children) - 1, name[1])
                continue
            elif name[0] == "checkbox":
                if len(subList) == 0:
                    status = False
                elif len(subList) == 1:
                    if subList[0] in ['True', 'true', '1']:
                        status = True
                    elif subList[0] in ['False', 'false', '0']:
                        status = False
                    else:
                        raise ValueError("The parameters for checkbox can only be \"True\", \"False\", \"0\" or \"1\". "
                                         "The parameter for the checkbox {} was:{}".format(name[1], subList[0]))
                else:
                    raise SyntaxError("The number of parameters for Checkbox can be 1 or 0."
                                      "Checkbox {} has {} parameters.".format(name[1], len(subList)))
                iWidget = widgets.Checkbox(description=name[1], layout=Layout(width='66%'), value=status,
                                           disabled=False)
            elif name[0] == "dropdown":
                values = list(subList)
                if len(values) == 0:
                    raise ValueError("dropdown menu quires at least 1 option. The dropdown menu {} has no options",
                                     format(name[1]))
                iWidget = widgets.Dropdown(description=name[1], options=values, layout=Layout(width='66%'),
                                           value=values[0])
            elif name[0] == "slider":
                if len(subList) == 4:
                    iWidget = widgets.FloatSlider(description=name[1], layout=Layout(width='66%'),
                                                  min=float(subList[0]), max=float(subList[1]),
                                                  step=float(subList[2]), value=float(subList[3]))
                elif len(subList) == 5:
                    iWidget = widgets.FloatRangeSlider(description=name[1], layout=Layout(width='66%'),
                                                       min=float(subList[0]), max=float(subList[1]),
                                                       step=float(subList[2]),
                                                       value=[float(subList[3]), float(subList[4])])
                else:
                    raise SyntaxError(
                        "The number of parameters for Sliders can be 4 for Single value sliders and 5 for ranged sliders. "
                        "Slider {} has {} parameters.".format(name[1], len(subList)))
            elif name[0] == "query":
                iWidget = widgets.Text(value='', placeholder='Type a query', description='Query:', disabled=False)
            else:
                if (self.verbosity >> 1) & 1:
                    logging.info("type of the widget\"" + name[0] + "\" is not specified. Assuming it is a slider.")
                iWidget = self.createWidgets([["slider." + name[0], subList]])  # For backward compatibility
            iWidget.observe(self.updateInteractive, names='value')
            widgetSubList.append(iWidget)
        self.allWidgets += widgetSubList
        if len(accordion.children) > 0:
            widgetSubList.append(accordion)
        if len(tab.children) > 0:
            widgetSubList.append(tab)
        return widgetSubList

    def updateInteractive(self, b):
        """
        callback function to update drawing CDS (Column data source) of drawing object

        :param b: not used
        :return: none
        """
        sliderQuery = ""

        for iWidget in self.allWidgets:
            if isinstance(iWidget, widgets.FloatRangeSlider):
                sliderQuery += str(
                    "{0}>={1}&{2}<={3}&".format(str(iWidget.description), str(iWidget.value[0]),
                                                str(iWidget.description), str(iWidget.value[1])))
            elif isinstance(iWidget, widgets.FloatSlider):
                sliderQuery += str(
                    "{0}>={1}-{2}&{3}<={4}+{5}&".format(str(iWidget.description), str(iWidget.value), str(iWidget.step),
                                                        str(iWidget.description), str(iWidget.value),
                                                        str(iWidget.step)))
            elif isinstance(iWidget, widgets.Text):
                if iWidget.value:   sliderQuery += str(str(iWidget.value) + "&")
            else:
                sliderQuery += str(str(iWidget.description) + "==" + str(iWidget.value) + "&")
        sliderQuery = sliderQuery[:-1]
        self.bokehSource.data = self.dataSource.query(sliderQuery)
        # print(sliderQuery, newSource.data["index"].size)
        if self.verbosity & 1:
            logging.info(sliderQuery)
        isNotebook = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        if isNotebook:
            push_notebook(self.handle)

    verbosity = 0

def constructVariables(query, varX, varY, varColor, widgetString, verbosity, **kwargs):
    varList = []
    varSource = [varColor, varX, varY, widgetString, query]
    if 'variables' in kwargs.keys():
        varSource.append(kwargs['variables'])
    if 'errY' in kwargs.keys():
        varSource.append(kwargs['errY'])
    if 'tooltips' in kwargs.keys():
        for tip in kwargs["tooltips"]:
            varSource.append(tip[1].replace("@", ""))
    toRemove = [r"^tab.*", r"^query.*", r"^accordion.*", "^False", "^True", "^false", "^true"]
    toReplace = ["^slider.", "^checkbox.", "^dropdown."]
    varList += getAndTestVariableList(varSource, toRemove, toReplace, verbosity)
    if 'tooltip' in kwargs.keys():
        tool = str([str(a[1]) for a in kwargs["tooltip"]])
        varList += filter(None, re.split('[^a-zA-Z0-9_]', tool))
    variableList = ""
    for var in set(varList):
        if len(variableList) > 0: variableList += ":"
        variableList += var
    return variableList
