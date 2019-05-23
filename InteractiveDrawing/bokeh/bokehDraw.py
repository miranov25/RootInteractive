# from bokeh.palettes import *
import re
#from itertools import izip
import pyparsing
from bokeh.models import *
from bokeh.models import ColumnDataSource

from .bokehTools import *
from ipywidgets import *
from Tools.aliTreePlayer import *
from IPython.display import display
import ROOT


class bokehDraw(object):

    def __init__(self, source, query, varX, varY, varColor, widgetString, p, **options):
        """
        :param source:           input data frame
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
        if isinstance(source, pd.DataFrame):
            if (self.verbosity >> 1) & 1:
                print("Panda Dataframe is parsing...")
            df = source
        else:
            if (self.verbosity >> 1) & 1:
                print('source is not a Panda DataFrame, assuming it is ROOT::TTree')
            varList = []
            if 'variables' in options.keys():
                varList = options['variables'].split(":")
            varSource = [varColor, varX, varY, widgetString, query]
            toRemove = ["^tab\..*", "^accordion\..*", "^False", "^True", "^false", "^true"]
            toReplace = ["^slider.", "^checkbox.", "^dropdown."]
            varList += getAndTestVariableList(varSource, toRemove, toReplace, source, self.verbosity)
            if 'tooltip' in options.keys():
                tool = str([str(a[1]) for a in options["tooltip"]])
                varList += filter(None, re.split('[^a-zA-Z0-9_]', tool))
            variableList = ""
            for var in set(varList):
                if len(variableList) > 0: variableList += ":"
                variableList += var

            if 'nEntries' in options.keys():
                nEntries = options['nEntries']
            else:
                nEntries = source.GetEntries()
            if 'firstEntry' in options.keys():
                firstEntry = options['firstEntry']
            else:
                firstEntry = 0
            if 'mask' in options.keys():
                columnMask = options['mask']
            else:
                columnMask = 'default'

            df = tree2Panda(source, variableList, query, nEntries, firstEntry, columnMask)

        self.query = query
        self.dataSource = df.query(query)
        if ":" not in varX:
            self.dataSource.sort_values(varX, inplace=True)
        self.sliderWidgets = 0
        self.accordArray = []
        self.tabArray = []
        self.widgetArray = []
        self.all = []
        self.accordion = widgets.Accordion()
        self.tab = widgets.Tab()
        self.Widgets = widgets.VBox()
        self.varX = varX
        self.varY = varY
        self.varColor = varColor
        self.options = options
        self.initWidgets(widgetString)
        self.figure, self.handle, self.bokehSource = drawColzArray(df, query, varX, varY, varColor, p, **options)
        self.updateInteractive("")
        display(self.Widgets)

    def initWidgets(self, widgetString):
        # type: (str) -> None
        """
        parse widgetString string and create widgets
        :param widgetString:   example string - slider.name0(min,max,step,valMin,valMax),tab.tabName(checkbox.name1())
        :return: s sliders
        """
        self.parseWidgetString(widgetString)
        accordBox = []
        tabBox = []
        for acc in self.accordArray:
            newBox = widgets.VBox(acc[1:])
            accordBox.append(newBox)
        for tabs in self.tabArray:
            newBox = widgets.VBox(tabs[1:])
            tabBox.append(newBox)

        self.accordion = widgets.Accordion(children=accordBox)
        for i, iWidget in enumerate(self.accordArray):
            self.accordion.set_title(i, iWidget[0])
        self.tab = widgets.Tab(children=tabBox)
        for i, iWidget in enumerate(self.tabArray):
            self.tab.set_title(i, iWidget[0])
        self.all = []
        if len(self.widgetArray) != 0:
            self.all += self.widgetArray
        if len(self.tabArray) != 0:
            self.all.append(self.tab)
        if len(self.accordArray) != 0:
            self.all.append(self.accordion)

        self.Widgets = widgets.VBox(self.all, layout=Layout(width='66%'))

    def fillArray(self, iWidget, array):
        """
        Gets create the specified widget and append it into the given widget array.
        :param iWidget:          is a list with 2 entry: 1.entry is a string: "type.name"
                                                        2.entry is a list of parameters
        :param array:           is the list of widgets to be added
        """
        global title
        title = iWidget[0].split('.')
        localWidget = 0
        if title[0] == "checkbox":
            if len(iWidget[1]) == 0:
                status = False
            else:
                if iWidget[1][0] in ['True', 'true', '1']:
                    status = True
                elif iWidget[1][0] in ['False', 'false', '0']:
                    status = False
                else:
                    raise ValueError("The parameters for checkbox can only be \"True\", \"False\", \"0\" or \"1\". "
                                     "The parameter for the checkbox {} was:{}".format(title[1], iWidget[1][0]))
            localWidget = widgets.Checkbox(description=title[1], layout=Layout(width='66%'), value=status,
                                           disabled=False)
        elif title[0] == "dropdown":
            values = list(iWidget[1])
            if len(values) == 0:
                raise ValueError("dropdown menu requires at least 1 option. The dropdown menu {} has no options",
                                 format(title[1]))
            localWidget = widgets.Dropdown(description=title[1], options=values, layout=Layout(width='66%'),
                                           values=values[0])
        elif title[0] == "slider":
            if len(iWidget[1]) == 4:
                localWidget = widgets.FloatSlider(description=title[1], layout=Layout(width='66%'),
                                                  min=float(iWidget[1][0]), max=float(iWidget[1][1]),
                                                  step=float(iWidget[1][2]), value=float(iWidget[1][3]))
            elif len(iWidget[1]) == 5:
                localWidget = widgets.FloatRangeSlider(description=title[1], layout=Layout(width='66%'),
                                                       min=float(iWidget[1][0]), max=float(iWidget[1][1]),
                                                       step=float(iWidget[1][2]),
                                                       value=[float(iWidget[1][3]), float(iWidget[1][4])])
            else:
                raise SyntaxError(
                    "The number of parameters for Sliders can be 4 for Single value sliders and 5 for ranged sliders. "
                    "Slider {} has {} parameters.".format(title[1], len(iWidget[1])))
        else:
            if (self.verbosity >> 1) & 1:
                print("type of the widget\"" + title[0] + "\" is not specified. Assuming it is a slider.")
            self.fillArray(["slider." + title[0], iWidget[1]], array)  # For backward compatibility
        if localWidget != 0:
            localWidget.observe(self.updateInteractive, names='value')
            array.append(localWidget)

    def updateInteractive(self, b):
        sliderQuery = ""
        allWidgets = []
        for iWidget in [item[1:] for item in self.accordArray] + [item[1:] for item in
                                                                  self.tabArray]: allWidgets += iWidget
        allWidgets += self.widgetArray
        for iWidget in allWidgets:
            if isinstance(iWidget, widgets.FloatRangeSlider):
                sliderQuery += str(
                    "{0}>={1}&{2}<={3}&".format(str(iWidget.description), str(iWidget.value[0]),
                                                str(iWidget.description), str(iWidget.value[1])))
            elif isinstance(iWidget, widgets.FloatSlider):
                sliderQuery += str(
                    "{0}>={1}-{2}&{3}<={4}+{5}&".format(str(iWidget.description), str(iWidget.value), str(iWidget.step), str(iWidget.description), str(iWidget.value),
                                                        str(iWidget.step)))
            else:
                sliderQuery += str(str(iWidget.description) + "==" + str(iWidget.value) + "&")
        sliderQuery = sliderQuery[:-1]
        newSource = ColumnDataSource(self.dataSource.query(sliderQuery))
        self.bokehSource.data = newSource.data
        if self.verbosity & 1:
            print(sliderQuery)
        push_notebook(self.handle)

    def parseWidgetString(self, widgetString):
        toParse = "(" + widgetString + ")"
        theContent = pyparsing.Word(pyparsing.alphanums + ".+-_") | '#' | pyparsing.Suppress(',') | pyparsing.Suppress(':')
        widgetParser = pyparsing.nestedExpr('(', ')', content=theContent)
        widgetList0 = widgetParser.parseString(toParse)[0]
        for widgetTitle, iWidget in zip(*[iter(widgetList0)] * 2):
            name = widgetTitle.split('.')
            if name[0] == 'accordion':
                if findInList(name[1], self.accordArray) == -1:
                    self.accordArray.append([name[1]])
                for name, param in zip(*[iter(iWidget)] * 2):
                    self.fillArray([name, param], self.accordArray[findInList(name[1], self.accordArray)])
            elif name[0] == 'tab':
                if findInList(name[1], self.tabArray) == -1:
                    self.tabArray.append([name[1]])
                for name, param in zip(*[iter(iWidget)] * 2):
                    self.fillArray([name, param], self.tabArray[findInList(name[1], self.tabArray)])
            else:
                self.fillArray([widgetTitle, iWidget], self.widgetArray)

    verbosity = 0


def findInList(c, classes):
    for i, sublist in enumerate(classes):
        if c in sublist:
            return i
    return -1


def tree2Panda(tree, variables, selection, nEntries, firstEntry, columnMask):
    entries = tree.Draw(str(variables), selection, "goffpara", nEntries, firstEntry)  # query data
    columns = variables.split(":")
    # replace column names
    #    1.) pandas does not allow dots in names
    #    2.) user can specified own mask
    for i, iColumn in enumerate(columns):
        if columnMask == 'default':
            iColumn = iColumn.replace(".fElements", "").replace(".fX$", "X").replace(".fY$", "Y")
        else:
            masks = columnMask.split(":")
            for mask in masks:
                iColumn = iColumn.replace(mask, "")
        columns[i] = iColumn.replace(".", "_")
    #    print(i, column)
    # print(columns)
    ex_dict = {}
    for i, a in enumerate(columns):
        # print(i,a)
        val = tree.GetVal(i)
        ex_dict[a] = np.frombuffer(val, dtype=float, count=entries)
    df = pd.DataFrame(ex_dict, columns=columns)
    for i, a in enumerate(columns):  # change type to time format if specified
        if (ROOT.TStatToolkit.GetMetadata(tree, a + ".isTime")):
            print(a, "isTime")
            df[a] = pd.to_datetime(df[a], unit='s')
    return df
