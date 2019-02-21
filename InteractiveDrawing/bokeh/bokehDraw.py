# from bokeh.palettes import *
import re
from itertools import izip
import pyparsing
from bokeh.models import *
from bokehTools import *
from ipywidgets import *
from Tools.aliTreePlayer import *
from IPython.display import display


class bokehDraw(object):

    def __init__(self, source, query, varX, varY, varColor, widgetString, p, **options):
        """
        :param source:           input data frame
        :param query:            query string
        :param varX:             X variable name
        :param varY:             : separated list of the Y variables
        :param varXerr:          variable name of the errors on X 
        :param varYerr:          : separated list of the errors on Y variables
        :param varColor:         color map variable name
        :param widgetString:     :  separated sting - list of sliders var(min,max,step, minValue,maxValue)     /// TODO - add detailed description
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
                print('Panda Dataframe is parsing...')
            df = source
        else:
            if (self.verbosity >> 1) & 1:
                print('source is not a Panda Dataframe, assuming it is ROOT::TTree')
            if 'variables' in options.keys():
                variableList = options['variables']
            else:
                variableList = str(re.sub(r'\([^)]*\)', '', widgetString) + ":" + varColor + ":" + varX + ":" + varY)
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

            df = self.tree2Panda(source, variableList, query, nEntries, firstEntry, columnMask)

        self.query = query
        self.dataSource = df.query(query)
        self.sliderWidgets = 0
        self.accordArray = []
        self.tabArray = []
        self.widgetArray = []
        self.varX = varX
        self.varY = varY
        self.varColor = varColor
        self.options = options
        self.initWidgets(widgetString)
        self.figure, self.handle, self.bokehSource = drawColzArray(df, query, varX, varY, varColor, p, **options)
        display(self.Widgets)

    def initWidgets(self, widgetString):
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
        for i, wdgt in enumerate(self.accordArray):
            self.accordion.set_title(i, wdgt[0])
        self.tab = widgets.Tab(children=tabBox)
        for i, wdgt in enumerate(self.tabArray):
            self.tab.set_title(i, wdgt[0])
        self.all = []
        if len(self.widgetArray) != 0:
            self.all += self.widgetArray
        if len(self.tabArray) != 0:
            self.all.append(self.tab)
        if len(self.accordArray) != 0:
            self.all.append(self.accordion)

        self.Widgets = widgets.VBox(self.all, layout=Layout(width='66%'))

    def fillArray(self, widget, array):
        title = widget[0].split('.')
        localWidget = 0
        if title[0] == "checkbox":
            localWidget = widgets.Checkbox(description=title[1], layout=Layout(width='66%'), value=False, disabled=False)
        elif title[0] == "dropdown":
            values = list(widget[1])
            localWidget = widgets.Dropdown(description=title[1], options=values, layout=Layout(width='66%'), values=values[0])
        elif title[0] == "slider":
            if len(widget[1]) == 4:
                localWidget = widgets.FloatSlider(description=title[1], layout=Layout(width='66%'), min=float(widget[1][0]), max=float(widget[1][1]), step=float(widget[1][2]),
                                                  value=float(widget[1][3]))
            if len(widget[1]) == 5:
                localWidget = widgets.FloatRangeSlider(description=title[1], layout=Layout(width='66%'), min=float(widget[1][0]), max=float(widget[1][1]), step=float(widget[1][2]),
                                                       value=[float(widget[1][3]), float(widget[1][4])])
        if localWidget != 0:
            localWidget.observe(self.updateInteractive, names='value')
            array.append(localWidget)

    def updateInteractive(self, b):
        sliderQuery = ""
        allWidgets = []
        for widget in [item[1:] for item in self.accordArray] + [item[1:] for item in self.tabArray]: allWidgets += widget
        allWidgets += self.widgetArray
        for widget in allWidgets:
            if isinstance(widget, widgets.FloatRangeSlider):
                sliderQuery += str(str(widget.description) + ">=" + str(widget.value[0]) + "&" + str(widget.description) + "<=" + str(widget.value[1]) + "&")
            else:
                sliderQuery += str(str(widget.description) + "==" + str(widget.value) + "&")
        sliderQuery = sliderQuery[:-1]
        newSource = ColumnDataSource(self.dataSource.query(sliderQuery))
        self.bokehSource.data = newSource.data
        if self.verbosity & 1:
            print(sliderQuery)
        push_notebook(self.handle)

    @staticmethod
    def tree2Panda(tree, variables, selection, nEntries, firstEntry, columnMask):
        entries = tree.Draw(str(variables), selection, "goffpara", nEntries, firstEntry)  # query data
        columns = variables.split(":")
        # replace column names
        #    1.) pandas does not allow dots in names
        #    2.) user can specified own mask
        for i, column in enumerate(columns):
            if columnMask == 'default':
                column = column.replace(".fElements", "").replace(".fX$", "X").replace(".fY$", "Y")
            else:
                masks = columnMask.split(":")
                for mask in masks:
                    column = column.replace(mask, "")
            columns[i] = column.replace(".", "_")
        #    print(i, column)
        # print(columns)
        ex_dict = {}
        for i, a in enumerate(columns):
            # print(i,a)
            val = tree.GetVal(i)
            ex_dict[a] = np.frombuffer(val, dtype=float, count=entries)
        df = pd.DataFrame(ex_dict, columns=columns)
        return df

    @staticmethod
    def findInList(c, classes):
        for i, sublist in enumerate(classes):
            if c in sublist:
                return i
        return -1

    def parseWidgetString(self, widgetString):
        toParse = "(" + widgetString + ")"
        theContent = pyparsing.Word(pyparsing.alphanums + ".+-") | '#' | pyparsing.Suppress(',') | ':'
        widgetParser = pyparsing.nestedExpr('(', ')', content=theContent)
        widgetList0 = widgetParser.parseString(toParse)[0]
        for title, wdgt in izip(*[iter(widgetList0)] * 2):
            name = title.split('.')
            if name[0] == 'accordion':
                if self.findInList(name[1], self.accordArray) == -1:
                    self.accordArray.append([name[1]])
                for name, widget in izip(*[iter(wdgt)] * 2):
                    self.fillArray([name, widget], self.accordArray[self.findInList(name[1], self.accordArray)])
            elif name[0] == 'tab':
                if self.findInList(name[1], self.tabArray) == -1:
                    self.tabArray.append([name[1]])
                for name, widget in izip(*[iter(wdgt)] * 2):
                    self.fillArray([name, widget], self.tabArray[self.findInList(name[1], self.tabArray)])
            else:
                self.fillArray([title, wdgt], self.widgetArray)

    verbosity = 0
