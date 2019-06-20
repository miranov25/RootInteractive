from bokeh.models import *
from .bokehTools import *
import logging


class bokehDrawSA(object):

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
        self.isNotebook = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        if isinstance(source, pd.DataFrame):
            if (self.verbosity >> 1) & 1:
                logging.info("Panda DataFrame is parsing...")
            df = source
        else:
            if (self.verbosity >> 1) & 1:
                logging.info('source is not a Panda DataFrame, assuming it is ROOT::TTree')
            varList = []
            if 'variables' in options.keys():
                varList = options['variables'].split(":")
            varSource = [varColor, varX, varY, widgetString, query]
            if 'errY' in options.keys():
                varSource.append(options['errY'])
            if 'tooltips' in options.keys():
                for tip in options["tooltips"]:
                    varSource.append(tip[1].replace("@", ""))
            toRemove = [r"^tab.*", r"^accordion.*", "^False", "^True", "^false", "^true"]
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
        dataSource = df.query(query)
        if len(varX)==0:
            return
        if ":" not in varX:
            dataSource.sort_values(varX, inplace=True)
        self.cdsOrig = ColumnDataSource(dataSource)
        self.figure, self.cdsSel, self.plotArray = drawColzArray(df, query, varX, varY, varColor, p, **options)
        self.plotArray.append(self.initWidgets(widgetString))
        pAll=gridplotRow(self.plotArray)
        self.handle=show(pAll,notebook_handle=self.isNotebook)

    @classmethod
    def fromArray(cls, dataFrame, query, figureArray, widgetString, **kwargs):
        """
        Constructor of  figure array
        :param widgetInput:
        :param dataFrame:
        :param query:
        :param figureArray:
        :param kwargs:
        :return:
        """
        self=cls(dataFrame,query,"","","","",None)
        self.figure, self.cdsSel, self.plotArray, dataFrameOrig = bokehDrawArray(dataFrame, query, figureArray, **kwargs)
        self.cdsOrig=ColumnDataSource(dataFrameOrig)
        self.plotArray.append(self.initWidgets(widgetString))
        pAll=gridplotRow(self.plotArray)
        self.handle=show(pAll,notebook_handle=self.isNotebook)
        return self



    def initWidgets(self, widgetString):
        r"""
        Initialize widgets 

        :param widgetString:
            example string
                >>>  slider.name0(min,max,step,valMin,valMax),tab.tabName(checkbox.name1())
        :return: VBox includes all widgets
        """
        widgetList = []
        try:
            widgetList = parseWidgetString(widgetString)
        except:
            logging.error("Invalid widget string", widgetString)
        widgetDict = {"cdsOrig":self.cdsOrig, "cdsSel":self.cdsSel}
        widgetList=self.createWidgets(widgetList)
        for iWidget in widgetList:
            widgetDict[iWidget.title]=iWidget
        callback=makeJScallback(widgetDict)
#        callback.code = callback.code.replace("A","PA")     # Contemporary correction until makeJScallback is fixed
        for iWidget in widgetList:
            iWidget.js_on_change("value", callback)
            iWidget.js_on_event("value", callback)
        #display(callback.code)
        return widgetList

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
        for widgetTitle, subList in zip(*[iter(widgetList0)] * 2):
            name = widgetTitle.split('.')
            if name[0] == "dropdown":
                values = list(subList)
                if len(values) == 0:
                    raise ValueError("dropdown menu quires at least 1 option. The dropdown menu {} has no options",
                                     format(name[1]))
                iWidget = widgets.Select(title=name[1], value=values[0], options=values)
#            elif name[0] == "checkbox":
#                if len(subList) == 0:
#                    active = []
#                elif len(subList) == 1:
#                    if subList[0] in ['True', 'true', '1']:
#                        active = [0]
#                    elif subList[0] in ['False', 'false', '0']:
#                        active = []
#                    else:
#                        raise ValueError("The parameters for checkbox can only be \"True\", \"False\", \"0\" or \"1\". "
#                                         "The parameter for the checkbox {} was:{}".format(name[1], subList[0]))
#                else:
#                    raise SyntaxError("The number of parameters for Checkbox can be 1 or 0."
#                                      "Checkbox {} has {} parameters.".format(name[1], len(subList)))
#                values = list(subList)
#                iWidget = widgets.CheckboxGroup(labels=[name[1]], active=active)
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
