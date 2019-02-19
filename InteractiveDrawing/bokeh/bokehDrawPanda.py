#from bokeh.palettes import *
import re
from bokeh.models import *
from bokehTools import *
from ipywidgets import *
from Tools.aliTreePlayer import *
#from functools import partial
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
        :param widgetString:     :  separated sting - list of sliders var(min,max,step, minValue,maxValue)
        :param p:                template figure
        :param options:          optional drawing parameters
                                 - ncols - number fo columns in drawing
                                 - commonX=?,commonY=? - switch share axis
                                 - size
                                 - errX=?  - query for errors on X-axis 
                                 - errY=?  - array of queries for errors on Y
                                 Tree options:
                                 - variables    - List of variables which will extract from ROOT File 
                                 - nEntries     - number of entries which will extract from ROOT File
                                 - firstEntry   - Starting entry number 
                                 - mask         - mask for variable names
        """
        if isinstance(source, pd.DataFrame):
            print('Panda Dataframe is parsing...')
            df=source
        else:
            print('source is not a Panda Dataframe, assuming it is ROOT::TTree')
            if 'variables' in options.keys():  vari=options['variables']
            else:   vari=str(re.sub(r'\([^)]*\)', '', widgetString)+":"+varColor+":"+varX+":"+varY)                
            if 'nEntries' in options.keys():  nEntries=options['nEntries']
            else:   nEntries=source.GetEntries()
            if 'firstEntry' in options.keys():  firstEntry=options['firstEntry']
            else:   firstEntry=0
            if 'mask' in options.keys():  columnMask=options['mask']
            else:  columnMask='default'
            df=self.tree2Panda(source, vari, query, nEntries, firstEntry, columnMask)
        
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
        :param sliderString:   example string - slider#name0(min,max,step,valMin,valMax):tab#tabName(checkbox#name1): ...
        :return: s sliders
        """
        sliderList = []
        dropList = []        
        sliderRangeList = []
        checkList = []
        accordList = []
        tabList = []
        widgetList = []
        widgetList0 = widgetString.split(":")
        for widget in widgetList0:
            if widget.split("#")[0] == 'accordion':
                if self.findInList(widget[0:widget.find("(")].split('#')[1],accordList) == -1:
                    accordList.append([widget[0:widget.find("(")].split('#')[1]])
                accordList[self.findInList(widget[0:widget.find("(")].split('#')[1],accordList)].append(widget[widget.find("(")+1:widget.rfind(")")])
            elif widget.split("#")[0] == 'tab': 
                if self.findInList(widget[0:widget.find("(")].split('#')[1],tabList) == -1:
                    tabList.append([widget[0:widget.find("(")].split('#')[1]])
                tabList[self.findInList(widget[0:widget.find("(")].split('#')[1],tabList)].append(widget[widget.find("(")+1:widget.rfind(")")]) 
            else:
                widgetList.append(widget)
                
        for i,lst in enumerate(accordList,len(self.accordArray)):
            self.accordArray.append([lst[0]])
            for wdgt in lst[1:]:
                widget0=wdgt.split('#')
                if widget0[0] == 'slider':
                    if widget0[1].count(',') == 3:      self.initSliders(widget0[1],self.accordArray[i])
                    elif widget0[1].count(',') == 4:    self.initRangeSliders(widget0[1],self.accordArray[i]) 
                elif widget0[0] == 'checkbox':          self.initCheck(widget0[1],self.accordArray[i]) 
                elif widget0[0] == 'dropdown':          self.initDropdown(widget0[1],self.accordArray[i]) 
            
        for i,lst in enumerate(tabList,len(self.tabArray)):
            self.tabArray.append([lst[0]])
            for wdgt in lst[1:]:
                widget0=wdgt.split('#')
                if widget0[0] == 'slider':
                    if widget0[1].count(',') == 3:      self.initSliders(widget0[1],self.tabArray[i])
                    elif widget0[1].count(',') == 4:    self.initRangeSliders(widget0[1],self.tabArray[i]) 
                elif widget0[0] == 'checkbox':          self.initCheck(widget0[1],self.tabArray[i]) 
                elif widget0[0] == 'dropdown':          self.initDropdown(widget0[1],self.tabArray[i]) 
            

        for wdgt in widgetList:
            widget0=wdgt.split('#')
            if widget0[0] == 'slider':
                if widget0[1].count(',') == 3:      self.initSliders(widget0[1],self.widgetArray)
                elif widget0[1].count(',') == 4:    self.initRangeSliders(widget0[1],self.widgetArray) 
            elif widget0[0] == 'checkbox':          self.initCheck(widget0[1],self.widgetArray) 
            elif widget0[0] == 'dropdown':          self.initDropdown(widget0[1],self.widgetArray) 
        
        accordBox = []
        tabBox = []
        for acc in self.accordArray:
            newBox = widgets.VBox(acc[1:])
            accordBox.append(newBox)
        for tabs in self.tabArray:
            newBox = widgets.VBox(tabs[1:])
            tabBox.append(newBox)
            
            
        self.accordion = widgets.Accordion(children= accordBox)
        for i,wdgt in enumerate(self.accordArray):
            self.accordion.set_title(i,wdgt[0])
        self.tab = widgets.Tab(children= tabBox)
        for i,wdgt in enumerate(self.tabArray):
            self.tab.set_title(i,wdgt[0])
        self.all=[]
        if len(self.widgetArray) != 0:
            self.all+=self.widgetArray
        if len(self.tabArray) != 0:
            self.all.append(self.tab)
        if len(self.accordArray) != 0:
            self.all.append(self.accordion)
        
        self.Widgets = widgets.VBox(self.all, layout=Layout(width='66%'))
        #self.sliderWidgets = widgets.VBox(self.sliderArray, layout=Layout(width='66%'))   
        #self.dropWidgets = widgets.VBox(self.dropDownArray, layout=Layout(width='66%'))   
        #self.checkWidgets = widgets.VBox(self.checkboxArray, layout=Layout(width='66%'))   
        #self.accordion = widgets.Accordion(children=[self.sliderWidgets, self.dropWidgets, self.checkWidgets])
        #self.accordion.set_title(0, 'Slider')
        #self.accordion.set_title(1, 'Drop Down')  
        #self.accordion.set_title(2, 'Check Box')              
        
    def initCheck(self, title, array):
        """
        parse string, create Checkbox and append it to the array
        :param title:   example string - name0
        :param array:   array name
        :return: s sliders
        """
        slider = widgets.Checkbox(description=title, layout=Layout(width='66%'), value=False, disabled=False)
        slider.observe(self.updateInteractive, names='value')
        array.append(slider)
            
                       
    def initDropdown(self, title, array):
        """
        parse string, create drop menu and append it to the array
        :param title:   example string - name0(option1, option2, ...)
        :param array:   array name
        :return: s sliders
        """
        values = re.split('[(,)]', title)
        slider = widgets.Dropdown(description=values.pop(0), options=values, layout=Layout(width='66%'), value=values[0])
        slider.observe(self.updateInteractive, names='value')
        array.append(slider)
                                     
    def initSliders(self, title, array):
        """
        parse string, create slider and append it to the array
        :param title:   example string - name0(min, max, step, value)
        :param array:   array name
        :return: s sliders
        """
        values = re.split('[(,)]', title)
        slider = widgets.FloatSlider(description=values[0], layout=Layout(width='66%'), min=float(values[1]), max=float(values[2]), step=float(values[3]),value=float(values[4]))
        slider.observe(self.updateInteractive, names='value')
        array.append(slider)
                   
    def initRangeSliders(self, title, array):
        """
        parse string, create ranged slider and append it to the array
        :param title:   example string - name0(min, max, step, min value, max value)
        :param array:   array name
        :return: s sliders
        """
        values = re.split('[(,)]', title)
        slider = widgets.FloatRangeSlider(description=values[0], layout=Layout(width='66%'), min=float(values[1]), max=float(values[2]), step=float(values[3]), value=[float(values[4]), float(values[5])])
        slider.observe(self.updateInteractive, names='value')
        array.append(slider)
            
        
    def updateInteractive(self, b):
        sliderQuery = ""
        allWidgets = []
        for wdgt in [item[1:] for item in self.accordArray] + [item[1:] for item in self.tabArray]: allWidgets += wdgt
        allWidgets += self.widgetArray 
        for widget in allWidgets:
            if isinstance(widget, widgets.FloatRangeSlider):
                sliderQuery += str(str(widget.description) + ">=" + str(widget.value[0]) + "&" + str(widget.description) + "<=" + str(widget.value[1]) + "&")
            else:
                sliderQuery += str(str(widget.description) + "==" + str(widget.value) + "&" )
        sliderQuery = sliderQuery[:-1]
        newSource = ColumnDataSource(self.dataSource.query(sliderQuery))
        self.bokehSource.data = newSource.data
        print(sliderQuery)
        push_notebook(self.handle)

    def tree2Panda(self, tree, variables, selection, nEntries, firstEntry, columnMask):
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
    
    def findInList(self, c, classes):
        for i, sublist in enumerate(classes):
            if c in sublist:
                return i
        return -1