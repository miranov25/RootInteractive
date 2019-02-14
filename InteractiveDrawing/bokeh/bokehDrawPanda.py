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
        self.checkboxArray = []
        self.dropDownArray = []
        self.sliderArray = []
        self.varX = varX
        self.varY = varY
        self.varColor = varColor
        self.options = options
        self.initWidgets(widgetString)
        self.figure, self.handle, self.bokehSource = drawColzArray(df, query, varX, varY, varColor, p, **options)
        display(self.accordion)

    def initWidgets(self, widgetString):
        """
        parse sliderString string and create range sliders
        :param sliderString:   example string - name0(min,max,step,valMin,valMax): ....
        :return: s sliders
        """
        sliderList = []
        dropList = []        
        sliderRangeList = []
        checkList = []
        widgetList0 = widgetString.split(":")
        for widget in widgetList0:
            widget0 = widget.split("#")
            if widget0[0] == 'slider':
                if widget0[1].count(',') == 4:      sliderRangeList.append(widget0[1])
                elif widget0[1].count(',') == 3:    sliderList.append(widget0[1]) 
            elif widget0[0] == 'checkbox':          checkList.append(widget0[1]) 
            elif widget0[0] == 'dropdown':          dropList.append(widget0[1]) 
        self.initCheck(checkList)
        self.initDropdown(dropList)
        self.initSliders(sliderList)
        self.initRangeSliders(sliderRangeList)
        self.sliderWidgets = widgets.VBox(self.sliderArray, layout=Layout(width='66%'))   
        self.dropWidgets = widgets.VBox(self.dropDownArray, layout=Layout(width='66%'))   
        self.checkWidgets = widgets.VBox(self.checkboxArray, layout=Layout(width='66%'))   
        self.accordion = widgets.Accordion(children=[self.sliderWidgets, self.dropWidgets, self.checkWidgets])
        self.accordion.set_title(0, 'Slider')
        self.accordion.set_title(1, 'Drop Down')  
        self.accordion.set_title(2, 'Check Box')              
        
    def initCheck(self, checkList):
        """
        parse sliderString string and create range sliders
        :param sliderString:   example string - name0(min,max,step,valMin,valMax): ....
        :return: s sliders
        """
        for box in checkList:
            slider = widgets.Checkbox(description=box, layout=Layout(width='66%'), value=False, disabled=False)
            slider.observe(self.updateInteractive, names='value')
            self.checkboxArray.append(slider)
            
                       
    def initDropdown(self, dropList):
        """
        parse sliderString string and create range sliders
        :param sliderString:   example string - name0(min,max,step,valMin,valMax): ....
        :return: s sliders
        """
        for i, slider in enumerate(dropList):
            values = re.split('[(,)]', slider)
            slider = widgets.Dropdown(description=values.pop(0), options=values, layout=Layout(width='66%'), value=values[0])
            slider.observe(self.updateInteractive, names='value')
            self.dropDownArray.append(slider)
                                     
    def initSliders(self, sliderList):
        """
        parse sliderString string and create range sliders
        :param sliderString:   example string - name0(min,max,step,valMin,valMax): ....
        :return: s sliders
        """
        for i, slider in enumerate(sliderList):
            values = re.split('[(,)]', slider)
            slider = widgets.FloatSlider(description=values[0], layout=Layout(width='66%'), min=float(values[1]), max=float(values[2]), step=float(values[3]),value=float(values[4]))
            slider.observe(self.updateInteractive, names='value')
            self.sliderArray.append(slider)
                   
    def initRangeSliders(self, sliderRangeList):
        """
        parse sliderString string and create range sliders
        :param sliderString:   example string - name0(min,max,step,valMin,valMax): ....
        :return: s sliders
        """
        for i, slider in enumerate(sliderRangeList):
            values = re.split('[(,)]', slider)
            slider = widgets.FloatRangeSlider(description=values[0], layout=Layout(width='66%'), min=float(values[1]), max=float(values[2]), step=float(values[3]),
                                              value=[float(values[4]), float(values[5])])
            slider.observe(self.updateInteractive, names='value')
            self.sliderArray.append(slider)
            
        
    def updateInteractive(self, b):
        sliderQuery = ""
        for slider in self.sliderArray:
            if isinstance(slider.value, tuple):   
                sliderQuery += str(str(slider.description) + ">=" + str(slider.value[0]) + "&" + str(slider.description) + "<=" + str(slider.value[1]) + "&")
            elif isinstance(slider.value, float):
                sliderQuery += str(str(slider.description) + "==" + str(slider.value) + "&" )
        for dropDown in self.dropDownArray:
            sliderQuery += str(str(dropDown.description) +  "==" +  str(dropDown.value)  + "&")
        for checkbox in self.checkboxArray:
            sliderQuery += str(str(checkbox.description) +  "==" + str(checkbox.value) + "&")
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
