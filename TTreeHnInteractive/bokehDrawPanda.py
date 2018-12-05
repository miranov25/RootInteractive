#from bokeh.palettes import *
import re
from bokeh.models import *
from bokehTools import *
from ipywidgets import *
#from functools import partial
from IPython.display import display


class bokehDrawPanda(object):

    def __init__(self, source, query, varX, varY, varColor, sliderString, p, **options):
        """
        :param source:           input data frame
        :param query:            query string
        :param varX:             X variable name
        :param varY:             : separated list of the Y variables
        :param varColor:         color map variable name
        :param sliderString:     :  separated sting - list of sliders var(min,max,step, minValue,maxValue)
        :param p:                template figure
        :param options:          optional drawing parameters
                                 - ncols - number fo columns in drawing
                                 - commonX=?,commonY=? - switch share axis
                                 - size
        """
        self.query = query
        self.dataSource = source.query(query)
        self.sliderWidgets = 0
        self.sliderArray = []
        self.varX = varX
        self.varY = varY
        self.varColor = varColor
        self.options = options
        self.initSliders(sliderString)
        self.figure, self.handle, self.bokehSource = drawColzArray(source, query, varX, varY, varColor, p, **options)
        display(self.sliderWidgets)

    def initSliders(self, sliderString):
        """
        parse sliderString string and create range sliders
        :param sliderString:   example string - name0(min,max,step,valMin,valMax): ....
        :return: s sliders
        """
        self.sliderArray = []
        sliderList0 = sliderString.split(":")
        for i, slider in enumerate(sliderList0):
            values = re.split('[(,)]', slider)
            # slider = RangeSlider(start=float(values[1]), end=float(values[2]), step=float(values[3]), value=(float(values[4]), float(values[5])), title=values[0])
            slider = widgets.FloatRangeSlider(description=values[0], layout=Layout(width='66%'), min=float(values[1]), max=float(values[2]), step=float(values[3]),
                                              value=[float(values[4]), float(values[5])])
            slider.observe(self.updateInteractive, names='value')
            self.sliderArray.append(slider)
        self.sliderWidgets = widgets.VBox(self.sliderArray, layout=Layout(width='66%'))

    def updateInteractive(self, b):
        sliderQuery = ""
        for slider in self.sliderArray:
            sliderQuery += str(str(slider.description) + ">" + str(slider.value[0]) + "&" + str(slider.description) + "<" + str(slider.value[1]) + "&")
        sliderQuery = sliderQuery[:-1]
        newSource = ColumnDataSource(self.dataSource.query(sliderQuery))
        self.bokehSource.data = newSource.data
        print(sliderQuery)
        push_notebook(self.handle)
