from bokeh.palettes import *
import re
from bokeh.models import *
from bokehTools import *
from ipywidgets import *


class bokehDrawPanda:
    def __init__(self, source, query, varX, varY, varColor, sliderString, p, **options):
        self.query = query
        self.dataSource = source.query(query)
        self.sliderWidgets = []
        self.varX = varX
        self.varY = varY
        self.varColor = varColor
        self.options = options
        self.initSliders(sliderString)
        print(options)
        self.figure, self.handle, self.bokehSource = drawColzArray(source, query, varX, varY, varColor, p, **options)
        print(options)

    def initSliders(self, sliderString):
        """
        parse sliderString string and create range sliders
        :param sliderString:   example string - name0(min,max,step,valMin,valMax): ....
        :return: srt sliders
        """
        self.sliderWidgets = []
        sliderList0 = sliderString.split(":")
        for i, slider in enumerate(sliderList0):
            values = re.split('[(,)]', slider)
            slider = RangeSlider(start=float(values[1]), end=float(values[2]), step=float(values[3]), value=(float(values[4]), float(values[5])), title=values[0])
            #interact(self.updateInteractive(), slider)
            self.sliderWidgets.append(slider)
        show(widgetbox(self.sliderWidgets))

    def updateInteractive(self):
        sliderQuery = ""
        for slider in self.sliderWidgets:
            sliderQuery += str(str(slider.title) + ">" + str(slider.value[0]) + "&" + str(slider.title) + "<" + str(slider.value[1]) + "&")
        sliderQuery = sliderQuery[:-1]
        newSource = self.dataSource.query(sliderQuery)
        self.bokehSource = newSource
        push_notebook(self.handle)

    query = 0  # init query
    dataSource = 0  # original data source
    bokehSource = 0  # source used for bokeh
    sliderWidgets = []  # slider widgets to interact with figures
    handle = 0  # handle to interact with the jupyter
    figure = 0  # master figure
    varX = 0  # x axis variable name
    varY = 0  # y axis variable names
    varColor = 0  # color variable name
    options = 0  # drawing options
