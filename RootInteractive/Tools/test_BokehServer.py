# source https://stackoverflow.com/questions/48669119/interactive-plot-of-data-frame-drop-down-menu-to-select-columns-to-display-bok

import pandas as pd
import numpy as np
#Pandas version 0.22.0
#Bokeh version 0.12.10
#Numpy version 1.12.1
from bokeh.io import output_file, show,curdoc
from bokeh.models import Quad
from bokeh.layouts import row, layout,widgetbox
from bokeh.models.widgets import Select,MultiSelect
from bokeh.plotting import ColumnDataSource,Figure,reset_output,gridplot
d= {'A': [1,1,1,2,2,3,4,4,4,4,4], 'B': [1,2,2,2,3,3,4,5,6,6,6], 'C' : [2,2,2,2,2,3,4,5,6,6,6]}
df = pd.DataFrame(data=d)
names = ["A","B", "C"]
#Since bokeh.charts are deprecated so using the new method using numpy histogram
hist,edge = np.histogram(df['A'],bins=4)
#This is the method you need to pass the histogram objects to source data here it takes edge values for each bin start and end and hist gives count.
source = ColumnDataSource(data={'hist': hist, 'edges_rt': edge[1:], 'edges_lt':edge[:-1]})
plot = Figure(plot_height = 300,plot_width = 400)
#The quad is used to display the histogram using bokeh.
plot.quad(top='hist', bottom=0, left='edges_lt', right='edges_rt',fill_color="#036564",
 line_color="#033649",source = source)
#When you change the selection it will this function and changes the source data so that values are updated.
def callback_menu(attr, old, new):
    hist,edge = np.histogram(df[menu.value],bins=4)
    source.data={'hist': hist,'edges_rt': edge[1:], 'edges_lt': edge[:-1]}
#These are interacting tools in the final graph
menu = MultiSelect(options=names,value= ['A','B'], title='Sensor Data')
menu.on_change('value', callback_menu)
layout = gridplot([[widgetbox(menu),plot]])
curdoc().add_root(layout)