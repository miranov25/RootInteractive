from MLpipeline.NDFunctionInterface import DataContainer, Fitter
from TTreeHnInteractive.TTreeHnBrowser import *
from InteractiveDrawing.bokeh.bokehTools import *
from InteractiveDrawing.bokeh.bokehDraw import *
from Tools.aliTreePlayer import readDataFrameURL

#
df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
df.head(10)
layout="((0,1),(2,x_visible=0),(3), plot_height=200,plot_width=800,commonX=3,commonY=3,y_visible=0)"
bokehDraw(df,"A>0","A","A:B:C:D","C","slider.A(0,100.1,0,1),slider.B(0,100,100,100,300)",None,layout=layout)



