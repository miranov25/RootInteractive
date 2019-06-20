from bokeh.io import show
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import CustomJS, ColumnDataSource, Slider, RangeSlider
from RootInteractive.InteractiveDrawing.bokeh.bokehTools import *



def test_FigureArray(dataframe, query, figureArray,**kwargs):
    """
    Test of the bokehDrawArray to draw figure array -
    :return:
    """
    # 1.) create figures
    figureArray = [
        [['A'], ['D+A','C-A'], {"size": 8}],
        [['A'], ['C+A', 'C-A']],
        [['A'], ['sin(A/10)', 'sin(A/20)*0.5', 'sin(A/40)*0.25'], {"size": 10}]
    ]
    figureLayout: str = '((0,1, plot_height=200),(2, x_visible=1),commonX=1,x_visible=1,y_visible=0,plot_height=150,plot_width=1000)'
    tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
    #dummy,source,figureList  = bokehDrawArray(dataframe, "A>0", figureArray, layout=figureLayout, color="blue", size=4, tooltips=tooltips)
    dummy,source,figureList  = bokehDrawArray(dataframe, query, figureArray, **kwargs )

    # 2.) make CDS
    cdsOrig = ColumnDataSource(dataframe.query("A>0"))
    cdsSel =  source
    widgetDict={"cdsOrig":cdsOrig, "cdsSel":cdsSel}
    sliders=[]
    # 3.) create sliders
    for a in dataframe.columns:
        length=dataframe[a].max()-dataframe[a].min()
        sliderRange= RangeSlider(start=dataframe[a].min(), end=dataframe[a].max(), value=(dataframe[a].min(), dataframe[a].max()), step=length*0.01, title=a+"Range")
        sliders.append([sliderRange])
        widgetDict[a+"Range"]=sliderRange
    # make callback
    mycallback=makeJScallback(widgetDict)
#    display(mycallback)
    for a in sliders:
        figureList.append(a)
        a[0].js_on_event('value',mycallback)
        a[0].js_on_change('value',mycallback)

    sliderRange2= RangeSlider(start=0.1, end=4, value=(0.1,4) , step=0.1, title="Range2")
    sliderRange2.js_on_change('value', mycallback)
    figureList.append([sliderRange2])
    pAll=    gridplotRow(figureList)
    show(pAll)
    return pAll