# Figure options could be parameterized on client. Following parameters to be  provided:
#   * parameterArray, widgets, widgetLayout, figureOptions ...
# The naming convention is following naming in the bokehDraw:
#    *  parameterArray - array of parameters  to be added to the parameterArray
#    *  widgets        - array of widgets controlling parameters to be added to wifgets
#    #  widgetLayout   - array of widgets ID to be added to the widget
#    #  figureOptions  - map of the options to be added to the figureArray

figureParameters={}
figureParameters["markers"]={}
figureParameters["legend"]={}
figureParameters["histogram"]={}
figureParameters["StatParam"]={}
# marker parameters
figureParameters["markers"]["parameterArray"]=[
    {"name": "markerSize", "value":7, "range":[0, 20]}, ## somethig else ????
]
figureParameters["markers"]["widgets"]=[
    ['slider',["markerSize"], {"name": "markerSize"}],
]
# legend parameters - as used in bokehDrawSA
figureParameters["legend"]["parameterArray"]=[
    {"name": "legendFontSize", "value":"11px", "options":['3px','4px','5px','7px',"9px", "11px", "13px", "15px", "17px", "19px"]},
    {"name": "legendLocation", "value":"top_right", "options":["top_right","top_left", "bottom_right","bottom_left"]},
    {"name": "legendVisible", "value":True},
]
figureParameters["legend"]["widgets"]=[
    ['select',["legendFontSize"], {"name": "legendFontSize"}],
    ['select',["legendLocation"], {"name": "legendLocation"}],
    ['toggle', ['legendVisible'], {"name": "legendVisible"}],
]
figureParameters["legend"]["widgetLayout"]=[
    ["legendFontSize","legendLocation","legendVisible"]
]
figureParameters["legend"]["figureOptions"]={"legend_options": {"label_text_font_size": "legendFontSize", "visible": "legendVisible","location":"legendLocation"}}
# histogram parameters
figureParameters["histogram"]["parameterArray"]=[
     {"name": "his_Count", "value":"top_right", "options":["linear_count","log_count","sqrt_count"]}, ###  ??? - supporting aliases for histogram to be created automatically
     #{"name": "his_CountErr", "value":"top_right", "options":["sqrt","sumw2"]},                       ###  ??? - supporting aliases for histogram to be created automatically
]
#
# statistic parameterization
figureParameters["StatParam"]["parameterArray"]=[
        {"name": "nPointsRender", "range":[10, 1000], "value": 50},
]

figureParameters["StatParam"]["widgets"]=[
    ['slider', ['nPointsRender'], {"name": "nPointsRender"}],
    ['range', ['index'], {"index": True}],
]
figureParameters["StatParam"]["widgetLayout"]=[
    ["nPointsRender","index"]
]
