from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar, HoverTool
# from bokeh.palettes import *
from bokeh.transform import *
from bokehTools import *
from bokeh.layouts import *
from bokeh.palettes import *
from bokeh.io import push_notebook
import copy


def drawColz(dataFrame, query, varX, varY, varColor, p=0):
    """
    drawing example - functionality like the tree->Draw colz
    :param dataFrame:   data frame
    :param query:
    :param varX:        x query
    :param varY:        y query
    :param varColor:    z query
    :return:
    """
    dfQuery = dataFrame.query(query)
    source = ColumnDataSource(dfQuery)
    mapper = linear_cmap(field_name=varColor, palette=Spectral6, low=min(dfQuery[varColor]), high=max(dfQuery[varColor]))
    if p == 0:
        p = figure(plot_width=500, plot_height=500, title="XXX")
    p.circle(x=varX, y=varY, line_color=mapper, color=mapper, fill_alpha=1, size=2, source=source)
    fig = color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0, 0))
    p.add_layout(color_bar, 'right')
    show(p)
    return fig


def drawColzArray(dataFrame, query, varX, varY, varColor, p, **options):
    """
    drawing example - functionality like the tree->Draw colz

    :param options1:
    :param dataFrame:   data frame
    :param query:
    :param varX:        x query
    :param varY:        y query array of queries
    :param varYerr:     errors on y query array of queries
    :param varColor:    z query
    :param p:           figure template TODO - check if some easier way to pass parameters -CSS string ?
    :param options      optional drawing parameters
      option - ncols - number fo columns in drawing
      option - commonX=?,commonY=? - switch share axis
      option - size
      option errX  - query for errors on X
      option errY  - array of queries for errors on Y
      option tooltip - tooltip to show

    :return:
    TODO  use other options if specified: size file and line color - p.circle(x, factors, size=15, fill_color="orange", line_color="green", line_width=3)
    """
    dfQuery = dataFrame.query(query)
    source = ColumnDataSource(dfQuery)
    mapper = linear_cmap(field_name=varColor, palette=Spectral6, low=min(dfQuery[varColor]), high=max(dfQuery[varColor]))
    
    varYArray = varY.split(":")
    plotArray = []
    pFirst = None
    size = 2
    if 'line' in options.keys(): line = options['line']
    else: line=0
    if 'size' in options.keys(): size = options['size']
    tools = 'pan,box_zoom, wheel_zoom,box_select,lasso_select,reset'
    if 'tooltip' in options.keys(): tools = [HoverTool(tooltips=options['tooltip']), tools]
    if 'y_axis_type' in options.keys(): y_axis_type=options['y_axis_type']
    else: y_axis_type='auto'
    if 'x_axis_type' in options.keys():
        x_axis_type = 'datetime'
    else:
        x_axis_type = 'auto'
    if 'errX' in options.keys(): varXerr = options['errX']
    if 'errY' in options.keys():
        varYerrArray = options['errY'].split(":")
    else:
        varYerrArray = varYArray
    for y, yerr in zip(varYArray, varYerrArray):
        if p:
            p2 = figure(plot_width=p.plot_width, plot_height=p.plot_height, title=y + " vs " + varX + "  Color=" + varColor, tools=tools, x_axis_type=x_axis_type, y_axis_type=y_axis_type)
        else:
            p2 = figure(plot_width=500, plot_height=500, title=y + " vs " + varX + "  Color=" + varColor, tools=tools, x_axis_type=x_axis_type)
        if 'varXerr' in locals():
            err_x_x = []
            err_x_y = []
            for coord_x, coord_y, x_err in zip(source.data[varX], source.data[y], source.data[varXerr]):
                err_x_y.append((coord_y, coord_y))
                err_x_x.append((coord_x - x_err, coord_x + x_err))
            p2.multi_line(err_x_x, err_x_y)
        if 'errY' in options.keys():
            err_y_x = []
            err_y_y = []
            for coord_x, coord_y, y_err in zip(source.data[varX], source.data[y], source.data[yerr]):
                err_y_x.append((coord_x, coord_x))
                err_y_y.append((coord_y - y_err, coord_y + y_err))
            p2.multi_line(err_y_x, err_y_y)
        p2.circle(x=varX, y=y, line_color=mapper, color=mapper, fill_alpha=1, source=source, size=size)
        if line ==1: p2.line(x=varX, y=y, source=source)
        if pFirst:
            if 'commonX' in options.keys(): p2.x_range = pFirst.x_range
            if 'commonY' in options.keys(): p2.y_range = pFirst.y_range
        else:
            pFirst = p2
        plotArray.append(p2)
        color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0, 0))
        p2.add_layout(color_bar, 'right')
    ncols = 1
    if 'ncols' in options.keys():
        ncols = options['ncols']
    # nRows=len(plotArray)/ncols+1
    plotArray2D = []
    for i, plot in enumerate(plotArray):
        pRow = i / ncols
        pCol = i % ncols
        if pCol == 0: plotArray2D.append([])
        plotArray2D[pRow].append(plot)
    pAll = gridplot(plotArray2D)
    #    print(plotArray2D)
    handle = show(pAll, notebook_handle=True)  # TODO make it OPTIONAL
    return pAll, handle, source



def drawColzNotebook(myfigure, dataFrame, query, varX, varY, varColor):
    """
    draw interactive colz figure in notebook
    push_notebook should guarantee automatic update of figure
    figure should be registered before  registering notebook_handle  show(p,notebook_handle=True)
    :param myfigure:            - figure to update
    :param dataFrame:           - source data frame
    :param query:
    :param varX:
    :param varY:
    :param varColor:
    :return:
    """
    glyphName = str("df(x=" + varX + ",y=" + varY + "colz=" + varColor + ",q=" + query + ")")
    dfQuery = dataFrame.query(query)
    source = ColumnDataSource(dfQuery)
    mapper = linear_cmap(field_name=varColor, palette=Spectral6, low=min(dfQuery[varColor]), high=max(dfQuery[varColor]))
    for r in myfigure.renderers:
        if glyphName == str(r.name):
            myfigure.renderers.remove(r)
    myfigure.circle(x=varX, y=varY, line_color=mapper, color=mapper, fill_alpha=1, size=2, source=source, name=glyphName)
    barName = "bar" + glyphName
    oldBar = 0
    for r in myfigure.renderers:
        if barName in str(r.name):
            oldBar = r
            # myfigure.renderers.remove(r)
    if oldBar == 0:
        color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0, 0), name=barName)
        myfigure.add_layout(color_bar, 'right')
    push_notebook()
