from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar
# from bokeh.palettes import *
from bokeh.transform import *
from bokehTools import *
from bokeh.layouts import *
from bokeh.palettes import *
from bokeh.io import push_notebook


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


def testExample():
    df2 = treeBrowser.sliderArray.queryDataFrame(treeBrowser.fDataFrame)
    df2 = df2.query("EN")
    source = ColumnDataSource(df2)
    # Use the field name of the column source
    mapper = linear_cmap(field_name='Z', palette=Spectral6, low=min(df2["Z"]), high=max(df2["Z"]))
    p = figure(plot_width=500, plot_height=500, title="XXX")
    p.circle(x='E', y='dEdx', line_color=mapper, color=mapper, fill_alpha=1, size=2, source=source)
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0, 0))
    p.add_layout(color_bar, 'right')
    # p.y_axis_type="log"
    show(p)
