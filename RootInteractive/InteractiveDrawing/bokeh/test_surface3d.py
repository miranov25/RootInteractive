import numpy as np
from bokeh.models import ColumnDataSource, Slider, CustomJS
from bokeh.io import show
from RootInteractive.InteractiveDrawing.bokeh.bokehVisJS3DGraph import BokehVisJSGraph3D
from bokeh.layouts import gridplot, row
from bokeh.plotting import figure, output_file
from random import random


def test_Surface3d():
    output_file("test_Surface3d.html")

    x = np.arange(0, 300, 10)
    y = np.arange(0, 300, 10)
    xx, yy = np.meshgrid(x, y)
    xx = xx.ravel()
    yy = yy.ravel()
    value = np.sin(xx / 50) * np.cos(yy / 50) * 50 + 50
    colorValue = np.cos(xx / 50)
    colorValue = xx

    source = ColumnDataSource(data=dict(x=xx, y=yy, z=value, colorValue=colorValue))

    surface = BokehVisJSGraph3D(x="x", y="y", z="z", style="colorValue", data_source=source, width=300,
                                height=300, options3D={"style": "surface"})

    show(surface)


def test_Surface3d_sliders():
    output_file("test_Surface3d_sliders.html")

    x = np.arange(0, 300, 10)
    y = np.arange(0, 300, 10)
    xx, yy = np.meshgrid(x, y)
    xx = xx.ravel()
    yy = yy.ravel()
    value = np.sin(xx / 50) * np.cos(yy / 50) * 50 + 50
    colorValue = np.cos(xx / 50)
    colorValue = xx

    source = ColumnDataSource(data=dict(x=xx, y=yy, z=value, colorValue=colorValue))

    surface = BokehVisJSGraph3D(x="x", y="y", z="z", style="colorValue", data_source=source, width=300,
                                height=300, options3D={"style": "surface"})

    sliderX = Slider(start=0, end=10, value=1, step=.1, title="FrequencyX")
    sliderY = Slider(start=0, end=10, value=1, step=.1, title="FrequencyY")
    callback = CustomJS(args=dict(source=source, sliderX=sliderX, sliderY=sliderY), code="""
        let data = source.data;
        let fx = sliderX.value
        let fy = sliderY.value
        let x = data['x']
        let y = data['y']
        let z = data['z']
        for (var i = 0; i < x.length; i++) {
            z[i] = Math.sin(x[i]*fx / 50) * Math.cos(y[i]*fy / 50) + 50
        }
        source.change.emit();
    """)
    sliderY.js_on_change("value_throttled", callback)
    sliderX.js_on_change("value_throttled", callback)
    show(gridplot([[surface], [sliderX], [sliderY]]))


def test_Surface3d_select():
    output_file("test_Surface3d_select.html")

    x = np.random.rand(500) * 300
    y = np.random.rand(500) * 300
    z = np.sin(x / 50) * np.cos(y / 50) * 50 + 50
    colorValue = z

    s1 = ColumnDataSource(data=dict(x=x, y=y, z=z, colorValue=colorValue))
    s2 = ColumnDataSource(data=dict(x=[0], y=[0], z=[0], colorValue=[0]))

    plot1 = figure(plot_width=400, plot_height=400, tools="lasso_select", title="Select Here")
    plot1.circle('x', 'y', source=s1, alpha=0.6)

    plot2 = BokehVisJSGraph3D(x="x", y="y", z="z", style="colorValue", data_source=s2, width=300,
                              height=300, options3D={"style": "dot_color", "xMin": 0, "xMax": 300,
                                                     "yMin": 0, "yMax": 300})

    s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2), code="""
        var inds = cb_obj.indices;
        var d1 = s1.data;
        var d2 = s2.data;
        d2['x'] = []
        d2['y'] = []
        d2['z'] = []
        d2['colorValue'] = []
        for (var i = 0; i < inds.length; i++) {
            d2['x'].push(d1['x'][inds[i]])
            d2['y'].push(d1['y'][inds[i]])
            d2['z'].push(d1['z'][inds[i]])
            d2['colorValue'].push(d1['colorValue'][inds[i]])
        }
        s2.change.emit();
    """))

    show(row(plot1, plot2))
