import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.io import show
from RootInteractive.InteractiveDrawing.bokeh.bokeh3DSurface import Surface3d


def test_Surface3d():
    x = np.arange(0, 300, 10)
    y = np.arange(0, 300, 10)
    xx, yy = np.meshgrid(x, y)
    xx = xx.ravel()
    yy = yy.ravel()
    value = np.sin(xx / 50) * np.cos(yy / 50) * 50 + 50
    colorValue = np.cos(xx / 50)
    colorValue = xx

    source = ColumnDataSource(data=dict(x=xx, y=yy, z=value, colorValue=colorValue))

    surface = Surface3d(x="x", y="y", z="z", style="colorValue", myWidth="100px", data_source=source, width=300,
                        height=300, options3D={"style": "dot-size"})

    show(surface)
