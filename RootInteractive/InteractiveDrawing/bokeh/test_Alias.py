from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
from bokeh.plotting import output_file
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.random_sample(size=(100000, 2)), columns=list('XY'))

aliasArray = [
    {
        "name": "saxpy", 
        "v_func": """
            let result = []
            for(let i=0; i<x.length; i++){
                result.push(a*x[i]+y[i])
            }
            return result
        """,
        "fields": ["a", "x", "y"]
    },
    {
        "name": "saxpy_v2", 
        "func": """
            return a*x+y
        """,
        "fields": ["a", "x", "y"]
    }
]

parameterArray = [
 {"name": "paramA", "value":1, "range":[0, 10]}
]

figureArray = [
    [["X"], ["Y", "10*X+Y", "saxpy(paramA, X, Y)"]],
    [["X"], ["Y", "10*X+Y", "saxpy_v2(paramA, X, Y)"]]
]

figureLayoutDesc=[
        [0, 1]
        ]

widgetParams=[
    ['range', ['X']],
    ['range', ['Y']],
    ['slider',["paramA"], {"callback": "parameter"}],
]

widgetLayoutDesc={
    "Selection": [[0, 1, 2], [3, 4], [5, 6],[7,8], {'sizing_mode': 'scale_width'}],
    "Graphics": [[9, 10, 11], {'sizing_mode': 'scale_width'}]
    }

def test_Alias():
    output_file("test_Alias.html")
    xxx=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, widgetLayout=widgetLayoutDesc, parameterArray=parameterArray, aliasArray=aliasArray)
