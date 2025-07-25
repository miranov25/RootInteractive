import numpy as np
import pandas as pd
import base64
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import output_file, figure
from bokeh.io import show
from bokeh.models.layouts import Column

from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
from RootInteractive.InteractiveDrawing.bokeh.bokehTools import bokehDrawArray
from RootInteractive.InteractiveDrawing.bokeh.CDSAlias import CDSAlias
from RootInteractive.InteractiveDrawing.bokeh.DownsamplerCDS import DownsamplerCDS
from RootInteractive.Tools.compressArray import arrayCompressionRelative16
from RootInteractive.InteractiveDrawing.bokeh.OrtFunction import OrtFunction

# Prepare model
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

clr = LogisticRegression()
clr.fit(X_train, y_train)
print(clr)

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)

y_pred_skl = clr.predict(X_test)

s = onx.SerializeToString()
onx_b64 = base64.b64encode(s).decode('utf-8')

X_test_T = X_test.T

print(X_test_T)
print(y_test)

def test_onnx_base64():
    ort_func_js = OrtFunction(v_func = onx_b64)
    cds = ColumnDataSource(data={
        "A": X_test_T[0],
        "B": X_test_T[1],
        "C": X_test_T[2],
        "D": X_test_T[3],
        "y_true": y_test,
        "y_pred_skl": y_pred_skl
    })

    cds_derived = CDSAlias(source=cds, mapping={
        "y_pred_client":{
            "transform":ort_func_js,
            "out_idx": 0,
            "fields": {"float_input":["A","B","C","D"]}
        }
    }, columnDependencies=[ort_func_js])

    cds_shown = DownsamplerCDS(source=cds_derived)

    print(cds.data)
    output_file("test_ort_web.html", "Test ONNX runtime web")
    f_skl = figure(title="Reference true vs predicted by scikit learn", width=800, height=600)
    f_skl.scatter(x="y_true", y="y_pred_skl", source=cds_shown, color="blue", legend_label="Predicted by scikit learn")
    f_client = figure(title="ONNX Runtime Client", width=800, height=600)
    f_client.scatter(x="y_true", y="y_pred_client", source=cds_shown, color="red", legend_label="Predicted by ONNX")
    show(Column(f_skl, f_client))

test_onnx_base64()