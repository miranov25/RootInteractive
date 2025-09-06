import numpy as np
import pandas as pd
import base64
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import output_file, figure
from bokeh.io import show
from bokeh.models.layouts import Column

from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
from RootInteractive.InteractiveDrawing.bokeh.CDSAlias import CDSAlias
from RootInteractive.InteractiveDrawing.bokeh.DownsamplerCDS import DownsamplerCDS
from RootInteractive.Tools.compressArray import arrayCompressionRelative16
from RootInteractive.InteractiveDrawing.bokeh.OrtFunction import OrtFunction
from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate import getDefaultVarsDiff
from RootInteractive.InteractiveDrawing.bokeh.bokehTools import mergeFigureArrays

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

df = pd.DataFrame({
        "A": X_test_T[0],
        "B": X_test_T[1],
        "C": X_test_T[2],
        "D": X_test_T[3],
        "y_true": y_test,
        "y_pred_skl": y_pred_skl
    })

df2_train = pd.DataFrame(np.random.random_sample(size=(50000, 4)), columns=list('ABCD'))
df2_train["y"] = df2_train.A - df2_train.B + np.random.normal(0, 0.05, df2_train.shape[0])
df2 = pd.DataFrame(np.random.random_sample(size=(1000000, 4)), columns=list('ABCD'))
df2["y_true"] = df2.A - df2.B + np.random.normal(0, 0.05, df2.shape[0])
rfr = RandomForestRegressor(n_estimators=10, max_depth=3)
rfr.fit(df2_train[["A", "B", "C", "D"]], df2_train["y"])
df2["y_pred_server_rf10"] = rfr.predict(df2[["A", "B", "C", "D"]])
initial_type = [('float_input', FloatTensorType([None, 4]))]
onx_rfr = convert_sklearn(rfr, initial_types=initial_type)
s = onx_rfr.SerializeToString()
onx_rfr_b64 = base64.b64encode(s).decode('utf-8')
rfr50 = RandomForestRegressor(n_estimators=50, max_depth=5)
rfr50.fit(df2_train[["A", "B", "C", "D"]], df2_train["y"])
df2["y_pred_server_rf50"] = rfr50.predict(df2[["A", "B", "C", "D"]])
onx_rfr50 = convert_sklearn(rfr50, initial_types=initial_type)
s = onx_rfr50.SerializeToString()
onx_rfr50_b64 = base64.b64encode(s).decode('utf-8')
ridgeReg = Ridge(alpha=0.1)
ridgeReg.fit(df2_train[["A", "B", "C", "D"]], df2_train["y"])
df2["y_pred_server_ridge"] = ridgeReg.predict(df2[["A", "B", "C", "D"]])
initial_type = [('float_input', FloatTensorType([None, 4]))]
onx_ridge = convert_sklearn(ridgeReg, initial_types=initial_type)
s = onx_ridge.SerializeToString()
onx_ridge_b64 = base64.b64encode(s).decode('utf-8')


def test_onnx_bokehDrawArray():
    output_file("test_ort_web_bokehDrawSA.html", "Test ONNX runtime web")
    figureArray = [
        [["A"], ["y_pred_skl"], {"size":10, "color":"blue", "legend_label":"Predicted by scikit learn"}],
        [["A"], ["y_pred_client"], {"size":10, "color":"red", "legend_label":"Predicted by ONNX"}]
    ]
    widgetParams = [["multiSelect", ["y_pred_skl"]]]
    parameterArray = []
    jsFunctionArray = [{"name": "ort_func_js","v_func":onx_b64,"type":"onnx"}]
    aliasArray = [{"name": "y_pred_client","transform":"ort_func_js","variables": {"float_input":["A","B","C","D"]},"out":"output_label"}]
    figureLayoutDesc=[[0,1, {"height":400}], {"sizing_mode":"scale_width"}]
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, parameterArray=parameterArray,
                          jsFunctionArray=jsFunctionArray, widgetLayout = [[0]],
                           aliasArray=aliasArray)
    
def test_onnx_templateWeights():
    output_file("test_ort_web_template.html", "Test ONNX runtime web - using template")
    aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVarsDiff(variables=["A", "B", "C", "D", "y_true", "y_pred_skl", "y_pred_client", "y_pred_skl == y_pred_client"], weights=[None, "A>.5", "B>C"], multiAxis="weights")
    jsFunctionArray = [{"name": "ort_func_js","v_func":onx_b64,"type":"onnx"}]
    aliasArray += [{"name": "y_pred_client","transform":"ort_func_js","variables": {"float_input":["A","B","C","D"]},"out":"output_label"}]
    widgetParams = mergeFigureArrays(widgetParams, [["multiSelect", ["y_pred_skl"],{"name":"y_pred_skl"}]])
    widgetLayoutDesc["Select"] += ["y_pred_skl"]
    bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, parameterArray=parameterArray,
                          jsFunctionArray=jsFunctionArray, widgetLayout = widgetLayoutDesc, histogramArray=histoArray, 
                           aliasArray=aliasArray)
    
def test_onnx_multimodels():
    output_file("test_ort_web_multimodels.html", "Test ONNX runtime web - using multiple models")
    aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVarsDiff(variables=["A", "B", "C", "D",
            "y_true", "y_pred_server_rf10", "y_pred_server_rf50", "y_pred_server_ridge", "y_pred_client_rf10", "y_pred_client_rf50", "y_pred_client_ridge",
            "y_pred_customjs_ridge", "y_pred_customjs_ridge_naive", "y_pred_server_rf10 == y_pred_client_rf10"], weights=[None, "A>.5", "B>C"], multiAxis="weights")
    jsFunctionArray = [
        {"name": "ort_func_js_rf10","v_func":onx_rfr_b64,"type":"onnx"},
        {"name": "ort_func_js_rf50","v_func":onx_rfr50_b64,"type":"onnx"},
        {"name": "ort_func_js_ridge","v_func":onx_ridge_b64,"type":"onnx"}
    ]
    aliasArray += [
        {"name": "y_pred_client_rf10","transform":"ort_func_js_rf10","variables": {"float_input":["A","B","C","D"]},"out":"variable"},
        {"name": "y_pred_client_rf50","transform":"ort_func_js_rf50","variables": {"float_input":["A","B","C","D"]},"out":"variable"},
        {"name": "y_pred_client_ridge","transform":"ort_func_js_ridge","variables": {"float_input":["A","B","C","D"]},"out":"variable"},
        {"name": "y_pred_customjs_ridge","v_func":"""
            if($output == null || $output.length !== A.length){
                $output = new Float64Array(A.length)
            }
            $output.fill(intercept)
            for(let i=0; i<$output.length; i++){
                $output[i] += coefs[0]*A[i]
            }
            for(let i=0; i<$output.length; i++){
                $output[i] += coefs[1]*B[i]
            }
            for(let i=0; i<$output.length; i++){
                $output[i] += coefs[2]*C[i]
            }
            for(let i=0; i<$output.length; i++){
                $output[i] += coefs[3]*D[i]
            }
            return $output
         """, "parameters":{"intercept":ridgeReg.intercept_, "coefs":ridgeReg.coef_}, "fields":["A","B","C","D"]},
         {"name":"y_pred_customjs_ridge_naive","func":"return intercept + coefs[0]*A + coefs[1]*B + coefs[2]*C + coefs[3]*D","parameters":{"intercept":ridgeReg.intercept_, "coefs":ridgeReg.coef_}, "variables":["A","B","C","D"]}
    ]
    widgetParams = mergeFigureArrays(widgetParams, [["range", ["A"],{"name":"A"}],["range",["B"],{"name":"B"}]])
    widgetLayoutDesc["Select"] += [["A","B"]]
    bokehDrawSA.fromArray(df2, None, figureArray, widgetParams, layout=figureLayoutDesc, parameterArray=parameterArray,
                          jsFunctionArray=jsFunctionArray, widgetLayout = widgetLayoutDesc, histogramArray=histoArray, 
                           aliasArray=aliasArray)

test_onnx_multimodels()