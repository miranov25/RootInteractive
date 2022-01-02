## Dummy README https://github.com/miranov25/RootInteractive/issues/177


## Dependencies
* pytest json +jq install
  * https://github.com/mattcl/pytest-json
    ``` 
    pip install --target=/data/miranov/JIRA/ATO-500/tensorflow_latest-gpuCUDA/usr/local/lib/python3.6/dist-packages pytest-json-report 
    pip install --target=/data/miranov/JIRA/ATO-500/tensorflow_latest-gpuCUDA/usr/local/lib/python3.6/dist-packages jq
    ```
* Data regression pytest-regressions
  * https://readthedocs.org/projects/pytest-regressions/downloads/pdf/latest/
    ```
    pip install pytest-regressions --target=/data/miranov/JIRA/ATO-500/tensorflow_latest-gpuCUDA/usr/local/lib/python3.6/dist-packages
    ```
    
## Example usage

```
pytest --json-report --json-report-file  test_bokehAll.json
jq '.tests[] | .nodeid, .call.duration'  test_bokehAll.json
```
-->
```
...
RootInteractive/InteractiveDrawing/bokeh/test_Compression.py::test_compressCDSPipe"
8.266445378001663
"RootInteractive/InteractiveDrawing/bokeh/test_Compression.py::test_CompressionCDSPipeDraw"
10.080579579996993
"RootInteractive/InteractiveDrawing/bokeh/test_bokehClientHistogram.py::testBokehClientHistogram"
12.836207999003818
"RootInteractive/InteractiveDrawing/bokeh/test_bokehClientHistogram.py::testBokehClientHistogramOnlyHisto"
7.535426512011327
"RootInteractive/InteractiveDrawing/bokeh/test_bokehClientHistogram.py::testBokehClientHistogramProfileA"
15.859309413004667
...
```


