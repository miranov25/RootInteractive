## Dummy README https://github.com/miranov25/RootInteractive/issues/177


## Dependencies
* pytest json +jq install
  * https://github.com/mattcl/pytest-json
    ``` 
    pip install --target=/data/miranov/JIRA/ATO-500/tensorflow_latest-gpuCUDA/usr/local/lib/python3.6/dist-packages pytest-json-report 
    pip install --target=/data/miranov/JIRA/ATO-500/tensorflow_latest-gpuCUDA/usr/local/lib/python3.6/dist-packages jq
    ```
  * we can use other 3 plugins  but the syntax is different  
    * https://pypi.org/project/pytest-json-report/ 
    * this one looks like more active ...
* Data regression pytest-regressions
  * https://readthedocs.org/projects/pytest-regressions/downloads/pdf/latest/
    ```
    pip install pytest-regressions --target=/data/miranov/JIRA/ATO-500/tensorflow_latest-gpuCUDA/usr/local/lib/python3.6/dist-packages
    ```
* Data dir pytest-datadir 
  ```
    pip install --target=/data/miranov/JIRA/ATO-500/tensorflow_latest-gpuCUDA/usr/local/lib/python3.6/dist-packages pytest-datadir   
  ```
* pytest-parallel
```angular2html
   pip install --target=/data/miranov/JIRA/ATO-500/tensorflow_latest-gpuCUDA/usr/local/lib/python3.6/dist-packages pytest-parallel   

```

## Example usage

### Simple dump to the test_bokehAll.json
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
...
```
### Parallel dump to json 
* specify number of parallel jobs
* add metadata to json
```
time pytest --workers 6 --metadata nCPU  5  --tests-per-worker 20 test_Compression.py --json-report --json-report-file test_Compression.json
```

## JQ filters manuals and links

### Blog explaining how it works  (tried)
To learn to make filters following blogs an manuals recommended (tried)
I tried after having first experience
* https://earthly.dev/blog/jq-select/


### Run jq tutorial (recomended above)
https://github.com/rjz/jq-tutorial
tutorial to download from the github

* "objects" completed with a gold star! 
* ★ "mapping" completed with a gold star!
* "filtering" completed with a gold star!
* ★ "output" completed with a gold star!
★ "reduce" completed with a gold star!



### Try the query interactivelly:
* get your document into clibpoard (e.g)
  ```
   cat test_Compression.json | jq .| clipboard
  ``` 
* copy to interactive web page  e.g: https://jqterm.com/
* exercise filters
* Example filter:
* https://jqterm.com/7e2ce231a4bfcbb4a1025e3c04f97ae5?query=%5B%7Benvironment%3A.environment.Platform%2C%20%22id%22%3A.tests%5B%5D.nodeid%2C%20%22tests%22%3A%20.tests%5B%5D.call%7D%5D%7C.%5B%5D%7C%5B.environment%2C.id%2C.tests.duration%5D%0A
```
[{environment:.environment.Platform, "id":.tests[].nodeid, "tests": .tests[].call}]|.[]|[.environment,.id,.tests.duration]
```
