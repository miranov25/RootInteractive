import jq
import json
import pandas as pd
import regex as re

jdata=0
def example0():
    """
    this is example jq query
    :return:
    """
    input="/home2/miranov/github/RootInteractive/RootInteractive/InteractiveDrawing/bokeh/testAll.json"
    with open(input, "r") as f: jdata = json.load(f)
    jquery='[{environment:.environment.Platform,nCPU:.environment.nCPU, "id":.tests[].nodeid, "test": .tests[].call,"duration":.tests[].call.duration}]|.[]|{environment,id,duration,nCPU}'
    arrayIn=jq.compile(jquery).input(jdata).all()
    arrayOut = pd.DataFrame(arrayIn)
    arrayOut.sample(10).head(3)
    arrayIn=jq.compile('{environment:.environment,"call":.tests[].call,"id":.tests[].nodeid}').input(jdata).all()
    arrayOut = pd.DataFrame(arrayIn)

#
# https://unix.stackexchange.com/questions/561460/how-to-print-path-and-key-values-of-json-file

def queryStrings():
    pathJQ0='[paths | map(.|tostring) | join(".")]'
    #  jq  '[paths | map(.|tostring) | join(".")| gsub("(.\\d+)"; "[]")]' testAll.json
    pathJQ1=r'[paths | map(.|tostring) | join(".")| gsub("(.\\d+)"; "[]")]'
    pathJQ2=r'[paths | map(.|tostring) | join(".")| gsub("(.\\d+)"; "[]")]|unique'
    pathJQ="""
    paths(scalars) as $p
  | [ ( [ $p[] | tostring ] | join(".") )
    , ( getpath($p) | tojson )
    ]
  | join(": ")
    """

def jsonToPanda(arrayIn, varArray):
    varArray=[".*ize.*"]
    regExpArray=[]
    for regExp in varArray:
        regExpArray.append(re.compile(regExp))

    pathJQ2=r'[paths | map(.|tostring) | join(".")| gsub("(.\\d+)"; "[]")]|unique'
    arrayPath0=jq.compile(pathJQ2).input(jdata).all()
    arrayPath=[]
    #query=???
    for path in arrayPath0[0]:
        for regExp in regExpArray:
            if re.match(regExp,path):
                print(path)
