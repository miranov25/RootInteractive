# from RootInteractive.Tools.RDataFrame.test_RDataFrame_Array  import *
from RootInteractive.Tools.RDataFrame.RDataFrame_Array  import *

import ROOT
import ast
#import RDataFrame_Array

import pprint
def makeTestDictionary():
    dictData = """
       #include "ROOT/RDataFrame.hxx"
       #include "ROOT/RVec.hxx"
       #include "ROOT/RDF/RInterface.hxx"
       #include  "TParticle.h"
       #pragma link C++ class ROOT::RVec < ROOT::RVec < float>> + ;
       #pragma link C++ class ROOT::RVec < ROOT::RVec < double>> + ;
       #pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<float>>+;
       #pragma link C++ class ROOT::VecOps::RVec <TParticle> + ;  
    """
    print(dictData,  file=open('dict.C', 'w'))
    ROOT.gInterpreter.ProcessLine(".L dict.C+")

def makeTestRDataFrame():
    # 2 hacks - instatiate classes needed for RDataframe
    x=ROOT.TParticle();
    ROOT.gInterpreter.ProcessLine(".L dict.C+")

    ROOT.gInterpreter.Declare("""
        auto makeUnitRVec1D = [](int n){
            auto array = ROOT::RVecF(n);
            array.resize(n);
            for (size_t i=0; i<n; i++) array[i]=i;
            return array;
        ;};
        auto makeUnitRVec2D = [](int n1, int n2){
            auto array2D = ROOT::RVec<ROOT::RVec<float>>(n1);
            array2D.resize(n2);
            for (size_t i=0; i<n1; i++) {
                array2D[i].resize(n2);
                for (size_t j=0; j<n2; j++) array2D[i][j]=i+j;
                }
            return array2D;
        ;};
        auto makeUnitRVec1DTrack = [](int n){
            auto array = ROOT::RVec<TParticle>(n);
            array.resize(n);
            //for (size_t i=0; i<n; i++) array=i;
            return array;
        ;};
    """)
    #
    nTracks=100
    df= ROOT.RDataFrame(nTracks);
    rdf=df.Define("nPoints", "int(40 + gRandom->Rndm() * 200)")
    rdf=rdf.Define("nPoints2", "int(40 + gRandom->Rndm() * 200)")
    #rdf= rdf.Define('array', 'ROOT::RVecF(mean)')
    rdf= rdf.Define("array1D0","makeUnitRVec1D(nPoints)")
    rdf= rdf.Define("array1D2","makeUnitRVec1D(nPoints)")
    rdf= rdf.Define("array2D0","makeUnitRVec2D(nPoints,nPoints2)")
    rdf= rdf.Define("array2D1","makeUnitRVec2D(nPoints,nPoints2)")
    rdf= rdf.Define('array1DTrack',"makeUnitRVec1DTrack(nPoints)")
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrame.root")
    return rdf

def generateASTExample():
    astOut=ast.parse("track.GetP() / mass", mode="eval")
    ast.dump(astOut,True,False)
def dumpAST(expression):
    astOut=ast.parse(expression, mode="eval")
    print(ast.dump(astOut,True,False))


def test_define1D(rdf, name, expression,verbosity):
    """
    :param rdf:
    :param expression:
    :return:
    1.) make AST
    2.) check get data

    """
    makeTestDictionary()
    rdf=makeTestRDataFrame()
    #
    rdf2 = makeDefine("arrayD","array1D0[1:10]-array1D2[:20:2]", rdf,3, True);
    rdf2 = makeDefine("arrayCos","cos(array1D0[1:10])", rdf,3, True);
    rdf2 = makeDefine("arrayAbs","TMath.Abs(array1D0[1:10])", rdf,3, True);  # this is failing - has many possible return values


    rdf2 = makeDefine("arrayPx","array1DTrack[1:10].Px()", rdf,3, True);

    #

    #
    #rdfNew=rdf
    return rdf

def getClassMethod(className, methodName):
    """
    TODO:  this is a hack - we should get return method description
    return class Mmthod information
    :param className:
    :param methodName:
    :return:  type of the method if exist
    className = "AliExternalTrackParam" ; methodName="GetX"
    """
    import re
    try:
        docString= eval(f"ROOT.{className}.{methodName}.func_doc")
        returnType = re.sub(f"{className}.*","",docString)
        return (returnType,docString)
    except:
        pass
    return ("","")
def makeFun():
    ROOT.gInterpreter.Declare(""" 
     ROOT::VecOps::RVec<double> arrayD(ROOT::VecOps::RVec<float> &array1D0, ROOT::VecOps::RVec<float> &array1D2){
    ROOT::VecOps::RVec<double> result(10);
    for(size_t i=0; i<10; i++){
        result[i] = (array1D0[1+i*1]) - (array1D2[0+i*2]);
    }
    return result;
} 

   """)


