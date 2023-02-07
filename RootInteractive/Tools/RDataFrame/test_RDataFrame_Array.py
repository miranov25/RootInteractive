# from RootInteractive.Tools.RDataFrame.test_RDataFrame_Array  import *
from RootInteractive.Tools.RDataFrame.RDataFrame_Array  import *
import pprint
import ROOT
import ast
import awkward as ak
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
       #pragma link C++ class ROOT::RVec < ROOT::RVec < long double>> + ;
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
    # test 0 -  1D delta fixed range -OK
    parsed= makeDefine("arrayD","array1D0[1:10]-array1D2[:20:2]", rdf,3, True)  # working
    rdf = makeDefineRDFv2("arrayD0", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameD0.root")
    # test 1 -  1D delta auto range -OK
    parsed= makeDefine("arrayDAll","array1D0[:]-array1D2[:]", rdf,3, True)  # working
    rdf = makeDefineRDFv2("arrayDAall", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameDAll.root")
    # test 2 -  - 1D  function  fix range -OK
    parsed = makeDefine("arrayCos","cos(array1D0[1:10])", rdf,3, True);
    rdf = makeDefineRDFv2("arrayCos0", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameCos0.root");
    # test 3 -  - 1D  function  full range -OK
    parsed = makeDefine("arrayCosAll","cos(array1D0[:])", rdf,3, True);
    rdf = makeDefineRDFv2("arrayCosAll", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameCosAll.root");
    # test 4  - 1D member function failing
    parsed = makeDefine("arrayPx","array1DTrack[1:10].Px()", rdf,3, True);
    rdf = makeDefineRDFv2("arrayPx", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayPx.root");
    #
    parsed = makeDefine("arrayD2D", "array2D0[1:10,:]-array2D1[1:10,:]", rdf, 3, True)
    rdf = makeDefineRDFv1("arrayD2D", parsed["name"], parsed, rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameD2D.root")
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


def makeDefineRDFv1(parsed, rdf):
    ROOT.gInterpreter.Declare( parsed["implementation"])
    rdf.Describe();                            ## workimg
    ROOT.gInterpreter.ProcessLine("arrayD")    ## prinintg implementation
    ROOT.gInterpreter.ProcessLine("ROOT::RDF::RInterface<ROOT::Detail::RDF::RLoopManager, void>  *rdfgener_prdf=0;")            ## add to global space
    ROOT.gInterpreter.ProcessLine("rdfgener_prdf");              # shows 0 points
    ROOT.rdfgener_prdf=rdf
    ROOT.gInterpreter.ProcessLine("rdfgener_prdf->Describe()");  ## print content of the rdf

    defineLine="""
        rdfgener_prdf->Describe().Print();
        auto rdfOut=rdfgener_prdf->Define("arrayDOut1",arrayD,{"array1D0", "array1D2"});
        rdfgener_prdf=&rdfOut;
        rdfgener_prdf->Describe().Print();   
    """
    print(defineLine)
    ROOT.gInterpreter.ProcessLine(defineLine)




def makeDefineRDFv2(columnName, funName, parsed,  rdf, verbose=1):
    if verbose & 0x1:
        print(f"{columnName}\t{funName}\t{parsed}")
    # 0.) Define function if does not exist yet

    try:
        ROOT.gInterpreter.Declare( parsed["implementation"])
    except:
        pass
    # 1.) set  rdf to ROOT space - the RDataFrame_Array should be owner
    try:
         ROOT.rdfgener_rdf
    except:
        ROOT.gInterpreter.ProcessLine("ROOT::RDF::RInterface<ROOT::Detail::RDF::RLoopManager, void>  *rdfgener_rdf=0;")            ## add to global space
    if ROOT.rdfgener_rdf!=rdf:
        ROOT.rdfgener_rdf=rdf
    # 2.) be verbose
    if verbose&0x2:
        rdf.Describe().Print();                                      ## workimg
        ROOT.gInterpreter.ProcessLine("rdfgener_rdf->Describe()");  ## print content of the rdf
    #
    dependency="{"+f'{parsed["dependencies"]}'[1:-1]+"}".replace("'","\"")
    dependency=dependency.replace("'","\"")

    defineLine=f"""
        auto rdfOut=rdfgener_rdf->Define("{columnName}",{funName},{dependency});
        rdfgener_rdf=&rdfOut;   
    """
    if verbose>0:
        print(defineLine)

    ROOT.gInterpreter.ProcessLine(defineLine)
    return  ROOT.rdfgener_rdf

