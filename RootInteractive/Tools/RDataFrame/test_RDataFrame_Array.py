# https://root.cern/doc/master/classROOT_1_1RDataFrame.html
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
       #include "DataFormatsTPC/TrackTPC.h"
       #pragma link C++ class ROOT::RVec < ROOT::RVec < char>> + ;
       #pragma link C++ class ROOT::RVec < ROOT::RVec < float>> + ;
       #pragma link C++ class ROOT::RVec < ROOT::RVec < double>> + ;
       #pragma link C++ class ROOT::RVec < ROOT::RVec < long double>> + ;
       #pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<float>>+;
       #pragma link C++ class ROOT::VecOps::RVec <TParticle> + ;  
       #pragma link C++ class ROOT::VecOps::RVec <o2::tpc::TrackTPC> + ;
    """
    print(dictData,  file=open('dict.C', 'w'))
    ROOT.gInterpreter.ProcessLine(".L dict.C+")

def makeTestRDataFrame():
    # 2 hacks - instatiate classes needed for RDataframe
    x=ROOT.TParticle();
    ROOT.gInterpreter.ProcessLine(".L dict.C+")

    ROOT.gInterpreter.Declare("""
        
        o2::tpc::TrackTPC t;
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
        auto makeUnitRVec1DTPCTrack = [](int n){
            auto array = ROOT::RVec<o2::tpc::TrackTPC>(n);
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
    rdf = rdf.Define('array1DTPCTrack', "makeUnitRVec1DTPCTrack(nPoints)")
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrame.root")
    return rdf

def generateASTExample():
    astOut=ast.parse("track.GetP() / mass", mode="eval")
    ast.dump(astOut,True,False)
def dumpAST(expression):
    astOut=ast.parse(expression, mode="eval")
    print(ast.dump(astOut,True,False))


def test_define(logLevel=logging.DEBUG, nCores=0):
    """
    :param rdf:
    :param expression:
    :return:
    1.) make AST
    2.) check get data

    """
    # from RootInteractive.Tools.RDataFrame.test_RDataFrame_Array  import *
    if (nCores>0):
            ROOT.EnableImplicitMT(nCores)
    log = logging.getLogger('')
    log.setLevel(logLevel)
    makeTestDictionary()
    rdf=makeTestRDataFrame()
    # test 0 -  1D delta fixed range -OK
    parsed= makeDefine("arrayD","array1D0[1:10]-array1D2[:20:2]", rdf,3, 0x4)  # working
    rdf = makeDefineRDF("arrayD0", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameD0.root")
    # test 1 -  1D delta auto range -OK
    parsed= makeDefine("arrayDAll","array1D0[:]-array1D2[:]", rdf,3, 0x4)  # working
    rdf = makeDefineRDF("arrayDAall", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameDAll.root")
    # test 2 -  - 1D  function  fix range -OK
    parsed = makeDefine("arrayCos","cos(array1D0[1:10])", rdf,3, 0x4);
    rdf = makeDefineRDF("arrayCos0", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameCos0.root");
    # test 3 -  - 1D  function  full range -OK
    parsed = makeDefine("arrayCosAll","cos(array1D0[:])", rdf,3, 0x4);
    rdf = makeDefineRDF("arrayCosAll", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameCosAll.root");
    # test 4  - 1D member function OK
    parsed = makeDefine("arrayPx","array1DTrack[1:10].Px()", rdf,3, 0x4);
    rdf = makeDefineRDF("arrayPx", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayPx.root");
    # test 4.b  - 1D member function  OK
    parsed = makeDefine("arrayAlpha","array1DTPCTrack[:].getAlpha()", rdf,3, 0x4);
    rdf = makeDefineRDF("arrayAlpha", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayGetAlpha.root");
    # test 4.c  - 1D member function  OK
    rdf=makeDefine("arrayAlpha3","array1DTPCTrack[:].getAlpha()", rdf,3);
    # test 4.d  - 1D member function   - not yet working - need public getter for private and protected
    # parsed = makeDefine("arrayfPdgCode","array1DTrack[:].fPdgCode", rdf,3, 0x4);
    # rdf = makeDefineRDF("arrayfPdgCode", parsed["name"], parsed,  rdf, verbose=1)
    # rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayfPdgCode.root");
    # test 5  - 2D delta auto range
    parsed = makeDefine("arrayD2D","array2D0[1:10,:]-array2D1[1:10,:]", rdf,3, 0x4);
    rdf = makeDefineRDF("arrayD2D", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayD2D.root");
    # test 6  - 2D delta against 1D
    parsed = makeDefine("arrayD1D2D","array2D0[:,:]-array1D0[:]", rdf,3, 0x4);
    rdf = makeDefineRDF("arrayD1D2D", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayD1D2D.root");
    # test 7   - 2D boolen test
    parsed=makeDefine("arrayD20Bool","array2D0[:,:]>0", rdf,3, 0x4);
    rdf = makeDefineRDF("arrayD20Bool", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayD1D2D.root");
    #
    rdf = makeDefine("arrayD1D2DP","array2D0[:,:]+array1D0[:]", rdf,3, False);
    rdf.Describe()
    # Test of invarainces
    # tt.Draw("arrayCosAll:cos(array1D0)","arrayCos0!=0","*")
    # entries = tt.Draw("arrayCosAll-cos(array1D0)","","");
    #  ROOT.TMath.RMS(entries,tt.GetV1())
    # tt.GetHistogram().GetRms()  # should be 0 +- errror  cut e.g 10^-5
    # tt.GetHistogram().GetRms()  # should be 0 +- errorr

    return rdf

def test_exception(rdf):
    # here we should test that the code is not crashing but showing error message and continue
    # this is OK
    rdf = makeDefine("arrayAlpha1", "array1DTPCTrack[:].getAlpha()", rdf, 1)
    # raising properly exception
    rdf = makeDefine("arrayAlphaT1", "array1DTPCTrack[:].getAlpha1()", rdf, 1)
    # this is crashing with seg fault - private attributes should be tested
    rdf = makeDefine("arrayPdgCode", "array1DTrack[:].fPdgCode", rdf, 1)
    # this is crashing with seg fault - private attributes should be tested
    rdf = makeDefine("arraymAlphaT1", "array1DTPCTrack[:].mAlpha", rdf, 1)
    #test new
    parsed = makeDefine("arrayD2D_RDF", "array2D0[1:10,:]-array2D1[1:10,:]", rdf, 3, 0x4);


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

