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
            for (size_t i=0; i<n; i++) array[i].setZ(gRandom->Gaus());
            return array;
        ;};
        auto makeUnitRVec1DTPCTrackSorted = [](int n){
            auto array = makeUnitRVec1DTPCTrack(n);
            std::sort(array.begin(), array.end(), [](o2::tpc::TrackTPC &x, o2::tpc::TrackTPC &y){return x.getZ() < y.getZ();});
            return array;
        };
        auto makeRVecPermutation = [](int n){
            auto array = ROOT::RVec<size_t>(n);
            array.resize(n);
            for (size_t i=0; i<n; i++) array[i]=i;
            return array;
        ;};
        auto scatter = [](ROOT::RVec<size_t> x){
            auto array = ROOT::RVec<size_t>(x.size());
            for(size_t i=0; i<x.size(); i++) array[x[i]] = i;
            return array;
            ;};
        auto is_arange = [](ROOT::RVec<size_t> x){
            char acc = 1;
            for(auto i=x.begin(); i<x.end(); ++i){
              acc &= (*i != i-x.begin());
              };
            return acc;
        ;};
    """)
    #
    nTracks=50
    df= ROOT.RDataFrame(nTracks);
    rdf=df.Define("nPoints", "int(40 + gRandom->Rndm() * 200)")
    rdf=rdf.Define("nPoints2", "int(40 + gRandom->Rndm() * 200)")
    #rdf= rdf.Define('array', 'ROOT::RVecF(mean)')
    rdf= rdf.Define("array1D0","makeUnitRVec1D(nPoints)")
    rdf= rdf.Define("array1D2","makeUnitRVec1D(nPoints)")
    rdf= rdf.Define("array2D0","makeUnitRVec2D(nPoints,nPoints2)")
    rdf= rdf.Define("array2D1","makeUnitRVec2D(nPoints,nPoints2)")
    rdf= rdf.Define('array1DTrack',"makeUnitRVec1DTrack(nPoints)")
    rdf = rdf.Define('array1DTPCTrack2',"makeUnitRVec1DTPCTrackSorted(nPoints2)")
    rdf = rdf.Define('array1DTPCTrack', "makeUnitRVec1DTPCTrack(nPoints)")
    rdf = rdf.Define('arrayPermutation', "makeRVecPermutation(nPoints)")
    rdf = rdf.Define('arrayPermutationInverse', "scatter(arrayPermutation)")
    rdf = rdf.Define('index_tracksT0V0', "arrayPermutation")
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
    Example use;
    from RootInteractive.Tools.RDataFrame.test_RDataFrame_Array  import *
    rdf, lib = test_define(logging.DEBUG, 0)
    """

    if (nCores>0):
            ROOT.EnableImplicitMT(nCores)
    log = logging.getLogger('')
    log.setLevel(logLevel)
    makeTestDictionary()
    rdf=makeTestRDataFrame()
    cppLibrary={}
    # test 0 -  1D delta fixed range -OK
    parsed= makeDefine("arrayD","array1D0[1:10]-array1D2[:20:2]",rdf, cppLibrary,3, 0x4)  # working
    rdf = makeDefineRNode("arrayD0", parsed["name"], parsed,  rdf, verbose=1)

    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameD0.root")
    #return rdf
    # test 1 -  1D delta auto range -OK
    parsed= makeDefine("arrayDAll","array1D0[:]-array1D2[:]",rdf, cppLibrary,3, 0x4)  # working
    rdf = makeDefineRNode("arrayDAall", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameDAll.root")
    # test 2 -  - 1D  function  fix range -OK
    parsed = makeDefine("arrayCos","cos(array1D0[1:10])",rdf, cppLibrary,3, 0x4);
    rdf = makeDefineRNode("arrayCos0", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameCos0.root");
    # test 3 -  - 1D  function  full range -OK
    parsed = makeDefine("arrayCosAll","cos(array1D0[:])",rdf, cppLibrary,3, 0x4);
    rdf = makeDefineRNode("arrayCosAll", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameCosAll.root");
    # test 4  - 1D member function OK
    parsed = makeDefine("arrayPx","array1DTrack[1:10].Px()",rdf, cppLibrary,3, 0x4);
    rdf = makeDefineRNode("arrayPx", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayPx.root");
    # test 4.b  - 1D member function  OK
    parsed = makeDefine("arrayAlpha","array1DTPCTrack[:].getAlpha()",rdf, cppLibrary,3, 0x4);
    rdf = makeDefineRNode("arrayAlpha", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayGetAlpha.root");
    # test 4.c  - 1D member function  OK
    rdf=makeDefine("arrayAlpha3","array1DTPCTrack[:].getAlpha()", rdf, cppLibrary, 3);
    # test 4.d  - 1D member function   - not yet working - need public getter for private and protected
    # parsed = makeDefine("arrayfPdgCode","array1DTrack[:].fPdgCode",rdf, cppLibrary,3, 0x4);
    # rdf = makeDefineRNode("arrayfPdgCode", parsed["name"], parsed,  rdf, verbose=1)
    # rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayfPdgCode.root");
    # test 5  - 2D delta auto range
    parsed = makeDefine("arrayD2D","array2D0[1:10,:]-array2D1[1:10,:]",rdf, cppLibrary,3, 0x4);
    rdf = makeDefineRNode("arrayD2D", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayD2D.root");
    # test 6  - 2D delta against 1D
    parsed = makeDefine("arrayD1D2D","array2D0[:,:]-array1D0[:]",rdf, cppLibrary,3, 0x4);
    rdf = makeDefineRNode("arrayD1D2D", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayD1D2D.root");
    #
    parsed = makeDefine("arrayD2D1D","-array1D0[:]+array2D0[:,:]",rdf, cppLibrary,3, 0x4);
    rdf = makeDefineRNode("arrayD2D1DD", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayD2D1D.root");
    # test 7   - 2D boolen test
    parsed=makeDefine("arrayD20Bool","array2D0[:,:]>0",rdf, cppLibrary,3, 0x4);
    rdf = makeDefineRNode("arrayD20Bool", parsed["name"], parsed,  rdf, verbose=1)
    rdf.Snapshot("makeTestRDataFrame","makeTestRDataFrameArrayD1D2D.root");
    #
    rdf = makeDefine("arrayD1D2DP","array2D0[:,:]+array1D0[:]", rdf, cppLibrary,3);
    rdf.Describe();
    makeLibrary(cppLibrary,"cppRDFtest.C", includes="")
    # Test of invarainces
    # tt.Draw("arrayCosAll:cos(array1D0)","arrayCos0!=0","*")
    # entries = tt.Draw("arrayCosAll-cos(array1D0)","","");
    #  ROOT.TMath.RMS(entries,tt.GetV1())
    # tt.GetHistogram().GetRms()  # should be 0 +- errror  cut e.g 10^-5
    # tt.GetHistogram().GetRms()  # should be 0 +- errorr

    return rdf,cppLibrary

def test_define2(rdf):
    # support for the TMath:: functions
    rdf = makeDefine("array2D0_cos0", "cos(array2D0[0,:])", rdf, None, 3);         # this is working
    rdf = makeDefine("array2D0_cos1", "TMath.Cos(array2D0[0,:])", rdf, None, 3);   # this is working
    rdf = makeDefine("array2D0_cos_diff", "array2D0_cos0[:]-array2D0_cos1[:]", rdf, None, 3)
    assert abs(rdf.Mean("array2D0_cos_diff").GetValue()) < 1e-5
    # support for the operator [index]
    rdf = makeDefine("array2D0_0", "array2D0[0,:]", rdf, None, 3);
    #rdf = makeDefine("array2D0_0", "array2D0[0]", rdf, None, 3);       # should return 1D RVec at position 0, now it is failing
    rdf = makeDefine("arrayJoin_0", "arrayPermutation[arrayPermutationInverse[:]]", rdf, None, 3)
    ### todo
    rdf3 = rdf.Define("inv_PermSum", "is_arange(arrayJoin_0)")
    inv_PermSum= rdf3.Histo1D("inv_PermSum").GetSum(); # raise error if not 0
    assert inv_PermSum == 0

    # test array "tracks", array "V0", index_tracksTOV0, N:N   index_tracksTOV0, - better to do not create Alice object can be tested with root classes
    rdf = makeDefine("arrayJoin_0_Func", "array1DTrack[arrayPermutation[:]].Px()", rdf, None, 3)
    # test array "tracks", array "collisions" both has key for  which the distance can be used
    #         track.getZ(),   collision.getZ()
    # ???? upper and lower bound resp, nearest - syntax to be defined
    # rdf = makeDefine("arrayJoin_0", "nearest(tracks[:],collisions,track.getZ(),collision.getZ())", rdf, None, 3);
    rdf = makeDefine("arrayJoin_1", "upperBound(array1DTPCTrack2, array1DTPCTrack[:], lambda x,y: x.getZ() < y.getZ())", rdf, None, 3);
    rdf = makeDefine("arrayJoin_2", "lowerBound(array1DTPCTrack2, array1DTPCTrack[:], lambda x,y: x.getZ() < y.getZ())", rdf, None, 3);
    # rdf = makeDefine("arrayJoin_3", "inrange(tracks[:],collisions,track.getZ(),collision.getZ(),min,max)", rdf, None, 3);
    rdf = makeDefine("inv_arrayJoin", "arrayJoin_1[:] > arrayJoin_2[:] or arrayJoin_1[:] >= 0 and array1DTPCTrack2[arrayJoin_1[:]] < array1DTPCTrack[:]", rdf, None, 3)
    assert rdf.Sum("inv_arrayJoin").GetValue() == 0
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
    parsed = makeDefine("arrayD2D_RDF", "array2D0[1:10,:]-array2D1[1:10,:]", rdf, 3, 0x4,cppLibrary);

