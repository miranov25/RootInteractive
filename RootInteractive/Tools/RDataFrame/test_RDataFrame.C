/*
 gSystem->AddIncludePath("-I$RootInteractive/")
.L $RootInteractive/RootInteractive/Tools/RDataFrame/test_RDataFrame.C+g

*/
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "RootInteractive/Tools/RDataFrame/RDataFrame_TimeSeries.h"
#include "TRandom.h"
#pragma link C++ class map<string,ROOT::VecOps::RVec<double> >+;

void testDevel(){
    int nPoints=1001;
    ROOT::RVec<float>  sampleV(nPoints);
    ROOT::RVec<float>  sampleTime(nPoints);
    sampleTime.resize(nPoints);
    for (int i=0; i<nPoints; i++) {sampleTime[i]=float(i)/nPoints;};
    //ROOT::RVec<float> subSampleTime(sampleTime.data()[int(0.1*nPoints)], 0.9*nPoints);
    ROOT::RVec<float> subSampleTime(&(sampleTime.data()[int(0.1*nPoints)]), 0.8*nPoints);
    float mean = Mean(subSampleTime);
    float rms = StdDev(subSampleTime);
    //float median =
    auto sampleSin = sin(sampleTime*6.28);
    auto sampleCos = cos(sampleTime*6.28);
    std::vector<std::string> statVector={"mean","std","median"};
    auto statMap   = getStat0<double,double>(sampleSin, sampleTime,sampleTime,statVector,0.01);
}

ROOT::RDF::RInterface<ROOT::Detail::RDF::RLoopManager, void> testRDFSeries(int nTracks){
    // nTracks=100
     ROOT::RDataFrame df(nTracks);
     auto rdf = df.Define("nPoints", "size_t(10+gRandom->Rndm()*100)");
     auto vecUni =  [] (size_t nPoints){auto vecTime=ROOT::RVec<double>(nPoints); vecTime.resize(nPoints); for (size_t i=0; i<nPoints; i++) vecTime[i]=float(i)/nPoints; return vecTime;};
     auto vecNoise =  [] (size_t nPoints){auto vecNoise=ROOT::RVec<double>(nPoints); vecNoise.resize(nPoints); for (size_t i=0; i<nPoints; i++) vecNoise[i]=gRandom->Gaus()*0.1; return vecNoise;};
     rdf=  rdf.Define("vecTime",vecUni,{"nPoints"});
     rdf=  rdf.Define("vecNoise",vecNoise,{"nPoints"});
     rdf=  rdf.Define("sinTime","sin(vecTime*TMath::TwoPi())");
     rdf=  rdf.Define("sinTimeN","sinTime+vecNoise");
     rdf=  rdf.Define("range0","float(0.02)");
     //
     {
        std::vector<std::string> statVector={"mean","std","median"};
        rdf=  rdf.Define("statVector",[statVector](){return statVector;});
     }
     rdf=  rdf.Define("statMap",getStat0<double,double>,{"sinTimeN","vecTime","vecTime","statVector","range0"});
     rdf = rdf.Define("sinTimeMean","statMap[\"mean\"]");
     rdf = rdf.Define("sinTimeMedian","statMap[\"median\"]");
     rdf = rdf.Define("sinTimeStd","statMap[\"std\"]");
     rdf.Snapshot("rdfSeries","rdfSeriesTest1.root");
     return rdf;
}


