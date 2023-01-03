/*
  #include "$RootInteractive/RootInteractive/Tools/RDataFrame_TimeSeries.h"
  .L $RootInteractive/RootInteractive/Tools/RDataFrame_TimeSeries.h
*/
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include <map>

#pragma link map<string,ROOT::VecOps::RVec<double> >

/// make convolution of  vecRef at time stamp defined - assuming data are sorted in time/1D space
/// \param vec0       -  input data vector
/// \param vecRef     -  reference date vector
/// \param time0      -  time/1D space input vector
/// \param timeRef    -  time/1D space  for reference  vector
/// \return              ROOT::RVec<float> with convolution around the nearest value
template <typename DataTime, typename DataVal>
auto getConvolution0(const ROOT::RVec<DataVal>& vecRef, const ROOT::RVec<DataTime>& timeRef, const ROOT::RVec<DataTime>& time0, float width, float deltaMax)
{
  bool lower = true;
  int vecSize = time0.size();
  int vecSizeRef = vecRef.size();
  ROOT::RVec<float> vecNearestRef(vecSize);
  ROOT::RVec<float> vecNearestDist(vecSize);
  for (int i = 0; i < vecSize; i++) {
    double timeI = time0[i];
    int indexRef = 0;
    auto lower = std::lower_bound(timeRef.begin(), timeRef.end(), timeI); ///
    int indexRefLower = std::distance(timeRef.begin(), lower);
    float sum = 0;
    float sumW = 0;
    // TODO: indexRefLower: can it get larger than timeRef.size() - I think yes -
    for (int j = indexRefLower; j > 0; j--) {
      float delta = time0[i] - timeRef[j];
      if (abs(delta) > deltaMax) {
        break;
      }
      float weight = TMath::Gaus(delta, 0, width);
      sum += weight * vecRef[j];
      sumW += weight;
    }
    for (int j = indexRefLower; j < vecSizeRef; j++) {
      float delta = time0[i] - timeRef[j];
      if (abs(delta) > deltaMax) {
        break;
      }
      float weight = TMath::Gaus(delta, 0, width);
      sum += weight * vecRef[j];
      sumW += weight;
    }
    vecNearestDist[i] = sumW;
    vecNearestRef[i] = sum / sumW;
  }
  return std::tuple(vecNearestRef, vecNearestDist);
}

template <typename DataTime, typename DataVal>
///
/// \tparam DataTime       - type for the time - should be double
/// \tparam DataVal        - type fot the values - float/double
/// \param vecRef          - reference  measurement to proces
/// \param timeRef         - time/1D space measurement
/// \param time0           - time to exctract local statitical properties
/// \param statVector      - dictionary - statistic to extract  (dictionary of vectors )
///                           - mean, median, std - local
///                           - Robust statistics - ord stat, LTS
///                           - local filters - local weighted fits
///                           - https://bookdown.org/rdpeng/timeseriesbook/filtering-time-series.html
/// \param deltaMax        - local neighborhood to process
/// \return
auto getStat0(const ROOT::RVec<DataVal>& vecRef, const ROOT::RVec<DataTime>& timeRef, const ROOT::RVec<DataTime>& time0, std::vector<std::string> statVector, float deltaMax)
{
  bool lower = true;
  int vecSize = time0.size();
  int vecSizeRef = vecRef.size();
  Long64_t  work[vecSizeRef];
  std::map<string, ROOT::RVec<DataVal>> statMap;
  for (auto   & stat : statVector){
    statMap[stat]=ROOT::RVec<DataVal>(vecSize);
  }
  //
  DataVal* vecRefVal = (DataVal*)vecRef.data();   ///
  for (int i = 0; i < vecSize; i++) {
    double timeI = time0[i];
    auto lower = std::lower_bound(timeRef.begin(), timeRef.end(), timeI); ///
    int indexRefLower = std::distance(timeRef.begin(), lower);
    int indexMin=indexRefLower>=0? indexRefLower:0;
    int indexMax=indexRefLower<vecSizeRef? indexRefLower:vecSizeRef;
    while (timeI<time0[indexMin]+deltaMax && indexMin>0) indexMin--;
    while (timeI>time0[indexMax]-deltaMax && indexMax<vecSizeRef) indexMax++;
    //ROOT::RVec<DataVal> take(&(vecRef[indexMin]),indexMax-indexMin+1);

    const ROOT::RVec<DataVal> rtake( &(vecRefVal[indexMin]),indexMax-indexMin+1);
    for (auto   & stat : statVector){
        if (stat=="mean")    statMap[stat][i]=Mean(rtake);
        if (stat=="std")     statMap[stat][i]=StdDev(rtake);
        if (stat=="median")  statMap[stat][i]=TMath::Median(indexMax-indexMin+1,&(vecRefVal[indexMin]));
    }

  }
  return statMap;
}

