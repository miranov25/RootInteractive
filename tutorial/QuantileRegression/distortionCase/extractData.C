void extractData(){
    chain = AliXRDPROOFtoolkit::MakeChainRandom("timeSeriesData.list", "distortions", 0, 100);

    TString branchList=""
   for (Int_t i=0; i<chain->GetListOfBranches()->GetEntries(); i++){
       if (TString(chain->GetListOfBranches()->At(i)->GetName()).Contains(".")) continue;
       branchList+=chain->GetListOfBranches()->At(i)->GetName();
       if (i<chain->GetListOfBranches()->GetEntries()-1)branchList+=":";
   }
   AliTreePlayer::selectWhatWhereOrderBy(chain,branchList,"1","",0,1000,"csvroot","distortion.csv");

}