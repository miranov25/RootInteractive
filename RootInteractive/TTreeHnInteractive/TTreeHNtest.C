void TTreeHNtest(){
  tree = AliTreePlayer::LoadTrees("cat mapLong.list","his.*_proj_0_1Dist","XXX",".*","","");
  AliTreePlayer::selectWhatWhereOrderBy(tree,"hisQptTgldAlphaQN80_proj_0_1Dist.mean:hisQptTgldAlphaQN80_proj_0_1Dist.rms:sqPtCenter","hisQptTgldAlphaQN80_proj_0_1Dist.entries>0","",0,10000,"csvroot","data.csv");

}


