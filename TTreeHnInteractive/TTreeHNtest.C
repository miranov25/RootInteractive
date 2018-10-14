void TTreeHNtest(){
  tree = AliTreePlayer::LoadTrees("cat mapLong.list","his.*_proj_0_1Dist","XXX",".*","","");
  AliTreePlayer::selectWhatWhereOrderBy(tree,"hisQptTgldAlphaQN80_proj_0_1Dist.mean:sqPtCenter","hisQptTgldAlphaQN80_proj_0_1Dist.entries>0","",0,10000,"csv","mypt.csv");

}