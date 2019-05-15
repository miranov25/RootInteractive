
from TTreeHnInteractive.TTreeHnBrowser import *
from InteractiveDrawing.bokeh.bokehDraw import *
from InteractiveDrawing.bokeh.bokehDrawPanda import *
#output_notebook()
import ROOT
ROOT.gSystem.Load("$ALICE_ROOT/lib/libSTAT.so")
from Tools.aliTreePlayer import *
import pyparsing
from bokeh.io import curdoc
curdoc().theme = 'caliber'

ROOT.TFile.SetCacheFileDir("../data/")
treeQA=ROOT.AliTreePlayer.LoadTrees("echo https://aliqat.web.cern.ch/aliqat/qcml/data/2018/LHC18q/trending_merged_LHC18q_withStatusTree.root",".*",".*sta.*",".*","","")
treeQA.RemoveFriend(treeQA.GetFriend("Tstatus"))
ROOT.TStatToolkit.AddMetadata(treeQA,"chunkBegin.isTime","1")
ROOT.TStatToolkit.AddMetadata(treeQA,"chunkMedian.isTime","1")
treeQA.RemoveFriend(treeQA.GetFriend("tpcQA"))


varDraw="(meanMIP,meanMIPele):meanMIPele:resolutionMIP"
varX="meanMIP:chunkMedian:chunkMedian"
tooltips=[("MIP","(@meanMIP)"),  ("Electron","@meanMIPele"), ("Global status","(@global_Outlier,@global_Warning)"), \
          ("MIP status(Warning,Outlier,Acc.)","@MIPquality_Warning,@MIPquality_Outlier,@MIPquality_PhysAcc")]
widgets="tab.sliders(slider.meanMIP(45,55,0.1,45,55),slider.meanMIPele(50,80,0.2,50,80), slider.resolutionMIP(0,0.15,0.01,0,0.15)),"
widgets+="tab.checkboxGlobal(slider.global_Warning(0,1,1,0,1),checkbox.global_Outlier(0)),"
widgets+="tab.checkboxMIP(slider.MIPquality_Warning(0,1,1,0,1),checkbox.MIPquality_Outlier(0), checkbox.MIPquality_PhysAcc(1))"
layout="((0),(1),(2,x_visible=1),commonX=2,x_visible=1,y_visible=0,plot_height=250,plot_width=1000)"
xxx=bokehDraw(treeQA,"meanMIP>0","chunkMedian",varDraw,"MIPquality_Warning",widgets,0,commonX=1,size=6,tooltip=tooltips,x_axis_type='datetime',layout=layout)

layout= '((0,commonX=0),(1),(2,x_visible=1),commonX=2,x_visible=1,y_visible=0,plot_height=250,plot_width=1000)'
#xxx=bokehDraw(treeQA,"meanMIP>0",varX,"meanMIPele:meanMIPele:resolutionMIP","MIPquality_Warning",widgets,0,size=6,tooltip=tooltips,x_axis_type='datetime',layout=layout)


xxx=bokehDraw(treeQA,"meanMIP>0",varX,"meanMIPele:meanMIPele:resolutionMIP","MIPquality_Warning",widgets,0,size=6,tooltip=tooltips,layout=layout)

