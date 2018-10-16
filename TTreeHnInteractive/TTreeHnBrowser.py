# import
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipywidgets import Button, Layout
import numpy as np
from IPython.display import display
import ROOT
ROOT.gSystem.Load("$ALICE_ROOT/lib/libSTAT.so")


#
# x1(1:2:0.1:1.5:1.75)
# TODO:
#   1.) switch edit mode  with syntax x1(1:2:0.1:1.5:1.75) <-> slider mode
#   2.) syntax for the min,max, mean, median, rms pointer to array needed
class TSliderArray:
    def __init__(self):
        pass

    def addSlider(self, sliderDescription):
        # parse slider description title(<min>:<max>:<step:<valueMin>:<valueMax>)
        sliderDesc0=ROOT.AliParser.ExtractBetween(sliderDescription,'(',')') # tokenize space between ()
        sliderDesc1=sliderDesc0[sliderDesc0.size()-1].Data()          # get last string - assuming slider description
        sliderDesc2=ROOT.AliParser.Split(sliderDesc1,':')
        #
        newSlider=widgets.FloatRangeSlider(description=sliderDescription,layout=Layout(width='66%'))
        if sliderDesc2.size()>0: newSlider.min=sliderDesc2[0].Atof()
        if sliderDesc2.size()>1: newSlider.max=sliderDesc2[1].Atof()
        if sliderDesc2.size()>2: newSlider.step=sliderDesc2[2].Atof()
        if sliderDesc2.size()>4:
            newSlider.value=[sliderDesc2[3].Atof(), sliderDesc2[4].Atof()]
        else:
            newSlider.value=[newSlider.min,newSlider.max]
        newButton=widgets.Button(description='Remove',tooltip="Remove slider")
        enableButton=widgets.ToggleButton(description='Status',tooltip="Enable/disable")
        newBox=widgets.HBox([newSlider,enableButton,newButton],layout=Layout(width='100%'))
        self.fSliderWidgets.children+=(newBox,)
        newButton.on_click(self.removeSlider)
        self.fSliderDictionary.update({newButton:newBox})
    def removeSlider(self,b):
        # remove parent box from the tuple
        # TODO - check with ipythone team if no side effect
        box=self.fSliderDictionary.get(b)
        box.close()
        self.fSliderWidgets.children=[x for x in self.fSliderWidgets.children  if x !=box]  #tuple  build again removing box

    fSliderWidgets=widgets.VBox(layout=Layout(width='100%'))
    fSliderDictionary={'':''}





# TTree browser - using the python interactive widgets in Jupyter notebook
#
class TTreeHnBrowser:
    def __init__(self):
        pass
    def initTree(self,tree):
        self.friendListROOT = tree.GetListOfFriends()
        self.branchListROOT = tree.GetListOfBranches()
        self.aliasListROOT = tree.GetListOfAliases()
        self.funUpdateList()
        self.addQueryButton.on_click(self.funAddQuery)
        self.addSelectionButton.on_click(self.funAddSelection)
        self.addSliderButton.on_click(self.funAddSlider)
        self.regExpButton.on_click(self.funUpdateMask)
        self.drawSliderButton.on_click(self.funAddNewSlider)
        #self.branchMaskWidget.observe(funUpdateMask,branchMaskWidget.value) # looks like not working
        return 0
    def testInit(self):
        self.tree=ROOT.AliTreePlayer.LoadTrees("cat mapLong.list","his.*_proj_0_1Dist","$#",".*","","")
        self.initTree(self.tree)

    def funUpdateList(self):          # update list according to the current RegExp selection
        friendMaskReg=ROOT.TPRegexp(self.friendRegExp)
        branchMaskReg=ROOT.TPRegexp(self.branchRegExp)
        self.friendList[:]=['']
        for i in range(0,self.friendListROOT.GetEntries()):
            if friendMaskReg.GetPattern().Length() > 0:
                if friendMaskReg.Match(self.friendListROOT.At(i).GetName()):
                    self.friendList.append(self.friendListROOT.At(i).GetName())
            else:
                self.friendList.append(self.friendListROOT.At(i).GetName())
        self.branchList[:]=['']
        for i in range(0,self.branchListROOT.GetEntries()):
            if branchMaskReg.GetPattern().Length() > 0:
                if branchMaskReg.Match(self.branchListROOT.At(i).GetName()):
                    self.branchList.append(self.branchListROOT.At(i).GetName())
            else:
                self.branchList.append(self.branchListROOT.At(i).GetName())
        #for i in range(0,self.aliasListROOT.GetEntries()):
        #    self.aliasList.append(self.aliasListROOT.At(i).GetName())
        self.friendDropDown.options=self.friendList
        self.branchDropDown.options=self.branchList

    def funAddQuery(self,b):
        self.drawQuery.value+=self.friendDropDown.value
        self.drawQuery.value+='.'
        self.drawQuery.value+=self.branchDropDown.value
    def funAddSelection(self, b):
        self.drawCut.value+=self.friendDropDown.value
        self.drawCut.value+='.'
        self.drawCut.value+=self.branchDropDown.value
    def funAddSlider(self, b):
        self.drawSlider.value+=self.friendDropDown.value
        self.drawSlider.value+='.'
        self.drawSlider.value+=self.branchDropDown.value
    def funUpdateMask(self, b):
        self.branchRegExp=str(self.branchRegExpWidget.value)
        self.friendRegExp=str(self.friendRegExpWidget.value)
        #print(branchMask)
        self.funUpdateList()
    def funAddNewSlider(self, b):
        newSlider=widgets.FloatRangeSlider(description=str(self.drawSlider.value),layout=Layout(width='66%'))
        self.sliderWidgets.children+=(newSlider,)


    # data members
    # root tree
    tree=ROOT.TTree()
    friendListROOT=[]
    branchListROOT=[]
    aliasListROOT=[]
    # tree variables selection
    friendList=['']
    branchList=[]
    aliasList=['']
    friendRegExp='.*'
    branchRegExp='.*'
    aliasRegExp='.*'
    # regular expression selection
    friendRegExpWidget=widgets.Text(description='Friend RegExp', value=friendRegExp)
    branchRegExpWidget=widgets.Text(description='Branch RegExp',value=branchRegExp)
    aliasRegExpWidget=widgets.Text(description='Alias RegExp')
    regExpButton=widgets.Button(description='RegExp',tooltip="RegExp")
    # dropdonw list of varaibles
    friendDropDown=widgets.Dropdown( options=friendList, description='Friend list:')
    branchDropDown=widgets.Dropdown( options=branchList, description='Branch list:')
    aliasDropDown=widgets.Dropdown( options=aliasList, description='Alias list:')
    # export buttons
    addQueryButton=widgets.Button(description='Add Query',tooltip="Add query")
    addSelectionButton=widgets.Button(description='Add selection',tooltip="Add selection")
    addSliderButton=widgets.Button(description='Add slider',tooltip="Add slider")
    # tree expression Text box
    drawQuery=widgets.Text(description='Query',layout=Layout(width='66%'))
    drawQueryButton=widgets.Button(description='Add query',tooltip="Add/test query")
    drawQueryBox=widgets.HBox([drawQuery,drawQueryButton])
    drawCut=widgets.Text(description='Selection',layout=Layout(width='66%'))
    drawSlider=widgets.Text(description='Slider',layout=Layout(width='66%'))
    drawSliderButton=widgets.Button(description='Append slider',tooltip="Append slider")
    sliderWidgets=widgets.VBox(layout=Layout(width='100%'));
    drawSliderBox=widgets.HBox([drawSlider,drawSliderButton])
    #
    boxMask=widgets.HBox([friendRegExpWidget,branchRegExpWidget, aliasRegExpWidget,regExpButton]);
    boxSelect=widgets.HBox([friendDropDown,branchDropDown, aliasDropDown]);
    boxButton=widgets.HBox([addQueryButton,addSelectionButton,addSliderButton]);
    boxDraw=widgets.VBox([drawCut,drawQueryBox,drawSliderBox,sliderWidgets], width=1000);
    #boxSlider=widgets.VBox(sliderWidgets);
    boxAll=widgets.VBox([boxMask,boxSelect,boxButton,boxDraw])

