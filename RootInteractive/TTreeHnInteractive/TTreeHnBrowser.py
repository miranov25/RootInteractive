# import
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipywidgets import Button, Layout
import numpy as np
from IPython.display import display
from bqplot import *
import bqplot.pyplot as pyplt
import re
import sys
import pandas as pd
import qgrid
from RootInteractive.Tools.aliTreePlayer import *
#
from bokeh.layouts import *
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, show

ROOT.gSystem.Load("$ALICE_ROOT/lib/libSTAT.so")


def readCSV(fName, withHeader):
    """
    read cvs file as in the root format together with header
    :param fName:             input file
    :param withHeader:        reading as named numpy array or simple numpy array
    :return:                  named/simple numpy array
    """
    line = open(fName).readline().rstrip()  # get header q- using root cvs convention
    line.replace("/D:", "/<f8:")
    names = line.split(":")  # split header line
    variables = []
    for a in names: variables.append(a.split('/')[0])  #
    if withHeader:
        csv = np.genfromtxt(fName, delimiter="\t", skip_header=1, names=variables)
    else:
        csv = np.genfromtxt(fName, delimiter="\t", skip_header=1)
    return csv


def readDataFrame(fName):
    line = open(fName).readline().rstrip()  # get header - using root cvs convention
    names = line.replace("/D", "").replace("/I", "").split(":")
    variables = []
    for a in names: variables.append(a.split('\t')[0])  #
    dataFrame = pd.read_csv(fName, sep='\t', index_col=False, names=variables, skiprows=1)
    return dataFrame


# Slider array:
#   x1(1:2:0.1:1.5:1.75)
# TODO:
#   1.) switch edit mode  with syntax x1(1:2:0.1:1.5:1.75) <-> slider mode
#   2.) syntax for the min,max, mean, median, rms pointer to array needed
class TSliderArray:
    """
    TSlider array  - vertical box with dynamic array of sliders for Jupyter ipywidget
    slider+active button+remove button
    Example usage:
        sliderArray.addSlider("x1(1:2:0.1:1.5:1.75)")
    """

    def __init__(self):
        pass

    def addSlider(self, sliderDescription):
        # parse slider description title(<min>:<max>:<step:<valueMin>:<valueMax>)
        sliderDescription0 = ROOT.AliParser.ExtractBetween(sliderDescription, '(', ')')  # tokenize space between ()
        sliderDescription1 = sliderDescription0[sliderDescription0.size() - 1].Data()  # get last string - assuming slider description
        variableName = sliderDescription.replace("(" + sliderDescription1 + ")", "")
        sliderDescription2 = ROOT.AliParser.Split(sliderDescription1, ':')
        #
        newSlider = widgets.FloatRangeSlider(description=variableName, layout=Layout(width='66%'))
        newSlider.description = variableName
        if sliderDescription2.size() > 0: newSlider.min = sliderDescription2[0].Atof()
        if sliderDescription2.size() > 1: newSlider.max = sliderDescription2[1].Atof()
        if sliderDescription2.size() > 2: newSlider.step = sliderDescription2[2].Atof()
        if sliderDescription2.size() > 4:
            newSlider.value = [sliderDescription2[3].Atof(), sliderDescription2[4].Atof()]
        else:
            newSlider.value = [newSlider.min, newSlider.max]
        newButton = widgets.Button(description='Remove', tooltip="Remove slider")
        enableButton = widgets.ToggleButton(description='Status', tooltip="Enable/disable")
        newBox = widgets.HBox([newSlider, enableButton, newButton], layout=Layout(width='100%'))
        self.fSliderWidgets.children += (newBox,)
        self.fVariableList.append(variableName)
        newButton.on_click(self.removeSlider)
        self.fSliderDictionary.update({newButton: newBox})

    def removeSlider(self, b):
        # remove parent box from the tuple
        # TODO - check with ipython team if no side effect
        box = self.fSliderDictionary.get(b)
        box.close()
        self.fSliderWidgets.children = [x for x in self.fSliderWidgets.children if x != box]  # tuple  build again removing box

    def queryDataFrame(self, dataFrame):
        '''
        query python data frame
        :return: result of query
        '''
        query = ""
        for box in self.fSliderWidgets.children:
            query0 = box.children[0].description + "<" + str(box.children[0].value[1]) + "&" + box.children[0].description + ">" + str(box.children[0].value[0])
            query += query0 + "&"
        query = query[:-1]
        df2 = dataFrame.query(query)
        return df2

    fSliderWidgets = widgets.VBox(layout=Layout(width='100%'))  # slider widget
    fSliderDictionary = {'': ''}  # dictionary button->box
    fVariableList = []  # variable list


#
#
class TDrawVarArray:
    """
    TDrawVarArray
    """

    def __init__(self):
        pass

    def addVariables(self, variableList):
        for var in variableList: self.addVariable(var)

    def addVariable(self, variableDescription):
        """
        addVariable
        :param variableDescription:
        :return:
        """
        variables = widgets.Text(description='', value=variableDescription, layout=Layout(width='66%'))
        newButton = widgets.Button(description='Remove', tooltip="Remove slider")
        enableButton = widgets.ToggleButton(description='Status', tooltip="Enable/disable")
        newBox = widgets.HBox([variables, enableButton, newButton], layout=Layout(width='100%'))
        self.fDrawVarWidgets.children += (newBox,)
        self.fDrawVarDictionary.update({newButton: newBox})
        for val in variables.value.split(":"): self.fVariableList.append(val)
        newButton.on_click(self.removeVariable)

    def removeVariable(self, b):
        box = self.fDrawVarDictionary.get(b)
        for a in box.children[0].value.split(":"): self.fVariableList.remove(a)
        # self.fVariableList.remove(str(box.children[0].value).split(":"))
        box.close()
        self.fDrawVarWidgets.children = [x for x in self.fDrawVarWidgets.children if x != box]  # tuple  build again removing box

    fDrawVarWidgets = widgets.VBox(layout=Layout(width='100%'))
    fDrawVarDictionary = {'': ''}
    fVariableList = []


# TTree browser - using the python interactive widgets in Jupyter notebook

class TTreeHnBrowser:
    def __init__(self):
        self.addQueryButton.on_click(self.funAddQuery)
        self.addSelectionButton.on_click(self.funAddSelection)
        self.addSliderButton.on_click(self.funAddSlider)
        self.regExpButton.on_click(self.funUpdateMask)
        self.drawSliderButton.on_click(self.funAddNewSlider)
        self.drawQueryButton.on_click(self.funAddNewQuery)
        self.loadSelectionButton.on_click(self.loadDataInMemory)

    def initTree(self, tree):
        """
        initialize browser using tree ()
        :param tree:
        :return:
        """
        self.fTree = tree
        self.friendListROOT = tree.GetListOfFriends()
        self.branchListROOT = tree.GetListOfBranches()
        self.aliasListROOT = tree.GetListOfAliases()
        self.funUpdateList()
#        if ROOT.TStatToolkit.GetMetadata(tree): self.loadMetadata(tree)
        if GetMetadata(tree): self.loadMetadata(tree)
        return 0

    def loadDataInMemory(self, b):
        """
        Load numpy array with selected entries from tree into memory
        Temporary data.csv file created
        TODO: make in memory transformation
        :return: success stratus
        """
        varSet = set()
        for var in self.sliderArray.fVariableList:  varSet.add(var)
        for var in self.drawVarArray.fVariableList:  varSet.add(var)
        variables = ""
        for var in varSet: variables += var + ":"
        value = ROOT.AliTreePlayer.selectWhatWhereOrderBy(self.fTree, str(variables), str(self.drawSelection.value), "", 0, 10000000, "csvroot", "data.csv")
        print(value)
        if value <= 0: return value
        self.fDataFrame = readDataFrame('data.csv')
        self.boxAll.children = [x for x in self.boxAll.children if x != self.fQgrid]
        self.fQgrid = qgrid.show_grid(self.fDataFrame)
        self.boxAll.children += (self.fQgrid,)

    def loadMetadata(self, tree):
        """ Load sliders according description in tree metadata
        """
#        treeMeta = ROOT.TStatToolkit.GetMetadata(tree)
        treeMeta = GetMetadata(tree)
        for index in range(treeMeta.GetEntries()):
            j = treeMeta.At(index)
            if ROOT.TString(j.GetName()).Contains(".Slider"):
                slider = str(j.GetName())
                slider = re.sub(".Slider$", "", slider)
                slider += j.GetTitle()
                self.sliderArray.addSlider(slider)

    def funUpdateList(self):
        """
        update lists according to  RegExp selections
        :return:
        """
        friendMaskReg = ROOT.TPRegexp(self.friendRegExp)
        branchMaskReg = ROOT.TPRegexp(self.branchRegExp)
        self.friendList[:] = ['']
        if (self.friendListROOT != None):
            for i in range(0, self.friendListROOT.GetEntries()):
                if friendMaskReg.GetPattern().Length() > 0:
                    if friendMaskReg.Match(self.friendListROOT.At(i).GetName()):
                        self.friendList.append(self.friendListROOT.At(i).GetName())
                else:
                    self.friendList.append(self.friendListROOT.At(i).GetName())
        self.branchList[:] = ['']
        for i in range(0, self.branchListROOT.GetEntries()):
            if branchMaskReg.GetPattern().Length() > 0:
                if branchMaskReg.Match(self.branchListROOT.At(i).GetName()):
                    self.branchList.append(self.branchListROOT.At(i).GetName())
            else:
                self.branchList.append(self.branchListROOT.At(i).GetName())
        # for i in range(0,self.aliasListROOT.GetEntries()):
        #    self.aliasList.append(self.aliasListROOT.At(i).GetName())
        self.friendDropDown.options = self.friendList
        self.branchDropDown.options = self.branchList

    def funAddQuery(self, b):
        self.drawQuery.value += self.friendDropDown.value
        self.drawQuery.value += '.'
        self.drawQuery.value += self.branchDropDown.value

    def funAddSelection(self, b):
        self.drawSelection.value += self.friendDropDown.value
        self.drawSelection.value += '.'
        self.drawSelection.value += self.branchDropDown.value

    def funAddSlider(self, b):
        self.drawSlider.value += self.friendDropDown.value
        self.drawSlider.value += '.'
        self.drawSlider.value += self.branchDropDown.value

    def funUpdateMask(self, b):
        self.branchRegExp = str(self.branchRegExpWidget.value)
        self.friendRegExp = str(self.friendRegExpWidget.value)
        # print(branchMask)
        self.funUpdateList()

    def funAddNewSlider(self, b):
        self.sliderArray.addSlider(self.drawSlider.value)
        with self.fOut:
            print('funAddNewSlider')

    def funAddNewQuery(self, b):
        self.drawVarArray.addVariable(self.drawQuery.value)
        with self.fOut:
            print('funAddNewQuery')

    # data members
    # root tree
    tree = ROOT.TTree()
    friendListROOT = []
    branchListROOT = []
    aliasListROOT = []
    # tree variables selection
    friendList = ['']
    branchList = []
    aliasList = ['']
    friendRegExp = '.*'
    branchRegExp = '.*'
    aliasRegExp = '.*'
    # regular expression selection
    friendRegExpWidget = widgets.Text(description='Friend RegExp', value=friendRegExp)
    branchRegExpWidget = widgets.Text(description='Branch RegExp', value=branchRegExp)
    aliasRegExpWidget = widgets.Text(description='Alias RegExp')
    regExpButton = widgets.Button(description='RegExp', tooltip="RegExp")
    # drop-down list of variables
    friendDropDown = widgets.Dropdown(options=friendList, description='Friend list:')
    branchDropDown = widgets.Dropdown(options=branchList, description='Branch list:')
    aliasDropDown = widgets.Dropdown(options=aliasList, description='Alias list:')
    # tree expression Text box
    drawSelection = widgets.Text(description='Selection', layout=Layout(width='66%'))
    addSelectionButton = widgets.Button(description='Copy selection', tooltip="Copy selection")
    loadSelectionButton = widgets.Button(description='Load selection', tooltip="Load selected entries from tree to memory")
    drawSelectionBox = widgets.HBox([drawSelection, addSelectionButton, loadSelectionButton])
    #
    drawQuery = widgets.Text(description='Query', layout=Layout(width='66%'))
    addQueryButton = widgets.Button(description='Copy Query', tooltip="Copy query")
    drawQueryButton = widgets.Button(description='Append query', tooltip="Append query")
    drawQueryBox = widgets.HBox([drawQuery, addQueryButton, drawQueryButton])
    #
    drawSlider = widgets.Text(description='Slider', layout=Layout(width='66%'))
    addSliderButton = widgets.Button(description='Copy slider', tooltip="Copy slider")
    drawSliderButton = widgets.Button(description='Append slider', tooltip="Append slider")
    drawSliderBox = widgets.HBox([drawSlider, addSliderButton, drawSliderButton])
    #
    boxMask = widgets.HBox([friendRegExpWidget, branchRegExpWidget, aliasRegExpWidget, regExpButton])
    boxSelect = widgets.HBox([friendDropDown, branchDropDown, aliasDropDown])
    # boxButton = widgets.HBox([addQueryButton, addSelectionButton, addSliderButton])
    boxDraw = widgets.VBox([drawQueryBox, drawSliderBox, drawSelectionBox], width=1000)
    sliderArray = TSliderArray()
    drawVarArray = TDrawVarArray()
    fTree = ROOT.TTree()
    fDataFrame = 0
    fQgrid = 0
    fOut = widgets.Output()
    boxAll = widgets.VBox([boxMask, boxSelect, boxDraw, drawVarArray.fDrawVarWidgets, sliderArray.fSliderWidgets, fOut])
