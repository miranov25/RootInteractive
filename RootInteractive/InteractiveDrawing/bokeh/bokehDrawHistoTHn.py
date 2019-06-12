from IPython.core.display import display
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.glyphs import Quad
from bokeh.io import push_notebook
from ipywidgets import *
import pyparsing

class drawHisto(object):
    def __init__(self, histograms, selection, **options):
        """
        :param histograms:      TObjArray consists of multidimensional TTree's ( THnT )
                                    for detailed info check:
                                        https://root.cern.ch/doc/master/classTObjArray.html
                                        https://root.cern.ch/doc/master/classTHnT.html

        :param selection:       String, list of the  histograms to draw. Separated by commas (,)
                                Dimensions to draw must be included inside of parenthesis' as an integer.
                                Ex:
                                    "hisdZ(3),hisdZ(0),hisPullZHM1(3),hisPullZCL(1)"

        :param options:         -ncols        :the number of columns
                                -tooltips     :tooltips to show
                                -plot_width   :the width of each plot
                                -plot_height  :the height of each plot
                                -bg_color     :Background color
                                -color        :Histogram color
                                -line_color   :Line color
 


        """
        
        self.selectionList = parseSelectionString(selection)
        self.histArray = histograms
        self.sliderList = []
        self.sliderNames = []
        self.initSlider("")
        WidgetBox = widgets.VBox(self.sliderList)

        self.figure, self.handle, self.source = self.drawGraph(**options)
        self.updateInteractive("")
        display(WidgetBox)

    def updateInteractive(self, b):
        for hisTitle, projectionList in zip(*[iter(self.selectionList)] * 2):
            for iDim in range(self.histArray.FindObject(hisTitle).GetNdimensions() - 1,-1,-1):
                iSlider = self.sliderNames.index(self.histArray.FindObject(hisTitle).GetAxis(iDim).GetTitle())
                value = self.sliderList[iSlider].value
                self.histArray.FindObject(hisTitle).GetAxis(iDim).SetRangeUser(value[0], value[1])
        iterator = 0
        for hisTitle, projectionList in zip(*[iter(self.selectionList)] * 2):
            dimList = list(map(int, projectionList))
            nDim = len(dimList)
            if nDim > 1:
                raise NotImplementedError("Sorry!!.. Multidimensional projections have not been implemented, yet")
            histogram = self.histArray.FindObject(hisTitle).Projection(dimList[0])
            binsLowEdge = []
            binsUpEdge = []
            top = []
            bottom = []
            for i in range(1, histogram.GetNbinsX() + 1):
                binsLowEdge.append(histogram.GetXaxis().GetBinLowEdge(i))
                binsUpEdge.append(histogram.GetXaxis().GetBinUpEdge(i))
                top.append(histogram.GetBinContent(i))
                bottom.append(0)
            newSource = ColumnDataSource(dict(
                left=binsLowEdge,
                right=binsUpEdge,
                top=top,
                bottom=bottom
            ))
            self.source[iterator].data = newSource.data
            iterator = iterator + 1
        push_notebook(self.handle)

    def initSlider(self, b):
        for hisTitle, projectionList in zip(*[iter(self.selectionList)] * 2):
            for iDim in range(self.histArray.FindObject(hisTitle).GetNdimensions() - 1,-1,-1):
                axis = self.histArray.FindObject(hisTitle).GetAxis(iDim)
                title = axis.GetTitle()
                if title not in self.sliderNames:
                    maxRange = axis.GetXmax()
                    minRange = axis.GetXmin()
                    nBin = axis.GetNbins()
                    step = (maxRange - minRange) / nBin
                    slider = makeSlider(title, minRange, maxRange, step)
                    slider.observe(self.updateInteractive, names='value')
                    self.sliderList.append(slider)
                    self.sliderNames.append(title)

    def drawGraph(self, **kwargs):
        
        # define default options
        options = {
            'nCols': 2,
            'tooltips': 'pan,box_zoom, wheel_zoom,box_select,lasso_select,reset',
            'y_axis_type': 'auto',
            'x_axis_type': 'auto',
            'plot_width': 400,
            'plot_height': 400,
            'bg_color': '#fafafa',
            'color' : "navy",
            'line_color' : "white"
        }
        options.update(kwargs)
        isNotebook = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        p = []
        source = []
        iterator = 0
        for hisTitle, projectionList in zip(*[iter(self.selectionList)] * 2):
            dimList = list(map(int, projectionList))
            nDim = len(dimList)
            if nDim > 1:
                raise NotImplementedError("Sorry!!.. Multidimensional projections have not been implemented, yet")
            histogram = self.histArray.FindObject(hisTitle).Projection(dimList[0])
            binsLowEdge = []
            binsUpEdge = []
            top = []
            bottom = []
            for i in range(1, histogram.GetNbinsX() + 1):
                binsLowEdge.append(histogram.GetXaxis().GetBinLowEdge(i))
                binsUpEdge.append(histogram.GetXaxis().GetBinUpEdge(i))
                top.append(histogram.GetBinContent(i))
                bottom.append(0)
            histLabel = histogram.GetTitle()
            xLabel = histogram.GetXaxis().GetTitle()
            yLabel = histogram.GetYaxis().GetTitle()
            source.append(ColumnDataSource(dict(
                left=binsLowEdge,
                right=binsUpEdge,
                top=top,
                bottom=bottom
            )  )  )
            localHist = figure(title=histLabel, tools=options['tooltips'], background_fill_color=options['bg_color'], y_axis_type=options['y_axis_type'], x_axis_type=options['x_axis_type'])
            glyph = Quad(top="top", bottom="bottom", left="left", right="right", fill_color=options['color'], line_color=options['line_color'])
            localHist.add_glyph(source[iterator], glyph)
            localHist.y_range.start = 0
            localHist.xaxis.axis_label = xLabel
            localHist.yaxis.axis_label = yLabel
            p.append(localHist)
            iterator = iterator + 1
            pAll = gridplot(p, ncols=options['nCols'], plot_width=options['plot_width'], plot_height=options['plot_height'])
        handle = show(pAll, notebook_handle=True)
        return pAll, handle, source


def makeSlider(title, minRange, maxRange, step):
    slider = widgets.FloatRangeSlider(description=title, layout=Layout(width='66%'), min=minRange, max=maxRange,
                                      step=step, value=[minRange, maxRange])
    return slider


def parseSelectionString(selectionString):
    toParse = "(" + selectionString + ")"
    theContent = pyparsing.Word(pyparsing.alphanums + ".+-") | '#' | pyparsing.Suppress(',')
    selectionParser = pyparsing.nestedExpr('(', ')', content=theContent)
    selectionList = selectionParser.parseString(toParse)[0]
    return selectionList
