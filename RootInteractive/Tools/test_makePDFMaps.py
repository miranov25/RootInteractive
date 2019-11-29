from makePDFMaps import *
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *


#


def randomData(nPoints=100000):
    df = pd.DataFrame(np.random.random_sample(size=(nPoints, 4)), columns=list('ABCD'))
    df["noise"] = np.random.normal(0, 0.1, nPoints)
    df["csin"] = np.sin(6.28 * df["C"])
    df["valueOrig"] = df["A"] + np.exp(0 * df["B"]) * df["csin"]
    df["value"] = df["valueOrig"] + df["noise"]
    return df


def test_makePDF():
    df = randomData(1000000)
    histo = makeHistogram(df, "value:A:B:C:D:#A>0>>myhisto(40,-20,20,40,0,1,30,0,1,50,0,1,10,0,1)")

    slices = ((0, 40, 1, 0), (10, 30, 1, 4), (10, 24, 5, 3), (1, 49, 3, 1), (0, 10, 3, 0))
    dimI = 0
    dframe = makePdfMaps(histo, slices, dimI)
    dframe["csin"] = np.sin(6.28 * dframe["CBinCenter"])
    dframe["valueOrig"] = dframe["ABinCenter"] + np.exp(0 * dframe["BBinCenter"]) * dframe["csin"]

    figureArray = [
        [['CBinCenter'], ['valueOrig', 'means-valueOrig', 'medians-valueOrig']]
    ]
    tooltips = [("A", "(@ABinCenter)"), ("B", "(@BBinCenter)"), ("C", "(@CBinCenter)"), ("D", "(@DBinCenter)"), ]
    widgets = 'slider.DBinCenter(0,1,0.1,0,1)'
    figureLayout: str = '((0),x_visible=1,y_visible=0,plot_height=250,plot_width=1000)'
    bokehDrawSA.fromArray(dframe, "ABinNumber>0", figureArray, widgets, layout=figureLayout, tooltips=tooltips)


test_makePDF()
