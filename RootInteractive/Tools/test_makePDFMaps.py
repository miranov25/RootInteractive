from RootInteractive.Tools.makePDFMaps import *
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA


#


def randomData(nPoints=100000):
    df = pd.DataFrame(np.random.random_sample(size=(nPoints, 5)), columns=list('ABCDE'))
    df["noise"] = np.random.normal(0, 0.1, nPoints)
    df["csin"] = np.sin(6.28 * df["C"])
    df["valueOrig"] = df["A"] + np.exp(0 * df["B"]) * df["csin"]
    df["value"] = df["valueOrig"] + df["noise"]
    return df


def test_makePDF():
    df = randomData(1000000)
    histo = makeHistogram(df, "value:A:B:C:D:#A>0>>myhisto(40,-2,2,40,0,1,30,0,1,50,0,1,20,0,1)")

    slices = ((0, 40, 1, 0), (10, 30, 1, 1), (10, 24, 5, 1), (1, 49, 3, 1), (0, 10, 3, 0))
    dimI = 0
    dframe = makePdfMaps(histo, slices, dimI)
    #
    aliasArray = [ 
        ("csin", "sin(6.28*CBinCenter)"),
        ("valueOrig", "ABinCenter+csin"),
        ("deltaMean", "means-valueOrig")
    ]
    tooltips = [("A", "(@ABinCenter)"), ("B", "(@BBinCenter)"), ("C", "(@CBinCenter)"), ("Orig. Value", "(@valueOrig)"), ("Delta", "(@deltaMean)")]
    widgetParams=[
            ['range', ['ABinCenter']],
            ['range', ['BBinCenter']],
            ['range', ['CBinCenter']],
            ['range', ['valueOrig']],
            ['range', ['deltaMean']],
        ]
    widgetLayoutDesc=[ [2], [0,1], [3,4], {'sizing_mode':'scale_width'} ]
    #
    figureArray = [
        [['CBinCenter'], ['valueOrig', 'means-valueOrig', 'medians-valueOrig']],
        [['CBinCenter'], ['rmsd']],
    ]
    figureLayoutDesc=[
        [0, {'commonX':1,'y_visible':2,'plot_height':400}],
        [1,{'plot_height':200}],
        {'plot_height':200,'sizing_mode':'scale_width'}
    ]

    output_file("test_makePDF.html")
    bokehDrawSA.fromArray(dframe, "ABinNumber>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips,widgetLayout=widgetLayoutDesc, aliasArray=aliasArray)
    #

def DistMap():
    df = randomData(100000)
    # histogram binning as for the distortion calibration
    histo = makeHistogram(df, "value:A:B:C:D:#A>0>>myhisto(200,-2,2,180,0,1,33,0,33,40,0,40,8,0,1)")
    print(180*33*33*40*8)
    slices = ((0, 200, 1, 0), (0, 180, 1, 0), (0, 33, 1, 0), (0, 40, 1, 0), (0, 8, 1, 0))
    dimI = 0
    dframe = makePdfMaps(histo, slices, dimI)


test_makePDF()
#DistMap()
