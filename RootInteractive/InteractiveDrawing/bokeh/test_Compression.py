import pandas as pd
import numpy as np
from pandas import CategoricalDtype
from scipy.stats import entropy

def estimateEntropy():
    random = np.random.randn(10000,3)
    nBins=10
    H, edges = np.histogramdd(random, bins = (nBins, nBins, nBins), range=[[-6,6], [-6,6], [-6,6]])
    a=np.empty(H.size)
    b=np.empty(H.size)
    c=np.empty(H.size)
    for i in range(H.size):
        a[i]=edges[0][i%nBins]
        b[i]=edges[1][(i//nBins)%nBins]
        c[i]=edges[2][i//(nBins*nBins)]
    #    c[i]=edges[2][i%(nBins*nBins)]
    df=pd.DataFrame(data={'A': a, 'B': b, 'C': c,'H': H.flatten()})
    # Test - send to cds
    entropy(df["A"].value_counts(),base=2)
    entropy(((df[0:-1]-df[1:])["A"])[1:-1].value_counts(),  base=2)
    #entropy=0
    entropy(df["B"].value_counts(),base=2)
    entropy(((df[0:-1]-df[1:])["B"])[1:-1].value_counts(),  base=2)
    #entropy=0
    entropy(df["H"].value_counts(),base=2)
#entropy=3.1

# TODO
#      if (User coding){
#          float-> integer
#          entropy coding
#      }else{
#       1. Unique gives amount of distinct value
#       1. if Unique<<size
#           1. Entropy for value_counts
#           2. Entropy for delta.value-counts
#           3. Use coding with smaller entopy
#   }

def simulatePandaDCA():
    sigma0=0.1
    sigma1=1
    entries=1000
    # qPt,tgl.mdEdx.alpha, dCA
    range=([-5,5],[-1,1],[0,1],[0,2*np.pi],[-10*sigma0,10*sigma0])
    bins=[50,20,20,12,100]
    H, edges = np.histogramdd(sample=np.array([[0, 0, 0, 0, 0]]), bins=bins, range=range)
    indexH = np.arange(H.size)
    indexC = np.unravel_index(indexH, bins)
    qPtCenter = (edges[0][indexC[0]]+edges[0][indexC[0]+1])*.5
    tgl = edges[1][indexC[1]]
    mdEdx = edges[2][indexC[2]]
    value = edges[4][indexC[4]]
    valueSigma = sigma0 * np.sqrt(1 + sigma1 * mdEdx * qPtCenter * qPtCenter)
    weight = np.exp(-value * value / (2 * valueSigma * valueSigma))
    weightPoisson = np.random.poisson(weight * entries, H.size)
    H = weightPoisson
    df = pd.DataFrame({"qPtCenter": qPtCenter, "tglCenter": tgl, "mdEdxCenter": mdEdx, "weight": H})
    # to export  panda, columns
    #       qPtCenter,tglCenter,mdEdxCenter,alphaCenter,
    #       DCA mean, DCA rms
    #       DCA, w
    return H, edges, df

H,edges, df = simulatePandaDCA()