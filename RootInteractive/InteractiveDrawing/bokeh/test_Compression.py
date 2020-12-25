import pandas as pd
import numpy as np
from pandas import CategoricalDtype
from scipy.stats import entropy

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

def simulateHistoDCA():
    sigma0=0.1
    sigma1=1
    qPt      = np.linspace(-5, 5, 50)
    tgl      = np.linspace(-1, 1, 20)
    mdEdx    = np.linspace(0, 1, 20)
    alpha    = np.linspace(0, np.pi, 12)
    value    = np.linspace(-10*sigma0, 10*sigma0, 100)

    # dcaSigma = sigma0* np.sqrt(1+sigma1*mdEdx*qPt)
    range=([-5,5],[-1,1],[0,1],[0,2*np.pi],[-10*sigma0,10*sigma0])
    bins=[50,20,20,12,100]
    H, edges= np.histogramdd(sample=np.array([[0,0,0,0,0]]), bins=bins,range=range)
    indexH=np.arange(H.size)
    indexC=np.unravel_index(indexH,bins)

    for i in indexH:
        qPt=indexC[0][i]
        mdEdx=indexC[2][i]
        value=indexC[4][i]
        valueSigma=sigma0*np.sqrt(1+sigma1*mdEdx*qPt*qPt)
        weight=np.exp(value*value/(2*valueSigma*valueSigma))
        weigthPoison=np.random.poisson()


