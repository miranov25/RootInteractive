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
df["A"].astype(CategoricalDtype(ordered=True))

entropy(df["A"].value_counts(),base=2)
entropy(((df[0:-1]-df[1:])["A"])[1:-1].value_counts(),  base=2)
entropy(df["B"].value_counts(),base=2)
entropy(((df[0:-1]-df[1:])["B"])[1:-1].value_counts(),  base=2)