import numpy as np
import pytest
import torch
from RootInteractive.Tools.Histograms.histogramdd_pytorch import histogramdd as histogramdd_pytorch

def test_histogram_singular():
    H,axes = histogramdd_pytorch(torch.zeros(5,100000),9)
    assert H[4,4,4,4,4] == 100000, "Test for all zeros failed"
    assert len(axes) == 5, "Number of axes invalid"

def test_histogram_empty():
    sample = torch.empty(5,0)
    H,axes = histogramdd_pytorch(sample,bins = [2,3,4,5,6], range=[(1,2),None,None,None,None])
    assert torch.sum(H) == 0, "Sum of empty histogram is not zero"
    assert len(axes) == 5, "Number of axes invalid"
