import numpy as np
import pytest
import torch
from RootInteractive.Tools.Histograms.histogramdd_pytorch import histogramdd as histogramdd_pytorch
import itertools

def test_histogram_singular():
    H,axes = histogramdd_pytorch(torch.zeros(5,100000),9)
    assert H[4,4,4,4,4] == 100000, "Test for all zeros failed"
    assert len(axes) == 5, "Number of axes invalid"

def test_histogram_empty():
    sample = torch.empty(5,0)
    H,axes = histogramdd_pytorch(sample,bins = [2,3,4,5,6], range=[(1,2),None,None,None,None])
    assert torch.sum(H) == 0, "Sum of empty histogram is not zero"
    assert len(axes) == 5, "Number of axes invalid"

@pytest.mark.parametrize("seed, N, D, bins, device",[(665162135,100000,5,[9,8,9,10,2],"cpu:0"),(665162135,100000,4,[9,2,4,42],"cuda:0")])
def test_histogram_uniform(seed,N,D,bins,device):
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available with pytorch")
    torch.random.manual_seed(seed)
    sample = torch.rand(D,N,device=device)
    H,axes = histogramdd_pytorch(sample,bins)
    Hnp,npaxes = np.histogramdd(sample.T.cpu().numpy(),bins)
    assert len(axes) == D, "Number of axes invalid"
    assert torch.sum(H) <= N, "Sum of histogram too high"
    assert np.max(np.abs(H.cpu().numpy() - Hnp)) <= 1, "Histogram inconsistent with numpy histogram"
    for i in range(len(axes)):
        assert len(axes[i]) == bins[i]+1, "Invalid number of edges"

@pytest.mark.parametrize("seed, N, D, bins, device",[(665162135,100000,5,[9,8,9,10,2],"cpu:0"),(665162135,100000,4,[9,2,4,42],"cuda:0")])
def test_histogram_uniform_randombins(seed,N,D,bins,device):
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available with pytorch")
    try:
        from torch import searchsorted
    except ImportError:
        pytest.skip("Current Pytorch version doesn't supoport searchsorted")
    torch.random.manual_seed(seed)
    sample = torch.rand(D,N,device=device)
    nbins = bins
    bins = [torch.sort(torch.rand(i+1,device=device))[0] for i in nbins] #+1 because there's 1 more fence post than the length of the fence
    H,axes = histogramdd_pytorch(sample,bins)
    Hnp,npaxes = np.histogramdd(sample.T.cpu().numpy(),[i.cpu().numpy() for i in bins])
    assert len(axes) == D, "Number of axes invalid"
    assert torch.sum(H) <= N, "Sum of histogram too high"
    assert np.max(np.abs(H.cpu().numpy() - Hnp)) <= 1, "Histogram inconsistent with numpy histogram"
    for i in range(len(axes)):
        assert len(axes[i]) == nbins[i]+1, "Invalid number of edges"

def test_invalid_range():
    sample = torch.Tensor([[1,2,0,7],[2,1,4,6],[3,5,1,0]])
    try:
        H,axes = histogramdd_pytorch(sample,bins = [3,4,8], range=[(1,2),None,(5,4)])
        ok = False
    except ValueError:
        ok = True
    assert ok, "Accepted invalid input: invalid range"

def test_invalid_bins():
    sample = torch.Tensor([[1,2,0,7],[2,1,4,6],[3,5,1,0]])
    try:
        H,axes = histogramdd_pytorch(sample,bins = [3,4,0])
        ok = False
    except ValueError:
        ok = True
    assert ok, "Accepted invalid input: bins must be positive"

def test_invalid_bins_custombinning_empty_axis():
    try:
        from torch import searchsorted
    except ImportError:
        pytest.skip("Current Pytorch version doesn't supoport searchsorted")
    sample = torch.Tensor([[1,2,0,7],[2,1,4,6],[3,5,1,0]])
    try:
        H,axes = histogramdd_pytorch(sample,bins = [torch.tensor([1,2,3]),torch.tensor([2,3,4]),torch.tensor([1])])
        ok = False
    except ValueError:
        ok = True
    assert ok, "Accepted invalid input: each axis must have at least 2 edges for custom binning"
