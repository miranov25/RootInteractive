import numpy as np
import torch
from RootInteractive.Tools.Histograms.histogramdd_pytorch import histogramdd as histogramdd_pytorch

def _randombins(nbins,d):
    x = [np.sort(np.random.randn(nbins)*2) for i in range(d)]
    return x

class HistogramBenchmark:
    params = ([1000,10000,100000,1000000,10000000],[3,4,5,6],["uniform","random"])

    def setup(self,n,d,bins_type):
        self.sample = np.random.randn(n,d)
        self.sample_torch_cpu = torch.tensor(self.sample).T.contiguous()
        binsfunc = {"uniform":lambda x,y:x,"random":_randombins}
        print(self.sample_torch_cpu.dtype)
        self.bins = binsfunc[bins_type](10,d)
        if torch.cuda.is_available:
            self.sample_torch_cuda = self.sample_torch_cpu.cuda()
            torch.cuda.synchronize()

    def time_histogramdd_numpy(self,n,d,bins_type):
        np.histogramdd(self.sample,self.bins)

    def time_histogramdd_torch_cpu(self,n,d,bins_type):
        histogramdd_pytorch(self.sample_torch_cpu,self.bins)

    def time_histogramdd_torch_cuda(self,n,d,bins_type):
        if torch.cuda.is_available:
            histogramdd_pytorch(self.sample_torch_cuda,self.bins)
            torch.cuda.synchronize()
        else:
             raise Exception("CUDA is not available")
