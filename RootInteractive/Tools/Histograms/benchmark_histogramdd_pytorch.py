# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:32:34 2020

@author: majoi
"""

import numpy as np
import torch
import timeit
from tabulate import tabulate
import io
from matplotlib import pyplot as plt
from histogramdd_pytorch import histogramdd

try:
    from torch import searchsorted
    searchsorted_available = True
except ImportError:
    searchsorted_available = False

torch.random.manual_seed(19680801)
torch.cuda.manual_seed_all(19680801)
np.random.seed(19680801)

number = 5
repeat = 4

create_output_file = True

def get_tensors(n,d=3,device=None):
    x = torch.rand((d,n),device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return x

def get_arrays(n,d=3):
    x = np.random.rand(n,d)
    return x

def get_tensors_edges(n,d=3,device=None):
    x = torch.rand((d,n),device=device)
    v = [torch.arange(0,1.1,.1,device=device)]*d
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return x,v

def get_arrays_edges(n,d=3):
    x = np.random.rand(n,d)
    v = [np.arange(0,1.1,.1)]*d
    return x,v

def get_tensors_ranges(n,d=3,device=None):
    sample = torch.rand((d,n),device=device)
    ranges = torch.empty(2,d)
    ranges[0,:] = 0
    ranges[1,:] = 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return sample,ranges

def histogramdd_synchronized(*args,**kwargs):
    hist = histogramdd(*args,**kwargs)
    torch.cuda.synchronize()
    return hist

n = 100
time_cpu = np.empty((4,6))
time_cuda = np.empty((4,6))
time_numpy = np.empty((4,6))

time_cpu_e = np.empty((4,6))
time_cuda_e = np.empty((4,6))
time_numpy_e = np.empty((4,6))
for j in range(6):
    for i in range(3,7):
        print("n= ",n,"d= ",i)
        t = timeit.repeat(
                stmt="histogramdd(sample)",
                setup="sample = get_tensors(n,i,device='cpu')",
                globals=globals(),
                repeat=repeat,
                number=number
                )
        time_cpu[i-3,j] = min(t)/number
        print("CPU: ",min(t)/number,sep='\t')
        if torch.cuda.is_available():
            t = timeit.repeat(
                    stmt="histogramdd_synchronized(sample)",
                    setup="sample = get_tensors(n,i,device='cuda')",
                    globals=globals(),
                    repeat=repeat,
                    number=number
                    )
            time_cuda[i-3,j] = min(t)/number
            print("CUDA: ",min(t)/number,sep='\t')
        t = timeit.repeat(
                stmt="np.histogramdd(sample)",
                setup="sample = get_arrays(n,i)",
                globals=globals(),
                repeat=repeat,
                number=number
                )
        time_numpy[i-3,j] = min(t)/number
        print("Numpy: ",min(t)/number,sep='\t')

        if searchsorted_available:
            t = timeit.repeat(
                    stmt="histogramdd(sample,bins)",
                    setup="sample,bins = get_tensors_edges(n,i,device='cpu')",
                    globals=globals(),
                    repeat=repeat,
                    number=number
                    )
            time_cpu_e[i-3,j] = min(t)/number
            print("CPU: ",min(t)/number,sep='\t')
            if torch.cuda.is_available():
                t = timeit.repeat(
                        stmt="histogramdd_synchronized(sample,bins)",
                        setup="sample,bins = get_tensors_edges(n,i,device='cuda')",
                        globals=globals(),
                        repeat=repeat,
                        number=number
                        )
                time_cuda_e[i-3,j] = min(t)/number
                print("CUDA: ",min(t)/number,sep='\t')
            t = timeit.repeat(
                    stmt="np.histogramdd(sample,bins)",
                    setup="sample,bins = get_arrays_edges(n,i)",
                    globals=globals(),
                    repeat=repeat,
                    number=number
                    )
            time_numpy_e[i-3,j] = min(t)/number
            print("Numpy: ",min(t)/number,sep='\t')

    n *= 10

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
ax1.set_title("d=3")
ax1.loglog([100,1000,10000,100000,1000000,10000000],time_cpu[0,:],label='CPU')
if torch.cuda.is_available():
    ax1.loglog([100,1000,10000,100000,1000000,10000000],time_cuda[0,:],label='CUDA')
ax1.loglog([100,1000,10000,100000,1000000,10000000],time_numpy[0,:],label='Numpy')
ax2.set_title("d=4")
ax2.loglog([100,1000,10000,100000,1000000,10000000],time_cpu[1,:],label='CPU')
if torch.cuda.is_available():
    ax2.loglog([100,1000,10000,100000,1000000,10000000],time_cuda[1,:],label='CUDA')
ax2.loglog([100,1000,10000,100000,1000000,10000000],time_numpy[1,:],label='Numpy')
ax3.set_title("d=5")
ax3.loglog([100,1000,10000,100000,1000000,10000000],time_cpu[2,:],label='CPU')
if torch.cuda.is_available():
    ax3.loglog([100,1000,10000,100000,1000000,10000000],time_cuda[2,:],label='CUDA')
ax3.loglog([100,1000,10000,100000,1000000,10000000],time_numpy[2,:],label='Numpy')
ax4.set_title("d=6")
ax4.loglog([100,1000,10000,100000,1000000,10000000],time_cpu[3,:],label='CPU')
if torch.cuda.is_available():
    ax4.loglog([100,1000,10000,100000,1000000,10000000],time_cuda[3,:],label='CUDA')
ax4.loglog([100,1000,10000,100000,1000000,10000000],time_numpy[3,:],label='Numpy')
ax4.legend()
fig.show()

output_string = "## Benchmark results: \n"
output_string += "### Uniform binning: \n"
output_string += "Numpy: \n"
output_string += tabulate(time_numpy*1000,["1e2","1e3","1e4","1e5","1e6","1e7"],showindex=[3,4,5,6],tablefmt="github")
output_string += "\n\n"
output_string += "PyTorch CPU: \n"
output_string += tabulate(time_cpu*1000,["1e2","1e3","1e4","1e5","1e6","1e7"],showindex=[3,4,5,6],tablefmt="github")
output_string += "\n\n"
if torch.cuda.is_available():
    output_string += "PyTorch CUDA: \n"
    output_string += tabulate(time_cuda*1000,["1e2","1e3","1e4","1e5","1e6","1e7"],showindex=[3,4,5,6],tablefmt="github")
    output_string += "\n\n"

if searchsorted_available:
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.set_title("d=3")
    ax1.loglog([100,1000,10000,100000,1000000,10000000],time_cpu_e[0,:],label='CPU')
    if torch.cuda.is_available():
        ax1.loglog([100,1000,10000,100000,1000000,10000000],time_cuda_e[0,:],label='CUDA')
    ax1.loglog([100,1000,10000,100000,1000000,10000000],time_numpy_e[0,:],label='Numpy')
    ax2.set_title("d=4")
    ax2.loglog([100,1000,10000,100000,1000000,10000000],time_cpu_e[1,:],label='CPU')
    if torch.cuda.is_available():
        ax2.loglog([100,1000,10000,100000,1000000,10000000],time_cuda_e[1,:],label='CUDA')
    ax2.loglog([100,1000,10000,100000,1000000,10000000],time_numpy_e[1,:],label='Numpy')
    ax3.set_title("d=5")
    ax3.loglog([100,1000,10000,100000,1000000,10000000],time_cpu_e[2,:],label='CPU')
    if torch.cuda.is_available():
        ax3.loglog([100,1000,10000,100000,1000000,10000000],time_cuda_e[2,:],label='CUDA')
    ax3.loglog([100,1000,10000,100000,1000000,10000000],time_numpy_e[2,:],label='Numpy')
    ax4.set_title("d=6")
    ax4.loglog([100,1000,10000,100000,1000000,10000000],time_cpu_e[3,:],label='CPU')
    if torch.cuda.is_available():
        ax4.loglog([100,1000,10000,100000,1000000,10000000],time_cuda_e[3,:],label='CUDA')
    ax4.loglog([100,1000,10000,100000,1000000,10000000],time_numpy_e[3,:],label='Numpy')
    ax4.legend()
    fig.show()
    output_string += "### Custom binning: \n"
    output_string += "Numpy: \n"
    output_string += tabulate(time_numpy_e*1000,["1e2","1e3","1e4","1e5","1e6","1e7"],showindex=[3,4,5,6],tablefmt="github")
    output_string += "\n\n"
    output_string += "PyTorch CPU: \n"
    output_string += tabulate(time_cpu_e*1000,["1e2","1e3","1e4","1e5","1e6","1e7"],showindex=[3,4,5,6],tablefmt="github")
    output_string += "\n\n"
    if torch.cuda.is_available():
        output_string += "PyTorch CUDA: \n"
        output_string += tabulate(time_cuda_e*1000,["1e2","1e3","1e4","1e5","1e6","1e7"],showindex=[3,4,5,6],tablefmt="github")
        output_string += "\n\n"

print(output_string)
if create_output_file:
    f = open("benchmark_histogramdd_pytorch.md","w")
    f.write(output_string)
    f.close()
