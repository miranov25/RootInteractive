# -*- coding: utf-8 -*-
"""

hsgh
"""

import torch

_range = range

def histogramdd(sample,bins=None,range=None,weights=None,remove_overflow=True):
    r"""
    Function to create N-dimensional histograms
    :param sample:            Tensor
        D by N Tensor, i-th row being the list of positions in the i-th coordinate, j-th column being the coordinates of j-th item
    :param bins:       Tensor, sequence or int, optional
        Tensor:
        (Will probably remove this)
        sequence of Tensors:
        Each Tensor defines the bins on each axis
        sequence of D ints:
        each int signifies the number of bins in each dimension
        int:
        signifies the number of ints in each dimension
    :param range:           sequence, optional
        Each item in the sequence is either a (min,max) Tuple, or None, in which case the edges are calculated as the minimum and maximum of input data
    :param remove_overflow: bool, optional, default True
        Whether the overflow bins should be included in the final histogram or not
    :return:
        Histogram with axes
            * H              - Tensor - the histogram
            * axis           - list of Tensors - axis description
        Example usage:
        >>> r = torch.randn(3,100)
        >>> H,axes = histogramdd(r,bins = (4,3,7))
        >>> H.shape
        (4,3,7)
    """
    edges=None
    device=None
    custom_edges = False
    D,N = sample.shape
    if device == None:
        device = sample.device
    if bins == None:
        if edges == None:
            bins = 10
            custom_edges = False
        else:
            try:
                bins = edges.size(1)-1
            except AttributeError:
                bins = torch.empty(D)
                for i in _range(len(edges)):
                    bins[i] = edges[i].size(0)-1
                bins = bins.to(device)
            custom_edges = True
    try:
        M = bins.size(0)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except AttributeError:
        # bins is either an integer or a list
        if type(bins) == int:
            bins = torch.full([D],bins,dtype=torch.long,device=device)
        elif torch.is_tensor(bins[0]):
            custom_edges = True
            edges = bins
            bins = torch.empty(D,dtype=torch.long)
            for i in _range(len(edges)):
                bins[i] = edges[i].size(0)-1
            bins = bins.to(device)
        else:
            bins = torch.as_tensor(bins).to(device)
    if bins.dim() == 2:
        custom_edges = True
        edges = bins
        bins = torch.full([D],bins.size(1)-1,dtype=torch.long,device=device)
    if torch.any(bins <= 0):
        raise ValueError(
        'The number of bins must be a positive integer.'
        )
    if custom_edges:
        use_old_edges = False
        if not torch.is_tensor(edges):
            use_old_edges = True
            edges_old = edges
            m = max(i.size(0) for i in edges)
            tmp = torch.full([D,m],float("inf"),device=edges[0].device)
            for i in _range(D):
                s = edges[i].size(0)
                tmp[i,:]=edges[i][-1]
                tmp[i,:s]=edges[i][:]
            edges = tmp.to(device)
        k = torch.searchsorted(edges,sample)
        k = torch.min(k,(bins+1).reshape(-1,1))
        if use_old_edges:
            edges = edges_old
        else:
            edges = torch.unbind(edges)
    else:
            if range == None: #range is not defined
                range = torch.empty(2,D,device=device)
                if N == 0: #Empty histogram
                    range[0,:] = 0
                    range[1,:] = 1
                else:
                    range[0,:]=torch.min(sample,1)[0]
                    range[1,:]=torch.max(sample,1)[0]
            elif not torch.is_tensor(range): #range is a tuple
                r = torch.empty(2,D)
                for i in _range(D):
                    if range[i] is not None:
                        r[:,i] = torch.as_tensor(range[i])
                    else:
                        if N == 0: #Edge case: empty histogram
                            r[0,i] = 0
                            r[1,i] = 1
                        else:
                            r[0,i]=torch.min(sample[:,i])
                            r[1,i]=torch.max(sample[:,i])
                range = r.to(device=device,dtype=sample.dtype)
            singular_range = torch.eq(range[0],range[1]) #If the range consists of only one point, pad it up
            range[0,singular_range] -= .5
            range[1,singular_range] += .5
            if torch.any(range[0] > range[1]):
                 raise ValueError("Max must be greater than min in range parameters.")
            edges = [torch.linspace(range[0,i],range[1,i],bins[i]+1) for i in _range(len(bins))]
            tranges = torch.empty_like(range)
            tranges[1,:] = bins/(range[1,:]-range[0,:])
            tranges[0,:] = 1-range[0,:]*tranges[1,:]
            k = torch.addcmul(tranges[0,:].reshape(-1,1),sample,tranges[1,:].reshape(-1,1)).long() #Get the right index
            k = torch.max(k,torch.zeros([],device=device,dtype=torch.long)) #Underflow bin
            k = torch.min(k,(bins+1).reshape(-1,1))


    multiindex = torch.ones_like(bins)
    multiindex[1:] = torch.cumprod(torch.flip(bins[1:],[0])+2,-1).long()
    multiindex = torch.flip(multiindex,[0])
    k *= multiindex.reshape(-1,1)
    l = torch.sum(k,0)
    hist = torch.bincount(l,minlength=(multiindex[0]*(bins[0]+2)),weights=weights)
    hist = hist.reshape(tuple(bins+2))
    if remove_overflow:
        core = D * (slice(1, -1),)
        hist = hist[core]
    return hist,edges
