"""An exact port of RootInteractive's present weighted, unbinned profile, for the defect models of PHASE_5 Batches 2A
and 2B (test-only correction after GPT24.RI's review of the test commits, 2026-11-13).

Plain words. A profile of y per x bin is computed by HistoNdProfile.ts from a 2-D histogram (HistoNdCDS.ts) of x and y.
The rows are put in bin order once (a stable counting sort, x the fast axis); then, for each weight in the profile's
weight list, entries (the sum of the weights), mean and std are summed row by row, and each requested quantile is found
by MathUtils.weighted_kth_value, which reorders the values AND that series' weights in place. The values are reordered
for good: the next weight series pairs its own weights, in the original order, with the reordered values. That is
WT-D4. A first row of weight 0 in a bin makes the running mean 0/0 = NaN: that is WT-D2. weighted_kth_value can return a
row of weight 0 at an exact cumulative boundary: that is WT-D3.

This port does the same operations in the same order on Python floats (IEEE doubles, like JavaScript), so the page and
the port agree to the last bit. Each defect is a switch:
  "WT-D2"  the running mean starts at a row of weight 0 (switched off: rows of weight 0 are skipped in the sums);
  "WT-D4"  later series use the values as reordered by earlier series (off: each series starts from the bin order).
WT-D3 is not a switch: weighted_kth_value is ported as it is, and the 2B tests judge its result separately.
Sources: HistoNdProfile.ts 120-262, HistoNdCDS.ts 290-312 (histogram_sorted) and 521-556 (compute_sorted_indices),
MathUtils.ts 59-116 (weighted_kth_value), at master f1cbe0f.
"""
import math
from typing import Dict, List, Sequence


def weighted_kth_value(sample: List[float], weights: List[float], k: float, begin: int, end: int) -> int:
    """MathUtils.ts weighted_kth_value, line by line: a three-way partition around the middle element, moving the
    weights with the values, until the k-th unit of weight is found. Reorders sample[begin:end] and weights[begin:end]
    in place; returns an index into them."""
    end_orig = end
    while True:
        pivot = sample[(begin + end) >> 1]
        pivot_idx = begin
        lower = 0.0
        upper = 0.0
        for i in range(begin, end):
            pivot_idx += 1 if sample[i] < pivot else 0
            lower += weights[i] if sample[i] < pivot else 0
            upper += weights[i] if sample[i] <= pivot else 0
        p_mid, p_high, p_low = pivot_idx, end - 1, begin
        while p_low < pivot_idx:
            x, tmp = sample[p_low], weights[p_low]
            if x > pivot:
                sample[p_low], sample[p_high] = sample[p_high], x
                weights[p_low], weights[p_high] = weights[p_high], tmp
                p_high -= 1
            elif x == pivot:
                sample[p_low], sample[p_mid] = sample[p_mid], x
                weights[p_low], weights[p_mid] = weights[p_mid], tmp
                p_mid += 1
            else:
                p_low += 1
        while p_mid <= p_high:
            x, tmp = sample[p_mid], weights[p_mid]
            if x == pivot:
                p_mid += 1
            else:
                sample[p_mid], sample[p_high] = sample[p_high], x
                weights[p_mid], weights[p_high] = weights[p_high], tmp
                p_high -= 1
        if k < lower:
            end = pivot_idx
        elif k > upper:
            k -= upper
            begin = p_mid
            if p_mid == end_orig:
                return p_mid - 1
        else:
            return pivot_idx


def _bin(values: Sequence[float], nbins: int, lo: float, hi: float) -> List[int]:
    """HistoNdCDS.compute_bins for one axis: -1 outside [lo, hi] or not finite; the maximum belongs to the last bin."""
    scale = nbins / (hi - lo)
    origin = lo * scale
    out = []
    for v in values:
        if v < lo or v > hi or not math.isfinite(v):
            out.append(-1)
            continue
        b = int(v * scale - origin)
        out.append(min(b, nbins - 1))
    return out


def profile(cells: Sequence[tuple], y: Sequence[float], ybins: tuple, weights: Dict[str, Sequence[float]],
            quantiles=(0.5,), defects=("WT-D2", "WT-D4")) -> Dict[str, dict]:
    """The profile of y per cell, with the weights in `weights` (in that order). `cells` lists the histogram's other
    axes as (values, nbins, (lo, hi)), the first the fastest; y is the last axis, ybins = (nbins, (lo, hi)). Per series:
    entries, mean, std and quantile_<i> per cell (cells in the page's order), as the page computes them with the given
    defects switched on."""
    strides, ncell = [], 1
    for _, nb, _ in cells:
        strides.append(ncell)
        ncell *= nb
    ny = ybins[0]
    binned = [_bin(v, nb, *r) for v, nb, r in cells] + [_bin(y, ny, *ybins[1])]
    flat = []
    for row in zip(*binned):
        flat.append(-1 if min(row) < 0 else sum(b * s for b, s in zip(row, strides + [ncell])))
    counts = [0] * (ncell * ny)
    for f in flat:
        if f >= 0:
            counts[f] += 1
    cum, acc = [], 0
    for c in counts:
        acc += c
        cum.append(acc)
    start = [cum[i] - counts[i] for i in range(len(counts))]
    order = [0] * acc
    fill = list(start)
    for i, f in enumerate(flat):                       # stable counting sort: rows of a bin in table order
        if f >= 0:
            order[fill[f]] = i
            fill[f] += 1
    shared = [float(y[i]) for i in order]              # sorted_column_orig(y): made once, reordered for good
    out = {}
    for name, w in weights.items():
        values = shared if "WT-D4" in defects else [float(y[i]) for i in order]
        sw = [float(w[i]) for i in order]              # sorted_column_orig(weight): fresh for each series
        bin_count = []
        for b in range(ncell * ny):                     # histogram_sorted: sums in bin order
            s = 0.0
            for j in range((cum[b - 1] if b else 0), cum[b]):
                s += float(w[order[j]])
            bin_count.append(s)
        res = {"entries": [], "mean": [], "std": [], **{f"quantile_{q}": [] for q in range(len(quantiles))}}
        for z in range(ncell):                          # cells; y sub-bins z, z+ncell, z+2*ncell, ...
            subs = [z + ncell * j for j in range(ny)]
            cumulative, entries = [], 0.0
            for b in subs:
                entries += bin_count[b]
                cumulative.append(entries)
            if not entries > 0:
                res["entries"].append(entries)
                res["mean"].append(math.nan)
                res["std"].append(math.nan)
                for q in range(len(quantiles)):
                    res[f"quantile_{q}"].append(math.nan)
                continue
            mean = std = entries_ = 0.0
            for b in subs:
                for i in range((cum[b - 1] if b else 0), cum[b]):
                    v, wi = values[i], sw[i]
                    if not math.isfinite(v):
                        continue
                    if wi == 0 and "WT-D2" not in defects:
                        continue
                    entries_ += wi
                    old_mean = mean
                    mean = mean + (v - mean) * wi / entries_ if entries_ != 0 else math.nan
                    std += (v - mean) * (v - old_mean) * wi
            std = math.sqrt(std / entries) if std / entries >= 0 else math.nan
            res["entries"].append(entries)
            res["mean"].append(mean)
            res["std"].append(std)
            low, iq = 0.0, 0
            while iq < len(quantiles) and quantiles[iq] < 0:
                res[f"quantile_{iq}"].append(math.nan)
                iq += 1
            for j, b in enumerate(subs):
                if iq >= len(quantiles):
                    break
                high = cumulative[j]
                lo_i, hi_i = (cum[b - 1] if b else 0), cum[b]
                while iq < len(quantiles) and high > quantiles[iq] * entries:
                    k = quantiles[iq] * entries - low
                    idx = weighted_kth_value(values, sw, k, lo_i, hi_i)
                    res[f"quantile_{iq}"].append(values[idx])
                    iq += 1
                low = high
            while iq < len(quantiles):
                res[f"quantile_{iq}"].append(math.nan)
                iq += 1
        out[name] = res
    return out
