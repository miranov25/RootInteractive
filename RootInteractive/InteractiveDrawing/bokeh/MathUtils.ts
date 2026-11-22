export function lerp(a:number, b:number, t:number){
    return a + (b - a) * t
}

export function kth_value(sample:any[], k:number, compare:(a:number, b:number)=>number, begin=0, end=-1){
    // TODO: Use a smarter algorithm
    // This algorithm should be linear average case, but worst case is quadratic
    if(end < 0) end += sample.length + 1
    if(!(k>=begin && k < end)) {
      throw Error("Element out of range: " + (k-begin))
    }
    while(begin < end){
      //debugger;
      let pivot = sample[k]
      let pivot_idx = begin
      for(let i=begin; i < end; i++){
        pivot_idx += isNaN(sample[i]) || compare(sample[i], pivot) < 0 ? 1 : 0
      }
      if(pivot_idx >= end && isNaN(pivot)) return
      let pointer_mid = pivot_idx
      let pointer_high = end-1
      let pointer_low=begin
      while(pointer_low < pivot_idx){
        let x = sample[pointer_low]
        let cmp = compare(x, pivot)
        if(cmp>0){
          sample[pointer_low] = sample[pointer_high]
          sample[pointer_high] = x
          --pointer_high
        } else if(cmp === 0){
          sample[pointer_low] = sample[pointer_mid]
          sample[pointer_mid] = x
          ++pointer_mid
        } else {
          ++pointer_low
        }
      }
      while(pointer_mid <= pointer_high){
        let x = sample[pointer_mid]
        let cmp = compare(x, pivot)     
        if(cmp === 0){
          ++pointer_mid
        } else {
          sample[pointer_mid] = sample[pointer_high]
          sample[pointer_high] = x
          --pointer_high
        }
      }
      if(k < pivot_idx){
        end = pivot_idx
      } else if(k >= pointer_mid){
        begin = pointer_mid
      } else {
        return
      }
    }
  }
  
export function weighted_kth_value(sample:number[], weights: number[], k:number, begin=0, end=-1){
    // TODO: Use a smarter algorithm
    // This algorithm should be linear average case, but worst case is quadratic
    if(end < 0) end += sample.length + 1
    const end_orig = end
    while(true){
      let pivot = sample[(begin+end)>>1]
      let pivot_idx = begin
      let pivot_cumsum_lower = 0
      let pivot_cumsum_upper = 0
      for(let i=begin; i < end; i++){
        pivot_idx += sample[i] < pivot ? 1 : 0
        pivot_cumsum_lower += sample[i] < pivot ? weights[i] : 0
        pivot_cumsum_upper += sample[i] <= pivot ? weights[i] : 0
      }
      let pointer_mid = pivot_idx
      let pointer_high = end-1
      let pointer_low=begin
      while(pointer_low < pivot_idx){
        let x = sample[pointer_low]
        let tmp = weights[pointer_low]
        if(x>pivot){
          sample[pointer_low] = sample[pointer_high]
          sample[pointer_high] = x
          weights[pointer_low] = weights[pointer_high]
          weights[pointer_high] = tmp       
          --pointer_high
        } else if(x === pivot){
          sample[pointer_low] = sample[pointer_mid]
          sample[pointer_mid] = x
          weights[pointer_low] = weights[pointer_mid]
          weights[pointer_mid] = tmp
          ++pointer_mid
        } else {
          ++pointer_low
        }
      }
      while(pointer_mid <= pointer_high){
        let x = sample[pointer_mid]
        let tmp = weights[pointer_mid]    
        if(x === pivot){
          ++pointer_mid
        } else {
          sample[pointer_mid] = sample[pointer_high]
          sample[pointer_high] = x
          weights[pointer_mid] = weights[pointer_high]
          weights[pointer_high] = tmp
          --pointer_high
        }
      }
      // PHASE_5 Batch 2B (WT-D3): rows whose weight is 0 are never the answer. At an exact cumulative boundary the
      // pivot's rows could be returned although they carry no weight; then the search goes on below the pivot (k
      // reached by smaller values with weight) or above it (k = 0, or k beyond them)
      const no_weight = pivot_cumsum_upper === pivot_cumsum_lower
      if(k < pivot_cumsum_lower || (no_weight && k === pivot_cumsum_lower && pivot_cumsum_lower > 0)){
        end = pivot_idx
      } else if (k > pivot_cumsum_upper || no_weight) {
        k -= pivot_cumsum_upper
        begin = pointer_mid
        if (pointer_mid === end_orig) return pointer_mid-1
      } else {
        return pivot_idx
      }
    }
  }
  
export function quantile(sample:number[], q:number, begin=0, end=-1){
  // Returns the q-th quantile of the sample

  // q should be in the range [0,1]
  if(q < 0 || q > 1) throw Error("Quantile out of range: " + q)
  if(end < 0) end += sample.length + 1
  if(!(begin >= 0 && begin < end)) {
    throw Error("Element out of range: " + (begin))
  }
  if(end - begin <= 0) return NaN
  if(end - begin === 1) return sample[begin]
  if(end - begin === 2){
    // PHASE_5 Batch 2B (B5): interpolate from the smaller to the larger value - it was (a + b) * q
    const a = Math.min(sample[begin], sample[begin+1]), b = Math.max(sample[begin], sample[begin+1])
    return a + (b - a) * q
  }
  const k = begin + (end - begin - 1) * q
  const k1 = Math.floor(k)
  // PHASE_5 Batch 2B (B5b): the ends of the range are its minimum and maximum - not the first and last element of the
  // unsorted range, which depended on the row order
  if(k === begin){
    let lo = sample[begin]
    for(let i=begin+1; i < end; ++i) if(sample[i] < lo) lo = sample[i]
    return lo
  }
  if(k1 === end - 1){
    let hi = sample[begin]
    for(let i=begin+1; i < end; ++i) if(sample[i] > hi) hi = sample[i]
    return hi
  }
  kth_value(sample, k1, (a, b) => a - b, begin, end)
  if(k % 1 === 0) {
    return sample[k1]
  } else {
    let k2 = Math.ceil(k)
    let value_k2 = sample[k1 + 1]
    for(let k3=Math.ceil(k); k3 < end; ++k3){
      if(sample[k3] < value_k2) {
        value_k2 = sample[k3]
        k2 = k3
      }
    }
    return lerp(sample[k1], sample[k2], k-k1)
  }
}
  
  // Cholesky decomposition without pivoting - inplace
// TODO: Perhaps add a version with pivoting too?
// PHASE_5 Group 1 (ST-D5): a diagonal element that is not positive when its turn comes means that the matrix cannot
// be solved; it is marked NaN, and solve() then returns NaN. There is no threshold: before, any element below the
// absolute number 1e-6 counted as zero, so the result depended on the units of the variables.
export function chol(X: number[], nRows: number){
  let iRow = 0
  let jRow, kRow
  for(let i=0; i < nRows; ++i){
    if(!(X[i+iRow] > 0)) X[i+iRow] = NaN
    const pivotDiag = 1/X[i+iRow]
    jRow = iRow+i+1
    for(let j=i+1; j < nRows; ++j) {
      const pivotRow = pivotDiag*X[i+jRow]
      kRow = iRow+i+1
      // PHASE_5 Group 1 (ST-D2): row j, columns i+1..j-1 were already updated by the second loop, when each of those
      // columns was the j of an earlier turn; only the diagonal element (j, j) is updated here. Updating them again,
      // with an already scaled value, made every fit with two or more predictors wrong.
      for(let k=i+1; k <= j; ++k){
	if(k === j) X[k+jRow] -= pivotRow*X[i+kRow]
	kRow += k+1
      }
      for(let k=j+1; k < nRows; ++k) {
	X[j+kRow] -= pivotRow*X[i+kRow]
	kRow += k+1
      }
      X[i+jRow] = pivotRow
      jRow += j+1
    }
    iRow += i+1
  }
  return X
}

// Solves a system of linear equations using Cholesky decomposition
export function solve(x:number[], y:number[]){
  let nRows = y.length
  chol(x,nRows)
  // PHASE_5 Group 1 (ST-D5): a matrix that cannot be solved gives NaN for every unknown, not a wrong number
  for(let i=0, iDiag=0; i < nRows; iDiag += i+2, ++i){
    if(!(x[iDiag] > 0)) return y.fill(NaN)
  }
  let iRow = 0
  for(let i=0; i < nRows; ++i){
    for(let j=0; j < i; ++j){
      y[i] -= y[j]*x[iRow+j]
    }
    iRow += i+1
  }
  let iDiag=0
  for(let i=0; i < nRows; i++){
    y[i] /= x[iDiag]
    iDiag += i+2
  }
  let jRow = 0
  for(let i=nRows-1; i >= 0; --i){
    jRow = 1+((i*(i+5))>>1)
    for(let j=i+1; j < nRows; ++j){
      y[i] -= y[j]*x[jRow]
      jRow += j+1
    }
  }
  return y
}

// ------------------------------------------------------------------------------------------------------------------
// PHASE_5_2: the quantile estimators of dfdraw (plots/_hist_quantiles.py, the Architect's decisions of 2026-11-10),
// for profiles (quantile_method) and for the exact quantile lines of histograms.
// - without weights: NumPy's method "linear", position (n-1) q among the sorted finite values;
// - with weights, method "inverted_cdf" (dfdraw's default): the smallest value whose cumulative weight share reaches q,
//   as NumPy 2 computes it (cumulative sum divided by the total, then the first index with share >= q);
// - with weights, method "linear": h = q (W - w1) + w1, interpolated over the cumulative weights C_k of the sorted
//   values (w1 the weight of the smallest value); with unit weights it equals NumPy's linear;
// - a weight of 0 carries no mass; a negative or non-finite weight gives NaN (refused); no positive weight gives NaN.

// The finite values with positive weight, sorted, with their weights (null when no weights are given); null when a
// weight is negative or not finite. `indices`: the rows to take, or null for all rows. Typed arrays, so that 10^7
// selected rows stay affordable.
export function sorted_with_weights(values: ArrayLike<number>, weights: ArrayLike<number> | null,
                                    indices: ArrayLike<number> | null): {xs: Float64Array, ws: Float64Array | null} | null {
  // memory: 8 bytes per kept row without weights, 20 with weights (one typed array per column and the row order)
  const n = indices == null ? values.length : indices.length
  let count = 0
  for (let j = 0; j < n; j++) {
    const i = indices == null ? j : indices[j]
    if (!isFinite(values[i])) continue
    if (weights != null) {
      const w = weights[i]
      if (!isFinite(w) || w < 0) return null
      if (w === 0) continue
    }
    count++
  }
  if (weights == null) {
    const xs = new Float64Array(count)
    let k = 0
    for (let j = 0; j < n; j++) {
      const v = values[indices == null ? j : indices[j]]
      if (isFinite(v)) xs[k++] = v
    }
    xs.sort()
    return {xs, ws: null}
  }
  const kept = new Uint32Array(count)
  let k = 0
  for (let j = 0; j < n; j++) {
    const i = indices == null ? j : indices[j]
    if (isFinite(values[i]) && weights[i] > 0) kept[k++] = i
  }
  kept.sort((a, b) => values[a] - values[b] || a - b)
  const xs = new Float64Array(count)
  const ws = new Float64Array(count)
  for (let j = 0; j < count; j++) { xs[j] = values[kept[j]]; ws[j] = weights[kept[j]] }
  return {xs, ws}
}

// One level q in [0, 1] of sorted values (see sorted_with_weights); NaN for an empty set or a level outside [0, 1].
export function quantile_of_sorted(xs: ArrayLike<number>, ws: ArrayLike<number> | null, q: number, method: string): number {
  const n = xs.length
  if (n === 0 || !(q >= 0 && q <= 1)) return NaN
  if (ws == null || method === "linear" && Array.prototype.every.call(ws, (w: number) => w === 1)) {
    if (method === "inverted_cdf") {
      const k = Math.max(Math.ceil(n * q) - 1, 0)
      return xs[k]
    }
    const h = (n - 1) * q
    const r0 = Math.floor(h)
    return h > r0 ? lerp(xs[r0], xs[r0 + 1], h - r0) : xs[r0]
  }
  const cumulative = new Float64Array(n)
  let total = 0
  for (let k = 0; k < n; k++) { total += ws[k]; cumulative[k] = total }
  if (method === "inverted_cdf") {
    for (let k = 0; k < n; k++) {
      if (cumulative[k] / total >= q) return xs[k]
    }
    return xs[n - 1]
  }
  // weighted linear
  const h = q * (total - ws[0]) + ws[0]
  let before = 0
  for (let k = 0; k < n; k++) {
    if (cumulative[k] >= h) {
      if (k === 0) return xs[0]
      return lerp(xs[k - 1], xs[k], (h - before) / (cumulative[k] - before))
    }
    before = cumulative[k]
  }
  return xs[n - 1]
}
