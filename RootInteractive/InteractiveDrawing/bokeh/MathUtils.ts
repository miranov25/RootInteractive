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
      if(k < pivot_cumsum_lower){
        end = pivot_idx
      } else if (k > pivot_cumsum_upper) {
        k -= pivot_cumsum_upper
        begin = pointer_mid
        if (pointer_mid === end_orig) return pointer_mid-1
      } else {
        return pivot_idx
      }
    }
  }
  
  // Cholesky decomposition without pivoting - inplace
// TODO: Perhaps add a version with pivoting too?
function chol(X: number[], nRows: number){
  let iRow = 0
  let jRow, kRow
  for(let i=0; i < nRows; ++i){
    const pivotDiag = 1/X[i+iRow]
    jRow = iRow+i+1
    for(let j=i+1; j < nRows; ++j) {
      const pivotRow = pivotDiag*X[i+jRow]
      kRow = iRow+i+1
      for(let k=i+1; k <= j; ++k){
	X[k+jRow] -= pivotRow*X[i+kRow]
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
