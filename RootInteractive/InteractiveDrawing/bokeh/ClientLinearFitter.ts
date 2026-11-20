import {Model} from "model"
import {ColumnarDataSource} from "models/sources/columnar_data_source"
import * as p from "core/properties"
import { solve } from "./MathUtils"

export namespace ClientLinearFitter {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    varX: p.Property<Array<string>>
    source: p.Property<ColumnarDataSource>
    varY: p.Property<string>
    weights: p.Property<string|null>
    alpha: p.Property<number>
  }
}

export interface ClientLinearFitter extends ClientLinearFitter.Attrs {}

export class ClientLinearFitter extends Model {
  properties: ClientLinearFitter.Props

  constructor(attrs?: Partial<ClientLinearFitter.Attrs>) {
    super(attrs)
  }

  static __name__ = "ClientLinearFitter"

  static {
    this.define<ClientLinearFitter.Props>(({Array, String, Number, Ref, Nullable})=>({
      varX: [Array(String), []],
      source: [Ref(ColumnarDataSource)],
      alpha: [Number, 0],
      varY: [String],
      weights: [Nullable(String), null]
    }))
  }

  initialize(){
    super.initialize()
    this._lock = false
    this._is_fresh =  false
  }

  args_keys: Array<string>
  args_values: Array<any>

  _lock: boolean
  _is_fresh: boolean

  parameters: Array<number>

  connect_signals(): void {
    super.connect_signals()
    this.connect(this.source.change, this.onChange)
  }

  onChange(){
    if(this._lock){
      return
    }
    this._lock = true
    this._is_fresh = false
    this.change.emit()
    this._lock = false
  }

  fit(){
    // The least-squares fit y = beta_0 + sum_i beta_i * x_i, with optional weights w and the ridge term alpha:
    // it minimises sum_k w_k (y_k - beta_0 - sum_i beta_i x_ki)^2 + alpha (beta_0^2 + sum_i beta_i^2), the
    // intercept included in the penalty, as before.
    //
    // PHASE_5 Group 3 (defect ST-D6): the fit works with the centred and scaled predictors u_i = (x_i - m_i) / s_i,
    // with m_i and s_i the weighted mean and spread of x_i, and the coefficients are transformed back. Before, the
    // sums were formed from x_i itself: with a large offset (a position far from 0, a run number, a timestamp) they
    // lost the digits that carry the information, and the fit was silently wrong (offset 1e6) or meaningless (1e9).
    // This is the standard remedy (numpy.polynomial.Polynomial.fit maps its data to [-1, 1] for the same reason). It
    // is an exact change of variables: m_i and s_i need not be exact, and the ridge term is transformed with it, so
    // the fit minimises the same function as before.
    //
    // PHASE_5 Group 3 (defect ST-D7): without the ridge term, a fit with fewer rows of positive weight than unknowns
    // is not determined, and gives no fit (NaN), as a fit that cannot be solved does (MathUtils.solve). Before,
    // rounding could leave a tiny positive pivot and give finite, meaningless coefficients.
    //
    // PHASE_5 Group 5 (review finding G3-P1-B): with the ridge term, the penalty of the intercept, written in the
    // centred unknowns, adds alpha * (m_i/s_i) * (m_j/s_j) to the sums. With predictors far from 0 that is about
    // 1e17 to 1e19 against data sums of the order of the number of rows, and the data were lost in the rounding:
    // the page gave no fit, or a wrong fit without a message (a reviewer's case, 2 predictors near 1e9 and
    // alpha = 10: slopes -4.9 and 7.9 instead of 1.86 and 1.14). With the ridge term the fit is now solved without
    // forming these sums: the rows sqrt(w_k) (u_k, 1) with the target sqrt(w_k) y_k, and the rows sqrt(alpha) T of
    // the penalty with the target 0, are reduced one by one by plane rotations (Givens rotations) to a small
    // triangle R, and R gamma = Q^T y is solved. This is the standard way to add rows to a least-squares problem
    // without forming its normal equations. The function minimised is the same.
    // Without the ridge term the sums are kept: their solver says "no fit" for exactly collinear predictors, while
    // the rotations would give wrong fitted values without a message for collinear predictors far from 0.
    //
    // PHASE_5 Group 5 (review finding G3-P1-A): with no row of positive weight (all weights 0, or no row), the mean
    // and spread are not defined; they are taken as 0 and 1. With the ridge term the fit then gives the only
    // minimum of its function, every coefficient 0 (as before Group 3); without it, no fit.
    //
    // PHASE_5 Group 5a (review finding F4; the Architect's decision of 2026-11-18): the rotations need the square root
    // of each weight. With a negative weight (signed weights such as sWeights), the fit with the ridge term uses the
    // sums of Group 3 instead, with the penalty added to them: their result for such data is the one before Group 5
    // (exact near 0, with Group 3's limits far from 0).
    const {alpha, source, varX, varY, weights} = this
    const nX = varX.length
    const colY = source.get_column(varY)
    if(colY == null){
      throw ReferenceError("Column not defined: " + varY)
    }
    const cols: ArrayLike<number>[] = []
    for(const name of varX){
      const col = source.get_column(name)
      if(col == null){
        throw ReferenceError("Column not defined: " + name)
      }
      cols.push(col)
    }
    let w: ArrayLike<number> | null = null
    if(weights != null){
      w = source.get_column(weights)
      if(w == null){
        throw ReferenceError("Column not defined: " + weights)
      }
    }
    const len = source.get_length() ?? colY.length

    // 1. the weighted mean and spread of each predictor, and the number of rows with a positive weight
    let sumW = 0
    let nPositive = 0
    let negative = false
    for(let k=0; k < len; ++k){
      const wk = w == null ? 1 : w[k]
      sumW += wk
      if(wk > 0) ++nPositive
      if(wk < 0) negative = true
    }
    const mean: number[] = []
    const scale: number[] = []
    for(const col of cols){
      if(!(sumW > 0)){
        // no row of positive weight: the mean and spread are not defined (G3-P1-A)
        mean.push(0)
        scale.push(1)
        continue
      }
      let acc = 0
      for(let k=0; k < len; ++k){
        acc += w == null ? col[k] : col[k]*w[k]
      }
      const m = acc / sumW
      acc = 0
      for(let k=0; k < len; ++k){
        const d = col[k] - m
        acc += w == null ? d*d : d*d*w[k]
      }
      const s = Math.sqrt(acc / sumW)
      mean.push(m)
      // a constant predictor stays a column of zeros, so the fit cannot be solved and gives NaN, as before
      scale.push(s > 0 ? s : 1)
    }

    // 2. the coefficients gamma of u (the intercept is the last unknown)
    const gamma = alpha > 0 && !negative ? this._ridge_by_rotations(cols, colY, w, len, mean, scale)
                                         : this._normal_equations(cols, colY, w, len, mean, scale, nPositive)

    // 3. transform back: beta_i = gamma_i / s_i, beta_0 = gamma_0 - sum_i beta_i * m_i
    this.parameters = []
    let intercept = gamma[nX]
    for(let i=0; i < nX; ++i){
      const beta = gamma[i]/scale[i]
      this.parameters.push(beta)
      intercept -= beta*mean[i]
    }
    this.parameters.push(intercept)
    this._is_fresh = true
  }

  _normal_equations(cols: ArrayLike<number>[], colY: ArrayLike<number>, w: ArrayLike<number> | null, len: number,
                    mean: number[], scale: number[], nPositive: number): number[]{
    // The normal equations of u (packed lower triangle, row by row), as in Group 3: without the ridge term, and with
    // it when a weight is negative (Group 5a)
    const {alpha} = this
    const nX = cols.length
    const nUnknowns = nX + 1
    const x: number[] = new Array(nUnknowns*(nUnknowns+1)/2).fill(0)
    const rhs: number[] = new Array(nUnknowns).fill(0)
    const u: number[] = new Array(nX).fill(0)
    const lastRow = nX*(nX+1)/2
    for(let k=0; k < len; ++k){
      const wk = w == null ? 1 : w[k]
      const yk = colY[k]
      for(let i=0; i < nX; ++i){
        u[i] = (cols[i][k] - mean[i]) / scale[i]
      }
      let iRow = 0
      for(let i=0; i < nX; ++i){
        const uw = u[i]*wk
        for(let j=0; j <= i; ++j){
          x[iRow+j] += uw*u[j]
        }
        x[lastRow+i] += uw
        rhs[i] += uw*yk
        iRow += i+1
      }
      x[lastRow+nX] += wk
      rhs[nX] += wk*yk
    }
    if(alpha > 0){
      // the ridge term alpha * T^T T of Group 3 (beta = T gamma)
      let iRow = 0
      for(let i=0; i < nX; ++i){
        const ri = mean[i]/scale[i]
        for(let j=0; j <= i; ++j){
          x[iRow+j] += alpha*ri*mean[j]/scale[j]
        }
        x[iRow+i] += alpha/(scale[i]*scale[i])
        x[lastRow+i] -= alpha*ri
        iRow += i+1
      }
      x[lastRow+nX] += alpha
    } else if(nPositive < nUnknowns){
      return rhs.fill(NaN)
    }
    return solve(x, rhs)
  }

  _ridge_by_rotations(cols: ArrayLike<number>[], colY: ArrayLike<number>, w: ArrayLike<number> | null, len: number,
                      mean: number[], scale: number[]): number[]{
    // With the ridge term (G3-P1-B): least squares of the stacked rows, by Givens rotations. R is the upper triangle
    // (row-major, n x n), q the rotated target. A row with weight 0 adds nothing and is skipped; a negative or NaN
    // weight makes its row NaN, and the fit gives NaN.
    const {alpha} = this
    const nX = cols.length
    const n = nX + 1
    const R: number[] = new Array(n*n).fill(0)
    const q: number[] = new Array(n).fill(0)
    const row: number[] = new Array(n).fill(0)
    const addRow = (t: number) => {
      for(let i=0; i < n; ++i){
        const ai = row[i]
        if(ai === 0) continue
        const ii = i*n + i
        let r = Math.sqrt(R[ii]*R[ii] + ai*ai)
        if(!(r > 0 && r < Infinity)){
          r = Math.hypot(R[ii], ai)     // the squares overflowed or underflowed (Group 5a, review finding F6)
        }
        const c = R[ii]/r
        const sn = ai/r
        for(let j=i; j < n; ++j){
          const rij = R[i*n + j]
          const aj = row[j]
          R[i*n + j] = c*rij + sn*aj
          row[j] = c*aj - sn*rij
        }
        const qi = q[i]
        q[i] = c*qi + sn*t
        t = c*t - sn*qi
      }
    }
    for(let k=0; k < len; ++k){
      const wk = w == null ? 1 : w[k]
      if(wk === 0) continue
      const sw = Math.sqrt(wk)
      for(let i=0; i < nX; ++i){
        row[i] = sw*(cols[i][k] - mean[i]) / scale[i]
      }
      row[nX] = sw
      addRow(sw*colY[k])
    }
    // the rows of the penalty, sqrt(alpha) T: beta_i = gamma_i / s_i, beta_0 = gamma_0 - sum_i gamma_i * m_i / s_i
    const sa = Math.sqrt(alpha)
    for(let i=0; i < nX; ++i){
      row.fill(0)
      row[i] = sa/scale[i]
      addRow(0)
    }
    for(let i=0; i < nX; ++i){
      row[i] = -sa*mean[i]/scale[i]
    }
    row[nX] = sa
    addRow(0)
    // R gamma = q, from the last unknown up
    const gamma: number[] = new Array(n).fill(0)
    for(let i=n-1; i >= 0; --i){
      let acc = q[i]
      for(let j=i+1; j < n; ++j){
        acc -= R[i*n + j]*gamma[j]
      }
      gamma[i] = acc / R[i*n + i]
    }
    if(!gamma.every((g) => Number.isFinite(g))){
      gamma.fill(NaN)
    }
    return gamma
  }

  v_compute(xs: any[], data_source: any, output: any[] | null =null){
    if(!this._is_fresh){
	this.fit()
    }
    if(xs.length + 1 !== this.parameters.length){
      return output
      // Workaround for a bug caused by this being recomputed as varX is changed, the alias receiving the wrong number of parameters
      // throw Error("Invalid number of parameters, expected " + (this.parameters.length-1) + " got " + xs.length)
    }
    if(output != null && output.length === data_source.get_length()){
      output.fill(this.parameters[xs.length])
      for(let i=0; i < xs.length; i++){
        let x = xs[i]
        for(let j=0; j < x.length; ++j){
          output[j] += x[j]*this.parameters[i]
        }
      } 
    } 
    return output
  }

  compute(xs: any[]){
    if(!this._is_fresh){
      this.fit()
    }
    let acc = this.parameters[this.parameters.length-1]!
    for(let i=0; i < xs.length && i < this.parameters.length; i++){
	 acc += xs[i]*this.parameters[i]
     }
    return acc     
  }

}
