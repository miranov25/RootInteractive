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
    // The least-squares fit y = beta_0 + sum_i beta_i * x_i, with optional weights w and the ridge term alpha (added,
    // as before, to every diagonal element of the normal equations of the original predictors, the intercept
    // included).
    //
    // PHASE_5 Group 3 (defect ST-D6): the sums of the normal equations are formed from the centred and scaled
    // predictors u_i = (x_i - m_i) / s_i, with m_i and s_i the weighted mean and spread of x_i, and the coefficients
    // are transformed back. Before, the sums were formed from x_i itself: with a large offset (a position far from
    // 0, a run number, a timestamp) they lost the digits that carry the information, and the fit was silently wrong
    // (offset 1e6) or meaningless (1e9). This is the standard remedy (numpy.polynomial.Polynomial.fit maps its data
    // to [-1, 1] for the same reason). It is an exact change of variables: m_i and s_i need not be exact, and the
    // ridge term is transformed with it, so the fit minimises the same function as before.
    //
    // PHASE_5 Group 3 (defect ST-D7): without the ridge term, a fit with fewer rows of positive weight than unknowns
    // is not determined, and gives no fit (NaN), as a fit that cannot be solved does (MathUtils.solve). Before,
    // rounding could leave a tiny positive pivot and give finite, meaningless coefficients.
    const {alpha, source, varX, varY, weights} = this
    const nX = varX.length
    const nUnknowns = nX + 1
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
    for(let k=0; k < len; ++k){
      const wk = w == null ? 1 : w[k]
      sumW += wk
      if(wk > 0) ++nPositive
    }
    const mean: number[] = []
    const scale: number[] = []
    for(const col of cols){
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

    // 2. the normal equations of u (packed lower triangle, row by row; the intercept is the last unknown)
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

    // 3. the ridge term alpha*(sum_i beta_i^2 + beta_0^2) in the new unknowns gamma: beta = T gamma with
    // beta_i = gamma_i/s_i and beta_0 = gamma_0 - sum_i gamma_i*m_i/s_i, so alpha * T^T T is added
    if(alpha > 0){
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
    }

    // 4. solve, and transform back
    if(!(alpha > 0) && nPositive < nUnknowns){
      rhs.fill(NaN)
    } else {
      solve(x, rhs)
    }
    this.parameters = []
    let intercept = rhs[nX]
    for(let i=0; i < nX; ++i){
      const beta = rhs[i]/scale[i]
      this.parameters.push(beta)
      intercept -= beta*mean[i]
    }
    this.parameters.push(intercept)
    this._is_fresh = true
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
