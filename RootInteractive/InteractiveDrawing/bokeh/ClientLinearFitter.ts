import {Model} from "model"
import * as p from "core/properties"

export namespace ClientLinearFitter {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Model.Props & {
    parameters: p.Property<Record<string, any>>
    fields: p.Property<Array<string>>
    func: p.Property<string>
    v_func: p.Property<string>
  }
}

// Cholesky decomposition without pivoting - inplace
// TODO: Perhaps add a version with pivoting too?
function chol(X: number[], nRows: number){
  let iRow = 0
  let jRow, kRow
  for(let i=0; i<nRows; ++i){
    iPivot = i
    pivotDiag = 1/X[i+iRow]
    jRow = iRow+i+1
    for(let j=i+1; j<nRows; ++j) {
      pivotRow = pivotDiag*X[i+jRow]
      kRow = iRow+i+1
      for(let k=i+1; k<=j; ++k){
	X[k+jRow] -= pivotRow*X[i+kRow]
	kRow += k+1
      }
      kRow = jRow+j+1
      for(let k=j+1; k<nRows; ++k) {
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
function solve(x:number[], y:number[]){
  let nRows = y.length
  chol(x,nRows)
  iRow = 0
  for(let i=0; i<nRows; ++i){
    for(let j=0; j<i; ++j){
      y[i] -= y[j]*x[iRow+j]
    }
    iRow += i+1
  }
  let iDiag=0
  for(let i=0; i<nRows; i++){
    y[i] /= x[iDiag]
    iDiag += i+2
  }
  let jRow = 0
  for(let i=nRows-1; i>=0; --i){
    jRow = 1+((x*(x+5))>>1)
    for(let j=i+1; j<nRows; ++j){
      y[i] -= y[j]*x[jRow]
      jRow += j+1
    }
  }
  return y
}

export interface ClientLinearFitter extends ClientLinearFitter.Attrs {}

export class ClientLinearFitter extends Model {
  properties: ClientLinearFitter.Props

  constructor(attrs?: Partial<ClientLinearFitter.Attrs>) {
    super(attrs)
  }

  static __name__ = "ClientLinearFitter"

  static init_ClientLinearFitter() {
    this.define<ClientLinearFitter.Props>(({Array, String})=>({
      parameters:  [p.Instance, {}],
      fields: [Array(String), []],
      func:    [ String ],
      v_func:  [String]
    }))
  }

  initialize(){
    super.initialize()
    this.update_func()
    this.update_vfunc()
  }

  args_keys: Array<string>
  args_values: Array<any>

  scalar_func: Function
  vector_func: Function

  connect_signals(): void {
    super.connect_signals()
  }

  update_func(){
    this.args_keys = Object.keys(this.parameters)
    this.args_values = Object.values(this.parameters)
    this.scalar_func = new Function(...this.args_keys, ...this.fields, '"use strict";\n'+this.func)
    this.change.emit()
  }

  compute(x: any[]){
    return this.scalar_func!(...this.args_values, ...x)
  }

  update_vfunc(){
    this.args_keys = Object.keys(this.parameters)
    this.args_values = Object.values(this.parameters)
    this.vector_func = new Function(...this.args_keys, ...this.fields, "data_source", "$output",'"use strict";\n'+this.v_func)
    this.change.emit()
  }

  update_args(){
    this.args_keys = Object.keys(this.parameters)
    this.args_values = Object.values(this.parameters)
    this.change.emit()
  }

  v_compute(xs: any[], data_source: any, output: any[] | null =null){
      if(this.vector_func){
          return this.vector_func(...this.args_values, ...xs, data_source, output)
      } else {
          return null
      }
  }

}
