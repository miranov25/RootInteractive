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

  static init_ClientLinearFitter() {
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
    const {alpha, source, varX, varY, weights} = this
    let x: number[] = []
    this.parameters = []
    if(weights == null){
    for(let i=0; i < varX.length; ++i){
      let iField = source.get_column(varX[i])
      if(iField == null){
	throw ReferenceError("Column not defined: " + iField)
      }
      for(let j=0; j <= i; ++j){
	let acc = 0
	let jField = source.get_column(varX[j])
	if(jField == null) continue
	// HACK: for some reason ES6 reduce eats too much memory
	for(let k=0; k < iField.length; ++k){
	  acc += iField[k]*jField[k]
	}
	x.push(acc)
      }
    }
    const colY = source.get_column(varY)
    if(colY == null){
      throw ReferenceError("Column not defined: " + this.varY)
    }
    for(let i=0; i < varX.length; ++i){
      const iField = varX[i]
      let col = source.get_column(iField)
      if(col == null){
	throw ReferenceError("Column not defined: " + iField)
      }
      let acc = 0
      for(let k=0; k < col.length; k++){
	acc += col[k]
      }
      x.push(acc)
      acc=0
      for(let k=0; k < col.length; k++){
	acc += col[k]*colY[k]
      }
      this.parameters.push(acc)
    }
    let len = source.get_length()
    if(len == null){
      x.push(1)
    } else {
      x.push(len)
    }
    let acc=0
    for(let k=0; k < colY.length; k++){
      acc += colY[k]
    }
    this.parameters.push(acc)
    } else {
      const weightsCol = source.get_column(weights)
      if(weightsCol == null){
        throw ReferenceError("Column not defined: " + weights)
      }
      const colY = source.get_column(varY)
      if(colY == null) {
	throw ReferenceError("Column not defined: " + varY)
      }
      let len = source.get_length()
      if(len == null){
	len = 0
      }
      for(let i=0; i < varX.length; ++i){
	let iField = source.get_column(varX[i])
	if(iField == null){
	  throw ReferenceError("Column not defined: " + iField)
	}
	for(let j=0; j <= i; ++j){
	  let jField = source.get_column(varX[j])
	  if(jField == null) {
	    break
	  }
	  let acc = 0
	  for(let k=0; k < len; ++k){
	    acc += iField[k] * jField[k] * weightsCol[k]
	  }
	  x.push(acc)
	}
      }
      for(let i=0; i < varX.length; ++i){
	let col = source.get_column(varX[i])
	if(col == null){
	  throw ReferenceError("Column not defined: " + col)
	}
        let acc=0
	for(let k=0; k < len; ++k){
	  acc += col[k] * weightsCol[k]
	}
        x.push(acc)
	acc = 0
	for(let k=0; k < len; ++k){
	  acc += col[k] * colY[k] * weightsCol[k]
	}
	this.parameters.push(acc)
      } 
      let acc = 0
      for(let k=0; k < len; ++k){
	acc += weightsCol[k]
      }
      x.push(acc)
      acc = 0
      for(let k=0; k < len; ++k){
	acc += colY[k] * weightsCol[k]
      }
      this.parameters.push(acc)      
    }
    if(alpha > 0){
      let iRow = 0
      for(let i=0; i < this.parameters.length; ++i){
	x[iRow] += alpha
	iRow += i+2
      }
    }
    solve(x,this.parameters)
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
