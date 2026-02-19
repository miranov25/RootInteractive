import {RIFilter} from "./RIFilter"
import * as p from "core/properties"

export namespace LazyIntersectionFilter {
  export type Attrs = p.AttrsOf<Props>

  export type Props = RIFilter.Props & {
    filters: p.Property<RIFilter[]>
  }
}

export interface LazyIntersectionFilter extends LazyIntersectionFilter.Attrs {}

export function spread_nibble(x: number): number {
  let y = x & 1
  y |= (x & 2) << 7
  y |= (x & 4) << 14
  y |= (x & 8) << 21
  return y
}

export class LazyIntersectionFilter extends RIFilter {
  properties: LazyIntersectionFilter.Props

  constructor(attrs?: Partial<LazyIntersectionFilter.Attrs>) {
    super(attrs)
  }

  static __name__ = "LazyIntersectionFilter"

  static {
    this.define<LazyIntersectionFilter.Props>(({Ref, Array})=>({
      filters:  [Array(Ref(RIFilter)), []],
    }))
  }

  changed: Set<number>
  counts: number[] | Int32Array
  cached_vector: boolean[]
  old_values: Int32Array[]
  cached_indices: number[]
  changed_indices: boolean
  _changed_values: boolean
  _old_old_values: Int32Array | null
  _vectorize: boolean

  initialize(){
    super.initialize()
    this.changed = new Set()
    for(let i=0; i < this.filters.length; i++){
        this.changed.add(i)
    }
    this._vectorize = this.filters.length < 256 ? true : false
    this.changed_indices = true
    this.old_values = []
  }

  connect_signals(): void {
    super.connect_signals()

    for (let i=0; i < this.filters.length; i++) {
        this.connect(this.filters[i].change, ()=>this.change_filter(i))
    }
  }

  change_filter(i: number){
    // Until we add dependency trees, just eagerly compute the values, if all equal don't emit change
    this.changed.add(i)
    this.v_compute()
    if(!this._changed_values){
      return
    }
    this.changed_indices = true
    this.change.emit()
  }

  resize_counts(new_length: number){
    if(this.counts == null || (this._vectorize && this.counts.length < new_length*8) || (!this._vectorize && this.counts.length < new_length*32)){
      if(this._vectorize){
        this.counts = new Int32Array(new_length*8)
      } else {
        this.counts = Array(new_length * 32).fill(0)
        this.counts.length = new_length * 32
      }
    }
  }

  update_from_bits(bits: Int32Array, old_values: Int32Array | null, invert: boolean ): Int32Array {
    let mask = 1
    this.resize_counts(bits.length)
    if(old_values == null || old_values.length < Math.ceil(bits.length)){
      old_values = new Int32Array(Math.ceil(bits.length))
      for(let i=0; i < old_values.length; i++){
        old_values[i] = -1
      }
    }
    for(let i=0; i < bits.length; i++){
      const value = bits[i]
      const old_value_word = old_values[i]
      this._changed_values ||= bits[i] !== old_values[i]
      if(invert){
        for(let j=0; j < 32; j++){
          const new_value = (value & mask) === 0
          const old_value = (old_value_word & mask) === 0
          this.counts[i * 32 + j] += new_value ? 1 : 0
          this.counts[i * 32 + j] -= old_value ? 1 : 0
          mask = mask << 1
          if (mask === 0) mask = 1
        }
      } else {
        for(let j=0; j < 32; j++){
          const new_value = (value & mask) !== 0
          const old_value = (old_value_word & mask) !== 0
          this.counts[i * 32 + j] += new_value ? 1 : 0
          this.counts[i * 32 + j] -= old_value ? 1 : 0
          mask = mask << 1
          if (mask === 0) mask = 1
        }
      }
    }
    return old_values
  }

  public v_compute(): boolean[]{
    let {cached_vector, filters} = this
    if (this.changed.size === 0 && cached_vector != null){
        return cached_vector
    }
    this._changed_values = false
    for(const x of this.changed){
	      let mask = 1
        if(filters[x].active){
          if(this.old_values.length <= x){
            this.old_values.length = x+1
          }
          const values_bits = filters[x].as_bits(this._old_old_values)
          if(values_bits != null){
            this._old_old_values = this.update_from_bits(values_bits, this.old_values[x], filters[x].invert)
            this.old_values[x] = values_bits
            continue
          }
          const values = filters[x].v_compute()
          if(this.old_values[x] == null){
              this.old_values[x] = new Int32Array(Math.ceil(values.length / 32))
              for(let i=0; i < this.old_values[x].length; i++){
                this.old_values[x][i] = -1 
              }
          }
          let old_values = this.old_values[x]
          if(old_values.length < Math.ceil(values.length / 32)){
            console.warn("Resizing old_values for filter " + x + " from " + old_values.length + " to " + (Math.ceil(values.length / 32)))
            const new_old_values = new Int32Array(Math.ceil(values.length / 32))
            for(let i=0; i < old_values.length; i++){
              new_old_values[i] = old_values[i]
            }
            for(let i=old_values.length; i < new_old_values.length; i++){
              new_old_values[i] = -1
            }
            this.old_values[x] = new_old_values
            old_values = new_old_values
          }
          this.resize_counts(values.length)
          const invert = filters[x].invert
          for(let i=0; i < values.length; i++){
            const new_value = values[i]!==invert
            let old_count = this.counts[i]
            this.counts[i] += new_value ? 1 : 0
            this.counts[i] -= old_values[i >> 5] & mask ? 1 : 0
            this._changed_values ||= (!!old_count !== !!this.counts[i])
            if (new_value){
              old_values[i >> 5] |= mask
            } else {
              old_values[i >> 5] &= ~mask
            }
            mask = mask << 1
            if (mask === 0) mask = 1
          }
        } else {
          const old_values = this.old_values[x]
          if(old_values == null){
            continue
          }
          const l = this.counts.length
          for(let i=0; i < l; i++){
            let old_count = this.counts[i]
            this.counts[i] += 1
            this.counts[i] -= old_values[i >> 5] & mask ? 1 : 0
	          this._changed_values ||= (!!old_count !== !!this.counts[i])
            old_values[i >> 5] |= mask
	          mask = mask << 1
	          if (mask === 0) mask = 1
          }            
        }
    }
    this.changed.clear()
    let new_vector: boolean[] = this.cached_vector
    if (new_vector == null){
        new_vector = Array(this.counts.length)
    } else {
        new_vector.length = this.counts.length
    }
    for(let i=0; i < this.counts.length; i++){
        new_vector[i] = !this.counts[i]
    }
    this.cached_vector = new_vector
    return new_vector
  }

  public get_indices(): number[] {
    if(!this.changed_indices){
      return this.cached_indices
    }
    const first = 0
    const isSelected = this.v_compute()
    const last = isSelected.length
    const indicesAll = this.cached_indices ?? []
    indicesAll.length = 0
    for (let i = first; i < last; i++){
      if (isSelected[i]){
          indicesAll.push(i);
      }
    }    
    this.cached_indices = indicesAll
    this.changed_indices = false
    return indicesAll
  }
}
