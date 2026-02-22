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

let nibble_lut = new Int32Array(16)
for(let i=0; i<16; i++){
  nibble_lut[i] = spread_nibble(i)
}

function from_bits_8(front: Int32Array | null, back: Int32Array | null, counter: Int32Array): boolean{
  let changed = false
  const len = front ? front.length : back ? back.length : 0
  for(let i=0; i<len; i++){
    let front_i32 = front ? front[i] : -1
    let back_i32 = back ? back[i] : -1
    changed ||= front_i32 !== back_i32
    for(let j=0; j<8; j++){
      const front_nibble = front_i32 & 15
      const back_nibble = back_i32 & 15
      const delta = nibble_lut[front_nibble] - nibble_lut[back_nibble]
      counter[i*8+j] += delta
      front_i32 >>>= 4
      back_i32 >>>= 4
    }
  }
  return changed
}

function decode_8(counter: Int32Array, out: any[]){
  for(let i=0; i<counter.length; i++){
    let c = counter[i]
    for(let j=0; j<4; j++){
      out[4*i+j] = (c & 0xff) === 0
      c >>>= 8
    }
  }
}

// Adapt booleans from old interface
function pack_booleans(values: any[], out: Int32Array | null){
  if(out == null || out.length <= values.length / 32){
    out = new Int32Array((values.length / 32) + 1)
  } else {
    out.fill(0)
  }
  for(let i=0; i<values.length; i++){
    if(values[i]){
      out[Math.floor(i / 32)] |= (1 << (i % 32))
    }
  }
  return out
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
  counts: Int32Array
  cached_vector: boolean[]
  old_values: (Int32Array | null)[]
  cached_indices: number[]
  changed_indices: boolean
  changed_booleans: boolean
  _changed_values: boolean
  _old_old_values: Int32Array | null
  _vectorize: boolean

  initialize(){
    super.initialize()
    this.changed = new Set()
    for(let i=0; i < this.filters.length; i++){
        this.changed.add(i)
    }
    this._vectorize = this.filters.length < 256
    this.changed_booleans = true
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
    this.update_counts()
    if(!this._changed_values){
      return
    }
    this.changed_indices = true
    this.changed_booleans = true
    this.change.emit()
  }

  resize_counts(new_length: number){
    if(this.counts == null || (this._vectorize && this.counts.length < new_length*8) || (!this._vectorize && this.counts.length < new_length*32)){
      if(this._vectorize){
        return this.counts = new Int32Array(new_length*8)
      } else {
        return this.counts = new Int32Array(new_length*32)
      }
    }
    return this.counts
  }

  update_from_bits(bits: Int32Array, old_values: Int32Array | null, invert: boolean ): Int32Array {
    let mask = 1
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
          this.counts[i * 32 + j] -= new_value ? 1 : 0
          this.counts[i * 32 + j] += old_value ? 1 : 0
          mask = mask << 1
          if (mask === 0) mask = 1
        }
      } else {
        for(let j=0; j < 32; j++){
          const new_value = (value & mask) !== 0
          const old_value = (old_value_word & mask) !== 0
          this.counts[i * 32 + j] -= new_value ? 1 : 0
          this.counts[i * 32 + j] += old_value ? 1 : 0
          mask = mask << 1
          if (mask === 0) mask = 1
        }
      }
    }
    return old_values
  }

  update_counts(){
    const {filters} = this
    this._changed_values = false
    for(const x of this.changed){
	      let mask = 1
        if(filters[x].active){
          if(this.old_values.length <= x){
            this.old_values.length = x+1
          }
          let values_bits = filters[x].as_bits(this._old_old_values)
          if(values_bits == null){
            values_bits = pack_booleans(filters[x].v_compute(), this._old_old_values)
          }
          const counts = this.resize_counts(values_bits.length)
          if(this._vectorize){
              if(filters[x].invert){
                this._changed_values = from_bits_8(values_bits, this.old_values[x], counts)
              } else {
                this._changed_values = from_bits_8(this.old_values[x], values_bits, counts)
              }
              [this._old_old_values, this.old_values[x]] = [this.old_values[x], values_bits]
          } else {
              this._old_old_values = this.update_from_bits(values_bits, this.old_values[x], filters[x].invert)
              this.old_values[x] = values_bits
          }
        } else {
          const old_values = this.old_values[x]
          if(old_values == null){
            continue
          }
          const counts = this.counts
          if(this._vectorize){
              if(filters[x].invert){
                this._changed_values = from_bits_8(null, this.old_values[x], counts)
              } else {
                this._changed_values = from_bits_8(this.old_values[x], null, counts)
              }
          } else {
            const l = this.counts.length
            for(let i=0; i < l; i++){
              let old_count = counts[i]
              counts[i] -= 1
              counts[i] += old_values[i >> 5] & mask ? 1 : 0
              this._changed_values ||= (!!old_count !== !!counts[i])
              old_values[i >> 5] |= mask
              mask = mask << 1
              if (mask === 0) mask = 1
            }   
          }         
        }
    }
    this.changed.clear()
  }

  public v_compute(): boolean[]{
    let {cached_vector} = this
    if (!this.changed_booleans){
        return cached_vector
    }
    this.update_counts()
    let new_vector: boolean[] = this.cached_vector
    if (new_vector == null){
        new_vector = this._vectorize ? Array(this.counts.length * 4).fill(false) : Array(this.counts.length).fill(false)
    } else {
        new_vector.length = this._vectorize ? this.counts.length * 4: this.counts.length
    }
    if (this._vectorize){
      decode_8(this.counts, new_vector)
    } else{
      for(let i=0; i < this.counts.length; i++){
          new_vector[i] = !this.counts[i]
      }
    }
    this.cached_vector = new_vector
    this.changed_booleans = false
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
