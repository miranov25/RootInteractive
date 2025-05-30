import {RIFilter} from "./RIFilter"
import * as p from "core/properties"

export namespace LazyIntersectionFilter {
  export type Attrs = p.AttrsOf<Props>

  export type Props = RIFilter.Props & {
    filters: p.Property<RIFilter[]>
  }
}

export interface LazyIntersectionFilter extends LazyIntersectionFilter.Attrs {}

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
  counts: number[]
  cached_vector: boolean[]
  old_values: number[][]
  cached_indices: number[]
  changed_indices: boolean
  _changed_values: boolean

  initialize(){
    super.initialize()
    this.changed = new Set()
    for(let i=0; i < this.filters.length; i++){
        this.changed.add(i)
    }
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
          if(this.old_values[x] == undefined){
              this.old_values[x] = []
          }
          const old_values = this.old_values[x]
          const values = filters[x].v_compute()
          while(this.old_values[x].length < values.length / 32 + 1){
              this.old_values[x].push(-1)
          }
          if(this.counts == null){
              this.counts = Array(values.length).fill(0)
          }
          this.counts.length = values.length
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
