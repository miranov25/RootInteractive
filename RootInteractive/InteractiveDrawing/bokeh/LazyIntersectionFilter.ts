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

  static init_LazyIntersectionFilter() {
    this.define<LazyIntersectionFilter.Props>(({Ref, Array})=>({
      filters:  [Array(Ref(RIFilter)), []],
    }))
  }

  changed: Set<number>
  counts: number[]
  cached_vector: boolean[]
  old_values: boolean[][]

  initialize(){
    super.initialize()
    this.changed = new Set()
    for(let i=0; i<this.filters.length; i++){
        this.changed.add(i)
    }
    this.old_values = []
  }

  connect_signals(): void {
    super.connect_signals()

    for (let i=0; i<this.filters.length; i++) {
        this.connect(this.filters[i].change, ()=>this.changed.add(i))
    }
  }

  public v_compute(): boolean[]{
    let {cached_vector, filters} = this
    if (this.changed.size === 0 && cached_vector != null){
        return cached_vector
    }
    for(const x of this.changed){
        if(this.old_values.length <= x){
            this.old_values.length = x+1
        }
        if(this.old_values[x] == undefined){
            this.old_values[x] = []
        }
        const values = filters[x].v_compute()
        while(this.old_values[x].length < values.length){
            this.old_values[x].push(true)
        }
        if(this.counts == null){
            this.counts = Array(values.length).fill(0)
        }
        this.counts.length = values.length
        const invert = filters[x].invert
        const old_values = this.old_values[x]
        for(let i=0; i<values.length; i++){
            const new_value = values[i]!==invert
            this.counts[i] += new_value ? 1 : 0
            this.counts[i] -= old_values[i] ? 1 : 0
            old_values[i] = new_value
        }
    }
    this.changed.clear()
    let new_vector: boolean[] = this.cached_vector
    if (new_vector == null){
        new_vector = Array(this.counts.length)
    } else {
        new_vector.length = this.counts.length
    }
    for(let i=0; i<this.counts.length; i++){
        new_vector[i] = !this.counts[i]
    }
    this.cached_vector = new_vector
    return new_vector
  }
}
