import {Tabs, TabsView} from "models/layouts/tabs"
import { DownsamplerCDS } from "./DownsamplerCDS"
import * as p from "core/properties"

export namespace LazyTabs {
    export type Attrs = p.AttrsOf<Props>
  
    export type Props = Tabs.Props & {
      renderers: p.Property<(DownsamplerCDS | LazyTabs)[][]>
      watched: p.Property<boolean>
    }
  }
  
  export interface LazyTabs extends LazyTabs.Attrs {}
  
  export class LazyTabs extends Tabs {
    properties: LazyTabs.Props
  
    _last_index: number

    constructor(attrs?: Partial<LazyTabs.Attrs>) {
      super(attrs)
    }
  
    // The ``__name__`` class attribute should generally match exactly the name
    // of the corresponding Python class. Note that if using TypeScript, this
    // will be automatically filled in during compilation, so except in some
    // special cases, this shouldn't be generally included manually, to avoid
    // typos, which would prohibit serialization/deserialization of this model.
    static __name__ = "LazyTabs"
  
    static init_LazyTabs() {
      // This is usually boilerplate. In some cases there may not be a view.
      this.prototype.default_view = TabsView
      this.define<LazyTabs.Props>(({Ref, Array, Or, Boolean})=>({
        renderers:  [Array(Array(Or(Ref(LazyTabs), Ref(DownsamplerCDS))))],
        watched: [Boolean, true]
      }))
      }

    initialize(): void {
        super.initialize()
        const {active, watched, tabs} = this
        this._last_index = active
        if(this.renderers == null){
          this.renderers = tabs.map(_=>[])
        }
        for(let i=0; i<this.renderers.length; i++){
          for(let j=0; j<this.renderers[i].length; j++){
            this.renderers[i][j].watched = false
            this.renderers[i][j].on_visible_change()
          }
        }
        for(let j=0; j<this.renderers[active].length; j++){
          this.renderers[active][j].watched = watched
          this.renderers[active][j].on_visible_change()
        } 
      }

    connect_signals(): void {
        super.connect_signals()

        this.connect(this.properties.active.change, () => this.on_active_change())
        this.connect(this.properties.watched.change, () => this.on_visible_change())
    }

    on_active_change() {
        const {active, _last_index} = this
        const old_renderers = this.renderers[_last_index]
        for(let i=0; i<old_renderers.length; i++){
          old_renderers[i].watched = false
        }
        const active_renderers = this.renderers[active]
        for(let i=0; i<active_renderers.length; i++){
          active_renderers[i].watched = true
        }
        this._last_index = active
    }

    on_visible_change(){
      const {_last_index, watched} = this
      const active_renderers = this.renderers[_last_index]
      for(let i=0; i<active_renderers.length; i++){
        active_renderers[i].watched = watched
      }
   //   tabs[_last_index].child.visible = visible
    }
  }
