import {Tabs, TabsView} from "models/layouts/tabs"
import {Renderer} from "models/renderers/renderer"
import {LayoutDOM} from "models/layouts/layout_dom"
import * as p from "core/properties"

export namespace LazyTabs {
    export type Attrs = p.AttrsOf<Props>
  
    export type Props = Tabs.Props & {
      renderers: p.Property<(LayoutDOM | Renderer)[][]>
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
      this.define<LazyTabs.Props>(({Ref, Array, Or})=>({
        renderers:  [Array(Array(Or(Ref(LayoutDOM), Ref(Renderer))))]
      }))
      }

    initialize(): void {
        super.initialize()
    
        this._last_index = this.active
        if(this.renderers == null){
          this.renderers = this.tabs.map(_=>[])
        }
      }

    connect_signals(): void {
        super.connect_signals()

        this.connect(this.properties.active.change, () => this.on_active_change())
        this.connect(this.properties.visible.change, () => this.on_visible_change())
    }

    on_active_change() {
        const {active, tabs, _last_index} = this
        const old_renderers = this.renderers[_last_index]
        for(let i=0; i<old_renderers.length; i++){
          old_renderers[i].visible = false
        }
  //      tabs[_last_index].child.visible = false
        const active_renderers = this.renderers[active]
        for(let i=0; i<active_renderers.length; i++){
          active_renderers[i].visible = true
        }
        tabs[active].child.visible = true
  //      this._last_index = active
    }

    on_visible_change(){
      const {tabs, _last_index, visible} = this
      const active_renderers = this.renderers[_last_index]
      for(let i=0; i<active_renderers.length; i++){
        active_renderers[i].visible = visible
      }
   //   tabs[_last_index].child.visible = visible
    }
  }
