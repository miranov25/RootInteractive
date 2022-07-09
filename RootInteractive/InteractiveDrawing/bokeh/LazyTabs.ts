import {Tabs, TabsView} from "models/layouts/tabs"
import * as p from "core/properties"

export namespace LazyTabs {
    export type Attrs = p.AttrsOf<Props>
  
    export type Props = Tabs.Props
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
      }

    initialize(): void {
        super.initialize()
    
        this._last_index = this.active
      }

    connect_signals(): void {
        super.connect_signals()

        this.connect(this.properties.active.change, () => this.on_active_change())
    }

    on_active_change() {
        const {active, tabs, _last_index} = this
        tabs[active].child.visible = true
        tabs[_last_index].child.visible = false
        this._last_index = active
    }
  }
