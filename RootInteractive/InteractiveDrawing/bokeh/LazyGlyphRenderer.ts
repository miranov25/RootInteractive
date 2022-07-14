import {GlyphRenderer, GlyphRendererView} from "models/renderers/glyph_renderer"
import * as p from "core/properties"

export class LazyGlyphRendererView extends GlyphRendererView {
  model: GlyphRenderer

  set_data(indices: boolean) {
    if(this.model.visible){
      super.set_data(indices);
    }
  }
  
}

export namespace LazyGlyphRenderer {
    export type Attrs = p.AttrsOf<Props>
  
    export type Props = GlyphRenderer.Props
  }
  
  export interface LazyGlyphRenderer extends LazyGlyphRenderer.Attrs {}
  
  export class LazyGlyphRenderer extends GlyphRenderer {
    properties: LazyGlyphRenderer.Props
  
    constructor(attrs?: Partial<LazyGlyphRenderer.Attrs>) {
      super(attrs)
    }
  
    static __name__ = "LazyGlyphRenderer"
  
    static init_LazyGlyphRenderer() {
      // This is usually boilerplate. In some cases there may not be a view.
      this.prototype.default_view = LazyGlyphRendererView
      }

  }
