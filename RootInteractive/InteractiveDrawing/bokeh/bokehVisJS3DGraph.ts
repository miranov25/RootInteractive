// This custom model wraps one part of the third-party vis.js library:
//
//     http://visjs.org/index.html
//
// Making it easy to hook up python data analytics tools (NumPy, SciPy,
// Pandas, etc.) to web presentations using the Bokeh server.
import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"
import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {LayoutItem} from "core/layout"
import * as p from "core/properties"



declare namespace vis {
  class Graph3d {
    constructor(el: HTMLElement, data: object, OPTIONS: object)
    setData(data: vis.DataSet): void
    setOptions(options: object): void
  }

  class DataSet {
    add(data: unknown): void
  }
}

// This defines some default options for the Graph3d feature of vis.js
// See: http://visjs.org/graph3d_examples.html for more details.
const OPTIONS = {
  width: '400px',
  height: '400px',
  // style: 'surface',
  style: 'dot-color',
  showPerspective: true,
  showGrid: true,
  keepAspectRatio: true,
  verticalRatio: 1.0,
  legendLabel: 'legendLabel',
  cameraPosition: {
    horizontal: -0.35,
    vertical: 0.22,
    distance: 1.8,
  },
}
// To create custom model extensions that will render on to the HTML canvas
// or into the DOM, we must create a View subclass for the model.
//
// In this case we will subclass from the existing BokehJS ``LayoutDOMView``
export class BokehVisJSGraph3DView extends LayoutDOMView {
  model: BokehVisJSGraph3D

  private _graph: vis.Graph3d

  initialize(): void {
    super.initialize()

    const url = "https://cdnjs.cloudflare.com/ajax/libs/vis/4.16.1/vis.min.js"
    const script = document.createElement("script")
    script.onload = () => this._init()
    script.async = false
    script.src = url
    document.head.appendChild(script)
  }

  private _init(): void {
    // Create a new Graph3s using the vis.js API. This assumes the vis.js has
    // already been loaded (e.g. in a custom app template). In the future Bokeh
    // models will be able to specify and load external scripts automatically.
    //
    // BokehJS Views create <div> elements by default, accessible as this.el.
    // Many Bokeh views ignore this default <div>, and instead do things like
    // draw to the HTML canvas. In this case though, we use the <div> to attach
    // a Graph3d to the DOM.
    this._graph = new vis.Graph3d(this.el, this.get_data(), OPTIONS)
    if(this.model.options3D !== null){
      this.model.options3D.xLabel=this.model.x
      this.model.options3D.yLabel=this.model.y
      this.model.options3D.zLabel=this.model.z
      this.model.options3D.legendLabel=this.model.style
      this._graph.setOptions(this.model.options3D)
    }

    // Set a listener so that when the Bokeh data source has a change
    // event, we can process the new data
    this.connect(this.model.data_source.change, () => {
      this._graph.setData(this.get_data())
      this._graph.setOptions(this.model.options3D)
    })

  }

  // This is the callback executed when the Bokeh data has an change. Its basic
  // function is to adapt the Bokeh data source to the vis.js DataSet format.
  get_data(): vis.DataSet {
    const data = new vis.DataSet()
    const source = this.model.data_source
    const x = source.get_column(this.model.x)
    const y = source.get_column(this.model.y)
    const z = source.get_column(this.model.z)
    const style = source.get_column(this.model.style)
    const n = source.get_length()
    if(x != null && y != null && z != null && style != null && n != null)
    for (let i = 0; i < n; i++) {
      data.add({
        x: x[i],
        y: y[i],
        z: z[i],
        style: style[i],
      })
    }
    return data
  }

  get child_models(): LayoutDOM[] {
    return []
  }

  _update_layout(): void {
    this.layout = new LayoutItem()
    this.layout.set_sizing(this.box_sizing())
  }
}

// We must also create a corresponding JavaScript BokehJS model subclass to
// correspond to the python Bokeh model subclass. In this case, since we want
// an element that can position itself in the DOM according to a Bokeh layout,
// we subclass from ``LayoutDOM``
export namespace BokehVisJSGraph3D {
  export type Attrs = p.AttrsOf<Props>

  export type Props = LayoutDOM.Props & {
    x: p.Property<string>
    y: p.Property<string>
    z: p.Property<string>
    style: p.Property<string>
    options3D:  p.Property<Record<string, any>>
    data_source: p.Property<ColumnarDataSource>
  }
}

export interface BokehVisJSGraph3D extends BokehVisJSGraph3D.Attrs {}

export class BokehVisJSGraph3D extends LayoutDOM {
  properties: BokehVisJSGraph3D.Props
  __view_type__: BokehVisJSGraph3DView

  constructor(attrs?: Partial<BokehVisJSGraph3D.Attrs>) {
    super(attrs)
  }

  // The ``__name__`` class attribute should generally match exactly the name
  // of the corresponding Python class. Note that if using TypeScript, this
  // will be automatically filled in during compilation, so except in some
  // special cases, this shouldn't be generally included manually, to avoid
  // typos, which would prohibit serialization/deserialization of this model.
  static __name__ = "BokehVisJSGraph3D"

  static init_BokehVisJSGraph3D() {
    // This is usually boilerplate. In some cases there may not be a view.
    this.prototype.default_view = BokehVisJSGraph3DView

    // The @define block adds corresponding "properties" to the JS model. These
    // should basically line up 1-1 with the Python model class. Most property
    // types have counterparts, e.g. ``bokeh.core.properties.String`` will be
    // ``p.String`` in the JS implementatin. Where the JS type system is not yet
    // as rich, you can use ``p.Any`` as a "wildcard" property type.
    this.define<BokehVisJSGraph3D.Props>({
      x:            [ p.String   ],
      y:            [ p.String   ],
      z:            [ p.String   ],
      style:        [ p.String   ],
      options3D:    [ p.Instance ],
      data_source:  [ p.Instance ],
    })
  }
}
