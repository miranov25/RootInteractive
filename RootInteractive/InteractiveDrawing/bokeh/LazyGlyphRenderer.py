from bokeh.models.renderers import GlyphRenderer

class LazyGlyphRenderer(GlyphRenderer):

    __implementation__ = "LazyGlyphRenderer.ts"
