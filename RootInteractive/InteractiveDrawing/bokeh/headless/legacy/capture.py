"""Build one original example dashboard and save the real page (PHASE_4_7, Claude20.RI).

Runs in a CHILD process, so the example module is imported fresh with the opt-in table and Bokeh's output state stays
isolated: `RI_DASHBOARD_ROWS` and `RI_DASHBOARD_SEED` are set before the import (the module builds its table on
import), the working directory is the output folder (the example calls `output_file`), and `bokehDrawSA.show` is
replaced only during the call, to keep the real root `pAll` that `fromArray` passes to it. The dashboard function and
`fromArray` are not touched.

    python -m RootInteractive.InteractiveDrawing.bokeh.headless.legacy.capture FUNCTION ROWS SEED OUTDIR [--graph-only]

ROWS = 0 builds the original 500,000 unseeded rows (no opt-in); --graph-only skips the HTML (fidelity check).

writes OUTDIR/page.html (Bokeh inline resources), OUTDIR/table.npz (the raw columns A, B, C, D as built by the module),
OUTDIR/manifest.json (identity, model graph, timings).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

EXAMPLE = "RootInteractive.InteractiveDrawing.bokeh.test_bokehClientHistogram"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table_sha256(a) -> str:
    """Hash of the ordered raw table: float64 little-endian bytes of A, B, C, D, column by column."""
    import numpy as np
    h = hashlib.sha256()
    for col in "ABCD":
        h.update(np.ascontiguousarray(a[col], dtype="<f8").tobytes())
    return h.hexdigest()


def _field(v):
    """The column a glyph property reads: a plain name, a Field (with or without a transform), else None."""
    if isinstance(v, str):
        return v
    f = getattr(v, "field", None)
    if isinstance(f, str):
        return f + (" (transformed)" if getattr(v, "transform", None) is not None else "")
    if isinstance(v, dict) and isinstance(v.get("field"), str):
        return v["field"] + (" (transformed)" if v.get("transform") is not None else "")
    return None


def model_graph(root) -> dict:
    """What the page consists of, without data values or random ids: named models, widgets, tabs, renderers."""
    from bokeh.models import GlyphRenderer, Tabs
    named, widgets, tabs, renderers, histograms = [], [], [], [], []
    for m in root.references():
        t = type(m).__name__
        if getattr(m, "name", None):
            named.append([t, m.name])
        if t in ("RangeSlider", "Slider", "Select", "MultiSelect", "MultiChoice", "Spinner", "TextInput", "Toggle",
                 "CheckboxGroup", "RadioButtonGroup"):
            widgets.append([t, getattr(m, "title", None) or "", getattr(m, "name", None) or ""])
        if isinstance(m, Tabs):
            tabs.append([p.title for p in m.tabs])
        if isinstance(m, GlyphRenderer):
            g = m.glyph
            src = m.data_source
            fields = {k: _field(getattr(g, k)) for k in ("x", "y", "left", "right", "top", "bottom")
                      if hasattr(g, k) and _field(getattr(g, k)) is not None}
            renderers.append([type(g).__name__, type(src).__name__, getattr(src, "name", None) or "", fields])
        if t in ("HistogramCDS", "HistoNdCDS", "HistoNdProfile"):
            histograms.append([t, m.name, getattr(m, "nbins", None)])
    key = lambda x: json.dumps(x, sort_keys=True, default=str)
    return {"named": sorted(named, key=key), "widgets": sorted(widgets, key=key), "tabs": tabs,
            "renderers": sorted(renderers, key=key), "histograms": sorted(histograms, key=key)}


def widget_state(root) -> dict:
    """The widgets' own values as built, keyed by title: what the oracle and the test start from."""
    state = {}
    for m in root.references():
        t = type(m).__name__
        title = getattr(m, "title", None)
        if not title:
            continue
        if t == "RangeSlider":
            state[title] = {"type": t, "start": float(m.start), "end": float(m.end), "step": float(m.step),
                            "value": [float(v) for v in m.value]}
        elif t in ("MultiSelect", "Select"):
            state[title] = {"type": t, "value": m.value, "options": [o if isinstance(o, str) else o[0] for o in m.options]}
        elif t == "Slider":
            state[title] = {"type": t, "start": float(m.start), "end": float(m.end), "step": float(m.step),
                            "value": float(m.value)}
    return state


def main(function: str, rows: int, seed: int, outdir: str, html: bool = True) -> None:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)                                    # the example's own output_file(...) lands here, not in the repo
    if rows > 0:
        os.environ["RI_DASHBOARD_ROWS"], os.environ["RI_DASHBOARD_SEED"] = str(rows), str(seed)
    else:                                            # rows == 0: the original 500,000 unseeded rows (fidelity check)
        os.environ.pop("RI_DASHBOARD_ROWS", None)
        os.environ.pop("RI_DASHBOARD_SEED", None)
    t0 = time.perf_counter()
    import importlib
    import numpy as np
    example = importlib.import_module(EXAMPLE)
    sa = importlib.import_module("RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA")
    t_import = time.perf_counter() - t0
    captured = []
    original_show = sa.show
    sa.show = lambda obj, *a, **k: captured.append(obj)
    try:
        t1 = time.perf_counter()
        getattr(example, function)()
        t_build = time.perf_counter() - t1
    finally:
        sa.show = original_show
    if len(captured) != 1:
        raise SystemExit(f"capture: expected one page from {function}, got {len(captured)}")
    root = captured[0]
    t_html = None
    if html:                                         # html=False: model graph only (the 500,000-row fidelity check)
        from bokeh.embed import file_html
        from bokeh.resources import INLINE
        t2 = time.perf_counter()
        (out / "page.html").write_text(file_html(root, INLINE, f"RI legacy dashboard {function}"), encoding="utf-8")
        t_html = round(time.perf_counter() - t2, 3)
    df = example.df
    np.savez(out / "table.npz", **{c: df[c].to_numpy(dtype="<f8") for c in "ABCD"})
    src = Path(example.__file__)
    manifest = {
        "function": function, "rows": int(len(df)), "seed": seed if rows > 0 else None,
        "opt_in": rows > 0, "table_sha256": table_sha256({c: df[c].to_numpy() for c in "ABCD"}),
        "example_file": str(src.name), "example_sha256": _sha256_file(src),
        "page_sha256": _sha256_file(out / "page.html") if html else None,
        "page_bytes": (out / "page.html").stat().st_size if html else None,
        "seconds": {"import_and_table": round(t_import, 3), "build": round(t_build, 3), "html": t_html},
        "graph": model_graph(root), "widget_state": widget_state(root),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("function", "rows", "page_bytes", "seconds")}))


if __name__ == "__main__":
    fn, n, s, o = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    main(fn, n, s, o, html="--graph-only" not in sys.argv[5:])
