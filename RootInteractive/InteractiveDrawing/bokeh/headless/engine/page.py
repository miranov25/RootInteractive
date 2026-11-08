"""Page builder (proposal v0.2 §4, field `page`): turns a page declaration into RootInteractive's own
`bokehDrawSA.fromArray` call and writes a self-contained HTML file (inline Bokeh resources) for the browser, or the
page document (JSON) for the Node model harness (`build_document`, proposal v0.3).

The declaration, not the built page, is what the automatic oracle reads, so the oracle never depends on how
RootInteractive interpreted it. `histogram_models()` lists the histogram model names the page must contain.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .scenario import Hist, Page, Scenario

os.environ.setdefault("BOKEH_BROWSER", "none")


def histogram_models(page: Page) -> List[Tuple[str, Hist, str]]:
    """(model name, declaration, weights column or None) for every histogram model the page creates.
    A histogram weighted by a parameter becomes one model per option, named `name[option]`."""
    params = {p.name: p for p in page.parameters}
    out = []
    for h in page.histograms:
        if h.weights in params:
            for opt in params[h.weights].options:
                out.append((f"{h.name}[{opt}]", h, opt))
        else:
            out.append((h.name, h, h.weights))
    return out


def figure_array(page: Page) -> list:
    figs = []
    for h in page.histograms:
        x = "bin_center" if len(h.variables) == 1 else "bin_center_0"
        figs.append([[x], ["bin_count"], {"source": h.name}])
    for p in page.profiles:
        figs.append([["bin_center_0"], ["mean"], {"source": p}])
    return figs


def widget_params(page: Page) -> list:
    out = []
    for w in page.widgets:
        out.append(["range", [w.column] + ([] if w.limits is None else list(w.limits))])
    for p in page.parameters:
        out.append(["multiSelect", [p.name], {"name": p.name}])
    return out


def build(scenario: Scenario, df: pd.DataFrame, html_path: Path) -> Path:
    from bokeh.plotting import output_file
    from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA

    page = scenario.page
    figs = figure_array(page)
    widgets = widget_params(page)
    output_file(str(html_path), mode="inline")
    bokehDrawSA.fromArray(
        df, None, figs, widgets,
        sourceArray=[h.source() for h in page.histograms],
        aliasArray=[tuple(a) for a in page.aliases],
        parameterArray=[{"name": p.name, "value": list(p.value), "options": list(p.options)} for p in page.parameters],
        layout=[list(range(len(figs)))],
        widgetLayout=[list(range(len(widgets)))],
        arrayCompression=[(pat, list(actions)) for pat, actions in scenario.transport.array_compression],
    )
    return Path(html_path)


def build_document(scenario: Scenario, df: pd.DataFrame, json_path: Path, models_only=()) -> Path:
    """The page *document* for the Node model harness (proposal v0.3 D-24): the same `fromArray` call as `build`, but
    `show` is suppressed, so no HTML is written and no model set is compiled; the layout `fromArray` produced is put into
    a fresh Bokeh Document and written as JSON with its data inlined. Node builds it with Bokeh's own deserializer.
    `models_only` (stage 4.2c-2b): the document's roots are only the named data models (histograms, profiles, …) with
    everything they reference — the same builder, the same objects, without plots, axes and tools (measured: 157 → 15
    models, 11 times faster per case in Node). For mass cases that read only those models."""
    import json
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa

    page = scenario.page
    figs = figure_array(page)
    widgets = widget_params(page)
    shown = sa.show
    sa.show = lambda *a, **k: None                       # build only: nothing written, nothing compiled
    try:
        obj = sa.bokehDrawSA.fromArray(
            df, None, figs, widgets,
            sourceArray=[h.source() for h in page.histograms],
            aliasArray=[tuple(a) for a in page.aliases],
            parameterArray=[{"name": p.name, "value": list(p.value), "options": list(p.options)} for p in page.parameters],
            layout=[list(range(len(figs)))],
            widgetLayout=[list(range(len(widgets)))],
            arrayCompression=[(pat, list(actions)) for pat, actions in scenario.transport.array_compression],
        )
    finally:
        sa.show = shown
    doc = Document()
    if models_only:                                      # stage 4.2c-2b: only the named models and what they reference
        wanted = [m for m in obj.pAll.references() if getattr(m, "name", None) in set(models_only)
                  and type(m).__name__ in ("HistogramCDS", "HistoNdCDS", "HistoNdProfile", "CDSJoin", "CDSStack")]
        if sorted(m.name for m in wanted) != sorted(models_only):
            raise ValueError(f"models {sorted(models_only)} requested, found {sorted(m.name for m in wanted)}")
        for m in wanted:
            doc.add_root(m)
    else:
        doc.add_root(obj.pAll)
    Path(json_path).write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return Path(json_path)


def input_columns(scenario: Scenario) -> List[str]:
    """Columns of the main table the page sends (they must arrive decoded exactly, §7.5)."""
    cols = set(w.column for w in scenario.page.widgets)
    for h in scenario.page.histograms:
        cols.update(v for v in h.variables if v in scenario.data.main.columns)
    for _, h, wcol in histogram_models(scenario.page):
        if wcol is not None:
            cols.add(wcol)
    for _, expr in scenario.page.aliases:                 # aliases are computed in the browser from their inputs
        cols.update(t for t in re.findall(r"[A-Za-z_]\w*", expr) if t in scenario.data.main.columns)
    return sorted(cols)
