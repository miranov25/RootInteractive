"""PHASE_5 Group 7: the pages of the invariance and oracle tests (Claude21.RI, 2026-11-22).

In plain words: each page is a real RootInteractive dashboard, built from a table without random numbers, so that the
tests and a person who opens the page see exactly the same data. An invariance is two definitions that must agree, for
example A+B normalized to B+A must give 0 (difference) or 1 (ratio) in every row; an oracle is the page compared with
numpy and pandas, the selected rows one by one and the histograms bin by bin.

D1, the template without weights (`getDefaultVarsNormAll`): the difference, ratio or log-ratio of two variables is
switched live with the control "diffFunc". The page opens with the invariances visible:
- varY = A+B normalized to varYNorm = B+A, and varZ = A*B normalized to varZNorm = B*A: the tabs histoXYNorm,
  histoXYNormZ and the diffZ maps show 0 (difference), 1 (ratio) or 0 (log-ratio) everywhere;
- the first tab, "Invariances", draws four alias columns against varX, for any selection:
  (A+B)-(B+A), A*B-B*A and ((A+B)-B)-A must lie on the line 0, (A+A)/A on the line 2, each point with no error bar
  (the error bar is the std of the rows in the X bin).
The tab "Select" has a range slider for A, B, C and D and a list for the categories cat and DDC, as a user's page.

D2, the template with weights (the page of `test_interactiveTemplateWeights`: `getDefaultVarsDiff` with several weights
at once, `multiAxis="weights"`, the lists varXMulti and varYMulti and their histograms): the list "weights" chooses any
of None (no weight), W, W1, W2, Wsel, A>8 and A<=8; every histogram then shows one series per chosen weight. Its first
tab, "Invariances", draws, against varX, four ratios or differences of weighted counts of one histogram that must be
1, 0, 2 and 1 in every bin: W1 against no weight, W2 - 2 W, W2 / W, and (A>8) + (A<=8) against no weight.

The table (`template_table`), row number i, every value a multiple of 1/8, none zero, so that sums, differences and
the ratio (A+A)/A are exact in binary floating point:
- A = (i % 32)/2 + 0.5, B = ((7i + i//32) % 32)/2 + 0.5, C = ((11i + i//128) % 32)/2 + 0.5, D = ((5i + i//64) % 32)/2
  + 0.5: halves from 0.5 to 16; within every block of 32 rows A and B take each of their 32 values once;
- cat = (i//32) % 4 (an integer category), DDC = "A0".."A3" from (i//128) % 4 (a text category);
- a sparse region, the last 16 rows: A = 16.5 (1 row), 17 (2 rows), 17.5 (3), 18 (4), 18.5 (6), B = 1 + k/8 for
  k = 0..15, so the X bins at the top hold 1 to 6 rows with different values (the profile's special cases).
The table of D2 adds the weights: W = 1 + ((13i + i//64) % 9)/8 (eighths from 1 to 2), W1 = 1, W2 = 2 W, and Wsel = 1
where cat > 0, else 0. Sums of such weights are exact.

D3, the ToyMC joins dashboard (Tools/generators/toy_mc_joins_dashboard.py, unchanged): the ToyMC generator's
collisions, tracks, V0s and cascades (9,200 collisions, seed 42: about 101,000 tracks, 22,800 V0s and 7,150 cascades)
and the generator's own rows without a parent (inject_orphan_test_rows: a track without a collision, a V0 without its
positive track, three cascades without a V0, without a bachelor, and with a V0 that misses a track). Every figure of a
track, a V0 or a cascade reads its parents through joins by row number; under the figures, one selection tab per table
(Collisions, Tracks, Pos, Neg, V0s, Cascades), each with its range sliders and its on/off switches. What you can check
by eye is the page's own tab "Consistency": the mass of every V0 and cascade computed in the browser from the joined
tracks against its stored mass (every point on the diagonal; the differences and the difference of the collisions'
vz of two tracks of one candidate at 0), for any selection. With --text-query: a small page (300 collisions) without
the selection tabs, with a text query on the joins TrkJ and V0J (test J18: today such a query does not see the
join's columns).

Build a page yourself (the tests build it the same way, in a child process):

    python -m RootInteractive.InteractiveDrawing.bokeh.headless.dashboards.invariance_dashboards template \\
        --rows 100000 --output /tmp/invariance_template.html

    python -m RootInteractive.InteractiveDrawing.bokeh.headless.dashboards.invariance_dashboards weights \\
        --rows 100000 --output /tmp/invariance_weights.html

    python -m RootInteractive.InteractiveDrawing.bokeh.headless.dashboards.invariance_dashboards toymc \\
        --collisions 9200 --output /tmp/invariance_toymc.html

It writes the page and, next to it, <page>.manifest.json: the table's checksum, the page's checksum, the versions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE_ROWS = 100_000
SPARSE_A = (16.5, 17.0, 17.0, 17.5, 17.5, 17.5, 18.0, 18.0, 18.0, 18.0, 18.5, 18.5, 18.5, 18.5, 18.5, 18.5)
# the variables of the template's lists varX ... varZNorm (the template adds funCustom0 to funCustom2); cat is listed
# so that a text query can use it
TEMPLATE_VARIABLES = ["A", "B", "C", "D", "cat", "A+B", "B+A", "A*B", "B*A", "(A+B)-B", "A+A", "A+exponentX*B"]
# the page opens with the invariances visible in the template's own tabs
TEMPLATE_DEFAULTS = {"varX": "A", "varY": "A+B", "varYNorm": "B+A", "varZ": "A*B", "varZNorm": "B*A"}
# the tab "Invariances": (alias name, expression, the value every row must have, plot title)
INVARIANCES = [
    ("invSum", "(A+B)-(B+A)", 0.0, "(A+B)-(B+A): every point at 0"),
    ("invProd", "A*B-B*A", 0.0, "A*B-B*A: every point at 0"),
    ("invSub", "((A+B)-B)-A", 0.0, "((A+B)-B)-A: every point at 0"),
    ("invDouble", "(A+A)/A", 2.0, "(A+A)/A: every point at 2"),
]
# D2: the variables, the weights of the list "weights" (None: no weight), the defaults, and the tab "Invariances":
# (alias of a figure, expression of weighted counts, the value every non-empty bin must have, plot title)
WEIGHTS_VARIABLES = ["A", "B", "C", "D", "cat", "A+B", "B+A", "A*B", "B*A"]
WEIGHTS = [None, "W", "W1", "W2", "Wsel", "A>8", "A<=8"]
WEIGHTS_DEFAULTS = {"varX": "A", "varY": "A+B", "varYNorm": "B+A", "varZ": "A*B", "varZNorm": "B*A"}
WEIGHT_COUNTS = {"nW": "W", "nW1": "W1", "nW2": "W2", "nP": "A>8", "nNotP": "A<=8"}
WEIGHT_INVARIANCES = [
    ("nW1/bin_count", 1.0, "W1 / no weight: every bin at 1"),
    ("nW2-2*nW", 0.0, "W2 - 2 W: every bin at 0"),
    ("nW2/nW", 2.0, "W2 / W: every bin at 2"),
    ("(nP+nNotP)/bin_count", 1.0, "((A>8) + (A<=8)) / no weight: every bin at 1"),
]
SELECT_RANGES = ["A", "B", "C", "D"]
SELECT_LISTS = ["cat", "DDC"]
# D3: the ToyMC size and seed; the small page with a text query on joins (test J18)
TOYMC_COLLISIONS = 9200
TOYMC_SEED = 42
TOYMC_QUERY_COLLISIONS = 300
TOYMC_QUERIES = [("TrkJ", "query_tracks"), ("V0J", "query_v0s")]       # (the join it acts on, the field's name)


def template_table(rows: int = TEMPLATE_ROWS) -> pd.DataFrame:
    """The table of D1 (see the module text). Deterministic: the same rows on every machine."""
    if rows < 64:
        raise ValueError("the template table needs at least 64 rows (16 of them are the sparse region)")
    i = np.arange(rows, dtype=np.int64)
    df = pd.DataFrame({"A": (i % 32) / 2 + 0.5, "B": ((7 * i + i // 32) % 32) / 2 + 0.5,
                       "C": ((11 * i + i // 128) % 32) / 2 + 0.5, "D": ((5 * i + i // 64) % 32) / 2 + 0.5})
    df.loc[rows - 16:, "A"] = SPARSE_A
    df.loc[rows - 16:, "B"] = 1 + np.arange(16) / 8
    df["cat"] = ((i // 32) % 4).astype(np.int64)
    df["DDC"] = np.asarray(["A0", "A1", "A2", "A3"])[(i // 128) % 4]
    return df


def weights_table(rows: int = TEMPLATE_ROWS) -> pd.DataFrame:
    """The table of D2: the table of D1 and the weights W, W1, W2, Wsel (see the module text)."""
    df = template_table(rows)
    i = np.arange(rows, dtype=np.int64)
    df["W"] = 1 + ((13 * i + i // 64) % 9) / 8
    df["W1"] = 1.0
    df["W2"] = 2 * df["W"]
    df["Wsel"] = (df["cat"] > 0).astype(np.float64)
    return df


def table_sha256(df: pd.DataFrame) -> str:
    """Checksum of the ordered table: the float64 little-endian bytes of A, B, C, D, cat (and of the weights W, W1, W2,
    Wsel when the table has them), then the DDC texts."""
    h = hashlib.sha256()
    for c in ["A", "B", "C", "D", "cat"] + [w for w in ("W", "W1", "W2", "Wsel") if w in df.columns]:
        h.update(np.ascontiguousarray(df[c].to_numpy(), dtype="<f8").tobytes())
    h.update("\n".join(df["DDC"].tolist()).encode())
    return h.hexdigest()


def toymc_tables(collisions: int = TOYMC_COLLISIONS, seed: int = TOYMC_SEED, orphans: bool = True) -> dict:
    """The four tables of D3: the ToyMC generator's events and, with `orphans`, its rows without a parent."""
    from RootInteractive.Tools.generators import toy_mc_events as gen
    raw = gen.generate_toy_mc_events(n_collisions=collisions, seed=seed)
    return gen.inject_orphan_test_rows(raw) if orphans else raw


def toymc_sha256(raw: dict) -> str:
    """Checksum of the four tables: per table (by name) and column (in order), the float64 little-endian bytes of a
    numeric column, the texts of any other."""
    h = hashlib.sha256()
    for name in sorted(raw):
        for c in raw[name].columns:
            h.update(f"{name}.{c}\n".encode())
            col = raw[name][c]
            if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                h.update(np.ascontiguousarray(col.to_numpy(dtype=np.float64), dtype="<f8").tobytes())
            else:
                h.update("\n".join(map(str, col.tolist())).encode())
    return h.hexdigest()


def toymc_layout(raw: dict, text_query: bool = False):
    """The ToyMC page with its selection tabs; or, with `text_query`, the figures and a text query on each join of
    TOYMC_QUERIES (no selection tabs)."""
    from RootInteractive.Tools.generators import toy_mc_joins_dashboard as dash
    if not text_query:
        return dash.toy_mc_joins_dashboard(raw)
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    d = dash.page_description(raw)
    shown = sa.show
    sa.show = lambda *a, **k: None
    try:
        return sa.bokehDrawSA.fromArray(d["main"], None, d["figures"],
                                        [["textQuery", {"source": j, "name": n}] for j, n in TOYMC_QUERIES],
                                        sourceArray=d["sources"], aliasArray=d["aliases"], layout=d["figure_layout"],
                                        widgetLayout=[list(range(len(TOYMC_QUERIES)))], nPointRender=5000).pAll
    finally:
        sa.show = shown


def build_toymc_page(output: Path, collisions: int = TOYMC_COLLISIONS, seed: int = TOYMC_SEED,
                     text_query: bool = False) -> dict:
    """Build D3 (or, with `text_query`, the small page of test J18) into `output` and write <output>.manifest.json."""
    import bokeh
    from bokeh.io import save
    from bokeh.resources import INLINE
    from ..engine import compile_cache
    compile_cache.install()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    raw = toymc_tables(collisions, seed)
    t1 = time.perf_counter()
    save(toymc_layout(raw, text_query), filename=str(output), resources=INLINE,
         title="RootInteractive invariance dashboard D3: the ToyMC joins" + (" (text queries on joins)" if text_query else ""))
    manifest = {
        "dashboard": "D3 ToyMC joins" + (", text queries on joins (test J18)" if text_query else ""),
        "collisions": collisions, "seed": seed, "rows": {k: len(v) for k, v in raw.items()},
        "table_sha256": toymc_sha256(raw), "page": output.name,
        "page_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "page_bytes": output.stat().st_size,
        "seconds": round(time.perf_counter() - t0, 2), "seconds_tables": round(t1 - t0, 2),
        "versions": {"python": sys.version.split()[0], "bokeh": bokeh.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    Path(str(output) + ".manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def template_arrays():
    """The template's arrays for D1: getDefaultVarsNormAll with the variables and defaults above, plus the tab
    "Select" and the tab "Invariances". Returns the keyword arguments of bokehDrawSA.fromArray, and the figures."""
    from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate import getDefaultVarsNormAll
    from RootInteractive.InteractiveDrawing.bokeh.bokehTools import mergeFigureArrays
    aliases, jsf, variables, parameters, widgets, widget_layout, histograms, figures, layout = getDefaultVarsNormAll(
        variables=list(TEMPLATE_VARIABLES), defaultVariables=dict(TEMPLATE_DEFAULTS))
    widgets = mergeFigureArrays(widgets, [["range", [c], {"name": c}] for c in SELECT_RANGES]
                                + [["multiSelect", [c], {"name": c}] for c in SELECT_LISTS])
    widget_layout["Select"] = [["A", "B", "C"], ["D", "cat", "DDC"]]
    options = figures.pop()                                       # the template's global figure options stay last
    first = len(figures)
    for name, expr, _value, title in INVARIANCES:
        aliases.append({"name": name, "expr": expr})
        histograms.append({"name": "histo_" + name, "variables": ["varX", name], "nbins": ["nbinsX", "nbinsY"],
                           "axis": [1], "quantiles": [0.5], "unbinned_projections": True})
        figures.append([["bin_center_0"], ["mean"], {"source": "histo_" + name + "_1", "errY": "std",
                                                     "yAxisTitle": title}])
    figures.append(options)
    idx = list(range(first, first + len(INVARIANCES)))
    layout = {"Invariances": [idx[:2], idx[2:], {"plot_height": 250}], **layout}
    return {"aliasArray": aliases, "jsFunctionArray": jsf, "parameterArray": parameters, "widgetLayout": widget_layout,
            "histogramArray": histograms, "layout": layout}, widgets, figures


def weights_arrays():
    """The arrays of D2, as `test_interactiveTemplateWeights` makes them (getDefaultVarsDiff with several weights, the
    lists varXMulti and varYMulti and their histograms), on the variables and weights above, plus the tab "Select" and
    the tab "Invariances". Returns the keyword arguments of bokehDrawSA.fromArray, the widgets and the figures."""
    from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate import getDefaultVarsDiff
    from RootInteractive.InteractiveDrawing.bokeh.bokehTools import mergeFigureArrays
    aliases, variables, parameters, widgets, widget_layout, histograms, figures, layout = getDefaultVarsDiff(
        variables=list(WEIGHTS_VARIABLES), defaultVariables=dict(WEIGHTS_DEFAULTS), weights=list(WEIGHTS),
        multiAxis="weights")
    # as in test_interactiveTemplateWeights
    parameters = parameters + [{"name": "varXMulti", "value": ["A", "B"], "options": variables},
                               {"name": "varYMulti", "value": ["A", "B"], "options": variables}]
    widgets = mergeFigureArrays(widgets, [["range", [c], {"name": c}] for c in SELECT_RANGES]
                                + [["multiSelect", [c], {"name": c}] for c in SELECT_LISTS]
                                + [["multiSelect", ["varXMulti"], {"name": "varXMulti"}],
                                   ["multiSelect", ["varYMulti"], {"name": "varYMulti"}]])
    widget_layout["Select"] = [["A", "B", "C"], ["D", "cat", "DDC"]]
    widget_layout["Histograms"][0].extend(["varXMulti", "varYMulti"])
    histograms = histograms + [
        {"name": "histo1D", "variables": ["varX"], "nbins": "nbinsX"},
        {"name": "histo1DMulti", "variables": ["varXMulti"], "nbins": "nbinsX"},
        {"name": "histoNDMultiX", "variables": ["varXMulti", "varY"], "quantiles": [0.35, 0.5],
         "unbinned_projections": True, "nbins": ["nbinsX", "nbinsY"]},
        {"name": "histoNDMultiY", "variables": ["varX", "varYMulti"], "quantiles": [0.35, 0.5],
         "unbinned_projections": True, "nbins": ["nbinsX", "nbinsY"]},
        # the tab "Invariances": one histogram of varX with the weighted counts as extra columns
        {"name": "histoInvW", "variables": ["varX"], "nbins": "nbinsX",
         "histograms": {name: {"weights": w} for name, w in WEIGHT_COUNTS.items()}},
    ]
    extra = [
        [["bin_center"], ["bin_count"], {"source": "histo1D", "name": "histo1D"}],
        [["bin_center"], ["bin_count"], {"source": "histo1DMulti", "name": "histo1DMulti"}],
        [["bin_center_0"], ["mean"], {"source": "histoNDMultiX_1", "name": "histoNDMultiX_1_Mean", "errY": "std/sqrt(entries)"}],
        [["bin_center_0"], ["quantile_1"], {"source": "histoNDMultiX_1", "name": "histoNDMultiX_1_Median", "errY": "std/sqrt(entries)"}],
        [["mean"], ["bin_center_1"], {"source": "histoNDMultiX_0", "name": "histoNDMultiX_0_Mean", "errX": "std/sqrt(entries)"}],
        [["quantile_1"], ["bin_center_1"], {"source": "histoNDMultiX_0", "name": "histoNDMultiX_0_Median", "errX": "std/sqrt(entries)"}],
        [["bin_center_0"], ["mean"], {"source": "histoNDMultiY_1", "name": "histoNDMultiY_1_Mean", "errY": "std/sqrt(entries)"}],
        [["bin_center_0"], ["quantile_1"], {"source": "histoNDMultiY_1", "name": "histoNDMultiY_1_Median", "errY": "std/sqrt(entries)"}],
        [["mean"], ["bin_center_1"], {"source": "histoNDMultiY_0", "name": "histoNDMultiY_0_Mean", "errX": "std/sqrt(entries)"}],
        [["quantile_1"], ["bin_center_1"], {"source": "histoNDMultiY_0", "name": "histoNDMultiY_0_Median", "errX": "std/sqrt(entries)"}],
    ] + [[["bin_center"], [expr], {"source": "histoInvW", "name": f"inv{k}", "yAxisTitle": title}]
         for k, (expr, _v, title) in enumerate(WEIGHT_INVARIANCES)]
    figures = mergeFigureArrays(figures, extra)
    layout["Histo1D"] = [["histo1D", "histo1DMulti"], {"plot_height": 350}]
    layout["HistoMultiX"] = [["histoNDMultiX_1_Mean", "histoNDMultiX_1_Median"], ["histoNDMultiX_0_Mean", "histoNDMultiX_0_Median"], {"plot_height": 240}]
    layout["HistoMultiY"] = [["histoNDMultiY_1_Mean", "histoNDMultiY_1_Median"], ["histoNDMultiY_0_Mean", "histoNDMultiY_0_Median"], {"plot_height": 240}]
    layout = {"Invariances": [["inv0", "inv1"], ["inv2", "inv3"], {"plot_height": 250}], **layout}
    return {"aliasArray": aliases, "parameterArray": parameters, "widgetLayout": widget_layout,
            "histogramArray": histograms, "layout": layout}, widgets, figures


def build_template_page(output: Path, rows: int = TEMPLATE_ROWS, weights: bool = False) -> dict:
    """Build D1 (or D2 with `weights`) into `output` (Bokeh inline resources) and write <output>.manifest.json.
    Returns the manifest."""
    import bokeh
    from bokeh.plotting import output_file
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    from ..engine import compile_cache
    # the disk cache of the compiled TypeScript models, as in capture.py: the folder comes from RI_HEADLESS_BUNDLE
    # (inherited from the test), and RI_COMPILE_CACHE=0 turns it off
    compile_cache.install()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if weights:
        df = weights_table(rows)
        kwargs, widgets, figures = weights_arrays()
        title, what = "RootInteractive invariance dashboard D2: the template with weights", "D2 template with weights (getDefaultVarsDiff, multiAxis weights)"
        about = {"variables": WEIGHTS_VARIABLES, "weights": WEIGHTS, "defaults": WEIGHTS_DEFAULTS,
                 "invariances": [[e, v] for e, v, _ in WEIGHT_INVARIANCES]}
    else:
        df = template_table(rows)
        kwargs, widgets, figures = template_arrays()
        title, what = "RootInteractive invariance dashboard D1: the template", "D1 template (getDefaultVarsNormAll)"
        about = {"variables": TEMPLATE_VARIABLES, "defaults": TEMPLATE_DEFAULTS,
                 "invariances": [[n, e, v] for n, e, v, _ in INVARIANCES]}
    output_file(str(output), mode="inline", title=title)
    sa.bokehDrawSA.fromArray(df, None, figures, widgets, **kwargs)
    manifest = {
        "dashboard": what, "rows": rows, "table_sha256": table_sha256(df),
        "page": output.name, "page_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "page_bytes": output.stat().st_size, **about, "seconds": round(time.perf_counter() - t0, 2),
        "versions": {"python": sys.version.split()[0], "bokeh": bokeh.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    Path(str(output) + ".manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a page of the Group 7 invariance and oracle tests.")
    sub = ap.add_subparsers(dest="dashboard", required=True)
    t = sub.add_parser("template", help="D1: the template without weights (getDefaultVarsNormAll)")
    t.add_argument("--rows", type=int, default=TEMPLATE_ROWS)
    t.add_argument("--output", required=True)
    w = sub.add_parser("weights", help="D2: the template with weights (the page of test_interactiveTemplateWeights)")
    w.add_argument("--rows", type=int, default=TEMPLATE_ROWS)
    w.add_argument("--output", required=True)
    m = sub.add_parser("toymc", help="D3: the ToyMC joins dashboard with its selection tabs")
    m.add_argument("--collisions", type=int, default=TOYMC_COLLISIONS)
    m.add_argument("--seed", type=int, default=TOYMC_SEED)
    m.add_argument("--text-query", action="store_true", help="the small page of test J18: text queries on joins")
    m.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    if a.dashboard == "toymc":
        m = build_toymc_page(Path(a.output), a.collisions, a.seed, text_query=a.text_query)
    else:
        m = build_template_page(Path(a.output), a.rows, weights=(a.dashboard == "weights"))
    print(json.dumps({k: m[k] for k in ("dashboard", "rows", "page_bytes", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
