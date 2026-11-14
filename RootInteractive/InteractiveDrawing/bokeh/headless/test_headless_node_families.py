"""The first Node families (proposal v0.3 §7 items D, E, H, J; stage 4.2c-2a, v0.2 after the review).

Histograms (widget-free pages, mass cases by template substitution — erratum V-1). One CASE is one data table put
into a page with a 1-D, a 2-D and a weighted 1-D histogram; Node builds the page, Python compares every bin.
  - per configuration: single-row NAMED cases at every bin edge and its rounding neighbours, the range limits,
    +-Infinity and far values, each with y = 0.5 and with y = 1.0 (the last bin of y); GENERATED cases from the
    edge-value column kind;
  - check 1, the implementation: the page equals the oracle that copies RootInteractive's arithmetic, in every cell;
  - check 2, the RULE (oracles.histograms.rule_bounds, the per-cell rule): every finite value inside the ranges is
    counted once, in a cell that contains it (interior-edge values may go to either side). PHASE_5 Batch 2A fixed
    B10: the page must satisfy the rule as it is, the "edge rows" (a value one rounding step below the top edge of x
    whose bin computes as nbins) included - they are counted in the last bin of x. With B10 put back by the fault
    B10_REINTRODUCED (configuration claude3_b10_6, the only one with such rows), the rule's violations must be
    exactly the B10 rows: in 1-D they are dropped, in 2-D they move to bin 0 of x in the NEXT bin of y, or are
    dropped when y is in its last bin. Each B10 row is classified by its destination, repaired, and the repaired
    page must satisfy the rule — else FAIL (the classification stays proven, so the fault stays a faithful B10);
  - the gate of V-1 (substituted documents equal fresh builds as model graphs, references kept); isolation; timings.
Filters (document cases): B1 with the refined rule (b1_first_changed) swept over 1-200 rows; 2 to 5 filters in every
  move order; the 20 000 / 20 001-row pair. Every expected B1 step is computed from the rule, every other must PASS.
  PHASE_5 Batch 1 fixed B1: each filter family runs twice - on the fixed code, where every step must PASS (the
  intersection of all filters), and with B1 put back by the fault B1_REINTRODUCED, where the refined rule must still
  predict every step exactly (the rule stays proven, so the fault keeps being a faithful copy of B1).
"""
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import data as data_mod
from RootInteractive.InteractiveDrawing.bokeh.headless.engine import faults
from RootInteractive.InteractiveDrawing.bokeh.headless.engine import sweep
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.expect import EqualsSelection, expected_failure
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import LOAD_CHECK, run
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
    Labels, Page, Scenario, TableSpec, Tables, auto, hist, load, range_, set_)
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import filters as filt
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import histograms as ho

pytestmark = [pytest.mark.backend("node"), pytest.mark.layer("integration")]

CONFIGS = (("unit_10", 10, 0.0, 1.0), ("claude3_b10_6", 6, -2.0, 2.28), ("odd_7", 7, -1.3, 2.9))
GENERATED_PER_CONFIG = {"fast": 300, "full": 3400}  # generated tables per configuration (full: 3 x 3400 = 10 200)
MODELS = ("h1", "h2", "hw")                      # documents hold only these models and what they reference (4.2c-2b)
READ = """() => [window.__riH.col('h1', 'bin_count', ['HistogramCDS']), window.__riH.col('h2', 'bin_count', ['HistoNdCDS']),
                 window.__riH.col('hw', 'bin_count', ['HistogramCDS'])]"""
B1_RULE = "b1_later_filters_keep_rows_excluded_at_load"
CODES = [pytest.param(False, id="fixed"), pytest.param(True, id="b1_put_back")]   # PHASE_5 Batch 1
B10_CONFIG = CONFIGS[1]                          # the configuration with edge rows (B10's rows); PHASE_5 Batch 2A


@pytest.fixture(scope="module")
def b1_back_session(node_session, node_bundle_root):
    """A Node session whose bundle copy has B1 put back (engine/faults.py B1_REINTRODUCED). One window for all its
    documents, isolated per document as the healthy session is (timers, channels, every added global - the fault
    counter too): a fresh window per document, the default of fault sessions, exhausts Node's memory over these
    families' hundreds of documents (measured: the process ended after about 170)."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import NodeSession
    s = NodeSession(node_bundle_root, fault=faults.B1_REINTRODUCED, fresh_window=False)
    yield s
    s.close()


@pytest.fixture(scope="module")
def b10_back_session(node_session, node_bundle_root):
    """A Node session whose bundle copy has B10 put back (engine/faults.py B10_REINTRODUCED), one window as above."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import NodeSession
    s = NodeSession(node_bundle_root, fault=faults.B10_REINTRODUCED, fresh_window=False)
    yield s
    s.close()


def _filter_run(b1_live, node_session, b1_back_session, sc, out):
    if b1_live:
        return run(b1_back_session, sc, out / "b1_put_back", test_ref=__file__, fault=faults.B1_REINTRODUCED)
    return run(node_session, sc, out, test_ref=__file__)


def _page(name, nb, lo, hi):
    return Scenario(
        id=f"families/histograms_node_{name}",
        data=Tables(TableSpec(200, 1, {"x": f"edges({lo},{hi},{nb})", "y": "edges(0,1,4)", "w": "uniform(0.5,2)"})),
        page=Page(histograms=(hist("h1", ["x"], nb, [lo, hi]), hist("h2", ["x", "y"], [nb, 4], [[lo, hi], [0, 1]]),
                              hist("hw", ["x"], nb, [lo, hi], weights="w"))),
        steps=(load(),), checks=(auto(),), labels=Labels(family="histograms", technique="template sweep"))


def _named_frames(nb, lo, hi):
    edges = lo + (hi - lo) * np.arange(nb + 1) / nb
    edges[-1] = hi
    values = [np.inf, -np.inf, lo - 1.0, hi + 1.0]
    for e in edges:
        values += [e, np.nextafter(e, -np.inf), np.nextafter(e, np.inf)]
    return [pd.DataFrame({"x": [v], "y": [y], "w": [1.5]}) for y in (0.5, 1.0) for v in values]


def _generated_frames(nb, lo, hi, count):
    out = []
    for seed in range(count):
        n = 50 + (seed * 37) % 251
        out.append(data_mod.generate(TableSpec(n, 10_000 + seed, {"x": f"edges({lo},{hi},{nb})", "y": "edges(0,1,4)",
                                                                  "w": "uniform(0.5,2)"})))
    return out


@dataclass
class Judgement:
    """The verdict on one case."""
    implementation_equal: bool          # check 1: page = implementation oracle in every cell (1-D, 2-D, weighted)
    unexplained: List[str]              # check 2: rule violations (B10 put back: after repairing its rows; must be empty)
    edge_rows: int                      # rows one rounding step below the top edge whose bin computes as nbins
    edge_rows_y_inside: int             # ... of them with y inside its range (the ones the 2-D histogram counts)
    b10_1d_dropped: int                 # B10 put back: B10 rows dropped by the 1-D histogram
    b10_2d_moved: int                   # ... moved by the 2-D histogram to bin 0 of x in the next bin of y
    b10_2d_dropped: int                 # ... dropped by the 2-D histogram (y in its last bin)


def judge(df, page, nb, lo, hi, b10_live: bool = False) -> Judgement:
    """The verdict on one case. `b10_live`: the page runs with B10 put back (fault B10_REINTRODUCED); its B10 rows are
    then classified and repaired before the rule is applied. Without it (the fixed code) nothing is repaired."""
    x, y, w = (df[c].to_numpy() for c in ("x", "y", "w"))
    p1, p2, pw = (np.asarray(p, dtype=np.float64) for p in page)
    impl_equal = (np.array_equal(p1, ho.counts([x], [nb], [(lo, hi)], b10=b10_live))
                  and np.array_equal(p2, ho.counts([x, y], [nb, 4], [(lo, hi), (0.0, 1.0)], b10=b10_live))
                  and np.allclose(pw, ho.counts([x], [nb], [(lo, hi)], weights=w, b10=b10_live), rtol=1e-12, atol=1e-9))
    with np.errstate(invalid="ignore"):
        scale = nb / (hi - lo)
        x_in = np.isfinite(x) & (x >= lo) & (x <= hi)
        y_in = np.isfinite(y) & (y >= 0.0) & (y <= 1.0)
        raw = np.where(x_in, x, lo) * scale - lo * scale                 # RootInteractive's arithmetic on axis x
        b10 = x_in & (x < hi) & (np.trunc(raw) == nb)                    # the B10 signature
    edge, edge_y = int(b10.sum()), int((b10 & y_in).sum())
    iy = np.clip(ho.bin_index_1d(np.where(y_in, y, 0.0), 4, 0.0, 1.0), 0, 3)
    if not b10_live:                                                     # the fixed code: nothing to repair
        b10 = np.zeros_like(b10)
    moved = b10 & y_in & (iy < 3)
    dropped2 = b10 & y_in & (iy == 3)
    fix1, fixw, fix2 = p1.copy(), pw.copy(), p2.copy()                   # repair the B10 rows, then apply the rule
    np.add.at(fix1, np.full(b10.sum(), nb - 1), 1.0)
    np.add.at(fixw, np.full(b10.sum(), nb - 1), w[b10])
    np.add.at(fix2, (nb - 1) + nb * iy[moved | dropped2], 1.0)
    np.add.at(fix2, nb * (iy[moved] + 1), -1.0)
    problems = []
    for name, cols, nbins, rng, wt, fixed in (("1-D", [x], [nb], [(lo, hi)], None, fix1),
                                              ("2-D", [x, y], [nb, 4], [(lo, hi), (0.0, 1.0)], None, fix2),
                                              ("weighted", [x], [nb], [(lo, hi)], w, fixw)):
        lower, upper, total = ho.rule_bounds(cols, nbins, rng, weights=wt)
        problems += [f"{name}: {v}" for v in ho.rule_violations(fixed, lower, upper, total)]
    return Judgement(impl_equal, problems, edge, edge_y, int(b10.sum()), int(moved.sum()), int(dropped2.sum()))


def _histogram_family_run(node_session, tpl, frames):
    docs, t_sub = sweep.substitute_all(tpl, frames)
    pages, t = sweep.run_batch(node_session, docs, READ)
    assert len(pages) == len(frames)                                     # (run_batch also checks per chunk)
    return pages, t_sub, t["node_s"]


# targets (4.2c-2c): the sample size belongs to node_histograms / node_filters, the full size to the *_full targets;
# `slow` remains only for the run without --target (today's default suite leaves the full sizes out)
HIST_SIZES = [pytest.param("fast", id="fast", marks=pytest.mark.node_histograms),
              pytest.param("full", id="full", marks=[pytest.mark.slow, pytest.mark.node_histograms_full])]
B1_SIZES = [pytest.param("fast", id="fast", marks=pytest.mark.node_filters),
            pytest.param("full", id="full", marks=[pytest.mark.slow, pytest.mark.node_filters_full])]


@pytest.mark.parametrize("size", HIST_SIZES)
@pytest.mark.parametrize("config", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_histogram_family(config, size, node_session, headless_out):
    """One test per configuration and size (parallel workers run them side by side). The fast default suite judges
    every named case and 300 generated tables per configuration; the full suite (marker `slow`, the existing RootInteractive marker) 3 400."""
    name, nb, lo, hi = config
    outdir = headless_out / f"families__histograms_node_{name}_{size}"
    outdir.mkdir(parents=True, exist_ok=True)
    res = ScenarioResult(id=f"families/histograms_node_{name}_{size} [node]", outdir=str(outdir),
                         versions=dict(node_session.versions), reproduce=f"pytest '{os.environ.get('PYTEST_CURRENT_TEST', __file__).rsplit(' (', 1)[0]}'")
    res.backend = "node"
    timings, failures = {}, []
    sc = _page(name, nb, lo, hi)
    t0 = time.perf_counter()
    tpl = sweep.Template(sc, data_mod.generate(sc.data.main), outdir / "template_document.json",
                         models_only=MODELS)
    timings["template_build_s"] = round(time.perf_counter() - t0, 3)
    sample = _generated_frames(nb, lo, hi, 11) + [_named_frames(nb, lo, hi)[4]]   # includes a one-row table
    eq = sum(tpl.equivalent(df, outdir / "fresh_document.json") for df in sample)
    res.add(0, name, f"template equals a fresh build ({len(sample)} sampled tables)", "PASS" if eq == len(sample) else "FAIL",
            f"{eq} of {len(sample)} substituted documents equal a fresh Python build as model graphs")
    for kind, frames in (("named", _named_frames(nb, lo, hi)), ("generated", _generated_frames(nb, lo, hi, GENERATED_PER_CONFIG[size]))):
        pages, t_sub, t_node = _histogram_family_run(node_session, tpl, frames)
        verdicts = [judge(df, p, nb, lo, hi) for df, p in zip(frames, pages)]
        compared = len(verdicts)
        bad_impl = [i for i, v in enumerate(verdicts) if not v.implementation_equal]
        bad_rule = [i for i, v in enumerate(verdicts) if v.unexplained]
        edge = sum(v.edge_rows for v in verdicts)
        failures += [(kind, i) for i in bad_impl + bad_rule]
        res.add(0, name, f"{kind} cases ({compared}): page = implementation oracle (1-D, 2-D, weighted)",
                "PASS" if not bad_impl and compared == len(frames) else "FAIL",
                f"{compared - len(bad_impl)} of {len(frames)} cases compared and equal in every cell"
                + (f"; first failing case {bad_impl[0]}" if bad_impl else ""), feature="HIST.histogram_1d")
        res.add(0, name, f"{kind} cases ({compared}): rule check, edge rows {edge} counted in the last bin of x",
                "FAIL" if bad_rule else "PASS",
                "the per-cell rule holds in all three histograms, the edge rows included (B10 fixed, PHASE_5 Batch 2A)"
                + (f"; unexplained in case {bad_rule[0]}: {verdicts[bad_rule[0]].unexplained[:2]}" if bad_rule else ""),
                feature="HIST.histogram_nd")
        timings[kind] = {"cases": len(frames), "compared": compared, "substitution_s": round(t_sub, 3),
                         "node_s": round(t_node, 3), "ms_per_case": round(1000 * (t_sub + t_node) / len(frames), 3)}
    frames = _generated_frames(nb, lo, hi, 200)                          # isolation
    docs, _ = sweep.substitute_all(tpl, frames)
    order = list(range(len(docs)))
    random.Random(nb).shuffle(order)
    straight, _ = sweep.run_batch(node_session, docs + [docs[0]], READ)
    shuffled, _ = sweep.run_batch(node_session, [docs[i] for i in order], READ)
    same = all(json.dumps(shuffled[k]) == json.dumps(straight[i]) for k, i in enumerate(order))
    res.add(0, name, "isolation: 200 cases in shuffled order and a repeated first case",
            "PASS" if same and json.dumps(straight[-1]) == json.dumps(straight[0]) else "FAIL",
            "equal results in both orders; the first case repeated at the end equal")
    (outdir / "timings.json").write_text(json.dumps(timings, indent=1))
    res.write()
    assert res.ok and not failures, res.text() + f"\nfailures (kind, case): {failures[:10]}"


@pytest.mark.node_histograms
def test_histogram_family_with_b10_put_back(b10_back_session, headless_out):
    """PHASE_5 Batch 2A: the named and the fast generated cases of the configuration with edge rows, with B10 put back
    (fault B10_REINTRODUCED). The page must equal the implementation oracle with B10 (RB-2.6, `b10=True`) in every
    cell, every edge row must be a B10 row at its predicted destination (dropped in 1-D; moved, or dropped in the last
    bin of y, in 2-D), and once repaired the rule must hold: the classification stays proven."""
    name, nb, lo, hi = B10_CONFIG
    sc = _page(name, nb, lo, hi)
    tpl = sweep.Template(sc, data_mod.generate(sc.data.main), headless_out / f"b10_put_back_{name}_template.json",
                         models_only=MODELS)
    for kind, frames in (("named", _named_frames(nb, lo, hi)),
                         ("generated", _generated_frames(nb, lo, hi, GENERATED_PER_CONFIG["fast"]))):
        pages, _, _ = _histogram_family_run(b10_back_session, tpl, frames)
        verdicts = [judge(df, p, nb, lo, hi, b10_live=True) for df, p in zip(frames, pages)]
        bad = [(i, v) for i, v in enumerate(verdicts) if not v.implementation_equal or v.unexplained]
        assert not bad, f"{kind}: {bad[:3]}"
        edge, edge_y, d1, m2, d2 = (sum(getattr(v, f) for v in verdicts) for f in
                                    ("edge_rows", "edge_rows_y_inside", "b10_1d_dropped", "b10_2d_moved", "b10_2d_dropped"))
        assert edge > 0 and d1 == edge and m2 + d2 == edge_y and m2 > 0 and d2 > 0, (kind, edge, edge_y, d1, m2, d2)


# ------------------------------------------------------------------- controls: the reviewers' counterexamples
@pytest.mark.node_histograms
def test_rule_check_catches_a_row_missing_from_page_and_oracle():
    """GPT23: oracle and page both missing a healthy row must fail the rule (the oracle shares the error)."""
    df = pd.DataFrame({"x": [0.33], "y": [0.5], "w": [1.5]})
    empty = [np.zeros(10), np.zeros(40), np.zeros(10)]
    v = judge(df, empty, 10, 0.0, 1.0)
    assert v.unexplained and any("total" in u for u in v.unexplained), v


def _oracle_pages(df, nb, lo, hi):
    return [ho.counts([df.x], [nb], [(lo, hi)]), ho.counts([df.x, df.y], [nb, 4], [(lo, hi), (0.0, 1.0)]),
            ho.counts([df.x], [nb], [(lo, hi)], weights=df.w)]


@pytest.mark.node_histograms
def test_rule_guards_2d_and_weighted_on_their_own():
    """4.2c-2a approval A-1 (Claude10, GPT23): each must leave an unexplained violation in ITS histogram, while the
    other histograms are healthy — a 2-D row moved to another bin of y (total unchanged), a weight moved to a far
    bin, and a row missing from the 2-D histogram only (also from its oracle)."""
    df = pd.DataFrame({"x": [0.33], "y": [0.5], "w": [1.5]})
    p1, p2, pw = _oracle_pages(df, 10, 0.0, 1.0)
    moved_y = p2.copy(); moved_y[3 + 10 * 2] = 0; moved_y[3 + 10 * 0] = 1
    moved_w = pw.copy(); moved_w[3] = 0; moved_w[7] = 1.5
    missing_2d = np.zeros_like(p2)
    for page, which in (([p1, moved_y, pw], "2-D"), ([p1, p2, moved_w], "weighted"), ([p1, missing_2d, pw], "2-D")):
        v = judge(df, page, 10, 0.0, 1.0)
        assert v.unexplained and all(u.startswith(which) for u in v.unexplained), (which, v)


@pytest.mark.node_histograms
def test_rule_tolerance_at_an_edge_at_zero():
    """4.2c-2a approval A-2 (Claude10): a range [-1, 1] has an interior edge at 0; values a few 1e-17 below it may land
    on either side, and a healthy page must not be flagged."""
    for xv in (-1e-18, -1e-17, -5e-17, -1e-16, 1e-17):
        df = pd.DataFrame({"x": [xv], "y": [0.5], "w": [1.5]})
        v = judge(df, _oracle_pages(df, 10, -1.0, 1.0), 10, -1.0, 1.0)
        assert not v.unexplained, (xv, v)


def _b10_pages(df, b10):
    return [ho.counts([df.x], [6], [(-2.0, 2.28)], b10=b10),
            ho.counts([df.x, df.y], [6, 4], [(-2.0, 2.28), (0.0, 1.0)], b10=b10),
            ho.counts([df.x], [6], [(-2.0, 2.28)], weights=df.w, b10=b10)]


@pytest.mark.node_histograms
def test_b10_destination_in_2d_moved_or_dropped():
    """GPT22/GPT23/GPT24/Claude10: with B10 put back, a B10 value moves to bin 0 of x in the next bin of y, or is
    dropped when y is in its last bin; both are classified, repaired, and leave no unexplained violation."""
    x = np.nextafter(2.28, -np.inf)
    for y, moved, dropped in ((0.5, 1, 0), (1.0, 0, 1)):
        df = pd.DataFrame({"x": [x], "y": [y], "w": [1.5]})
        v = judge(df, _b10_pages(df, True), 6, -2.0, 2.28, b10_live=True)
        assert (v.b10_2d_moved, v.b10_2d_dropped, v.b10_1d_dropped) == (moved, dropped, 1) and not v.unexplained, v


@pytest.mark.node_histograms
def test_b10_fixed_edge_row_is_in_the_last_bin_and_b10_is_a_violation():
    """PHASE_5 Batch 2A: on the fixed code an edge row is counted in the last bin of x and satisfies the rule as it is;
    a page with B10 (the edge row dropped or moved) judged as fixed code is a violation - nothing is repaired."""
    x = np.nextafter(2.28, -np.inf)
    for y in (0.5, 1.0):
        df = pd.DataFrame({"x": [x], "y": [y], "w": [1.5]})
        fixed = _b10_pages(df, False)
        assert fixed[0][5] == 1 and fixed[2][5] == 1.5 and fixed[1][5 + 6 * (3 if y == 1.0 else 2)] == 1
        v = judge(df, fixed, 6, -2.0, 2.28)
        assert v.implementation_equal and not v.unexplained and v.edge_rows == 1 and v.b10_1d_dropped == 0, v
        bad = judge(df, _b10_pages(df, True), 6, -2.0, 2.28)
        assert not bad.implementation_equal and bad.unexplained, bad


@pytest.mark.node_histograms
def test_template_check_sees_swapped_data_sources(headless_out):
    """GPT23/GPT24: two renderers with exchanged data sources must not count as the same document."""
    name, nb, lo, hi = CONFIGS[0]
    sc = _page(name, nb, lo, hi)
    tpl = sweep.Template(sc, data_mod.generate(sc.data.main), headless_out / "swap_template.json")
    doc = json.loads(tpl.document(data_mod.generate(sc.data.main)))
    renderers = []

    def find(o):
        if isinstance(o, dict):
            if o.get("type") == "object" and o.get("name") == "GlyphRenderer":
                renderers.append(o)
            for v in o.values():
                find(v)
        elif isinstance(o, list):
            for v in o:
                find(v)
    find(doc)
    r1, r2 = [r for r in renderers if "data_source" in r["attributes"]][:2]
    swapped = json.loads(json.dumps(doc))
    renderers.clear(); find(swapped)
    s1, s2 = [r for r in renderers if "data_source" in r["attributes"]][:2]
    s1["attributes"]["data_source"], s2["attributes"]["data_source"] = r2["attributes"]["data_source"], r1["attributes"]["data_source"]
    assert sweep.content(doc) == sweep.content(json.loads(json.dumps(doc)))
    assert sweep.content(doc) != sweep.content(swapped)


@pytest.mark.node_histograms
def test_short_batch_reply_is_an_error():
    """GPT23/GPT24: fewer results than documents must fail, with the batch's index."""
    class Short:
        startup_page_errors, startup_console_errors = [], []

        def request(self, op, **kw):
            return {"ok": True, "result": [{"result": 1, "isolation": {"not_removed": []}}] * (len(kw["docs"]) - 1)}
    with pytest.raises(RuntimeError, match="3 documents sent, 2 results returned"):
        sweep.run_batch(Short(), ["{}", "{}", "{}"], "() => 1")


@pytest.mark.node_histograms
def test_batch_refuses_a_session_with_startup_errors(node_bundle_root, tmp_path):
    """GPT24: the start-up error policy applies to batches too."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import NodeSession
    bad = tmp_path / "startup_error.js"
    bad.write_text("console.error('unexpected error at start-up');")
    s = NodeSession(node_bundle_root, extra_files=[bad])
    try:
        name, nb, lo, hi = CONFIGS[0]
        sc = _page(name, nb, lo, hi)
        valid = sweep.Template(sc, data_mod.generate(sc.data.main), tmp_path / "t.json", models_only=MODELS)
        with pytest.raises(RuntimeError, match="start-up"):       # a valid document (approval note A-3)
            sweep.run_batch(s, [valid.document(data_mod.generate(sc.data.main))], READ)
    finally:
        s.close()


# ------------------------------------------------------------------------------------------- filters: B1 sweep
def _b1_scenario(n, first_has_nan, b1_live=True):
    a_spec = "normal(0,1) + 10% NaN" if first_has_nan else "normal(0,1)"
    base = Scenario(
        id=f"families/b1_sweep_n{n}_{'first_nan' if first_has_nan else 'first_finite'}",
        data=Tables(TableSpec(n, 4000 + n, {"x": "uniform(0,1)", "y": "normal(0,1)", "a": a_spec, "b": "normal(0,1) + 10% NaN"})),
        page=Page(histograms=(hist("h2", ["x", "y"], [10, 2], [[0, 1], [-50, 50]]),), widgets=(range_("a"), range_("b"))),
        steps=(load(), set_("a", [-1, 1]), set_("b", [-1, 1])), checks=(auto(),),
        labels=Labels(family="known_defects", models=("RangeFilter", "LazyIntersectionFilter", "HistoNdCDS"),
                      verified=("RangeFilter", "LazyIntersectionFilter", "HistoNdCDS"), features=("FILTER.intersection",),
                      technique="B1 row-count sweep (refined rule)", priority="p1"))
    df = data_mod.generate(base.data.main)
    a, b = df.a.to_numpy(), df.b.to_numpy()
    if not (np.isfinite(a).any() and np.isfinite(b).any()):
        return None, df                                              # a widget needs at least one finite value
    lim = lambda v: (np.nanmin(v), np.nanmax(v))
    load_masks = [filt.range_mask(a, *lim(a)), filt.range_mask(b, *lim(b))]
    steps_now = [load_masks,
                 [filt.range_mask(a, -1, 1), load_masks[1]],
                 [filt.range_mask(a, -1, 1), filt.range_mask(b, -1, 1)]]
    b1_steps = tuple(k for k, now in enumerate(steps_now)
                     if b1_live and not np.array_equal(filt.b1_selection(now, load_masks), now[0] & now[1]))
    expect = []
    if b1_steps:
        expect.append(expected_failure("B1", "h2.bin_count", b1_steps, EqualsSelection(B1_RULE)))
        if 0 in b1_steps:
            expect.append(expected_failure("B1", LOAD_CHECK, (0,), EqualsSelection(B1_RULE)))
    from dataclasses import replace
    return replace(base, expect=tuple(expect)), df


ROWS = {"fast": (1, 2, 3, 31, 32, 33, 63, 64, 65, 200), "full": tuple(range(1, 201))}


@pytest.mark.parametrize("size", B1_SIZES)
@pytest.mark.parametrize("first_has_nan", [False, True], ids=["first_filter_finite", "first_filter_with_nan"])
@pytest.mark.parametrize("b1_live", CODES)
def test_b1_row_count_sweep(b1_live, first_has_nan, size, node_session, b1_back_session, headless_out):
    out = headless_out / "b1_sweep"
    bad, with_b1, without, skipped, trigger = [], [], [], [], []
    t0 = time.perf_counter()
    for n in ROWS[size]:
        sc, df = _b1_scenario(n, first_has_nan, b1_live)
        if sc is None:
            skipped.append(n)
            continue
        r = _filter_run(b1_live, node_session, b1_back_session, sc, out)
        (with_b1 if sc.expect else without).append(n)
        trigger.append((n, bool(sc.expect), bool(np.isnan(df.b.to_numpy()).any())))
        if not r.ok:
            bad.append((n, [f"{o.step}:{o.check}={o.outcome}" for o in r.outcomes if o.outcome not in ("PASS", "XFAIL")][:3]))
    dt = time.perf_counter() - t0
    (out / f"summary_{'b1_put_back' if b1_live else 'fixed'}_{'first_nan' if first_has_nan else 'first_finite'}_"
           f"{size}.json").write_text(json.dumps(
        {"with_b1": with_b1, "without_b1": without, "skipped_no_finite_value": skipped, "seconds": round(dt, 1)}))
    assert not bad, f"{len(bad)} row counts not as the refined rule predicts: {bad[:5]}"
    if not b1_live:
        assert not with_b1                                           # fixed code: no step expects B1
    elif not first_has_nan:
        # the trigger of the refined rule: with a finite first filter, B1 shows exactly when the row count is not a
        # multiple of 32 AND the second filter excludes a row at load (its missing values)
        wrong = [(n, shows) for n, shows, b_nan in trigger if shows != (n % 32 != 0 and b_nan)]
        assert not wrong, wrong


# ------------------------------------------------------------- filters: 2 to 5 filters, every move order (item E)
COLS = "abcde"


def _filters_scenario(n, k, order, first_has_nan=False, seed=None, b1_live=True):
    """k range filters on columns a.. ; the first without missing values unless first_has_nan, the others with 10 %;
    each filter moved once to [-1, 1] in the given order. Expected B1 steps from the refined rule."""
    cols = {"x": "uniform(0,1)", "y": "normal(0,1)"}
    for i, c in enumerate(COLS[:k]):
        cols[c] = "normal(0,1) + 10% NaN" if (i or first_has_nan) else "normal(0,1)"
    base = Scenario(
        id=f"families/filters_k{k}_n{n}_order_{''.join(COLS[i] for i in order)}",
        data=Tables(TableSpec(n, seed or (7000 + 10 * k + n), cols)),
        page=Page(histograms=(hist("h2", ["x", "y"], [10, 2], [[0, 1], [-50, 50]]),),
                  widgets=tuple(range_(c) for c in COLS[:k])),
        steps=(load(),) + tuple(set_(COLS[i], [-1, 1]) for i in order), checks=(auto(),),
        labels=Labels(family="known_defects", models=("RangeFilter", "LazyIntersectionFilter", "HistoNdCDS"),
                      verified=("RangeFilter", "LazyIntersectionFilter", "HistoNdCDS"), features=("FILTER.intersection",),
                      technique="filters in every move order (refined B1 rule)", priority="p1"))
    df = data_mod.generate(base.data.main)
    vals = [df[c].to_numpy() for c in COLS[:k]]
    if not all(np.isfinite(v).any() for v in vals):
        return None
    load_masks = [filt.range_mask(v, np.nanmin(v), np.nanmax(v)) for v in vals]
    now, steps_now = list(load_masks), [list(load_masks)]
    for i in order:
        now[i] = filt.range_mask(vals[i], -1, 1)
        steps_now.append(list(now))
    correct = lambda masks: np.logical_and.reduce(masks)
    b1_steps = tuple(s for s, m in enumerate(steps_now)
                     if b1_live and not np.array_equal(filt.b1_selection(m, load_masks), correct(m)))
    expect = []
    if b1_steps:
        expect.append(expected_failure("B1", "h2.bin_count", b1_steps, EqualsSelection(B1_RULE)))
        if 0 in b1_steps:
            expect.append(expected_failure("B1", LOAD_CHECK, (0,), EqualsSelection(B1_RULE)))
    from dataclasses import replace
    return replace(base, expect=tuple(expect))


@pytest.mark.parametrize("k", [pytest.param(2, id="2_filters", marks=pytest.mark.node_filters),
                               pytest.param(3, id="3_filters", marks=pytest.mark.node_filters),
                               pytest.param(4, id="4_filters", marks=[pytest.mark.slow, pytest.mark.node_filters_full]),
                               pytest.param(5, id="5_filters", marks=[pytest.mark.slow, pytest.mark.node_filters_full])])
@pytest.mark.parametrize("b1_live", CODES)
def test_filters_every_move_order(b1_live, k, node_session, b1_back_session, headless_out):
    import itertools
    out, bad, counts = headless_out / "filter_orders", [], {"with_b1": 0, "without_b1": 0}
    for n in (32, 33):                                   # both triggers of the refined rule
        for order in itertools.permutations(range(k)):
            sc = _filters_scenario(n, k, order, b1_live=b1_live)
            if sc is None:
                continue
            r = _filter_run(b1_live, node_session, b1_back_session, sc, out)
            counts["with_b1" if sc.expect else "without_b1"] += 1
            if not r.ok:
                bad.append((n, order, [f"{o.step}:{o.check}={o.outcome}" for o in r.outcomes if o.outcome not in ("PASS", "XFAIL")][:3]))
    (out / f"summary_k{k}_{'b1_put_back' if b1_live else 'fixed'}.json").write_text(json.dumps(counts))
    assert not bad, f"{len(bad)} move orders not as the refined rule predicts: {bad[:3]}"
    assert counts["with_b1"] > 0 if b1_live else counts["with_b1"] == 0, counts


@pytest.mark.node_filters
@pytest.mark.parametrize("b1_live", CODES)
def test_b1_twenty_thousand_rows_boundary(b1_live, node_session, b1_back_session, headless_out):
    """Claude3's pair: first filter without missing values — 20 000 rows (a multiple of 32): no B1; 20 001: B1.
    PHASE_5 Batch 1: on the fixed code both select the intersection; with B1 put back, the pair behaves as before."""
    out = headless_out / "filter_orders"
    results = {}
    for n in (20_000, 20_001):
        sc = _filters_scenario(n, 2, (0, 1), seed=47, b1_live=b1_live)
        r = _filter_run(b1_live, node_session, b1_back_session, sc, out)
        results[n] = (bool(sc.expect), r.ok)
    want = {20_000: (False, True), 20_001: (True, True)} if b1_live else {20_000: (False, True), 20_001: (False, True)}
    assert results == want, results


@pytest.mark.node_histograms
@pytest.mark.parametrize("config", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_models_only_document_computes_what_the_full_page_computes(config, node_session, tmp_path):
    """Review of 4.2c-2b (Claude10, GPT23, GPT24): the family reads documents holding only the histogram models; the
    same tables read from the FULL page must give identical counts in every bin (1-D, 2-D, weighted), including the
    B10 values (one rounding step below the top edge, with y in a middle and in the last bin)."""
    name, nb, lo, hi = config
    sc = _page(name, nb, lo, hi)
    df0 = data_mod.generate(sc.data.main)
    full = sweep.Template(sc, df0, tmp_path / "full.json")
    reduced = sweep.Template(sc, df0, tmp_path / "reduced.json", models_only=MODELS)
    named = _named_frames(nb, lo, hi)
    frames = _generated_frames(nb, lo, hi, 10) + [f for f in named if f.x[0] == np.nextafter(hi, -np.inf)]
    a, _ = sweep.run_batch(node_session, sweep.substitute_all(full, frames)[0], READ)
    b, _ = sweep.run_batch(node_session, sweep.substitute_all(reduced, frames)[0], READ)
    assert len(a) == len(b) == len(frames) >= 12
    differ = [i for i, (x, y) in enumerate(zip(a, b)) if json.dumps(x) != json.dumps(y)]
    assert not differ, f"tables whose counts differ between the full page and the reduced document: {differ}"
