"""The first Node families (proposal v0.3 §7 items D, E, H, J; stage 4.2c-2a).

Histograms (widget-free pages, mass cases by template substitution — erratum V-1):
  - per configuration, single-value NAMED cases at every bin edge and its rounding neighbours, the range limits,
    +-Infinity and far values, and GENERATED cases from the edge-value column kind;
  - each case: the page (1-D, N-D, weighted 1-D) must equal the implementation oracle in every cell (else FAIL);
    the RULE check "every finite value inside [lo, hi] is counted in a bin of that range" exposes what the oracle,
    which copies RootInteractive's arithmetic, cannot: its differences must be exactly the B10 rows (a value one
    rounding step below the top edge), recorded as the B10 candidate; any other difference is a FAIL;
  - the gate of V-1: substituted documents equal fresh Python builds (sample); isolation: shuffled order and a repeated
    first case give identical results; timings for V-2 and D-21 are written to timings.json.
Filters (document cases): known defect B1 swept over row counts with the refined rule (b1_first_changed): the first
  filter with or without missing values; every expected B1 step is computed from the rule, every other step must PASS.
"""
import json
import random
import time

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import data as data_mod
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
GENERATED_PER_CONFIG = 3400                     # 3 x 3400 = 10 200 generated cases (proposal item E: >= 10 000)
READ = """() => [window.__riH.col('h1', 'bin_count', ['HistogramCDS']), window.__riH.col('h2', 'bin_count', ['HistoNdCDS']),
                 window.__riH.col('hw', 'bin_count', ['HistogramCDS'])]"""
B1_RULE = "b1_later_filters_keep_rows_excluded_at_load"


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
    return [pd.DataFrame({"x": [v], "y": [0.5], "w": [1.5]}) for v in values]


def _generated_frames(nb, lo, hi, count):
    out = []
    for seed in range(count):
        n = 50 + (seed * 37) % 251
        out.append(data_mod.generate(TableSpec(n, 10_000 + seed, {"x": f"edges({lo},{hi},{nb})", "y": "edges(0,1,4)",
                                                                  "w": "uniform(0.5,2)"})))
    return out


def _judge(df, page, nb, lo, hi):
    """(page = implementation in every cell?, rule differences all B10?, B10 rows in 1-D, B10 rows in N-D).
    Rule (RB-2 as a statement, not as arithmetic): every finite value in [lo, hi] is counted in a bin of that range.
    Which of two neighbouring bins an interior-edge value lands in is arithmetic, not the rule."""
    x, y, w = (df[c].to_numpy() for c in ("x", "y", "w"))
    impl1 = ho.counts([x], [nb], [(lo, hi)])
    impl2 = ho.counts([x, y], [nb, 4], [(lo, hi), (0.0, 1.0)])
    implw = ho.counts([x], [nb], [(lo, hi)], weights=w)
    p1, p2, pw = (np.asarray(p, dtype=np.float64) for p in page)
    impl_ok = (np.array_equal(p1, impl1) and np.array_equal(p2, impl2)
               and np.allclose(pw, implw, rtol=1e-12, atol=1e-9))
    with np.errstate(invalid="ignore"):
        inside = np.isfinite(x) & (x >= lo) & (x <= hi)
        scale = nb / (hi - lo)
        raw = np.zeros_like(x)
        raw[inside] = x[inside] * scale + (-lo * scale)                # RootInteractive's 1-D arithmetic
        dropped = inside & (ho.bin_index_1d(x, nb, lo, hi) == -1)       # the rule is violated for these rows
        b10_1d = dropped & (x < hi) & (np.trunc(raw) == nb)             # ... B10: one rounding step below the top
        y_in = np.isfinite(y) & (y >= 0.0) & (y <= 1.0)
        raw_nd = np.zeros_like(x)
        raw_nd[inside] = x[inside] * scale - lo * scale                 # the N-D arithmetic of axis x
        b10_nd = inside & y_in & (x < hi) & (np.trunc(raw_nd) == nb)    # counted in the neighbouring cell or dropped
    return impl_ok, bool((dropped == b10_1d).all()), int(b10_1d.sum()), int(b10_nd.sum())


@pytest.mark.parametrize("config", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_histogram_family(config, node_session, headless_out):
    """One test per configuration (parallel workers can run them side by side)."""
    name, nb, lo, hi = config
    outdir = headless_out / f"families__histograms_node_{name}"
    outdir.mkdir(parents=True, exist_ok=True)
    res = ScenarioResult(id=f"families/histograms_node_{name} [node]", outdir=str(outdir),
                         versions=dict(node_session.versions), reproduce=f"pytest {__file__} -k {name}")
    res.backend = "node"
    timings, failures = {}, []
    sc = _page(name, nb, lo, hi)
    t0 = time.perf_counter()
    tpl = sweep.Template(sc, data_mod.generate(sc.data.main), outdir / "template_document.json")
    timings["template_build_s"] = round(time.perf_counter() - t0, 3)
    sample = _generated_frames(nb, lo, hi, 12)
    eq = sum(tpl.equivalent(df, outdir / "fresh_document.json") for df in sample)
    res.add(0, name, "template equals a fresh build (erratum V-1 gate)", "PASS" if eq == len(sample) else "FAIL",
            f"{eq} of {len(sample)} substituted documents equal a fresh Python build (ids, order, toolbar order ignored)")
    for kind, frames in (("named", _named_frames(nb, lo, hi)), ("generated", _generated_frames(nb, lo, hi, GENERATED_PER_CONFIG))):
        docs, t_sub = sweep.substitute_all(tpl, frames)
        pages, t = sweep.run_batch(node_session, docs, READ)
        judged = [_judge(df, p, nb, lo, hi) for df, p in zip(frames, pages)]
        bad_impl = [i for i, j in enumerate(judged) if not j[0]]
        bad_rule = [i for i, j in enumerate(judged) if not j[1]]
        b10_cases = sum(1 for j in judged if j[2] or j[3])
        b10_rows, b10_nd = sum(j[2] for j in judged), sum(j[3] for j in judged)
        failures += [(kind, i) for i in bad_impl + bad_rule]
        res.add(0, name, f"{kind} cases: page = implementation oracle (1-D, N-D, weighted)",
                "PASS" if not bad_impl else "FAIL", f"{len(frames) - len(bad_impl)} of {len(frames)} cases equal in every "
                f"cell" + (f"; first failing case {bad_impl[0]}" if bad_impl else ""), feature="HIST.histogram_1d")
        res.add(0, name, f"{kind} cases: rule check, every in-range value counted (B10 candidate)",
                "FAIL" if bad_rule else ("XFAIL" if (b10_rows or b10_nd) else "PASS"),
                f"every in-range value counted except the B10 rows: 1-D {b10_rows} rows dropped, N-D {b10_nd} rows in the "
                f"neighbouring cell, in {b10_cases} of {len(frames)} cases"
                + (f"; unexplained difference in case {bad_rule[0]}" if bad_rule else ""), feature="HIST.histogram_1d")
        timings[kind] = {"cases": len(frames), "substitution_s": round(t_sub, 3), "node_s": round(t["node_s"], 3),
                         "ms_per_case": round(1000 * (t_sub + t["node_s"]) / len(frames), 3)}
    frames = _generated_frames(nb, lo, hi, 200)                          # isolation
    docs, _ = sweep.substitute_all(tpl, frames)
    order = list(range(len(docs)))
    random.Random(nb).shuffle(order)
    straight, _ = sweep.run_batch(node_session, docs + [docs[0]], READ)
    shuffled, _ = sweep.run_batch(node_session, [docs[i] for i in order], READ)
    same = all(json.dumps(shuffled[k]) == json.dumps(straight[i]) for k, i in enumerate(order))
    res.add(0, name, "isolation: shuffled order and a repeated first case",
            "PASS" if same and json.dumps(straight[-1]) == json.dumps(straight[0]) else "FAIL",
            "200 cases equal in both orders; the first case repeated at the end equal")
    (outdir / "timings.json").write_text(json.dumps(timings, indent=1))
    res.write()
    assert res.ok and not failures, res.text() + f"\nfailures (kind, case): {failures[:10]}"
    assert sum(v["cases"] for v in timings.values() if isinstance(v, dict)) >= 3400 + 20


# ------------------------------------------------------------------------------------------- filters: B1 sweep
def _b1_scenario(n, first_has_nan):
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
                     if not np.array_equal(filt.b1_selection(now, load_masks), now[0] & now[1]))
    expect = []
    if b1_steps:
        expect.append(expected_failure("B1", "h2.bin_count", b1_steps, EqualsSelection(B1_RULE)))
        if 0 in b1_steps:
            expect.append(expected_failure("B1", LOAD_CHECK, (0,), EqualsSelection(B1_RULE)))
    from dataclasses import replace
    return replace(base, expect=tuple(expect)), df


@pytest.mark.parametrize("first_has_nan", [False, True], ids=["first_filter_finite", "first_filter_with_nan"])
def test_b1_row_count_sweep(first_has_nan, node_session, headless_out):
    out = headless_out / "b1_sweep"
    bad, with_b1, without, skipped, trigger = [], [], [], [], []
    t0 = time.perf_counter()
    for n in range(1, 201):
        sc, df = _b1_scenario(n, first_has_nan)
        if sc is None:
            skipped.append(n)
            continue
        r = run(node_session, sc, out, test_ref=__file__)
        (with_b1 if sc.expect else without).append(n)
        trigger.append((n, bool(sc.expect), bool(np.isnan(df.b.to_numpy()).any())))
        if not r.ok:
            bad.append((n, [f"{o.step}:{o.check}={o.outcome}" for o in r.outcomes if o.outcome not in ("PASS", "XFAIL")][:3]))
    dt = time.perf_counter() - t0
    (out / f"summary_{'first_nan' if first_has_nan else 'first_finite'}.json").write_text(json.dumps(
        {"with_b1": with_b1, "without_b1": without, "skipped_no_finite_value": skipped, "seconds": round(dt, 1)}))
    assert not bad, f"{len(bad)} row counts not as the refined rule predicts: {bad[:5]}"
    if not first_has_nan:
        # the trigger of the refined rule: with a finite first filter, B1 shows exactly when the row count is not a
        # multiple of 32 AND the second filter excludes a row at load (its missing values)
        wrong = [(n, shows) for n, shows, b_nan in trigger if shows != (n % 32 != 0 and b_nan)]
        assert not wrong, wrong
