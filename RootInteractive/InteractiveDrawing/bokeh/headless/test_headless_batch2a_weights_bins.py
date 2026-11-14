"""PHASE_5 Batch 2A — binning, Boolean transport, integers and basic weighted moments, in Node. Tests only
(Batch 2A CRR v0.1, Claude20.RI, 2026-11-13).

Plain words. Batch 2A of the PHASE_5 bug fixes repairs what a compiled ADF page needs before any quantile question
(that is Batch 2B, after dfdraw's estimator appendix F1): histogram bins, the transport of Boolean and integer columns,
and weighted counts, means and standard deviations. These tests come before the fix (gate G5a of the audit errata
v0.3a). They use the shapes the ADF/dfdraw compiler will emit and compare every number with numpy or pandas. Every
defect shows RED on unfixed code with its exact signature. No production code is changed.

Defects (register PHASE_5_0_Defect_Register_v0_1):
  B8     a 1-D histogram counts NaN in bin 0 (HistogramCDS.ts 220-227). For the compiler: every orphan row points at the
         NaN fill row of its related table, so every orphan lands in bin 0 of a 1-D histogram of a gathered column.
         The N-D histogram has a finite guard and is right (the healthy control).
  B10    a value one rounding step below a histogram's upper edge lands one bin too far: dropped in 1-D, moved to bin 0
         of the next group in 2-D (HistoNdCDS.ts 345, HistogramCDS.ts 227). Common axes are hit: [0, 50] in 20 bins
         (T1's r axis), [-4, 4] in 8, [-2, 2] in 20.
  WT-D1  a Boolean column cannot be decoded in a compressed page ("Unknown dtype: bool"; SerializationUtils.ts).
         For the compiler: selection branches compiled to Boolean weights.
  B7     relative compression rounds integer columns (65537 -> 65536 at 16 bits; compressArray.py 31).
  TR-D3  NEW (found here): a 64-bit integer above 2^53 loses precision in the page even with lossless coding, because
         the page decodes it into a JavaScript double (2^53 + 1 arrives as 2^53). Keys and identifiers are hit.
  B4/WT-D2  a zero weight in the first row a profile visits in a bin makes that bin's mean and std NaN
         (HistoNdProfile.ts 190). Boolean branch weights put zeros everywhere.
  WT-D4  in a profile with several weights, every series after the first is computed with weights paired to the wrong
         rows (HistoNdProfile.ts 130). `weights_vector` and `selection_vector` compile to exactly this.

The PHASE_4 strict tests of these defects stay (test_headless_node_weights.py, test_headless_node_histograms.py,
test_headless_node_families.py, scenarios/known_defects.py). These add the compiler's shapes, independent signatures
for TR-D3 and B7 (no test before), and the RI-ADF-01 acceptance page (hist2d and profile2d), which passes today.

THE STRICT LOGIC is that of Batch 1: a strict test passes when the page equals the oracle; it raises KnownDefect only
when the page shows exactly the declared defects, each by its exact signature; any other deviation is an ordinary
failure. Quantiles are read but never judged here (Batch 2B, decision D-B7).

TEST CORRECTION v0.2 (GPT24.RI's review of the test commits, 2026-11-13, findings 2A-P1-01 and 2A-P1-02):
- the weight series are explained only by an exact port of the present code (oracles/profile_port.py): the page must
  equal it bit for bit - entries, mean, std and quantile of every series in every X bin - with the smallest set of
  WT-D2 and WT-D4 switched on. Any other wrong value (a mean of 1e12, a NaN std, another quantile, other pairings)
  is UNEXPLAINED;
- an integer column counts as refused only when the build raises the deliberate refusal UnsafeIntegerColumn that
  names the column and 2^53; any other exception is an ordinary failure. Integers within 2^53 must arrive exactly.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import histograms as ho
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import profile_port
from .test_headless_node_weights import (BOOL_ERROR, SPEC, XEDGE, _first_visited_weight, _is_wt_d1, _open, _page,
                                         _profile, _series, _weighted_ref)


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of registered defects."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


STRICT = dict(strict=True, raises=KnownDefect)
LOSSLESS = [("code", 0), ("zip", 0), ("base64", 0)]
RELATIVE16 = [("relative", 16), "code", "zip"]


def _node(*marks, target=pytest.mark.node_stats):
    """Node test marks: the target (node_stats for weights and profiles, node_histograms for bins, node_transport for
    column coding), backend, layer, then the given marks."""
    def wrap(test):
        for m in reversed([target, pytest.mark.backend("node"), pytest.mark.layer("integration")] + list(marks)):
            test = m(test)
        return test
    return wrap


def _unit(test):
    return pytest.mark.headless_unit(pytest.mark.backend("python")(pytest.mark.layer("unit")(test)))


def _build(tmp_path, df, figures, name, *, sources=(), histograms=(), compression=None):
    """A page through bokehDrawSA.fromArray (show suppressed) as a Bokeh document."""
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    shown = sa.show
    sa.show = lambda *a, **k: None
    try:
        root = sa.bokehDrawSA.fromArray(df, None, figures, [], sourceArray=list(sources) + list(histograms),
                                        aliasArray=[], parameterArray=[], layout=[list(range(len(figures)))],
                                        widgetLayout=[], arrayCompression=compression or [(".*", LOSSLESS)]).pAll
    finally:
        sa.show = shown
    doc = Document()
    doc.add_root(root)
    out = tmp_path / name
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


def _node_page(node_session, tmp_path, path):
    report = ScenarioResult(id=f"batch2a/{Path(path).stem} [node]", outdir=str(tmp_path))
    page = node_session.new_page(report)
    page.goto(path)
    page.wait_for_function(page.load_condition)
    return page, report


def _no_errors(report):
    assert not report.page_errors and not report.console_errors, (report.page_errors, report.console_errors)


# ===================================================================================== 1. weights in the compiler's shape
SEED = 20261113
N_W = 800


def weight_table() -> pd.DataFrame:
    """x, y in [0, 4); two selection branches as 0/1 indicators (b1, b2: side 0 or 1), two positive weights (wA, wB),
    and the branch x weight products of vector_compose='inner' (p1 = b1*wA, p2 = b2*wB), precomputed as the compiler
    would export them."""
    rng = np.random.default_rng(SEED)
    d = pd.DataFrame({"x": rng.uniform(0, 4, N_W), "y": rng.uniform(0, 4, N_W)})
    side = rng.integers(0, 2, N_W)
    d["b1"], d["b2"] = (side == 0).astype(float), (side == 1).astype(float)
    d["wA"], d["wB"] = 0.5 + rng.random(N_W), 0.25 + 2 * rng.random(N_W)
    d["p1"], d["p2"] = d.b1 * d.wA, d.b2 * d.wB
    return d


WEIGHT_PAGES = {
    # contract C04 weights_vector: two positive weights in one profile
    "weights_vector": ["wA", "wB"],
    # contract C03 selection_vector compiled to Boolean weights (as 0/1 numbers, the WT-D1 workaround)
    "selection_vector": ["b1", "b2"],
    # vector_compose='inner': each branch times its weight (RI-ADF-04 shape)
    "inner_compose": ["p1", "p2"],
}
EXPECTED_WEIGHT_DEFECTS = {"weights_vector": ["WT-D4"], "selection_vector": ["WT-D2", "WT-D4"],
                           "inner_compose": ["WT-D2", "WT-D4"]}
ALL_WEIGHTS = ["b1", "b2", "wA", "wB", "p1", "p2"]


def _close(a, b, keys=("entries", "mean", "std")) -> bool:
    return all(np.allclose(np.asarray(a[k], float), np.asarray(b[k], float), rtol=1e-12, atol=1e-12, equal_nan=True)
               for k in keys)


WT_SWITCHES = ("WT-D2", "WT-D4")
SERIES_KEYS = ("entries", "mean", "std", "quantile_0")


def port_series(d, names, defects) -> dict:
    """The present code's profile (oracles/profile_port.py) of the weight table, the series in the order `names`."""
    out = profile_port.profile([(d.x.to_numpy(float), 4, (0.0, 4.0))], d.y.to_numpy(float), (4, (0.0, 4.0)),
                               {s: d[s].to_numpy(float) for s in names}, quantiles=(0.5,), defects=defects)
    return {s: {k: np.asarray(out[s][k], float) for k in SERIES_KEYS} for s in names}


def _exact(a, b) -> bool:
    return all(len(a[k]) == len(b[k]) and np.array_equal(np.asarray(a[k], float), np.asarray(b[k], float), equal_nan=True)
               for k in SERIES_KEYS)


def classify_series(d, names, multi) -> tuple:
    """PASS: every series of the multi-weight profile equals numpy (entries, mean, std; 1e-12). KNOWN: the page equals,
    bit for bit, the port of the present code with the smallest set of WT-D2 and WT-D4 switched on - entries, mean,
    std and quantile of every series in every X bin. UNEXPLAINED: anything else."""
    intended = {s: _weighted_ref(d, s) for s in names}
    if set(multi) == set(names) and all(_close(multi[s], intended[s]) for s in names):
        return "PASS", []
    for size in range(1, len(WT_SWITCHES) + 1):
        for subset in itertools.combinations(WT_SWITCHES, size):
            port = port_series(d, names, subset)
            if set(multi) == set(names) and all(_exact(multi[s], port[s]) for s in names):
                return "KNOWN", sorted(subset)
    return "UNEXPLAINED", "neither numpy nor the present code's port with any set of WT-D2 and WT-D4"


def weight_series(node_session, tmp_path, names) -> tuple:
    """The page's profile with every weight of `names` (multi), and each weight in a page of its own (alone)."""
    d = weight_table()
    options = [None] + ALL_WEIGHTS
    _open(node_session, _page(tmp_path, d, names, options, suffix="multi_" + "_".join(names)))
    got = _profile(node_session)
    multi = {s: _series(got, s) for s in names}
    alone = {}
    for s in names:
        _open(node_session, _page(tmp_path, d, [s], options, suffix="alone_" + s))
        alone[s] = _series(_profile(node_session), s)
    return d, multi, alone


@_unit
def test_weight_table_reaches_both_defects():
    """Preconditions: with 0/1 branch weights some bins start with a zero-weight row (WT-D2 can act), with positive
    weights none does; an indicator-weighted mean equals the plain mean of the branch's rows (the equivalence of
    errata v0.3a section 3 for counts, mean and std, in the oracle itself)."""
    d = weight_table()
    slices = [d[(d.x >= XEDGE[i]) & (d.x < XEDGE[i + 1])] for i in range(4)]
    assert any(_first_visited_weight(s, "b1") == 0 or _first_visited_weight(s, "b2") == 0 for s in slices)
    assert all(_first_visited_weight(s, w) > 0 for s in slices for w in ("wA", "wB"))
    ref = _weighted_ref(d, "b1")
    plain = [d[(d.x >= XEDGE[i]) & (d.x < XEDGE[i + 1]) & (d.b1 == 1)].y for i in range(4)]
    np.testing.assert_allclose(ref["entries"], [len(p) for p in plain], rtol=0, atol=0)
    np.testing.assert_allclose(ref["mean"], [p.mean() for p in plain], rtol=1e-12)
    np.testing.assert_allclose(ref["std"], [p.std(ddof=0) for p in plain], rtol=1e-12)


@_node(pytest.mark.feature("HIST.stack"), pytest.mark.feature("HIST.profile_unbinned"),
       pytest.mark.xfail(**STRICT, reason="WT-D2 and WT-D4 as declared per page (EXPECTED_WEIGHT_DEFECTS)"))
@pytest.mark.parametrize("shape", sorted(WEIGHT_PAGES))
def test_strict_compiled_weight_series_each_equal_numpy(node_session, tmp_path, shape):
    """C03, C04, RI-ADF-04: every series of a multi-weight profile - entries (sum of weights), weighted mean and
    weighted population std per X bin - equals numpy, whatever series come before it."""
    names = WEIGHT_PAGES[shape]
    d, multi, _ = weight_series(node_session, tmp_path, names)
    verdict, detail = classify_series(d, names, multi)
    if verdict == "PASS":
        return
    if verdict == "KNOWN":
        known_defect(detail == EXPECTED_WEIGHT_DEFECTS[shape], "known defects " + ", ".join(detail))
        raise AssertionError(f"the page shows {detail}, declared {EXPECTED_WEIGHT_DEFECTS[shape]}")
    raise AssertionError(detail)


@_node(pytest.mark.feature("HIST.stack"))
def test_falsifier_swapped_weight_names_and_std_for_mean_are_unexplained(node_session, tmp_path):
    """Gate G4: the page's own results with two weight names swapped, or with the std reported as the mean, are
    neither PASS nor a known defect."""
    names = WEIGHT_PAGES["weights_vector"]
    d, multi, alone = weight_series(node_session, tmp_path, names)
    swapped = {names[0]: multi[names[1]], names[1]: multi[names[0]]}
    assert classify_series(d, names, swapped)[0] == "UNEXPLAINED"
    std_for_mean = {s: {**multi[s], "mean": multi[s]["std"]} for s in names}
    assert classify_series(d, names, std_for_mean)[0] == "UNEXPLAINED"
    intended = {s: {**_weighted_ref(d, s), "quantile_0": alone[s]["quantile_0"]} for s in names}
    assert classify_series(d, names, intended) == ("PASS", [])


@_unit
@pytest.mark.parametrize("shape", sorted(WEIGHT_PAGES))
def test_falsifier_a_later_series_changed_in_any_way_is_unexplained(shape):
    """2A-P1-01 (GPT24): the model of the present code itself, changed only in its later series - a mean of 1e12, a NaN
    std, a quantile moved to the next value of the bin, or the weights paired as if the series came in the other order
    - is UNEXPLAINED, never WT-D4. The unchanged model is KNOWN with exactly the declared defects."""
    d, names = weight_table(), WEIGHT_PAGES[shape]
    model = port_series(d, names, WT_SWITCHES)
    assert classify_series(d, names, model) == ("KNOWN", EXPECTED_WEIGHT_DEFECTS[shape])
    later = names[1]

    def changed(key, i, value):
        s = {k: v.copy() for k, v in model[later].items()}
        s[key][i] = value
        return {**model, later: s}
    q = model[later]["quantile_0"]
    shuffled = d.copy()                       # the later weight given to other rows of the same X bin
    rng = np.random.default_rng(SEED + 5)
    for i in range(4):
        rows = np.flatnonzero((d.x >= XEDGE[i]) & (d.x < XEDGE[i + 1]))
        shuffled.loc[rows, later] = d[later].to_numpy()[rng.permutation(rows)]
    mutants = {"mean 1e12": changed("mean", 1, 1e12), "NaN std": changed("std", 2, np.nan),
               "another quantile": changed("quantile_0", 0, float(np.nextafter(q[0], np.inf))),
               "other pairing": {**model, later: port_series(shuffled, names, WT_SWITCHES)[later]}}
    for name, page in mutants.items():
        assert classify_series(d, names, page)[0] == "UNEXPLAINED", name


@_unit
@pytest.mark.parametrize("shape", sorted(WEIGHT_PAGES))
def test_the_port_without_defects_is_numpy(shape):
    """The port of the present code with no defect switched on gives numpy's entries, means and stds (1e-12)."""
    d, names = weight_table(), WEIGHT_PAGES[shape]
    port = port_series(d, names, ())
    for s in names:
        assert _close(port[s], _weighted_ref(d, s)), s


@_node(pytest.mark.feature("HIST.stack"))
def test_control_a_single_positive_weight_equals_numpy(node_session, tmp_path):
    """Healthy control: a positive weight alone (no zero weights, no series before it) equals numpy."""
    d, _, alone = weight_series(node_session, tmp_path, ["wA"])
    assert _close(alone["wA"], _weighted_ref(d, "wA"))


# ===================================================================================== 2. Boolean branches (WT-D1)
@_node(pytest.mark.feature("ENC.cds_compress.page_decode"),
       pytest.mark.xfail(**STRICT, reason="WT-D1: a Boolean branch weight cannot be decoded in a compressed page"),
       target=pytest.mark.node_transport)
@pytest.mark.parametrize("compression", ["lossless", "relative16"])
def test_strict_boolean_branch_weights_are_read_and_equal_numpy(node_session, tmp_path, compression):
    """C03 as the compiler would first emit it: two selection branches as Boolean columns in one profile. Judged on the
    transport alone - the branch columns arrive as their values, and each series' entries are the branch's row count -
    so that WT-D2 and WT-D4 (judged above) cannot decide this test."""
    d = weight_table()
    d["c1"], d["c2"] = d.b1.astype(bool), d.b2.astype(bool)
    _open(node_session, _page(tmp_path, d, ["c1", "c2"], [None, "c1", "c2"], compression=compression,
                              suffix=f"bool_{compression}"))
    r = node_session.request("eval", code=SPEC, arg="c1")
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    spec = decode(r["result"]) or {}
    read = _profile(node_session)
    known_defect(_is_wt_d1(read, spec, compression), f"WT-D1: {BOOL_ERROR} ({compression})")
    for s, ref in (("c1", "b1"), ("c2", "b2")):
        np.testing.assert_allclose(_series(read, s)["entries"], _weighted_ref(d, ref)["entries"], rtol=0, atol=0)


# ===================================================================================== 3. orphans at the fill row (B8)
N_REF, N_TRACK = 500, 60


def orphan_tables():
    """References pointing at tracks by row number; 10 % orphans point at the track table's NaN fill row."""
    rng = np.random.default_rng(SEED + 2)
    tracks = {"p": np.append(rng.uniform(0.5, 9.5, N_TRACK), np.nan)}
    idx = rng.integers(0, N_TRACK, N_REF)
    idx[rng.uniform(size=N_REF) < 0.1] = N_TRACK
    refs = pd.DataFrame({"idx": idx.astype(np.int64), "x": rng.uniform(0, 1, N_REF), "w": 0.5 + rng.random(N_REF)})
    return refs, tracks


ORPHAN_HISTOGRAMS = [{"name": "h1", "variables": ["tracks.p[idx]"], "nbins": 10, "range": [0, 10]},
                     {"name": "h1w", "variables": ["tracks.p[idx]"], "nbins": 10, "range": [0, 10], "weights": "w"},
                     {"name": "h2", "variables": ["tracks.p[idx]", "x"], "nbins": [10, 2], "range": [[0, 10], [0, 1]]}]


def orphan_state(node_session, tmp_path):
    refs, tracks = orphan_tables()
    figures = [[["bin_center"], ["bin_count"], {"source": "h1"}], [["bin_center"], ["bin_count"], {"source": "h1w"}],
               [["bin_center_0"], ["bin_count"], {"source": "h2"}]]
    page, report = _node_page(node_session, tmp_path, _build(tmp_path, refs, figures, "orphans.json",
                                                              sources=[{"name": "tracks", "data": tracks}],
                                                              histograms=ORPHAN_HISTOGRAMS))
    state = page.evaluate("""() => { const H = window.__riH, c = (n, t) => Array.from(H.model(n, [t]).get_column('bin_count'));
      return {h1: c('h1', 'HistogramCDS'), h1w: c('h1w', 'HistogramCDS'), h2: c('h2', 'HistoNdCDS')}; }""")
    _no_errors(report)
    return refs, tracks, state


def orphan_oracle(refs, tracks) -> dict:
    p = tracks["p"][refs.idx.to_numpy()]
    return {"h1": ho.counts([p], [10], [(0, 10)]).tolist(),
            "h1w": ho.counts([p], [10], [(0, 10)], weights=refs.w).tolist(),
            "h2": ho.counts([p, refs.x], [10, 2], [(0, 10), (0, 1)]).tolist()}


def _is_b8(state, refs, tracks) -> bool:
    """B8, exact: bin 0 of each 1-D histogram exceeds numpy by the orphans (their count; their summed weight), every
    other bin equals numpy, and the 2-D histogram equals numpy."""
    want, orphan = orphan_oracle(refs, tracks), (refs.idx == N_TRACK).to_numpy()
    for key, excess in (("h1", float(orphan.sum())), ("h1w", float(refs.w[orphan].sum()))):
        diff = np.asarray(state[key], float) - np.asarray(want[key], float)
        if not (excess > 0 and np.isclose(diff[0], excess, rtol=1e-12) and np.allclose(diff[1:], 0, atol=1e-9)):
            return False
    return state["h2"] == want["h2"]


@_node(pytest.mark.feature("HIST.histogram_1d"), pytest.mark.feature("DSL.gather_operation"), target=pytest.mark.node_histograms)
def test_signature_b8_every_orphan_lands_in_bin_0_of_a_1d_histogram(node_session, tmp_path):
    """B8, exact signature, in the compiler's shape: 1-D histograms of a gathered column, plain and weighted."""
    refs, tracks, state = orphan_state(node_session, tmp_path)
    assert _is_b8(state, refs, tracks)


@_node(pytest.mark.feature("HIST.histogram_nd"), pytest.mark.feature("DSL.gather_operation"), target=pytest.mark.node_histograms)
def test_control_b8_the_2d_histogram_skips_the_orphans(node_session, tmp_path):
    """B8's control (errata C-06): the N-D histogram's finite guard excludes every orphan."""
    refs, tracks, state = orphan_state(node_session, tmp_path)
    assert state["h2"] == orphan_oracle(refs, tracks)["h2"]


@_node(pytest.mark.feature("HIST.histogram_1d"), pytest.mark.feature("DSL.gather_operation"),
       pytest.mark.xfail(**STRICT, reason="B8: NaN (every orphan) is counted in bin 0 of a 1-D histogram"), target=pytest.mark.node_histograms)
def test_strict_orphans_are_not_counted_in_a_1d_histogram(node_session, tmp_path):
    refs, tracks, state = orphan_state(node_session, tmp_path)
    known_defect(_is_b8(state, refs, tracks), "B8: the orphans are in bin 0")
    want = orphan_oracle(refs, tracks)
    assert state["h1"] == want["h1"]
    np.testing.assert_allclose(state["h1w"], want["h1w"], rtol=1e-12)


# ===================================================================================== 4. the top edge of a group axis (B10)
TOP = float(np.nextafter(50.0, -np.inf))              # one rounding step below 50: B10 on [0, 50] with 20 bins


def edge_table(with_edge_rows=True) -> pd.DataFrame:
    """r in [0, 50) (T1's r axis, 20 bins), a group index g in 0..4; with_edge_rows: two rows at r = TOP per group."""
    rng = np.random.default_rng(SEED + 3)
    d = pd.DataFrame({"r": rng.uniform(0, 49, 300), "g": rng.integers(0, 5, 300).astype(float)})
    if with_edge_rows:
        d = pd.concat([d, pd.DataFrame({"r": [TOP] * 10, "g": [float(g) for g in range(5)] * 2})], ignore_index=True)
    return d


EDGE_HISTOGRAMS = [{"name": "e1", "variables": ["r"], "nbins": 20, "range": [0, 50]},
                   {"name": "e2", "variables": ["r", "g"], "nbins": [20, 5], "range": [[0, 50], [0, 5]]}]


def intended_counts(columns, nbins, ranges) -> list:
    """A finite value inside [lo, hi] is counted once, in the bin that contains it; the top edge belongs to the last
    bin (rulebook RB-2.3 as a rule): RootInteractive's arithmetic, clamped to the last bin."""
    n = len(columns[0])
    flat, stride, keep = np.zeros(n, dtype=np.int64), 1, np.ones(n, dtype=bool)
    for col, nb, (lo, hi) in zip(columns, nbins, ranges):
        v = np.asarray(col, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            inside = np.isfinite(v) & (v >= lo) & (v <= hi)
        b = np.zeros(n, dtype=np.int64)
        b[inside] = np.minimum(np.trunc(v[inside] * (nb / (hi - lo)) - lo * (nb / (hi - lo))), nb - 1)
        flat += b * stride
        stride *= nb
        keep &= inside
    return np.bincount(flat[keep], minlength=stride).astype(float).tolist()


def edge_state(node_session, tmp_path, with_edge_rows=True):
    d = edge_table(with_edge_rows)
    figures = [[["bin_center"], ["bin_count"], {"source": "e1"}], [["bin_center_0"], ["bin_count"], {"source": "e2"}]]
    page, report = _node_page(node_session, tmp_path, _build(tmp_path, d, figures, f"edge_{with_edge_rows}.json",
                                                              histograms=EDGE_HISTOGRAMS))
    state = page.evaluate("""() => { const H = window.__riH, c = (n, t) => Array.from(H.model(n, [t]).get_column('bin_count'));
      return {e1: c('e1', 'HistogramCDS'), e2: c('e2', 'HistoNdCDS')}; }""")
    _no_errors(report)
    return d, state


def _is_b10(d, state) -> bool:
    """B10, exact: the page equals RootInteractive's unclamped arithmetic (rulebook oracle RB-2): the edge rows are
    dropped in 1-D and moved to bin 0 of the next group in 2-D (dropped in the last group); not the intended counts."""
    model = {"e1": ho.counts([d.r], [20], [(0, 50)]).tolist(),
             "e2": ho.counts([d.r, d.g], [20, 5], [(0, 50), (0, 5)]).tolist()}
    want = {"e1": intended_counts([d.r], [20], [(0, 50)]), "e2": intended_counts([d.r, d.g], [20, 5], [(0, 50), (0, 5)])}
    return state == model and model != want


@_unit
def test_edge_rows_are_b10_rows_and_the_intended_rule_counts_them():
    d = edge_table()
    edge = d.r == TOP
    assert int(np.trunc(TOP * (20 / 50))) == 20                         # the B10 arithmetic
    assert sum(intended_counts([d.r], [20], [(0, 50)])) == len(d)
    assert sum(ho.counts([d.r], [20], [(0, 50)])) == len(d) - edge.sum()
    clean = edge_table(False)
    assert intended_counts([clean.r, clean.g], [20, 5], [(0, 50), (0, 5)]) == \
        ho.counts([clean.r, clean.g], [20, 5], [(0, 50), (0, 5)]).tolist()


@_node(pytest.mark.feature("HIST.histogram_1d"), pytest.mark.feature("HIST.histogram_nd"), target=pytest.mark.node_histograms)
def test_signature_b10_edge_rows_are_dropped_or_moved_to_the_next_group(node_session, tmp_path):
    d, state = edge_state(node_session, tmp_path)
    assert _is_b10(d, state)


@_node(pytest.mark.feature("HIST.histogram_1d"), pytest.mark.feature("HIST.histogram_nd"), target=pytest.mark.node_histograms)
def test_control_b10_without_edge_rows_every_group_is_right(node_session, tmp_path):
    d, state = edge_state(node_session, tmp_path, with_edge_rows=False)
    assert state == {"e1": intended_counts([d.r], [20], [(0, 50)]),
                     "e2": intended_counts([d.r, d.g], [20, 5], [(0, 50), (0, 5)])}


@_node(pytest.mark.feature("HIST.histogram_1d"), pytest.mark.feature("HIST.histogram_nd"),
       pytest.mark.xfail(**STRICT, reason="B10: a value just below the top edge lands one bin too far"), target=pytest.mark.node_histograms)
def test_strict_edge_rows_are_counted_in_the_last_bin_of_their_group(node_session, tmp_path):
    """G02/F01: a group or facet axis must keep every row in its own group."""
    d, state = edge_state(node_session, tmp_path)
    known_defect(_is_b10(d, state), "B10: edge rows dropped or moved to the next group")
    assert state == {"e1": intended_counts([d.r], [20], [(0, 50)]),
                     "e2": intended_counts([d.r, d.g], [20, 5], [(0, 50), (0, 5)])}


# ===================================================================================== 5. integers: B7 and TR-D3
KEYS_SMALL = np.array([65537, 70001, 1000003, -5, 3, 2 ** 31 + 1], dtype=np.int64)      # B7: rounded by relative16
KEYS_BIG = np.array([2 ** 53 + 1, 2 ** 60 + 7, -(2 ** 55) - 3, 12345], dtype=np.int64)   # TR-D3: above 2^53
INTS_JS = """() => { const v = window.__riH.model('cdsOrig', ['CDSCompress']).get_column('key');
  return {type: v.constructor.name,
          values: Array.from(v, x => typeof x === 'bigint' ? x.toString()
                                     : (Number.isInteger(x) ? BigInt(x).toString() : 'not an integer: ' + x))}; }"""


REFUSAL = "UnsafeIntegerColumn"       # the deliberate build-time refusal of the 2A fix (RootInteractive/Tools/compressArray.py)


def is_refusal(exc: BaseException, column: str) -> bool:
    """2A-P1-02 (GPT24): only the deliberate refusal counts - the exception class UnsafeIntegerColumn (a ValueError),
    naming the column and the 2^53 limit. Any other exception at build time is an ordinary failure."""
    text = str(exc)
    return type(exc).__name__ == REFUSAL and isinstance(exc, ValueError) and repr(column) in text and "2^53" in text


def int_state(node_session, tmp_path, keys, compression):
    """None and the refusal's text if the build refuses the column deliberately; else the page's decoded key column
    as exact integers. Any other build error propagates (an ordinary failure)."""
    df = pd.DataFrame({"key": keys, "x": np.arange(len(keys), dtype=float)})
    try:
        path = _build(tmp_path, df, [[["x"], ["key"]]], f"ints_{compression}_{len(keys)}.json",
                      compression=[(".*", RELATIVE16 if compression == "relative16" else LOSSLESS)])
    except ValueError as exc:
        if is_refusal(exc, "key"):
            return None, f"{type(exc).__name__}: {exc}"
        raise
    page, report = _node_page(node_session, tmp_path, path)
    got = page.evaluate(INTS_JS)
    _no_errors(report)
    return [int(v) if not v.startswith("not") else v for v in got["values"]], got["type"]


def _is_b7(values) -> bool:
    from RootInteractive.Tools.compressArray import roundRelativeBinary
    rounded = [int(v) for v in roundRelativeBinary(KEYS_SMALL.astype(np.float64), 16)]
    return values == rounded and values != KEYS_SMALL.tolist()


def _is_tr_d3(values) -> bool:
    return values == [int(float(k)) for k in KEYS_BIG] and values != KEYS_BIG.tolist()


@_node(pytest.mark.feature("ENC.cds_compress.page_decode"), target=pytest.mark.node_transport)
def test_signature_b7_relative16_rounds_integer_keys(node_session, tmp_path):
    """B7, exact signature (no test before): 65537 -> 65536, 70001 -> 70000, ...; -5 and 3 survive."""
    values, kind = int_state(node_session, tmp_path, KEYS_SMALL, "relative16")
    assert values is not None and _is_b7(values), (values, kind)


@_node(pytest.mark.feature("ENC.cds_compress.page_decode"), target=pytest.mark.node_transport)
def test_control_integer_keys_below_2_53_are_exact_when_lossless(node_session, tmp_path):
    values, kind = int_state(node_session, tmp_path, KEYS_SMALL, "lossless")
    assert values == KEYS_SMALL.tolist(), (values, kind)


@_node(pytest.mark.feature("ENC.cds_compress.page_decode"),
       pytest.mark.xfail(**STRICT, reason="B7: relative compression rounds integer columns"), target=pytest.mark.node_transport)
def test_strict_integer_keys_survive_relative16_or_are_refused(node_session, tmp_path):
    """An integer key within 2^53 arrives exactly, also when the page uses relative 16-bit coding (the Architect's
    default of 2026-11-13: integer columns skip relative coding). A refusal is not expected for these keys."""
    values, kind = int_state(node_session, tmp_path, KEYS_SMALL, "relative16")
    assert values is not None, f"keys within 2^53 refused: {kind}"
    known_defect(_is_b7(values), "B7: integer keys rounded")
    assert values == KEYS_SMALL.tolist(), (values, kind)


@_node(pytest.mark.feature("ENC.cds_compress.page_decode"), target=pytest.mark.node_transport)
def test_signature_tr_d3_keys_above_2_53_lose_precision_even_lossless(node_session, tmp_path):
    """TR-D3, exact signature: every key arrives as the nearest double (2^53 + 1 -> 2^53), in a Float64Array."""
    values, kind = int_state(node_session, tmp_path, KEYS_BIG, "lossless")
    assert values is not None and _is_tr_d3(values) and kind == "Float64Array", (values, kind)


@_unit
def test_falsifier_only_the_deliberate_refusal_counts():
    """2A-P1-02 (GPT24): an unrelated ValueError, TypeError or KeyError at build time, or the refusal naming another
    column, is not a refusal of the key column."""
    class UnsafeIntegerColumn(ValueError):
        pass
    good = UnsafeIntegerColumn("column 'key': 64-bit integers beyond 2^53 cannot be represented exactly in the browser")
    assert is_refusal(good, "key")
    for bad in (ValueError(str(good)), TypeError(str(good)), KeyError("key"),
                UnsafeIntegerColumn("column 'other': 64-bit integers beyond 2^53"), UnsafeIntegerColumn("column 'key'")):
        assert not is_refusal(bad, "key"), repr(bad)


@_node(pytest.mark.feature("ENC.cds_compress.page_decode"),
       pytest.mark.xfail(**STRICT, reason="TR-D3: 64-bit integers above 2^53 lose precision in the page"), target=pytest.mark.node_transport)
def test_strict_keys_above_2_53_are_exact_or_refused(node_session, tmp_path):
    values, kind = int_state(node_session, tmp_path, KEYS_BIG, "lossless")
    if values is None:
        return
    known_defect(_is_tr_d3(values), "TR-D3: keys above 2^53 rounded to doubles")
    assert values == KEYS_BIG.tolist(), (values, kind)


# ===================================================================================== 6. RI-ADF-01: hist2d and profile2d
def adf01_table() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 4)
    x, y = rng.uniform(0, 4, 2000), rng.uniform(0, 3, 2000)
    return pd.DataFrame({"x": x, "y": y, "z": np.clip(0.5 * x - 0.3 * y + rng.normal(0, 0.8, 2000), -2.9, 2.9)})


ADF01_HISTOGRAMS = [{"name": "hxy", "variables": ["x", "y"], "nbins": [4, 3], "range": [[0, 4], [0, 3]]},
                    {"name": "hxyz", "variables": ["x", "y", "z"], "nbins": [4, 3, 6], "range": [[0, 4], [0, 3], [-3, 3]],
                     "unbinned_projections": True}]


def adf01_oracle(d) -> dict:
    """pandas: occupancy per (x, y) cell; mean and population std of z per (x, y) cell (rows inside all three ranges)."""
    ix, iy = np.floor(d.x).astype(int), np.floor(d.y).astype(int)
    occupancy = np.zeros((3, 4))
    np.add.at(occupancy, (iy, ix), 1)
    g = pd.DataFrame({"ix": ix, "iy": iy, "z": d.z}).groupby(["ix", "iy"]).z
    return {"hxy": occupancy.ravel().tolist(),                      # x fastest, as RootInteractive flattens
            "cells": {k: (float(n), float(m), float(s)) for k, n, m, s in
                      zip(g.size().index, g.size(), g.mean(), g.std(ddof=0))}}


def adf01_differences(state, want) -> list:
    out = [] if state["hxy"] == want["hxy"] else ["hxy"]
    got = {(int(np.floor(c0)), int(np.floor(c1))): (n, m, s)
           for c0, c1, n, m, s in zip(state["c0"], state["c1"], state["entries"], state["mean"], state["std"]) if n}
    if set(got) != set(want["cells"]):
        return out + ["cells"]
    for k, (n, m, s) in want["cells"].items():
        gn, gm, gs = got[k]
        if gn != n or not np.isclose(gm, m, rtol=1e-12, atol=1e-12) or not np.isclose(gs, s, rtol=1e-12, atol=1e-12):
            out.append(k)
    return out


@_node(pytest.mark.feature("HIST.histogram_nd"), pytest.mark.feature("HIST.profile_unbinned"))
def test_ri_adf_01_hist2d_and_profile2d_equal_pandas(node_session, tmp_path):
    """RI-ADF-01, hand-built (no compiler): the XY occupancy and the conditional mean and std of Z per XY cell (the
    projection of a 3-D histogram along Z) equal pandas. Passes today: the acceptance evidence for this shape. The
    contract's falsifier - X and Y transposed - must be rejected (checked in the same test)."""
    d = adf01_table()
    figures = [[["bin_center_0"], ["bin_count"], {"source": "hxy"}], [["bin_center_0"], ["mean"], {"source": "hxyz_2"}]]
    page, report = _node_page(node_session, tmp_path, _build(tmp_path, d, figures, "adf01.json", histograms=ADF01_HISTOGRAMS))
    state = page.evaluate("""() => { const H = window.__riH, p = H.model('hxyz_2', ['HistoNdProfile']);
      if (!p) throw new Error(H.why('hxyz_2', ['HistoNdProfile']));
      const c = n => Array.from(p.get_column(n));
      return {hxy: Array.from(H.model('hxy', ['HistoNdCDS']).get_column('bin_count')),
              c0: c('bin_center_0'), c1: c('bin_center_1'), entries: c('entries'), mean: c('mean'), std: c('std')}; }""")
    _no_errors(report)
    want = adf01_oracle(d)
    assert not adf01_differences(state, want), adf01_differences(state, want)[:5]
    transposed = {"hxy": np.asarray(want["hxy"]).reshape(3, 4).T.ravel().tolist(),
                  "cells": {(k[1], k[0]): v for k, v in want["cells"].items()}}
    assert adf01_differences(state, transposed), "the comparator must reject X and Y transposed"
