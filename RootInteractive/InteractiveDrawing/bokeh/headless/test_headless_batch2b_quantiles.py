"""PHASE_5 Batch 2B (well-defined part) — profile quantiles, in Node. Tests only (Batch 2B CRR v0.1, Claude20.RI,
2026-11-13).

Plain words. The Architect decided on 2026-11-13: statistics that are well defined are fixed now; statistics with
several valid definitions (for example the quantile of weighted data) get options later, when the ADF/dfdraw
compilation starts; RootInteractive keeps its present methods meanwhile. So these tests do not choose a quantile
definition. They fix the bugs that are wrong under ANY definition:
  B5     a quantile that falls into a sub-bin with exactly two rows is (a + b) * q, which can lie outside [a, b]
         (MathUtils.ts `quantile`, line 132);
  B5b    at the lower or upper end of a sub-bin's share the quantile is the first or last row in ROW order, not the
         smallest or largest value: the result depends on the row order (MathUtils.ts 135-136);
  q=1    the quantile at q = 1 is NaN instead of the maximum (HistoNdProfile.ts: `high > q * entries` is never true);
  ST-D4  quantiles requested in a non-ascending order are wrong (the profile's single pass assumes ascending order);
  WT-D3  a zero-weight row can be returned as a weighted quantile (MathUtils.ts weighted_kth_value).

RootInteractive's present unweighted method (the Architect's DEC-10, "sub-bin"): in each profile cell, find the
Y sub-bin where the cumulative count first exceeds q * N; rescale q to that sub-bin; take the linear quantile of the
raw values inside that sub-bin. THE ORACLE is this method with the bugs removed (two rows interpolate; the ends of a
share are the sub-bin's minimum and maximum; q = 1 is the cell's maximum; the request order does not matter). It is
cross-checked by properties every standard definition has (inside [min, max], q = 0 the minimum, q = 1 the maximum,
non-decreasing in q, independent of row and request order). THE DEFECT MODEL is an exact port of the present
TypeScript (kth_value, quantile and the profile's pass, including the in-place reordering) with each defect
switchable; a strict test raises KnownDefect only when the smallest set of defects whose model equals the page is the
declared one.

For WEIGHTED quantiles no definition is chosen: a value is accepted when it is one a standard weighted definition
gives from the rows with positive weight (inverse CDF, the next value at an exact boundary, or their midpoint;
PHASE_4's `_quantile_candidates`). WT-D3 is a zero-weight row's value that is none of them; q=1 is NaN.

Shapes: RI-ADF-05 (quantiles .15, .5, .9 per facet and group) and RI-ADF-06 (.45, .55), plus the ends 0 and 1, in a
profile of Y per (X bin, group) cell. Weighted: a selection branch as a 0/1 weight (contract C03), whose integer
cumulative weights reach the exact boundaries where WT-D3 acts.

TEST CORRECTION v0.2 (GPT24.RI's review of the test commits, 2026-11-13, findings 2B-P1-01 to -03):
- a page must have exactly the 15 cells, no cell twice, and exactly one value per requested level in each cell;
- an error must equal the expected one in full, with its number (not only up to the colon);
- the weighted page is KNOWN (WT-D3, q=1) only when it equals, bit for bit, the exact port of the present weighted code
  (oracles/profile_port.py) - not merely when its bad values are zero-weight values. Mutants of each kind are tests.
The set of accepted weighted values stays a temporary bounded set (inverse CDF, next value, midpoint), not the future
weighted method (2B-P2-01): the method-free properties remain the target.

FIX COMMIT (Batch 2B fix CRR v0.1, Claude20.RI, 2026-11-13): B5, B5b, q = 1, ST-D4 and WT-D3 are fixed. The strict
marks are off and the signature tests removed; a page equal to the port of master's code is now a REGRESSION.
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import histograms as ho
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import profile_port
from .test_headless_node_weights import _quantile_candidates


def regression(condition: bool, message: str) -> None:
    """FIX COMMIT: the exact signature of a fixed defect is a regression - an ordinary failure that names it."""
    if condition:
        raise AssertionError("REGRESSION, a fixed defect is back: " + message)
QUANTILE_DEFECTS = ("B5", "B5b", "q=1", "ST-D4")


def _node(*marks):
    def wrap(test):
        for m in reversed([pytest.mark.node_stats, pytest.mark.backend("node"), pytest.mark.layer("integration")]
                          + list(marks)):
            test = m(test)
        return test
    return wrap


def _unit(test):
    return pytest.mark.headless_unit(pytest.mark.backend("python")(pytest.mark.layer("unit")(test)))


# ===================================================================================== the table and the page
SEED = 20261113
N = 600
NBINS = [5, 3, 8]                                    # X, group, Y sub-bins
RANGES = [[0.0, 5.0], [0.0, 3.0], [-4.0, 4.0]]
ASCENDING = [0.0, 0.15, 0.45, 0.5, 0.55, 0.9, 1.0]   # RI-ADF-05 and -06 levels, plus both ends
MIXED = [0.55, 0.15, 0.9, 0.45]                      # a request list in no order (ST-D4)


def table(order="table") -> pd.DataFrame:
    """x in [0, 5), a group index g in 0..2, y ~ N(0, 1.2) clipped inside [-4, 4); a 0/1 branch weight b. With
    order='permuted' the same rows in another order (the oracle must not change)."""
    rng = np.random.default_rng(SEED)
    d = pd.DataFrame({"x": rng.uniform(0, 5, N), "g": rng.integers(0, 3, N).astype(float),
                      "y": np.clip(rng.normal(0, 1.2, N), -3.95, 3.95), "b": (rng.uniform(size=N) < 0.5).astype(float)})
    if order == "permuted":
        d = d.iloc[np.random.default_rng(SEED + 9).permutation(N)].reset_index(drop=True)
    return d


def build(tmp_path, d, quantiles, weights=None, name="q"):
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    hist = {"name": "hq", "variables": ["x", "g", "y"], "nbins": NBINS, "range": RANGES, "quantiles": list(quantiles),
            "unbinned_projections": True}
    if weights:
        hist["weights"] = weights
    shown = sa.show
    sa.show = lambda *a, **k: None
    try:
        root = sa.bokehDrawSA.fromArray(d, None, [[["bin_center_0"], ["mean"], {"source": "hq_2"}]], [],
                                        sourceArray=[hist], aliasArray=[], parameterArray=[], layout=[[0]],
                                        widgetLayout=[], arrayCompression=[(".*", [("code", 0), ("zip", 0), ("base64", 0)])]).pAll
    finally:
        sa.show = shown
    doc = Document()
    doc.add_root(root)
    out = tmp_path / f"batch2b_{name}.json"
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


READ_JS = """(k) => {
  const H = window.__riH, p = H.model('hq_2', ['HistoNdProfile']);
  if (!p) throw new Error(H.why('hq_2', ['HistoNdProfile']));
  const col = n => { try { return Array.from(p.get_column(n), x => Number.isFinite(x) ? x : null); }
                     catch (e) { return 'ERR ' + e.message; } };
  const out = {c0: col('bin_center_0'), c1: col('bin_center_1'), entries: col('entries')};
  for (let i = 0; i < k; i++) out['quantile_' + i] = col('quantile_' + i);
  return out;
}"""


def page_quantiles(node_session, tmp_path, d, quantiles, weights=None, name="q"):
    """Per cell (X bin, group), in the page's order: the quantiles as a list (None for NaN), or the error text."""
    report = ScenarioResult(id=f"batch2b/{name} [node]", outdir=str(tmp_path))
    page = node_session.new_page(report)
    page.goto(build(tmp_path, d, quantiles, weights, name))
    page.wait_for_function(page.load_condition)
    got = page.evaluate(READ_JS, len(quantiles))
    errors = [got[f"quantile_{i}"] for i in range(len(quantiles)) if isinstance(got[f"quantile_{i}"], str)]
    if errors:
        return {"error": errors[0]}
    assert not report.page_errors and not report.console_errors, (report.page_errors, report.console_errors)
    return {"cells": raw_cells(got, len(quantiles))}


def raw_cells(got, nq) -> dict:
    """Review F4 (GPT24, Batches 2A/2B fix v0.1): the page's raw output, checked BEFORE the cells are paired. Every
    column - both cell coordinates, entries and each quantile_i - must have exactly N_CELLS values, so an extra value
    at the end of one column, or a column of another length, can no longer be cut off unseen by zip(). Then every
    cell must appear once."""
    names = ["c0", "c1", "entries"] + [f"quantile_{j}" for j in range(nq)]
    lengths = {n: len(got[n]) for n in names}
    assert all(v == N_CELLS for v in lengths.values()), f"malformed page output, column lengths {lengths} (expected {N_CELLS} each)"
    cells = {}
    for i in range(N_CELLS):
        key = (int(got["c0"][i]), int(got["c1"][i]))
        assert key not in cells, f"the page has the cell {key} twice"
        cells[key] = [got[f"quantile_{j}"][i] for j in range(nq)]
    return cells


# ===================================================================================== the port of the present code
class QuantileOutOfRange(Exception):
    pass


def kth_value(s, k, begin, end):
    """MathUtils.ts kth_value, line by line (in place; no NaN in these tables)."""
    if not (begin <= k < end):
        raise ValueError("Element out of range")
    while begin < end:
        pivot = s[k]
        pivot_idx = begin + sum(1 for i in range(begin, end) if s[i] < pivot)
        mid, high, low = pivot_idx, end - 1, begin
        while low < pivot_idx:
            x = s[low]
            if x > pivot:
                s[low], s[high] = s[high], x
                high -= 1
            elif x == pivot:
                s[low], s[mid] = s[mid], x
                mid += 1
            else:
                low += 1
        while mid <= high:
            x = s[mid]
            if x == pivot:
                mid += 1
            else:
                s[mid], s[high] = s[high], x
                high -= 1
        if k < pivot_idx:
            end = pivot_idx
        elif k >= mid:
            begin = mid
        else:
            return


def quantile(s, q, begin, end, defects):
    """MathUtils.ts quantile (lines 124-150) with B5 and B5b switchable; without them: linear interpolation of the
    sorted slice, its minimum and maximum at the ends."""
    if q < 0 or q > 1:
        raise QuantileOutOfRange(f"Quantile out of range: {q}")
    if not (0 <= begin < end):
        raise ValueError("Element out of range")
    m = end - begin
    if m == 1:
        return s[begin]
    if m == 2:
        if "B5" in defects:
            return (s[begin] + s[begin + 1]) * q
        a, b = sorted(s[begin:end])
        return a + (b - a) * q
    k = begin + (m - 1) * q
    k1 = math.floor(k)
    if k == begin:
        return s[begin] if "B5b" in defects else min(s[begin:end])
    if k1 == end - 1:
        return s[end - 1] if "B5b" in defects else max(s[begin:end])
    kth_value(s, k1, begin, end)
    if k % 1 == 0:
        return s[k1]
    value_k2 = s[k1 + 1]
    for k3 in range(math.ceil(k), end):
        if s[k3] < value_k2:
            value_k2 = s[k3]
    return s[k1] + (value_k2 - s[k1]) * (k - k1)


def cell_quantiles(values_by_subbin, qs, defects):
    """HistoNdProfile.ts, the unbinned unweighted pass over one cell (lines 212-256): `values_by_subbin` holds each Y
    sub-bin's rows in row order. With ST-D4 the requests are processed in the given order in one ascending pass;
    without it, in ascending order. With q=1 a request at 1 is never reached (NaN); without it, it is the maximum."""
    arr = [v for sub in values_by_subbin for v in sub]
    bounds, at = [], 0
    for sub in values_by_subbin:
        bounds.append((at, at + len(sub)))
        at += len(sub)
    n = len(arr)
    order = list(range(len(qs))) if "ST-D4" in defects else sorted(range(len(qs)), key=lambda i: qs[i])
    out = [None] * len(qs)
    i_q, low = 0, 0
    while i_q < len(order) and qs[order[i_q]] < 0:
        i_q += 1
    for (b_lo, b_hi) in bounds:
        high = low + (b_hi - b_lo)
        while i_q < len(order) and high > qs[order[i_q]] * n:
            q_scaled = (qs[order[i_q]] * n - low) / (high - low)
            out[order[i_q]] = quantile(arr, q_scaled, b_lo, b_hi, defects)
            i_q += 1
        low = high
    if "q=1" not in defects:
        for j in order[i_q:]:
            if qs[j] == 1 and n:
                out[j] = max(arr)
    return out


def expected(d, qs, defects=()) -> dict:
    """Per cell (X bin, group): the quantiles the page shows with the given defects ({} = the oracle); or the error
    the page raises."""
    idx = ho.bin_index_nd([d.x, d.g, d.y], NBINS, RANGES)
    nx, ng, _ = NBINS
    cells = {}
    try:
        for ix in range(nx):
            for ig in range(ng):
                subs = []
                for iy in range(NBINS[2]):
                    flat = ix + nx * ig + nx * ng * iy
                    subs.append([float(v) for v in d.y.to_numpy()[idx == flat]])
                if sum(map(len, subs)):
                    cells[(ix, ig)] = cell_quantiles(subs, qs, set(defects))
                else:
                    cells[(ix, ig)] = [None] * len(qs)
    except QuantileOutOfRange as exc:
        return {"error": f"ERR {exc}"}
    return {"cells": cells}


N_CELLS = NBINS[0] * NBINS[1]


def _same(got, want) -> bool:
    """Equal outputs: the same error in full (its number included), or exactly the same N_CELLS cells, each with
    exactly one value per requested level, in the requested order, every value equal (NaN = NaN; 1e-12 relative)."""
    if "error" in got or "error" in want:
        return "error" in got and "error" in want and got["error"] == want["error"]
    if set(got["cells"]) != set(want["cells"]) or len(got["cells"]) != N_CELLS:
        return False
    for key, w in want["cells"].items():
        g = got["cells"][key]
        if len(g) != len(w):
            return False
        for a, b in zip(g, w):
            if (a is None) != (b is None) or (a is not None and not np.isclose(a, b, rtol=1e-12, atol=1e-12)):
                return False
    return True


def explanations(got, d, qs) -> list:
    """The smallest sets of quantile defects whose model equals the page ([[]] = the oracle; [] = none)."""
    for size in range(len(QUANTILE_DEFECTS) + 1):
        found = [sorted(c) for c in combinations(QUANTILE_DEFECTS, size) if _same(got, expected(d, qs, c))]
        if found:
            return found
    return []


def properties_hold(d, qs, result) -> list:
    """Definition-free properties of every cell's quantiles: inside [min, max]; q = 0 the minimum; q = 1 the maximum;
    non-decreasing in q. Returns the violations."""
    idx = ho.bin_index_nd([d.x, d.g, d.y], NBINS, RANGES)
    nx, ng, _ = NBINS
    bad = []
    for (ix, ig), values in result["cells"].items():
        rows = d.y.to_numpy()[(idx >= 0) & (idx % nx == ix) & ((idx // nx) % ng == ig)]
        if not len(rows):
            continue
        lo, hi = rows.min(), rows.max()
        pairs = sorted(zip(qs, values))
        for q, v in pairs:
            if v is None or not lo <= v <= hi or (q == 0 and v != lo) or (q == 1 and v != hi):
                bad.append(((ix, ig), q, v))
        vals = [v for _, v in pairs if v is not None]
        if any(b < a for a, b in zip(vals, vals[1:])):
            bad.append(((ix, ig), "not monotone", vals))
    return bad


EXPECTED_SETS = {"ascending": ["B5", "B5b", "q=1"], "permuted": ["B5", "B5b", "q=1"], "mixed": ["ST-D4"]}


# ===================================================================================== Python: the oracle and the port
@_unit
def test_the_oracle_satisfies_every_definition_free_property():
    """The oracle (the present method without its bugs) is inside [min, max], gives the minimum at q = 0 and the
    maximum at q = 1, is non-decreasing, and does not depend on the row order or on the request order."""
    d, p = table(), table("permuted")
    want = expected(d, ASCENDING)
    assert not properties_hold(d, ASCENDING, want), properties_hold(d, ASCENDING, want)[:3]
    assert _same(expected(p, ASCENDING), want)
    mixed = expected(d, MIXED)
    for key, values in mixed["cells"].items():
        assert values == [want["cells"][key][ASCENDING.index(q)] for q in MIXED]


@_unit
def test_the_port_reproduces_phase_4s_hand_signatures():
    """The port with its defects gives PHASE_4's hand-checked numbers (test_headless_node_histograms.py): B5 on the
    values 1, 2, 3 | 12, 14 at q = .7 and .9 gives 6.5 and 19.5; B5b on 1, 2 | 15, 12, 18 at q = .4 gives the first
    row of the sub-bin in row order (15, and 12 when the order is 12, 15); q = 1 gives NaN."""
    assert cell_quantiles([[1., 2., 3.], [12., 14.]], [.7, .9], {"B5"}) == [6.5, 19.5]
    assert cell_quantiles([[1., 2., 3.], [12., 14.]], [.7, .9], set()) == [12.5, 13.5]
    assert cell_quantiles([[1., 2.], [15., 12., 18.]], [.4], {"B5b"}) == [15.]
    assert cell_quantiles([[1., 2.], [12., 15., 18.]], [.4], {"B5b"}) == [12.]
    assert cell_quantiles([[1., 2.], [15., 12., 18.]], [.4], set()) == [12.]
    assert cell_quantiles([[1., 2., 3.]], [1.], {"q=1"}) == [None] and cell_quantiles([[1., 2., 3.]], [1.], set()) == [3.]


@_unit
@pytest.mark.parametrize("case", sorted(EXPECTED_SETS))
def test_the_model_explains_each_page_by_the_declared_defects(case):
    """The full model of each strict page is explained by exactly its declared defects, and differs from the oracle."""
    d = table("permuted" if case == "permuted" else "table")
    qs = MIXED if case == "mixed" else ASCENDING
    full = expected(d, qs, QUANTILE_DEFECTS)
    assert explanations(full, d, qs) == [EXPECTED_SETS[case]]
    assert not _same(full, expected(d, qs))


@_unit
def test_the_comparator_rejects_mutants():
    """Gate G4: one cell's value moved, two levels swapped, one cell's NaN filled with the median - none is explained."""
    d = table()
    want = expected(d, ASCENDING)
    key = sorted(want["cells"])[4]
    for mutate in (lambda v: [v[0]] + [v[1] + 1e-6] + v[2:], lambda v: [v[0], v[2], v[1]] + v[3:]):
        bad = {"cells": {**want["cells"], key: mutate(want["cells"][key])}}
        assert explanations(bad, d, ASCENDING) == [], mutate
    full = expected(d, ASCENDING, QUANTILE_DEFECTS)
    filled = {"cells": {k: [v if v is not None else 0.0 for v in vals] for k, vals in full["cells"].items()}}
    assert explanations(filled, d, ASCENDING) == []


def _raw_from(cells, nq) -> dict:
    """The page's raw column form of a cell map (inverse of raw_cells), for the mutants below."""
    keys = sorted(cells)
    out = {"c0": [k[0] + 0.5 for k in keys], "c1": [k[1] + 0.5 for k in keys], "entries": [1.0] * len(keys)}
    for j in range(nq):
        out[f"quantile_{j}"] = [cells[k][j] for k in keys]
    return out


@_unit
def test_the_raw_output_check_rejects_extra_and_missing_values():
    """Review F4 (GPT24): 15 correct coordinate pairs with quantile_0 one value longer (99999 at the end) passed the
    v0.1 comparator, because zip() cut the extra value off. Now each of these raw mutants is rejected before pairing:
    an extra value at the end of a quantile column, of a coordinate column, or of entries; a quantile column one value
    short; a cell twice. The unchanged raw form is accepted and gives back the same cells."""
    d = table()
    want = expected(d, ASCENDING)["cells"]
    nq = len(ASCENDING)
    good = _raw_from(want, nq)
    assert raw_cells(good, nq) == want
    mutants = {}
    for col in ("quantile_0", "c1", "entries"):
        m = {k: list(v) for k, v in good.items()}
        m[col] = m[col] + [99999.0]
        mutants[f"extra value at the end of {col}"] = m
    short = {k: list(v) for k, v in good.items()}
    short["quantile_3"] = short["quantile_3"][:-1]
    mutants["quantile_3 one value short"] = short
    twice = {k: list(v) for k, v in good.items()}
    twice["c0"][1], twice["c1"][1] = twice["c0"][0], twice["c1"][0]
    mutants["a cell twice"] = twice
    for name, m in mutants.items():
        with pytest.raises(AssertionError):
            raw_cells(m, nq)
            pytest.fail(f"accepted: {name}")


@_unit
def test_the_comparator_rejects_structure_and_error_mutants():
    """2B-P1-01/02 (GPT24): a cell with one value missing, a cell with an extra value 99999, a missing cell - each
    made from the oracle and from the full model - and the expected error with another number: none is explained."""
    d = table()
    for model in (expected(d, ASCENDING), expected(d, ASCENDING, QUANTILE_DEFECTS)):
        key = sorted(model["cells"])[4]
        short = {"cells": {**model["cells"], key: model["cells"][key][:-1]}}
        extra = {"cells": {**model["cells"], key: model["cells"][key] + [99999.0]}}
        missing = {"cells": {k: v for k, v in model["cells"].items() if k != key}}
        for name, bad in (("one value missing", short), ("an extra value", extra), ("a missing cell", missing)):
            assert not _same(bad, model) and explanations(bad, d, ASCENDING) == [], name
    want = expected(d, MIXED, QUANTILE_DEFECTS)
    # the page's exact text (measured in Node, equal to the model); the CRR v0.1 rounded it to -0.45
    assert want == {"error": "ERR Quantile out of range: -0.446875"}, want
    assert _same({"error": "ERR Quantile out of range: -0.446875"}, want)
    assert not _same({"error": "ERR Quantile out of range: -0.45"}, want)
    assert not _same({"error": "ERR Quantile out of range: 12345"}, want)
    assert explanations({"error": "ERR Quantile out of range: 12345"}, d, MIXED) == []


# ===================================================================================== Node: unweighted quantiles
CASES = {"ascending": ("table", ASCENDING), "permuted": ("permuted", ASCENDING), "mixed": ("table", MIXED)}


@_node(pytest.mark.feature("HIST.profile_unbinned"))
@pytest.mark.parametrize("case", sorted(CASES))
def test_strict_profile_quantiles_follow_the_present_method_without_its_bugs(node_session, tmp_path, case):
    """RI-ADF-05/06 shape: every quantile of every (X bin, group) cell equals RootInteractive's sub-bin method with
    the bugs removed; the same after permuting the rows, and for a request list in no order."""
    order, qs = CASES[case]
    d = table(order)
    got = page_quantiles(node_session, tmp_path, d, qs, name=case)
    if _same(got, expected(d, qs)):
        return
    sets = explanations(got, d, qs)
    regression(sets == [EXPECTED_SETS[case]], "known defects " + ", ".join(EXPECTED_SETS[case]))
    raise AssertionError(f"the page shows {sets or 'no known set of defects'}, declared {EXPECTED_SETS[case]}")


@_node(pytest.mark.feature("HIST.profile_unbinned"))
def test_control_interior_levels_in_wide_sub_bins_are_right_today(node_session, tmp_path):
    """Healthy control: a level whose share falls strictly inside a sub-bin with three or more rows is right today
    (the median of every populated cell here). Shows the method itself is not the problem."""
    d = table()
    got = page_quantiles(node_session, tmp_path, d, [0.5], name="median")
    want = expected(d, [0.5])
    full = expected(d, [0.5], QUANTILE_DEFECTS)
    healthy = [k for k in want["cells"] if want["cells"][k] == full["cells"][k] and want["cells"][k][0] is not None]
    assert len(healthy) >= 10
    for k in healthy:
        assert np.isclose(got["cells"][k][0], want["cells"][k][0], rtol=1e-12), k


# ===================================================================================== Node: weighted quantiles (WT-D3)
WEIGHTED = [0.15, 0.5, 0.9, 1.0]


def weighted_port(d) -> dict:
    """The present code's weighted quantiles per cell (oracles/profile_port.py, the exact port; a single weight, so
    WT-D4 cannot act), None for NaN - what the page shows today."""
    (nx, ng, ny), (rx, rg, ry) = NBINS, RANGES
    out = profile_port.profile([(d.x.to_numpy(float), nx, tuple(rx)), (d.g.to_numpy(float), ng, tuple(rg))],
                               d.y.to_numpy(float), (ny, tuple(ry)), {"b": d.b.to_numpy(float)}, quantiles=WEIGHTED,
                               defects=("WT-D2", "WT-D4"))["b"]
    return {"cells": {(z % nx, z // nx): [None if math.isnan(out[f"quantile_{i}"][z]) else out[f"quantile_{i}"][z]
                                          for i in range(len(WEIGHTED))] for z in range(nx * ng)}}


def weighted_violations(d, got) -> list:
    """The values that are no standard weighted value of the cell's positive-weight rows (PHASE_4
    `_quantile_candidates`), as ((ix, ig), q, value, kind); kind 'WT-D3' = a zero-weight row's value, 'q=1' = NaN at
    q = 1, 'other' = anything else. A cell without weight must give None."""
    idx = ho.bin_index_nd([d.x, d.g, d.y], NBINS, RANGES)
    nx, ng, _ = NBINS
    bad = []
    for (ix, ig), values in got["cells"].items():
        sel = (idx >= 0) & (idx % nx == ix) & ((idx // nx) % ng == ig)
        y, w = d.y.to_numpy()[sel], d.b.to_numpy()[sel]
        for q, v in zip(WEIGHTED, values):
            if w.sum() <= 0:
                if v is not None:
                    bad.append(((ix, ig), q, v, "other"))
            elif v is not None and float(v) in _quantile_candidates(y, w, q):
                continue
            elif q == 1 and v is None:
                bad.append(((ix, ig), q, v, "q=1"))
            elif v is not None and float(v) in set(y[w == 0].astype(float)):
                bad.append(((ix, ig), q, v, "WT-D3"))
            else:
                bad.append(((ix, ig), q, v, "other"))
    return bad


def weighted_verdict(d, got) -> tuple:
    """Structure first: exactly the N_CELLS cells, each with one value per requested level. PASS: no violation (every
    value a standard weighted value of the positive-weight rows). KNOWN: the page equals the exact port of the present
    code in every cell and at every level, and its violations are only WT-D3 and q=1 - returns the set found.
    UNEXPLAINED: anything else (a missing or extra cell or value, another value, a page unlike the port)."""
    cells = got.get("cells", {})
    if len(cells) != N_CELLS or set(cells) != set(weighted_port(d)["cells"]) or \
            any(len(v) != len(WEIGHTED) for v in cells.values()):
        return "UNEXPLAINED", "the structure: cells or values missing or extra"
    bad = weighted_violations(d, got)
    if not bad:
        return "PASS", set()
    port = weighted_port(d)
    if got["cells"] == port["cells"] and all(kind != "other" for *_, kind in bad):
        return "KNOWN", {kind for *_, kind in bad}
    return "UNEXPLAINED", [b for b in bad if b[3] == "other"][:5] or "the page is not the present code's port"


@_unit
def test_weighted_candidates_include_q1_as_the_maximum_and_exclude_zero_weight_rows():
    y, w = np.array([1., 2., 3., 4.]), np.array([1., 0., 1., 1.])
    assert _quantile_candidates(y, w, 1.0) == {4.0}
    assert 2.0 not in _quantile_candidates(y, w, 0.5)


@_node(pytest.mark.feature("HIST.profile_unbinned"))
def test_strict_weighted_quantiles_are_standard_values_of_the_positive_rows(node_session, tmp_path):
    """C03 shape: a selection branch as a 0/1 weight. Every weighted quantile of every cell is a value a standard
    weighted definition gives (no definition is chosen: Architect, 2026-11-13); never a zero-weight row's value."""
    d = table()
    got = page_quantiles(node_session, tmp_path, d, WEIGHTED, weights="b", name="weighted")
    assert "error" not in got, got
    verdict, detail = weighted_verdict(d, got)
    if verdict == "PASS":
        return
    regression(verdict == "KNOWN" and detail == {"WT-D3", "q=1"}, "WT-D3 and q=1")
    raise AssertionError(f"{verdict}: {detail}; declared ['WT-D3', 'q=1']")


@_unit
def test_falsifier_weighted_structure_and_value_mutants_are_unexplained():
    """2B-P1-03 (GPT24): made from the port of the present code - one cell left out (the reviewer's one-cell page),
    a zero-weight value replaced by another zero-weight row of the same cell, a value outside the cell, two levels
    swapped, an extra value - each is UNEXPLAINED, never WT-D3."""
    d = table()
    port = weighted_port(d)
    assert weighted_verdict(d, port) == ("KNOWN", {"WT-D3", "q=1"})
    bad = weighted_violations(d, port)
    key, q, v, _ = next(b for b in bad if b[3] == "WT-D3")
    i = WEIGHTED.index(q)
    idx = ho.bin_index_nd([d.x, d.g, d.y], NBINS, RANGES)
    nx, ng, _ = NBINS
    sel = (idx >= 0) & (idx % nx == key[0]) & ((idx // nx) % ng == key[1])
    zeros = [float(x) for x in d.y.to_numpy()[sel & (d.b.to_numpy() == 0)] if float(x) != v]
    def with_value(value, k=key, j=i):
        return {"cells": {**port["cells"], k: [value if n == j else x for n, x in enumerate(port["cells"][k])]}}
    swapped = list(port["cells"][key])
    swapped[0], swapped[1] = swapped[1], swapped[0]
    mutants = {"one cell only": {"cells": {key: port["cells"][key]}},
               "a cell left out": {"cells": {k: x for k, x in port["cells"].items() if k != key}},
               "another zero-weight row": with_value(zeros[0]),
               "a value outside the cell": with_value(123.0),
               "two levels swapped": {"cells": {**port["cells"], key: swapped}},
               "an extra value": {"cells": {**port["cells"], key: port["cells"][key] + [1.0]}}}
    for name, page in mutants.items():
        assert weighted_verdict(d, page)[0] == "UNEXPLAINED", name


# ===================================================================================== v0.2: ties and the weight domain
@_node(pytest.mark.feature("HIST.profile_unbinned"))
def test_weighted_ties_and_the_weight_domain(node_session, tmp_path):
    """Reviews F5 and F6 (Batches 2A/2B fix v0.1). The supported weight domain, stated without choosing a weighted
    method (the Architect's statistics decision of 2026-11-13): finite, non-negative weights; a cell has defined
    quantiles only when its total weight is positive.
    Cell (0, 0): y = 2, 2, 3.5 with weights 0, 2, 0 - a tie of equal values, one of them of weight 0. Every level
    (0, 0.5, 1) is 2: the returned VALUE always has positive weight (the row the search returns may be the 2 of weight
    0; its value is the same). 3.5 has weight 0 and is never the answer, also not at q = 1.
    Cell (1, 0): y = 1, 2, 3, all of weight 0 - total weight 0, so no level is defined (NaN)."""
    d = pd.DataFrame({"x": [0.5, 0.5, 0.5, 1.5, 1.5, 1.5], "g": [0.5] * 6, "y": [2.0, 2.0, 3.5, 1.0, 2.0, 3.0],
                      "w": [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]})
    got = page_quantiles(node_session, tmp_path, d, [0.0, 0.5, 1.0], weights="w", name="ties_domain")
    assert got["cells"][(0, 0)] == [2.0, 2.0, 2.0], got["cells"][(0, 0)]
    assert got["cells"][(1, 0)] == [None, None, None], got["cells"][(1, 0)]
