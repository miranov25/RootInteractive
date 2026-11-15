"""PHASE_5 Batch 4.a, part 3: the calibration chain. Tests only; RootInteractive's code is not changed.

In plain words. A measurement table is joined BY KEY to three calibration tables, one after the other, so that every
measurement row carries its three calibration values (decision L-23: dEdx -> three calibration tables):
  * the first calibration table is keyed by one integer (like a run number);
  * the second by two integers (like run and sector);
  * the third by a floating-point value that must agree within the join's tolerance (like a time).
The tables are synthetic and say nothing about a real calibration; the Architect decided that the chain is joined by
equal keys for now, and that a nearest-value join comes later (L-38).

Why this chain matters. A table that is joined by three different keys can be sorted by only one of them. So the
chain works only if unsorted keys are joined correctly (L-32). It also needs: a key without a partner (a row with no
calibration shows NaN in a `left` join and is dropped by an `inner` join, L-35), a NaN key that never matches (L-34),
two keys (L-37), the rows in the measurement table's order (L-39, a proposal), and a join whose child is itself a
join (defect J-D5: for key joins today's code then pairs the rows by position).

The words, the way a test decides (OK, KNOWN, anything else fails) and the helpers are those of part 1
(test_headless_batch4a_joins.py). The reference is plain Python on the source tables: a lookup of each row's keys.
"""
from __future__ import annotations

import math

import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4a_joins import NAN, STRICT, _run, _strict

pytestmark = [pytest.mark.node_joins]
nan = float("nan")

# The measurement table. It is sorted by none of its three keys. Row 6 has a run without calibration, row 5 a time
# without calibration, row 7 an unknown time (NaN), and row 3 a time equal to 20 only within the tolerance.
MAIN = {"mid": [0, 1, 2, 3, 4, 5, 6, 7],
        "run": [2, 1, 2, 3, 1, 2, 9, 1],
        "sector": [0, 1, 1, 0, 0, 0, 0, 1],
        "t": [10.0, 20.0, 30.0, 20.000004, 10.0, 40.0, 30.0, nan],
        "q": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]}
BY_RUN = {"run0": [1, 2, 3], "g0": [1.01, 1.02, 1.03]}
BY_RUN_SECTOR = {"run1": [1, 1, 2, 2, 3], "sector1": [0, 1, 0, 1, 0], "g1": [1.10, 1.11, 1.20, 1.21, 1.30]}
BY_TIME = {"t2": [10.0, 20.0, 30.0], "g2": [0.91, 0.92, 0.93]}
TABLES = {"M": MAIN, "C0": BY_RUN, "C1": BY_RUN_SECTOR, "C2": BY_TIME}
LINKS = {   # name: (calibration table, keys in the measurement table, keys in the calibration table, value column)
    "by run": ("C0", ["run"], ["run0"], "g0"),
    "by run and sector": ("C1", ["run", "sector"], ["run1", "sector1"], "g1"),
    "by time": ("C2", ["t"], ["t2"], "g2"),
}
TOLERANCE = 1e-5                                                         # the join's default


def _joins(links, how):
    """A chain: the measurement table joined to each calibration table in turn; each join's child is the join before."""
    joins, child = {}, "M"
    for i, link in enumerate(links):
        table, on_child, on_parent, _ = LINKS[link]
        joins[f"J{i}"] = dict(left=child, right=table, on_left=on_child, on_right=on_parent, how=how)
        child = f"J{i}"
    return joins, child


def _lookup(tables, link, row):
    """The calibration value of one measurement row: the row of the calibration table with equal keys, or None."""
    table, on_child, on_parent, value = LINKS[link]
    keys = [tables["M"][k][row] for k in on_child]
    cal = tables[table]
    for j in range(len(cal[value])):
        if all(not math.isnan(a) and abs(a - cal[p][j]) <= TOLERANCE for a, p in zip(keys, on_parent)):
            return cal[value][j]
    return None


def _reference(tables, links, how):
    """Every measurement row in its own order with its calibration values. `left`: NaN where there is none. `inner`:
    only the rows that have every calibration."""
    out = {"mid": [], "q": [], **{LINKS[link][3]: [] for link in links}}
    for row in range(len(tables["M"]["mid"])):
        values = [_lookup(tables, link, row) for link in links]
        if how == "inner" and any(v is None for v in values):
            continue
        out["mid"].append(tables["M"]["mid"][row]); out["q"].append(tables["M"]["q"][row])
        for link, v in zip(links, values):
            out[LINKS[link][3]].append(NAN if v is None else v)
    return out


def _case(node_session, tmp_path, tables, links, how):
    joins, target = _joins(links, how)
    want = _reference(tables, links, how)
    return _run(node_session, tmp_path, tables, joins, target, list(want), want, ordered=True)


CHAINS = [("by run",), ("by run and sector",), ("by time",), ("by run", "by run and sector"),
          ("by run", "by run and sector", "by time"), ("by time", "by run and sector", "by run")]


@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.parametrize("links", CHAINS, ids=[" then ".join(c) for c in CHAINS])
@pytest.mark.parametrize("how", ["left", "inner"])
@pytest.mark.xfail(**STRICT, reason="the calibration chain: unsorted keys (L-32), a key without a partner (J-D4, L-35), "
                                    "a NaN key (L-34), child columns (J-D1), a key join of a join (J-D5)")
def test_strict_every_measurement_gets_its_calibration(node_session, tmp_path, links, how):
    """One calibration table alone, two in a chain, and all three in a chain in two different orders."""
    _strict(_case(node_session, tmp_path, TABLES, links, how), f"{' then '.join(links)}, {how}")


SORTED = {"mid": [0, 1, 2], "run": [1, 2, 3], "sector": [0, 0, 0], "t": [10.0, 20.0, 30.0], "q": [5.0, 5.5, 6.0]}


@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.parametrize("link", ["by run", "by time"])
@pytest.mark.parametrize("how", ["left", "inner"])
def test_control_one_sorted_complete_calibration_is_right_today(node_session, tmp_path, link, how):
    """The healthy side: sorted keys, every row with exactly one partner, one join. Right today, and it must stay."""
    assert _case(node_session, tmp_path, {**TABLES, "M": SORTED}, (link,), how) == "OK"


# ============================================================================================ no Node: the checks themselves
def test_the_measurement_table_is_sorted_by_none_of_its_keys():
    """Otherwise the chain test would not test unsorted keys."""
    for link, (_, on_child, _, _) in LINKS.items():
        keys = [tuple(MAIN[k][i] for k in on_child) for i in range(len(MAIN["mid"])) if not math.isnan(MAIN[on_child[0]][i])]
        assert keys != sorted(keys), link


def test_the_calibration_reference_follows_the_decisions():
    all_three = ("by run", "by run and sector", "by time")
    left = _reference(TABLES, all_three, "left")
    assert left["mid"] == MAIN["mid"] and left["q"] == MAIN["q"]                       # every row, in its own order
    assert left["g0"] == [1.02, 1.01, 1.02, 1.03, 1.01, 1.02, NAN, 1.01]               # run 9 has no calibration
    assert left["g1"] == [1.20, 1.11, 1.21, 1.30, 1.10, 1.20, NAN, 1.11]               # two keys
    assert left["g2"] == [0.91, 0.92, 0.93, 0.92, 0.91, NAN, 0.93, NAN]                # tolerance; 40 missing; NaN
    inner = _reference(TABLES, all_three, "inner")
    assert inner["mid"] == [0, 1, 2, 3, 4]                                             # rows 5, 6 and 7 lack one
    assert _reference(TABLES, all_three[::-1], "inner") == inner                       # the order of the joins is free
