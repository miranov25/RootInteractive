"""PHASE_4_5 A4, test-only shared numerical oracle for V0 -> track -> event.

Standalone fixture and reference, independent of RootInteractive implementation.
To run: python -m pytest -q PHASE_4_5_A4_Shared_Oracle_GPT24_RI_v0_2.py
To create the shared JSON fixture: python PHASE_4_5_A4_Shared_Oracle_GPT24_RI_v0_2.py output.json

CONTRACT v0.2 (proposal, not an approved RootInteractive API specification):
  - Each V0 row is counted once, even when daughter tracks are shared.
  - Invalid event or either invalid track index excludes that V0 row.
  - An event cut selects the parent event. A daughter cut requires BOTH daughters.
  - Event and daughter cuts combine by AND. A histogram of V0 is over V0 rows.
  - Alias 'mass_plus_pos_pt' is mass + 0.1 * gathered positive-daughter pt.
  - Weight parameter chooses either wA or wB, and must recompute counts.
Additional *separate* probe describes in-range cross-event daughters, and
 distinguishes gather-resolution from this fixture's AND-filter policy.
 This is a proposed policy for this fixture, NOT a ratified RI/ADF API contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

EVENTS = [
    {"event_id": 0, "centrality": 12.0},
    {"event_id": 1, "centrality": 38.0},
    {"event_id": 2, "centrality": 55.0},
    {"event_id": 3, "centrality": 71.0},
    {"event_id": 4, "centrality": 85.0},
]
TRACKS = [
    {"track_id": 0, "event_id": 0, "pt": 0.8},
    {"track_id": 1, "event_id": 0, "pt": 1.8},
    {"track_id": 2, "event_id": 1, "pt": 1.3},
    {"track_id": 3, "event_id": 1, "pt": 2.2},
    {"track_id": 4, "event_id": 2, "pt": 2.5},
    {"track_id": 5, "event_id": 2, "pt": 1.6},
    {"track_id": 6, "event_id": 3, "pt": 0.9},
    {"track_id": 7, "event_id": 4, "pt": 1.1},
    {"track_id": 8, "event_id": 4, "pt": 2.3},
]
V0 = [
    {"v0_id": 0, "event_idx": 0, "pos_idx": 0, "neg_idx": 1, "mass": 0.5, "wA": 1, "wB": 12},
    {"v0_id": 1, "event_idx": 0, "pos_idx": 1, "neg_idx": 0, "mass": 1.4, "wA": 2, "wB": 11},
    {"v0_id": 2, "event_idx": 1, "pos_idx": 2, "neg_idx": 3, "mass": 1.6, "wA": 3, "wB": 10},
    {"v0_id": 3, "event_idx": 1, "pos_idx": 2, "neg_idx": 3, "mass": 2.4, "wA": 4, "wB": 9},
    {"v0_id": 4, "event_idx": 2, "pos_idx": 4, "neg_idx": 5, "mass": 2.6, "wA": 5, "wB": 8},
    {"v0_id": 5, "event_idx": 2, "pos_idx": 4, "neg_idx": 5, "mass": 0.8, "wA": 6, "wB": 7},
    {"v0_id": 6, "event_idx": 3, "pos_idx": 6, "neg_idx": 6, "mass": 1.2, "wA": 7, "wB": 6},
    {"v0_id": 7, "event_idx": 4, "pos_idx": 7, "neg_idx": 8, "mass": 2.9, "wA": 8, "wB": 5},
    {"v0_id": 8, "event_idx": 4, "pos_idx": 8, "neg_idx": 7, "mass": 2.1, "wA": 9, "wB": 4},
    {"v0_id": 9, "event_idx": 2, "pos_idx": 4, "neg_idx": -1, "mass": 1.8, "wA": 10, "wB": 3},
    {"v0_id": 10, "event_idx": -1, "pos_idx": 0, "neg_idx": 1, "mass": 0.2, "wA": 11, "wB": 2},
    {"v0_id": 11, "event_idx": 4, "pos_idx": 7, "neg_idx": 99, "mass": 1.3, "wA": 12, "wB": 1},
]
EDGES = [0.0, 1.0, 2.0, 3.0]
ALIAS_EDGES = [0.0, 1.0, 2.0, 3.0, 4.0]
CASES = {
    "all_valid": {"event_indices": [0, 1, 2, 3, 4], "min_pt": 0.0},
    "events_0_2_4": {"event_indices": [0, 2, 4], "min_pt": 0.0},
    "events_0_2_4_pt_ge_1": {"event_indices": [0, 2, 4], "min_pt": 1.0},
    "empty_events": {"event_indices": [], "min_pt": 0.0},
    "all_pt_ge_2": {"event_indices": [0, 1, 2, 3, 4], "min_pt": 2.0},
    "events_2_pt_ge_1_5": {"event_indices": [2], "min_pt": 1.5},
}


def valid_row(v, tracks=TRACKS, events=EVENTS):
    e, p, n = (v[k] for k in ("event_idx", "pos_idx", "neg_idx"))
    if not (0 <= e < len(events) and 0 <= p < len(tracks) and 0 <= n < len(tracks)):
        return False
    return tracks[p]["event_id"] == e and tracks[n]["event_id"] == e


def selected_rows(case, v0=V0, tracks=TRACKS, events=EVENTS):
    selected_events = set(case["event_indices"])
    out = []
    for i, row in enumerate(v0):
        if not valid_row(row, tracks, events):
            continue
        if row["event_idx"] not in selected_events:
            continue
        if tracks[row["pos_idx"]]["pt"] < case["min_pt"]:
            continue
        if tracks[row["neg_idx"]]["pt"] < case["min_pt"]:
            continue
        out.append(i)
    return out


def reference(case, v0=V0):
    rows = selected_rows(case, v0=v0)
    masses = np.asarray([v0[i]["mass"] for i in rows], dtype=float)
    pospt = np.asarray([TRACKS[v0[i]["pos_idx"]]["pt"] for i in rows], dtype=float)
    alias = masses + 0.1 * pospt
    weights_a = np.asarray([v0[i]["wA"] for i in rows], dtype=float)
    weights_b = np.asarray([v0[i]["wB"] for i in rows], dtype=float)
    # Independent *BINNED* per-mass-slice profile. y is positive-daughter pt,
    # so moments use pt-bin CENTERS, not raw pt. Null means empty slice.
    counts2d = np.histogramdd(
        np.column_stack([masses, pospt]) if len(masses) else np.empty((0, 2)),
        bins=[EDGES, EDGES],
    )[0]
    centres = 0.5 * (np.asarray(EDGES[:-1]) + np.asarray(EDGES[1:]))
    entries = np.sum(counts2d, axis=1)
    means = []
    stds = []
    for n, histrow in zip(entries, counts2d):
        if n == 0:
            means.append(None)
            stds.append(None)
        else:
            mu = float(np.dot(histrow, centres) / n)
            variance = float(np.dot(histrow, (centres-mu)**2) / n)
            means.append(mu)
            stds.append(float(np.sqrt(max(variance, 0.0))))
    return {
        "profile_entries": entries.astype(int).tolist(),
        "profile_mean": means,
        "profile_std": stds,
        "profile_isOK": [bool(n > 0) for n in entries],
        "selected_row_ids": [v0[i]["v0_id"] for i in rows],
        "selected_mask": [int(i in rows) for i in range(len(v0))],
        "mass_counts": np.histogram(masses, bins=EDGES)[0].astype(int).tolist(),
        "mass_wA": np.histogram(masses, bins=EDGES, weights=weights_a)[0].astype(int).tolist(),
        "mass_wB": np.histogram(masses, bins=EDGES, weights=weights_b)[0].astype(int).tolist(),
        "alias_counts": np.histogram(alias, bins=ALIAS_EDGES)[0].astype(int).tolist(),
        "mass_pospt_2d": np.histogramdd(
            np.column_stack([masses, pospt]) if len(masses) else np.empty((0, 2)),
            bins=[EDGES, EDGES],
        )[0].astype(int).ravel(order="F").tolist(),
    }


def inspect_foreign_key(row, tracks=TRACKS, events=EVENTS):
    """Fixture-only two separate questions: can a daughter pt be GATHERED,
    and does the proposed same-event selection policy accept the row?

    A *valid* reference with a cross-event daughter still gathers its pt;
    this fixture's stricter event/AND filter separately rejects the row.
    We do not assert this fill behavior is the production ADF contract.
    """
    p, n, e = (row[k] for k in ("pos_idx", "neg_idx", "event_idx"))
    return {
        "pospt_gather": float(tracks[p]["pt"]) if 0 <= p < len(tracks) else None,
        "negpt_gather": float(tracks[n]["pt"]) if 0 <= n < len(tracks) else None,
        "parent_valid": bool(0 <= e < len(events)),
        "accepted_by_fixture_filter": bool(valid_row(row, tracks, events)),
    }


def assert_snapshot_matches(actual, expected, keys=None):
    """Shared failure-inducing comparator for real Node production replies.

    Note: `None` in the fixture corresponds to nonfinite/empty profile output;
    numerical comparisons use exact bin arrays and tolerant profile moments.
    This function is NOT evidence that a production snapshot was executed.
    """
    exact = {"selected_mask", "selected_row_ids", "mass_counts", "mass_wA", "mass_wB",
             "alias_counts", "mass_pospt_2d", "profile_entries", "profile_isOK"}
    keys = list(keys) if keys is not None else list(expected)
    for key in keys:
        if key not in actual:
            raise AssertionError(f"missing production output {key}")
        if key in exact:
            observed = list(actual[key])
            wanted = list(expected[key])
            assert observed == wanted, (key, observed, wanted)
        elif key in ("profile_mean", "profile_std"):
            assert len(actual[key]) == len(expected[key]), key
            for i, (value, want) in enumerate(zip(actual[key], expected[key])):
                if want is None:
                    assert value is None or (isinstance(value, (float, int)) and np.isnan(value)), (key, i, value)
                else:
                    assert value is not None and np.isfinite(value), (key, i, value)
                    np.testing.assert_allclose(float(value), float(want), rtol=1e-10, atol=1e-12,
                                               err_msg=f"{key}[{i}]")
        else:
            raise AssertionError(f"unregistered A4 comparison key: {key}")


def build_fixture():
    cross_event = {"v0_id": 12, "event_idx": 2, "pos_idx": 4, "neg_idx": 3,
                   "mass": 1.7, "wA": 13, "wB": 0}
    # Orphans already in main 12-row fixture; the cross-event row is intentionally
    # kept OUT of the primary selection-universe to preserve v0.1's golden arrays.
    return {"format": "RootInteractive PHASE_4_5 A4 oracle v0.2", "tables": {
        "events": EVENTS, "tracks": TRACKS, "v0": V0},
        "bins": {"mass": EDGES, "pospt": EDGES, "alias": ALIAS_EDGES},
        "policy_probes": {"cross_event_daughter": {"row": cross_event,
                         "expected": inspect_foreign_key(cross_event)},
                          "orphan_negative_daughter": {"row": V0[9],
                         "expected": inspect_foreign_key(V0[9])}},
        "cases": {name: {"selection": spec, "expected": reference(spec)}
                  for name, spec in CASES.items()}}


def test_known_hand_audited_selected_rows_and_weights():
    got = reference(CASES["events_0_2_4_pt_ge_1"])
    assert got["selected_row_ids"] == [4, 5, 7, 8]
    assert got["mass_counts"] == [1, 0, 3]
    assert got["mass_wA"] == [6, 0, 22]
    assert got["mass_wB"] == [7, 0, 17]
    assert got["selected_mask"][9:] == [0, 0, 0]  # invalid/orphan references


def test_conservation_and_orphans():
    for name, cut in CASES.items():
        got = reference(cut)
        assert sum(got["mass_counts"]) == len(got["selected_row_ids"]), name
        assert sum(got["alias_counts"]) == len(got["selected_row_ids"]), name
        assert sum(got["mass_pospt_2d"]) == len(got["selected_row_ids"]), name
        assert len(got["selected_mask"]) == len(V0)
        assert not any(set(got["selected_row_ids"]) & {9, 10, 11})


def test_row_permutation_preserves_population_and_bins():
    perm = [8, 5, 2, 11, 0, 4, 9, 7, 10, 1, 3, 6]
    permuted = [V0[i] for i in perm]
    for name, cut in CASES.items():
        baseline = reference(cut)
        got = reference(cut, v0=permuted)
        for key in ("mass_counts", "mass_wA", "mass_wB", "alias_counts", "mass_pospt_2d",
                    "profile_entries", "profile_mean", "profile_std", "profile_isOK"):
            assert got[key] == baseline[key], (name, key)
        assert sorted(got["selected_row_ids"]) == sorted(baseline["selected_row_ids"])


def test_shared_daughters_not_deduplicated():
    got = reference(CASES["all_valid"])
    assert {2, 3, 4, 5}.issubset(got["selected_row_ids"])
    assert len(got["selected_row_ids"]) == 9  # valid V0 rows, not unique daughter pairs


def test_falsification_wrong_weight_or_relation_is_detected():
    expected = reference(CASES["events_0_2_4_pt_ge_1"])
    mutated_weight = expected["mass_wA"].copy()
    mutated_weight[0] += 1
    with pytest.raises(AssertionError):
        assert_snapshot_matches({**expected, "mass_wA": mutated_weight}, expected)
    mutated_mask = expected["selected_mask"].copy()
    mutated_mask[4], mutated_mask[2] = 0, 1  # same number of rows, different identities
    assert sum(mutated_mask) == sum(expected["selected_mask"])
    with pytest.raises(AssertionError):
        assert_snapshot_matches({**expected, "selected_mask": mutated_mask}, expected)


def test_nonempty_high_pt_and_empty_profile_slice():
    got = reference(CASES["events_2_pt_ge_1_5"])
    assert got["selected_row_ids"] == [4, 5]
    assert got["profile_entries"] == [1, 0, 1]
    assert got["profile_mean"][1] is None
    assert got["profile_std"][1] is None
    assert got["profile_isOK"] == [True, False, True]


def test_cross_event_and_missing_daughter_gather_vs_selection():
    probes = build_fixture()["policy_probes"]
    assert probes["cross_event_daughter"]["expected"] == {
        "pospt_gather": 2.5, "negpt_gather": 2.2,
        "parent_valid": True, "accepted_by_fixture_filter": False,
    }
    assert probes["orphan_negative_daughter"]["expected"] == {
        "pospt_gather": 2.5, "negpt_gather": None,
        "parent_valid": True, "accepted_by_fixture_filter": False,
    }


def test_same_production_comparator_rejects_profile_mutations():
    expected = reference(CASES["events_2_pt_ge_1_5"])
    assert_snapshot_matches(expected, expected)
    wrong = {**expected, "profile_mean": expected["profile_mean"].copy()}
    wrong["profile_mean"][1] = 1.5  # should be an empty slice
    with pytest.raises(AssertionError):
        assert_snapshot_matches(wrong, expected)
    wrong = {**expected, "profile_std": expected["profile_std"].copy()}
    wrong["profile_std"][0] = 0.25
    with pytest.raises(AssertionError):
        assert_snapshot_matches(wrong, expected)


def test_fixture_json_has_complete_case_contract(tmp_path):
    output = build_fixture()
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(output, sort_keys=True))
    reread = json.loads(path.read_text())
    assert reread == output
    assert len(reread["cases"]) == len(CASES)


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("PHASE_4_5_A4_shared_fixture_v0_2.json")
    target.write_text(json.dumps(build_fixture(), indent=2, sort_keys=True) + "\n")
    print(f"Wrote {target}")
