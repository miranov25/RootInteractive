"""
Tests for toy_event_generator.py — Phase 0.1.F.ext

15 tests for the MC generator extension:
  F-ext-01 through F-ext-15

Existing 14 tests in test_toy_event_generator.py are NOT modified.
"""

import numpy as np
import pandas as pd
import pytest
import os
import tempfile

from toy_event_generator import (
    generate_event_display,
    generate_mc_reco,
    generate_mc_reco_its_only,
    generate_mc_reco_tpc_only,
    build_dense_index,
    track_efficiency,
    fit_track_linear,
    fit_track_circle,
    save_tables_extended,
    load_tables_extended,
    LAYERS_ALL,
    LAYERS_ITS,
    LAYERS_TPC,
    N_ITS,
)


@pytest.fixture
def mc_reco_data():
    """Generate extended dataset (100 events, full detector)."""
    return generate_mc_reco(n_events=100, seed=42)


@pytest.fixture
def mc_reco_its_data():
    """Generate ITS-only dataset (10 events)."""
    return generate_mc_reco_its_only(n_events=10, seed=42)


# ============================================================================
# F-ext-01: Base invariance — bit-exact match with generate_event_display
# ============================================================================

def test_mc_reco_base_invariance(mc_reco_data):
    """Base 3 tables must be bit-identical to generate_event_display()."""
    events_base, tracks_base, clusters_base = generate_event_display(
        n_events=100, seed=42
    )
    assert mc_reco_data["events"].equals(events_base), "events not bit-identical"
    assert mc_reco_data["tracks"].equals(tracks_base), "tracks not bit-identical"
    assert mc_reco_data["clusters"].equals(clusters_base), "clusters not bit-identical"


# ============================================================================
# F-ext-02: reco_tracks schema
# ============================================================================

def test_reco_tracks_schema(mc_reco_data):
    """Verify reco_tracks column names, dtypes, and FK integrity."""
    reco = mc_reco_data["reco_tracks"]
    expected_cols = [
        'reco_track_id', 'event_id', 'mc_track_idx',
        'pt_reco', 'tgl_reco', 'phi_reco',
        'rms_y', 'rms_z', 'n_cls_fit', 'chi2_per_cls',
    ]
    assert list(reco.columns) == expected_cols

    # Dense PK
    assert (reco['reco_track_id'].values == np.arange(len(reco))).all()

    # FK integrity: event_id exists in events
    assert set(reco['event_id']).issubset(set(mc_reco_data["events"]['event_id']))

    # FK integrity: mc_track_idx in valid range
    n_tracks = len(mc_reco_data["tracks"])
    assert (reco['mc_track_idx'] >= 0).all()
    assert (reco['mc_track_idx'] < n_tracks).all()

    # Dtypes
    assert reco['reco_track_id'].dtype == np.int64
    assert reco['event_id'].dtype == np.int64
    assert reco['mc_track_idx'].dtype == np.int64
    assert reco['n_cls_fit'].dtype == np.int64
    assert reco['pt_reco'].dtype == np.float64


# ============================================================================
# F-ext-03: mc_labels schema
# ============================================================================

def test_mc_labels_schema(mc_reco_data):
    """Verify mc_labels column names, dtypes, FK integrity (both directions)."""
    labels = mc_reco_data["mc_labels"]
    reco = mc_reco_data["reco_tracks"]
    tracks = mc_reco_data["tracks"]

    expected_cols = ['reco_track_id', 'mc_track_idx', 'weight']
    assert list(labels.columns) == expected_cols

    # FK → reco_tracks
    assert set(labels['reco_track_id']).issubset(set(reco['reco_track_id']))

    # FK → tracks (dense row index)
    assert (labels['mc_track_idx'] >= 0).all()
    assert (labels['mc_track_idx'] < len(tracks)).all()

    # Weight is 1.0 for all (current model)
    assert (labels['weight'] == 1.0).all()

    # 1:1 mapping: same count as reco_tracks
    assert len(labels) == len(reco)

    # Consistency: mc_track_idx matches between reco_tracks and mc_labels
    np.testing.assert_array_equal(
        labels['mc_track_idx'].values,
        reco['mc_track_idx'].values,
    )


# ============================================================================
# F-ext-04: Efficiency subset
# ============================================================================

def test_efficiency_subset(mc_reco_data):
    """reco_tracks is a proper subset of tracks."""
    n_tracks = len(mc_reco_data["tracks"])
    n_reco = len(mc_reco_data["reco_tracks"])

    # Strict subset
    assert n_reco < n_tracks, "Efficiency should remove some tracks"
    assert n_reco > 0, "At least some tracks should be reconstructed"

    # All mc_track_idx are unique (no duplicates — 1:1 mapping)
    mc_idx = mc_reco_data["reco_tracks"]['mc_track_idx'].values
    assert len(np.unique(mc_idx)) == len(mc_idx), "Duplicate mc_track_idx"


# ============================================================================
# F-ext-05: Efficiency pT dependence
# ============================================================================

def test_efficiency_pt_dependence(mc_reco_data):
    """Low-pT tracks have lower acceptance rate than high-pT tracks."""
    tracks = mc_reco_data["tracks"]
    reco = mc_reco_data["reco_tracks"]

    pt_all = tracks['pt'].values
    reco_mask = np.zeros(len(tracks), dtype=bool)
    reco_mask[reco['mc_track_idx'].values] = True

    # Split at median pT
    median_pt = np.median(pt_all)
    low_mask = pt_all < median_pt
    high_mask = pt_all >= median_pt

    eff_low = reco_mask[low_mask].mean()
    eff_high = reco_mask[high_mask].mean()

    assert eff_low < eff_high, (
        f"Low-pT efficiency ({eff_low:.3f}) should be < high-pT ({eff_high:.3f})"
    )


# ============================================================================
# F-ext-06: pT fit accuracy (full tracks, pT > 0.5 GeV)
# ============================================================================

def test_fit_pt_accuracy(mc_reco_data):
    """pt_reco within 5% of pt_mc for ≥95% of full tracks with pT > 0.5."""
    reco = mc_reco_data["reco_tracks"]
    tracks = mc_reco_data["tracks"]

    mc_idx = reco['mc_track_idx'].values
    pt_mc = tracks['pt'].values[mc_idx]
    pt_reco = reco['pt_reco'].values

    # Full tracks only (57 layers), pT > 0.5 GeV
    full_mask = reco['n_cls_fit'].values == len(LAYERS_ALL)
    pt_mask = pt_mc > 0.5
    mask = full_mask & pt_mask & np.isfinite(pt_reco)

    if mask.sum() < 10:
        pytest.skip("Not enough full tracks with pT > 0.5")

    rel_err = np.abs(pt_reco[mask] - pt_mc[mask]) / pt_mc[mask]
    frac_within_5pct = np.mean(rel_err < 0.05)

    assert frac_within_5pct >= 0.95, (
        f"Only {frac_within_5pct*100:.1f}% of full tracks have |Δpt/pt| < 5% "
        f"(need ≥95%). Median rel error: {np.median(rel_err)*100:.2f}%"
    )


# ============================================================================
# F-ext-07: tgl fit accuracy
# ============================================================================

def test_fit_tgl_accuracy(mc_reco_data):
    """|tgl_reco - tgl_mc| / (1 + |tgl_mc|) < 5% for ≥95% of tracks."""
    reco = mc_reco_data["reco_tracks"]
    tracks = mc_reco_data["tracks"]

    mc_idx = reco['mc_track_idx'].values
    eta_mc = tracks['eta'].values[mc_idx]
    # tgl_mc = dz/dr. From eta: theta = 2*arctan(exp(-eta)), tgl = 1/tan(theta)
    theta_mc = 2.0 * np.arctan(np.exp(-eta_mc))
    tgl_mc = np.where(
        np.abs(np.tan(theta_mc)) > 1e-10,
        1.0 / np.tan(theta_mc),
        1e10
    )
    tgl_reco = reco['tgl_reco'].values

    # Exclude extreme eta tracks
    valid = np.abs(eta_mc) < 0.95
    if valid.sum() < 10:
        pytest.skip("Not enough tracks with |eta| < 0.95")

    rel_err = np.abs(tgl_reco[valid] - tgl_mc[valid]) / (1.0 + np.abs(tgl_mc[valid]))
    frac_ok = np.mean(rel_err < 0.05)

    assert frac_ok >= 0.95, (
        f"Only {frac_ok*100:.1f}% of tracks have tgl within 5%. "
        f"Median: {np.median(rel_err)*100:.2f}%"
    )


# ============================================================================
# F-ext-08: ITS-only linear fit produces valid results
# ============================================================================

def test_fit_linear_its_only(mc_reco_its_data):
    """ITS-only tracks use linear fit, produce finite pt_reco and tgl_reco."""
    reco = mc_reco_its_data["reco_tracks"]

    assert len(reco) > 0, "Should have some reconstructed tracks"
    assert (reco['n_cls_fit'] == len(LAYERS_ITS)).all(), "All should be ITS-only"
    assert np.all(np.isfinite(reco['pt_reco'])), "pt_reco should be finite"
    assert np.all(np.isfinite(reco['tgl_reco'])), "tgl_reco should be finite"
    assert np.all(reco['pt_reco'] > 0), "pt_reco should be positive"


# ============================================================================
# F-ext-09: RMS positive
# ============================================================================

def test_rms_positive(mc_reco_data):
    """All rms_y > 0 and rms_z > 0."""
    reco = mc_reco_data["reco_tracks"]
    assert (reco['rms_y'] > 0).all(), "rms_y must be positive"
    assert (reco['rms_z'] > 0).all(), "rms_z must be positive"


# ============================================================================
# F-ext-10: ITS-only tracks have smaller RMS than full tracks
# ============================================================================

def test_rms_its_vs_full():
    """ITS-only tracks have smaller rms_y than full (ITS+TPC) tracks.

    ITS resolution (10μm) is better than TPC (100μm). Full tracks include
    TPC clusters which dominate the RMS.
    """
    its_tables = generate_mc_reco_its_only(n_events=50, seed=42)
    full_tables = generate_mc_reco(n_events=50, seed=42)

    rms_its = its_tables["reco_tracks"]['rms_y'].median()
    rms_full = full_tables["reco_tracks"]['rms_y'].median()

    assert rms_its < rms_full, (
        f"ITS-only median rms_y ({rms_its:.6f}) should be < "
        f"full track ({rms_full:.6f})"
    )


# ============================================================================
# F-ext-11: Reproducibility
# ============================================================================

def test_reproducibility_extended():
    """Same seed → bit-exact tables."""
    t1 = generate_mc_reco(n_events=20, seed=99)
    t2 = generate_mc_reco(n_events=20, seed=99)

    for name in ("events", "tracks", "clusters", "reco_tracks", "mc_labels"):
        assert t1[name].equals(t2[name]), f"{name} not reproducible"


# ============================================================================
# F-ext-12: Dense FK indices
# ============================================================================

def test_dense_fk(mc_reco_data):
    """All dense FK columns are in valid range [0, parent_len-1]."""
    reco = mc_reco_data["reco_tracks"]
    labels = mc_reco_data["mc_labels"]
    n_tracks = len(mc_reco_data["tracks"])
    n_events = len(mc_reco_data["events"])
    n_reco = len(reco)

    # reco_tracks.mc_track_idx → tracks
    assert (reco['mc_track_idx'] >= 0).all()
    assert (reco['mc_track_idx'] < n_tracks).all()

    # reco_tracks.event_id → events (events.event_id is already 0..N-1)
    assert (reco['event_id'] >= 0).all()
    assert (reco['event_id'] < n_events).all()

    # mc_labels.reco_track_id → reco_tracks
    assert (labels['reco_track_id'] >= 0).all()
    assert (labels['reco_track_id'] < n_reco).all()

    # mc_labels.mc_track_idx → tracks
    assert (labels['mc_track_idx'] >= 0).all()
    assert (labels['mc_track_idx'] < n_tracks).all()


# ============================================================================
# F-ext-13: Save/load roundtrip
# ============================================================================

def test_save_load_extended(mc_reco_data):
    """Parquet roundtrip for all 5 tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_tables_extended(mc_reco_data, tmpdir)
        loaded = load_tables_extended(tmpdir)

        for name in ("events", "tracks", "clusters", "reco_tracks", "mc_labels"):
            assert name in loaded, f"Missing table: {name}"
            pd.testing.assert_frame_equal(
                mc_reco_data[name], loaded[name],
                check_exact=True,
                obj=name,
            )


# ============================================================================
# F-ext-14: MC tracks missing from reco (efficiency holes)
# ============================================================================

def test_mc_missing_reco(mc_reco_data):
    """Some MC tracks have no entry in mc_labels (efficiency holes exist)."""
    n_tracks = len(mc_reco_data["tracks"])
    mc_idx_in_labels = set(mc_reco_data["mc_labels"]['mc_track_idx'].values)

    all_mc = set(range(n_tracks))
    missing = all_mc - mc_idx_in_labels

    assert len(missing) > 0, "All MC tracks have reco — efficiency should create holes"


# ============================================================================
# F-ext-15: build_dense_index utility
# ============================================================================

def test_build_dense_index(mc_reco_data):
    """build_dense_index correctly maps composite keys to row indices."""
    tracks = mc_reco_data["tracks"]
    clusters = mc_reco_data["clusters"]
    events = mc_reco_data["events"]

    # clusters → tracks
    cl_track_idx = build_dense_index(tracks, clusters, ['event_id', 'track_id'])
    assert len(cl_track_idx) == len(clusters)
    assert cl_track_idx.min() >= 0
    assert cl_track_idx.max() < len(tracks)

    # Verify: for each cluster, the mapped track has matching event_id and track_id
    for i in range(min(100, len(clusters))):
        row = clusters.iloc[i]
        trk = tracks.iloc[cl_track_idx[i]]
        assert row['event_id'] == trk['event_id'], f"event_id mismatch at cluster {i}"
        assert row['track_id'] == trk['track_id'], f"track_id mismatch at cluster {i}"

    # tracks → events
    tr_event_idx = build_dense_index(events, tracks, ['event_id'])
    assert len(tr_event_idx) == len(tracks)
    assert tr_event_idx.min() >= 0
    assert tr_event_idx.max() < len(events)

    # Since events.event_id == np.arange(n_events), tr_event_idx should equal event_id
    np.testing.assert_array_equal(
        tr_event_idx,
        tracks['event_id'].values,
    )
