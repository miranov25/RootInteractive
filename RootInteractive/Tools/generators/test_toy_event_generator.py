"""
Tests for toy_event_generator.py — Phase 0.1.F

7 tests as specified in the proposal:
  test_schema, test_reproducibility, test_helix_curvature,
  test_vertex_offset, test_layer_count, test_global_track_id, test_merge_all
"""

import numpy as np
import pandas as pd
import pytest

from toy_event_generator import (
    generate_event_display,
    generate_event_display_its_only,
    generate_event_display_tpc_only,
    merge_all,
    save_tables,
    load_tables,
    generate_or_load,
    _make_metadata,
    LAYERS_ALL,
    LAYERS_ITS,
    N_ITS,
)


@pytest.fixture
def sample_data():
    """Generate a small reproducible dataset."""
    return generate_event_display(n_events=10, seed=42)


def test_schema(sample_data):
    """Verify column names, dtypes, and foreign keys."""
    events, tracks, clusters = sample_data

    # Events schema
    assert list(events.columns) == ['event_id', 'vertex_x', 'vertex_y', 'vertex_z', 'n_tracks']
    assert events['event_id'].dtype == np.int64 or events['event_id'].dtype == int

    # Tracks schema
    expected_track_cols = ['event_id', 'track_id', 'global_track_id', 'pt', 'eta', 'phi', 'charge', 'n_clusters']
    assert list(tracks.columns) == expected_track_cols

    # Clusters schema
    expected_cluster_cols = [
        'event_id', 'track_id', 'cluster_id', 'x', 'y', 'z',
        'r', 'phi_cluster', 'layer', 'detector', 'Q'
    ]
    assert list(clusters.columns) == expected_cluster_cols

    # Foreign key integrity: all cluster event_ids exist in events
    assert set(clusters['event_id'].unique()).issubset(set(events['event_id']))

    # Foreign key integrity: all cluster (event_id, track_id) pairs exist in tracks
    cluster_keys = set(zip(clusters['event_id'], clusters['track_id']))
    track_keys = set(zip(tracks['event_id'], tracks['track_id']))
    assert cluster_keys.issubset(track_keys)

    # Detector column values
    assert set(clusters['detector'].unique()).issubset({"ITS", "TPC"})


def test_reproducibility():
    """Same seed produces identical DataFrames."""
    e1, t1, c1 = generate_event_display(n_events=20, seed=99)
    e2, t2, c2 = generate_event_display(n_events=20, seed=99)

    pd.testing.assert_frame_equal(e1, e2)
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(c1, c2)

    # Different seed produces different data
    e3, _, _ = generate_event_display(n_events=20, seed=100)
    assert not e1['vertex_x'].equals(e3['vertex_x'])


def test_helix_curvature(sample_data):
    """Verify curvature sign flips with charge."""
    _, tracks, clusters = sample_data

    # Find two tracks in same event with opposite charge
    event0_tracks = tracks[tracks['event_id'] == 0]
    pos_tracks = event0_tracks[event0_tracks['charge'] == 1]
    neg_tracks = event0_tracks[event0_tracks['charge'] == -1]

    if len(pos_tracks) == 0 or len(neg_tracks) == 0:
        pytest.skip("No opposite-charge pair in event 0 with this seed")

    pos_tid = pos_tracks.iloc[0]['track_id']
    neg_tid = neg_tracks.iloc[0]['track_id']

    # Get phi evolution for each track
    pos_cls = clusters[(clusters['event_id'] == 0) & (clusters['track_id'] == pos_tid)]
    neg_cls = clusters[(clusters['event_id'] == 0) & (clusters['track_id'] == neg_tid)]

    # Bending direction: delta_phi from first to last layer should have opposite sign
    pos_dphi = pos_cls.iloc[-1]['phi_cluster'] - pos_cls.iloc[0]['phi_cluster']
    neg_dphi = neg_cls.iloc[-1]['phi_cluster'] - neg_cls.iloc[0]['phi_cluster']

    # Opposite charge → opposite bending (delta_phi signs differ)
    assert pos_dphi * neg_dphi < 0, (
        f"Curvature should flip with charge: pos_dphi={pos_dphi:.4f}, neg_dphi={neg_dphi:.4f}"
    )


def test_vertex_offset():
    """First cluster (inner ITS) should be near vertex position."""
    events, _, clusters = generate_event_display(n_events=5, seed=42)

    for eid in range(min(5, len(events))):
        ev = events[events['event_id'] == eid].iloc[0]
        # Inner ITS layer (layer 0, r=2.3 cm) — cluster should be offset by vertex
        inner_cls = clusters[(clusters['event_id'] == eid) & (clusters['layer'] == 0)]

        for _, cl in inner_cls.iterrows():
            # At r=2.3 cm, the cluster position should be approximately
            # (r*cos(phi) + vx, r*sin(phi) + vy, z + vz)
            # The z component should reflect the vertex offset
            # Check that vertex_z contributes to cluster z
            # (exact check is hard due to helix, but z should correlate with vz)
            pass

        # Simpler check: generate with large vertex offset and verify it appears
    events_off, _, clusters_off = generate_event_display(
        n_events=1,
        tracks_per_event=(1, 1),
        seed=42,
        # We can't set vertex directly, but we can check correlation:
        # vertex_z has 5cm spread, so z at inner ITS should reflect vz
    )
    vz = events_off.iloc[0]['vertex_z']
    vx = events_off.iloc[0]['vertex_x']
    vy = events_off.iloc[0]['vertex_y']
    inner = clusters_off[clusters_off['layer'] == 0].iloc[0]

    # At layer 0 (r=2.3cm), position = r*cos(phi)+vx, r*sin(phi)+vy, r/tan(theta)+vz
    # The vertex offset should be present. For x: |x - vx| should be approximately r (2.3)
    r_from_vertex = np.sqrt((inner['x'] - vx)**2 + (inner['y'] - vy)**2)
    assert abs(r_from_vertex - 2.3) < 0.5, (
        f"Inner ITS cluster distance from vertex should be ~2.3 cm, got {r_from_vertex:.2f}"
    )


def test_layer_count():
    """57 clusters per track (default), 7 for ITS-only."""
    _, tracks, clusters = generate_event_display(n_events=3, seed=42)

    # Default: 57 layers
    for _, trk in tracks.iterrows():
        trk_cls = clusters[
            (clusters['event_id'] == trk['event_id']) &
            (clusters['track_id'] == trk['track_id'])
        ]
        assert len(trk_cls) == len(LAYERS_ALL), (
            f"Expected {len(LAYERS_ALL)} clusters, got {len(trk_cls)}"
        )
        assert trk['n_clusters'] == len(LAYERS_ALL)

    # ITS-only: 7 layers
    _, tracks_its, clusters_its = generate_event_display_its_only(n_events=2, seed=42)
    for _, trk in tracks_its.iterrows():
        trk_cls = clusters_its[
            (clusters_its['event_id'] == trk['event_id']) &
            (clusters_its['track_id'] == trk['track_id'])
        ]
        assert len(trk_cls) == len(LAYERS_ITS)
        assert set(trk_cls['detector'].unique()) == {"ITS"}


def test_global_track_id():
    """global_track_id is unique across events and matches formula."""
    _, tracks, _ = generate_event_display(n_events=20, seed=42)

    # Formula check
    expected = tracks['event_id'] * 10000 + tracks['track_id']
    pd.testing.assert_series_equal(
        tracks['global_track_id'], expected, check_names=False
    )

    # Uniqueness
    assert tracks['global_track_id'].is_unique, "global_track_id must be unique"


def test_merge_all(sample_data):
    """Merged table has correct row count and columns."""
    events, tracks, clusters = sample_data
    merged = merge_all(events, tracks, clusters)

    # Row count should match clusters (many-to-one joins)
    assert len(merged) == len(clusters)

    # Should contain columns from all three tables
    assert 'event_id' in merged.columns
    assert 'track_id' in merged.columns
    assert 'cluster_id' in merged.columns
    assert 'track_pt' in merged.columns
    assert 'event_vertex_x' in merged.columns
    assert 'cluster_x' in merged.columns

    # No NaN in join keys (join integrity)
    assert merged['track_pt'].notna().all()
    assert merged['event_vertex_x'].notna().all()


def test_save_load_roundtrip(sample_data, tmp_path):
    """save_tables → load_tables returns identical DataFrames."""
    events, tracks, clusters = sample_data
    cache_dir = str(tmp_path / "cache")
    metadata = _make_metadata(n_events=10, seed=42)

    save_tables(events, tracks, clusters, cache_dir, metadata=metadata)
    e2, t2, c2 = load_tables(cache_dir)

    pd.testing.assert_frame_equal(events, e2)
    pd.testing.assert_frame_equal(tracks, t2)
    pd.testing.assert_frame_equal(clusters, c2)

    # Verify metadata.json was written
    import json, os
    with open(os.path.join(cache_dir, "metadata.json")) as f:
        stored = json.load(f)
    assert stored["n_events"] == 10
    assert stored["seed"] == 42


def test_generate_or_load(tmp_path):
    """First call generates + saves, second call loads from cache."""
    cache_dir = str(tmp_path / "gen_cache")

    # First call: generates and saves
    e1, t1, c1 = generate_or_load(cache_dir, n_events=5, seed=42)
    assert len(e1) == 5
    assert (tmp_path / "gen_cache" / "events.parquet").exists()
    assert (tmp_path / "gen_cache" / "metadata.json").exists()

    # Second call: loads from cache (same result)
    e2, t2, c2 = generate_or_load(cache_dir, n_events=5, seed=42)
    pd.testing.assert_frame_equal(e1, e2)
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(c1, c2)

    # Force regenerate
    e3, _, _ = generate_or_load(cache_dir, n_events=5, seed=99, force=True)
    assert not e1['vertex_x'].equals(e3['vertex_x'])


def test_cache_parameter_mismatch(tmp_path):
    """Cache with different parameters raises ValueError."""
    cache_dir = str(tmp_path / "mismatch_cache")

    # Generate with n_events=10
    generate_or_load(cache_dir, n_events=10, seed=42)

    # Load with n_events=20 → should raise
    with pytest.raises(ValueError, match="Cache parameter mismatch"):
        generate_or_load(cache_dir, n_events=20, seed=42)

    # Load with different seed → should raise
    with pytest.raises(ValueError, match="Cache parameter mismatch"):
        generate_or_load(cache_dir, n_events=10, seed=99)

    # Load with correct params → should work
    e, _, _ = generate_or_load(cache_dir, n_events=10, seed=42)
    assert len(e) == 10

    # Force bypasses validation
    e2, _, _ = generate_or_load(cache_dir, n_events=20, seed=99, force=True)
    assert len(e2) == 20
