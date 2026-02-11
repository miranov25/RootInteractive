"""
Toy Event Display Data Generator for RootInteractive.

Generates 3 flat pandas DataFrames with realistic helix track topology
from the ALICE detector for testing CDSJoin and multi-table workflows.

NOTE: This implements simplified circular bending for visualization purposes.
It does not implement full helix transport with proper vertex-to-layer propagation.
Simplifications: no multiple scattering, no energy loss, uniform B field, no material
budget. Sufficient for testing CDSJoin and multi-table workflows — not for physics analysis.

Usage:
    events_df, tracks_df, clusters_df = generate_event_display(n_events=100)

Phase: 0.1.F
"""

import json
import numpy as np
import pandas as pd
from typing import Tuple, Optional

__version__ = "0.1.0"

# ============================================================================
# Detector geometry (ALICE-realistic radii)
# ============================================================================

# ITS (Inner Tracking System): 7 layers [cm]
LAYERS_ITS = np.array([2.3, 3.1, 3.9, 7.6, 12.0, 18.0, 24.0])

# TPC (Time Projection Chamber): 50 layers [cm]
LAYERS_TPC = np.linspace(85, 250, 50)

# Combined: 57 layers total
LAYERS_ALL = np.concatenate([LAYERS_ITS, LAYERS_TPC])

# Layer counts
N_ITS = len(LAYERS_ITS)
N_TPC = len(LAYERS_TPC)


# ============================================================================
# Physics model
# ============================================================================

# Reference implementation — production path is vectorized in generate_event_display()
def helix_position(
    pt: float,
    eta: float,
    phi: float,
    charge: int,
    r: float,
    b_field: float = 0.5,
    vertex_x: float = 0.0,
    vertex_y: float = 0.0,
    vertex_z: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Calculate (x, y, z) position on helix at detector radius r.

    This is a simplified circular bending approximation for visualization.
    It does NOT implement full helix transport — no multiple scattering,
    no energy loss, uniform B field, no material budget.

    Parameters
    ----------
    pt : float
        Transverse momentum [GeV/c]
    eta : float
        Pseudorapidity
    phi : float
        Initial azimuthal angle [rad]
    charge : int
        Electric charge (+1 or -1)
    r : float
        Target detector layer radius [cm]
    b_field : float
        Magnetic field [Tesla]
    vertex_x, vertex_y, vertex_z : float
        Collision vertex position [cm]

    Returns
    -------
    x, y, z : float
        Position coordinates [cm], with vertex offset applied.
    """
    # Helix radius: pT/(0.3*B) gives R in meters, *100 converts to cm
    R_helix = pt / (0.3 * b_field * abs(charge)) * 100  # [cm]

    # Bending angle at radius r (clamp for very low pT)
    sin_arg = r / (2 * R_helix)
    if abs(sin_arg) > 1:
        sin_arg = np.sign(sin_arg)

    delta_phi = charge * np.arcsin(sin_arg)

    # Position in transverse plane
    phi_at_r = phi + delta_phi
    x = r * np.cos(phi_at_r) + vertex_x
    y = r * np.sin(phi_at_r) + vertex_y

    # Z from pseudorapidity
    theta = 2 * np.arctan(np.exp(-eta))
    z = (r / np.tan(theta) + vertex_z) if abs(np.tan(theta)) > 1e-10 else vertex_z

    return x, y, z


# ============================================================================
# Generators
# ============================================================================

def generate_event_display(
    n_events: int = 100,
    tracks_per_event: Tuple[int, int] = (3, 10),
    pt_range: Tuple[float, float] = (0.3, 5.0),
    eta_range: Tuple[float, float] = (-1.0, 1.0),
    b_field: float = 0.5,
    layers: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate toy event display data with helix tracks.

    Parameters
    ----------
    n_events : int
        Number of events to generate.
    tracks_per_event : tuple of (int, int)
        (min, max) tracks per event.
    pt_range : tuple of (float, float)
        (min, max) transverse momentum [GeV/c].
    eta_range : tuple of (float, float)
        (min, max) pseudorapidity.
    b_field : float
        Magnetic field [Tesla].
    layers : array, optional
        Detector layer radii [cm]. Default: ITS + TPC (57 layers).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    events_df : DataFrame
        Event-level data (event_id, vertex, n_tracks).
    tracks_df : DataFrame
        Track-level data (event_id, track_id, global_track_id, pt, eta, phi, charge).
    clusters_df : DataFrame
        Cluster-level data (event_id, track_id, cluster_id, x, y, z, r, ...).

    Example
    -------
    >>> events, tracks, clusters = generate_event_display(n_events=10, seed=42)
    >>> len(events)
    10
    """
    rng = np.random.default_rng(seed)

    if layers is None:
        layers = LAYERS_ALL
    n_layers = len(layers)

    # --- Events (vectorized) ---
    vx = rng.normal(0, 0.01, n_events)    # 100 um spread
    vy = rng.normal(0, 0.01, n_events)
    vz = rng.normal(0, 5.0, n_events)     # 5 cm spread in z
    n_tracks_arr = rng.integers(tracks_per_event[0], tracks_per_event[1] + 1, n_events)

    events_df = pd.DataFrame({
        'event_id': np.arange(n_events, dtype=np.int64),
        'vertex_x': vx,
        'vertex_y': vy,
        'vertex_z': vz,
        'n_tracks': n_tracks_arr.astype(np.int64),
    })

    # --- Tracks (vectorized) ---
    total_tracks = int(n_tracks_arr.sum())
    trk_event_id = np.repeat(np.arange(n_events), n_tracks_arr)
    trk_track_id = np.concatenate([np.arange(n) for n in n_tracks_arr])
    trk_pt = rng.uniform(pt_range[0], pt_range[1], total_tracks)
    trk_eta = rng.uniform(eta_range[0], eta_range[1], total_tracks)
    trk_phi = rng.uniform(-np.pi, np.pi, total_tracks)
    trk_charge = rng.choice([-1, 1], total_tracks)

    tracks_df = pd.DataFrame({
        'event_id': trk_event_id.astype(np.int64),
        'track_id': trk_track_id.astype(np.int64),
        'global_track_id': (trk_event_id * 10000 + trk_track_id).astype(np.int64),
        'pt': trk_pt,
        'eta': trk_eta,
        'phi': trk_phi,
        'charge': trk_charge.astype(np.int64),
        'n_clusters': np.full(total_tracks, n_layers, dtype=np.int64),
    })

    # --- Clusters (vectorized over layers for each track) ---
    # Expand track arrays: each track × n_layers
    total_clusters = total_tracks * n_layers
    cl_trk_idx = np.repeat(np.arange(total_tracks), n_layers)
    cl_layer = np.tile(np.arange(n_layers), total_tracks)
    cl_r_layer = np.tile(layers, total_tracks)

    cl_pt = trk_pt[cl_trk_idx]
    cl_eta = trk_eta[cl_trk_idx]
    cl_phi = trk_phi[cl_trk_idx]
    cl_charge = trk_charge[cl_trk_idx]
    cl_event_id = trk_event_id[cl_trk_idx]
    cl_track_id = trk_track_id[cl_trk_idx]

    # Vertex offsets per cluster
    cl_vx = vx[cl_event_id]
    cl_vy = vy[cl_event_id]
    cl_vz = vz[cl_event_id]

    # Helix computation (vectorized)
    # R_helix [cm] = pT [GeV] / (0.3 * B [T]) * 100
    R_helix = cl_pt / (0.3 * b_field * np.abs(cl_charge)) * 100.0
    sin_arg = np.clip(cl_r_layer / (2.0 * R_helix), -1.0, 1.0)
    delta_phi = cl_charge * np.arcsin(sin_arg)
    phi_at_r = cl_phi + delta_phi

    cl_x = cl_r_layer * np.cos(phi_at_r) + cl_vx
    cl_y = cl_r_layer * np.sin(phi_at_r) + cl_vy

    theta = 2.0 * np.arctan(np.exp(-cl_eta))
    tan_theta = np.tan(theta)
    cl_z = np.where(np.abs(tan_theta) > 1e-10, cl_r_layer / tan_theta + cl_vz, cl_vz)

    cl_r = np.sqrt(cl_x**2 + cl_y**2)
    cl_phi_cluster = np.arctan2(cl_y, cl_x)

    cl_detector = np.where(cl_layer < N_ITS, "ITS", "TPC")  # String for readability; future: pd.Categorical
    cl_Q = rng.gamma(2.0, 1.0, total_clusters) * 100.0

    clusters_df = pd.DataFrame({
        'event_id': cl_event_id.astype(np.int64),
        'track_id': cl_track_id.astype(np.int64),
        'cluster_id': cl_layer.astype(np.int64),
        'x': cl_x,
        'y': cl_y,
        'z': cl_z,
        'r': cl_r,
        'phi_cluster': cl_phi_cluster,
        'layer': cl_layer.astype(np.int64),
        'detector': cl_detector,
        'Q': cl_Q,
    })

    return events_df, tracks_df, clusters_df


def generate_event_display_its_only(n_events: int = 100, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate with ITS layers only (7 layers) — fast for unit tests."""
    return generate_event_display(n_events=n_events, layers=LAYERS_ITS, **kwargs)


def generate_event_display_tpc_only(n_events: int = 100, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate with TPC layers only (50 layers)."""
    return generate_event_display(n_events=n_events, layers=LAYERS_TPC, **kwargs)


# ============================================================================
# Utility functions
# ============================================================================

def add_derived_columns(clusters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add commonly used derived columns for visualization.

    Adds: phi_deg, theta, r_xy.
    """
    df = clusters_df.copy()
    df['phi_deg'] = np.degrees(df['phi_cluster'])
    df['theta'] = np.arctan2(df['r'], df['z'])
    df['r_xy'] = df['r']
    return df


def merge_all(
    events_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all tables into single flat DataFrame.

    Column names prefixed: event_, track_, cluster_.
    Uses sort=False to preserve deterministic ordering.
    """
    events = events_df.add_prefix('event_').rename(columns={'event_event_id': 'event_id'})
    tracks = tracks_df.add_prefix('track_').rename(columns={
        'track_event_id': 'event_id',
        'track_track_id': 'track_id',
    })
    clusters = clusters_df.add_prefix('cluster_').rename(columns={
        'cluster_event_id': 'event_id',
        'cluster_track_id': 'track_id',
        'cluster_cluster_id': 'cluster_id',
    })

    merged = clusters.merge(tracks, on=['event_id', 'track_id'], how='left', sort=False)
    merged = merged.merge(events, on='event_id', how='left', sort=False)

    return merged


# ============================================================================
# Parquet cache (generate once, load fast)
# ============================================================================

_DEFAULT_PARAMS = {
    "n_events": 100,
    "tracks_per_event": (3, 10),
    "pt_range": (0.3, 5.0),
    "eta_range": (-1.0, 1.0),
    "b_field": 0.5,
    "seed": 42,
}


def _make_metadata(n_events, **kwargs):
    """Build metadata dict from generation parameters."""
    params = {**_DEFAULT_PARAMS, "n_events": n_events, **kwargs}
    # Convert numpy arrays / tuples to JSON-serializable lists
    if "layers" in params and params["layers"] is not None:
        params["layers"] = list(float(x) for x in params["layers"])
    else:
        params["layers"] = None
    for key in ("tracks_per_event", "pt_range", "eta_range"):
        if key in params:
            params[key] = list(params[key])
    params["generator_version"] = __version__
    return params


def save_tables(
    events_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    path: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save 3 DataFrames as parquet files in directory.

    Creates:
        {path}/events.parquet
        {path}/tracks.parquet
        {path}/clusters.parquet
        {path}/metadata.json  (if metadata provided)
    """
    import os
    os.makedirs(path, exist_ok=True)
    events_df.to_parquet(os.path.join(path, "events.parquet"))
    tracks_df.to_parquet(os.path.join(path, "tracks.parquet"))
    clusters_df.to_parquet(os.path.join(path, "clusters.parquet"))
    if metadata is not None:
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)


def load_tables(
    path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load cached tables from parquet directory.

    Returns (events_df, tracks_df, clusters_df).
    Raises FileNotFoundError if cache directory or files missing.
    """
    import os
    events_df = pd.read_parquet(os.path.join(path, "events.parquet"))
    tracks_df = pd.read_parquet(os.path.join(path, "tracks.parquet"))
    clusters_df = pd.read_parquet(os.path.join(path, "clusters.parquet"))
    return events_df, tracks_df, clusters_df


def generate_or_load(
    path: str,
    n_events: int = 100,
    force: bool = False,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate and cache, or load from existing cache.

    If cache exists and force=False, validates stored metadata against
    requested parameters. Raises ValueError on mismatch (use force=True
    to regenerate). Missing metadata (legacy cache) triggers a warning
    but loads anyway.
    """
    import os
    import warnings
    cache_file = os.path.join(path, "events.parquet")
    meta_file = os.path.join(path, "metadata.json")
    requested = _make_metadata(n_events, **kwargs)

    if os.path.exists(cache_file) and not force:
        # Validate metadata
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                stored = json.load(f)
            # Compare all params except generator_version
            mismatches = []
            for key in requested:
                if key == "generator_version":
                    continue
                if key in stored and stored[key] != requested[key]:
                    mismatches.append(f"  {key}: cached={stored[key]}, requested={requested[key]}")
            if mismatches:
                raise ValueError(
                    f"Cache parameter mismatch in '{path}':\n"
                    + "\n".join(mismatches)
                    + "\nUse force=True to regenerate."
                )
        else:
            warnings.warn(
                f"Cache at '{path}' has no metadata.json (legacy cache). "
                "Loading without validation. Re-generate with force=True to add metadata.",
                stacklevel=2,
            )
        return load_tables(path)

    events, tracks, clusters = generate_event_display(n_events=n_events, **kwargs)
    save_tables(events, tracks, clusters, path, metadata=requested)
    return events, tracks, clusters


# ============================================================================
# Main (demo / smoke test)
# ============================================================================

if __name__ == "__main__":
    import time

    print("Generating toy event display data...")

    t0 = time.perf_counter()
    events, tracks, clusters = generate_event_display(
        n_events=100,
        tracks_per_event=(3, 8),
        seed=42,
    )
    t1 = time.perf_counter()

    print(f"\nGenerated in {t1 - t0:.3f}s:")
    print(f"  Events:   {len(events):,} rows")
    print(f"  Tracks:   {len(tracks):,} rows")
    print(f"  Clusters: {len(clusters):,} rows")

    print(f"\nEvents columns:   {list(events.columns)}")
    print(f"Tracks columns:   {list(tracks.columns)}")
    print(f"Clusters columns: {list(clusters.columns)}")

    print(f"\nSample clusters (first 3):")
    print(clusters.head(3).to_string())

    # Helix curvature check
    t0_cls = clusters[(clusters.event_id == 0) & (clusters.track_id == 0)]
    print(f"\nTrack 0 curvature check:")
    print(f"  phi range: {t0_cls.phi_cluster.min():.4f} to {t0_cls.phi_cluster.max():.4f} rad")
    print(f"  (Curved track should show phi variation)")

    # 1K benchmark
    t0 = time.perf_counter()
    generate_event_display(n_events=1000, seed=123)
    t1 = time.perf_counter()
    print(f"\n1K events benchmark: {t1 - t0:.2f}s (target: < 1s)")
