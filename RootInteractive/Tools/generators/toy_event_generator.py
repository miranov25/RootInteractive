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

__version__ = "0.2.0"

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


def save_to_root(
    events_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    filename: str,
) -> None:
    """
    Export as ROOT file with 4 TTrees using uproot.

    Trees:
        "events"   — event-level (event_id, vertex, n_tracks)
        "tracks"   — track-level (event_id, track_id, pt, eta, phi, charge)
        "clusters" — cluster-level (event_id, track_id, cluster_id, x, y, z, ...)
        "flat"     — fully merged (all columns from all 3 tables)

    String columns (e.g. 'detector') are replaced by detector_id (0=ITS, 1=TPC).
    Requires: pip install uproot
    """
    import uproot

    with uproot.recreate(filename) as f:
        # Events
        f["events"] = {col: events_df[col].values for col in events_df.columns}

        # Tracks
        f["tracks"] = {col: tracks_df[col].values for col in tracks_df.columns}

        # Clusters — exclude string columns, add detector_id
        numeric_cols = [c for c in clusters_df.columns if clusters_df[c].dtype.kind != 'O']
        cluster_dict = {col: clusters_df[col].values for col in numeric_cols}
        cluster_dict["detector_id"] = (clusters_df["detector"] == "TPC").astype(np.int32).values
        f["clusters"] = cluster_dict

        # Flat — merged table with all columns
        merged = merge_all(events_df, tracks_df, clusters_df)
        # Replace string columns with integer encoding
        if 'cluster_detector' in merged.columns:
            merged['cluster_detector_id'] = (merged['cluster_detector'] == "TPC").astype(np.int32)
            merged = merged.drop(columns=['cluster_detector'])
        flat_dict = {col: merged[col].values for col in merged.columns}
        f["flat"] = flat_dict


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
    root_file: Optional[str] = None,
) -> None:
    """
    Save 3 DataFrames as parquet files in directory.

    Creates:
        {path}/events.parquet
        {path}/tracks.parquet
        {path}/clusters.parquet
        {path}/metadata.json  (if metadata provided)
        {root_file}           (if root_file provided, or {path}/event_display.root if root_file=True)
    """
    import os
    os.makedirs(path, exist_ok=True)
    events_df.to_parquet(os.path.join(path, "events.parquet"))
    tracks_df.to_parquet(os.path.join(path, "tracks.parquet"))
    clusters_df.to_parquet(os.path.join(path, "clusters.parquet"))
    if metadata is not None:
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    if root_file is not None:
        if root_file is True:
            root_file = os.path.join(path, "event_display.root")
        save_to_root(events_df, tracks_df, clusters_df, root_file)


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
    root_file: Optional[str] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate and cache, or load from existing cache.

    If cache exists and force=False, validates stored metadata against
    requested parameters. Raises ValueError on mismatch (use force=True
    to regenerate). Missing metadata (legacy cache) triggers a warning
    but loads anyway.

    Parameters
    ----------
    path : str
        Cache directory for parquet files.
    n_events : int
        Number of events to generate.
    force : bool
        If True, regenerate even if cache exists.
    root_file : str or True or None
        If str: export ROOT file to this path.
        If True: export to {path}/event_display.root.
        If None: no ROOT export.
    **kwargs
        Passed to generate_event_display().
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
    save_tables(events, tracks, clusters, path, metadata=requested, root_file=root_file)
    return events, tracks, clusters


# ============================================================================
# Phase 0.1.F.ext — MC Generator Extension
# Track reconstruction, efficiency, mc_labels
# ============================================================================

# Detector position resolution [cm] for cluster smearing
SIGMA_ITS = 0.001    # 10 μm — ITS silicon pixel resolution
SIGMA_TPC = 0.010    # 100 μm — TPC pad/drift resolution


def track_efficiency(pt, eta=None, params=None):
    """
    Track reconstruction efficiency as function of pT.

    Default: ε(pT) = clip(0.98 - 0.06/pT, 0, 0.99)

    Parameters
    ----------
    pt : array_like
        Transverse momentum [GeV/c].
    eta : array_like, optional
        Pseudorapidity (unused in default model, reserved for extension).
    params : dict, optional
        Override: {"a": 0.98, "b": 0.06, "max": 0.99}.

    Returns
    -------
    eff : ndarray
        Efficiency values in [0, params["max"]].
    """
    if params is None:
        params = {"a": 0.98, "b": 0.06, "max": 0.99}
    pt = np.asarray(pt, dtype=np.float64)
    eff = np.clip(params["a"] - params["b"] / pt, 0.0, params["max"])
    return eff


def build_dense_index(parent_df, child_df, key_cols):
    """
    Map composite FK columns in child_df to dense row indices into parent_df.

    Both tables should be consistently keyed (guaranteed by generator).

    Parameters
    ----------
    parent_df : DataFrame
        Parent table. Row order defines the dense index (0..len-1).
    child_df : DataFrame
        Child table with composite FK columns.
    key_cols : list of str
        Column names forming the composite key.

    Returns
    -------
    dense_fk : ndarray of int64
        Dense row index into parent_df for each row of child_df.
    """
    parent_idx = parent_df[key_cols].copy()
    parent_idx['_row_idx_'] = np.arange(len(parent_df), dtype=np.int64)
    merged = child_df[key_cols].merge(parent_idx, on=key_cols, how='left')
    result = merged['_row_idx_'].values.astype(np.int64)
    assert not np.any(result < 0), "build_dense_index: unmatched rows found"
    return result


def fit_track_linear(x, y, z, r, b_field=0.5):
    """
    Linear fit for ITS tracklets (3–7 layers).

    Fits y vs r and z vs r linearly.  pT estimated from 3-point sagitta.

    Parameters
    ----------
    x, y, z : ndarray — Smeared cluster positions [cm].
    r : ndarray — Layer radii [cm].
    b_field : float — Magnetic field [T].

    Returns
    -------
    pt_reco, tgl_reco, phi_reco, rms_y, rms_z, chi2_per_cls : floats
    """
    n = len(x)
    # y vs r
    cy = np.polyfit(r, y, 1)
    y_fit = np.polyval(cy, r)
    rms_y = np.sqrt(np.mean((y - y_fit)**2))

    # z vs r
    cz = np.polyfit(r, z, 1)
    z_fit = np.polyval(cz, r)
    rms_z = np.sqrt(np.mean((z - z_fit)**2))

    tgl_reco = cz[0]  # dz/dr ≈ tan(lambda)

    # pT from 3-point sagitta
    # sagitta = midpoint deviation from chord
    # R_helix ≈ chord² / (8 * |sagitta|)
    # pT = 0.3 * B * R_helix / 100
    pt_reco = np.nan
    if n >= 3:
        mid = n // 2
        dr = r[-1] - r[0]
        if abs(dr) > 1e-10:
            y_chord = y[0] + (y[-1] - y[0]) * (r[mid] - r[0]) / dr
            sagitta = y[mid] - y_chord
            if abs(sagitta) > 1e-8:
                chord = np.sqrt(dr**2 + (y[-1] - y[0])**2)
                R_helix_est = chord**2 / (8.0 * abs(sagitta))
                pt_reco = 0.3 * b_field * R_helix_est / 100.0
            else:
                pt_reco = 100.0  # Very straight → very high pT
    phi_reco = np.arctan2(y[0], x[0])
    chi2 = np.sum((y - y_fit)**2 + (z - z_fit)**2) / max(n - 2, 1)
    return pt_reco, tgl_reco, phi_reco, rms_y, rms_z, chi2


def fit_track_circle(x, y, z, r, b_field=0.5):
    """
    Algebraic circle fit (Kasa) in (x,y), linear fit in (r,z).

    Parameters
    ----------
    x, y, z : ndarray — Smeared cluster positions [cm].
    r : ndarray — Layer radii [cm].
    b_field : float — Magnetic field [T].

    Returns
    -------
    pt_reco, tgl_reco, phi_reco, rms_y, rms_z, chi2_per_cls : floats
    """
    n = len(x)
    # Circle fit: minimize algebraic distance
    A = np.column_stack([x, y, np.ones(n)])
    b_vec = x**2 + y**2
    result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    cx = result[0] / 2.0
    cy = result[1] / 2.0
    R_fit = np.sqrt(result[2] + cx**2 + cy**2)

    # pT from curvature
    pt_reco = 0.3 * b_field * R_fit / 100.0

    # Transverse residuals
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    res_y = dist - R_fit
    rms_y = np.sqrt(np.mean(res_y**2))

    # z vs r (linear)
    cz = np.polyfit(r, z, 1)
    z_fit = np.polyval(cz, r)
    rms_z = np.sqrt(np.mean((z - z_fit)**2))
    tgl_reco = cz[0]

    # phi at innermost point
    phi_reco = np.arctan2(y[0] - cy, x[0] - cx)

    # chi2 per cluster (5 params: cx, cy, R, tgl, z0)
    dof = max(2 * n - 5, 1)
    chi2 = np.sum(res_y**2) / max(rms_y**2, 1e-30) + np.sum((z - z_fit)**2) / max(rms_z**2, 1e-30)
    chi2_per_cls = chi2 / dof

    return pt_reco, tgl_reco, phi_reco, rms_y, rms_z, chi2_per_cls


def generate_mc_reco(
    n_events: int = 100,
    tracks_per_event: Tuple[int, int] = (3, 10),
    pt_range: Tuple[float, float] = (0.3, 5.0),
    eta_range: Tuple[float, float] = (-1.0, 1.0),
    b_field: float = 0.5,
    layers: Optional[np.ndarray] = None,
    seed: int = 42,
    efficiency_params: Optional[dict] = None,
) -> dict:
    """
    Generate extended MC + reconstruction tables (Phase 0.1.F.ext).

    Returns dict with keys:
        "events"      — identical to generate_event_display()
        "tracks"      — identical to generate_event_display() (= MC tracks)
        "clusters"    — identical to generate_event_display()
        "reco_tracks" — reconstructed tracks (subset after efficiency)
        "mc_labels"   — junction: reco_track_id ↔ mc_track_idx

    The first 3 tables are bit-identical to generate_event_display() with the
    same parameters and seed.  Extension uses child RNG (seed+1000).

    Parameters
    ----------
    n_events, tracks_per_event, pt_range, eta_range, b_field, layers, seed :
        Same as generate_event_display().
    efficiency_params : dict, optional
        Override efficiency model: {"a": 0.98, "b": 0.06, "max": 0.99}.
    """
    if layers is None:
        layers = LAYERS_ALL
    n_layers = len(layers)

    # --- Base tables (bit-identical to generate_event_display) ---
    events_df, tracks_df, clusters_df = generate_event_display(
        n_events=n_events,
        tracks_per_event=tracks_per_event,
        pt_range=pt_range,
        eta_range=eta_range,
        b_field=b_field,
        layers=layers,
        seed=seed,
    )

    total_tracks = len(tracks_df)
    # Child RNG — does not perturb base tables
    rng_ext = np.random.default_rng(seed + 1000)

    # --- Cluster smearing (internal copy for fitting) ---
    cl_layer = clusters_df['layer'].values
    sigma = np.where(cl_layer < N_ITS, SIGMA_ITS, SIGMA_TPC)
    x_smeared = clusters_df['x'].values + rng_ext.normal(0, sigma)
    y_smeared = clusters_df['y'].values + rng_ext.normal(0, sigma)
    z_smeared = clusters_df['z'].values + rng_ext.normal(0, sigma)
    r_smeared = np.sqrt(x_smeared**2 + y_smeared**2)

    # --- Efficiency selection ---
    trk_pt = tracks_df['pt'].values
    trk_eta = tracks_df['eta'].values
    eff = track_efficiency(trk_pt, trk_eta, params=efficiency_params)
    u = rng_ext.uniform(0, 1, total_tracks)
    reco_mask = u < eff  # bool array, True = reconstructed

    # Dense row indices of MC tracks that pass efficiency
    mc_indices = np.where(reco_mask)[0]  # indices into tracks_df
    n_reco = len(mc_indices)

    # --- Build dense mapping: cluster rows for each track ---
    # clusters are ordered: track 0 layers, track 1 layers, ...
    # Each track has exactly n_layers clusters
    # cluster rows for track i: [i*n_layers, (i+1)*n_layers)

    # --- Fit each reconstructed track ---
    pt_reco_arr = np.empty(n_reco, dtype=np.float64)
    tgl_reco_arr = np.empty(n_reco, dtype=np.float64)
    phi_reco_arr = np.empty(n_reco, dtype=np.float64)
    rms_y_arr = np.empty(n_reco, dtype=np.float64)
    rms_z_arr = np.empty(n_reco, dtype=np.float64)
    chi2_arr = np.empty(n_reco, dtype=np.float64)
    n_cls_arr = np.empty(n_reco, dtype=np.int64)

    for i, mc_idx in enumerate(mc_indices):
        cl_start = mc_idx * n_layers
        cl_end = cl_start + n_layers
        x_cl = x_smeared[cl_start:cl_end]
        y_cl = y_smeared[cl_start:cl_end]
        z_cl = z_smeared[cl_start:cl_end]
        r_cl = r_smeared[cl_start:cl_end]
        layer_cl = cl_layer[cl_start:cl_end]

        # Count ITS and TPC clusters
        n_its = int(np.sum(layer_cl < N_ITS))
        n_tpc = int(np.sum(layer_cl >= N_ITS))

        if n_tpc == 0 and n_its >= 3:
            # ITS-only tracklet → linear fit
            pt_r, tgl_r, phi_r, ry, rz, c2 = fit_track_linear(
                x_cl, y_cl, z_cl, r_cl, b_field
            )
        else:
            # Full track → circle fit
            pt_r, tgl_r, phi_r, ry, rz, c2 = fit_track_circle(
                x_cl, y_cl, z_cl, r_cl, b_field
            )

        pt_reco_arr[i] = pt_r
        tgl_reco_arr[i] = tgl_r
        phi_reco_arr[i] = phi_r
        rms_y_arr[i] = ry
        rms_z_arr[i] = rz
        chi2_arr[i] = c2
        n_cls_arr[i] = n_its + n_tpc

    # --- reco_tracks table ---
    reco_tracks_df = pd.DataFrame({
        'reco_track_id': np.arange(n_reco, dtype=np.int64),
        'event_id': tracks_df['event_id'].values[mc_indices].astype(np.int64),
        'mc_track_idx': mc_indices.astype(np.int64),
        'pt_reco': pt_reco_arr,
        'tgl_reco': tgl_reco_arr,
        'phi_reco': phi_reco_arr,
        'rms_y': rms_y_arr,
        'rms_z': rms_z_arr,
        'n_cls_fit': n_cls_arr,
        'chi2_per_cls': chi2_arr,
    })

    # --- mc_labels junction table ---
    mc_labels_df = pd.DataFrame({
        'reco_track_id': np.arange(n_reco, dtype=np.int64),
        'mc_track_idx': mc_indices.astype(np.int64),
        'weight': np.ones(n_reco, dtype=np.float64),
    })

    return {
        "events": events_df,
        "tracks": tracks_df,
        "clusters": clusters_df,
        "reco_tracks": reco_tracks_df,
        "mc_labels": mc_labels_df,
    }


def generate_mc_reco_its_only(n_events: int = 100, **kwargs) -> dict:
    """Generate MC+reco with ITS layers only (7 layers, linear fit)."""
    return generate_mc_reco(n_events=n_events, layers=LAYERS_ITS, **kwargs)


def generate_mc_reco_tpc_only(n_events: int = 100, **kwargs) -> dict:
    """Generate MC+reco with TPC layers only (50 layers)."""
    return generate_mc_reco(n_events=n_events, layers=LAYERS_TPC, **kwargs)


# ============================================================================
# Extended save/load (5 tables)
# ============================================================================

def save_tables_extended(
    tables: dict,
    path: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save all 5 tables as parquet files.

    Backward compatible: saves events/tracks/clusters in the same format
    as save_tables(), plus reco_tracks and mc_labels.

    Creates:
        {path}/events.parquet
        {path}/tracks.parquet
        {path}/clusters.parquet
        {path}/reco_tracks.parquet
        {path}/mc_labels.parquet
        {path}/metadata.json  (if provided)
    """
    import os
    os.makedirs(path, exist_ok=True)
    for name in ("events", "tracks", "clusters", "reco_tracks", "mc_labels"):
        tables[name].to_parquet(os.path.join(path, f"{name}.parquet"))
    if metadata is not None:
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)


def load_tables_extended(path: str) -> dict:
    """
    Load all 5 tables from parquet directory.

    Falls back to 3-table load if reco_tracks/mc_labels are missing.
    """
    import os
    result = {}
    for name in ("events", "tracks", "clusters"):
        result[name] = pd.read_parquet(os.path.join(path, f"{name}.parquet"))
    for name in ("reco_tracks", "mc_labels"):
        fpath = os.path.join(path, f"{name}.parquet")
        if os.path.exists(fpath):
            result[name] = pd.read_parquet(fpath)
    return result


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

    # ROOT export
    try:
        import os
        root_file = "toy_event_display.root"
        save_to_root(events, tracks, clusters, root_file)
        abs_path = os.path.abspath(root_file)
        print(f"\nROOT file written: {abs_path}")

        import uproot
        with uproot.open(root_file) as f:
            for tree_name in ["events", "tracks", "clusters", "flat"]:
                tree = f[tree_name]
                print(f"\n  Tree '{tree_name}': {tree.num_entries} entries, {len(tree.keys())} branches")
                print(f"    Branches: {tree.keys()}")
    except ImportError:
        print("\nSkipping ROOT export (uproot not installed)")
