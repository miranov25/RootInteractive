"""
generate_ao2d_fixture.py — Phase 0.2.A Python Oracle

Generates ao2d_fixture.json containing:
1. Fixture tables from generate_mc_reco() (Phase 0.1.F.ext)
2. Dense FK arrays from build_dense_index()
3. Oracle results for all cross-backend TC-* test cases

The oracle results are the reference truth that Node.js must match.

Usage:
    python generate_ao2d_fixture.py [output_path]

Default output: ao2d_fixture.json in current directory.
"""

import json
import sys
import os
import numpy as np

# Add generators directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Tools', 'generators'))
# Also support running from the same directory as toy_event_generator.py
sys.path.insert(0, os.path.dirname(__file__))

from toy_event_generator import generate_mc_reco, build_dense_index


def build_inverse_index_csr(fk_array, parent_count):
    """
    Build CSR inverse index: parent → list of child indices.

    Parameters
    ----------
    fk_array : ndarray of int — dense FK column (child → parent row index)
    parent_count : int — number of parent rows

    Returns
    -------
    offsets : list of int — length parent_count + 1
    indices : list of int — child row indices, grouped by parent
    """
    # Count children per parent
    counts = np.zeros(parent_count, dtype=np.int64)
    for v in fk_array:
        if 0 <= v < parent_count:
            counts[v] += 1

    # Prefix sum → offsets
    offsets = np.zeros(parent_count + 1, dtype=np.int64)
    for j in range(parent_count):
        offsets[j + 1] = offsets[j] + counts[j]

    # Fill indices
    indices = np.zeros(int(offsets[-1]), dtype=np.int64)
    pos = offsets[:-1].copy()
    for i, v in enumerate(fk_array):
        if 0 <= v < parent_count:
            indices[pos[v]] = i
            pos[v] += 1

    return offsets.tolist(), indices.tolist()


def build_inverse_index_sorted_range(fk_array, parent_count):
    """
    Build SortedRangeIndex: parent → (start, count) in the child array.

    Precondition: fk_array must be non-decreasing.

    Parameters
    ----------
    fk_array : ndarray of int — sorted dense FK column
    parent_count : int — number of parent rows

    Returns
    -------
    starts : list of int — length parent_count
    counts : list of int — length parent_count
    """
    starts = [0] * parent_count
    counts = [0] * parent_count
    n = len(fk_array)
    if n == 0:
        return starts, counts

    i = 0
    while i < n:
        v = int(fk_array[i])
        if v < 0 or v >= parent_count:
            i += 1
            continue
        start = i
        while i < n and int(fk_array[i]) == v:
            i += 1
        starts[v] = start
        counts[v] = i - start

    return starts, counts


def children_of_csr(offsets, indices, parent_idx):
    """Get sorted child indices for a parent from CSR index."""
    children = indices[offsets[parent_idx]:offsets[parent_idx + 1]]
    return sorted(children)


def children_of_range(starts, counts, parent_idx):
    """Get child indices for a parent from SortedRangeIndex."""
    s = starts[parent_idx]
    c = counts[parent_idx]
    return list(range(s, s + c))


def cross_table_filter(offsets_or_starts, indices_or_counts, selected_parents,
                       child_count, is_csr=True):
    """
    Compute child include-mask from parent selection.

    Returns list of int (0/1), length = child_count.
    """
    mask = [0] * child_count
    for p in selected_parents:
        if is_csr:
            for j in range(offsets_or_starts[p], offsets_or_starts[p + 1]):
                mask[indices_or_counts[j]] = 1
        else:
            s = offsets_or_starts[p]
            c = indices_or_counts[p]
            for j in range(s, s + c):
                mask[j] = 1
    return mask


def generate_oracle(tables, dense_fk):
    """
    Compute oracle results for all cross-backend TC-* tests.

    Returns dict of oracle results keyed by test case ID.
    """
    n_events = len(tables["events"])
    n_tracks = len(tables["tracks"])
    n_clusters = len(tables["clusters"])
    n_reco = len(tables["reco_tracks"])

    tr_event_idx = np.array(dense_fk["tracks_event_idx"])
    cl_track_idx = np.array(dense_fk["clusters_track_idx"])
    cl_event_idx = np.array(dense_fk["clusters_event_idx"])
    mc_track_idx = tables["mc_labels"]["mc_track_idx"]

    oracle = {}

    # ----------------------------------------------------------------
    # TC-INV-01: InverseIndexCSR — tracks_event_idx → events
    # ----------------------------------------------------------------
    offsets_te, indices_te = build_inverse_index_csr(tr_event_idx, n_events)
    # Children of each event (sorted ascending)
    inv01_children = {}
    for j in range(n_events):
        inv01_children[str(j)] = children_of_csr(offsets_te, indices_te, j)
    oracle["TC-INV-01"] = {
        "offsets": offsets_te,
        "indices": indices_te,
        "children_per_parent": inv01_children,
        "total_children": sum(len(v) for v in inv01_children.values()),
    }

    # ----------------------------------------------------------------
    # TC-INV-02: SortedRangeIndex — tracks_event_idx (sorted)
    # ----------------------------------------------------------------
    starts_te, counts_te = build_inverse_index_sorted_range(tr_event_idx, n_events)
    inv02_children = {}
    for j in range(n_events):
        inv02_children[str(j)] = children_of_range(starts_te, counts_te, j)
    oracle["TC-INV-02"] = {
        "starts": starts_te,
        "counts": counts_te,
        "children_per_parent": inv02_children,
    }

    # ----------------------------------------------------------------
    # TC-INV-03: CSR vs SortedRange invariance
    # For every parent, sorted child sets must be identical.
    # Oracle: the canonical sorted children (same as INV-01).
    # JS must verify both variants match this.
    # ----------------------------------------------------------------
    oracle["TC-INV-03"] = {
        "canonical_children": inv01_children,
    }

    # ----------------------------------------------------------------
    # TC-INV-04: Sentinel handling (efficiency holes)
    # Build inverse index: mc_labels.mc_track_idx → tracks
    # MC tracks NOT in mc_labels have 0 children.
    # ----------------------------------------------------------------
    mc_track_idx_arr = np.array(mc_track_idx)
    # mc_labels acts as child table; mc_track_idx is FK to tracks (parent)
    # "children of track j" = which mc_label rows point to track j
    offsets_ml, indices_ml = build_inverse_index_csr(mc_track_idx_arr, n_tracks)

    matched_tracks = sorted(set(int(v) for v in mc_track_idx_arr if 0 <= v < n_tracks))
    all_tracks = set(range(n_tracks))
    unmatched_tracks = sorted(all_tracks - set(matched_tracks))

    oracle["TC-INV-04"] = {
        "matched_mc_tracks": matched_tracks,
        "unmatched_mc_tracks": unmatched_tracks,
        "n_matched": len(matched_tracks),
        "n_unmatched": len(unmatched_tracks),
    }

    # ----------------------------------------------------------------
    # TC-INV-05: Empty parent (0 children)
    # Events or tracks with 0 children.
    # ----------------------------------------------------------------
    empty_events = [j for j in range(n_events) if offsets_te[j] == offsets_te[j + 1]]
    empty_tracks_in_labels = [j for j in range(n_tracks)
                               if offsets_ml[j] == offsets_ml[j + 1]]
    oracle["TC-INV-05"] = {
        "empty_events": empty_events,
        "empty_tracks_in_mc_labels": empty_tracks_in_labels[:50],  # first 50
        "n_empty_tracks_in_mc_labels": len(empty_tracks_in_labels),
    }

    # ----------------------------------------------------------------
    # TC-CTF-01: Single parent selection — event 5
    # ----------------------------------------------------------------
    sel_01 = [5]
    mask_01 = cross_table_filter(offsets_te, indices_te, sel_01, n_tracks, is_csr=True)
    oracle["TC-CTF-01"] = {
        "selected_events": sel_01,
        "expected_mask": mask_01,
        "expected_track_indices": sorted([i for i, v in enumerate(mask_01) if v == 1]),
    }

    # ----------------------------------------------------------------
    # TC-CTF-02: Range selection — events 0–9
    # ----------------------------------------------------------------
    sel_02 = list(range(10))
    mask_02 = cross_table_filter(offsets_te, indices_te, sel_02, n_tracks, is_csr=True)
    oracle["TC-CTF-02"] = {
        "selected_events": sel_02,
        "expected_mask": mask_02,
        "expected_track_indices": sorted([i for i, v in enumerate(mask_02) if v == 1]),
    }

    # ----------------------------------------------------------------
    # TC-CTF-03: Select all → all tracks
    # ----------------------------------------------------------------
    sel_03 = list(range(n_events))
    mask_03 = cross_table_filter(offsets_te, indices_te, sel_03, n_tracks, is_csr=True)
    oracle["TC-CTF-03"] = {
        "selected_events": sel_03,
        "expected_mask": mask_03,
        "n_selected_tracks": sum(mask_03),
    }

    # ----------------------------------------------------------------
    # TC-CTF-05: Cascade — event 5 → tracks → clusters
    # ----------------------------------------------------------------
    # Step 1: event 5 → tracks
    mask_tracks_05 = cross_table_filter(offsets_te, indices_te, [5], n_tracks, is_csr=True)
    selected_tracks_05 = [i for i, v in enumerate(mask_tracks_05) if v == 1]

    # Step 2: selected tracks → clusters
    offsets_ct, indices_ct = build_inverse_index_csr(cl_track_idx, n_tracks)
    mask_clusters_05 = cross_table_filter(offsets_ct, indices_ct, selected_tracks_05,
                                           n_clusters, is_csr=True)
    oracle["TC-CTF-05"] = {
        "selected_event": 5,
        "intermediate_track_indices": selected_tracks_05,
        "expected_cluster_mask": mask_clusters_05,
        "n_selected_clusters": sum(mask_clusters_05),
    }

    # ----------------------------------------------------------------
    # TC-CTF-06: M:N via junction (mc_labels)
    # Select tracks [10, 20, 30] → filter mc_labels → get reco_track_ids
    # ----------------------------------------------------------------
    sel_tracks_06 = [10, 20, 30]
    # mc_labels.mc_track_idx → tracks: inverse index already built (offsets_ml)
    # But here we want: "which mc_label rows have mc_track_idx in {10,20,30}?"
    mask_labels_06 = cross_table_filter(offsets_ml, indices_ml, sel_tracks_06,
                                         len(tables["mc_labels"]), is_csr=True)
    surviving_label_indices = [i for i, v in enumerate(mask_labels_06) if v == 1]
    reco_track_ids = [int(tables["mc_labels"]["reco_track_id"][i])
                      for i in surviving_label_indices]

    oracle["TC-CTF-06"] = {
        "selected_mc_tracks": sel_tracks_06,
        "expected_label_mask": mask_labels_06,
        "surviving_label_indices": surviving_label_indices,
        "reached_reco_track_ids": sorted(reco_track_ids),
    }

    # ----------------------------------------------------------------
    # TC-CTF-08: Round-trip invariance
    # Select all events → all tracks. Then gather event_id from those tracks.
    # ----------------------------------------------------------------
    mask_08 = cross_table_filter(offsets_te, indices_te, list(range(n_events)),
                                  n_tracks, is_csr=True)
    oracle["TC-CTF-08"] = {
        "all_tracks_mask": mask_08,
        "n_total_tracks": n_tracks,
        "all_selected": sum(mask_08) == n_tracks,
    }

    # ----------------------------------------------------------------
    # Inverse index for clusters → tracks (for cascade tests & perf)
    # ----------------------------------------------------------------
    oracle["inverse_clusters_track"] = {
        "csr_offsets": offsets_ct,
        "csr_indices": indices_ct,
    }

    starts_ct, counts_ct = build_inverse_index_sorted_range(cl_track_idx, n_tracks)
    oracle["inverse_clusters_track_sorted"] = {
        "starts": starts_ct,
        "counts": counts_ct,
    }

    return oracle


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate AO2D fixture + oracle")
    parser.add_argument("output", nargs="?", default="ao2d_fixture.json",
                        help="Output JSON path")
    parser.add_argument("--n_events", type=int, default=1000,
                        help="Number of events (default 1000)")
    parser.add_argument("--tracks_per_event", type=int, nargs=2, default=[3, 10],
                        metavar=("MIN", "MAX"),
                        help="Tracks per event range (default 3 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    args = parser.parse_args()
    output_path = args.output

    print(f"Generating fixture: n_events={args.n_events}, "
          f"tracks_per_event={tuple(args.tracks_per_event)}, seed={args.seed}")
    tables = generate_mc_reco(n_events=args.n_events, seed=args.seed,
                              tracks_per_event=tuple(args.tracks_per_event))

    print("Computing dense FK arrays...")
    tr_event_idx = build_dense_index(tables["events"], tables["tracks"], ["event_id"])
    cl_track_idx = build_dense_index(tables["tracks"], tables["clusters"],
                                      ["event_id", "track_id"])
    cl_event_idx = build_dense_index(tables["events"], tables["clusters"], ["event_id"])

    dense_fk = {
        "tracks_event_idx": tr_event_idx.tolist(),
        "clusters_track_idx": cl_track_idx.tolist(),
        "clusters_event_idx": cl_event_idx.tolist(),
    }

    print("Computing oracle results...")
    oracle = generate_oracle(tables, dense_fk)

    print("Building fixture JSON...")
    fixture = {
        "metadata": {
            "generator": "Phase 0.1.F.ext generate_mc_reco()",
            "seed": args.seed,
            "n_events": args.n_events,
            "tracks_per_event": args.tracks_per_event,
            "version": "0.2.0",
        },
        "tables": {
            "events": {
                "event_id": tables["events"]["event_id"].tolist(),
                "n_tracks": tables["events"]["n_tracks"].tolist(),
            },
            "tracks": {
                "event_id": tables["tracks"]["event_id"].tolist(),
                "track_id": tables["tracks"]["track_id"].tolist(),
                "pt": tables["tracks"]["pt"].tolist(),
            },
            "clusters": {
                "event_id": tables["clusters"]["event_id"].tolist(),
                "track_id": tables["clusters"]["track_id"].tolist(),
                "layer": tables["clusters"]["layer"].tolist(),
            },
            "reco_tracks": {
                "reco_track_id": tables["reco_tracks"]["reco_track_id"].tolist(),
                "event_id": tables["reco_tracks"]["event_id"].tolist(),
                "mc_track_idx": tables["reco_tracks"]["mc_track_idx"].tolist(),
            },
            "mc_labels": {
                "reco_track_id": tables["mc_labels"]["reco_track_id"].tolist(),
                "mc_track_idx": tables["mc_labels"]["mc_track_idx"].tolist(),
                "weight": tables["mc_labels"]["weight"].tolist(),
            },
        },
        "dense_fk": dense_fk,
        "oracle": oracle,
        "row_counts": {
            "events": len(tables["events"]),
            "tracks": len(tables["tracks"]),
            "clusters": len(tables["clusters"]),
            "reco_tracks": len(tables["reco_tracks"]),
            "mc_labels": len(tables["mc_labels"]),
        },
    }

    print(f"Writing {output_path}...")
    with open(output_path, "w") as f:
        json.dump(fixture, f, separators=(",", ":"))

    file_size = os.path.getsize(output_path)
    print(f"Done. {file_size / 1024 / 1024:.1f} MB")
    print(f"  Events:      {fixture['row_counts']['events']}")
    print(f"  Tracks:      {fixture['row_counts']['tracks']}")
    print(f"  Clusters:    {fixture['row_counts']['clusters']}")
    print(f"  Reco tracks: {fixture['row_counts']['reco_tracks']}")
    print(f"  MC labels:   {fixture['row_counts']['mc_labels']}")
    print(f"  Oracle tests: {len(oracle)} sections")


if __name__ == "__main__":
    main()
