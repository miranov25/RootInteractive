"""PHASE_5_1: export the approved v0.2 ToyMC into a Node-test JSON fixture.

Generated data, not committed source.  Python/pandas independently supplies the
expected gathers and reverse-selections; JavaScript must match those outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from toy_mc_events import (TABLES, generate_toy_mc_events, independent_gather_oracle,
                           inject_orphan_test_rows, select_collision_descendants,
                           validate_relations)

SOURCE_MD5 = "10a9546d2b1c2632c1496d1be267197d"
SCHEMA = {
    "collisions": ("rowId", "vz", "nTracks", "nV0", "nCascades"),
    "tracks": ("rowId", "collisionIdx", "pt", "charge"),
    "v0s": ("rowId", "collisionIdx", "posTrackIdx", "negTrackIdx", "mass"),
    "cascades": ("rowId", "collisionIdx", "v0Idx", "bachelorTrackIdx", "mass"),
}


def _json_numbers(array) -> list:
    """JSON numbers only; missing scientific values become null (never NaN)."""
    out = []
    for v in np.asarray(array):
        if pd.isna(v):
            out.append(None)
        elif isinstance(v, (np.integer, int)):
            out.append(int(v))
        elif isinstance(v, (np.bool_, bool)):
            out.append(bool(v))
        else:
            out.append(float(v))
    return out


def _check_pandas_oracle(tables, oracle):
    """Second scientific path: merge by explicit rowId, independent of NumPy take."""
    c = tables["collisions"][["rowId", "vz"]]
    t = tables["tracks"][["rowId", "collisionIdx", "pt"]]
    v = tables["v0s"][["rowId", "posTrackIdx", "negTrackIdx", "mass"]]
    ca = tables["cascades"][["rowId", "v0Idx", "bachelorTrackIdx"]]
    # left-merge retains all children including the deliberately orphaned rows.
    tz = t.merge(c, how="left", left_on="collisionIdx", right_on="rowId", sort=False,
                 validate="many_to_one")
    vp = v.merge(t[["rowId", "pt"]], how="left", left_on="posTrackIdx", right_on="rowId",
                 sort=False, validate="many_to_one")
    cv = ca.merge(v[["rowId", "mass"]], how="left", left_on="v0Idx", right_on="rowId",
                  sort=False, validate="many_to_one")
    for key, a in (("tracks.collisionVz", tz.vz), ("v0s.posPt", vp.pt),
                   ("cascades.v0Mass", cv.mass)):
        np.testing.assert_allclose(np.asarray(oracle[key], dtype=float), a.to_numpy(dtype=float),
                                   rtol=0, atol=0, equal_nan=True, err_msg=key)


def build_fixture(n_collisions=10, seed=42, orphans=False) -> dict:
    source_path = Path(__file__).with_name("toy_mc_events.py")
    digest = hashlib.md5(source_path.read_bytes()).hexdigest()
    if digest != SOURCE_MD5:
        raise RuntimeError(f"Wrong ToyMC generator MD5 {digest}; expected {SOURCE_MD5}. "
                           "Review and update the fixture contract before proceeding.")
    if n_collisions < 5:
        raise ValueError("At least 5 collisions required for Xi/Omega and empty-row controls")
    tables = generate_toy_mc_events(n_collisions=n_collisions, seed=seed)
    if orphans:
        tables = inject_orphan_test_rows(tables)
    validate_relations(tables, allow_orphans=orphans)
    oracle = independent_gather_oracle(tables)
    _check_pandas_oracle(tables, oracle)
    selections = {"collision_0": [0], "collisions_0_2": [0, 2],
                  "empty_collision": [n_collisions - 1], "none": [],
                  "collision_1": [1]}
    return {
        "metadata": {"fixture_version": "0.1", "generator_md5": digest,
                     "n_collisions": n_collisions, "seed": seed, "orphans": orphans,
                     "row_counts": {key: len(tables[key]) for key in TABLES}},
        "tables": {name: {col: _json_numbers(tables[name][col]) for col in SCHEMA[name]}
                   for name in TABLES},
        "oracle": {key: _json_numbers(value) for key, value in oracle.items()},
        "selections": {name: {"input": values,
                   "expected": {k: _json_numbers(x) for k, x in
                                select_collision_descendants(tables, values).items()}}
                       for name, values in selections.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path, help="JSON output path; use test_logs/, not a tracked directory")
    p.add_argument("--n-collisions", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--orphans", action="store_true")
    a = p.parse_args()
    payload = build_fixture(a.n_collisions, a.seed, a.orphans)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    counts = payload["metadata"]["row_counts"]
    print(f"TOYMC_NODE_FIXTURE PASS: {a.output} counts={counts}, orphans={a.orphans}")


if __name__ == "__main__":
    main()
