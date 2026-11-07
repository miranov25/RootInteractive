"""Seeded generation of input tables from short column specifications (proposal v0.2 §4, field `data`).

Column spec grammar (stage 4.2a):  "uniform(lo,hi)" | "normal(mu,sigma)", optionally followed by "+ p% NaN".
Each column draws from its own stream, seeded by (table seed, crc32 of the column name), so adding a column
never changes the others. The generated table is saved with every run (`case.npz`), so a comparison on another
platform can replay exactly the same values rather than relying on the same seed.
"""
import hashlib
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = re.compile(r"^\s*(uniform|normal)\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)"
                   r"\s*(?:\+\s*([0-9.]+)\s*%\s*NaN)?\s*$")


def parse_spec(spec: str):
    m = _SPEC.match(spec)
    if not m:
        raise ValueError(f"column spec {spec!r} not understood; use 'uniform(lo,hi)' or 'normal(mu,sigma)', "
                         f"optionally with '+ p% NaN'")
    kind, p1, p2, nan = m.groups()
    return kind, float(p1), float(p2), (float(nan) / 100.0 if nan else 0.0)


def generate(table) -> pd.DataFrame:
    """`table` is an engine.scenario.TableSpec (n, seed, columns)."""
    out = {}
    for name, spec in table.columns.items():
        kind, p1, p2, nan_frac = parse_spec(spec)
        rng = np.random.default_rng([int(table.seed), zlib.crc32(name.encode())])
        values = rng.uniform(p1, p2, table.n) if kind == "uniform" else rng.normal(p1, p2, table.n)
        if nan_frac > 0:
            values[rng.random(table.n) < nan_frac] = np.nan
        out[name] = values.astype(np.float64)
    return pd.DataFrame(out)


def content_sha256(df: pd.DataFrame) -> str:
    """sha256 of the table's content (column names, dtypes and values in column order) — identical on every
    platform that generated identical values; independent of file timestamps."""
    h = hashlib.sha256()
    for c in df.columns:
        v = np.ascontiguousarray(df[c].to_numpy())
        h.update(c.encode() + b"\0" + str(v.dtype).encode() + b"\0")
        h.update(v.tobytes())
    return h.hexdigest()


def save_case(df: pd.DataFrame, path: Path) -> str:
    """Save the generated table (for replay); return its content sha256 (see content_sha256)."""
    path = Path(path)
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **{c: df[c].to_numpy() for c in df.columns})
    return content_sha256(df)


def load_case(path: Path) -> pd.DataFrame:
    with np.load(path) as z:
        return pd.DataFrame({k: z[k] for k in z.files})
