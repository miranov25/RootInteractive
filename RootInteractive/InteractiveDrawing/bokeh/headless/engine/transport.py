"""Transport check (proposal v0.2 §7.5): what the browser decoded must equal an independent expectation
computed in Python from the original data and the declared codec — before any statistic is compared.

The browser's decoder is never the reference. Stage 4.2a supports lossless transport of float64 columns:
the expected decoded column is the original column, value for value (NaN at the same rows), as a Float64Array.
Further codecs (lossy, integer, Boolean, categories) arrive with their rulebook entries (RB-7).
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .compare import compare
from .scenario import LOSSLESS


def expected_decoded(df: pd.DataFrame, columns: List[str], array_compression) -> Dict[str, Tuple[np.ndarray, str]]:
    """column -> (expected values, expected typed-array name)."""
    if tuple(array_compression) != LOSSLESS:
        raise NotImplementedError("stage 4.2a: only lossless transport has an independent expectation")
    out = {}
    for c in columns:
        v = df[c].to_numpy()
        if v.dtype != np.float64:
            raise NotImplementedError(f"stage 4.2a: column {c!r} has dtype {v.dtype}; only float64 is covered")
        out[c] = (v, "Float64Array")
    return out


def check(decoded: Dict[str, dict], expected: Dict[str, Tuple[np.ndarray, str]]) -> List[Tuple[str, bool, str]]:
    """Compare the browser's decoded columns (`{col: {"type": ..., "values": [...]}}`) with the expectation."""
    results = []
    for c, (exp, typ) in expected.items():
        got = decoded.get(c)
        if got is None:
            results.append((c, False, "column not found in the page's decoded source"))
            continue
        if got["type"] != typ:
            results.append((c, False, f"typed array {got['type']}, expected {typ}"))
            continue
        cmp = compare(got["values"], exp, kind="float", nan_equal=True)
        results.append((c, cmp.ok, cmp.message))
    return results
