"""Negative controls for two test tools (post-audit order 0; GPT22's audit, approved consolidated summary).

`test_dsl_customjs.mjs` (the DSL comparator) and `test_wasm_cross_backend.mjs` (the WASM runner) must fail when they
could not really compare: before this fix, NaN against 42 counted as a match, a run without test cases passed, and a
missing WASM export or comparison mode was skipped ("0 passed, 0 failed", exit 0). Each case runs the real tool.
"""
import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
NODE = shutil.which("node")
pytestmark = [pytest.mark.backend("node"), pytest.mark.layer("unit"),
              pytest.mark.skipif(NODE is None, reason="node not available")]


def _b64(values):
    return {"data": base64.b64encode(np.asarray(values, dtype=np.float64)).decode(), "dtype": "float64"}


def _dsl(tmp_path, lhs, rhs, cases=True):
    data = {"data": {"T": {"x": _b64(lhs), "y": _b64(rhs)}},
            "test_cases": [{"type": "EQ", "table": "T", "lhs": "x", "rhs": "y", "epsilon": 1e-10}] if cases else []}
    f = tmp_path / "dsl.json"
    f.write_text(json.dumps(data))
    return subprocess.run([NODE, str(HERE / "test_dsl_customjs.mjs"), str(f)], capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("lhs, rhs", [([np.nan], [42.0]), ([42.0], [np.nan]), ([np.inf], [42.0]), ([42.0], [-np.inf]),
                                      ([np.inf], [-np.inf]), ([1.0], [1.001])],
                         ids=["nan_vs_42", "42_vs_nan", "inf_vs_42", "42_vs_-inf", "inf_vs_-inf", "finite_mismatch"])
def test_the_dsl_comparator_rejects_a_mismatch(tmp_path, lhs, rhs):
    r = _dsl(tmp_path, lhs, rhs)
    assert r.returncode != 0 and "Test case failed" in r.stderr, r.stdout + r.stderr


@pytest.mark.parametrize("lhs, rhs", [([np.nan], [np.nan]), ([np.inf], [np.inf]), ([-np.inf], [-np.inf]),
                                      ([1.0, 2.0], [1.0, 2.0 + 1e-12])],
                         ids=["nan_vs_nan", "inf_vs_inf", "-inf_vs_-inf", "finite_within_tolerance"])
def test_the_dsl_comparator_accepts_a_match(tmp_path, lhs, rhs):
    r = _dsl(tmp_path, lhs, rhs)
    assert r.returncode == 0 and "Test case passed" in r.stdout, r.stdout + r.stderr


def test_the_dsl_comparator_fails_without_a_comparison(tmp_path):
    r = _dsl(tmp_path, [1.0], [1.0], cases=False)
    assert r.returncode != 0 and "No comparison was made" in r.stderr, r.stdout + r.stderr


WASM = HERE / "functions.wasm"


def _wasm(tmp_path, cases, consistency=False):
    f = tmp_path / "wasm.json"
    f.write_text(json.dumps(cases))
    return subprocess.run([NODE, str(HERE / "test_wasm_cross_backend.mjs"), str(WASM), str(f)] +
                          (["--consistency"] if consistency else []), capture_output=True, text=True, timeout=60)


def _case(func, mode, a, b):
    ref = np.asarray(a) * np.asarray(b) + 1.0                     # fun2(a, b) = a * b + 1 (wasm_functions.cpp)
    return {"func": func, "func_v": func + "_v", "mode": mode, "fields": ["a", "b"],
            "inputs_b64": {"a": _b64(a)["data"], "b": _b64(b)["data"]}, "ref_b64": _b64(ref)["data"],
            "length": len(a), "abs_tol": 0.0, "rel_tol": 0.0, "description": "control"}


@pytest.mark.skipif(not WASM.exists(), reason="functions.wasm not present")
def test_the_wasm_runner_passes_a_real_comparison(tmp_path):
    a, b = [1.0, 2.0, -3.5], [4.0, 0.5, 2.0]
    r = _wasm(tmp_path, [_case("fun2", "scalar", a, b), _case("fun2", "vector", a, b)], consistency=True)
    assert r.returncode == 0 and "0 failed" in r.stdout, r.stdout + r.stderr


@pytest.mark.skipif(not WASM.exists(), reason="functions.wasm not present")
@pytest.mark.parametrize("case", ["missing_export", "missing_mode", "unknown_mode", "no_comparison"])
def test_the_wasm_runner_fails_without_a_real_comparison(tmp_path, case):
    a, b = [1.0, 2.0], [3.0, 4.0]
    if case == "missing_export":
        r = _wasm(tmp_path, [_case("fun_absent", "scalar", a, b)])
    elif case == "missing_mode":
        r = _wasm(tmp_path, [_case("fun2", "scalar", a, b)], consistency=True)
    elif case == "unknown_mode":
        r = _wasm(tmp_path, [_case("fun2", "matrix", a, b)])
    else:
        r = _wasm(tmp_path, [])
    assert r.returncode != 0, r.stdout + r.stderr
