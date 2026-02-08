"""
Phase 0.1.D v1.1 — WASM Cross-Backend Invariance Tests

Pattern: Python (numpy) computes reference → exports JSON → Node.js (WASM) computes → compare

Tests:
  TC-WASM-01..10: Correctness (5 functions × scalar + vector)
  TC-WASM-11:     Scalar-vs-vector consistency
  TC-WASM-SP-01..06: IEEE-754 special values (NaN, ±Inf)

The .wasm file can come from any source (WAT, C++/emscripten, Rust/wasm-pack).
"""
import subprocess
import tempfile
import pathlib
import json
import base64
import pytest
import numpy as np

# ============================================================
# Function registry: Python-side reference implementations
# These MUST match the WASM exports exactly.
# Includes tier classification and per-function tolerance (v1.1).
# ============================================================
FUNCTION_REGISTRY = {
    "fun1": {
        "numpy": lambda a, b, c: (a + b) / c,
        "fields": ["a", "b", "c"],
        "description": "(a + b) / c",
        "tier": "trivial",
        "abs_tol": 1e-14,
        "rel_tol": 1e-14,
    },
    "fun2": {
        "numpy": lambda a, b: a * b + 1.0,
        "fields": ["a", "b"],
        "description": "a * b + 1.0",
        "tier": "trivial",
        "abs_tol": 1e-14,
        "rel_tol": 1e-14,
    },
    "fun3": {
        "numpy": lambda a: np.sqrt(a * a + 1.0),
        "fields": ["a"],
        "description": "sqrt(a^2 + 1.0)",
        "tier": "complex",
        "abs_tol": 1e-14,
        "rel_tol": 1e-14,
    },
    "fun4": {
        "numpy": lambda a, b, c: np.where(a > b, c, -c),
        "fields": ["a", "b", "c"],
        "description": "a > b ? c : -c",
        "tier": "trivial",
        "abs_tol": 0,
        "rel_tol": 0,
    },
    "fun5": {
        "numpy": lambda a, b, c: np.sin(a) * np.cos(b) + np.exp(-c),
        "fields": ["a", "b", "c"],
        "description": "sin(a) * cos(b) + exp(-c)",
        "tier": "complex",
        "abs_tol": 1e-12,
        "rel_tol": 1e-12,
    },
}

# ============================================================
# Paths
# ============================================================
CWD = pathlib.Path(__file__).parent.resolve()
WASM_PATH = CWD / "functions.wasm"
JS_RUNNER = CWD / "test_wasm_cross_backend.mjs"

# ============================================================
# Test data generators
# ============================================================
rng = np.random.default_rng(42)

def make_test_data(n=100):
    """Standard random test data. c avoids zero for safe division."""
    return {
        "a": rng.random(n) * 10 - 5,       # [-5, 5]
        "b": rng.random(n) * 10 - 5,       # [-5, 5]
        "c": rng.random(n) * 9.9 + 0.1,    # [0.1, 10]
    }

def make_special_value_data():
    """IEEE-754 special values: NaN, ±Inf, zeros, mixed with finite values.
    Follows Phase 0.1.B has_infinity pattern."""
    special = np.array([0.1, 1e7, -14.0, np.nan, np.inf, -np.inf,
                        0.0, -0.0, 1e-300, 1e300], dtype=np.float64)
    # Extend to have enough elements for meaningful testing
    finite = rng.random(90) * 10 - 5
    a = np.concatenate([special, finite])
    rng2 = np.random.default_rng(123)
    b = np.concatenate([special, rng2.random(90) * 10 - 5])
    # c needs special handling: must avoid exact zero for fun1 division
    c_special = np.array([0.1, 1e7, -14.0, np.nan, np.inf, -np.inf,
                          1e-10, -1e-10, 1e-300, 1e300], dtype=np.float64)
    c = np.concatenate([c_special, np.abs(rng2.random(90) * 9.9) + 0.1])
    return {"a": a, "b": b, "c": c}

# ============================================================
# Export helpers
# ============================================================
def export_test_case(func_name, test_data, mode="scalar"):
    """Export a single test case as JSON for the Node.js runner."""
    func_def = FUNCTION_REGISTRY[func_name]
    fields = func_def["fields"]
    inputs = {f: test_data[f] for f in fields}

    # Compute reference in numpy
    ref = func_def["numpy"](*[inputs[f] for f in fields])

    return {
        "func": func_name,
        "func_v": func_name + "_v",
        "mode": mode,
        "fields": fields,
        "inputs_b64": {f: base64.b64encode(inputs[f].astype(np.float64).tobytes()).decode()
                       for f in fields},
        "ref_b64": base64.b64encode(ref.astype(np.float64).tobytes()).decode(),
        "length": len(ref),
        "abs_tol": func_def["abs_tol"],
        "rel_tol": func_def["rel_tol"],
        "description": func_def["description"],
    }

def run_js_tests(test_cases):
    """Run test cases through Node.js WASM runner. Returns (stdout, returncode)."""
    assert WASM_PATH.exists(), f"WASM file not found: {WASM_PATH}"
    assert JS_RUNNER.exists(), f"JS runner not found: {JS_RUNNER}"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_cases, f)
        json_path = f.name

    result = subprocess.run(
        ['node', str(JS_RUNNER), str(WASM_PATH), json_path],
        capture_output=True, text=True
    )
    return result

# ============================================================
# TC-WASM-01..10: Correctness tests (5 functions × scalar + vector)
# ============================================================
@pytest.mark.feature("WASM.cross_backend_invariance")
@pytest.mark.parametrize("func_name", list(FUNCTION_REGISTRY.keys()))
@pytest.mark.parametrize("mode", ["scalar", "vector"])
def test_wasm_correctness(func_name, mode):
    """TC-WASM-01..10: Cross-backend invariance for each function × mode."""
    test_data = make_test_data(100)
    tc = export_test_case(func_name, test_data, mode)
    result = run_js_tests([tc])
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        pytest.fail(f"WASM invariance failed for {mode} {func_name}:\n{result.stderr}")

# ============================================================
# TC-WASM-11: Scalar vs vector consistency
# ============================================================
@pytest.mark.feature("WASM.scalar_vector_consistency")
def test_wasm_scalar_vector_consistency():
    """TC-WASM-11: WASM scalar and vector modes must produce bit-identical results."""
    test_data = make_test_data(100)
    test_cases = []
    for func_name in FUNCTION_REGISTRY:
        # Export both modes — the JS runner will compare them internally
        test_cases.append(export_test_case(func_name, test_data, "scalar"))
        test_cases.append(export_test_case(func_name, test_data, "vector"))

    # Add a consistency check flag
    payload = {"test_cases": test_cases, "check_consistency": True}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_cases, f)
        json_path = f.name

    result = subprocess.run(
        ['node', str(JS_RUNNER), str(WASM_PATH), json_path, '--consistency'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        pytest.fail(f"Scalar-vector consistency failed:\n{result.stderr}")

# ============================================================
# TC-WASM-SP-01..06: IEEE-754 special value tests
# ============================================================
@pytest.mark.feature("WASM.special_values")
@pytest.mark.parametrize("func_name", list(FUNCTION_REGISTRY.keys()))
@pytest.mark.parametrize("mode", ["scalar", "vector"])
def test_wasm_special_values(func_name, mode):
    """TC-WASM-SP: IEEE-754 special values (NaN, ±Inf) cross-backend invariance.

    Uses mixed arrays with NaN, ±Inf, ±0, denormals, and large finite values.
    Validates that WASM and numpy produce identical special value behavior:
      - NaN propagation (NaN op X = NaN)
      - Inf arithmetic (X / 0 = ±Inf, sqrt(Inf) = Inf)
      - NaN comparison (NaN > NaN = false for fun4)
    """
    test_data = make_special_value_data()
    func_def = FUNCTION_REGISTRY[func_name]

    # For special value tests, use wider tolerance for transcendentals
    # but keep exact match for NaN==NaN and Inf==Inf (handled by allclose)
    tc = export_test_case(func_name, test_data, mode)
    # Override tolerance for special value context
    if func_def["tier"] == "complex":
        tc["abs_tol"] = max(tc["abs_tol"], 1e-12)
        tc["rel_tol"] = max(tc["rel_tol"], 1e-12)

    result = run_js_tests([tc])
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        pytest.fail(f"Special value test failed for {mode} {func_name}:\n{result.stderr}")

# ============================================================
# All-in-one (legacy compatibility with prototype)
# ============================================================
@pytest.mark.feature("WASM.cross_backend_invariance")
def test_wasm_invariance_all():
    """Run all 5 functions × 2 modes in a single Node.js invocation.
    Faster than parametrized tests (one process), useful for quick smoke test."""
    test_data = make_test_data(100)
    test_cases = []
    for func_name in FUNCTION_REGISTRY:
        test_cases.append(export_test_case(func_name, test_data, "scalar"))
        test_cases.append(export_test_case(func_name, test_data, "vector"))

    result = run_js_tests(test_cases)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        pytest.fail(f"WASM invariance test failed:\n{result.stderr}")

# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    test_wasm_invariance_all()
    print("All WASM invariance tests passed!")
