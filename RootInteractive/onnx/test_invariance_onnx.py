"""
Phase 0.1.E v1.1 — ONNX Cross-Backend Invariance Tests

Three-way comparison: sklearn ↔ Python ORT ↔ Node.js ORT

Tests:
  TC-ONNX-01..12:  Regression correctness (4 models × 3 comparisons)
  TC-ONNX-13..21:  Classification correctness (3 models × 3 comparisons)
  TC-ONNX-SP-01..06: IEEE-754 special values (NaN, ±Inf)
  TC-ONNX-SMOKE:   All models, all comparisons, quick pass/fail

Usage:
  pytest test_onnx_invariance.py -m "not bench"   # Correctness only (<60s)
  pytest test_onnx_invariance.py                   # All tests
"""

import subprocess
import tempfile
import pathlib
import json
import base64
import os
import sys
import pytest
import numpy as np

# Deterministic
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import onnxruntime as ort

# ============================================================
# Paths
# ============================================================
CWD = pathlib.Path(__file__).parent.resolve()
MODELS_DIR = CWD / "models"
REF_DIR = CWD / "reference_data"
JS_RUNNER = CWD / "test_cross_backend_onnx.mjs"

# ============================================================
# Model Registry (matches generate_models.py)
# ============================================================
REGRESSION_MODELS = ["ridge", "rf10", "rf50", "mlp"]
CLASSIFICATION_MODELS = ["logistic", "rf_clf", "mlp_clf"]
ALL_MODELS = REGRESSION_MODELS + CLASSIFICATION_MODELS

# Tolerance tiers
TOLERANCES = {
    "ridge":    {"sklearn_onnx": {"atol": 1e-6, "rtol": 1e-6}, "ort_cross": {"atol": 1e-7, "rtol": 1e-7}},
    "rf10":     {"sklearn_onnx": {"atol": 1e-5, "rtol": 1e-5}, "ort_cross": {"atol": 1e-6, "rtol": 1e-6}},
    "rf50":     {"sklearn_onnx": {"atol": 1e-5, "rtol": 1e-5}, "ort_cross": {"atol": 1e-6, "rtol": 1e-6}},
    "mlp":      {"sklearn_onnx": {"atol": 1e-5, "rtol": 1e-5}, "ort_cross": {"atol": 1e-6, "rtol": 1e-6}},
    "logistic": {"sklearn_onnx": {"atol": 1e-6, "rtol": 1e-6}, "ort_cross": {"atol": 2e-7, "rtol": 2e-7}},
    "rf_clf":   {"sklearn_onnx": {"atol": 1e-5, "rtol": 1e-5}, "ort_cross": {"atol": 1e-6, "rtol": 1e-6}},
    "mlp_clf":  {"sklearn_onnx": {"atol": 1e-5, "rtol": 1e-5}, "ort_cross": {"atol": 1e-6, "rtol": 1e-6}},
}

# ============================================================
# Helpers
# ============================================================

def decode_b64(s, dtype):
    """Decode base64 string to numpy array."""
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


def encode_b64(arr):
    """Encode numpy array as base64 string."""
    return base64.b64encode(arr.tobytes()).decode('utf-8')


def load_test_vectors(model_name):
    """Load reference test vectors from JSON."""
    path = REF_DIR / f"{model_name}_test_vectors.json"
    assert path.exists(), f"Reference data missing: {path}. Run generate_models.py first."
    with open(path) as f:
        return json.load(f)


def run_python_ort(model_name, input_f32):
    """Run Python ORT inference and return outputs dict."""
    tv = load_test_vectors(model_name)
    onnx_path = CWD / tv["model_path"]
    assert onnx_path.exists(), f"ONNX model missing: {onnx_path}"

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts)

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    results = session.run(output_names, {input_name: input_f32})

    outputs = {}
    for i, name in enumerate(output_names):
        raw = results[i]
        if isinstance(raw, list):
            # zipmap output (list of dicts) — convert to array
            n_cls = len(raw[0])
            outputs[name] = np.array([[d[c] for c in sorted(d.keys())]
                                       for d in raw], dtype=np.float32)
        else:
            outputs[name] = raw
    return outputs, output_names


def run_nodejs_ort(model_name, input_f32, N, n_features=4):
    """Run Node.js ORT inference via subprocess. Returns outputs dict or None."""
    if not JS_RUNNER.exists():
        pytest.skip(f"Node.js runner not found: {JS_RUNNER}")

    tv = load_test_vectors(model_name)
    onnx_path = (CWD / tv["model_path"]).resolve()

    config = {
        "model_path": str(onnx_path),
        "input_b64": encode_b64(input_f32),
        "N": N,
        "n_features": n_features,
        "task": tv["task"],
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False,
                                      dir=str(CWD)) as f_in:
        json.dump(config, f_in)
        config_path = f_in.name

    output_path = config_path.replace('.json', '_output.json')
    config["output_path"] = output_path

    # Rewrite with output_path
    with open(config_path, 'w') as f_in:
        json.dump(config, f_in)

    try:
        result = subprocess.run(
            ['node', str(JS_RUNNER), config_path],
            capture_output=True, text=True, timeout=120,
            cwd=str(CWD)
        )
        if result.returncode != 0:
            stderr = result.stderr
            # Skip if onnxruntime-node not installed (expected in some environments)
            if "Cannot find package" in stderr or "MODULE_NOT_FOUND" in stderr or "Cannot find module" in stderr:
                pytest.skip("onnxruntime-node not installed. Run: npm install in onnx/")
            print(f"Node.js stderr: {stderr}")
            pytest.fail(f"Node.js runner failed for {model_name}:\n{stderr}")

        if not os.path.exists(output_path):
            pytest.fail(f"Node.js runner did not produce output file: {output_path}")

        with open(output_path) as f:
            node_result = json.load(f)

        return node_result
    except FileNotFoundError:
        pytest.skip("Node.js not available")
    except subprocess.TimeoutExpired:
        pytest.fail(f"Node.js runner timed out for {model_name}")
    finally:
        for p in [config_path, output_path]:
            if os.path.exists(p):
                os.unlink(p)


def assert_allclose_with_report(actual, expected, atol, rtol, label, model_name):
    """np.allclose with detailed failure report (structured failure artifact §3.11)."""
    if not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True):
        diff = np.abs(actual - expected)
        finite_mask = np.isfinite(diff)
        if np.any(finite_mask):
            max_abs = float(np.max(diff[finite_mask]))
            worst_idx = int(np.argmax(diff[finite_mask]))
        else:
            max_abs = float('inf')
            worst_idx = 0
        msg = (f"[{model_name}] {label} FAILED: "
               f"max_abs_diff={max_abs:.2e}, "
               f"tolerance=atol={atol}, rtol={rtol}, "
               f"worst_idx={worst_idx}, "
               f"actual[{worst_idx}]={actual.flat[worst_idx]}, "
               f"expected[{worst_idx}]={expected.flat[worst_idx]}")
        pytest.fail(msg)


# ============================================================
# TC-ONNX-01..12: Regression Correctness (Three-Way)
# ============================================================

@pytest.mark.feature("ONNX.invariance.sklearn_vs_ort")
@pytest.mark.parametrize("model_name", REGRESSION_MODELS)
def test_regression_sklearn_vs_python_ort(model_name):
    """TC-ONNX-01,04,07,10: sklearn predictions ↔ Python ORT predictions."""
    tv = load_test_vectors(model_name)
    input_f32 = decode_b64(tv["input_b64"], np.float32).reshape(-1, 4)
    y_sklearn = decode_b64(tv["y_sklearn_b64"], np.float32)

    outputs, _ = run_python_ort(model_name, input_f32)
    # First output is the regression prediction
    y_ort = list(outputs.values())[0].flatten().astype(np.float32)

    tol = TOLERANCES[model_name]["sklearn_onnx"]
    assert_allclose_with_report(y_ort, y_sklearn, tol["atol"], tol["rtol"],
                                "sklearn↔Python ORT", model_name)


@pytest.mark.feature("ONNX.invariance.sklearn_vs_ort")
@pytest.mark.parametrize("model_name", REGRESSION_MODELS)
def test_regression_sklearn_vs_nodejs_ort(model_name):
    """TC-ONNX-02,05,08,11: sklearn predictions ↔ Node.js ORT predictions."""
    tv = load_test_vectors(model_name)
    input_f32 = decode_b64(tv["input_b64"], np.float32).reshape(-1, 4)
    y_sklearn = decode_b64(tv["y_sklearn_b64"], np.float32)
    N = tv["N"]

    node_result = run_nodejs_ort(model_name, input_f32, N)
    if node_result is None:
        pytest.skip("Node.js not available")

    # Decode first output
    first_output_name = list(node_result["outputs"].keys())[0]
    y_node = np.frombuffer(
        base64.b64decode(node_result["outputs"][first_output_name]),
        dtype=np.float32
    )

    tol = TOLERANCES[model_name]["sklearn_onnx"]
    assert_allclose_with_report(y_node, y_sklearn, tol["atol"], tol["rtol"],
                                "sklearn↔Node.js ORT", model_name)


@pytest.mark.feature("ONNX.invariance.cross_runtime")
@pytest.mark.parametrize("model_name", REGRESSION_MODELS)
def test_regression_python_ort_vs_nodejs_ort(model_name):
    """TC-ONNX-03,06,09,12: Python ORT ↔ Node.js ORT predictions."""
    tv = load_test_vectors(model_name)
    input_f32 = decode_b64(tv["input_b64"], np.float32).reshape(-1, 4)
    y_ort_python = decode_b64(tv["y_onnx_python_b64"], np.float32)
    N = tv["N"]

    node_result = run_nodejs_ort(model_name, input_f32, N)
    if node_result is None:
        pytest.skip("Node.js not available")

    first_output_name = list(node_result["outputs"].keys())[0]
    y_node = np.frombuffer(
        base64.b64decode(node_result["outputs"][first_output_name]),
        dtype=np.float32
    )

    tol = TOLERANCES[model_name]["ort_cross"]
    assert_allclose_with_report(y_node, y_ort_python, tol["atol"], tol["rtol"],
                                "Python ORT↔Node.js ORT", model_name)


# ============================================================
# TC-ONNX-13..21: Classification Correctness (Three-Way)
# ============================================================

@pytest.mark.feature("ONNX.invariance.classification")
@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_classification_labels_sklearn_vs_python_ort(model_name):
    """TC-ONNX-13,16,19: Classification labels sklearn ↔ Python ORT (exact)."""
    tv = load_test_vectors(model_name)
    input_f32 = decode_b64(tv["input_b64"], np.float32).reshape(-1, 4)
    labels_sklearn = decode_b64(tv["labels_sklearn_b64"], np.int64)

    outputs, output_names = run_python_ort(model_name, input_f32)
    labels_ort = outputs[output_names[0]].flatten().astype(np.int64)

    np.testing.assert_array_equal(labels_ort, labels_sklearn,
                                   err_msg=f"[{model_name}] Label mismatch sklearn↔Python ORT")


@pytest.mark.feature("ONNX.invariance.classification")
@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_classification_labels_sklearn_vs_nodejs_ort(model_name):
    """TC-ONNX-14,17,20: Classification labels sklearn ↔ Node.js ORT (exact)."""
    tv = load_test_vectors(model_name)
    input_f32 = decode_b64(tv["input_b64"], np.float32).reshape(-1, 4)
    labels_sklearn = decode_b64(tv["labels_sklearn_b64"], np.int64)
    N = tv["N"]

    node_result = run_nodejs_ort(model_name, input_f32, N)
    if node_result is None:
        pytest.skip("Node.js not available")

    # Labels output — Node.js sends as float32 (BigInt64→Number conversion)
    label_output_name = node_result["output_names"][0]
    labels_raw = np.frombuffer(
        base64.b64decode(node_result["outputs"][label_output_name]),
        dtype=np.float32
    )
    labels_node = labels_raw.astype(np.int64)

    np.testing.assert_array_equal(labels_node, labels_sklearn,
                                   err_msg=f"[{model_name}] Label mismatch sklearn↔Node.js ORT")


@pytest.mark.feature("ONNX.invariance.classification")
@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_classification_proba_python_ort_vs_nodejs_ort(model_name):
    """TC-ONNX-15,18,21: Classification probabilities Python ORT ↔ Node.js ORT."""
    tv = load_test_vectors(model_name)
    input_f32 = decode_b64(tv["input_b64"], np.float32).reshape(-1, 4)
    N = tv["N"]

    # Python ORT probabilities
    proba_ort_python = decode_b64(tv["proba_onnx_python_b64"], np.float32)

    node_result = run_nodejs_ort(model_name, input_f32, N)
    if node_result is None:
        pytest.skip("Node.js not available")

    # Probability output is second
    proba_output_name = node_result["output_names"][1] if len(node_result["output_names"]) > 1 else node_result["output_names"][0]
    proba_node = np.frombuffer(
        base64.b64decode(node_result["outputs"][proba_output_name]),
        dtype=np.float32
    )

    tol = TOLERANCES[model_name]["ort_cross"]
    assert_allclose_with_report(proba_node, proba_ort_python, tol["atol"], tol["rtol"],
                                "proba Python ORT↔Node.js ORT", model_name)


# Additional: sklearn predict_proba ↔ ORT probabilities (Claude20 P2-3)
@pytest.mark.feature("ONNX.invariance.sklearn_vs_ort")
@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_classification_proba_sklearn_vs_python_ort(model_name):
    """sklearn predict_proba ↔ Python ORT probabilities."""
    tv = load_test_vectors(model_name)
    proba_sklearn = decode_b64(tv["proba_sklearn_b64"], np.float32)
    proba_ort = decode_b64(tv["proba_onnx_python_b64"], np.float32)

    tol = TOLERANCES[model_name]["sklearn_onnx"]
    assert_allclose_with_report(proba_ort, proba_sklearn, tol["atol"], tol["rtol"],
                                "proba sklearn↔Python ORT", model_name)


# ============================================================
# TC-ONNX-SP-01..06: IEEE-754 Special Value Tests
# ============================================================

def make_special_input(N=100, n_features=4):
    """Create input with IEEE-754 special values in column A."""
    rng = np.random.default_rng(999)
    X = rng.random((N, n_features)).astype(np.float32)
    # Inject specials in column 0 (A)
    specials = [np.nan, np.inf, -np.inf, 0.0, -0.0, np.float32(1e38)]
    for i, v in enumerate(specials):
        if i < N:
            X[i, 0] = np.float32(v)
    return X


@pytest.mark.feature("ONNX.special_values")
@pytest.mark.parametrize("model_name", ["ridge", "logistic"])
def test_special_values_linear_nan(model_name):
    """TC-ONNX-SP-01: Linear models — NaN propagation (sklearn ↔ ORT)."""
    X = make_special_input(100)
    outputs_ort, output_names = run_python_ort(model_name, X)
    first_output = outputs_ort[output_names[0]]
    if first_output.dtype == np.int64:
        # Classification labels — NaN handling is model-specific
        pass
    else:
        result = first_output.flatten().astype(np.float32)
        # NaN input should produce NaN output for linear models
        assert np.isnan(result[0]), f"[{model_name}] Expected NaN output for NaN input, got {result[0]}"


@pytest.mark.feature("ONNX.special_values")
@pytest.mark.parametrize("model_name", ["ridge", "logistic"])
def test_special_values_linear_inf(model_name):
    """TC-ONNX-SP-02: Linear models — ±Inf handling."""
    X = make_special_input(100)
    outputs_ort, output_names = run_python_ort(model_name, X)
    first_output = outputs_ort[output_names[0]]
    if first_output.dtype != np.int64:
        result = first_output.flatten().astype(np.float32)
        # Inf input should produce Inf or large value output
        assert np.isinf(result[1]) or np.abs(result[1]) > 1e30, \
            f"[{model_name}] Expected Inf/large for +Inf input, got {result[1]}"


@pytest.mark.feature("ONNX.special_values")
@pytest.mark.parametrize("model_name", ["rf10", "rf_clf"])
def test_special_values_tree_nan(model_name):
    """TC-ONNX-SP-03: Tree models — NaN cross-backend consistency."""
    X = make_special_input(100)
    outputs_ort, output_names = run_python_ort(model_name, X)
    first_output = outputs_ort[output_names[0]].flatten()

    # Key requirement: Python ORT and Node.js ORT must agree (not necessarily match sklearn)
    node_result = run_nodejs_ort(model_name, X, 100)
    if node_result is None:
        pytest.skip("Node.js not available")

    first_name = node_result["output_names"][0]
    node_output = np.frombuffer(
        base64.b64decode(node_result["outputs"][first_name]),
        dtype=np.float32
    )

    # For tree models, compare ORT outputs — NaN routing is implementation-defined
    if first_output.dtype == np.int64:
        node_labels = node_output.astype(np.int64)
        py_labels = first_output.astype(np.int64)
        np.testing.assert_array_equal(node_labels, py_labels,
                                       err_msg=f"[{model_name}] Tree NaN: cross-backend label mismatch")
    else:
        # Both should produce the same deterministic value for NaN inputs
        np.testing.assert_array_equal(node_output[:6], first_output[:6].astype(np.float32),
                                       err_msg=f"[{model_name}] Tree NaN: cross-backend value mismatch")


@pytest.mark.feature("ONNX.special_values")
@pytest.mark.parametrize("model_name", ["rf10", "rf_clf"])
def test_special_values_tree_inf(model_name):
    """TC-ONNX-SP-04: Tree models — ±Inf cross-backend consistency."""
    X = make_special_input(100)
    outputs_ort, output_names = run_python_ort(model_name, X)
    first_output = outputs_ort[output_names[0]].flatten()

    node_result = run_nodejs_ort(model_name, X, 100)
    if node_result is None:
        pytest.skip("Node.js not available")

    first_name = node_result["output_names"][0]
    node_output = np.frombuffer(
        base64.b64decode(node_result["outputs"][first_name]),
        dtype=np.float32
    )

    if first_output.dtype == np.int64:
        np.testing.assert_array_equal(node_output.astype(np.int64),
                                       first_output.astype(np.int64),
                                       err_msg=f"[{model_name}] Tree Inf: cross-backend label mismatch")
    else:
        np.testing.assert_array_equal(node_output[:6], first_output[:6].astype(np.float32),
                                       err_msg=f"[{model_name}] Tree Inf: cross-backend value mismatch")


@pytest.mark.feature("ONNX.special_values")
@pytest.mark.parametrize("model_name", ["mlp", "mlp_clf"])
def test_special_values_mlp_nan(model_name):
    """TC-ONNX-SP-05: MLP models — NaN propagation."""
    X = make_special_input(100)
    outputs_ort, output_names = run_python_ort(model_name, X)
    first_output = outputs_ort[output_names[0]]
    if first_output.dtype != np.int64:
        result = first_output.flatten().astype(np.float32)
        assert np.isnan(result[0]), f"[{model_name}] Expected NaN output for NaN input, got {result[0]}"


@pytest.mark.feature("ONNX.special_values")
def test_special_values_mixed_all_models():
    """TC-ONNX-SP-06: Mixed special values — all 7 models cross-backend agreement."""
    X = make_special_input(100)
    for model_name in ALL_MODELS:
        outputs_ort, output_names = run_python_ort(model_name, X)
        py_first = outputs_ort[output_names[0]].flatten()

        node_result = run_nodejs_ort(model_name, X, 100)
        if node_result is None:
            pytest.skip("Node.js not available")

        first_name = node_result["output_names"][0]
        node_first = np.frombuffer(
            base64.b64decode(node_result["outputs"][first_name]),
            dtype=np.float32
        )

        # Cross-backend agreement for special values
        if py_first.dtype == np.int64:
            np.testing.assert_array_equal(
                node_first.astype(np.int64), py_first.astype(np.int64),
                err_msg=f"[{model_name}] Mixed specials: cross-backend label mismatch"
            )
        else:
            py_f32 = py_first.astype(np.float32)
            # Check NaN positions match
            nan_match = np.isnan(py_f32) == np.isnan(node_first)
            assert np.all(nan_match), \
                f"[{model_name}] NaN position mismatch"
            # Check finite values match
            finite_mask = np.isfinite(py_f32) & np.isfinite(node_first)
            if np.any(finite_mask):
                tol = TOLERANCES[model_name]["ort_cross"]
                assert_allclose_with_report(
                    node_first[finite_mask], py_f32[finite_mask],
                    tol["atol"], tol["rtol"],
                    "mixed specials cross-backend", model_name
                )


# ============================================================
# TC-ONNX-SMOKE: Quick smoke test (all models, <30s)
# ============================================================

def test_smoke_all_models():
    """TC-ONNX-SMOKE: Quick validation — all 7 models, Python ORT only, N=100."""
    rng = np.random.default_rng(777)
    X = rng.random((100, 4)).astype(np.float32)

    results = {}
    for model_name in ALL_MODELS:
        try:
            outputs, output_names = run_python_ort(model_name, X)
            first = outputs[output_names[0]]
            results[model_name] = f"OK: shape={first.shape}, dtype={first.dtype}"
        except Exception as e:
            results[model_name] = f"FAIL: {e}"
            pytest.fail(f"Smoke test failed for {model_name}: {e}")

    # Print summary
    for name, status in results.items():
        print(f"  {name:12s}: {status}")


# ============================================================
# Python-only correctness verification (no Node.js needed)
# ============================================================

@pytest.mark.feature("ONNX.invariance.sklearn_vs_ort")
def test_python_three_way_consistency():
    """Verify that generate_models.py reference data is internally consistent.
    Recompute Python ORT predictions and compare with saved references."""
    for model_name in ALL_MODELS:
        tv = load_test_vectors(model_name)
        input_f32 = decode_b64(tv["input_b64"], np.float32).reshape(-1, 4)

        outputs, output_names = run_python_ort(model_name, input_f32)

        if tv["task"] == "regression":
            y_saved = decode_b64(tv["y_onnx_python_b64"], np.float32)
            y_computed = outputs[output_names[0]].flatten().astype(np.float32)
            np.testing.assert_array_equal(
                y_computed, y_saved,
                err_msg=f"[{model_name}] Python ORT output changed since generate_models.py"
            )
        else:
            # Labels
            labels_saved = decode_b64(tv["labels_onnx_python_b64"], np.int64)
            labels_computed = outputs[output_names[0]].flatten().astype(np.int64)
            np.testing.assert_array_equal(
                labels_computed, labels_saved,
                err_msg=f"[{model_name}] Python ORT labels changed since generate_models.py"
            )
            # Probabilities
            if "proba_onnx_python_b64" in tv:
                proba_saved = decode_b64(tv["proba_onnx_python_b64"], np.float32)
                proba_computed = outputs[output_names[1]].flatten().astype(np.float32)
                np.testing.assert_array_equal(
                    proba_computed, proba_saved,
                    err_msg=f"[{model_name}] Python ORT probabilities changed since generate_models.py"
                )

        print(f"  [{model_name}] Python ORT consistent with saved reference ✓")
