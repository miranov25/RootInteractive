"""
Phase 0.1.E — ONNX Model Generation & Reference Data Pipeline

Trains 7 scikit-learn models (4 regression + 3 classification) on synthetic data,
exports each to ONNX via skl2onnx, runs inference in Python (sklearn + onnxruntime),
and saves canonical reference data for cross-backend invariance testing.

Usage:
    python generate_models.py              # Generate all models + reference data
    python generate_models.py --check      # Verify existing artifacts are valid

Output:
    models/*.onnx                          # ONNX model files
    reference_data/*_test_vectors.json     # Test vectors with predictions
    reference_data/precision_budget.json   # Measured max diffs per model

IMPORTANT: Run once (or when models change). Tests load cached artifacts — never retrain.
"""

import os
import sys
import json
import base64
import pathlib
import argparse
import numpy as np

# Deterministic BLAS — must be set before sklearn import
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sklearn
import skl2onnx
import onnxruntime as ort
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ============================================================
# Configuration
# ============================================================

ONNX_OPSET = 17
N_FEATURES = 4
FEATURE_NAMES = ['A', 'B', 'C', 'D']

# Seed isolation (v1.1, Claude20 P1-2):
# Separate RNG per data split — adding/removing models does not shift test data.
SEED_TRAIN = 42
SEED_TEST = 123
SEED_BENCH = 456

N_TRAIN = 50_000
N_TEST = 1_000
N_BENCH = 1_000_000

CWD = pathlib.Path(__file__).parent.resolve()
MODELS_DIR = CWD / "models"
REF_DIR = CWD / "reference_data"

# ============================================================
# Model Registry — matches proposal §3.4
# ============================================================

MODEL_REGISTRY = {
    # --- Regression ---
    "ridge": {
        "class": Ridge,
        "params": {"alpha": 0.1},
        "task": "regression",
        "onnx_file": "ridge_4feat.onnx",
        "tolerance_sklearn_onnx": {"atol": 1e-6, "rtol": 1e-6},
        "tolerance_ort_cross": {"atol": 1e-7, "rtol": 1e-7},
        "description": "Linear regression — baseline, near-exact match expected",
    },
    "rf10": {
        "class": RandomForestRegressor,
        "params": {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1},
        "task": "regression",
        "onnx_file": "rf10_4feat.onnx",
        "tolerance_sklearn_onnx": {"atol": 1e-5, "rtol": 1e-5},
        "tolerance_ort_cross": {"atol": 1e-6, "rtol": 1e-6},
        "description": "Small random forest — tree ensemble operator",
    },
    "rf50": {
        "class": RandomForestRegressor,
        "params": {"n_estimators": 50, "max_depth": 5, "random_state": 42, "n_jobs": 1},
        "task": "regression",
        "onnx_file": "rf50_4feat.onnx",
        "tolerance_sklearn_onnx": {"atol": 1e-5, "rtol": 1e-5},
        "tolerance_ort_cross": {"atol": 1e-6, "rtol": 1e-6},
        "description": "Large random forest — stress test",
    },
    "mlp": {
        "class": MLPRegressor,
        "params": {"hidden_layer_sizes": (100, 50), "max_iter": 500, "random_state": 42},
        "task": "regression",
        "onnx_file": "mlp_4feat.onnx",
        "tolerance_sklearn_onnx": {"atol": 1e-5, "rtol": 1e-5},
        "tolerance_ort_cross": {"atol": 1e-6, "rtol": 1e-6},
        "description": "Multi-layer perceptron — dense + activation",
    },
    # --- Classification (v1.1) ---
    "logistic": {
        "class": LogisticRegression,
        "params": {"random_state": 42, "n_jobs": 1},
        "task": "classification",
        "onnx_file": "logistic_4feat.onnx",
        "tolerance_sklearn_onnx": {"atol": 1e-6, "rtol": 1e-6},
        "tolerance_ort_cross": {"atol": 2e-7, "rtol": 2e-7},
        "description": "Logistic regression — linear classifier baseline",
    },
    "rf_clf": {
        "class": RandomForestClassifier,
        "params": {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1},
        "task": "classification",
        "onnx_file": "rf_clf_4feat.onnx",
        "tolerance_sklearn_onnx": {"atol": 1e-5, "rtol": 1e-5},
        "tolerance_ort_cross": {"atol": 1e-6, "rtol": 1e-6},
        "description": "RF classifier — tree ensemble classifier",
    },
    "mlp_clf": {
        "class": MLPClassifier,
        "params": {"hidden_layer_sizes": (50, 25), "max_iter": 500, "random_state": 42},
        "task": "classification",
        "onnx_file": "mlp_clf_4feat.onnx",
        "tolerance_sklearn_onnx": {"atol": 1e-5, "rtol": 1e-5},
        "tolerance_ort_cross": {"atol": 1e-6, "rtol": 1e-6},
        "description": "MLP classifier — dense + softmax",
    },
}


# ============================================================
# Data Generation
# ============================================================

def generate_data():
    """Generate synthetic datasets with isolated RNG per split."""
    rng_train = np.random.default_rng(SEED_TRAIN)
    rng_test = np.random.default_rng(SEED_TEST)

    # Training data
    X_train = rng_train.random((N_TRAIN, N_FEATURES)).astype(np.float64)
    y_train_reg = X_train[:, 0] - X_train[:, 1] + rng_train.normal(0, 0.05, N_TRAIN)
    y_train_cls = (X_train[:, 0] - X_train[:, 1] > 0).astype(np.int64)

    # Test data (invariance) — stable regardless of training changes
    X_test = rng_test.random((N_TEST, N_FEATURES)).astype(np.float64)
    y_test_cls = (X_test[:, 0] - X_test[:, 1] > 0).astype(np.int64)

    return X_train, y_train_reg, y_train_cls, X_test, y_test_cls


def encode_b64(arr):
    """Encode numpy array as base64 string."""
    return base64.b64encode(arr.tobytes()).decode('utf-8')


def decode_b64(s, dtype):
    """Decode base64 string to numpy array."""
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


# ============================================================
# Model Training, Export, and Reference Generation
# ============================================================

def train_and_export(model_name, model_def, X_train, y_train_reg, y_train_cls, X_test):
    """Train a model, export to ONNX, generate reference predictions."""
    task = model_def["task"]
    cls = model_def["class"]
    params = model_def["params"]

    # Train
    model = cls(**params)
    y_train = y_train_cls if task == "classification" else y_train_reg
    model.fit(X_train, y_train)
    print(f"  [{model_name}] Trained ({task})")

    # Convert input to float32 for ONNX comparison
    X_test_f32 = X_test.astype(np.float32)

    # sklearn predictions — predict with float64 input, cast to float32
    # (sklearn internally uses float64; we compare in float32 space per §3.7)
    if task == "regression":
        y_sklearn_f64 = model.predict(X_test_f32.astype(np.float64))
        y_sklearn_f32 = y_sklearn_f64.astype(np.float32)
        proba_sklearn = None
        labels_sklearn = None
    else:
        labels_sklearn_raw = model.predict(X_test_f32.astype(np.float64))
        labels_sklearn = labels_sklearn_raw.astype(np.int64)
        proba_sklearn_f64 = model.predict_proba(X_test_f32.astype(np.float64))
        proba_sklearn_f32 = proba_sklearn_f64.astype(np.float32)
        y_sklearn_f32 = proba_sklearn_f32  # primary comparison target for classifiers
        proba_sklearn = proba_sklearn_f32
        y_sklearn_f64 = proba_sklearn_f64

    # Export to ONNX
    initial_type = [('float_input', FloatTensorType([None, N_FEATURES]))]
    # zipmap=False: classifiers output numpy arrays instead of list-of-dicts
    export_options = {}
    if task == "classification":
        export_options = {cls: {'zipmap': False}}
    onnx_model = convert_sklearn(model, initial_types=initial_type,
                                  target_opset=ONNX_OPSET,
                                  options=export_options if export_options else None)
    onnx_path = MODELS_DIR / model_def["onnx_file"]
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    onnx_size = onnx_path.stat().st_size
    print(f"  [{model_name}] Exported ONNX: {onnx_path.name} ({onnx_size:,} bytes)")

    # Python onnxruntime inference
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts)

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    ort_results = session.run(output_names, {input_name: X_test_f32})

    # Build output dict
    ort_outputs = {}
    for i, name in enumerate(output_names):
        ort_outputs[name] = ort_results[i]

    # Compute diffs for precision budget
    precision_info = {}
    if task == "regression":
        y_ort_python = ort_outputs[output_names[0]].flatten().astype(np.float32)
        diff = np.abs(y_sklearn_f32.flatten() - y_ort_python)
        finite_mask = np.isfinite(diff)
        max_abs = float(np.max(diff[finite_mask])) if np.any(finite_mask) else 0.0
        y_sklearn_nz = np.abs(y_sklearn_f32.flatten())
        y_sklearn_nz = np.where(y_sklearn_nz > 0, y_sklearn_nz, 1.0)
        rel_diff = diff / y_sklearn_nz
        max_rel = float(np.max(rel_diff[finite_mask])) if np.any(finite_mask) else 0.0
        precision_info = {
            "sklearn_vs_ort": {"max_abs_diff": max_abs, "max_rel_diff": max_rel},
        }
        print(f"  [{model_name}] sklearn↔ORT: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}")
    else:
        # Classification: compare labels and probabilities
        ort_labels = ort_outputs[output_names[0]].flatten()
        if ort_labels.dtype == np.int64 or ort_labels.dtype == object:
            ort_labels = ort_labels.astype(np.int64)
        label_match = np.all(labels_sklearn == ort_labels)
        # Probabilities — with zipmap=False, output is numpy array
        max_abs = 0.0
        if len(output_names) > 1:
            ort_proba_raw = ort_outputs[output_names[1]]
            if isinstance(ort_proba_raw, np.ndarray):
                ort_proba = ort_proba_raw.astype(np.float32)
            elif isinstance(ort_proba_raw, list):
                # zipmap=True fallback: list of dicts → array
                n_classes = len(ort_proba_raw[0])
                ort_proba = np.array([[d[c] for c in sorted(d.keys())]
                                      for d in ort_proba_raw], dtype=np.float32)
            else:
                ort_proba = np.array(ort_proba_raw, dtype=np.float32)
            diff = np.abs(proba_sklearn.flatten() - ort_proba.flatten())
            finite_mask = np.isfinite(diff)
            max_abs = float(np.max(diff[finite_mask])) if np.any(finite_mask) else 0.0
        precision_info = {
            "sklearn_vs_ort": {"labels_match": bool(label_match), "max_abs_proba_diff": max_abs},
        }
        print(f"  [{model_name}] Labels match: {label_match}, max_proba_diff: {max_abs:.2e}")

    # Build test vector JSON
    test_vector = {
        "model_name": model_name,
        "model_path": f"models/{model_def['onnx_file']}",
        "task": task,
        "N": N_TEST,
        "n_features": N_FEATURES,
        "dtype": "float32",
        "endianness": "little",
        "input_b64": encode_b64(X_test_f32),
        "tolerance_sklearn_onnx": model_def["tolerance_sklearn_onnx"],
        "tolerance_ort_cross": model_def["tolerance_ort_cross"],
        "onnx_input_name": input_name,
        "onnx_output_names": output_names,
        "versions": {
            "sklearn": sklearn.__version__,
            "skl2onnx": skl2onnx.__version__,
            "onnxruntime_python": ort.__version__,
            "onnx_opset": ONNX_OPSET,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
    }

    if task == "regression":
        test_vector["y_sklearn_b64"] = encode_b64(y_sklearn_f32.flatten())
        y_ort_flat = ort_outputs[output_names[0]].flatten().astype(np.float32)
        test_vector["y_onnx_python_b64"] = encode_b64(y_ort_flat)
    else:
        test_vector["labels_sklearn_b64"] = encode_b64(labels_sklearn)
        test_vector["proba_sklearn_b64"] = encode_b64(proba_sklearn.flatten())
        test_vector["n_classes"] = int(proba_sklearn.shape[1])
        # ORT outputs
        ort_labels_i64 = ort_outputs[output_names[0]].flatten().astype(np.int64)
        test_vector["labels_onnx_python_b64"] = encode_b64(ort_labels_i64)
        if len(output_names) > 1:
            ort_proba_raw = ort_outputs[output_names[1]]
            if isinstance(ort_proba_raw, np.ndarray):
                ort_proba_f32 = ort_proba_raw.flatten().astype(np.float32)
            elif isinstance(ort_proba_raw, list):
                n_cls = len(ort_proba_raw[0])
                ort_proba_f32 = np.array([[d[c] for c in sorted(d.keys())]
                                          for d in ort_proba_raw], dtype=np.float32).flatten()
            else:
                ort_proba_f32 = np.array(ort_proba_raw, dtype=np.float32).flatten()
            test_vector["proba_onnx_python_b64"] = encode_b64(ort_proba_f32)

    # Save test vector
    ref_path = REF_DIR / f"{model_name}_test_vectors.json"
    with open(ref_path, 'w') as f:
        json.dump(test_vector, f, indent=2)
    print(f"  [{model_name}] Saved: {ref_path.name}")

    return precision_info


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 0.1.E: Generate ONNX models + reference data")
    parser.add_argument("--check", action="store_true", help="Verify existing artifacts")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        print("Checking existing artifacts...")
        ok = True
        for name, mdef in MODEL_REGISTRY.items():
            onnx_path = MODELS_DIR / mdef["onnx_file"]
            ref_path = REF_DIR / f"{name}_test_vectors.json"
            if not onnx_path.exists():
                print(f"  MISSING: {onnx_path}")
                ok = False
            if not ref_path.exists():
                print(f"  MISSING: {ref_path}")
                ok = False
        budget_path = REF_DIR / "precision_budget.json"
        if not budget_path.exists():
            print(f"  MISSING: {budget_path}")
            ok = False
        if ok:
            print("All artifacts present.")
        else:
            print("Some artifacts missing. Run: python generate_models.py")
            sys.exit(1)
        return

    print(f"Phase 0.1.E — Generating ONNX models and reference data")
    print(f"  sklearn={sklearn.__version__}")
    print(f"  skl2onnx={skl2onnx.__version__}")
    print(f"  onnxruntime={ort.__version__}")
    print(f"  python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print()

    # Generate data
    print("Generating synthetic data...")
    X_train, y_train_reg, y_train_cls, X_test, y_test_cls = generate_data()
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print()

    # Train and export all models
    precision_budget = {}
    for model_name, model_def in MODEL_REGISTRY.items():
        print(f"Processing: {model_name}")
        precision_info = train_and_export(
            model_name, model_def, X_train, y_train_reg, y_train_cls, X_test
        )
        precision_budget[model_name] = precision_info
        print()

    # Save precision budget
    budget_path = REF_DIR / "precision_budget.json"
    with open(budget_path, 'w') as f:
        json.dump(precision_budget, f, indent=2)
    print(f"Precision budget saved: {budget_path}")

    # Summary
    print("\n=== Summary ===")
    for name in MODEL_REGISTRY:
        onnx_path = MODELS_DIR / MODEL_REGISTRY[name]["onnx_file"]
        size = onnx_path.stat().st_size
        print(f"  {name:12s}: {size:>8,} bytes  ({MODEL_REGISTRY[name]['task']})")
    print(f"\nAll {len(MODEL_REGISTRY)} models generated successfully.")


if __name__ == "__main__":
    main()
