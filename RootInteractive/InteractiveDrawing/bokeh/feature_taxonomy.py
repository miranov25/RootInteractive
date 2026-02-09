# RootInteractive/InteractiveDrawing/bokeh/feature_taxonomy.py
"""
Feature taxonomy for RootInteractive cross-backend testing.
Based on RDataFrameDSL pattern (Phase 13.6.B).

Usage:
    @pytest.mark.feature("DSL.arithmetic_expr")
    from feature_taxonomy import FEATURE_TAXONOMY

IMPORTANT: Feature IDs are IMMUTABLE. If renaming needed, use FEATURE_ALIASES.

Phase: 0.1.D
Date: 2026-02-07
Version: 1.2
"""

FEATURE_TAXONOMY_VERSION = "1.2"

# =============================================================================
# FEATURE ALIASES (for backward compatibility if renaming needed)
# =============================================================================
# Format: {"old_id": "new_id"}
# Usage: If feature ID must change, add mapping here instead of breaking tests
FEATURE_ALIASES = {}


# =============================================================================
# FEATURE TAXONOMY
# =============================================================================

FEATURE_TAXONOMY = {
    # =========================================================================
    # DSL Operations
    # =========================================================================
    "DSL.arithmetic_expr": {
        "name": "Arithmetic expressions",
        "description": "Basic arithmetic expressions (a + b * c)",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "unit",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },
    "DSL.math_functions": {
        "name": "Math functions",
        "description": "Math functions (arctan2, sqrt, sin, cos, etc.)",
        "priority": "P1",
        "backends": ["python", "node"],
        "layer": "unit",
        "proof": None,
        "planned": True,
    },
    "DSL.gather_operation": {
        "name": "Cross-table gather",
        "description": "Cross-table gather (dfB.a[i] syntax)",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "integration",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },
    "DSL.custom_js_func": {
        "name": "CustomJS function execution",
        "description": "CustomJS function execution in browser/Node",
        "priority": "P0",
        "backends": ["node", "browser"],
        "layer": "integration",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },

    # =========================================================================
    # Encoding / Data Transfer
    # =========================================================================
    "ENC.base64.float64": {
        "name": "Float64Array encoding",
        "description": "Base64 Float64Array encoding/decoding",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "unit",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },
    "ENC.base64.int32": {
        "name": "Int32Array encoding",
        "description": "Base64 Int32Array encoding/decoding",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "unit",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },
    "ENC.compression.relative": {
        "name": "Relative compression",
        "description": "Relative compression round-trip (roundRelativeBinary)",
        "priority": "P1",
        "backends": ["python", "node"],
        "layer": "integration",
        "proof": None,
        "planned": True,
    },
    "ENC.compression.delta": {
        "name": "Delta/Absolute compression",
        "description": "Fixed-step quantization round-trip (roundAbsolute)",
        "priority": "P1",
        "backends": ["python", "node"],
        "layer": "integration",
        "proof": "test_compression_integration.py::test_serializationutils",
        "planned": False,
    },
    "ENC.compression.sinh": {
        "name": "Sinh/Sqrt scaling compression",
        "description": "Arcsinh transform compression round-trip (roundSqrtScaling)",
        "priority": "P1",
        "backends": ["python", "node"],
        "layer": "integration",
        "proof": None,
        "planned": True,
    },
    "ENC.compression.zip": {
        "name": "ZIP compression",
        "description": "ZIP compression round-trip (zlib)",
        "priority": "P2",
        "backends": ["python", "node"],
        "layer": "integration",
        "proof": None,
        "planned": True,
    },
    "ENC.compression.roundtrip": {
        "name": "Compression roundtrip",
        "description": "Full compression pipeline Python->JS invariance",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "invariance",
        "proof": None,
        "planned": True,
    },

    # =========================================================================
    # Join Operations
    # =========================================================================
    "JOIN.cdsjoin.basic": {
        "name": "Basic CDSJoin",
        "description": "Basic CDSJoin index join",
        "priority": "P0",
        "backends": ["browser"],
        "layer": "integration",
        "proof": "test_ClientSideJoin.py::test_join",
    },
    "JOIN.cdsjoin.index0": {
        "name": "CDSJoin index-0 regression",
        "description": "CDSJoin with index 0 (regression test for x>=0 fix)",
        "priority": "P0",
        "backends": ["browser"],
        "layer": "integration",
        "proof": "test_ClientSideJoin.py::test_join",
        "regression": "PR-371",
    },
    "JOIN.cdsjoin.outer": {
        "name": "CDSJoin outer join",
        "description": "CDSJoin outer join with null handling",
        "priority": "P1",
        "backends": ["browser"],
        "layer": "integration",
        "proof": None,
        "planned": True,
    },
    "JOIN.cross_table": {
        "name": "Multi-CDS cross-table",
        "description": "Multi-CDS function with cross-table access",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "integration",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },

    # =========================================================================
    # Histogram Operations
    # =========================================================================
    "HIST.histogram_1d": {
        "name": "1D histogram",
        "description": "1D histogram binning (HistogramCDS)",
        "priority": "P1",
        "backends": ["browser"],
        "layer": "integration",
        "proof": None,
        "planned": True,
    },
    "HIST.histogram_nd": {
        "name": "N-D histogram",
        "description": "N-dimensional histogram (HistoNdCDS)",
        "priority": "P2",
        "backends": ["browser"],
        "layer": "integration",
        "proof": None,
        "planned": True,
    },

    # =========================================================================
    # Alias Operations
    # =========================================================================
    "ALIAS.cdsalias": {
        "name": "CDSAlias",
        "description": "CDSAlias column aliasing",
        "priority": "P1",
        "backends": ["browser"],
        "layer": "integration",
        "proof": "test_Alias.py",
        "planned": True,
    },

    # =========================================================================
    # WASM Operations (Phase 0.1.D)
    # =========================================================================
    "WASM.cross_backend_invariance": {
        "name": "WASM cross-backend invariance",
        "description": "WASM results match numpy reference at IEEE-754 precision",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "invariance",
        "proof": "test_wasm_invariance.py::test_wasm_correctness",
    },
    "WASM.scalar_vector_consistency": {
        "name": "WASM scalar-vector consistency",
        "description": "WASM scalar and vector modes produce bit-identical results",
        "priority": "P0",
        "backends": ["node"],
        "layer": "invariance",
        "proof": "test_wasm_invariance.py::test_wasm_scalar_vector_consistency",
    },
    "WASM.special_values": {
        "name": "WASM IEEE-754 special values",
        "description": "NaN/±Inf handling in WASM matches numpy",
        "priority": "P1",
        "backends": ["python", "node"],
        "layer": "invariance",
        "proof": "test_wasm_invariance.py::test_wasm_special_values",
    },
    "WASM.compile": {
        "name": "WASM compilation",
        "description": "C++ to WASM compilation via emscripten",
        "priority": "P0",
        "backends": ["native"],
        "layer": "build",
        "proof": "wasm/Makefile",
    },
    "WASM.scalar.arithmetic": {
        "name": "WASM scalar arithmetic",
        "description": "Scalar evaluation of basic arithmetic (fun1, fun2)",
        "priority": "P0",
        "backends": ["node"],
        "layer": "unit",
        "proof": "test_wasm_invariance.py::test_wasm_correctness[scalar-fun1]",
    },
    "WASM.scalar.transcendental": {
        "name": "WASM scalar transcendental",
        "description": "Scalar evaluation of transcendentals (fun3, fun5)",
        "priority": "P0",
        "backends": ["node"],
        "layer": "unit",
        "proof": "test_wasm_invariance.py::test_wasm_correctness[scalar-fun3]",
    },
    "WASM.scalar.conditional": {
        "name": "WASM scalar conditional",
        "description": "Scalar evaluation of conditionals (fun4)",
        "priority": "P0",
        "backends": ["node"],
        "layer": "unit",
        "proof": "test_wasm_invariance.py::test_wasm_correctness[scalar-fun4]",
    },
    "WASM.vector.arithmetic": {
        "name": "WASM vector arithmetic",
        "description": "Vector evaluation of basic arithmetic via WASM memory",
        "priority": "P0",
        "backends": ["node"],
        "layer": "unit",
        "proof": "test_wasm_invariance.py::test_wasm_correctness[vector-fun1]",
    },
    "WASM.vector.transcendental": {
        "name": "WASM vector transcendental",
        "description": "Vector evaluation of transcendentals via WASM memory",
        "priority": "P0",
        "backends": ["node"],
        "layer": "unit",
        "proof": "test_wasm_invariance.py::test_wasm_correctness[vector-fun3]",
    },
    "WASM.vector.conditional": {
        "name": "WASM vector conditional",
        "description": "Vector evaluation of conditionals via WASM memory",
        "priority": "P0",
        "backends": ["node"],
        "layer": "unit",
        "proof": "test_wasm_invariance.py::test_wasm_correctness[vector-fun4]",
    },
    "WASM.benchmark.scalar_overhead": {
        "name": "WASM benchmark scalar overhead",
        "description": "Per-call scalar overhead measurement (ns/call)",
        "priority": "P2",
        "backends": ["node"],
        "layer": "benchmark",
        "proof": "bench_wasm_vs_js.mjs",
    },
    "WASM.benchmark.vector_crossover": {
        "name": "WASM benchmark vector crossover",
        "description": "N-based JS/WASM crossover measurement",
        "priority": "P2",
        "backends": ["node"],
        "layer": "benchmark",
        "proof": "bench_wasm_vs_js.mjs",
    },
    "WASM.benchmark.memory_fraction": {
        "name": "WASM benchmark memory fraction",
        "description": "Memory copy fraction of total WASM time",
        "priority": "P2",
        "backends": ["node"],
        "layer": "benchmark",
        "proof": "bench_wasm_vs_js.mjs",
    },

    # =========================================================================
    # ONNX Inference (Phase 0.1.E)
    # =========================================================================
    "ONNX.export.linear": {
        "name": "ONNX export linear models",
        "description": "sklearn Ridge/LogisticRegression → ONNX via skl2onnx (LinearRegressor/LinearClassifier operators)",
        "priority": "P0",
        "backends": ["python"],
        "layer": "export",
        "proof": "onnx/generate_models.py",
    },
    "ONNX.export.tree_ensemble": {
        "name": "ONNX export tree ensembles",
        "description": "sklearn RandomForest Regressor/Classifier → ONNX (TreeEnsembleRegressor/Classifier operators)",
        "priority": "P0",
        "backends": ["python"],
        "layer": "export",
        "proof": "onnx/generate_models.py",
    },
    "ONNX.export.neural_net": {
        "name": "ONNX export neural networks",
        "description": "sklearn MLP Regressor/Classifier → ONNX (MatMul, Relu, Add, Softmax operators)",
        "priority": "P0",
        "backends": ["python"],
        "layer": "export",
        "proof": "onnx/generate_models.py",
    },
    "ONNX.invariance.sklearn_vs_ort": {
        "name": "ONNX sklearn↔ORT invariance",
        "description": "sklearn predictions match Python onnxruntime within float32 tolerance for all 7 models",
        "priority": "P0",
        "backends": ["python"],
        "layer": "invariance",
        "proof": "onnx/test_invariance_onnx.py::test_regression_sklearn_vs_python_ort",
    },
    "ONNX.invariance.cross_runtime": {
        "name": "ONNX cross-runtime invariance",
        "description": "Python ORT and Node.js ORT produce identical results (within float32 ULP) for same .onnx model",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "invariance",
        "proof": "onnx/test_invariance_onnx.py::test_regression_python_ort_vs_nodejs_ort",
    },
    "ONNX.invariance.classification": {
        "name": "ONNX classification invariance",
        "description": "Classification labels (exact) and probabilities (float32 tolerance) match across all 3 backends",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "invariance",
        "proof": "onnx/test_invariance_onnx.py::test_classification_labels_sklearn_vs_python_ort",
    },
    "ONNX.special_values": {
        "name": "ONNX IEEE-754 special values",
        "description": "NaN/±Inf handling verified across backends — linear propagation, tree cross-backend match, MLP NaN propagation",
        "priority": "P1",
        "backends": ["python", "node"],
        "layer": "invariance",
        "proof": "onnx/test_invariance_onnx.py::test_special_values_mixed_all_models",
    },
    "ONNX.benchmark.load_time": {
        "name": "ONNX model load time",
        "description": "Cold and warm model load time for all 7 models (all <36ms cold, <1ms warm)",
        "priority": "P2",
        "backends": ["node"],
        "layer": "benchmark",
        "proof": "onnx/bench_inference_onnx.mjs",
    },
    "ONNX.benchmark.inference": {
        "name": "ONNX inference latency",
        "description": "Per-model inference latency N=1 to 1M. Linear: interactive at N=1M (<10ms). Tree/MLP: batch at N=1M.",
        "priority": "P1",
        "backends": ["node"],
        "layer": "benchmark",
        "proof": "onnx/bench_inference_onnx.mjs",
    },
}


# =============================================================================
# KNOWN LIMITATIONS
# =============================================================================

KNOWN_LIMITATIONS = {
    # Add limitations as they are discovered
    # Format follows RDataFrameDSL pattern:
    # "L1": {
    #     "name": "Limitation name",
    #     "status": "⚠️ Partial",
    #     "description": "Description of the limitation",
    #     "workaround": "How to work around it",
    #     "resolution": None,  # or description of fix
    # },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_feature_ids():
    """Return sorted list of all feature IDs."""
    return sorted(FEATURE_TAXONOMY.keys())


def get_features_by_category(prefix: str) -> dict:
    """Get all features with a given prefix (e.g., 'DSL', 'ENC')."""
    return {k: v for k, v in FEATURE_TAXONOMY.items() if k.startswith(prefix + ".")}


def get_features_by_priority(priority: str) -> dict:
    """Get all features with a given priority."""
    return {k: v for k, v in FEATURE_TAXONOMY.items() if v.get("priority") == priority}


def get_all_categories() -> list:
    """Get list of all category prefixes."""
    categories = set()
    for feature_id in FEATURE_TAXONOMY:
        if "." in feature_id:
            categories.add(feature_id.split(".")[0])
    return sorted(categories)


def validate_feature_id(feature_id: str) -> bool:
    """Check if feature ID is valid (including aliases)."""
    if feature_id in FEATURE_TAXONOMY:
        return True
    if feature_id in FEATURE_ALIASES:
        return True
    return False


def resolve_feature_id(feature_id: str) -> str:
    """Resolve alias to canonical feature ID."""
    return FEATURE_ALIASES.get(feature_id, feature_id)


def get_planned_features() -> dict:
    """Get all features marked as planned."""
    return {k: v for k, v in FEATURE_TAXONOMY.items() if v.get("planned")}


def get_implemented_features() -> dict:
    """Get all features not marked as planned."""
    return {k: v for k, v in FEATURE_TAXONOMY.items() if not v.get("planned")}
