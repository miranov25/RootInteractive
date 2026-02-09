# CAPABILITY_MATRIX

**Generated:** 2026-02-09 00:39:42

**Generator:** `scripts/generate_capability_matrix.py`

**Phase:** 0.1.A

> This matrix shows test coverage for RootInteractive features.
> Status is derived from pytest outcomes using `pytest-json-report`.

---

## ALIAS

*Column aliasing operations*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `ALIAS.cdsalias` | P1 | 📋 Planned | browser | integration | - |


## DSL

*Domain-Specific Language operations*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `DSL.arithmetic_expr` | P0 | ✅ Working | python, node | unit | 1 test |
| `DSL.custom_js_func` | P0 | ✅ Working | node, browser | integration | 1 test |
| `DSL.gather_operation` | P0 | ✅ Working | python, node | integration | 2 tests |
| `DSL.math_functions` | P1 | ✅ Working | python, node | unit | 1 test |


## ENC

*Encoding and data transfer*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `ENC.base64.float64` | P0 | ✅ Working | python, node | unit | 5 tests |
| `ENC.base64.int32` | P0 | ✅ Working | python, node | unit | 5 tests |
| `ENC.compression.delta` | P1 | ✅ Working | python, node | integration | 2 tests |
| `ENC.compression.relative` | P1 | ✅ Working | python, node | integration | 1 test |
| `ENC.compression.roundtrip` | P0 | ✅ Working | python, node | invariance | 4 tests |
| `ENC.compression.sinh` | P1 | ✅ Working | python, node | integration | 2 tests |
| `ENC.compression.zip` | P2 | ✅ Working | python, node | integration | 5 tests |


## HIST

*Histogram operations*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `HIST.histogram_1d` | P1 | 📋 Planned | browser | integration | - |
| `HIST.histogram_nd` | P2 | 📋 Planned | browser | integration | - |


## JOIN

*Join and cross-table operations*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `JOIN.cdsjoin.basic` | P0 | ✅ Working | browser | integration | 1 test |
| `JOIN.cdsjoin.index0` | P0 | ✅ Working | browser | integration | 1 test |
| `JOIN.cdsjoin.outer` | P1 | 📋 Planned | browser | integration | - |
| `JOIN.cross_table` | P0 | ✅ Working | python, node | integration | 1 test |


## ONNX

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `ONNX.benchmark.inference` | P1 | ❌ No Tests | node | benchmark | - |
| `ONNX.benchmark.load_time` | P2 | ❌ No Tests | node | benchmark | - |
| `ONNX.export.linear` | P0 | ✅ Working | python | export | 3 tests |
| `ONNX.export.neural_net` | P0 | ✅ Working | python | export | 3 tests |
| `ONNX.export.tree_ensemble` | P0 | ✅ Working | python | export | 3 tests |
| `ONNX.invariance.classification` | P0 | ✅ Working | python, node | invariance | 6 tests |
| `ONNX.invariance.cross_runtime` | P0 | ✅ Working | python, node | invariance | 2 tests |
| `ONNX.invariance.sklearn_vs_ort` | P0 | ✅ Working | python | invariance | 8 tests |
| `ONNX.special_values` | P1 | ✅ Working | python, node | invariance | 12 tests |


## WASM

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `WASM.benchmark.memory_fraction` | P2 | ❌ No Tests | node | benchmark | - |
| `WASM.benchmark.scalar_overhead` | P2 | ❌ No Tests | node | benchmark | - |
| `WASM.benchmark.vector_crossover` | P2 | ❌ No Tests | node | benchmark | - |
| `WASM.compile` | P0 | ✅ Working | native | build | 1 test |
| `WASM.cross_backend_invariance` | P0 | ✅ Working | python, node | invariance | 2 tests |
| `WASM.scalar.arithmetic` | P0 | ✅ Working | node | unit | 1 test |
| `WASM.scalar.conditional` | P0 | ✅ Working | node | unit | 1 test |
| `WASM.scalar.transcendental` | P0 | ✅ Working | node | unit | 1 test |
| `WASM.scalar_vector_consistency` | P0 | ✅ Working | node | invariance | 1 test |
| `WASM.special_values` | P1 | ✅ Working | python, node | invariance | 1 test |
| `WASM.vector.arithmetic` | P0 | ✅ Working | node | unit | 1 test |
| `WASM.vector.conditional` | P0 | ✅ Working | node | unit | 1 test |
| `WASM.vector.transcendental` | P0 | ✅ Working | node | unit | 1 test |


---

## Test Coverage Details

Tests per feature (for traceability). Approval logic: Feature = ✅ Working iff **ALL** tests pass.

<details>
<summary><strong>ALIAS.cdsalias</strong> — CDSAlias (0 tests)</summary>

*Feature planned, not yet implemented*

</details>

<details>
<summary><strong>DSL.arithmetic_expr</strong> — Arithmetic expressions (1 tests)</summary>

- ✅ `test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>DSL.custom_js_func</strong> — CustomJS function execution (1 tests)</summary>

- ✅ `test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>DSL.gather_operation</strong> — Cross-table gather (2 tests)</summary>

- ✅ `test_ClientSideJoin.py::test_gather`
- ✅ `test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>DSL.math_functions</strong> — Math functions (1 tests)</summary>

- ✅ `test_dsl_customjs.py::test_mathutils`

</details>

<details>
<summary><strong>ENC.base64.float64</strong> — Float64Array encoding (5 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_delta`
- ✅ `test_compression_integration.py::test_compression_relative16`
- ✅ `test_compression_integration.py::test_compression_simple`
- ✅ `test_compression_integration.py::test_compression_sinh`
- ✅ `test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>ENC.base64.int32</strong> — Int32Array encoding (5 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_delta`
- ✅ `test_compression_integration.py::test_compression_relative16`
- ✅ `test_compression_integration.py::test_compression_simple`
- ✅ `test_compression_integration.py::test_compression_sinh`
- ✅ `test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>ENC.compression.delta</strong> — Delta/Absolute compression (2 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_delta`
- ✅ `test_compression_integration.py::test_serializationutils`

</details>

<details>
<summary><strong>ENC.compression.relative</strong> — Relative compression (1 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_relative16`

</details>

<details>
<summary><strong>ENC.compression.roundtrip</strong> — Compression roundtrip (4 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_delta`
- ✅ `test_compression_integration.py::test_compression_relative16`
- ✅ `test_compression_integration.py::test_compression_simple`
- ✅ `test_compression_integration.py::test_compression_sinh`

</details>

<details>
<summary><strong>ENC.compression.sinh</strong> — Sinh/Sqrt scaling compression (2 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_sinh`
- ✅ `test_compression_integration.py::test_serializationutils`

</details>

<details>
<summary><strong>ENC.compression.zip</strong> — ZIP compression (5 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_delta`
- ✅ `test_compression_integration.py::test_compression_relative16`
- ✅ `test_compression_integration.py::test_compression_simple`
- ✅ `test_compression_integration.py::test_compression_sinh`
- ✅ `test_compression_integration.py::test_serializationutils`

</details>

<details>
<summary><strong>HIST.histogram_1d</strong> — 1D histogram (0 tests)</summary>

*Feature planned, not yet implemented*

</details>

<details>
<summary><strong>HIST.histogram_nd</strong> — N-D histogram (0 tests)</summary>

*Feature planned, not yet implemented*

</details>

<details>
<summary><strong>JOIN.cdsjoin.basic</strong> — Basic CDSJoin (1 tests)</summary>

- ✅ `test_ClientSideJoin.py::test_join`

</details>

<details>
<summary><strong>JOIN.cdsjoin.index0</strong> — CDSJoin index-0 regression (1 tests)</summary>

- ✅ `test_ClientSideJoin.py::test_join`

</details>

<details>
<summary><strong>JOIN.cdsjoin.outer</strong> — CDSJoin outer join (0 tests)</summary>

*Feature planned, not yet implemented*

</details>

<details>
<summary><strong>JOIN.cross_table</strong> — Multi-CDS cross-table (1 tests)</summary>

- ✅ `test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>ONNX.benchmark.inference</strong> — ONNX inference latency (0 tests)</summary>

*No tests with @pytest.mark.feature marker*

</details>

<details>
<summary><strong>ONNX.benchmark.load_time</strong> — ONNX model load time (0 tests)</summary>

*No tests with @pytest.mark.feature marker*

</details>

<details>
<summary><strong>ONNX.export.linear</strong> — ONNX export linear models (3 tests)</summary>

- ❓ `test_invariance_onnx.py::test_classification_labels_sklearn_vs_python_ort`
- ❓ `test_invariance_onnx.py::test_regression_sklearn_vs_python_ort`
- ✅ `test_invariance_onnx.py::test_smoke_all_models`

</details>

<details>
<summary><strong>ONNX.export.neural_net</strong> — ONNX export neural networks (3 tests)</summary>

- ❓ `test_invariance_onnx.py::test_classification_labels_sklearn_vs_python_ort`
- ❓ `test_invariance_onnx.py::test_regression_sklearn_vs_python_ort`
- ✅ `test_invariance_onnx.py::test_smoke_all_models`

</details>

<details>
<summary><strong>ONNX.export.tree_ensemble</strong> — ONNX export tree ensembles (3 tests)</summary>

- ❓ `test_invariance_onnx.py::test_classification_labels_sklearn_vs_python_ort`
- ❓ `test_invariance_onnx.py::test_regression_sklearn_vs_python_ort`
- ✅ `test_invariance_onnx.py::test_smoke_all_models`

</details>

<details>
<summary><strong>ONNX.invariance.classification</strong> — ONNX classification invariance (6 tests)</summary>

- ❓ `test_invariance_onnx.py::test_classification_labels_sklearn_vs_nodejs_ort`
- ❓ `test_invariance_onnx.py::test_classification_labels_sklearn_vs_python_ort`
- ❓ `test_invariance_onnx.py::test_classification_proba_python_ort_vs_nodejs_ort`
- ❓ `test_onnx_invariance.py::test_classification_labels_sklearn_vs_nodejs_ort`
- ❓ `test_onnx_invariance.py::test_classification_labels_sklearn_vs_python_ort`
- ❓ `test_onnx_invariance.py::test_classification_proba_python_ort_vs_nodejs_ort`

</details>

<details>
<summary><strong>ONNX.invariance.cross_runtime</strong> — ONNX cross-runtime invariance (2 tests)</summary>

- ❓ `test_invariance_onnx.py::test_regression_python_ort_vs_nodejs_ort`
- ❓ `test_onnx_invariance.py::test_regression_python_ort_vs_nodejs_ort`

</details>

<details>
<summary><strong>ONNX.invariance.sklearn_vs_ort</strong> — ONNX sklearn↔ORT invariance (8 tests)</summary>

- ❓ `test_invariance_onnx.py::test_classification_proba_sklearn_vs_python_ort`
- ✅ `test_invariance_onnx.py::test_python_three_way_consistency`
- ❓ `test_invariance_onnx.py::test_regression_sklearn_vs_nodejs_ort`
- ❓ `test_invariance_onnx.py::test_regression_sklearn_vs_python_ort`
- ❓ `test_onnx_invariance.py::test_classification_proba_sklearn_vs_python_ort`
- ✅ `test_onnx_invariance.py::test_python_three_way_consistency`
- ❓ `test_onnx_invariance.py::test_regression_sklearn_vs_nodejs_ort`
- ❓ `test_onnx_invariance.py::test_regression_sklearn_vs_python_ort`

</details>

<details>
<summary><strong>ONNX.special_values</strong> — ONNX IEEE-754 special values (12 tests)</summary>

- ❓ `test_invariance_onnx.py::test_special_values_linear_inf`
- ❓ `test_invariance_onnx.py::test_special_values_linear_nan`
- ✅ `test_invariance_onnx.py::test_special_values_mixed_all_models`
- ❓ `test_invariance_onnx.py::test_special_values_mlp_nan`
- ❓ `test_invariance_onnx.py::test_special_values_tree_inf`
- ❓ `test_invariance_onnx.py::test_special_values_tree_nan`
- ❓ `test_onnx_invariance.py::test_special_values_linear_inf`
- ❓ `test_onnx_invariance.py::test_special_values_linear_nan`
- ✅ `test_onnx_invariance.py::test_special_values_mixed_all_models`
- ❓ `test_onnx_invariance.py::test_special_values_mlp_nan`
- ❓ `test_onnx_invariance.py::test_special_values_tree_inf`
- ❓ `test_onnx_invariance.py::test_special_values_tree_nan`

</details>

<details>
<summary><strong>WASM.benchmark.memory_fraction</strong> — WASM benchmark memory fraction (0 tests)</summary>

*No tests with @pytest.mark.feature marker*

</details>

<details>
<summary><strong>WASM.benchmark.scalar_overhead</strong> — WASM benchmark scalar overhead (0 tests)</summary>

*No tests with @pytest.mark.feature marker*

</details>

<details>
<summary><strong>WASM.benchmark.vector_crossover</strong> — WASM benchmark vector crossover (0 tests)</summary>

*No tests with @pytest.mark.feature marker*

</details>

<details>
<summary><strong>WASM.compile</strong> — WASM compilation (1 tests)</summary>

- ✅ `test_wasm_invariance.py::test_wasm_invariance_all`

</details>

<details>
<summary><strong>WASM.cross_backend_invariance</strong> — WASM cross-backend invariance (2 tests)</summary>

- ❓ `test_wasm_invariance.py::test_wasm_correctness`
- ✅ `test_wasm_invariance.py::test_wasm_invariance_all`

</details>

<details>
<summary><strong>WASM.scalar.arithmetic</strong> — WASM scalar arithmetic (1 tests)</summary>

- ❓ `test_wasm_invariance.py::test_wasm_correctness`

</details>

<details>
<summary><strong>WASM.scalar.conditional</strong> — WASM scalar conditional (1 tests)</summary>

- ❓ `test_wasm_invariance.py::test_wasm_correctness`

</details>

<details>
<summary><strong>WASM.scalar.transcendental</strong> — WASM scalar transcendental (1 tests)</summary>

- ❓ `test_wasm_invariance.py::test_wasm_correctness`

</details>

<details>
<summary><strong>WASM.scalar_vector_consistency</strong> — WASM scalar-vector consistency (1 tests)</summary>

- ✅ `test_wasm_invariance.py::test_wasm_scalar_vector_consistency`

</details>

<details>
<summary><strong>WASM.special_values</strong> — WASM IEEE-754 special values (1 tests)</summary>

- ❓ `test_wasm_invariance.py::test_wasm_special_values`

</details>

<details>
<summary><strong>WASM.vector.arithmetic</strong> — WASM vector arithmetic (1 tests)</summary>

- ❓ `test_wasm_invariance.py::test_wasm_correctness`

</details>

<details>
<summary><strong>WASM.vector.conditional</strong> — WASM vector conditional (1 tests)</summary>

- ❓ `test_wasm_invariance.py::test_wasm_correctness`

</details>

<details>
<summary><strong>WASM.vector.transcendental</strong> — WASM vector transcendental (1 tests)</summary>

- ❓ `test_wasm_invariance.py::test_wasm_correctness`

</details>

---

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Working | 31 | 77.5% |
| 🧨 Broken | 0 | 0.0% |
| ⚠️ Known Issue | 0 | 0.0% |
| 📋 Planned | 4 | 10.0% |
| ❌ No Tests | 5 | 12.5% |
| ❓ Unknown | 0 | 0.0% |
| **Total** | **40** | **100%** |

---

## Legend

| Status | Meaning |
|--------|---------|
| ✅ Working | All tests pass |
| 🧨 Broken | At least one test fails |
| ⚠️ Known Issue | Expected failure (xfail) |
| 📋 Planned | Feature planned, not yet tested |
| ❌ No Tests | No tests cover this feature |
| ❓ Unknown | Test status unclear |

## Priority Levels

| Priority | Meaning |
|----------|---------|
| P0 | Critical - blocks release |
| P1 | Important - should fix before release |
| P2 | Nice to have - can defer |

---

*Auto-generated by `scripts/generate_capability_matrix.py`*

*Phase 0.1.A provides coverage accounting. Invariance validation requires Phase 0.1.B.*