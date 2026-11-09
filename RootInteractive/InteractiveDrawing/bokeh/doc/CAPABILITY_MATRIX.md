# CAPABILITY_MATRIX

**Generated:** 2026-11-09 09:50:44

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
| `DSL.gather_operation` | P0 | ✅ Working | python, node | integration | 3 tests |
| `DSL.math_functions` | P1 | ✅ Working | python, node | unit | 1 test |


## ENC

*Encoding and data transfer*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `ENC.base64.float64` | P0 | ✅ Working | python, node | unit | 6 tests |
| `ENC.base64.int32` | P0 | ✅ Working | python, node | unit | 6 tests |
| `ENC.compression.delta` | P1 | ✅ Working | python, node | integration | 4 tests |
| `ENC.compression.relative` | P1 | ✅ Working | python, node | integration | 2 tests |
| `ENC.compression.roundtrip` | P0 | ✅ Working | python, node | invariance | 5 tests |
| `ENC.compression.sinh` | P1 | ✅ Working | python, node | integration | 3 tests |
| `ENC.compression.zip` | P2 | ✅ Working | python, node | integration | 6 tests |


## FILTER

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `FILTER.intersection` | P1 | ⚠️ Known Issue | browser | integration | 709 tests |
| `FILTER.range` | P1 | ✅ Working | browser | integration | 7 tests |


## HIST

*Histogram operations*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `HIST.histogram_1d` | P1 | ⚠️ Known Issue | browser | integration | 13 tests |
| `HIST.histogram_nd` | P2 | ⚠️ Known Issue | browser | integration | 724 tests |
| `HIST.profile` | P2 | ❓ Partly verified (response only) | browser | integration | 1 test |


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
| `ONNX.invariance.classification` | P0 | 🧨 Broken | python, node | invariance | 6 tests |
| `ONNX.invariance.cross_runtime` | P0 | 🧨 Broken | python, node | invariance | 2 tests |
| `ONNX.invariance.sklearn_vs_ort` | P0 | 🧨 Broken | python | invariance | 8 tests |
| `ONNX.special_values` | P1 | 🧨 Broken | python, node | invariance | 12 tests |


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

## 🧨 Broken Features Detail

**ONNX classification invariance:**
- `RootInteractive/onnx/test_onnx_invariance.py::test_classification_labels_sklearn_vs_nodejs_ort[logistic]`
- `RootInteractive/onnx/test_onnx_invariance.py::test_classification_proba_python_ort_vs_nodejs_ort[mlp_clf]`
- `RootInteractive/onnx/test_invariance_onnx.py::test_classification_labels_sklearn_vs_nodejs_ort[logistic]`
- `RootInteractive/onnx/test_invariance_onnx.py::test_classification_proba_python_ort_vs_nodejs_ort[logistic]`

**ONNX cross-runtime invariance:**
- `RootInteractive/onnx/test_onnx_invariance.py::test_regression_python_ort_vs_nodejs_ort[mlp]`
- `RootInteractive/onnx/test_invariance_onnx.py::test_regression_python_ort_vs_nodejs_ort[rf10]`

**ONNX sklearn↔ORT invariance:**
- `RootInteractive/onnx/test_onnx_invariance.py::test_regression_sklearn_vs_nodejs_ort[rf10]`
- `RootInteractive/onnx/test_invariance_onnx.py::test_regression_sklearn_vs_nodejs_ort[rf50]`

**ONNX IEEE-754 special values:**
- `RootInteractive/onnx/test_onnx_invariance.py::test_special_values_tree_nan[rf10]`
- `RootInteractive/onnx/test_onnx_invariance.py::test_special_values_tree_inf[rf10]`
- `test_onnx_invariance.py::test_special_values_mixed_all_models`
- `RootInteractive/onnx/test_invariance_onnx.py::test_special_values_tree_nan[rf10]`
- `RootInteractive/onnx/test_invariance_onnx.py::test_special_values_tree_inf[rf_clf]`
- ... and 1 more

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
<summary><strong>DSL.gather_operation</strong> — Cross-table gather (3 tests)</summary>

- ✅ `test_ClientSideJoin.py::test_gather`
- ✅ `test_ClientSideJoin.py::test_gather_realistic`
- ✅ `test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>DSL.math_functions</strong> — Math functions (1 tests)</summary>

- ✅ `test_dsl_customjs.py::test_mathutils`

</details>

<details>
<summary><strong>ENC.base64.float64</strong> — Float64Array encoding (6 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_code`
- ✅ `test_compression_integration.py::test_compression_delta`
- ✅ `test_compression_integration.py::test_compression_relative16`
- ✅ `test_compression_integration.py::test_compression_simple`
- ✅ `test_compression_integration.py::test_compression_sinh`
- ✅ `test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>ENC.base64.int32</strong> — Int32Array encoding (6 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_code`
- ✅ `test_compression_integration.py::test_compression_delta`
- ✅ `test_compression_integration.py::test_compression_relative16`
- ✅ `test_compression_integration.py::test_compression_simple`
- ✅ `test_compression_integration.py::test_compression_sinh`
- ✅ `test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>ENC.compression.delta</strong> — Delta/Absolute compression (4 tests)</summary>

- ✅ `test_Compression.py::test_CompressionSampleDelta`
- ✅ `test_Compression.py::test_CompressionSampleDeltaCode`
- ✅ `test_compression_integration.py::test_compression_delta`
- ✅ `test_compression_integration.py::test_serializationutils`

</details>

<details>
<summary><strong>ENC.compression.relative</strong> — Relative compression (2 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_code`
- ✅ `test_compression_integration.py::test_compression_relative16`

</details>

<details>
<summary><strong>ENC.compression.roundtrip</strong> — Compression roundtrip (5 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_code`
- ✅ `test_compression_integration.py::test_compression_delta`
- ✅ `test_compression_integration.py::test_compression_relative16`
- ✅ `test_compression_integration.py::test_compression_simple`
- ✅ `test_compression_integration.py::test_compression_sinh`

</details>

<details>
<summary><strong>ENC.compression.sinh</strong> — Sinh/Sqrt scaling compression (3 tests)</summary>

- ✅ `test_Compression.py::test_CompressionSampleSinh`
- ✅ `test_compression_integration.py::test_compression_sinh`
- ✅ `test_compression_integration.py::test_serializationutils`

</details>

<details>
<summary><strong>ENC.compression.zip</strong> — ZIP compression (6 tests)</summary>

- ✅ `test_compression_integration.py::test_compression_code`
- ✅ `test_compression_integration.py::test_compression_delta`
- ✅ `test_compression_integration.py::test_compression_relative16`
- ✅ `test_compression_integration.py::test_compression_simple`
- ✅ `test_compression_integration.py::test_compression_sinh`
- ✅ `test_compression_integration.py::test_serializationutils`

</details>

<details>
<summary><strong>FILTER.intersection</strong> — Filter intersection (709 tests)</summary>

- ⚠️ headless scenario `families/b1_sweep_n100_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n100_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n101_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n101_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n102_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n102_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n103_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n103_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n104_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n104_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n105_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n105_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n106_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n106_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n107_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n107_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n108_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n108_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n109_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n109_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n10_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n10_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n110_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n110_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n111_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n111_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n112_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n112_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n113_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n113_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n114_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n114_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n115_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n115_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n116_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n116_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n117_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n117_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n118_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n118_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n119_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n119_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n11_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n11_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n120_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n120_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n121_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n121_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n122_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n122_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n123_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n123_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n124_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n124_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n125_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n125_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n126_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n126_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n127_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n127_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n128_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n128_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n129_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n129_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n12_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n12_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n130_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n130_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n131_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n131_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n132_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n132_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n133_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n133_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n134_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n134_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n135_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n135_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n136_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n136_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n137_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n137_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n138_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n138_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n139_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n139_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n13_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n13_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n140_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n140_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n141_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n141_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n142_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n142_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n143_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n143_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n144_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n144_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n145_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n145_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n146_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n146_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n147_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n147_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n148_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n148_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n149_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n149_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n14_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n14_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n150_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n150_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n151_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n151_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n152_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n152_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n153_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n153_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n154_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n154_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n155_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n155_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n156_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n156_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n157_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n157_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n158_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n158_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n159_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n159_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n15_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n15_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n160_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n160_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n161_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n161_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n162_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n162_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n163_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n163_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n164_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n164_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n165_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n165_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n166_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n166_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n167_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n167_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n168_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n168_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n169_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n169_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n16_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n16_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n170_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n170_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n171_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n171_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n172_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n172_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n173_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n173_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n174_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n174_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n175_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n175_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n176_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n176_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n177_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n177_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n178_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n178_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n179_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n179_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n17_first_finite` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n17_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n180_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n180_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n181_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n181_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n182_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n182_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n183_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n183_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n184_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n184_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n185_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n185_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n186_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n186_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n187_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n187_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n188_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n188_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n189_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n189_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n18_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n18_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n190_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n190_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n191_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n191_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n192_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n192_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n193_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n193_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n194_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n194_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n195_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n195_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n196_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n196_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n197_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n197_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n198_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n198_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n199_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n199_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n19_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n19_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n1_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n1_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n200_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n200_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n20_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n20_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n21_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n21_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n22_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n22_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n23_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n23_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n24_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n24_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n25_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n25_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n26_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n26_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n27_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n27_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n28_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n28_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n29_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n29_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n2_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n2_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n30_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n30_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n31_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n31_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n32_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n32_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n33_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n33_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n34_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n34_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n35_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n35_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n36_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n36_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n37_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n37_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n38_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n38_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n39_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n39_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n3_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n3_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n40_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n40_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n41_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n41_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n42_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n42_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n43_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n43_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n44_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n44_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n45_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n45_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n46_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n46_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n47_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n47_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n48_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n48_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n49_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n49_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n4_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n4_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n50_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n50_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n51_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n51_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n52_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n52_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n53_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n53_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n54_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n54_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n55_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n55_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n56_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n56_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n57_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n57_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n58_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n58_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n59_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n59_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n5_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n5_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n60_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n60_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n61_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n61_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n62_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n62_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n63_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n63_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n64_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n64_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n65_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n65_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n66_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n66_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n67_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n67_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n68_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n68_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n69_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n69_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n6_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n6_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n70_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n70_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n71_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n71_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n72_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n72_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n73_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n73_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n74_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n74_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n75_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n75_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n76_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n76_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n77_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n77_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n78_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n78_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n79_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n79_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n7_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n7_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n80_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n80_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n81_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n81_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n82_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n82_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n83_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n83_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n84_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n84_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n85_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n85_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n86_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n86_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n87_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n87_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n88_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n88_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n89_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n89_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n8_first_finite` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n8_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n90_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n90_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n91_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n91_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n92_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n92_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n93_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n93_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n94_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n94_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n95_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n95_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n96_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n96_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n97_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n97_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n98_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n98_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n99_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n99_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n9_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n9_first_nan` (node) — known defect B1
- ✅ headless scenario `families/filters_k2_n20000_order_ab` (node)
- ⚠️ headless scenario `families/filters_k2_n20001_order_ab` (node) — known defect B1
- ✅ headless scenario `families/filters_k2_n32_order_ab` (node)
- ✅ headless scenario `families/filters_k2_n32_order_ba` (node)
- ⚠️ headless scenario `families/filters_k2_n33_order_ab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k2_n33_order_ba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_abc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_acb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_bac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_bca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_cab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_cba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_abc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_acb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_bac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_bca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_cab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_cba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_abcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_abdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_acbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_acdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_adbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_adcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_bacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_badc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_bcad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_bcda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_bdac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_bdca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cbad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cbda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cdab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cdba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dbac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dbca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dcab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dcba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_abcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_abdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_acbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_acdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_adbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_adcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_bacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_badc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_bcad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_bcda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_bdac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_bdca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cbad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cbda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cdab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cdba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dbac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dbca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dcab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dcba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abcde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abced` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abdce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abdec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abecd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abedc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acbde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acbed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acdbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acdeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acebd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acedb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adbce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adbec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adcbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adceb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adebc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adecb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aebcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aebdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aecbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aecdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aedbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aedcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bacde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_baced` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_badce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_badec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_baecd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_baedc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bcade` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bcaed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bcdae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bcdea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bcead` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bceda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdace` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdaec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdcae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdcea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdeac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdeca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_beacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_beadc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_becad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_becda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bedac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bedca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cabde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cabed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cadbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cadeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_caebd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_caedb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbade` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbaed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbdae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbdea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbead` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbeda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdabe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdaeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdbae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdbea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdeab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdeba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ceabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ceadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cebad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cebda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cedab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cedba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dabce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dabec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dacbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_daceb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_daebc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_daecb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbace` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbaec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbcae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbcea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbeac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbeca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dcabe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dcaeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dcbae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dcbea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dceab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dceba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_deabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_deacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_debac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_debca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_decab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_decba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eabcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eabdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eacbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eacdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eadbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eadcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebadc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebcad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebcda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebdac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebdca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecbad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecbda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecdab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecdba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edbac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edbca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edcab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edcba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abcde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abced` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abdce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abdec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abecd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abedc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acbde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acbed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acdbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acdeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acebd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acedb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adbce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adbec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adcbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adceb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adebc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adecb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aebcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aebdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aecbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aecdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aedbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aedcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bacde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_baced` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_badce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_badec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_baecd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_baedc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bcade` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bcaed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bcdae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bcdea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bcead` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bceda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdace` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdaec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdcae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdcea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdeac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdeca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_beacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_beadc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_becad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_becda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bedac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bedca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cabde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cabed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cadbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cadeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_caebd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_caedb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbade` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbaed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbdae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbdea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbead` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbeda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdabe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdaeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdbae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdbea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdeab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdeba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ceabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ceadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cebad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cebda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cedab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cedba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dabce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dabec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dacbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_daceb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_daebc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_daecb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbace` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbaec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbcae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbcea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbeac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbeca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dcabe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dcaeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dcbae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dcbea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dceab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dceba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_deabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_deacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_debac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_debca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_decab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_decba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eabcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eabdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eacbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eacdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eadbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eadcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebadc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebcad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebcda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebdac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebdca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecbad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecbda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecdab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecdba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edbac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edbca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edcab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edcba` (node) — known defect B1
- ✅ headless scenario `histograms/claude10_two_ranges_alternating_moves` (browser, node)
- ⚠️ headless scenario `known_defects/b1_two_filters_missing_values` (browser, node) — known defect B1
- ⚠️ headless scenario `known_defects/claude3_b1_three_filters_missing_values` (browser, node) — known defect B1

</details>

<details>
<summary><strong>FILTER.range</strong> — Range filter (7 tests)</summary>

- ✅ headless scenario `chain/deep_widget_filter_alias_stack_profile` (browser, node)
- ✅ headless scenario `histograms/gpt24_empty_selection_shifted_range` (browser, node)
- ✅ headless scenario `histograms/plain_1d_2d_one_range_two_moves` (browser, node)
- ✅ headless scenario `known_defects/b8_1d_histogram_counts_nan_in_bin0` (browser, node)
- ✅ headless scenario `known_defects/claude10_b8_weighted_1d_histogram_nan_in_bin0` (browser, node)
- ✅ headless scenario `known_defects/claude3_b8_weighted_1d_histogram` (browser, node)
- ✅ headless scenario `selection/rb1_nan_excluded_at_full_bounds` (browser, node)

</details>

<details>
<summary><strong>HIST.histogram_1d</strong> — 1D histogram (13 tests)</summary>

- ✅ headless scenario `families/histograms_node_claude3_b10_6_fast` (node)
- ✅ headless scenario `families/histograms_node_claude3_b10_6_full` (node)
- ✅ headless scenario `families/histograms_node_odd_7_fast` (node)
- ✅ headless scenario `families/histograms_node_odd_7_full` (node)
- ✅ headless scenario `families/histograms_node_unit_10_fast` (node)
- ✅ headless scenario `families/histograms_node_unit_10_full` (node)
- ✅ headless scenario `histograms/claude10_two_ranges_alternating_moves` (browser, node)
- ✅ headless scenario `histograms/gpt24_empty_selection_shifted_range` (browser, node)
- ✅ headless scenario `histograms/plain_1d_2d_one_range_two_moves` (browser, node)
- ⚠️ headless scenario `known_defects/b2_1d_histogram_ignores_selection_at_load` (browser, node) — known defect B2
- ⚠️ headless scenario `known_defects/b8_1d_histogram_counts_nan_in_bin0` (browser, node) — known defect B8
- ⚠️ headless scenario `known_defects/claude10_b8_weighted_1d_histogram_nan_in_bin0` (browser, node) — known defect B8
- ⚠️ headless scenario `known_defects/claude3_b8_weighted_1d_histogram` (browser, node) — known defect B8

</details>

<details>
<summary><strong>HIST.histogram_nd</strong> — N-D histogram (724 tests)</summary>

- ✅ headless scenario `chain/deep_widget_filter_alias_stack_profile` (browser, node)
- ⚠️ headless scenario `families/b1_sweep_n100_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n100_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n101_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n101_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n102_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n102_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n103_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n103_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n104_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n104_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n105_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n105_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n106_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n106_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n107_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n107_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n108_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n108_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n109_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n109_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n10_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n10_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n110_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n110_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n111_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n111_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n112_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n112_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n113_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n113_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n114_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n114_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n115_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n115_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n116_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n116_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n117_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n117_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n118_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n118_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n119_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n119_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n11_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n11_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n120_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n120_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n121_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n121_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n122_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n122_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n123_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n123_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n124_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n124_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n125_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n125_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n126_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n126_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n127_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n127_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n128_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n128_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n129_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n129_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n12_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n12_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n130_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n130_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n131_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n131_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n132_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n132_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n133_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n133_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n134_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n134_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n135_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n135_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n136_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n136_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n137_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n137_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n138_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n138_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n139_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n139_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n13_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n13_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n140_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n140_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n141_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n141_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n142_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n142_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n143_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n143_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n144_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n144_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n145_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n145_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n146_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n146_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n147_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n147_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n148_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n148_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n149_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n149_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n14_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n14_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n150_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n150_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n151_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n151_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n152_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n152_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n153_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n153_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n154_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n154_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n155_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n155_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n156_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n156_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n157_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n157_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n158_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n158_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n159_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n159_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n15_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n15_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n160_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n160_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n161_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n161_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n162_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n162_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n163_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n163_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n164_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n164_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n165_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n165_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n166_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n166_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n167_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n167_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n168_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n168_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n169_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n169_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n16_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n16_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n170_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n170_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n171_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n171_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n172_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n172_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n173_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n173_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n174_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n174_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n175_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n175_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n176_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n176_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n177_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n177_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n178_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n178_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n179_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n179_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n17_first_finite` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n17_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n180_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n180_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n181_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n181_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n182_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n182_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n183_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n183_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n184_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n184_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n185_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n185_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n186_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n186_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n187_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n187_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n188_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n188_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n189_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n189_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n18_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n18_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n190_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n190_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n191_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n191_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n192_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n192_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n193_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n193_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n194_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n194_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n195_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n195_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n196_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n196_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n197_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n197_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n198_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n198_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n199_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n199_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n19_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n19_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n1_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n1_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n200_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n200_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n20_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n20_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n21_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n21_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n22_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n22_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n23_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n23_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n24_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n24_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n25_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n25_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n26_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n26_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n27_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n27_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n28_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n28_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n29_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n29_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n2_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n2_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n30_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n30_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n31_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n31_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n32_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n32_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n33_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n33_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n34_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n34_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n35_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n35_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n36_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n36_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n37_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n37_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n38_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n38_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n39_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n39_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n3_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n3_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n40_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n40_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n41_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n41_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n42_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n42_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n43_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n43_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n44_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n44_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n45_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n45_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n46_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n46_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n47_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n47_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n48_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n48_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n49_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n49_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n4_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n4_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n50_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n50_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n51_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n51_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n52_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n52_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n53_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n53_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n54_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n54_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n55_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n55_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n56_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n56_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n57_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n57_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n58_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n58_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n59_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n59_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n5_first_finite` (node)
- ✅ headless scenario `families/b1_sweep_n5_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n60_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n60_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n61_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n61_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n62_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n62_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n63_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n63_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n64_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n64_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n65_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n65_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n66_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n66_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n67_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n67_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n68_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n68_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n69_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n69_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n6_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n6_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n70_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n70_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n71_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n71_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n72_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n72_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n73_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n73_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n74_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n74_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n75_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n75_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n76_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n76_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n77_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n77_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n78_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n78_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n79_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n79_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n7_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n7_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n80_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n80_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n81_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n81_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n82_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n82_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n83_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n83_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n84_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n84_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n85_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n85_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n86_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n86_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n87_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n87_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n88_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n88_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n89_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n89_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n8_first_finite` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n8_first_nan` (node)
- ⚠️ headless scenario `families/b1_sweep_n90_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n90_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n91_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n91_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n92_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n92_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n93_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n93_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n94_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n94_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n95_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n95_first_nan` (node) — known defect B1
- ✅ headless scenario `families/b1_sweep_n96_first_finite` (node)
- ⚠️ headless scenario `families/b1_sweep_n96_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n97_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n97_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n98_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n98_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n99_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n99_first_nan` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n9_first_finite` (node) — known defect B1
- ⚠️ headless scenario `families/b1_sweep_n9_first_nan` (node) — known defect B1
- ✅ headless scenario `families/filters_k2_n20000_order_ab` (node)
- ⚠️ headless scenario `families/filters_k2_n20001_order_ab` (node) — known defect B1
- ✅ headless scenario `families/filters_k2_n32_order_ab` (node)
- ✅ headless scenario `families/filters_k2_n32_order_ba` (node)
- ⚠️ headless scenario `families/filters_k2_n33_order_ab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k2_n33_order_ba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_abc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_acb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_bac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_bca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_cab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n32_order_cba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_abc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_acb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_bac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_bca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_cab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k3_n33_order_cba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_abcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_abdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_acbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_acdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_adbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_adcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_bacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_badc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_bcad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_bcda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_bdac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_bdca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cbad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cbda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cdab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_cdba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dbac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dbca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dcab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n32_order_dcba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_abcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_abdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_acbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_acdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_adbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_adcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_bacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_badc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_bcad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_bcda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_bdac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_bdca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cbad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cbda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cdab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_cdba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dbac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dbca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dcab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k4_n33_order_dcba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abcde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abced` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abdce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abdec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abecd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_abedc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acbde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acbed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acdbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acdeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acebd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_acedb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adbce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adbec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adcbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adceb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adebc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_adecb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aebcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aebdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aecbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aecdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aedbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_aedcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bacde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_baced` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_badce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_badec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_baecd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_baedc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bcade` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bcaed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bcdae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bcdea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bcead` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bceda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdace` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdaec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdcae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdcea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdeac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bdeca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_beacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_beadc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_becad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_becda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bedac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_bedca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cabde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cabed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cadbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cadeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_caebd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_caedb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbade` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbaed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbdae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbdea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbead` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cbeda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdabe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdaeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdbae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdbea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdeab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cdeba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ceabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ceadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cebad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cebda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cedab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_cedba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dabce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dabec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dacbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_daceb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_daebc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_daecb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbace` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbaec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbcae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbcea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbeac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dbeca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dcabe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dcaeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dcbae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dcbea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dceab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_dceba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_deabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_deacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_debac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_debca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_decab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_decba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eabcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eabdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eacbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eacdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eadbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_eadcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebadc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebcad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebcda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebdac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ebdca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecbad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecbda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecdab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_ecdba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edbac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edbca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edcab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n32_order_edcba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abcde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abced` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abdce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abdec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abecd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_abedc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acbde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acbed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acdbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acdeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acebd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_acedb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adbce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adbec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adcbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adceb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adebc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_adecb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aebcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aebdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aecbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aecdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aedbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_aedcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bacde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_baced` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_badce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_badec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_baecd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_baedc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bcade` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bcaed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bcdae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bcdea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bcead` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bceda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdace` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdaec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdcae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdcea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdeac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bdeca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_beacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_beadc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_becad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_becda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bedac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_bedca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cabde` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cabed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cadbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cadeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_caebd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_caedb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbade` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbaed` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbdae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbdea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbead` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cbeda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdabe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdaeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdbae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdbea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdeab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cdeba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ceabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ceadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cebad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cebda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cedab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_cedba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dabce` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dabec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dacbe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_daceb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_daebc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_daecb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbace` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbaec` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbcae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbcea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbeac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dbeca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dcabe` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dcaeb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dcbae` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dcbea` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dceab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_dceba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_deabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_deacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_debac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_debca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_decab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_decba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eabcd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eabdc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eacbd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eacdb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eadbc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_eadcb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebacd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebadc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebcad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebcda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebdac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ebdca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecabd` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecadb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecbad` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecbda` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecdab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_ecdba` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edabc` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edacb` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edbac` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edbca` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edcab` (node) — known defect B1
- ⚠️ headless scenario `families/filters_k5_n33_order_edcba` (node) — known defect B1
- ⚠️ headless scenario `families/histograms_node_claude3_b10_6_fast` (node)
- ⚠️ headless scenario `families/histograms_node_claude3_b10_6_full` (node)
- ✅ headless scenario `families/histograms_node_odd_7_fast` (node)
- ✅ headless scenario `families/histograms_node_odd_7_full` (node)
- ✅ headless scenario `families/histograms_node_unit_10_fast` (node)
- ✅ headless scenario `families/histograms_node_unit_10_full` (node)
- ✅ headless scenario `histograms/claude10_two_ranges_alternating_moves` (browser, node)
- ✅ headless scenario `histograms/claude3_3d_widget_with_explicit_limits` (browser, node)
- ✅ headless scenario `histograms/gpt24_empty_selection_shifted_range` (browser, node)
- ✅ headless scenario `histograms/plain_1d_2d_one_range_two_moves` (browser, node)
- ⚠️ headless scenario `known_defects/b1_two_filters_missing_values` (browser, node) — known defect B1
- ✅ headless scenario `known_defects/b2_1d_histogram_ignores_selection_at_load` (browser, node)
- ✅ headless scenario `known_defects/b8_1d_histogram_counts_nan_in_bin0` (browser, node)
- ✅ headless scenario `known_defects/claude10_b8_weighted_1d_histogram_nan_in_bin0` (browser, node)
- ⚠️ headless scenario `known_defects/claude3_b1_three_filters_missing_values` (browser, node) — known defect B1
- ✅ headless scenario `known_defects/claude3_b8_weighted_1d_histogram` (browser, node)
- ✅ headless scenario `selection/rb1_nan_excluded_at_full_bounds` (browser, node)

</details>

<details>
<summary><strong>HIST.profile</strong> — Histogram profile (1 tests)</summary>

- ✅ headless scenario `chain/deep_widget_filter_alias_stack_profile` (browser, node)

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
- ❌ `test_invariance_onnx.py::test_special_values_mixed_all_models`
- ❓ `test_invariance_onnx.py::test_special_values_mlp_nan`
- ❓ `test_invariance_onnx.py::test_special_values_tree_inf`
- ❓ `test_invariance_onnx.py::test_special_values_tree_nan`
- ❓ `test_onnx_invariance.py::test_special_values_linear_inf`
- ❓ `test_onnx_invariance.py::test_special_values_linear_nan`
- ❌ `test_onnx_invariance.py::test_special_values_mixed_all_models`
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
| ✅ Working | 28 | 65.1% |
| 🧨 Broken | 4 | 9.3% |
| ⚠️ Known Issue | 3 | 7.0% |
| 📋 Planned | 2 | 4.7% |
| ❌ No Tests | 5 | 11.6% |
| ❓ Unknown | 1 | 2.3% |
| **Total** | **43** | **100%** |

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