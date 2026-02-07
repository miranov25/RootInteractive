# CAPABILITY_MATRIX

**Generated:** 2026-02-06 21:59:25

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

---

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Working | 14 | 77.8% |
| 🧨 Broken | 0 | 0.0% |
| ⚠️ Known Issue | 0 | 0.0% |
| 📋 Planned | 4 | 22.2% |
| ❌ No Tests | 0 | 0.0% |
| ❓ Unknown | 0 | 0.0% |
| **Total** | **18** | **100%** |

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