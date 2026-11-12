# CAPABILITY_MATRIX

**Generated:** 2026-11-12 23:16:06

**Generator:** `scripts/generate_capability_matrix.py`

**Phase:** 0.1.A

> This matrix shows test coverage for RootInteractive features.
> Status is derived from pytest outcomes using `pytest-json-report`.

---

## ALIAS

*Column aliasing operations*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `ALIAS.cdsalias` | P1 | ⚠️ Known Issue | browser, node | integration | 5 tests, 24 cases |
| `ALIAS.chain` | P0 | ⚠️ Known Issue | node | integration | 5 tests, 5 cases |
| `ALIAS.change_propagation` | P0 | ⚠️ Known Issue | node | integration | 6 tests, 6 cases |
| `ALIAS.context` | P1 | ✅ Working | node | integration | 1 test, 1 case |
| `ALIAS.parameter` | P0 | ⚠️ Known Issue | node | integration | 5 tests, 5 cases |


## DASHBOARD

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `DASHBOARD.histogram_only` | P0 | ⚠️ Known Issue | browser | e2e | 5 tests, 5 cases |
| `DASHBOARD.weights_reference` | P0 | ⚠️ Known Issue | browser | e2e | 4 tests, 4 cases |


## DISPLAY

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `DISPLAY.axis_label` | P2 | ⚠️ Known Issue | node, browser | integration | 4 tests, 9 cases |
| `DISPLAY.downsampler` | P1 | ⚠️ Known Issue | node | integration | 6 tests, 6 cases |
| `DISPLAY.graph3d` | P2 | ⚠️ Known Issue | node, browser | integration | 11 tests, 11 cases |
| `DISPLAY.lazy_tabs` | P2 | ✅ Working | node, browser | integration | 2 tests, 2 cases |


## DSL

*Domain-Specific Language operations*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `DSL.arithmetic_expr` | P0 | ✅ Working | python, node | unit | 3 tests, 18 cases |
| `DSL.custom_js_func` | P0 | ✅ Working | node, browser | integration | 3 tests, 3 cases |
| `DSL.gather_operation` | P0 | ⚠️ Known Issue | python, node | integration | 11 tests, 12 cases |
| `DSL.math_functions` | P1 | ⚠️ Known Issue | python, node | unit | 4 tests, 20 cases |
| `DSL.server_client_parity` | P0 | ⚠️ Known Issue | python, node | integration | 5 tests, 14 cases |


## ENC

*Encoding and data transfer*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `ENC.base64.float64` | P0 | ✅ Working | python, node | unit | 6 tests, 6 cases |
| `ENC.base64.int32` | P0 | ✅ Working | python, node | unit | 6 tests, 6 cases |
| `ENC.cds_compress.data_replace` | P1 | ⚠️ Known Issue | node | integration | 3 tests, 3 cases |
| `ENC.cds_compress.length_guard` | P2 | ⚠️ Known Issue | node | integration | 4 tests, 4 cases |
| `ENC.cds_compress.page_decode` | P1 | ✅ Working | node | integration | 3 tests, 3 cases |
| `ENC.compression.delta` | P1 | ✅ Working | python, node | integration | 4 tests, 4 cases |
| `ENC.compression.relative` | P1 | ✅ Working | python, node | integration | 2 tests, 2 cases |
| `ENC.compression.roundtrip` | P0 | ✅ Working | python, node | invariance | 5 tests, 5 cases |
| `ENC.compression.sinh` | P1 | ✅ Working | python, node | integration | 3 tests, 3 cases |
| `ENC.compression.zip` | P2 | ✅ Working | python, node | integration | 6 tests, 6 cases |


## FILTER

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `FILTER.intersection` | P1 | ⚠️ Known Issue | browser | integration | 710 tests, 2 cases |
| `FILTER.range` | P1 | ⚠️ Known Issue | browser, node | integration | 21 tests, 21 cases |
| `FILTER.text_query` | P1 | ⚠️ Known Issue | node, browser | integration | 14 tests, 16 cases |


## HIST

*Histogram operations*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `HIST.histogram_1d` | P1 | ⚠️ Known Issue | browser, node | integration | 18 tests, 10 cases |
| `HIST.histogram_nd` | P2 | ⚠️ Known Issue | browser, node | integration | 744 tests, 37 cases |
| `HIST.profile` | P2 | ❓ Partly verified (response only) | browser, node | integration | 7 tests, 6 cases |
| `HIST.profile_unbinned` | P1 | ⚠️ Known Issue | node | integration | 10 tests, 13 cases |
| `HIST.stack` | P1 | ✅ Working | node | integration | 5 tests, 5 cases |


## JOIN

*Join and cross-table operations*

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `JOIN.cdsjoin.basic` | P0 | ⚠️ Known Issue | browser, node | integration | 19 tests, 54 cases |
| `JOIN.cdsjoin.index0` | P0 | ⚠️ Known Issue | browser, node | integration | 4 tests, 4 cases |
| `JOIN.cdsjoin.outer` | P1 | ⚠️ Known Issue | browser, node | integration | 5 tests, 18 cases |
| `JOIN.cdsjoin.reverse_index` | P1 | ⚠️ Known Issue | node | integration | 3 tests, 3 cases |
| `JOIN.change_propagation` | P0 | ✅ Working | node | integration | 3 tests, 8 cases |
| `JOIN.cross_table` | P0 | ✅ Working | python, node | integration | 1 test, 1 case |
| `JOIN.multi_table_chain` | P1 | ⚠️ Known Issue | node, browser | integration | 7 tests, 10 cases |
| `JOIN.prototype_parity` | P1 | ⚠️ Known Issue | node | integration | 6 tests, 6 cases |
| `JOIN.recursive` | P0 | ⚠️ Known Issue | node | integration | 14 tests, 19 cases |


## ONNX

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `ONNX.benchmark.inference` | P1 | ❌ No Tests | node | benchmark | - |
| `ONNX.benchmark.load_time` | P2 | ❌ No Tests | node | benchmark | - |
| `ONNX.export.linear` | P0 | ✅ Working | python | export | 3 tests, 8 cases |
| `ONNX.export.neural_net` | P0 | ✅ Working | python | export | 3 tests, 8 cases |
| `ONNX.export.tree_ensemble` | P0 | ✅ Working | python | export | 3 tests, 8 cases |
| `ONNX.invariance.classification` | P0 | ✅ Working | python, node | invariance | 6 tests, 18 cases |
| `ONNX.invariance.cross_runtime` | P0 | ✅ Working | python, node | invariance | 2 tests, 8 cases |
| `ONNX.invariance.sklearn_vs_ort` | P0 | ✅ Working | python | invariance | 8 tests, 24 cases |
| `ONNX.special_values` | P1 | ✅ Working | python, node | invariance | 12 tests, 22 cases |


## STAT

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `STAT.histogram_table` | P1 | ⚠️ Known Issue | node | integration | 9 tests, 10 cases |
| `STAT.linear_fit` | P1 | ⚠️ Known Issue | node | integration | 9 tests, 17 cases |
| `STAT.sum_range` | P2 | ✅ Working | node | integration | 1 test, 4 cases |


## WASM

| Feature | Priority | Status | Backend | Layer | Tests |
|---------|----------|--------|---------|-------|-------|
| `WASM.benchmark.memory_fraction` | P2 | ❌ No Tests | node | benchmark | - |
| `WASM.benchmark.scalar_overhead` | P2 | ❌ No Tests | node | benchmark | - |
| `WASM.benchmark.vector_crossover` | P2 | ❌ No Tests | node | benchmark | - |
| `WASM.compile` | P0 | ✅ Working | native | build | 7 tests, 15 cases |
| `WASM.cross_backend_invariance` | P0 | ✅ Working | python, node | invariance | 2 tests, 11 cases |
| `WASM.integration` | P0 | ⚠️ Known Issue | node | integration | 6 tests, 6 cases |
| `WASM.scalar.arithmetic` | P0 | ✅ Working | node | unit | 1 test, 10 cases |
| `WASM.scalar.conditional` | P0 | ✅ Working | node | unit | 1 test, 10 cases |
| `WASM.scalar.transcendental` | P0 | ✅ Working | node | unit | 1 test, 10 cases |
| `WASM.scalar_vector_consistency` | P0 | ✅ Working | node | invariance | 1 test, 1 case |
| `WASM.special_values` | P1 | ✅ Working | python, node | invariance | 1 test, 10 cases |
| `WASM.vector.arithmetic` | P0 | ✅ Working | node | unit | 1 test, 10 cases |
| `WASM.vector.conditional` | P0 | ✅ Working | node | unit | 1 test, 10 cases |
| `WASM.vector.transcendental` | P0 | ✅ Working | node | unit | 1 test, 10 cases |


---

## Test Coverage Details

Tests per feature (for traceability). Approval logic: Feature = ✅ Working iff **ALL** tests pass.

<details>
<summary><strong>ALIAS.cdsalias</strong> — CDSAlias (5 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_control_a_d2_the_renamed_column_last_costs_only_itself`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_expression_alias_equals_numpy`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_function_forms_agree_with_the_expression_and_numpy`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d2_a_non_computed_alias_stops_the_alias_list`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_every_alias_of_the_list_is_usable`

</details>

<details>
<summary><strong>ALIAS.chain</strong> — Alias of an alias (5 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_chain_through_the_function_form`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d1_alias_of_an_alias_reads_the_whole_column`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d5_gather_of_an_alias_of_another_table_fails_the_build`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_alias_of_an_alias_equals_numpy`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_gather_of_an_alias_of_another_table`

</details>

<details>
<summary><strong>ALIAS.change_propagation</strong> — Alias change propagation (6 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_a_change_of_the_table_recomputes_every_alias`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_a_parameter_change_invalidates_exactly_its_dependents`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d6_a_gather_alias_keeps_old_values_after_the_gathered_table_changes`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d7_an_alias_with_automatic_fields_is_not_invalidated`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_a_gather_alias_follows_the_gathered_table`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_an_alias_with_automatic_fields_follows_its_input`

</details>

<details>
<summary><strong>ALIAS.context</strong> — Alias of a secondary table (1 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_alias_of_a_secondary_table`

</details>

<details>
<summary><strong>ALIAS.parameter</strong> — Aliases with parameters (5 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_column_selector_parameter_shown_directly`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_control_a_muted_widget_leaves_stale_values_and_the_check_sees_it`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_parameter_change_recomputes_the_alias`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d3_column_selector_in_an_expression_fails_the_build`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_column_selector_in_an_expression`

</details>

<details>
<summary><strong>DASHBOARD.histogram_only</strong> — Example dashboard: histograms only (testBokehClientHistogramOnlyHisto) (5 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_oracle_dashboards.py::test_control_h01_d1_end_key_and_mouse_drag_restore_the_full_range`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_oracle_dashboards.py::test_h01_histogram_dashboard_follows_the_A_range_and_the_cumulative_tab_and_restores`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_oracle_dashboards.py::test_h01_live_faults_in_the_browser_are_caught`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_oracle_dashboards.py::test_h01_page_is_the_original_dashboard_on_the_reproducible_table`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_oracle_dashboards.py::test_signature_h01_d1_arrow_keys_back_to_the_end_lose_the_largest_A`

</details>

<details>
<summary><strong>DASHBOARD.weights_reference</strong> — Example dashboard: analysis weights against reference weights (test_interactiveTemplateWeightsRef) (4 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_oracle_weightsref.py::test_wr1_control_weighted_counts_and_analysis_side_equal_numpy`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_oracle_weightsref.py::test_wr1_live_faults_in_the_browser_are_caught`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_oracle_weightsref.py::test_wr1_page_is_the_original_dashboard_on_the_reproducible_table`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_oracle_weightsref.py::test_wr1_weights_reference_dashboard_follows_weights_transform_bins_and_range_and_restores`

</details>

<details>
<summary><strong>DISPLAY.axis_label</strong> — Parametric axis label (ConcatenatedString) (4 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_browser_real_widget_change_updates_axis_label_and_redraws`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_control_cs_d1_after_each_change_the_value_and_the_axis_follow`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_node_axis_label_parameter_value_and_target`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_signature_cs_d1_the_label_value_is_undefined_at_load`

</details>

<details>
<summary><strong>DISPLAY.downsampler</strong> — Downsampled scatter (DownsamplerCDS) (6 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_control_ds_d1_when_the_selection_grows_the_columns_match_the_length`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_node_downsampler_exact_invariants_after_changes`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_node_downsampler_seeded_page_reproducibility`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_node_downsampler_selection_maps_back_without_losing_hidden_rows`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_signature_ds_d1_after_the_selection_shrinks_the_columns_keep_old_entries`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_strict_after_the_selection_shrinks_every_column_has_get_length_entries`

</details>

<details>
<summary><strong>DISPLAY.graph3d</strong> — 3D graph (BokehVisJSGraph3D, vis.js) (11 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_visualization_d3.py::test_control_g3_d1_the_same_graph_with_a_colour_variable_is_built`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_visualization_d3.py::test_control_g3_d2_a_partial_selection_reaches_vis_exactly`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_visualization_d3.py::test_d3_browser_real_vis_render_source_change_and_mouse_drag`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_visualization_d3.py::test_d3_node_options_and_model_identity`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_visualization_d3.py::test_d3_node_public_page_graph_follows_a_range_filter`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_visualization_d3.py::test_d3_node_public_page_mapping_under_row_reorder`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_visualization_d3.py::test_d3_node_public_page_maps_four_fields_to_production_source`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_visualization_d3.py::test_signature_g3_d1_a_3d_graph_without_colour_cannot_be_built`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_visualization_d3.py::test_signature_g3_d2_an_empty_selection_leaves_the_old_points_in_vis`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_visualization_d3.py::test_strict_a_3d_graph_without_colour_is_built`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_visualization_d3.py::test_strict_an_empty_selection_leaves_no_points_in_the_3d_graph`

</details>

<details>
<summary><strong>DISPLAY.lazy_tabs</strong> — Lazy tabs (LazyTabs) (2 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_browser_lazy_tabs_real_tab_click_refreshes_hidden_plot`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_node_lazy_tabs_watched_and_catchup_after_hidden_filter_moves`

</details>

<details>
<summary><strong>DSL.arithmetic_expr</strong> — Arithmetic expressions (3 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_expression_alias_equals_numpy`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_integer_and_boolean_columns`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>DSL.custom_js_func</strong> — CustomJS function execution (3 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_chain_through_the_function_form`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_function_forms_agree_with_the_expression_and_numpy`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>DSL.gather_operation</strong> — Cross-table gather (11 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_node_every_state_after_load_equals_the_reference`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_gather_of_a_column_equals_numpy_indexing`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d4_a_gather_alias_alone_does_not_send_the_gathered_column`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d5_gather_of_an_alias_of_another_table_fails_the_build`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d6_a_gather_alias_keeps_old_values_after_the_gathered_table_changes`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_a_gather_alias_alone_equals_numpy_indexing`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_a_gather_alias_follows_the_gathered_table`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_gather_of_an_alias_of_another_table`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_ClientSideJoin.py::test_gather`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_ClientSideJoin.py::test_gather_realistic`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>DSL.math_functions</strong> — Math functions (4 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_math_function_alias_equals_numpy`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d9_server_cosh_is_sinh`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_server_cosh_equals_numpy`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_dsl_customjs.py::test_mathutils`

</details>

<details>
<summary><strong>DSL.server_client_parity</strong> — One expression, one result (5 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_control_where_the_semantics_agree_both_paths_agree`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d8_python_operators_differ_in_the_browser`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_signature_a_d9_server_cosh_is_sinh`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_one_expression_one_result`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_aliases.py::test_strict_server_cosh_equals_numpy`

</details>

<details>
<summary><strong>ENC.base64.float64</strong> — Float64Array encoding (6 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_code`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_delta`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_relative16`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_simple`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_sinh`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>ENC.base64.int32</strong> — Int32Array encoding (6 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_code`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_delta`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_relative16`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_simple`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_sinh`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>ENC.cds_compress.data_replace</strong> — Compressed table: replacing its data in a live page (3 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_d1p2_transport.py::test_control_tr_d1_replacement_before_any_read_gives_the_new_table`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_d1p2_transport.py::test_d1p2_real_compressed_source_inputdata_replacement_invalidates_cache`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_d1p2_transport.py::test_signature_tr_d1_decoded_columns_stay_old_undecoded_columns_come_out_new`

</details>

<details>
<summary><strong>ENC.cds_compress.length_guard</strong> — Compressed table: a column of the wrong length is refused (4 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_d1p2_transport.py::test_control_tr_d2_a_column_of_the_right_length_is_cached_and_reused`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_d1p2_transport.py::test_d1p2_real_compressed_source_rejects_mismatched_column_length`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_d1p2_transport.py::test_signature_tr_d2_a_refused_column_is_returned_on_the_second_read`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_d1p2_transport.py::test_strict_a_column_of_the_wrong_length_is_refused_on_every_read`

</details>

<details>
<summary><strong>ENC.cds_compress.page_decode</strong> — Compressed table in a page (CDSCompress): lossless decoding (3 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_d1p2_transport.py::test_d1p2_real_cdscompress_decodes_a_column_only_when_it_is_read`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_d1p2_transport.py::test_d1p2_real_cdscompress_lazy_decode_and_cache`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_d1p2_transport.py::test_d1p2_real_compressed_source_histogram_filter_restoration`

</details>

<details>
<summary><strong>ENC.compression.delta</strong> — Delta/Absolute compression (4 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_Compression.py::test_CompressionSampleDelta`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_Compression.py::test_CompressionSampleDeltaCode`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_delta`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_serializationutils`

</details>

<details>
<summary><strong>ENC.compression.relative</strong> — Relative compression (2 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_code`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_relative16`

</details>

<details>
<summary><strong>ENC.compression.roundtrip</strong> — Compression roundtrip (5 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_code`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_delta`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_relative16`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_simple`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_sinh`

</details>

<details>
<summary><strong>ENC.compression.sinh</strong> — Sinh/Sqrt scaling compression (3 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_Compression.py::test_CompressionSampleSinh`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_sinh`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_serializationutils`

</details>

<details>
<summary><strong>ENC.compression.zip</strong> — ZIP compression (6 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_code`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_delta`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_relative16`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_simple`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_compression_sinh`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_compression_integration.py::test_serializationutils`

</details>

<details>
<summary><strong>FILTER.intersection</strong> — Filter intersection (710 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_node_every_state_after_load_equals_the_reference`
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
<summary><strong>FILTER.range</strong> — Range filter (21 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_histograms_follow_a_range_filter_and_its_restoration`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_chromium_real_keyboard_on_range_slider_updates_histogram`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_chromium_wg_d1_signature_the_page_raises_and_shows_no_widget`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_control_wg_d1_with_a_second_active_filter_the_page_loads`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_control_wg_d2_the_same_range_without_toggle_selects_its_rows`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_range_slider_selection_and_weighted_histogram`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_range_slider_updates_nd_profile_after_each_move`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_scalar_slider_selection_and_histogram`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_scratch_muted_widget_callback_is_caught`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_spinner_range_each_endpoint_callback_and_histogram`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_toggleable_range_activation_and_restore`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_signature_wg_d1_a_toggleable_range_disabled_at_load_breaks_the_page`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_signature_wg_d2_an_enabled_toggleable_range_selects_every_row`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_strict_wg_d2_an_enabled_toggleable_range_selects_its_rows`
- ✅ headless scenario `chain/deep_widget_filter_alias_stack_profile` (browser, node)
- ✅ headless scenario `histograms/gpt24_empty_selection_shifted_range` (browser, node)
- ✅ headless scenario `histograms/plain_1d_2d_one_range_two_moves` (browser, node)
- ✅ headless scenario `known_defects/b8_1d_histogram_counts_nan_in_bin0` (browser, node)
- ✅ headless scenario `known_defects/claude10_b8_weighted_1d_histogram_nan_in_bin0` (browser, node)
- ✅ headless scenario `known_defects/claude3_b8_weighted_1d_histogram` (browser, node)
- ✅ headless scenario `selection/rb1_nan_excluded_at_full_bounds` (browser, node)

</details>

<details>
<summary><strong>FILTER.text_query</strong> — Text query (textQuery widget, ColumnFilter) (14 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_browser_textquery_real_typed_input_updates_weighted_histogram`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_control_tq_d1_with_the_parameter_in_scope_the_query_computes_it`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_control_tq_d2_a_toggle_with_the_query_shown_logs_nothing`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_node_textquery_disconnected_callback_mutant_is_detected`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_node_textquery_exact_masks_and_weighted_bins`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_node_textquery_parameter_changed_after_input`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_node_textquery_toggle_with_second_filter`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_signature_tq_d1_a_query_without_a_matching_alias_cannot_see_the_parameter`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_signature_tq_d1_the_query_keeps_the_parameter_value_from_build_time`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_signature_tq_d2_after_a_cleared_query_the_next_query_logs_a_type_error`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_signature_wg_d1_a_toggleable_text_query_alone_breaks_the_page`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_strict_a_query_follows_its_parameter`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_strict_a_toggleable_text_query_alone_loads_and_selects_every_row`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_display.py::test_strict_clearing_and_retyping_a_query_logs_nothing`

</details>

<details>
<summary><strong>HIST.histogram_1d</strong> — 1D histogram (18 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_node_live_weight_switch_follows_the_parameter`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_signature_b2_at_load_the_1d_histograms_count_every_row`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_strict_the_load_state_equals_the_reference`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_histogram_counts_weights_and_projections_equal_numpy`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_histograms_follow_a_range_filter_and_its_restoration`
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
<summary><strong>HIST.histogram_nd</strong> — N-D histogram (744 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_control_b2_the_2d_histogram_and_the_profile_follow_the_selection_from_load`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_histogram_counts_weights_and_projections_equal_numpy`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_histograms_follow_a_range_filter_and_its_restoration`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_browser_distinct_widget_input_updates_dependent_state`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_chromium_real_keyboard_on_range_slider_updates_histogram`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_control_wg_d1_with_a_second_active_filter_the_page_loads`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_control_wg_d2_the_same_range_without_toggle_selects_its_rows`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_categorical_multiselect_selection_and_histogram`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_multiselect_real_widget_callback_and_histogram`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_parameter_widget_updates_nd_histogram_nbins`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_range_slider_selection_and_weighted_histogram`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_range_slider_updates_nd_profile_after_each_move`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_scalar_slider_selection_and_histogram`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_scratch_muted_widget_callback_is_caught`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_select_selection_callback_and_histogram`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_spinner_range_each_endpoint_callback_and_histogram`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_toggle_parameter_updates_legend_and_keeps_histogram`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_toggleable_range_activation_and_restore`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_strict_wg_d2_an_enabled_toggleable_range_selects_its_rows`
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
- ✅ headless scenario `widgets/gpt27_slider_profile_binned_mean` (node)

</details>

<details>
<summary><strong>HIST.profile</strong> — Histogram profile (7 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_control_b2_the_2d_histogram_and_the_profile_follow_the_selection_from_load`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_binned_profile_quantiles_interpolate_the_cumulative_count`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_binned_profile_uses_bin_moments_not_raw_values`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_empty_profile_slice_reports_no_valid_statistic`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_widgets.py::test_node_range_slider_updates_nd_profile_after_each_move`
- ✅ headless scenario `chain/deep_widget_filter_alias_stack_profile` (browser, node)
- ✅ headless scenario `widgets/gpt27_slider_profile_binned_mean` (node)

</details>

<details>
<summary><strong>HIST.profile_unbinned</strong> — Unbinned profile statistics (10 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_control_b4_a_zero_weight_after_the_first_row_is_harmless`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_signature_b4_zero_weight_first_gives_nan_moments`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_signature_b5_two_row_bin_quantile_is_the_sum_times_q`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_signature_b5b_quantile_depends_on_the_row_order`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_strict_a_quantile_does_not_depend_on_the_row_order`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_strict_a_quantile_lies_among_its_bin_values_and_equals_the_definition`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_strict_zero_weight_rows_do_not_spoil_the_weighted_moments`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_unbinned_profile_moments_equal_numpy`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_histograms.py::test_unbinned_quantiles_in_a_bin_of_three_rows`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_unbinned_weighted_quantile_invariants`

</details>

<details>
<summary><strong>HIST.stack</strong> — Stacked histograms (CDSStack) (5 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_multiselect_stack_selection_change`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_multiselect_stack_tracks_range_filter`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_multiselect_weights_stack_and_reverse_order`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_real_page_mutation_falsifies_stack_oracle`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_single_select_weights_recomputes`

</details>

<details>
<summary><strong>JOIN.cdsjoin.basic</strong> — Basic CDSJoin (19 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c1_key_join_row_pairs_equal_pandas`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c1_many_to_many_key_join_gives_every_combination`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c1_multi_key_join_rows_equal_pandas_in_every_mode`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c2_right_columns_correct`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c3_left_index_join_right_values`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c4_length_and_every_column`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c5_row_order_within_equal_keys_does_not_change_the_join`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_control_j_d1_aligned_rows`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_j_h1_exact_result_on_unsorted_keys_and_its_sorted_control`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_multi_key_join_equals_pandas_in_both_sort_orders`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_prefixed_names_route_to_their_side`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_signature_j_d1_left_columns`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_signature_j_d4_multi_key_unmatched_key_inside`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_signature_j_d4_unmatched_key_inside_the_tables`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_strict_key_join_columns_are_correct`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_strict_key_join_with_an_unmatched_key_inside`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_strict_multi_key_join_with_an_unmatched_key_inside`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_strict_unsorted_keys_join_correctly_or_refuse_loudly`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_ClientSideJoin.py::test_join`

</details>

<details>
<summary><strong>JOIN.cdsjoin.index0</strong> — CDSJoin index-0 regression (4 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_control_j_d2_without_a_reference_to_row_0`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_signature_j_d2_and_j_d1_inner_index_join`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_strict_inner_index_join_is_correct`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_ClientSideJoin.py::test_join`

</details>

<details>
<summary><strong>JOIN.cdsjoin.outer</strong> — CDSJoin outer join (5 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c1_key_join_row_pairs_equal_pandas`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c1_multi_key_join_rows_equal_pandas_in_every_mode`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_multi_key_join_equals_pandas_in_both_sort_orders`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_strict_key_join_with_an_unmatched_key_inside`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_strict_multi_key_join_with_an_unmatched_key_inside`

</details>

<details>
<summary><strong>JOIN.cdsjoin.reverse_index</strong> — CDSJoin reverse index join (3 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_control_j_d3_the_same_relation_in_the_supported_direction`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_signature_j_d3_reverse_index_join`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_strict_reverse_index_join_is_correct`

</details>

<details>
<summary><strong>JOIN.change_propagation</strong> — Join change propagation (3 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c8_a_change_of_the_innermost_table_reaches_the_outermost_join`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c8_cache_invalidation_on_an_already_used_model`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c8_rows_added_or_removed_reach_the_outermost_join`

</details>

<details>
<summary><strong>JOIN.cross_table</strong> — Multi-CDS cross-table (1 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_dsl_customjs.py::test_compileVarName`

</details>

<details>
<summary><strong>JOIN.multi_table_chain</strong> — Three linked tables on one page (V0 -> track -> event) (7 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_browser_real_input_equals_the_reference_and_node`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_control_a_muted_widget_is_caught_by_the_comparator`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_control_b2_the_2d_histogram_and_the_profile_follow_the_selection_from_load`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_node_every_state_after_load_equals_the_reference`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_node_live_weight_switch_follows_the_parameter`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_signature_b2_at_load_the_1d_histograms_count_every_row`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_a4_multitable.py::test_strict_the_load_state_equals_the_reference`

</details>

<details>
<summary><strong>JOIN.prototype_parity</strong> — Join prototype parity (6 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_j3_clusters_left_nested_is_empty`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_j3_clusters_right_nested_event_per_cluster`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_j3_inner_join_loses_exactly_the_tracks_of_event_0`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_j3_strict_clusters_left_nested_gives_each_cluster_its_event`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_j3_strict_inner_join_keeps_the_tracks_of_event_0`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_j3_tracks_to_events_three_references_agree`

</details>

<details>
<summary><strong>JOIN.recursive</strong> — Recursive joins (14 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c6_control_right_nested_identity_references`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c6_signature_cascade_v0_track_event`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c6_signature_v0_track_event`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c6_strict_cascade_v0_track_event_is_correct`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c6_strict_v0_track_event_is_correct`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c6_two_daughters_join_of_two_joins_signature`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c7_associativity_v0_track_event`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c7_control_clean_inner_key_joins_are_associative_and_correct`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c7_signature_inner_key_joins_with_an_inner_gap_are_not_associative`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c7_strict_inner_key_joins_with_an_inner_gap_are_associative`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_c7_strict_production_is_associative`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_signature_j_d5_key_join_of_joins_pairs_rows_by_position`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_signature_j_d5_left_nested_chain_is_empty_and_stays_empty`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_joins.py::test_strict_key_join_of_joins_pairs_by_v0_id`

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

- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_classification_labels_sklearn_vs_python_ort`
- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_regression_sklearn_vs_python_ort`
- ✅ `RootInteractive/onnx/test_invariance_onnx.py::test_smoke_all_models`

</details>

<details>
<summary><strong>ONNX.export.neural_net</strong> — ONNX export neural networks (3 tests)</summary>

- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_classification_labels_sklearn_vs_python_ort`
- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_regression_sklearn_vs_python_ort`
- ✅ `RootInteractive/onnx/test_invariance_onnx.py::test_smoke_all_models`

</details>

<details>
<summary><strong>ONNX.export.tree_ensemble</strong> — ONNX export tree ensembles (3 tests)</summary>

- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_classification_labels_sklearn_vs_python_ort`
- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_regression_sklearn_vs_python_ort`
- ✅ `RootInteractive/onnx/test_invariance_onnx.py::test_smoke_all_models`

</details>

<details>
<summary><strong>ONNX.invariance.classification</strong> — ONNX classification invariance (6 tests)</summary>

- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_classification_labels_sklearn_vs_nodejs_ort`
- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_classification_labels_sklearn_vs_python_ort`
- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_classification_proba_python_ort_vs_nodejs_ort`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_classification_labels_sklearn_vs_nodejs_ort`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_classification_labels_sklearn_vs_python_ort`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_classification_proba_python_ort_vs_nodejs_ort`

</details>

<details>
<summary><strong>ONNX.invariance.cross_runtime</strong> — ONNX cross-runtime invariance (2 tests)</summary>

- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_regression_python_ort_vs_nodejs_ort`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_regression_python_ort_vs_nodejs_ort`

</details>

<details>
<summary><strong>ONNX.invariance.sklearn_vs_ort</strong> — ONNX sklearn↔ORT invariance (8 tests)</summary>

- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_classification_proba_sklearn_vs_python_ort`
- ✅ `RootInteractive/onnx/test_invariance_onnx.py::test_python_three_way_consistency`
- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_regression_sklearn_vs_nodejs_ort`
- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_regression_sklearn_vs_python_ort`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_classification_proba_sklearn_vs_python_ort`
- ✅ `RootInteractive/onnx/test_onnx_invariance.py::test_python_three_way_consistency`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_regression_sklearn_vs_nodejs_ort`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_regression_sklearn_vs_python_ort`

</details>

<details>
<summary><strong>ONNX.special_values</strong> — ONNX IEEE-754 special values (12 tests)</summary>

- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_special_values_linear_inf`
- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_special_values_linear_nan`
- ✅ `RootInteractive/onnx/test_invariance_onnx.py::test_special_values_mixed_all_models`
- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_special_values_mlp_nan`
- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_special_values_tree_inf`
- ❓ `RootInteractive/onnx/test_invariance_onnx.py::test_special_values_tree_nan`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_special_values_linear_inf`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_special_values_linear_nan`
- ✅ `RootInteractive/onnx/test_onnx_invariance.py::test_special_values_mixed_all_models`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_special_values_mlp_nan`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_special_values_tree_inf`
- ❓ `RootInteractive/onnx/test_onnx_invariance.py::test_special_values_tree_nan`

</details>

<details>
<summary><strong>STAT.histogram_table</strong> — Histogram statistics table (HistoStatsCDS) (9 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_control_st_d4_an_ascending_merged_list_is_correct`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_signature_st_d4_an_unsorted_quantile_list_gives_one_pass_values`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_table_binned_stats_quantiles_integrals`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_table_empty_histogram_reports_invalid`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_table_filter_changes_and_restores_exact_numbers`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_table_include_exclude_selects_named_histograms`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_table_mixed_quantile_requests_preserve_column_titles`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_table_nd_axes_have_independent_statistics_no_quantiles`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_table_weight_choice_updates`

</details>

<details>
<summary><strong>STAT.linear_fit</strong> — Client-side linear fit (ClientLinearFitter) (9 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_control_st_d2_one_predictor_fit_equals_numpy`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_control_st_d3_after_an_expression_alias_the_page_builds_and_follows_the_predictor`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_signature_st_d1_a_fit_column_is_not_sent`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_signature_st_d2_two_predictors_give_the_double_update_solution`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_signature_st_d3_parameter_chosen_alias_variables_fail_the_build`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_client_fit_predictor_count_change_recomputes`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_client_fit_reacts_to_source_data_change`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_client_linear_fit_matches_numpy`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_strict_a_fit_on_a_page_as_a_user_writes_it_equals_numpy`

</details>

<details>
<summary><strong>STAT.sum_range</strong> — Integral and efficiency columns (sum_range) (1 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/headless/test_headless_node_stats.py::test_stat_profile_integral_efficiency_matches_uniform_bins`

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
<summary><strong>WASM.compile</strong> — WASM compilation (7 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_invariance.py::test_wasm_invariance_all`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_provenance.py::test_w1_bokeh_functions_wasm_is_the_built_binary`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_provenance.py::test_w1_file_is_the_recorded_one`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_provenance.py::test_w2_every_function_has_the_cpp_parameter_count_and_every_cpp_function_is_exported`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_provenance.py::test_w2_exports_are_the_declared_functions_plus_the_runtime`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_provenance.py::test_w2_fact_an_empty_import_object_cannot_instantiate_the_binary`
- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_provenance.py::test_w2_imports_are_exactly_the_memory_growth_notification`

</details>

<details>
<summary><strong>WASM.cross_backend_invariance</strong> — WASM cross-backend invariance (2 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_invariance.py::test_wasm_correctness`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_invariance.py::test_wasm_invariance_all`

</details>

<details>
<summary><strong>WASM.integration</strong> — WASM in RootInteractive (6 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_integration.py::test_signature_w_d1_custom_wasm_module_does_not_compile`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_integration.py::test_signature_w_d2_custom_wasm_func_does_not_compile`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_integration.py::test_strict_custom_wasm_func_compiles`
- ⚠️ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_integration.py::test_strict_custom_wasm_module_compiles`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_integration.py::test_w2_control_a_column_function_of_the_same_kind_compiles`
- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_integration.py::test_w9_fact_no_rootinteractive_module_uses_the_wasm_classes`

</details>

<details>
<summary><strong>WASM.scalar.arithmetic</strong> — WASM scalar arithmetic (1 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_invariance.py::test_wasm_correctness`

</details>

<details>
<summary><strong>WASM.scalar.conditional</strong> — WASM scalar conditional (1 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_invariance.py::test_wasm_correctness`

</details>

<details>
<summary><strong>WASM.scalar.transcendental</strong> — WASM scalar transcendental (1 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_invariance.py::test_wasm_correctness`

</details>

<details>
<summary><strong>WASM.scalar_vector_consistency</strong> — WASM scalar-vector consistency (1 tests)</summary>

- ✅ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_invariance.py::test_wasm_scalar_vector_consistency`

</details>

<details>
<summary><strong>WASM.special_values</strong> — WASM IEEE-754 special values (1 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_invariance.py::test_wasm_special_values`

</details>

<details>
<summary><strong>WASM.vector.arithmetic</strong> — WASM vector arithmetic (1 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_invariance.py::test_wasm_correctness`

</details>

<details>
<summary><strong>WASM.vector.conditional</strong> — WASM vector conditional (1 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_invariance.py::test_wasm_correctness`

</details>

<details>
<summary><strong>WASM.vector.transcendental</strong> — WASM vector transcendental (1 tests)</summary>

- ❓ `RootInteractive/InteractiveDrawing/bokeh/test_wasm_invariance.py::test_wasm_correctness`

</details>

---

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Working | 33 | 47.8% |
| 🧨 Broken | 0 | 0.0% |
| ⚠️ Known Issue | 30 | 43.5% |
| 📋 Planned | 0 | 0.0% |
| ❌ No Tests | 5 | 7.2% |
| ❓ Unknown | 1 | 1.4% |
| **Total** | **69** | **100%** |

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
| ❓ Partly verified | The tests that ran pass, but not every registered test or case ran (no result, or skipped), or a scenario checked only the response; counted with Unknown in the summary |

## Priority Levels

| Priority | Meaning |
|----------|---------|
| P0 | Critical - blocks release |
| P1 | Important - should fix before release |
| P2 | Nice to have - can defer |

---

*Auto-generated by `scripts/generate_capability_matrix.py`*

*Phase 0.1.A provides coverage accounting. Invariance validation requires Phase 0.1.B.*