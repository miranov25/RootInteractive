#!/usr/bin/env python3
"""PHASE_4_5 S3 v0.2a — verify *executed* pytest-json-report case accounting.

Review utility only; no RI production-code changes.  This utility cannot prove
that a passing assertion is correct; it prevents 26 parameterized test variants
from being silently counted as only 15 test functions or omitted altogether.

Integration (Claude20.RI, PHASE_4_5 A3): installed as scripts/verify_widget_stage3_report.py for the integrated file
headless/test_headless_widgets.py. Two changes only: the file name, and the 9 cases added on integration for the
two defects found (WG-D1, WG-D2: signatures, strict partners, controls) - 35 exact cases. The acceptance rule is
unchanged (every case passed), so it says NOT ACCEPTED while a strict expected failure (a known defect) remains;
the expected failures are listed per case in the output.

Usage after approved integration + required alma2 run:
  python verify_widget_stage3_report_GPT27_RI_v0_2a.py \
    --report /path/to/widgets_pytest_report.json \
    --source /path/to/test_headless_widgets_GPT27_RI_CANDIDATE_v0_2a.py \
    --out /path/to/widgets_case_evidence.json
"""
import argparse
import ast
import json
from collections import Counter
from pathlib import Path

TEST_NAME = "test_headless_widgets.py"
# Exact names from review-stage pytest --collect-only, not a function count.
EXPECTED_SUFFIXES = (
    "test_node_range_slider_selection_and_weighted_histogram[narrow-empty-restore]",
    "test_node_range_slider_selection_and_weighted_histogram[two-complementary-ranges]",
    "test_node_multiselect_real_widget_callback_and_histogram[bitmask-any]",
    "test_node_multiselect_real_widget_callback_and_histogram[bitmask-all]",
    "test_node_categorical_multiselect_selection_and_histogram",
    "test_node_range_slider_updates_nd_profile_after_each_move",
    "test_oracle_and_assertions_have_controls_that_fail",
    "test_chromium_real_keyboard_on_range_slider_updates_histogram",
    "test_node_scalar_slider_selection_and_histogram",
    "test_node_select_selection_callback_and_histogram",
    "test_node_spinner_range_each_endpoint_callback_and_histogram",
    "test_node_toggleable_range_activation_and_restore",
    "test_node_parameter_widget_updates_nd_histogram_nbins[slider-binning]",
    "test_node_parameter_widget_updates_nd_histogram_nbins[select-binning]",
    "test_node_parameter_widget_updates_nd_histogram_nbins[spinner-binning]",
    "test_node_toggle_parameter_updates_legend_and_keeps_histogram",
    "test_v02_oracle_controls_cover_scalar_param_and_restore",
    "test_browser_distinct_widget_input_updates_dependent_state[slider-keyboard]",
    "test_browser_distinct_widget_input_updates_dependent_state[select-keyboard]",
    "test_browser_distinct_widget_input_updates_dependent_state[multiselect-native]",
    "test_browser_distinct_widget_input_updates_dependent_state[bitmask-any-native]",
    "test_browser_distinct_widget_input_updates_dependent_state[bitmask-all-native]",
    "test_browser_distinct_widget_input_updates_dependent_state[spinner-range-both-endpoints]",
    "test_browser_distinct_widget_input_updates_dependent_state[spinner-parameter-input]",
    "test_browser_distinct_widget_input_updates_dependent_state[toggle-click]",
    "test_node_scratch_muted_widget_callback_is_caught",
    # integration (Claude20.RI): WG-D1 and WG-D2
    "test_signature_wg_d1_a_toggleable_range_disabled_at_load_breaks_the_page",
    "test_control_wg_d1_with_a_second_active_filter_the_page_loads",
    "test_signature_wg_d2_an_enabled_toggleable_range_selects_every_row[gate-first]",
    "test_signature_wg_d2_an_enabled_toggleable_range_selects_every_row[gate-second]",
    "test_strict_wg_d2_an_enabled_toggleable_range_selects_its_rows[gate-first]",
    "test_strict_wg_d2_an_enabled_toggleable_range_selects_its_rows[gate-second]",
    "test_control_wg_d2_the_same_range_without_toggle_selects_its_rows[gate-first]",
    "test_control_wg_d2_the_same_range_without_toggle_selects_its_rows[gate-second]",
    "test_chromium_wg_d1_signature_the_page_raises_and_shows_no_widget",
)
N_CASES = len(EXPECTED_SUFFIXES)
CONTROLS = {"test_oracle_and_assertions_have_controls_that_fail",
            "test_v02_oracle_controls_cover_scalar_param_and_restore"}
TAXONOMY_IDS = {"FILTER.range", "HIST.histogram_nd", "HIST.profile"}


def source_markers(source):
    tree = ast.parse(Path(source).read_text(encoding="utf8"))
    mapping = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
            continue
        features = []
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "feature" and decorator.args):
                features.append(ast.literal_eval(decorator.args[0]))
        mapping[node.name] = sorted(set(features))
    if {s.split("[", 1)[0] for s in EXPECTED_SUFFIXES} != set(mapping):
        raise ValueError(f"source functions do not match the {N_CASES}-case reviewed contract")
    for name, features in mapping.items():
        if name not in CONTROLS and not features:
            raise ValueError("missing canonical feature marker for " + name)
        if set(features) - TAXONOMY_IDS:
            raise ValueError("unrecognized / not-reviewed taxonomy feature: " + repr(features))
        if name in CONTROLS and features:
            raise ValueError("Python-only oracle controls cannot certify production features")
    return mapping


def verify(report, source):
    markers = source_markers(source)
    tests = json.loads(Path(report).read_text(encoding="utf8")).get("tests", [])
    matching = {}
    for test in tests:
        node = test.get("nodeid", "")
        if TEST_NAME not in node.split("::", 1)[0]:
            continue
        suffix = node.split("::", 1)[-1]
        if suffix in matching:
            raise ValueError("duplicate executed node ID: " + suffix)
        matching[suffix] = test.get("outcome", "missing")
    expected = set(EXPECTED_SUFFIXES)
    missing, extra = sorted(expected - set(matching)), sorted(set(matching) - expected)
    if missing or extra or len(matching) != N_CASES:
        raise ValueError(f"variant accounting failed: missing={missing}, extra={extra}, n={len(matching)}")
    rows = []
    for suffix in EXPECTED_SUFFIXES:
        fn = suffix.split("[", 1)[0]
        rows.append({"case": suffix, "outcome": matching[suffix], "features": markers[fn],
                     "backend": "python-control" if fn in CONTROLS else
                     "chromium" if fn.startswith("test_browser_") or fn.startswith("test_chromium_") else "node"})
    summary = dict(Counter(r["outcome"] for r in rows))
    by_backend = {b: dict(Counter(r["outcome"] for r in rows if r["backend"] == b))
                  for b in ("node", "chromium", "python-control")}
    return {"expected": N_CASES, "executed": len(rows), "outcomes": summary,
            "by_backend": by_backend, "rows": rows,
            "expected_failures": [r["case"] for r in rows if r["outcome"] == "xfailed"],
            "accepted": len(rows) == N_CASES and summary == {"passed": N_CASES}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    evidence = verify(args.report, args.source)
    if args.out:
        args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf8")
    print("Stage3 v0.2a: cases=%s outcomes=%s by_backend=%s" %
          (evidence["executed"], evidence["outcomes"], evidence["by_backend"]))
    for case in evidence["expected_failures"]:
        print("  strict expected failure (known defect):", case)
    if not evidence["accepted"]:
        raise SystemExit(f"NOT ACCEPTED: expected {N_CASES} passes; classify product/harness failures separately")


if __name__ == "__main__":
    main()
