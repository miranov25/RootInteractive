#!/usr/bin/env python3
# RootInteractive/scripts/generate_capability_matrix.py
"""
Auto-generate CAPABILITY_MATRIX.md from pytest markers.
Based on RDataFrameDSL pattern (Phase 13.6.F).

Usage:
    # Run tests and generate matrix (recommended)
    pytest RootInteractive/InteractiveDrawing/bokeh/ \\
        --json-report --json-report-file=report.json -m feature
    python scripts/generate_capability_matrix.py --from-reports report.json
    
    # Or let the script run pytest itself
    python scripts/generate_capability_matrix.py
    
    # Dry run (no file write)
    python scripts/generate_capability_matrix.py --dry-run
    
    # Skip running tests (just parse markers)
    python scripts/generate_capability_matrix.py --skip-tests

Prerequisites:
    pip install pytest-json-report

Features:
    - Parses @pytest.mark.feature markers (canonical form only)
    - Uses pytest-json-report for reliable test outcomes (P0-1 fix)
    - Multiple nodeid matching strategies (P1-2 fix)
    - Generates both Markdown and JSON output (P1-4 fix)
    - Supports --from-reports for pre-run reports (P1-5)

MARKER REQUIREMENTS (P1-4 Documentation):
========================================
This generator only supports the CANONICAL marker form:

    @pytest.mark.feature("CATEGORY.feature_id")
    def test_something():
        ...

Requirements:
    - Must be on single line, immediately above test function
    - Must use string literal (not variable reference)
    - Feature ID format: CATEGORY.feature_id (e.g., "DSL.arithmetic_expr")
    - CATEGORY must be uppercase letters only
    - feature_id can contain letters, numbers, underscores, dots

SUPPORTED examples:
    @pytest.mark.feature("DSL.arithmetic_expr")
    @pytest.mark.feature("JOIN.cdsjoin.basic")
    @pytest.mark.feature("ENC.base64.float64")

NOT SUPPORTED (will be missed):
    - Multiline decorators
    - Variable feature IDs: @pytest.mark.feature(FEATURE_CONSTANT)
    - Parametrized/indirect markers
    - Markers inside comments or docstrings

For feature taxonomy definitions, see:
    RootInteractive/InteractiveDrawing/bokeh/feature_taxonomy.py

Phase: 0.1.A
Date: 2026-02-02
Version: 1.1
"""

import ast
import subprocess
import json
import tempfile
import glob
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
if SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

# Add paths for imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RootInteractive" / "InteractiveDrawing" / "bokeh"))


# =============================================================================
# FEATURE MARKER PARSING (P1-1: Canonical form only)
# =============================================================================

FEATURE_ID = re.compile(r"[A-Z]+(?:\.\w+)+")


def _feature_ids(decorators) -> List[str]:
    """The ids of the canonical markers @pytest.mark.feature("ID") among a node's decorators."""
    out = []
    for d in decorators:
        if (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "feature"
                and isinstance(d.func.value, ast.Attribute) and d.func.value.attr == "mark"
                and isinstance(d.func.value.value, ast.Name) and d.func.value.value.id == "pytest"
                and d.args and isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str)
                and FEATURE_ID.fullmatch(d.args[0].value)):
            out.append(d.args[0].value)
    return out


def _test_file_id(filepath: Path) -> str:
    """The file part of a pytest node id: the path relative to the project root (pytest's rootdir), else as given."""
    try:
        return filepath.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return filepath.as_posix()


def parse_feature_markers_from_files(test_pattern: str, verbose: bool = False) -> Dict[str, List[str]]:
    """
    Parse @pytest.mark.feature("feature_id") markers from test files.

    Only the canonical form is supported: @pytest.mark.feature("DSL.arithmetic_expr"), ids with several dots allowed.

    PHASE_4_3 J4 v0.6 (review M-1): the file is read as Python syntax (ast), not line by line. A test function gets a
    class segment only if it is defined INSIDE that class - before, the class of the last `class` line was attached
    to every later module-level test (e.g. after a helper class `Port`), so its results were never found. The file
    part is the path relative to the project root, as in pytest's node ids; markers in strings or comments are
    ignored by construction. Marked classes are ignored, as before (no test file has one).

    Returns dict mapping feature_id -> [list of test nodeids]
    """
    feature_tests = {}
    test_files = sorted(glob.glob(str(PROJECT_ROOT / test_pattern)))

    if verbose:
        print(f"Scanning {len(test_files)} test files...")

    for filepath in test_files:
        filepath = Path(filepath)
        try:
            tree = ast.parse(filepath.read_text(), filename=str(filepath))
        except (FileNotFoundError, PermissionError, SyntaxError, UnicodeDecodeError) as e:
            if verbose:
                print(f"  Warning: Cannot read {filepath}: {e}")
            continue
        file_id = _test_file_id(filepath)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                found = [(f"{file_id}::{node.name}", _feature_ids(node.decorator_list))]
            elif isinstance(node, ast.ClassDef):
                found = [(f"{file_id}::{node.name}::{m.name}", _feature_ids(m.decorator_list)) for m in node.body
                         if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test")]
            else:
                continue
            for nodeid, features in found:
                for feature_id in features:
                    feature_tests.setdefault(feature_id, []).append(nodeid)
                    if verbose:
                        print(f"  Found: {feature_id} -> {nodeid}")

    return feature_tests


# =============================================================================
# PYTEST EXECUTION (P0-1: Use pytest-json-report)
# =============================================================================

def run_pytest_for_outcomes(
    test_pattern: str = "RootInteractive/InteractiveDrawing/bokeh/test_*.py",
    verbose: bool = False
) -> Dict[str, str]:
    """
    Run pytest to get actual pass/fail status.
    
    P0-1 Fix: Uses pytest-json-report instead of log parsing.
    
    Returns dict mapping nodeid -> outcome (passed/failed/xfailed/etc.)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        report_path = Path(f.name)
    
    try:
        expanded_files = glob.glob(str(PROJECT_ROOT / test_pattern))
        if not expanded_files:
            print(f"Warning: No files match pattern {test_pattern}")
            return {}
        
        cmd = [
            sys.executable, "-m", "pytest",
            *expanded_files,
            "-m", "feature",  # Only run feature-marked tests
            "-q",
            "--json-report",
            f"--json-report-file={report_path}",
            "--tb=no",
        ]
        
        if verbose:
            print(f"Running: {' '.join(cmd[:6])}...")
        else:
            print(f"Running pytest on {len(expanded_files)} file(s)...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=PROJECT_ROOT,
        )
        
        if not report_path.exists():
            print(f"Warning: pytest report not generated at {report_path}")
            if result.stderr:
                print(f"stderr: {result.stderr[:500]}")
            if result.stdout:
                print(f"stdout: {result.stdout[:500]}")
            return {}
        
        with open(report_path) as f:
            report = json.load(f)
        
        outcomes = {}
        for test in report.get("tests", []):
            nodeid = test.get("nodeid", "")
            outcome = test.get("outcome", "unknown")
            outcomes[nodeid] = outcome
            
            # Also store basename form for flexible matching
            if "/" in nodeid:
                basename_nodeid = nodeid.split("/")[-1]
                outcomes[basename_nodeid] = outcome
        
        return outcomes
    
    finally:
        if report_path.exists():
            report_path.unlink()


def load_outcomes_from_reports(report_paths: List[Path], verbose: bool = False) -> Dict[str, str]:
    """
    Load test outcomes from existing JSON report files.
    
    P1-5: Supports --from-reports for pre-run reports.
    """
    outcomes = {}
    
    for report_path in report_paths:
        if not report_path.exists():
            print(f"Warning: Report not found: {report_path}")
            continue
        
        try:
            with open(report_path) as f:
                report = json.load(f)
            
            count = 0
            for test in report.get("tests", []):
                nodeid = test.get("nodeid", "")
                outcome = test.get("outcome", "unknown")
                
                # Store multiple forms for flexible matching (P1-2)
                outcomes[nodeid] = outcome
                count += 1
                
                # Also store without path prefix
                if "/" in nodeid:
                    basename_nodeid = nodeid.split("/")[-1]
                    outcomes[basename_nodeid] = outcome
            
            if verbose:
                print(f"  Loaded {count} outcomes from {report_path.name}")
            else:
                print(f"Loaded {count} outcomes from {report_path.name}")
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse {report_path}: {e}")
    
    return outcomes



# =============================================================================
# HEADLESS EVIDENCE (PHASE_4 stage 4.2c-2d): the headless engine's own per-capability verdicts
# =============================================================================
# The headless tests express a known defect as an expected failure INSIDE a passing pytest test, so pytest's outcome
# cannot carry it. The engine writes a verdict per capability into every scenario report (expect.py::feature_status).
# With --from-headless, those verdicts enter the matrix: only reports of the folder's session (session.json), only
# healthy runs (no deliberate faults, no harness controls), only taxonomy capabilities (engine labels skipped, any
# other unknown id is an error). Order: Broken > Known Issue > Unverified > Working — a confirmed known defect is never
# hidden by an unverified result elsewhere. Evidence of only "response" checks (a value changes and returns) verifies
# a capability only in part. Without --from-headless nothing in this script's output changes.
HEADLESS_RANK = {"Working": 0, "Unverified": 1, "Known Issue": 2, "Broken": 3}


def load_headless_evidence(folders: List[str], taxonomy: Dict) -> Dict[str, dict]:
    evidence: Dict[str, dict] = {}
    for folder in folders:
        folder = Path(folder)
        meta_path = folder / "session.json"
        if not meta_path.is_file():
            raise SystemExit(f"--from-headless {folder}: no session.json (not the output folder of a finished headless session)")
        meta = json.loads(meta_path.read_text())
        sid = meta.get("session")
        if not sid:
            raise SystemExit(f"--from-headless {folder}: session.json names no session")
        found = 0
        for f in sorted(folder.rglob("report*.json")):
            try:
                r = json.loads(f.read_text())
            except Exception as exc:
                raise SystemExit(f"--from-headless: unreadable report {f}: {exc}")
            if r.get("session") != sid:
                continue                                 # another session's report in a re-used folder
            found += 1
            if r.get("kind") not in ("healthy", "fault", "harness"):
                raise SystemExit(f"--from-headless: report without a valid kind: {f}")
            if r["kind"] != "healthy":
                continue
            base = re.sub(r" \[node\]$", "", r["id"])
            scopes: Dict[str, set] = {}
            defects: Dict[str, set] = {}
            outs: Dict[str, set] = {}
            for o in r.get("outcomes", []):
                feat = o.get("feature") or ""
                if feat and o.get("assertion", True):
                    scopes.setdefault(feat, set()).add(o.get("scope", "value"))
                    outs.setdefault(feat, set()).add(o.get("outcome", ""))
                    m = re.search(r"known defect (\S+?):", o.get("detail", "")) if o.get("outcome") == "XFAIL" else None
                    if m:
                        defects.setdefault(feat, set()).add(m.group(1))
            for feat, status in (r.get("features") or {}).items():
                if feat.startswith("ENGINE."):
                    continue
                o = outs.get(feat)
                if o:            # this report's status from its checks: a known defect is never hidden by a skip
                    status = ("Broken" if o & {"FAIL", "ERROR", "XPASS"} else "Known Issue" if "XFAIL" in o
                              else "Unverified" if "SKIP" in o or "PASS" not in o else "Working")
                if feat not in taxonomy:
                    raise SystemExit(f"--from-headless: capability {feat!r} of {r['id']} is not in the taxonomy")
                e = evidence.setdefault(feat, {"status": "Working", "value_evidence": False, "scenarios": {}})
                if HEADLESS_RANK[status] > HEADLESS_RANK[e["status"]]:
                    e["status"] = status
                e["value_evidence"] |= "value" in scopes.get(feat, set())
                sc = e["scenarios"].setdefault(base, {"backends": set(), "defects": set(), "status": "Working",
                                                      "skipped": False})
                sc["backends"].add(r.get("backend", "browser"))
                sc["defects"] |= defects.get(feat, set())
                sc["skipped"] |= bool(o and "SKIP" in o and status != "Unverified")
                if HEADLESS_RANK[status] > HEADLESS_RANK[sc["status"]]:
                    sc["status"] = status
        if found == 0:
            raise SystemExit(f"--from-headless {folder}: no report of session {sid}")
    return evidence


MATRIX_RANK = [("Broken", 4), ("Known Issue", 3), ("Unknown", 2), ("Partly", 2), ("Working", 1)]


def _matrix_rank(status: str) -> int:
    return next((r for key, r in MATRIX_RANK if key in status), 0)      # Planned / No Tests: no evidence


def headless_matrix_status(e: dict) -> str:
    s = e["status"]
    if s == "Broken":
        return "🧨 Broken"
    if s == "Known Issue":
        return "⚠️ Known Issue"
    if s == "Unverified":
        return "❓ Unknown"
    return "✅ Working" if e["value_evidence"] else "❓ Partly verified (response only)"


def combine_status(status: str, test_count: int, tests: List[str], e: dict) -> Tuple[str, int]:
    """The worse of the decorator status (if it has tests) and the headless status; counts add up."""
    h = headless_matrix_status(e)
    best = h if not tests or _matrix_rank(h) >= _matrix_rank(status) else status
    return best, (test_count if tests else 0) + len(e["scenarios"])

# =============================================================================
# STATUS COMPUTATION (PHASE_4_3 J4 v0.6: exact matching, every variant)
# =============================================================================

def matched_outcomes(nodeid: str, test_outcomes: Dict[str, str]) -> List[Tuple[str, str]]:
    """The report entries of one test: its exact node id or its parametrized variants "<nodeid>[...]"; the
    project-relative form first, then the file-name form. Empty if the test has no result."""
    for form in (nodeid, nodeid.split("/")[-1]):
        hits = sorted((k, v) for k, v in test_outcomes.items() if k == form or k.startswith(form + "["))
        if hits:
            return hits
    return []


def compute_status(
    feature_id: str,
    feature_test_nodeids: List[str],
    test_outcomes: Dict[str, str],
    taxonomy: Dict,
) -> Tuple[str, int, List[str]]:
    """
    Compute feature status from test outcomes.
    
    PHASE_4_3 J4 v0.6: exact node-id matching, every parametrized variant counts (matched_outcomes).
    
    Returns (status_string, test_count, failed_tests)
    
    Status priority (highest to lowest):
    1. failed/error -> 🧨 Broken
    2. xpassed -> 🧨 Broken (unexpected pass)
    3. xfailed -> ⚠️ Known Issue
    4. unknown/skipped only -> ❓ Unknown
    5. all passed -> ✅ Working
    6. no tests + planned -> 📋 Planned
    7. no tests -> ❌ No Tests
    """
    if not feature_test_nodeids:
        # Check if feature is planned
        if taxonomy.get(feature_id, {}).get("planned"):
            return "📋 Planned", 0, []
        return "❌ No Tests", 0, []
    
    outcomes = []
    failed_tests = []

    for nodeid in feature_test_nodeids:
        # PHASE_4_3 J4 v0.6 (review M-1): EXACT matching only. A test's results are the report entries with exactly its
        # node id, or its node id followed by a parameter list "[...]" - EVERY variant counts. The project-relative
        # form is tried first, then the file-name form (reports from a run started elsewhere). The former suffix,
        # test-name and substring strategies are gone: they could attribute another test's result (test_x to
        # test_x_other) or only the first variant of a parametrized test. No match -> "unknown", never a pass.
        hits = matched_outcomes(nodeid, test_outcomes)
        if not hits:
            outcomes.append("unknown")
            continue
        for key, outcome in hits:
            outcomes.append(outcome)
            if outcome in ("failed", "error", "xpassed"):
                failed_tests.append(key)

    test_count = len(feature_test_nodeids)
    
    # Priority 1: Any failed = broken
    if "failed" in outcomes or "error" in outcomes:
        return "🧨 Broken", test_count, failed_tests
    
    # Priority 2: xpassed = broken (unexpected pass)
    if "xpassed" in outcomes:
        return "🧨 Broken", test_count, failed_tests
    
    # Priority 3: xfailed
    if "xfailed" in outcomes:
        return "⚠️ Known Issue", test_count, []
    
    # Priority 4: Only unknown/skipped
    known = [o for o in outcomes if o not in ("unknown", "skipped")]
    if not known:
        return "❓ Unknown", test_count, []
    
    # Priority 5: All passed
    if all(o == "passed" for o in known):
        return "✅ Working", test_count, []
    
    return "❓ Unknown", test_count, []


# =============================================================================
# MATRIX GENERATION
# =============================================================================

def generate_matrix(
    taxonomy: Dict,
    test_outcomes: Dict[str, str],
    feature_tests: Dict[str, List[str]],
    verbose: bool = False,
    headless: Optional[Dict[str, dict]] = None,
) -> Tuple[str, dict]:
    """
    Generate CAPABILITY_MATRIX content (Markdown + JSON).
    
    P1-4 Fix: Returns both markdown and JSON data.
    """
    lines = []
    json_data = {
        "generated": datetime.now().isoformat(),
        "generator": "scripts/generate_capability_matrix.py",
        "version": "1.0",
        "phase": "0.1.A",
        "features": {},
        "summary": {},
    }
    
    # Header
    lines.extend([
        "# CAPABILITY_MATRIX",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "**Generator:** `scripts/generate_capability_matrix.py`",
        "",
        "**Phase:** 0.1.A",
        "",
        "> This matrix shows test coverage for RootInteractive features.",
        "> Status is derived from pytest outcomes using `pytest-json-report`.",
        "",
        "---",
        "",
    ])
    
    # Summary counts
    counts = {"working": 0, "broken": 0, "known_issue": 0, "planned": 0, "no_tests": 0, "unknown": 0}
    broken_features = {}  # feature_name -> [failed_tests]
    
    # Get categories
    categories = set()
    for fid in taxonomy:
        if "." in fid:
            categories.add(fid.split(".")[0])
    
    # Generate by category
    for category in sorted(categories):
        cat_features = {k: v for k, v in taxonomy.items() if k.startswith(f"{category}.")}
        if not cat_features:
            continue
        
        # Category header with description
        cat_descriptions = {
            "DSL": "Domain-Specific Language operations",
            "ENC": "Encoding and data transfer",
            "JOIN": "Join and cross-table operations",
            "HIST": "Histogram operations",
            "ALIAS": "Column aliasing operations",
        }
        cat_desc = cat_descriptions.get(category, "")
        
        lines.extend([
            f"## {category}",
            "",
        ])
        if cat_desc:
            lines.append(f"*{cat_desc}*")
            lines.append("")
        
        lines.extend([
            "| Feature | Priority | Status | Backend | Layer | Tests |",
            "|---------|----------|--------|---------|-------|-------|",
        ])
        
        for feature_id, info in sorted(cat_features.items()):
            name = info.get("name", feature_id)
            priority = info.get("priority", "P2")
            backends = ", ".join(info.get("backends", []))
            layer = info.get("layer", "unit")
            tests = feature_tests.get(feature_id, [])
            
            status, test_count, failed = compute_status(feature_id, tests, test_outcomes, taxonomy)
            if headless and feature_id in headless:                         # PHASE_4 4.2c-2d
                status, test_count = combine_status(status, test_count, tests, headless[feature_id])
            
            # Update counts
            if "Working" in status:
                counts["working"] += 1
            elif "Broken" in status:
                counts["broken"] += 1
                if failed:
                    broken_features[name] = failed
            elif "Known Issue" in status:
                counts["known_issue"] += 1
            elif "Planned" in status:
                counts["planned"] += 1
            elif "No Tests" in status:
                counts["no_tests"] += 1
            else:
                counts["unknown"] += 1
            
            # Format tests column. PHASE_4_3 J4 v0.6 (review M-2): test functions, executed cases (every parameter
            # variant is a case) and functions without any result in the reports are shown separately.
            hits = {n: matched_outcomes(n, test_outcomes) for n in tests}
            cases = sum(len(h) for h in hits.values())
            without_result = sorted(n for n, h in hits.items() if not h)
            if tests or (headless and feature_id in headless):
                tests_display = f"{test_count} test{'s' if test_count != 1 else ''}"
                if tests:
                    tests_display += f", {cases} case{'s' if cases != 1 else ''}"
                    if without_result:
                        tests_display += f", {len(without_result)} without result"
            else:
                tests_display = "-"
            
            lines.append(f"| `{feature_id}` | {priority} | {status} | {backends} | {layer} | {tests_display} |")
            
            # JSON data
            json_data["features"][feature_id] = {
                "name": name,
                "description": info.get("description", ""),
                "status": status,
                "priority": priority,
                "backends": info.get("backends", []),
                "layer": layer,
                "test_count": test_count,
                "cases": cases,
                "without_result": without_result,
                "tests": tests,
                "failed_tests": failed,
                "planned": info.get("planned", False),
                "regression": info.get("regression"),
            }
            if headless and feature_id in headless:                         # PHASE_4 4.2c-2d
                json_data["features"][feature_id]["headless_scenarios"] = {
                    sid: {"backends": sorted(v["backends"]), "status": v["status"], "defects": sorted(v["defects"])}
                    for sid, v in sorted(headless[feature_id]["scenarios"].items())}
        
        lines.extend(["", ""])
    
    # Broken features detail (if any)
    if broken_features:
        lines.extend([
            "---",
            "",
            "## 🧨 Broken Features Detail",
            "",
        ])
        for name, failed in broken_features.items():
            lines.append(f"**{name}:**")
            for test in failed[:5]:
                lines.append(f"- `{test}`")
            if len(failed) > 5:
                lines.append(f"- ... and {len(failed) - 5} more")
            lines.append("")
    
    # ==========================================================================
    # Test Coverage Details (for manual verification)
    # ==========================================================================
    lines.extend([
        "---",
        "",
        "## Test Coverage Details",
        "",
        "Tests per feature (for traceability). Approval logic: Feature = ✅ Working iff **ALL** tests pass.",
        "",
    ])
    
    for feature_id, info in sorted(taxonomy.items()):
        name = info.get("name", feature_id)
        feature_test_list = feature_tests.get(feature_id, [])
        test_count = len(feature_test_list)
        hl = (headless or {}).get(feature_id)
        if hl:
            test_count += len(hl["scenarios"])
        
        lines.append("<details>")
        lines.append(f"<summary><strong>{feature_id}</strong> — {name} ({test_count} tests)</summary>")
        lines.append("")
        
        if feature_test_list:
            for nodeid in sorted(feature_test_list):
                # Try to get outcome for this test
                outcome = None
                
                # Strategy 1: Exact match
                outcome = test_outcomes.get(nodeid)
                
                # Strategy 2: Match with full path prefix
                if outcome is None:
                    for key, val in test_outcomes.items():
                        if key.endswith(nodeid):
                            outcome = val
                            break
                
                # Strategy 3: Match just the test name part
                if outcome is None:
                    test_name = nodeid.split("::")[-1]
                    for key, val in test_outcomes.items():
                        if key.endswith("::" + test_name):
                            outcome = val
                            break
                
                # Status emoji based on outcome
                if outcome == "passed":
                    status_emoji = "✅"
                elif outcome == "failed":
                    status_emoji = "❌"
                elif outcome == "error":
                    status_emoji = "💥"
                elif outcome == "skipped":
                    status_emoji = "⏭️"
                elif outcome == "xfailed":
                    status_emoji = "⚠️"
                elif outcome == "xpassed":
                    status_emoji = "🔄"
                else:
                    status_emoji = "❓"
                
                lines.append(f"- {status_emoji} `{nodeid}`")
        elif not hl:
            if info.get("planned"):
                lines.append("*Feature planned, not yet implemented*")
            else:
                lines.append("*No tests with @pytest.mark.feature marker*")
        if hl:                                                              # PHASE_4 4.2c-2d: headless evidence
            mark = {"Working": "✅", "Known Issue": "⚠️", "Unverified": "❓", "Broken": "🧨"}
            for sid, v in sorted(hl["scenarios"].items()):
                extra = f" — known defect {', '.join(sorted(v['defects']))}" if v["defects"] else ""
                extra += " — also skipped checks" if v.get("skipped") else ""
                lines.append(f"- {mark[v['status']]} headless scenario `{sid}` ({', '.join(sorted(v['backends']))}){extra}")
        
        lines.append("")
        lines.append("</details>")
        lines.append("")
    
    # Summary section
    total = sum(counts.values())
    lines.extend([
        "---",
        "",
        "## Summary",
        "",
        "| Status | Count | Percentage |",
        "|--------|-------|------------|",
        f"| ✅ Working | {counts['working']} | {100*counts['working']/max(total,1):.1f}% |",
        f"| 🧨 Broken | {counts['broken']} | {100*counts['broken']/max(total,1):.1f}% |",
        f"| ⚠️ Known Issue | {counts['known_issue']} | {100*counts['known_issue']/max(total,1):.1f}% |",
        f"| 📋 Planned | {counts['planned']} | {100*counts['planned']/max(total,1):.1f}% |",
        f"| ❌ No Tests | {counts['no_tests']} | {100*counts['no_tests']/max(total,1):.1f}% |",
        f"| ❓ Unknown | {counts['unknown']} | {100*counts['unknown']/max(total,1):.1f}% |",
        f"| **Total** | **{total}** | **100%** |",
        "",
    ])
    
    # Legend
    lines.extend([
        "---",
        "",
        "## Legend",
        "",
        "| Status | Meaning |",
        "|--------|---------|",
        "| ✅ Working | All tests pass |",
        "| 🧨 Broken | At least one test fails |",
        "| ⚠️ Known Issue | Expected failure (xfail) |",
        "| 📋 Planned | Feature planned, not yet tested |",
        "| ❌ No Tests | No tests cover this feature |",
        "| ❓ Unknown | Test status unclear |",
        "",
        "## Priority Levels",
        "",
        "| Priority | Meaning |",
        "|----------|---------|",
        "| P0 | Critical - blocks release |",
        "| P1 | Important - should fix before release |",
        "| P2 | Nice to have - can defer |",
        "",
        "---",
        "",
        "*Auto-generated by `scripts/generate_capability_matrix.py`*",
        "",
        "*Phase 0.1.A provides coverage accounting. Invariance validation requires Phase 0.1.B.*",
    ])
    
    json_data["summary"] = counts
    json_data["summary"]["total"] = total
    
    return "\n".join(lines), json_data


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate CAPABILITY_MATRIX from pytest markers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic generation (runs pytest internally)
    python scripts/generate_capability_matrix.py
    
    # Use existing pytest report
    pytest bokeh/ --json-report --json-report-file=report.json -m feature
    python scripts/generate_capability_matrix.py --from-reports report.json
    
    # Multiple reports
    python scripts/generate_capability_matrix.py --from-reports report1.json report2.json
    
    # Dry run
    python scripts/generate_capability_matrix.py --dry-run
        """
    )
    parser.add_argument(
        "--test-dir",
        nargs="*",
        default=[
            "RootInteractive/InteractiveDrawing/bokeh/test_*.py",
            "RootInteractive/InteractiveDrawing/bokeh/headless/test_*.py",   # PHASE_4_3 J4: the Node join tests
            "RootInteractive/Tools/test_*.py",
            "RootInteractive/onnx/test_*.py"
        ],
        help="Test file pattern(s) (default: bokeh and Tools directories)"
    )
    parser.add_argument(
        "--output",
        default="RootInteractive/InteractiveDrawing/bokeh/doc/CAPABILITY_MATRIX.md",
        help="Output path for Markdown"
    )
    parser.add_argument(
        "--from-reports",
        nargs="*",
        help="Read outcomes from existing JSON report files (pytest-json-report format)"
    )
    parser.add_argument(
        "--from-headless",
        nargs="*",
        default=[],
        help="Headless output folders of finished sessions (session.json + scenario reports): the engine's own "
             "per-capability verdicts of healthy runs enter the matrix (PHASE_4 stage 4.2c-2d)"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest (use with --from-reports or for marker parsing only)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print output without writing files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CAPABILITY_MATRIX Generator (Phase 0.1.A)")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # Import taxonomy
    # -------------------------------------------------------------------------
    FEATURE_TAXONOMY = None
    
    import_attempts = [
        "feature_taxonomy",
        "RootInteractive.InteractiveDrawing.bokeh.feature_taxonomy",
    ]
    
    for module_name in import_attempts:
        try:
            import importlib
            module = importlib.import_module(module_name)
            FEATURE_TAXONOMY = getattr(module, "FEATURE_TAXONOMY", None)
            if FEATURE_TAXONOMY is not None:
                if args.verbose:
                    print(f"Loaded taxonomy from: {module_name}")
                break
        except ImportError:
            continue
    
    if FEATURE_TAXONOMY is None:
        print("Error: Cannot import feature_taxonomy")
        print("Tried: " + ", ".join(import_attempts))
        print("Make sure feature_taxonomy.py exists in bokeh/ directory")
        sys.exit(1)
    
    print(f"Loaded {len(FEATURE_TAXONOMY)} features from taxonomy")
    
    # -------------------------------------------------------------------------
    # Parse feature markers from test files (support multiple patterns)
    # -------------------------------------------------------------------------
    test_patterns = args.test_dir if isinstance(args.test_dir, list) else [args.test_dir]
    print(f"\nParsing @pytest.mark.feature markers from {len(test_patterns)} pattern(s)...")
    
    feature_tests = {}
    for pattern in test_patterns:
        pattern_tests = parse_feature_markers_from_files(pattern, args.verbose)
        # Merge results
        for fid, tests in pattern_tests.items():
            if fid not in feature_tests:
                feature_tests[fid] = []
            feature_tests[fid].extend(tests)
    
    total_mappings = sum(len(v) for v in feature_tests.values())
    print(f"Found {total_mappings} test-feature mappings across {len(feature_tests)} features")
    
    if args.verbose:
        for fid, tests in sorted(feature_tests.items()):
            print(f"  {fid}: {len(tests)} tests")
    
    # -------------------------------------------------------------------------
    # Collect test outcomes
    # -------------------------------------------------------------------------
    test_outcomes = {}
    
    if args.from_reports:
        print(f"\nLoading outcomes from {len(args.from_reports)} report file(s)...")
        test_outcomes = load_outcomes_from_reports(
            [Path(p) for p in args.from_reports],
            args.verbose
        )
    elif args.skip_tests:
        print("\nSkipping pytest run (--skip-tests)")
    else:
        print("\nRunning pytest to collect outcomes...")
        test_outcomes = {}
        for pattern in test_patterns:
            pattern_outcomes = run_pytest_for_outcomes(pattern, args.verbose)
            test_outcomes.update(pattern_outcomes)
    
    print(f"Collected {len(test_outcomes)} test outcomes")
    
    if args.verbose and test_outcomes:
        outcome_counts = {}
        for outcome in set(test_outcomes.values()):
            outcome_counts[outcome] = sum(1 for v in test_outcomes.values() if v == outcome)
        print(f"Outcome distribution: {outcome_counts}")
    
    # -------------------------------------------------------------------------
    # Generate matrix
    # -------------------------------------------------------------------------
    print("\nGenerating capability matrix...")
    headless = load_headless_evidence(args.from_headless, FEATURE_TAXONOMY) if args.from_headless else None
    markdown, json_data = generate_matrix(
        FEATURE_TAXONOMY,
        test_outcomes,
        feature_tests,
        args.verbose,
        headless=headless,
    )
    
    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Markdown output (first 80 lines):")
        print("=" * 60)
        print("\n".join(markdown.split("\n")[:80]))
        print("...")
        print("=" * 60)
        print("\nJSON summary:")
        print(json.dumps(json_data["summary"], indent=2))
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write Markdown
        output_path.write_text(markdown)
        print(f"\n✅ Generated {output_path}")
        
        # Write JSON (P1-4)
        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(json_data, indent=2))
        print(f"✅ Generated {json_path}")
        
        # Print summary
        summary = json_data["summary"]
        print(f"\n📊 Summary: {summary['working']} working, {summary['broken']} broken, "
              f"{summary['planned']} planned, {summary['no_tests']} no tests")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
