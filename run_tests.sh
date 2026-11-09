#!/bin/bash
ri_main() {
# =============================================================================
# run_tests.sh — RootInteractive Test Runner
# =============================================================================
# 
# Runs all tests in parallel and generates CAPABILITY_MATRIX documentation.
# Saves logs and reports to test_logs/<dirname>_<date>/
#
# Usage:
#   ./run_tests.sh              # Run all tests + generate matrix
#   ./run_tests.sh --quick      # Run tests only (skip matrix generation)
#   ./run_tests.sh --matrix     # Generate matrix only (skip tests)
#   ./run_tests.sh -n 4         # Use 4 parallel workers (default: 6)
#   ./run_tests.sh -v           # Verbose output
#   ./run_tests.sh -h           # Show help
#
# Output structure:
#   test_logs/
#   ├── bokeh_20260203_001500/
#   │   ├── report.json
#   │   ├── test.log
#   │   └── summary.txt
#   ├── Tools_20260203_001500/
#   │   ├── report.json
#   │   ├── test.log
#   │   └── summary.txt
#   └── latest -> bokeh_20260203_001500  (symlink to most recent)
#
# Phase: 0.1.A
# Date: 2026-02-03
# =============================================================================


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Test directories (without glob - pytest discovers test_*.py)
TEST_DIRS=(
    "RootInteractive/InteractiveDrawing/bokeh"
    "RootInteractive/InteractiveDrawing/bokeh/headless"
    "RootInteractive/Tools"
    "RootInteractive/onnx"
    "RootInteractive/MLpipeline/tests"
)

# Output locations
LOG_BASE_DIR="test_logs"
MATRIX_OUTPUT="RootInteractive/InteractiveDrawing/bokeh/doc/CAPABILITY_MATRIX.md"

# Default parallel workers
PARALLEL_WORKERS=6

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE_SHORT=$(date +%Y%m%d)

# P0-3 FIX: Initialize counters at top to avoid unbound variable errors
RUN_FAILED=0
TOTAL_ERRORS=0
TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0

# =============================================================================
# P1-1 FIX: Dependency Check
# =============================================================================

check_dependencies() {
    local missing=0
    
    # Check pytest-json-report
    if ! python3 -c "import pytest_jsonreport" 2>/dev/null; then
        echo "⚠️  pytest-json-report not installed"
        echo "   Install with: pip install pytest-json-report"
        missing=1
    fi
    
    # Check pytest-xdist (only if parallel requested)
    if [[ $PARALLEL_WORKERS -gt 1 ]]; then
        if ! python3 -c "import xdist" 2>/dev/null; then
            echo "⚠️  pytest-xdist not installed, falling back to sequential execution"
            echo "   Install with: pip install pytest-xdist"
            PARALLEL_WORKERS=1
        fi
    fi
    
    return $missing
}

# Colors (if terminal supports it)
if [[ -t 1 ]] && command -v tput &>/dev/null; then
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4)
    BOLD=$(tput bold)
    RESET=$(tput sgr0)
else
    RED="" GREEN="" YELLOW="" BLUE="" BOLD="" RESET=""
fi

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat << 'EOF'
RootInteractive Test Runner
============================

Usage:
  ./run_tests.sh [OPTIONS]

Options:
  -h, --help       Show this help message
  -v, --verbose    Verbose pytest output
  -n N             Number of parallel workers (default: 6, use 1 for sequential)
  --quick          Run tests only (skip CAPABILITY_MATRIX generation)
  --matrix         Generate CAPABILITY_MATRIX only (skip running tests)
  --dry-run        Show what would be done without doing it
  --full           Headless tests: the full suite (all generated cases, every browser test);
                   default: the fast suite (tests marked slow are left out)
  --target NAME    Run only this target (repeatable; each target runs separately, with its own log folder,
                   timing and summary line). A target is a pytest marker of the headless tests (hyphens may be
                   used for underscores), or "joins" (node --test). "all": everything, as --full.
  --list-targets   List the targets, their test counts and descriptions (with --ids: also every test id)

Test Directories:
  - RootInteractive/InteractiveDrawing/bokeh/ (without headless/, which has its own entry)
  - RootInteractive/InteractiveDrawing/bokeh/headless/ (Node and browser tests; fast suite unless --full)
  - RootInteractive/joins/ (Node tests, node --test)
  - RootInteractive/Tools/

Output Structure:
  test_logs/
  ├── bokeh_20260203_001500/
  │   ├── report.json      # pytest-json-report output
  │   ├── test.log         # Full test output
  │   └── summary.txt      # Pass/fail summary
  └── latest -> bokeh_20260203_001500

  RootInteractive/InteractiveDrawing/bokeh/doc/
  ├── CAPABILITY_MATRIX.md
  └── CAPABILITY_MATRIX.json

Examples:
  # Full run: tests (6 workers) + matrix
  ./run_tests.sh

  # Run with 4 parallel workers
  ./run_tests.sh -n 4

  # Sequential run (no parallel)
  ./run_tests.sh -n 1

  # Quick test run (no matrix)
  ./run_tests.sh --quick

  # Regenerate matrix from existing reports
  ./run_tests.sh --matrix

  # Compare test runs
  diff test_logs/bokeh_20260201_*/summary.txt test_logs/bokeh_20260202_*/summary.txt

EOF
    return 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

VERBOSE=""
QUICK_MODE=0
MATRIX_ONLY=0
DRY_RUN=0
FULL_MODE=0
TARGETS=()
LIST_TARGETS=0
LIST_IDS=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_help; return 0 ;;
        -v|--verbose) VERBOSE="-v" ;;
        -n)
            shift
            PARALLEL_WORKERS="${1:-}"
            if [[ ! "$PARALLEL_WORKERS" =~ ^[1-9][0-9]*$ ]]; then echo "Invalid worker count"; return 2; fi
            ;;
        --quick) QUICK_MODE=1 ;;
        --matrix) MATRIX_ONLY=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --full) FULL_MODE=1 ;;
        --target)
            shift
            if [[ -z "${1:-}" ]]; then echo "--target needs a name (see --list-targets)"; return 2; fi
            TARGETS+=("$1")
            ;;
        --list-targets) LIST_TARGETS=1 ;;
        --ids) LIST_IDS=1 ;;
        *) echo "Unknown option: $1"; show_help; return 2 ;;
    esac
    shift
done

# =============================================================================
# Functions
# =============================================================================

print_header() {
    echo ""
    echo "${BLUE}${BOLD}=============================================="
    echo "$1"
    echo "==============================================${RESET}"
}

print_success() {
    echo "${GREEN}✅ $1${RESET}"
}

print_warning() {
    echo "${YELLOW}⚠️  $1${RESET}"
}

print_error() {
    echo "${RED}❌ $1${RESET}"
}

# =============================================================================
# Main
# =============================================================================

cd "$PROJECT_ROOT"

# P1-1 FIX: Check dependencies before running
check_dependencies
if [[ $? -ne 0 ]]; then RUN_FAILED=1; fi

print_header "RootInteractive Test Runner"
echo "Project root: $PROJECT_ROOT"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Parallel workers: $PARALLEL_WORKERS"
echo "Log directory: $LOG_BASE_DIR/"

# Create base log directory
mkdir -p "$LOG_BASE_DIR"

REPORT_FILES=()
if [[ $DRY_RUN -eq 0 && $MATRIX_ONLY -eq 0 ]]; then
    python3 - "$PROJECT_ROOT" "$LOG_BASE_DIR/source_before_${TIMESTAMP}.json" <<'BEFOREPY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); files=[root/'run_tests.sh',*(root/'RootInteractive/MLpipeline').rglob('*.py')]
pathlib.Path(sys.argv[2]).write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},indent=2))
BEFOREPY
    if [[ $? -ne 0 ]]; then RUN_FAILED=1; fi
fi

# =============================================================================
# Targets (PHASE_4 stage 4.2c-2c): pytest markers of the headless tests, plus "joins" (node --test)
# =============================================================================
HEADLESS_DIR="RootInteractive/InteractiveDrawing/bokeh/headless"
NODE_TAP=()                                  # TAP output gives a readable count on every Node version (C-3)
if command -v node > /dev/null 2>&1 && node --help 2>/dev/null | grep -q -- "--test-reporter"; then NODE_TAP=(--test-reporter=tap); fi

target_markers() {                           # the registered target markers ("<name>: target: <description>")
    python3 -m pytest --markers -p no:cacheprovider "$HEADLESS_DIR" 2>/dev/null \
        | sed -n 's/^@pytest\.mark\.\([a-z_0-9]*\): target: \(.*\)/\1|\2/p' | sort -u -t'|' -k1,1
}

publish_matrix() {                           # $1: destination .md; the rest: generator arguments
    # The Architect's decision (4.2c-2d, option C, approved by three reviewers): both files are generated in a temporary
    # folder and published TOGETHER only on success (both present, the JSON readable); if generation fails or crashes,
    # the previous pair stays unchanged and the caller fails the run. Measured failures are published.
    local dest="$1"; shift
    local tmp; tmp=$(mktemp -d)
    if python3 scripts/generate_capability_matrix.py "$@" --output "$tmp/matrix.md" \
            && [[ -s "$tmp/matrix.md" && -s "$tmp/matrix.json" ]] \
            && python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$tmp/matrix.json"; then
        mkdir -p "$(dirname "$dest")"
        mv "$tmp/matrix.md" "$dest" && mv "$tmp/matrix.json" "${dest%.md}.json"
        local ok=$?
        rm -rf "$tmp"
        return $ok
    fi
    rm -rf "$tmp"
    return 1
}

is_target() {                                # a registered target marker: exact name, never a pattern (4.2c-2c F-1)
    cut -d'|' -f1 <<< "$KNOWN_TARGETS" | grep -qxF -- "${1//-/_}"
}

tap_counts() {                               # "selected passed failed skipped test-errors collection-errors"
    local p f s
    p=$(grep -E "^# pass [0-9]+" "$1" | tail -1 | awk '{print $3}')
    f=$(grep -E "^# fail [0-9]+" "$1" | tail -1 | awk '{print $3}')
    s=$(grep -E "^# skipped [0-9]+" "$1" | tail -1 | awk '{print $3}')
    if [[ -z "$p" || -z "$f" ]]; then return 1; fi
    if [[ $((p + f + ${s:-0})) -le 0 ]]; then return 1; fi           # a run with no tests is not a pass (4.2c-2c v0.2)
    echo "$((p + f + ${s:-0})) $p $f ${s:-0} 0 0"
}

json_counts() {                              # the same six numbers from pytest-json-report; ids to selected_tests.txt
    python3 - "$1" "$2" <<'COUNTPY'
import json, sys
try:                                         # an unreadable or missing report is a failure, never counted as zero tests
    r = json.load(open(sys.argv[1]))
    s = r["summary"]
    coll = sum(1 for c in r.get("collectors", []) if c.get("outcome") == "failed")
    tests = [t["nodeid"] for t in r["tests"]]
    open(sys.argv[2], "w").write("".join(t + "\n" for t in tests))
except Exception as exc:
    print(f"report unreadable: {exc!r}", file=sys.stderr); sys.exit(3)
print(len(tests), s.get("passed", 0), s.get("failed", 0), s.get("skipped", 0), s.get("error", 0), coll)
COUNTPY
}

run_target() {
    local name="$1" marker="${1//-/_}" out="$LOG_BASE_DIR/target_${1}_${TIMESTAMP}" rc=0 counts note="" t0 t1
    local cmd=()
    mkdir -p "$out"
    if [[ "$name" == "joins" ]]; then
        local files=(RootInteractive/joins/test_*.mjs)
        cmd=(node --test "${NODE_TAP[@]}" "${files[@]}")
    elif is_target "$name"; then
        cmd=(env -u DISPLAY RI_HEADLESS_OUT="$PROJECT_ROOT/$out/headless_out"
             RI_HEADLESS_BUNDLE="${RI_HEADLESS_BUNDLE:-$PROJECT_ROOT/$HEADLESS_DIR/_out}"
             RI_HEADLESS_REQUIRED=1 RI_HEADLESS_CHECK_REQUIRED=1 RI_HEADLESS_TARGET="$marker"
             python3 -m pytest "$HEADLESS_DIR" -m "$marker" -n "$PARALLEL_WORKERS" --timeout=900 -q -p no:cacheprovider
             --json-report --json-report-file="$PROJECT_ROOT/$out/report.json")
    else
        echo "unknown target: $name (see ./run_tests.sh --list-targets)" | tee "$out/test.log"
        printf "%-22s %8s %6s %6s %7s %8s %11s %7s %4s  %s\n" "$name" 0 0 0 0 0 0 0.0 2 "$out  UNKNOWN TARGET" >> "$TARGET_SUMMARY"
        TARGET_FAILED=1; return
    fi
    printf '%q ' "${cmd[@]}" > "$out/command.txt"; echo >> "$out/command.txt"
    if [[ $DRY_RUN -eq 1 ]]; then echo "  [DRY RUN] $(cat "$out/command.txt")"; return; fi
    echo "${BOLD}Target: $name${RESET}"
    t0=$(date +%s.%N)
    if [[ "$name" == "joins" && ! -e "${files[0]}" ]]; then
        echo "no joins test files (RootInteractive/joins/test_*.mjs)" > "$out/test.log"; rc=1
    elif [[ "$name" == "joins" && ! -f RootInteractive/joins/ao2d_fixture.json ]]; then
        { echo "RootInteractive/joins/ao2d_fixture.json is missing (generated, not in git); create it with:"
          echo "    (cd RootInteractive/Tools/generators && python3 generate_ao2d_fixture.py ../../joins/ao2d_fixture.json)"; } | tee "$out/test.log"
        rc=1; note="  FIXTURE MISSING"
    else
        "${cmd[@]}" > "$out/test.log" 2>&1; rc=$?
    fi
    t1=$(date +%s.%N)
    if [[ "$name" == "joins" ]]; then
        counts=$(tap_counts "$out/test.log") || { counts="0 0 0 0 0 0"; note="${note:-  COUNT UNREADABLE OR NO TESTS}"; rc=1; }
    else
        if ! counts=$(json_counts "$out/report.json" "$out/selected_tests.txt" 2>> "$out/test.log"); then
            counts="0 0 0 0 0 0"; note="  REPORT UNREADABLE"; [[ $rc -eq 0 ]] && rc=1
        fi
        [[ $rc -eq 5 ]] && note="  EMPTY SELECTION"
        # the capability matrix of this target, in its own folder (4.2c-2d; never the canonical matrix)
        if [[ -f "$out/headless_out/session.json" ]] && find "$out/headless_out" -name 'report*.json' -print -quit | grep -q .; then
            if ! publish_matrix "$out/CAP_MATRIX_${marker}.md" --from-reports "$out/report.json" \
                    --from-headless "$out/headless_out" >> "$out/test.log" 2>&1; then
                note="${note}  MATRIX GENERATION FAILED"; [[ $rc -eq 0 ]] && rc=1
            fi
        fi
    fi
    read -r sel pas fai ski ter cer <<< "$counts"
    printf "%-22s %8s %6s %6s %7s %8s %11s %7.1f %4s  %s\n" "$name" "$sel" "$pas" "$fai" "$ski" "$ter" "$cer" \
        "$(python3 -c "print($t1 - $t0)")" "$rc" "$out$note" | tee -a "$TARGET_SUMMARY"
    [[ $rc -eq 0 ]] || TARGET_FAILED=1
}

if [[ $LIST_TARGETS -eq 1 ]]; then
    print_header "Targets"
    printf "%-22s %6s  %s\n" target tests description
    while IFS='|' read -r m about; do
        n=$(python3 -m pytest --collect-only -qq -p no:cacheprovider -m "$m" "$HEADLESS_DIR" 2>/dev/null | grep -c "::")
        printf "%-22s %6s  %s\n" "$m" "$n" "$about"
        if [[ $LIST_IDS -eq 1 ]]; then
            python3 -m pytest --collect-only -qq -p no:cacheprovider -m "$m" "$HEADLESS_DIR" 2>/dev/null | grep "::" | sed 's/^/    /'
        fi
    done < <(target_markers)
    printf "%-22s %6s  %s\n" joins "$(ls RootInteractive/joins/test_*.mjs 2>/dev/null | wc -l) files" "JavaScript tests of the joins prototypes (node --test, TAP)"
    if [[ $LIST_IDS -eq 1 ]]; then ls RootInteractive/joins/test_*.mjs 2>/dev/null | sed 's/^/    /'; fi
    echo "Run one:   ./run_tests.sh --target node_histograms      (the same tests: pytest $HEADLESS_DIR -m node_histograms)"
    echo "Test ids:  pytest --collect-only -qq -m <target> $HEADLESS_DIR"
    return 0
fi

if [[ ${#TARGETS[@]} -gt 0 ]]; then
    KNOWN_TARGETS=$(target_markers)
    bad=()                                   # every name is checked before anything runs, also with "all" (4.2c-2c v0.2)
    for t in "${TARGETS[@]}"; do
        if [[ "$t" != all && "$t" != joins ]] && ! is_target "$t"; then bad+=("$t"); fi
    done
    if [[ ${#bad[@]} -gt 0 ]]; then
        echo "unknown target(s): ${bad[*]} (see ./run_tests.sh --list-targets)"
        return 1
    fi
    if [[ " ${TARGETS[*]} " == *" all "* ]]; then
        FULL_MODE=1                          # --target all: the existing full acceptance path below
    else
        print_header "Targets: ${TARGETS[*]}"
        TARGET_SUMMARY="$LOG_BASE_DIR/targets_${TIMESTAMP}.txt"
        TARGET_FAILED=0
        {
            echo "commit $(git rev-parse --short HEAD 2>/dev/null) | workers $PARALLEL_WORKERS | $(python3 --version 2>&1) | pytest $(python3 -c 'import pytest; print(pytest.__version__)' 2>/dev/null) | node $(node --version 2>/dev/null) | bokeh $(python3 -c 'import bokeh; print(bokeh.__version__)' 2>/dev/null)"
            printf "%-22s %8s %6s %6s %7s %8s %11s %7s %4s  %s\n" target selected passed failed skipped test-err collect-err wall_s exit log
        } | tee "$TARGET_SUMMARY"
        all0=$(date +%s.%N)
        for t in "${TARGETS[@]}"; do run_target "$t"; done
        echo "total wall time: $(python3 -c "print(f'{$(date +%s.%N) - $all0:.1f}')") s" | tee -a "$TARGET_SUMMARY"
        if [[ ${#TARGETS[@]} -gt 1 && $DRY_RUN -eq 0 ]]; then                  # tests selected by several targets
            python3 - "$LOG_BASE_DIR" "$TIMESTAMP" <<'OVERPY' | tee -a "$TARGET_SUMMARY"
import collections, pathlib, sys
seen = collections.defaultdict(list)
for f in sorted(pathlib.Path(sys.argv[1]).glob(f"target_*_{sys.argv[2]}/selected_tests.txt")):
    for t in f.read_text().splitlines():
        seen[t].append(f.parent.name.split("_" + sys.argv[2])[0][len("target_"):])
over = {t: ts for t, ts in seen.items() if len(ts) > 1}
print(f"overlap between targets: {len(over)} test(s)" + "".join(f"\n  {t}: {', '.join(ts)}" for t, ts in sorted(over.items())))
OVERPY
        fi
        if [[ $DRY_RUN -eq 0 ]]; then                                        # the evidence of this invocation
            if (cd "$LOG_BASE_DIR" && zip -qr "targets_${TIMESTAMP}.zip" "targets_${TIMESTAMP}.txt" target_*_"${TIMESTAMP}" \
                    -x '*.npz' -x '*.html' -x '*_document.json') && [[ -s "$LOG_BASE_DIR/targets_${TIMESTAMP}.zip" ]]; then
                echo "Evidence: $LOG_BASE_DIR/targets_${TIMESTAMP}.zip"
            else
                print_warning "the evidence zip could not be written; the target folders remain in $LOG_BASE_DIR"
                TARGET_FAILED=1
            fi
        fi
        return $TARGET_FAILED
    fi
fi

# =============================================================================
# Phase 1: Run Tests
# =============================================================================

if [[ $MATRIX_ONLY -eq 0 ]]; then
    print_header "Phase 1: Running Tests"
    
    for test_dir in "${TEST_DIRS[@]}"; do
        if [[ ! -d "$test_dir" ]]; then
            print_warning "Directory not found: $test_dir (skipping)"
            RUN_FAILED=1
            continue
        fi
        
        # Check if there are test files
        test_count=$(ls -1 "$test_dir"/test*.py 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$test_count" -eq 0 ]]; then
            print_warning "No test files in: $test_dir (skipping)"
            RUN_FAILED=1
            continue
        fi
        
        # Create output directory: test_logs/<dirname>_<timestamp>/
        dir_name=$(basename "$test_dir")
        output_dir="$LOG_BASE_DIR/${dir_name}_${TIMESTAMP}"
        mkdir -p "$output_dir"
        
        report_file="$output_dir/report.json"
        log_file="$output_dir/test.log"
        summary_file="$output_dir/summary.txt"
        
        echo ""
        echo "${BOLD}Testing: $test_dir ($test_count test files)${RESET}"
        echo "Output: $output_dir/"
        
        # Per-folder arguments (PHASE_4 stage 4.2c-2b): the bokeh entry leaves out headless/, which has its own entry;
        # the headless entry runs the fast suite unless --full, writes its reports into this run's log folder and keeps
        # the Node model bundle in a persistent store (built once, about a minute).
        EXTRA_ARGS=()
        case "$test_dir" in
            RootInteractive/InteractiveDrawing/bokeh)
                EXTRA_ARGS=(--ignore=RootInteractive/InteractiveDrawing/bokeh/headless) ;;
            RootInteractive/InteractiveDrawing/bokeh/headless)
                # required browser and Node (a wrong jsdom version fails), the required-scenario check, an outer time
                # limit per test, and no X display for this entry (4.2c-2b v0.2)
                [[ $FULL_MODE -eq 1 ]] || EXTRA_ARGS=(-m "not slow")
                EXTRA_ARGS+=(--timeout=900)
                export RI_HEADLESS_OUT="$PROJECT_ROOT/$output_dir/headless_out"
                export RI_HEADLESS_BUNDLE="${RI_HEADLESS_BUNDLE:-$PROJECT_ROOT/RootInteractive/InteractiveDrawing/bokeh/headless/_out}"
                export RI_HEADLESS_REQUIRED=1 RI_HEADLESS_CHECK_REQUIRED=1
                HEADLESS_SAVED_DISPLAY="${DISPLAY-}"; unset DISPLAY ;;
        esac

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [DRY RUN] Would run: pytest $test_dir/ -n $PARALLEL_WORKERS ${EXTRA_ARGS[*]}"
        else
            # Run pytest with parallel workers
            if [[ "$PARALLEL_WORKERS" -gt 1 ]]; then
                python3 -m pytest "$test_dir/" \
                    -n "$PARALLEL_WORKERS" \
                    "${EXTRA_ARGS[@]}" \
                    $VERBOSE \
                    --json-report \
                    --json-report-file="$report_file" \
                    2>&1 | tee "$log_file"
            else
                # Sequential run
                python3 -m pytest "$test_dir/" \
                    "${EXTRA_ARGS[@]}" \
                    $VERBOSE \
                    --json-report \
                    --json-report-file="$report_file" \
                    2>&1 | tee "$log_file"
            fi
            exit_code=${PIPESTATUS[0]}
            if [[ $exit_code -ne 0 ]]; then RUN_FAILED=1; fi
            if [[ "$test_dir" == "RootInteractive/InteractiveDrawing/bokeh/headless" ]]; then
                # The outcome list, the timings and the required-scenario check are written and checked by the headless
                # session itself (its conftest.py): a failure there is pytest's own exit status, never ignored here.
                for f in outcomes.txt timings.txt; do
                    if [[ -f "$RI_HEADLESS_OUT/$f" ]]; then cp "$RI_HEADLESS_OUT/$f" "$output_dir/$f"; fi
                done
                HEADLESS_OUT_DIR="$RI_HEADLESS_OUT"           # for the capability matrix (4.2c-2d)
                unset RI_HEADLESS_OUT RI_HEADLESS_REQUIRED RI_HEADLESS_CHECK_REQUIRED
                if [[ -n "$HEADLESS_SAVED_DISPLAY" ]]; then export DISPLAY="$HEADLESS_SAVED_DISPLAY"; fi
            fi

            if [[ -f "$report_file" ]]; then
                REPORT_FILES+=("$report_file")
                
                # Extract counts from report
                passed=$(python3 -c "import json; r=json.load(open('$report_file')); print(r.get('summary',{}).get('passed',0))" 2>/dev/null || echo "0")
                failed=$(python3 -c "import json; r=json.load(open('$report_file')); print(r.get('summary',{}).get('failed',0))" 2>/dev/null || echo "0")
                skipped=$(python3 -c "import json; r=json.load(open('$report_file')); print(r.get('summary',{}).get('skipped',0))" 2>/dev/null || echo "0")
                errors=$(python3 -c "import json; r=json.load(open('$report_file')); print(r.get('summary',{}).get('error',0)+r.get('summary',{}).get('errors',0))" 2>/dev/null)
                errors=${errors:-0}
                TOTAL_ERRORS=$((TOTAL_ERRORS + errors))
                total=$((passed + failed + skipped + errors))
                
                TOTAL_PASSED=$((TOTAL_PASSED + passed))
                TOTAL_FAILED=$((TOTAL_FAILED + failed))
                TOTAL_SKIPPED=$((TOTAL_SKIPPED + skipped))
                
                # Write summary file
                cat > "$summary_file" << EOF
# Test Summary: $dir_name
# Date: $(date '+%Y-%m-%d %H:%M:%S')
# Directory: $test_dir

PASSED:  $passed
FAILED:  $failed
SKIPPED: $skipped
ERRORS:  $errors
PYTEST_EXIT_STATUS: $exit_code
TOTAL:   $total

# Failed tests:
EOF
                # Add failed test names to summary
                python3 -c "
import json
r = json.load(open('$report_file'))
for t in r.get('tests', []):
    if t.get('outcome') == 'failed':
        print(f\"  - {t['nodeid']}\")
" >> "$summary_file" 2>/dev/null || true
                
                echo "  Results: ${GREEN}${passed} passed${RESET}, ${RED}${failed} failed${RESET}, ${YELLOW}${skipped} skipped${RESET}"
                
                # Update 'latest' symlink
                rm -f "$LOG_BASE_DIR/${dir_name}_latest"
                ln -sf "${dir_name}_${TIMESTAMP}" "$LOG_BASE_DIR/${dir_name}_latest"
            else
                print_warning "Report not generated for $test_dir"
                RUN_FAILED=1
            fi
        fi
    done

    # Node tests of the joins prototypes (PHASE_4 stage 4.2c-2b). The files are listed explicitly: their names
    # (test_*.mjs) do not match node --test's default patterns, so a folder argument would find nothing. An empty list
    # fails the run; three of the files read ao2d_fixture.json, which is generated (joins/README.md), so it is checked
    # first and its generation command printed.
    if [[ -d RootInteractive/joins ]]; then
        joins_dir="$LOG_BASE_DIR/joins_${TIMESTAMP}"
        joins_files=(RootInteractive/joins/test_*.mjs)
        echo ""
        echo "${BOLD}Testing: RootInteractive/joins (node --test, ${#joins_files[@]} files)${RESET}"
        if [[ ! -e "${joins_files[0]}" ]]; then
            print_warning "no joins test files found (RootInteractive/joins/test_*.mjs)"
            RUN_FAILED=1
        elif [[ $DRY_RUN -eq 1 ]]; then
            echo "  [DRY RUN] Would run: node --test ${joins_files[*]}"
        elif ! command -v node > /dev/null 2>&1; then
            print_warning "node not found: joins tests not run"
            RUN_FAILED=1
        elif [[ ! -f RootInteractive/joins/ao2d_fixture.json ]]; then
            print_warning "RootInteractive/joins/ao2d_fixture.json is missing (generated, not in git); create it with:"
            echo "    (cd RootInteractive/Tools/generators && python3 generate_ao2d_fixture.py ../../joins/ao2d_fixture.json)"
            RUN_FAILED=1
        else
            mkdir -p "$joins_dir"
            node --test "${NODE_TAP[@]}" "${joins_files[@]}" 2>&1 | tee "$joins_dir/test.log"
            [[ ${PIPESTATUS[0]} -eq 0 ]] || RUN_FAILED=1
            joins_passed=$(grep -E "^# pass [0-9]+" "$joins_dir/test.log" | tail -1 | awk '{print $3}')
            joins_failed=$(grep -E "^# fail [0-9]+" "$joins_dir/test.log" | tail -1 | awk '{print $3}')
            if [[ -z "$joins_passed" || -z "$joins_failed" ]] || (( joins_passed + joins_failed <= 0 )); then
                print_warning "joins: no test counted, or the count could not be read from the log"; RUN_FAILED=1
            fi
            TOTAL_PASSED=$((TOTAL_PASSED + ${joins_passed:-0}))
            TOTAL_FAILED=$((TOTAL_FAILED + ${joins_failed:-0}))
            echo "  Results: ${joins_passed:-0} passed, ${joins_failed:-0} failed"
        fi
    fi
    
    # Summary
    print_header "Test Summary"
    echo "  ${BOLD}PASSED:${RESET}  ${GREEN}${TOTAL_PASSED}${RESET}"
    echo "  ${BOLD}FAILED:${RESET}  ${RED}${TOTAL_FAILED}${RESET}"
    echo "  ${BOLD}SKIPPED:${RESET} ${YELLOW}${TOTAL_SKIPPED}${RESET}"
    
    if [[ $RUN_FAILED -ne 0 || $TOTAL_FAILED -gt 0 || $TOTAL_ERRORS -gt 0 ]]; then
        print_error "Some tests failed!"
    else
        print_success "All tests passed!"
    fi
fi

# Focused example is separate from pytest collection and is included in the packet.
if [[ $MATRIX_ONLY -eq 0 && $DRY_RUN -eq 0 ]]; then
    python3 RootInteractive/MLpipeline/examples/phase_2_1_reducible_example.py > "$LOG_BASE_DIR/MLpipeline_example_${TIMESTAMP}.log" 2>&1
    EXAMPLE_STATUS=$?
    echo "MLpipeline example exit status: $EXAMPLE_STATUS"
    if [[ $EXAMPLE_STATUS -ne 0 ]]; then RUN_FAILED=1; fi
fi

# =============================================================================
# Phase 2: Generate CAPABILITY_MATRIX
# =============================================================================

if [[ $QUICK_MODE -eq 0 ]]; then
    print_header "Phase 2: Generating CAPABILITY_MATRIX"
    
    # Find report files if we didn't just run tests
    if [[ ${#REPORT_FILES[@]} -eq 0 ]]; then
        # Use latest reports (follow symlinks)
        for test_dir in "${TEST_DIRS[@]}"; do
            dir_name=$(basename "$test_dir")
            latest_link="$LOG_BASE_DIR/${dir_name}_latest"
            if [[ -L "$latest_link" ]]; then
                latest_report="$LOG_BASE_DIR/$(readlink "$latest_link")/report.json"
                if [[ -f "$latest_report" ]]; then
                    REPORT_FILES+=("$latest_report")
                fi
            fi
        done
    fi
    
    # PHASE_4 4.2c-2d (the Architect's decision, option C, approved by three reviewers): only a COMPLETE full-suite headless
    # session (session.json "complete") publishes the canonical matrix; any other run publishes CAP_MATRIX_fast (the
    # fast suite) or CAP_MATRIX_partial (--matrix without a complete session) in test_logs/, each with the evidence it
    # has. publish_matrix keeps the previous pair if generation fails, and the run then fails.
    HL_DIR="${HEADLESS_OUT_DIR:-}"
    if [[ -z "$HL_DIR" && -L "$LOG_BASE_DIR/headless_latest" ]]; then
        HL_DIR="$LOG_BASE_DIR/$(readlink "$LOG_BASE_DIR/headless_latest")/headless_out"
    fi
    HL_ARGS=()
    if [[ -n "$HL_DIR" && -f "$HL_DIR/session.json" ]] && find "$HL_DIR" -name 'report*.json' -print -quit | grep -q .; then
        HL_ARGS=(--from-headless "$HL_DIR")
    fi
    if [[ ${#HL_ARGS[@]} -gt 0 ]] && python3 -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("complete") else 1)' "$HL_DIR/session.json"; then
        MATRIX_DEST="$MATRIX_OUTPUT"
    elif [[ $MATRIX_ONLY -eq 1 ]]; then
        MATRIX_DEST="$LOG_BASE_DIR/CAP_MATRIX_partial.md"
        echo "Note: no complete headless session; the canonical matrix is not updated. Matrix: $MATRIX_DEST"
    else
        MATRIX_DEST="$LOG_BASE_DIR/CAP_MATRIX_fast.md"
        echo "Note: not a complete full-suite run; the canonical matrix is not updated. Matrix of this run: $MATRIX_DEST"
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] Would publish: $MATRIX_DEST (generator ${HL_ARGS[*]})"
        echo "  Reports: ${REPORT_FILES[*]}"
    else
        if [[ ${#REPORT_FILES[@]} -gt 0 ]]; then
            echo "Using reports:"
            for rf in "${REPORT_FILES[@]}"; do
                echo "  - $rf"
            done
            publish_matrix "$MATRIX_DEST" --from-reports "${REPORT_FILES[@]}" "${HL_ARGS[@]}"
        else
            echo "No reports found, running pytest internally..."
            publish_matrix "$MATRIX_DEST" "${HL_ARGS[@]}"
        fi
        if [[ $? -eq 0 ]]; then
            print_success "Generated: $MATRIX_DEST"
            print_success "Generated: ${MATRIX_DEST%.md}.json"
        else
            RUN_FAILED=1
            print_error "Capability matrix generation failed; the previous $MATRIX_DEST (if any) is unchanged"
        fi
    fi
fi

# =============================================================================
# Final Status
# =============================================================================

print_header "Done"

echo "Logs: $LOG_BASE_DIR/"
for test_dir in "${TEST_DIRS[@]}"; do
    dir_name=$(basename "$test_dir")
    latest_link="$LOG_BASE_DIR/${dir_name}_latest"
    if [[ -L "$latest_link" ]]; then
        echo "  ${dir_name}: $(readlink "$latest_link")/"
    fi
done

if [[ -s "${MATRIX_DEST:-$MATRIX_OUTPUT}" && -s "$(echo "${MATRIX_DEST:-$MATRIX_OUTPUT}" | sed 's/\.md$/.json/')" ]]; then
    echo "Matrix: ${MATRIX_DEST:-$MATRIX_OUTPUT}"
fi

echo ""
echo "Compare runs: diff test_logs/bokeh_<date1>/summary.txt test_logs/bokeh_<date2>/summary.txt"
echo ""

# P0-3 FIX: Only check TOTAL_FAILED if we actually ran tests
if [[ $MATRIX_ONLY -eq 0 ]] && [[ $TOTAL_FAILED -gt 0 ]]; then
    print_error "Some tests failed — see logs above"
fi

# =============================================================================
# Phase Diff — save for reviewers
# =============================================================================

PHASE_TAG="PHASE_BEGIN_RootInteractive"
DIFF_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if git rev-parse "$PHASE_TAG" >/dev/null 2>&1; then
    print_header "Phase Changes (since $PHASE_TAG)"
    DIFF_STAT=$(git diff --stat "$PHASE_TAG"..HEAD 2>/dev/null)
    COMMIT_COUNT=$(git rev-list --count "$PHASE_TAG"..HEAD 2>/dev/null)

    if [[ -n "$DIFF_STAT" ]]; then
        echo "Commits since phase start: $COMMIT_COUNT"
        echo ""
        echo "$DIFF_STAT"

        # Save diffs for reviewers
        DIFF_PHASE="$LOG_BASE_DIR/diff_to_phase_${DIFF_TIMESTAMP}.txt"
        DIFF_LAST="$LOG_BASE_DIR/diff_last_commit_${DIFF_TIMESTAMP}.txt"

        git diff "$PHASE_TAG"..HEAD -- . > "$DIFF_PHASE"
        git diff HEAD~1..HEAD -- . > "$DIFF_LAST"

        echo ""
        echo "Full diff: git diff $PHASE_TAG..HEAD"
        echo "Log:       git log --oneline $PHASE_TAG..HEAD"
    else
        echo "No changes since $PHASE_TAG"
    fi
else
    RUN_FAILED=1
    print_error "Tag $PHASE_TAG missing — phase review blocked (P0). Evidence is still packaged."
    echo "  Missing reference tag recorded; no tag is changed by this runner."
    echo ""
    echo "  Restore the approved phase reference before claiming a green canonical gate."
    echo "  This runner does not change baseline tags or approve a phase."
fi

# =============================================================================
# File listing (for easy upload to reviewers)
# =============================================================================

print_header "Files"

for test_dir in "${TEST_DIRS[@]}"; do
    dir_name=$(basename "$test_dir")
    latest_link="$LOG_BASE_DIR/${dir_name}_latest"
    if [[ -L "$latest_link" ]]; then
        latest_dir="$LOG_BASE_DIR/$(readlink "$latest_link")"
        [[ -f "$latest_dir/test.log" ]] && echo "  Log:       $PROJECT_ROOT/$latest_dir/test.log"
        [[ -f "$latest_dir/summary.txt" ]] && echo "  Summary:   $PROJECT_ROOT/$latest_dir/summary.txt"
        [[ -f "$latest_dir/report.json" ]] && echo "  JSON:      $PROJECT_ROOT/$latest_dir/report.json"
    fi
done

[[ -f "${MATRIX_DEST:-$MATRIX_OUTPUT}" ]] && echo "  Matrix:    $PROJECT_ROOT/${MATRIX_DEST:-$MATRIX_OUTPUT}"
[[ -f "$DIFF_PHASE" ]] && echo "  Diff tag:  $PROJECT_ROOT/$DIFF_PHASE"
[[ -f "$DIFF_LAST" ]] && echo "  Diff HEAD: $PROJECT_ROOT/$DIFF_LAST"

# =============================================================================
# Package reviewer.zip — all review artifacts in one file
# =============================================================================
# Uses a staging directory with prefixed filenames to avoid collisions
# (e.g. bokeh_test.log vs Tools_test.log)

REVIEWER_ZIP="$LOG_BASE_DIR/reviewer_${TIMESTAMP}.zip"
REVIEWER_STAGE="$LOG_BASE_DIR/.reviewer_stage_$$"
rm -rf "$REVIEWER_STAGE"
mkdir -p "$REVIEWER_STAGE"

REVIEWER_COUNT=0

# Collect only this run, never substitute stale latest logs for a failed suite.
for test_dir in "${TEST_DIRS[@]}"; do
    dir_name=$(basename "$test_dir")
    latest_dir="$LOG_BASE_DIR/${dir_name}_${TIMESTAMP}"
    if [[ $MATRIX_ONLY -eq 1 && -L "$LOG_BASE_DIR/${dir_name}_latest" ]]; then
        latest_dir="$LOG_BASE_DIR/$(readlink "$LOG_BASE_DIR/${dir_name}_latest")"
    fi
    for f in test.log summary.txt report.json outcomes.txt timings.txt; do     # outcomes/timings: headless entry
        if [[ -f "$latest_dir/$f" ]]; then
            cp "$latest_dir/$f" "$REVIEWER_STAGE/${dir_name}_${f}"
            REVIEWER_COUNT=$((REVIEWER_COUNT + 1))
        fi
    done
done

# Collect the joins Node tests' log (PHASE_4 stage 4.2c-2b)
if [[ -f "$LOG_BASE_DIR/joins_${TIMESTAMP}/test.log" ]]; then
    cp "$LOG_BASE_DIR/joins_${TIMESTAMP}/test.log" "$REVIEWER_STAGE/joins_test.log"
    REVIEWER_COUNT=$((REVIEWER_COUNT + 1))
fi

# Collect matrix (unique name, no prefix needed)
if [[ -s "${MATRIX_DEST:-$MATRIX_OUTPUT}" ]]; then                                  # the matrix of THIS run (4.2c-2d)
    cp "${MATRIX_DEST:-$MATRIX_OUTPUT}" "$REVIEWER_STAGE/CAPABILITY_MATRIX.md"
    REVIEWER_COUNT=$((REVIEWER_COUNT + 1))
fi

# Collect diffs
if [[ -n "${DIFF_PHASE:-}" ]] && [[ -f "$DIFF_PHASE" ]]; then
    cp "$DIFF_PHASE" "$REVIEWER_STAGE/$(basename "$DIFF_PHASE")"
    REVIEWER_COUNT=$((REVIEWER_COUNT + 1))
fi
if [[ -n "${DIFF_LAST:-}" ]] && [[ -f "$DIFF_LAST" ]]; then
    cp "$DIFF_LAST" "$REVIEWER_STAGE/$(basename "$DIFF_LAST")"
    REVIEWER_COUNT=$((REVIEWER_COUNT + 1))
fi

# Add git log
if git rev-parse "$PHASE_TAG" >/dev/null 2>&1; then
    git log --oneline "$PHASE_TAG"..HEAD > "$REVIEWER_STAGE/git_log.txt" 2>/dev/null
    if [[ -s "$REVIEWER_STAGE/git_log.txt" ]]; then
        REVIEWER_COUNT=$((REVIEWER_COUNT + 1))
    else
        rm -f "$REVIEWER_STAGE/git_log.txt"
    fi
fi

if [[ $DRY_RUN -eq 0 ]]; then
    export RI_REVIEW_STAGE="$REVIEWER_STAGE" RI_REVIEW_ROOT="$PROJECT_ROOT" RI_REVIEW_TIMESTAMP="$TIMESTAMP" RI_REVIEW_FAILURE="$RUN_FAILED" RI_REVIEW_EXAMPLE="${EXAMPLE_STATUS:-not-run}" RI_REVIEW_MATRIX_ONLY="$MATRIX_ONLY"
    python3 - <<'REVIEWPY'
import hashlib, importlib.metadata, json, os, pathlib, platform, subprocess
root=pathlib.Path(os.environ['RI_REVIEW_ROOT']); stage=root/os.environ['RI_REVIEW_STAGE']
def git(*args):
    r=subprocess.run(['git','-C',str(root),*args],capture_output=True)
    return r.stdout if r.returncode==0 else b'GIT COMMAND FAILED: '+r.stderr
for name,args in [('diff_worktree.patch',['diff','--binary','--','.']),('diff_staged.patch',['diff','--cached','--binary','--','.']),('diff_HEAD_to_worktree.patch',['diff','HEAD','--binary','--','.']),('git_status.txt',['status','--short','--','.']),('git_identity.txt',['log','-1','--format=fuller'])]:
    (stage/name).write_bytes(git(*args))
source_paths={pathlib.Path('run_tests.sh')}
for relative in git('diff','HEAD','--name-only','--','.').decode(errors='replace').splitlines(): source_paths.add(pathlib.Path(relative))
for relative in git('ls-files','--others','--exclude-standard','--','RootInteractive/MLpipeline').decode(errors='replace').splitlines(): source_paths.add(pathlib.Path(relative))
source_paths.update(p.relative_to(root) for p in (root/'RootInteractive/MLpipeline').rglob('*.py'))
manifest=[]
for relative in sorted(source_paths):
    p=root/relative
    if not p.is_file() or p.suffix not in ('.py','.sh','.ts','.mjs','.md'): continue
    if any(x in relative.parts for x in ('__pycache__','node_modules','test_logs')): continue
    data=p.read_bytes(); destination=stage/'source'/relative
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
    manifest.append(dict(path=str(relative),bytes=len(data),md5=hashlib.md5(data).hexdigest(),sha256=hashlib.sha256(data).hexdigest()))
(stage/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
(stage/'run_status.json').write_text(json.dumps(dict(failed_or_blocked=int(os.environ['RI_REVIEW_FAILURE']),example_status=os.environ['RI_REVIEW_EXAMPLE'],matrix_only=int(os.environ['RI_REVIEW_MATRIX_ONLY'])),indent=2))
before=root/'test_logs'/('source_before_'+os.environ['RI_REVIEW_TIMESTAMP']+'.json')
changed=[]
if before.is_file():
    original=json.loads(before.read_text())
    (stage/'source_before.json').write_bytes(before.read_bytes())
    current={entry['path']:entry['sha256'] for entry in manifest}
    changed=[name for name,sha in original.items() if current.get(name)!=sha]
(stage/'source_consistency.json').write_text(json.dumps(dict(changed_during_run=changed),indent=2))
# New files have complete source above and an explicit /dev/null patch here.
patch=b''
for relative in git('ls-files','--others','--exclude-standard','--','RootInteractive/MLpipeline').decode(errors='replace').splitlines():
    p=root/relative
    if p.is_file() and p.suffix=='.py':
        r=subprocess.run(['git','diff','--no-index','--','/dev/null',str(p)],capture_output=True)
        patch+=r.stdout
(stage/'diff_untracked.patch').write_bytes(patch)
packages={}
for name in ['numpy','pandas','scipy','scikit-learn','joblib','pytest','xgboost','onnxruntime','keras']:
    try:packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:packages[name]='NOT INSTALLED'
(stage/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),packages=packages),indent=2))
example=root/'test_logs'/('MLpipeline_example_'+os.environ['RI_REVIEW_TIMESTAMP']+'.log')
if example.is_file():(stage/'MLpipeline_example.log').write_bytes(example.read_bytes())
docs=os.environ.get('RootInteractive_DOCS')
missing=[]
if docs:
    names=['PHASE_2_1_MIForestErrPDFUpdate_Proposal_v0_2_1.md','PHASE_2_1_StageB_Reducible_v0_1_Consolidated_Official_Review_Summary_GPT10_20261030.md','PHASE_2_1_StageC_Noise_CRR_v0_2_GPT20_20261031.md','PHASE_2_1_StageC_Noise_v0_1_Consolidated_Official_Review_Summary_GPT10_20261031.md']
    for name in names:
        p=pathlib.Path(docs)/name
        if p.is_file():(stage/name).write_bytes(p.read_bytes())
        else:missing.append(name)
else:missing.append('RootInteractive_DOCS environment variable unavailable; formal docs not collected')
(stage/'review_packet_notes.txt').write_text('Source is the actual worktree, including staged/unstaged/new ML files.\nMissing formal references:\n'+'\n'.join(missing)+'\nFull-run failures are not waived; see each suite log.\n')
if changed:
    raise RuntimeError('Source changed during tests; review source_consistency.json')
REVIEWPY
    if [[ $? -ne 0 ]]; then RUN_FAILED=1; fi
fi

if [[ $REVIEWER_COUNT -gt 0 ]]; then
    # Create zip from staging directory (flat structure, no path issues)
    (cd "$REVIEWER_STAGE" && zip -qr "$PROJECT_ROOT/$REVIEWER_ZIP" .)
    if [[ $? -ne 0 ]]; then RUN_FAILED=1; fi
    if [[ -f "$REVIEWER_ZIP" ]]; then
        rm -f "$LOG_BASE_DIR/reviewer_latest.zip"
        ln -sf "reviewer_${TIMESTAMP}.zip" "$LOG_BASE_DIR/reviewer_latest.zip"
        print_header "Reviewer Package"
        echo "  ${BOLD}$PROJECT_ROOT/$REVIEWER_ZIP${RESET}"
        echo "  ($REVIEWER_COUNT files)"
        echo ""
        echo "  Upload this single file to a reviewer session."
        if [[ -n "${RootInteractive_DOCS:-}" ]]; then
            mkdir -p "$RootInteractive_DOCS/files_RI"
            cp -p "$REVIEWER_ZIP" "$RootInteractive_DOCS/files_RI/"
            if [[ $? -ne 0 ]]; then RUN_FAILED=1; else echo "Exchange packet: $RootInteractive_DOCS/files_RI/$(basename "$REVIEWER_ZIP")"; fi
        fi
    else
        print_error "Failed to create reviewer.zip"
        RUN_FAILED=1
    fi
else
    print_warning "No files to package for reviewer.zip"
fi

# Cleanup staging
rm -rf "$REVIEWER_STAGE"

# =============================================================================
# Return a truthful process status after preserving the review packet.
if [[ $RUN_FAILED -ne 0 || $TOTAL_FAILED -gt 0 || $TOTAL_ERRORS -gt 0 ]]; then
    return 1
fi
return 0

}
ri_main "$@"
