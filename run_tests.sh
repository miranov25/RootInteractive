#!/bin/bash
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

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Test directories (without glob - pytest discovers test_*.py)
TEST_DIRS=(
    "RootInteractive/InteractiveDrawing/bokeh"
    "RootInteractive/Tools"
    "RootInteractive/onnx"
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

Test Directories:
  - RootInteractive/InteractiveDrawing/bokeh/
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
    exit 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

VERBOSE=""
QUICK_MODE=0
MATRIX_ONLY=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_help ;;
        -v|--verbose) VERBOSE="-v" ;;
        -n)
            shift
            PARALLEL_WORKERS="$1"
            ;;
        --quick) QUICK_MODE=1 ;;
        --matrix) MATRIX_ONLY=1 ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown option: $1"; show_help ;;
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

print_header "RootInteractive Test Runner"
echo "Project root: $PROJECT_ROOT"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Parallel workers: $PARALLEL_WORKERS"
echo "Log directory: $LOG_BASE_DIR/"

# Create base log directory
mkdir -p "$LOG_BASE_DIR"

REPORT_FILES=()

# =============================================================================
# Phase 1: Run Tests
# =============================================================================

if [[ $MATRIX_ONLY -eq 0 ]]; then
    print_header "Phase 1: Running Tests"
    
    for test_dir in "${TEST_DIRS[@]}"; do
        if [[ ! -d "$test_dir" ]]; then
            print_warning "Directory not found: $test_dir (skipping)"
            continue
        fi
        
        # Check if there are test files
        test_count=$(ls -1 "$test_dir"/test*.py 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$test_count" -eq 0 ]]; then
            print_warning "No test files in: $test_dir (skipping)"
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
        
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [DRY RUN] Would run: pytest $test_dir/ -n $PARALLEL_WORKERS"
        else
            set +e
            # Run pytest with parallel workers
            if [[ "$PARALLEL_WORKERS" -gt 1 ]]; then
                pytest "$test_dir/" \
                    -n "$PARALLEL_WORKERS" \
                    $VERBOSE \
                    --json-report \
                    --json-report-file="$report_file" \
                    2>&1 | tee "$log_file"
            else
                # Sequential run
                pytest "$test_dir/" \
                    $VERBOSE \
                    --json-report \
                    --json-report-file="$report_file" \
                    2>&1 | tee "$log_file"
            fi
            exit_code=$?
            set -e
            
            if [[ -f "$report_file" ]]; then
                REPORT_FILES+=("$report_file")
                
                # Extract counts from report
                passed=$(python3 -c "import json; r=json.load(open('$report_file')); print(r.get('summary',{}).get('passed',0))" 2>/dev/null || echo "0")
                failed=$(python3 -c "import json; r=json.load(open('$report_file')); print(r.get('summary',{}).get('failed',0))" 2>/dev/null || echo "0")
                skipped=$(python3 -c "import json; r=json.load(open('$report_file')); print(r.get('summary',{}).get('skipped',0))" 2>/dev/null || echo "0")
                total=$((passed + failed + skipped))
                
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
            fi
        fi
    done
    
    # Summary
    print_header "Test Summary"
    echo "  ${BOLD}PASSED:${RESET}  ${GREEN}${TOTAL_PASSED}${RESET}"
    echo "  ${BOLD}FAILED:${RESET}  ${RED}${TOTAL_FAILED}${RESET}"
    echo "  ${BOLD}SKIPPED:${RESET} ${YELLOW}${TOTAL_SKIPPED}${RESET}"
    
    if [[ $TOTAL_FAILED -gt 0 ]]; then
        print_error "Some tests failed!"
    else
        print_success "All tests passed!"
    fi
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
    
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] Would run: python3 scripts/generate_capability_matrix.py"
        echo "  Reports: ${REPORT_FILES[*]}"
    else
        if [[ ${#REPORT_FILES[@]} -gt 0 ]]; then
            echo "Using reports:"
            for rf in "${REPORT_FILES[@]}"; do
                echo "  - $rf"
            done
            python3 scripts/generate_capability_matrix.py \
                --from-reports "${REPORT_FILES[@]}" \
                --output "$MATRIX_OUTPUT"
        else
            echo "No reports found, running pytest internally..."
            python3 scripts/generate_capability_matrix.py \
                --output "$MATRIX_OUTPUT"
        fi
        
        if [[ -f "$MATRIX_OUTPUT" ]]; then
            print_success "Generated: $MATRIX_OUTPUT"
            print_success "Generated: ${MATRIX_OUTPUT%.md}.json"
        else
            print_error "Failed to generate CAPABILITY_MATRIX"
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

if [[ -f "$MATRIX_OUTPUT" ]]; then
    echo "Matrix: $MATRIX_OUTPUT"
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

        git diff "$PHASE_TAG"..HEAD > "$DIFF_PHASE"
        git diff HEAD~1..HEAD > "$DIFF_LAST"

        echo ""
        echo "Full diff: git diff $PHASE_TAG..HEAD"
        echo "Log:       git log --oneline $PHASE_TAG..HEAD"
    else
        echo "No changes since $PHASE_TAG"
    fi
else
    print_error "Tag $PHASE_TAG not found — PHASE REVIEW NOT POSSIBLE (P0 violation per Organization-structure-v1_16.md)"
    echo "  Create with: git tag -f $PHASE_TAG && git push origin $PHASE_TAG -f"
    echo ""
    echo "  ⚠️  Without this tag, reviewers cannot verify git diff to phase start."
    echo "  ⚠️  Phase approval without diff review is a P0 process violation."
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

[[ -f "$MATRIX_OUTPUT" ]] && echo "  Matrix:    $PROJECT_ROOT/$MATRIX_OUTPUT"
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

# Collect test logs and summaries (prefix with dir name to avoid collisions)
for test_dir in "${TEST_DIRS[@]}"; do
    dir_name=$(basename "$test_dir")
    latest_link="$LOG_BASE_DIR/${dir_name}_latest"
    if [[ -L "$latest_link" ]]; then
        latest_dir="$LOG_BASE_DIR/$(readlink "$latest_link")"
        for f in test.log summary.txt report.json; do
            if [[ -f "$latest_dir/$f" ]]; then
                cp "$latest_dir/$f" "$REVIEWER_STAGE/${dir_name}_${f}"
                REVIEWER_COUNT=$((REVIEWER_COUNT + 1))
            fi
        done
    fi
done

# Collect matrix (unique name, no prefix needed)
if [[ -f "$MATRIX_OUTPUT" ]]; then
    cp "$MATRIX_OUTPUT" "$REVIEWER_STAGE/CAPABILITY_MATRIX.md"
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

if [[ $REVIEWER_COUNT -gt 0 ]]; then
    # Create zip from staging directory (flat structure, no path issues)
    (cd "$REVIEWER_STAGE" && zip -q "$PROJECT_ROOT/$REVIEWER_ZIP" *)
    if [[ -f "$REVIEWER_ZIP" ]]; then
        rm -f "$LOG_BASE_DIR/reviewer_latest.zip"
        ln -sf "reviewer_${TIMESTAMP}.zip" "$LOG_BASE_DIR/reviewer_latest.zip"
        print_header "Reviewer Package"
        echo "  ${BOLD}$PROJECT_ROOT/$REVIEWER_ZIP${RESET}"
        echo "  ($REVIEWER_COUNT files)"
        echo ""
        echo "  Upload this single file to a reviewer session."
    else
        print_error "Failed to create reviewer.zip"
    fi
else
    print_warning "No files to package for reviewer.zip"
fi

# Cleanup staging
rm -rf "$REVIEWER_STAGE"

# =============================================================================
# Phase Closure Helper
# =============================================================================
# Per Organization-structure-v1_16.md §PHASE_BEGIN Tag Convention:
# After phase approval, PHASE_BEGIN_<project> must be moved to HEAD
# to mark the start of the next phase.

if [[ $MATRIX_ONLY -eq 0 ]] && [[ $TOTAL_FAILED -eq 0 ]] && [[ $TOTAL_PASSED -gt 0 ]]; then
    if git rev-parse "$PHASE_TAG" >/dev/null 2>&1; then
        TAG_COMMIT=$(git rev-parse "$PHASE_TAG" 2>/dev/null)
        HEAD_COMMIT=$(git rev-parse HEAD 2>/dev/null)
        if [[ "$TAG_COMMIT" != "$HEAD_COMMIT" ]]; then
            print_header "Phase Closure"
            echo "  All tests passed. If this phase is approved, move the tag:"
            echo ""
            echo "    git tag -f $PHASE_TAG && git push origin $PHASE_TAG -f"
            echo ""
            echo "  This marks HEAD as the start of the next phase."
        fi
    fi
fi

echo ""

# Exit with failure if tests failed
if [[ $MATRIX_ONLY -eq 0 ]] && [[ $TOTAL_FAILED -gt 0 ]]; then
    exit 1
fi
exit 0
