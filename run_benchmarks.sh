#!/bin/bash
# =============================================================================
# RootInteractive Benchmark Runner
# =============================================================================
# Usage:
#   ./run_benchmarks.sh              # Run all benchmarks
#   ./run_benchmarks.sh wasm         # WASM element-wise only
#   ./run_benchmarks.sh conv         # Convolution only
#   ./run_benchmarks.sh onnx         # ONNX only (future)
#   ./run_benchmarks.sh all          # All benchmarks
#
# Output: bench_logs/<category>_<timestamp>.log
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"  # script lives at project root (alongside run_tests.sh)
BOKEH_DIR="$PROJECT_ROOT/RootInteractive/InteractiveDrawing/bokeh"
WASM_BUILD="$PROJECT_ROOT/RootInteractive/wasm/build"
LOG_DIR="$PROJECT_ROOT/bench_logs"

# Verify project structure
if [[ ! -d "$BOKEH_DIR" ]]; then
    echo "ERROR: Cannot find $BOKEH_DIR"
    echo "Run this script from the RootInteractive project root."
    exit 1
fi
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
RESET="\033[0m"

# --- Helpers -----------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}  $1${RESET}"
    echo -e "${BOLD}══════════════════════════════════════════════════════════════${RESET}"
}

print_ok()    { echo -e "  ${GREEN}✅${RESET} $1"; }
print_warn()  { echo -e "  ${YELLOW}⚠️${RESET}  $1"; }
print_fail()  { echo -e "  ${RED}❌${RESET} $1"; }

run_benchmark() {
    local name="$1"
    local cmd="$2"
    local logfile="$LOG_DIR/${name}_${TIMESTAMP}.log"
    local latest="$LOG_DIR/${name}_latest.log"

    echo ""
    echo -e "  Running: ${BOLD}${name}${RESET}"
    echo "  Command: $cmd"
    echo "  Log:     $logfile"
    echo ""

    if eval "$cmd" 2>&1 | tee "$logfile"; then
        print_ok "$name completed (exit 0)"
        # Update latest symlink
        ln -sf "$(basename "$logfile")" "$latest"
        BENCH_PASSED=$((BENCH_PASSED + 1))
    else
        print_fail "$name failed (exit $?)"
        BENCH_FAILED=$((BENCH_FAILED + 1))
    fi
    echo ""
}

# --- Prerequisites -----------------------------------------------------------

check_node() {
    if ! command -v node &>/dev/null; then
        print_fail "Node.js not found. Required for WASM and ONNX benchmarks."
        exit 1
    fi
    NODE_VERSION=$(node --version)
    echo "  Node.js: $NODE_VERSION"
}

check_wasm_binaries() {
    local missing=0
    if [[ ! -f "$WASM_BUILD/functions.wasm" ]]; then
        print_warn "functions.wasm not found at $WASM_BUILD/functions.wasm"
        print_warn "Build with: cd RootInteractive/wasm && make"
        missing=1
    fi
    if [[ ! -f "$WASM_BUILD/wasm_conv.wasm" ]]; then
        print_warn "wasm_conv.wasm not found at $WASM_BUILD/wasm_conv.wasm"
        missing=1
    fi
    return $missing
}

# --- Benchmark Categories ----------------------------------------------------

bench_wasm() {
    print_header "WASM Element-Wise Benchmarks (TC-BENCH-01..04)"

    if ! check_wasm_binaries; then
        print_fail "Skipping WASM benchmarks — missing .wasm files"
        BENCH_SKIPPED=$((BENCH_SKIPPED + 1))
        return
    fi

    run_benchmark "bench_wasm_vs_js" \
        "cd '$BOKEH_DIR' && node bench_wasm_vs_js.mjs '$WASM_BUILD/functions.wasm'"
}

bench_conv() {
    print_header "Convolution Benchmarks (1D/2D/3D)"

    if ! check_wasm_binaries; then
        print_fail "Skipping convolution benchmarks — missing .wasm files"
        BENCH_SKIPPED=$((BENCH_SKIPPED + 1))
        return
    fi

    run_benchmark "bench_convolution" \
        "cd '$BOKEH_DIR' && node bench_convolution.mjs '$WASM_BUILD/wasm_conv.wasm'"
}

bench_onnx() {
    print_header "ONNX Benchmarks"

    # Future: ONNX Runtime inference benchmarks
    # Expected files:
    #   bokeh/bench_onnx.mjs          — ONNX model inference timing
    #   bokeh/bench_onnx_models/      — Test models (.onnx)
    #
    # Benchmark scope:
    #   - Model load time (cold + warm)
    #   - Inference latency at N = 100, 1K, 10K, 100K
    #   - Memory footprint per model
    #   - Comparison: ONNX Runtime vs native JS computation

    if [[ -f "$BOKEH_DIR/bench_onnx.mjs" ]]; then
        run_benchmark "bench_onnx" \
            "cd '$BOKEH_DIR' && node bench_onnx.mjs"
    else
        print_warn "bench_onnx.mjs not found — ONNX benchmarks not yet implemented"
        BENCH_SKIPPED=$((BENCH_SKIPPED + 1))
    fi
}

# --- Main --------------------------------------------------------------------

BENCH_PASSED=0
BENCH_FAILED=0
BENCH_SKIPPED=0

# Parse arguments
CATEGORIES="${1:-all}"

print_header "RootInteractive Benchmark Runner"
echo "  Date:       $(date)"
echo "  Categories: $CATEGORIES"
echo "  Log dir:    $LOG_DIR"
check_node

mkdir -p "$LOG_DIR"

case "$CATEGORIES" in
    wasm)
        bench_wasm
        ;;
    conv|convolution)
        bench_conv
        ;;
    onnx)
        bench_onnx
        ;;
    all)
        bench_wasm
        bench_conv
        bench_onnx
        ;;
    *)
        echo "Unknown category: $CATEGORIES"
        echo "Usage: $0 [wasm|conv|onnx|all]"
        exit 1
        ;;
esac

# --- Summary -----------------------------------------------------------------

print_header "Benchmark Summary"
echo "  Passed:  $BENCH_PASSED"
echo "  Failed:  $BENCH_FAILED"
echo "  Skipped: $BENCH_SKIPPED"
echo ""
echo "  Logs: $LOG_DIR/"
ls -1t "$LOG_DIR"/*_${TIMESTAMP}.log 2>/dev/null | while read f; do
    echo "    $(basename "$f")"
done
echo ""
echo "  Latest symlinks:"
ls -1 "$LOG_DIR"/*_latest.log 2>/dev/null | while read f; do
    target=$(readlink "$f" 2>/dev/null || echo "?")
    echo "    $(basename "$f") -> $target"
done
echo ""

# Compare with previous runs:
echo "  Compare runs:"
echo "    diff bench_logs/bench_wasm_vs_js_latest.log bench_logs/bench_wasm_vs_js_<old>.log"
echo ""

if [[ $BENCH_FAILED -gt 0 ]]; then
    print_fail "Some benchmarks failed"
    exit 1
fi
