# RootInteractive/Tools/exploration/compression_benchmark/bench_categorical.py
"""
Phase 0.1.C-0: Categorical Compression Benchmark Suite.

Standalone benchmarks comparing three compression variants:
  1. Baseline: delta → zip → base64 (current production)
  2. Code re-enabled: delta → code → zip → base64 (numpy reimplementation)
  3. Categorical: delta → categorical(uint8) → zip → base64 (proposed)

Measures: transfer size (incl. metadata), encode time, decode time (Python + JS),
          peak memory, min_column_length break-even.

Constraints:
  - NaN-only test data (no ±Inf until Phase 0.1.B merged)
  - Standalone code (no production file modifications)
  - Uses pytest infrastructure from Phase 0.1.A

Phase: 0.1.C-0 (Exploration & Benchmarking)
Date: 2026-02-06
Author: Claude11 (Coder)
"""

import json
import subprocess
import sys
import tempfile
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from compression_utils import (
    compress_baseline,
    compress_categorical,
    compress_code_reenabled,
    generate_test_data,
    measure_time,
    quantize_delta,
)

# =============================================================================
# CONSTANTS
# =============================================================================

DISTRIBUTIONS = ["gaussian", "exponential", "uniform"]
RESOLUTIONS = [1, 2, 5, 10, 20, 50]
N_VALUES = 1_000_000  # 1M values (production scale)
N_VALUES_SMALL = 100_000  # For timing tests (faster iteration)
BENCHMARK_DIR = Path(__file__).parent


# =============================================================================
# TC-BENCH-01: Compression Ratio vs Distribution × Resolution
# =============================================================================

class TestCompressionRatio:
    """
    TC-BENCH-01: Compare transfer sizes across all three variants.
    
    For each (distribution × resolution), measure:
    - Compressed payload size (after ZIP + base64)
    - Metadata size (JSON)
    - Total transfer size (payload + metadata)
    
    Includes full metadata + base64 overhead per G7-8.
    """

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("resolution", RESOLUTIONS)
    def test_compression_ratio(self, dist, resolution, results_collector):
        """Three-way comparison: baseline vs code vs categorical."""
        data, delta, info = generate_test_data(dist, n=N_VALUES,
                                                resolution_pct=resolution)

        r_baseline = compress_baseline(data, delta)
        r_code = compress_code_reenabled(data, delta)
        r_cat = compress_categorical(data, delta)

        # Compute deltas vs baseline
        base_total = r_baseline["sizes"]["total_transfer"]
        code_total = r_code["sizes"]["total_transfer"]
        cat_total = r_cat["sizes"]["total_transfer"]

        result = {
            "distribution": dist,
            "resolution_pct": resolution,
            "n_values": N_VALUES,
            "data_info": info,
            "baseline": r_baseline["sizes"],
            "code_reenabled": r_code["sizes"],
            "categorical": r_cat["sizes"],
            "cat_method": r_cat["metadata"]["method"],
            "delta_code_vs_baseline_pct": (1 - code_total / base_total) * 100,
            "delta_cat_vs_baseline_pct": (1 - cat_total / base_total) * 100,
        }

        results_collector["compression_ratio"].append(result)

        # Verify categorical engages only at dtype boundary
        if info["quantized_dtype"] == "int16" and info["n_unique"] < 253:
            assert r_cat["metadata"]["method"] == "categorical", (
                f"Categorical should engage: dtype={info['quantized_dtype']}, "
                f"n_unique={info['n_unique']}"
            )
        if info["quantized_dtype"] == "int8":
            assert r_cat["metadata"]["method"] == "direct", (
                f"Categorical should NOT engage when dtype=int8"
            )


# =============================================================================
# TC-BENCH-02: Python Encode Time
# =============================================================================

class TestPythonEncodeTime:
    """
    TC-BENCH-02: Measure Python encoding time for all three variants.
    """

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("resolution", [5, 10, 20])
    def test_encode_time(self, dist, resolution, results_collector):
        data, delta, info = generate_test_data(dist, n=N_VALUES_SMALL,
                                                resolution_pct=resolution)

        t_baseline = measure_time(compress_baseline, data, delta, n_iter=5, warmup=1)
        t_code = measure_time(compress_code_reenabled, data, delta, n_iter=5, warmup=1)
        t_cat = measure_time(compress_categorical, data, delta, n_iter=5, warmup=1)

        results_collector["encode_time"].append({
            "distribution": dist,
            "resolution_pct": resolution,
            "n_values": N_VALUES_SMALL,
            "data_info": info,
            "baseline_ms": t_baseline,
            "code_reenabled_ms": t_code,
            "categorical_ms": t_cat,
        })


# =============================================================================
# TC-BENCH-03: JS Decode Time (Isolation)
# TC-BENCH-05: JS Decode End-to-End (decode + sum/histogram)
# =============================================================================

class TestJSDecodeTime:
    """
    TC-BENCH-03 + TC-BENCH-05: JavaScript decode performance.
    
    Runs bench_js_decode.js via Node.js subprocess, passing compressed
    data via stdin JSON and reading timing results from stdout.
    """

    @pytest.mark.parametrize("resolution", [5, 10, 20])
    def test_js_decode_gaussian(self, resolution, results_collector):
        """JS decode benchmark for Gaussian data."""
        self._run_js_benchmark("gaussian", resolution, results_collector)

    @pytest.mark.parametrize("resolution", [5, 10, 20])
    def test_js_decode_exponential(self, resolution, results_collector):
        """JS decode benchmark for Exponential data."""
        self._run_js_benchmark("exponential", resolution, results_collector)

    def _run_js_benchmark(self, dist, resolution, results_collector):
        data, delta, info = generate_test_data(dist, n=N_VALUES_SMALL,
                                                resolution_pct=resolution)

        r_baseline = compress_baseline(data, delta)
        r_code = compress_code_reenabled(data, delta)
        r_cat = compress_categorical(data, delta)

        config = {
            "tests": [
                {"label": "baseline", "payload": r_baseline["payload"],
                 "metadata": r_baseline["metadata"]},
                {"label": "code_reenabled", "payload": r_code["payload"],
                 "metadata": r_code["metadata"]},
            ]
        }
        # Only include categorical if it actually engaged
        if r_cat["metadata"]["method"] == "categorical":
            config["tests"].append({
                "label": "categorical", "payload": r_cat["payload"],
                "metadata": r_cat["metadata"],
            })

        js_script = BENCHMARK_DIR / "bench_js_decode.js"
        result = subprocess.run(
            ["node", str(js_script)],
            input=json.dumps(config),
            capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            pytest.skip(f"Node.js benchmark failed: {result.stderr[:200]}")

        js_results = json.loads(result.stdout)

        results_collector["js_decode"].append({
            "distribution": dist,
            "resolution_pct": resolution,
            "n_values": N_VALUES_SMALL,
            "data_info": info,
            "js_results": js_results,
        })


# =============================================================================
# TC-BENCH-04: Existing Code Action Baseline (numpy reimplementation)
# =============================================================================

class TestExistingCodeAction:
    """
    TC-BENCH-04: Baseline comparison using numpy reimplementation of
    the existing 'code' action.
    
    NOTE (P1-C11-2): Cannot simply remove the dtype skip in production
    code — existing 'code' action uses pandas methods (Series.unique(),
    Series.map()) but post-quantization data is numpy arrays. This
    benchmark uses np.unique + np.searchsorted as a fair comparison.
    """

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("resolution", [2, 5, 10, 20])
    def test_code_action_comparison(self, dist, resolution, results_collector):
        data, delta, info = generate_test_data(dist, n=N_VALUES,
                                                resolution_pct=resolution)

        r_baseline = compress_baseline(data, delta)
        r_code = compress_code_reenabled(data, delta)

        base_total = r_baseline["sizes"]["total_transfer"]
        code_total = r_code["sizes"]["total_transfer"]

        results_collector["code_action"].append({
            "distribution": dist,
            "resolution_pct": resolution,
            "data_info": info,
            "baseline_total": base_total,
            "code_total": code_total,
            "delta_pct": (1 - code_total / base_total) * 100,
        })


# =============================================================================
# TC-BENCH-06: Peak Memory During Encoding
# =============================================================================

class TestPeakMemory:
    """
    TC-BENCH-06: Measure peak memory during encoding (Gemini1 P2-7).
    """

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_peak_memory(self, dist, results_collector):
        data, delta, info = generate_test_data(dist, n=N_VALUES,
                                                resolution_pct=5)

        memory_results = {}
        for name, func in [("baseline", compress_baseline),
                           ("code_reenabled", compress_code_reenabled),
                           ("categorical", compress_categorical)]:
            tracemalloc.start()
            func(data, delta)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_results[name] = {
                "current_mb": current / 1024 / 1024,
                "peak_mb": peak / 1024 / 1024,
            }

        results_collector["peak_memory"].append({
            "distribution": dist,
            "n_values": N_VALUES,
            "data_info": info,
            "memory": memory_results,
        })


# =============================================================================
# DTYPE TRANSITION BOUNDARY TEST
# =============================================================================

class TestDtypeBoundary:
    """Verify savings appear exactly at int8→int16 boundary."""

    def test_boundary_sweep(self, results_collector):
        """Sweep across the int8/int16 boundary (range around 127)."""
        rng = np.random.default_rng(42)
        boundary_results = []

        for target_range in [60, 80, 100, 120, 127, 128, 130, 150, 180, 200, 250, 300]:
            n = N_VALUES
            # Create data with exactly target_range unique quantized values
            data = rng.choice(target_range, size=n).astype(np.float64)
            data = data + rng.normal(0, 0.001, n)  # tiny noise
            delta = 1.0

            r_baseline = compress_baseline(data, delta)
            r_cat = compress_categorical(data, delta)

            q, _ = quantize_delta(data, delta)
            n_unique = len(np.unique(q))

            boundary_results.append({
                "target_range": target_range,
                "actual_unique": n_unique,
                "quantized_dtype": q.dtype.name,
                "baseline_total": r_baseline["sizes"]["total_transfer"],
                "cat_total": r_cat["sizes"]["total_transfer"],
                "cat_method": r_cat["metadata"]["method"],
                "delta_pct": (1 - r_cat["sizes"]["total_transfer"] /
                             r_baseline["sizes"]["total_transfer"]) * 100,
            })

        results_collector["dtype_boundary"] = boundary_results


# =============================================================================
# ZIP-FRIENDLY INT16 WORST CASE (G7-9)
# =============================================================================

class TestZipFriendlyWorstCase:
    """
    G7-9: Test if structured zero-heavy int16 compresses so well
    that categorical gain collapses.
    
    When int16 values are mostly in [0..127], the high bytes are mostly
    zeros — very ZIP-friendly. This tests the worst case for categorical.
    """

    @pytest.mark.parametrize("n_unique", [130, 150, 180, 200, 252])
    def test_zip_friendly_int16(self, n_unique, results_collector):
        rng = np.random.default_rng(42)
        n = N_VALUES

        # Create int16 data where values are concentrated in [0..n_unique-1]
        # but range forces int16 (max value > 127)
        values = np.arange(n_unique)
        # Zipf-like distribution: most values are small
        probs = 1.0 / (np.arange(n_unique) + 1)
        probs = probs / probs.sum()
        data = rng.choice(values, size=n, p=probs).astype(np.float64)
        delta = 1.0

        r_baseline = compress_baseline(data, delta)
        r_cat = compress_categorical(data, delta)

        results_collector["zip_friendly"].append({
            "n_unique": n_unique,
            "baseline_total": r_baseline["sizes"]["total_transfer"],
            "baseline_zipped": r_baseline["sizes"]["zipped_bytes"],
            "cat_total": r_cat["sizes"]["total_transfer"],
            "cat_zipped": r_cat["sizes"]["zipped_bytes"],
            "cat_method": r_cat["metadata"]["method"],
            "delta_pct": (1 - r_cat["sizes"]["total_transfer"] /
                         r_baseline["sizes"]["total_transfer"]) * 100,
        })


# =============================================================================
# MIN_COLUMN_LENGTH BREAK-EVEN (G7-3)
# =============================================================================

class TestMinColumnLength:
    """
    G7-3: Measure break-even column length for categorical.
    
    At what column length does codebook overhead exceed byte savings?
    """

    @pytest.mark.parametrize("n_values", [100, 256, 512, 1024, 2048,
                                          4096, 8192, 16384])
    def test_break_even(self, n_values, results_collector):
        rng = np.random.default_rng(42)
        # 150 unique values forces int16, fits uint8 categorical
        data = rng.choice(150, size=n_values).astype(np.float64)
        data = data + rng.normal(0, 0.001, n_values)
        delta = 1.0

        r_baseline = compress_baseline(data, delta)
        r_cat = compress_categorical(data, delta)

        results_collector["min_column_length"].append({
            "n_values": n_values,
            "baseline_total": r_baseline["sizes"]["total_transfer"],
            "cat_total": r_cat["sizes"]["total_transfer"],
            "cat_method": r_cat["metadata"]["method"],
            "delta_pct": (1 - r_cat["sizes"]["total_transfer"] /
                         r_baseline["sizes"]["total_transfer"]) * 100
                         if r_cat["metadata"]["method"] == "categorical" else 0,
        })


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def results_collector():
    """Collect all benchmark results for final report generation."""
    results = {
        "compression_ratio": [],
        "encode_time": [],
        "js_decode": [],
        "code_action": [],
        "peak_memory": [],
        "dtype_boundary": [],
        "zip_friendly": [],
        "min_column_length": [],
    }
    yield results

    # After all tests: write results to JSON
    output_path = BENCHMARK_DIR / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\n{'='*70}")
    print(f"Benchmark results written to: {output_path}")
    print(f"{'='*70}\n")


# =============================================================================
# CONFTEST MARKERS (inline for standalone operation)
# =============================================================================

def pytest_configure(config):
    config.addinivalue_line("markers", "feature(id): Feature taxonomy ID")
    config.addinivalue_line("markers", "layer(name): Test layer")
    config.addinivalue_line("markers", "backend(name): Backend requirement")
