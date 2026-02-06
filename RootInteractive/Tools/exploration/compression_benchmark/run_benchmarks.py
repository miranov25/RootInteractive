#!/usr/bin/env python3
# RootInteractive/Tools/exploration/compression_benchmark/run_benchmarks.py
"""
Phase 0.1.C-0: Direct Benchmark Runner.

Executes all benchmarks and generates comprehensive results JSON + report.
This is the primary entry point for running the benchmark suite.

Phase: 0.1.C-0 (Exploration & Benchmarking)
Date: 2026-02-06
Author: Claude11 (Coder)
"""

import json
import subprocess
import sys
import tracemalloc
import time
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from compression_utils import (
    compress_baseline,
    compress_categorical,
    compress_code_reenabled,
    compress_bitpacked,
    compress_int8_baseline,
    generate_test_data,
    generate_bitmask_data,
    measure_time,
    quantize_delta,
)

BENCHMARK_DIR = Path(__file__).parent
N_VALUES = 1_000_000
N_VALUES_SMALL = 100_000

DISTRIBUTIONS = ["gaussian", "exponential", "uniform"]
RESOLUTIONS = [1, 2, 5, 10, 20, 50]

results = {
    "compression_ratio": [],
    "encode_time": [],
    "js_decode": [],
    "code_action": [],
    "peak_memory": [],
    "dtype_boundary": [],
    "zip_friendly": [],
    "min_column_length": [],
    "bit_coding": [],
    "bit_coding_scale": [],
    "bit_coding_js": [],
    "metadata": {
        "date": "2026-02-06",
        "phase": "0.1.C-0",
        "author": "Claude11",
        "n_values": N_VALUES,
        "n_values_small": N_VALUES_SMALL,
    }
}


def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


# =============================================================================
# 1. COMPRESSION RATIO (TC-BENCH-01)
# =============================================================================

def run_compression_ratio():
    print_header("TC-BENCH-01: Compression Ratio Benchmark")
    
    for dist in DISTRIBUTIONS:
        for resolution in RESOLUTIONS:
            data, delta, info = generate_test_data(dist, n=N_VALUES,
                                                    resolution_pct=resolution)
            r_base = compress_baseline(data, delta)
            r_code = compress_code_reenabled(data, delta)
            r_cat = compress_categorical(data, delta)

            base_t = r_base["sizes"]["total_transfer"]
            code_t = r_code["sizes"]["total_transfer"]
            cat_t = r_cat["sizes"]["total_transfer"]

            row = {
                "distribution": dist,
                "resolution_pct": resolution,
                "n_unique": info["n_unique"],
                "range_size": info["range_size"],
                "quantized_dtype": info["quantized_dtype"],
                "baseline_zip_bytes": r_base["sizes"]["zipped_bytes"],
                "baseline_total": base_t,
                "code_zip_bytes": r_code["sizes"]["zipped_bytes"],
                "code_meta_bytes": r_code["sizes"]["metadata_bytes"],
                "code_total": code_t,
                "cat_zip_bytes": r_cat["sizes"]["zipped_bytes"],
                "cat_meta_bytes": r_cat["sizes"]["metadata_bytes"],
                "cat_total": cat_t,
                "cat_method": r_cat["metadata"]["method"],
                "delta_code_pct": round((1 - code_t / base_t) * 100, 1),
                "delta_cat_pct": round((1 - cat_t / base_t) * 100, 1),
            }
            results["compression_ratio"].append(row)

            flag = "★" if row["cat_method"] == "categorical" else " "
            print(f"  {flag} {dist:12s} {resolution:3d}% | "
                  f"unique={info['n_unique']:4d} range={info['range_size']:4d} "
                  f"dtype={info['quantized_dtype']:5s} | "
                  f"base={base_t:8d} code={code_t:8d} cat={cat_t:8d} | "
                  f"Δcode={row['delta_code_pct']:+5.1f}% Δcat={row['delta_cat_pct']:+5.1f}%")


# =============================================================================
# 2. PYTHON ENCODE TIME (TC-BENCH-02)
# =============================================================================

def run_encode_time():
    print_header("TC-BENCH-02: Python Encode Time")

    for dist in DISTRIBUTIONS:
        for resolution in [5, 10, 20]:
            data, delta, info = generate_test_data(dist, n=N_VALUES_SMALL,
                                                    resolution_pct=resolution)

            t_base = measure_time(compress_baseline, data, delta, n_iter=5, warmup=1)
            t_code = measure_time(compress_code_reenabled, data, delta, n_iter=5, warmup=1)
            t_cat = measure_time(compress_categorical, data, delta, n_iter=5, warmup=1)

            results["encode_time"].append({
                "distribution": dist,
                "resolution_pct": resolution,
                "n_values": N_VALUES_SMALL,
                "baseline_ms": t_base,
                "code_reenabled_ms": t_code,
                "categorical_ms": t_cat,
            })

            print(f"  {dist:12s} {resolution:3d}% | "
                  f"base={t_base['mean_ms']:6.1f}ms "
                  f"code={t_code['mean_ms']:6.1f}ms "
                  f"cat={t_cat['mean_ms']:6.1f}ms")


# =============================================================================
# 3. JS DECODE TIME (TC-BENCH-03 + TC-BENCH-05)
# =============================================================================

def run_js_decode():
    print_header("TC-BENCH-03/05: JS Decode Time (Isolation + End-to-End)")

    js_script = BENCHMARK_DIR / "bench_js_decode.js"
    if not js_script.exists():
        print("  SKIP: bench_js_decode.js not found")
        return

    for dist in ["gaussian", "exponential"]:
        for resolution in [5, 10, 20]:
            data, delta, info = generate_test_data(dist, n=N_VALUES_SMALL,
                                                    resolution_pct=resolution)

            r_base = compress_baseline(data, delta)
            r_code = compress_code_reenabled(data, delta)
            r_cat = compress_categorical(data, delta)

            config = {"tests": [
                {"label": "baseline", "payload": r_base["payload"],
                 "metadata": r_base["metadata"]},
                {"label": "code_reenabled", "payload": r_code["payload"],
                 "metadata": r_code["metadata"]},
            ]}
            if r_cat["metadata"]["method"] == "categorical":
                config["tests"].append({
                    "label": "categorical",
                    "payload": r_cat["payload"],
                    "metadata": r_cat["metadata"],
                })

            try:
                proc = subprocess.run(
                    ["node", str(js_script)],
                    input=json.dumps(config),
                    capture_output=True, text=True, timeout=120
                )
                if proc.returncode != 0:
                    print(f"  SKIP {dist} {resolution}%: Node error: {proc.stderr[:100]}")
                    continue

                js_results = json.loads(proc.stdout)
                results["js_decode"].append({
                    "distribution": dist,
                    "resolution_pct": resolution,
                    "n_values": N_VALUES_SMALL,
                    "data_info": info,
                    "js_results": js_results,
                })

                # Print summary
                for label in ["baseline", "code_reenabled", "categorical"]:
                    if label in js_results:
                        r = js_results[label]
                        print(f"  {dist:12s} {resolution:3d}% {label:16s} | "
                              f"decode={r['decode_only']['mean_ms']:6.2f}ms "
                              f"full={r['full_pipeline']['mean_ms']:6.2f}ms "
                              f"e2e_sum={r['e2e_sum']['mean_ms']:6.2f}ms "
                              f"e2e_hist={r['e2e_histogram']['mean_ms']:6.2f}ms")

            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT: {dist} {resolution}%")
            except Exception as e:
                print(f"  ERROR: {dist} {resolution}%: {e}")


# =============================================================================
# 4. DTYPE BOUNDARY SWEEP
# =============================================================================

def run_dtype_boundary():
    print_header("Dtype Boundary Sweep (int8↔int16)")

    rng = np.random.default_rng(42)

    for target_range in [60, 80, 100, 120, 127, 128, 130, 150, 180, 200, 250, 300, 400]:
        n = N_VALUES
        data = rng.choice(target_range, size=n).astype(np.float64)
        data = data + rng.normal(0, 0.001, n)
        delta = 1.0

        r_base = compress_baseline(data, delta)
        r_cat = compress_categorical(data, delta)

        q, _ = quantize_delta(data, delta)
        n_unique = len(np.unique(q))

        row = {
            "target_range": target_range,
            "actual_unique": n_unique,
            "quantized_dtype": q.dtype.name,
            "baseline_total": r_base["sizes"]["total_transfer"],
            "cat_total": r_cat["sizes"]["total_transfer"],
            "cat_method": r_cat["metadata"]["method"],
            "delta_pct": round((1 - r_cat["sizes"]["total_transfer"] /
                               r_base["sizes"]["total_transfer"]) * 100, 1),
        }
        results["dtype_boundary"].append(row)

        flag = "★" if row["cat_method"] == "categorical" else " "
        print(f"  {flag} range={target_range:4d} unique={n_unique:4d} "
              f"dtype={q.dtype.name:5s} | "
              f"base={row['baseline_total']:8d} cat={row['cat_total']:8d} "
              f"Δ={row['delta_pct']:+5.1f}%")


# =============================================================================
# 5. ZIP-FRIENDLY INT16 WORST CASE (G7-9)
# =============================================================================

def run_zip_friendly():
    print_header("G7-9: ZIP-Friendly Int16 Worst Case")

    rng = np.random.default_rng(42)

    for n_unique in [130, 140, 150, 180, 200, 220, 252]:
        values = np.arange(n_unique)
        probs = 1.0 / (np.arange(n_unique) + 1)
        probs = probs / probs.sum()
        data = rng.choice(values, size=N_VALUES, p=probs).astype(np.float64)
        delta = 1.0

        r_base = compress_baseline(data, delta)
        r_cat = compress_categorical(data, delta)

        row = {
            "n_unique": n_unique,
            "baseline_zip": r_base["sizes"]["zipped_bytes"],
            "baseline_total": r_base["sizes"]["total_transfer"],
            "cat_zip": r_cat["sizes"]["zipped_bytes"],
            "cat_meta": r_cat["sizes"]["metadata_bytes"],
            "cat_total": r_cat["sizes"]["total_transfer"],
            "cat_method": r_cat["metadata"]["method"],
            "delta_pct": round((1 - r_cat["sizes"]["total_transfer"] /
                               r_base["sizes"]["total_transfer"]) * 100, 1),
        }
        results["zip_friendly"].append(row)

        print(f"  unique={n_unique:3d} | "
              f"base_zip={row['baseline_zip']:8d} base_total={row['baseline_total']:8d} | "
              f"cat_zip={row['cat_zip']:8d} cat_meta={row['cat_meta']:5d} "
              f"cat_total={row['cat_total']:8d} | "
              f"Δ={row['delta_pct']:+5.1f}%")


# =============================================================================
# 6. MIN_COLUMN_LENGTH BREAK-EVEN (G7-3)
# =============================================================================

def run_min_column_length():
    print_header("G7-3: Min Column Length Break-Even")

    rng = np.random.default_rng(42)

    for n_values in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536]:
        data = rng.choice(150, size=n_values).astype(np.float64)
        data = data + rng.normal(0, 0.001, n_values)
        delta = 1.0

        r_base = compress_baseline(data, delta)
        r_cat = compress_categorical(data, delta)

        row = {
            "n_values": n_values,
            "baseline_total": r_base["sizes"]["total_transfer"],
            "cat_total": r_cat["sizes"]["total_transfer"],
            "cat_method": r_cat["metadata"]["method"],
            "delta_pct": round((1 - r_cat["sizes"]["total_transfer"] /
                               r_base["sizes"]["total_transfer"]) * 100, 1)
                               if r_cat["metadata"]["method"] == "categorical" else 0,
        }
        results["min_column_length"].append(row)

        flag = "+" if row["delta_pct"] > 0 else "-"
        print(f"  {flag} n={n_values:6d} | "
              f"base={row['baseline_total']:8d} cat={row['cat_total']:8d} | "
              f"Δ={row['delta_pct']:+5.1f}% [{row['cat_method']}]")


# =============================================================================
# 7. PEAK MEMORY (TC-BENCH-06)
# =============================================================================

def run_peak_memory():
    print_header("TC-BENCH-06: Peak Memory During Encoding")

    for dist in DISTRIBUTIONS:
        data, delta, info = generate_test_data(dist, n=N_VALUES,
                                                resolution_pct=5)

        mem = {}
        for name, func in [("baseline", compress_baseline),
                           ("code_reenabled", compress_code_reenabled),
                           ("categorical", compress_categorical)]:
            tracemalloc.start()
            func(data, delta)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem[name] = {
                "current_mb": round(current / 1024 / 1024, 2),
                "peak_mb": round(peak / 1024 / 1024, 2),
            }

        results["peak_memory"].append({
            "distribution": dist,
            "n_values": N_VALUES,
            "memory": mem,
        })

        print(f"  {dist:12s} | "
              f"base={mem['baseline']['peak_mb']:5.1f}MB "
              f"code={mem['code_reenabled']['peak_mb']:5.1f}MB "
              f"cat={mem['categorical']['peak_mb']:5.1f}MB")


# =============================================================================
# 8. BIT CODING BENCHMARK (TC-BENCH-07)
# =============================================================================

def run_bit_coding():
    print_header("TC-BENCH-07: Bit Coding Benchmark")

    PATTERNS = [
        "boolean_balanced",
        "boolean_sparse_1pct",
        "boolean_sparse_01pct",
        "status_2bit",
        "status_2bit_skewed",
        "status_3bit",
        "status_4bit",
        "detector_mask_8bit",
        "quality_flag",
    ]

    results["bit_coding"] = []

    for pattern in PATTERNS:
        data, bits_needed, info = generate_bitmask_data(pattern, n=N_VALUES)

        # Variant A: int8 baseline (current production)
        r_int8 = compress_int8_baseline(data)

        # Variant B: bit packing + ZIP
        r_bit = compress_bitpacked(data, bits_needed)

        # Variant C: categorical (treat as float, apply code table)
        r_cat = compress_categorical(data.astype(np.float64), 1.0)

        # Compute savings
        int8_total = r_int8["sizes"]["total_transfer"]
        bit_total = r_bit["sizes"]["total_transfer"]
        cat_total = r_cat["sizes"]["total_transfer"]

        row = {
            "pattern": pattern,
            "bits_needed": bits_needed,
            "n_unique": info["n_unique"],
            "entropy_bits": info["entropy_bits"],
            "theoretical_min_bytes": info["theoretical_min_bytes"],
            "int8_raw": r_int8["sizes"]["raw_bytes"],
            "int8_zip": r_int8["sizes"]["zipped_bytes"],
            "int8_total": int8_total,
            "bitpack_raw": r_bit["sizes"]["raw_bytes"],
            "bitpack_zip": r_bit["sizes"]["zipped_bytes"],
            "bitpack_total": bit_total,
            "cat_total": cat_total,
            "cat_method": r_cat["metadata"]["method"],
            "delta_bitpack_pct": round((1 - bit_total / int8_total) * 100, 1),
            "delta_cat_pct": round((1 - cat_total / int8_total) * 100, 1),
            "raw_compression_ratio": round(r_int8["sizes"]["raw_bytes"] /
                                           r_bit["sizes"]["raw_bytes"], 1),
        }
        results["bit_coding"].append(row)

        print(f"  {pattern:25s} | bits={bits_needed} ent={info['entropy_bits']:.2f} | "
              f"int8={int8_total:8d} bitpack={bit_total:8d} cat={cat_total:8d} | "
              f"Δbit={row['delta_bitpack_pct']:+5.1f}% Δcat={row['delta_cat_pct']:+5.1f}% "
              f"raw={row['raw_compression_ratio']:.0f}x")


# =============================================================================
# 9. BIT CODING SCALE PROJECTION (10^8 rows)
# =============================================================================

def run_bit_coding_scale():
    print_header("TC-BENCH-08: Bit Coding Scale Projection (10^8 rows)")
    print("  NOTE: Using extrapolation from 10^6 measurements\n")

    results["bit_coding_scale"] = []

    # Measure at 10^6 to extrapolate
    N_MEASURE = 1_000_000
    N_TARGET = 100_000_000  # 10^8
    SCALE_FACTOR = N_TARGET / N_MEASURE

    patterns_to_project = [
        "boolean_balanced",
        "boolean_sparse_1pct",
        "status_2bit_skewed",
        "quality_flag",
        "status_4bit",
        "detector_mask_8bit",
    ]

    print(f"  {'Pattern':25s} | {'int8 @10^8':>12s} {'bitpack @10^8':>14s} "
          f"{'cat @10^8':>12s} | {'Memory saved/col':>16s}")
    print(f"  {'-'*25}-+-{'-'*12}-{'-'*14}-{'-'*12}-+-{'-'*16}")

    for pattern in patterns_to_project:
        data, bits_needed, info = generate_bitmask_data(pattern, n=N_MEASURE)

        r_int8 = compress_int8_baseline(data)
        r_bit = compress_bitpacked(data, bits_needed)
        r_cat = compress_categorical(data.astype(np.float64), 1.0)

        # Extrapolate: ZIP+base64 scales roughly linearly with data size
        int8_proj = int(r_int8["sizes"]["total_transfer"] * SCALE_FACTOR)
        bit_proj = int(r_bit["sizes"]["total_transfer"] * SCALE_FACTOR)
        cat_proj = int(r_cat["sizes"]["total_transfer"] * SCALE_FACTOR)

        # Client-side memory: what gets stored after decode
        # int8: 1 byte/value; bitpacked: bits_needed/8 bytes/value
        mem_int8_mb = N_TARGET * 1 / (1024 * 1024)
        mem_bit_mb = N_TARGET * bits_needed / 8 / (1024 * 1024)
        mem_saved_mb = mem_int8_mb - mem_bit_mb

        row = {
            "pattern": pattern,
            "bits_needed": bits_needed,
            "entropy_bits": info["entropy_bits"],
            "int8_proj_mb": round(int8_proj / (1024 * 1024), 1),
            "bitpack_proj_mb": round(bit_proj / (1024 * 1024), 1),
            "cat_proj_mb": round(cat_proj / (1024 * 1024), 1),
            "client_mem_int8_mb": round(mem_int8_mb, 1),
            "client_mem_bitpack_mb": round(mem_bit_mb, 1),
            "client_mem_saved_mb": round(mem_saved_mb, 1),
        }
        results["bit_coding_scale"].append(row)

        print(f"  {pattern:25s} | "
              f"{row['int8_proj_mb']:8.1f} MB  {row['bitpack_proj_mb']:8.1f} MB   "
              f"{row['cat_proj_mb']:8.1f} MB  | "
              f"{row['client_mem_saved_mb']:8.1f} MB saved")

    # Summary: impact on 500 MB JSON limit
    print(f"\n  --- Impact on 500 MB JSON/Bokeh Limit ---")
    print(f"  At 10^8 rows with N boolean columns:")
    bool_data, _, _ = generate_bitmask_data("boolean_balanced", n=N_MEASURE)
    r_int8 = compress_int8_baseline(bool_data)
    r_bit = compress_bitpacked(bool_data, 1)
    int8_per_col_mb = r_int8["sizes"]["total_transfer"] * SCALE_FACTOR / (1024 * 1024)
    bit_per_col_mb = r_bit["sizes"]["total_transfer"] * SCALE_FACTOR / (1024 * 1024)

    n_cols_int8 = int(500 / int8_per_col_mb)
    n_cols_bit = int(500 / bit_per_col_mb)
    print(f"    int8:     {int8_per_col_mb:.1f} MB/col → max {n_cols_int8} boolean columns in 500 MB")
    print(f"    bitpack:  {bit_per_col_mb:.1f} MB/col → max {n_cols_bit} boolean columns in 500 MB")
    print(f"    Gain:     {n_cols_bit / max(n_cols_int8, 1):.1f}x more boolean columns")

    results["bit_coding_scale"].append({
        "summary": "500MB_limit_boolean_columns",
        "int8_mb_per_col": round(int8_per_col_mb, 1),
        "bitpack_mb_per_col": round(bit_per_col_mb, 1),
        "max_cols_int8": n_cols_int8,
        "max_cols_bitpack": n_cols_bit,
    })


# =============================================================================
# 10. BIT CODING JS DECODE (TC-BENCH-09)
# =============================================================================

def run_bit_coding_js():
    print_header("TC-BENCH-09: Bit Coding JS Decode Time")

    js_script = BENCHMARK_DIR / "bench_js_decode.js"
    if not js_script.exists():
        print("  SKIP: bench_js_decode.js not found")
        return

    results["bit_coding_js"] = []

    for pattern in ["boolean_balanced", "boolean_sparse_1pct", "status_2bit_skewed"]:
        data, bits, info = generate_bitmask_data(pattern, n=N_VALUES_SMALL)

        r_int8 = compress_int8_baseline(data)

        config = {"tests": [
            {"label": f"{pattern}_int8",
             "payload": r_int8["payload"],
             "metadata": r_int8["metadata"]},
        ]}

        # For int8 baseline, treat as direct decode with scale=1, origin=0
        config["tests"][0]["metadata"]["scale"] = 1.0
        config["tests"][0]["metadata"]["origin"] = 0.0
        config["tests"][0]["metadata"]["method"] = "direct"

        try:
            proc = subprocess.run(
                ["node", str(js_script)],
                input=json.dumps(config),
                capture_output=True, text=True, timeout=60
            )
            if proc.returncode == 0:
                js_results = json.loads(proc.stdout)
                results["bit_coding_js"].append({
                    "pattern": pattern,
                    "js_results": js_results,
                })
                for label, r in js_results.items():
                    print(f"  {label:30s} | decode={r['decode_only']['mean_ms']:5.2f}ms "
                          f"full={r['full_pipeline']['mean_ms']:5.2f}ms")
            else:
                print(f"  SKIP {pattern}: {proc.stderr[:100]}")
        except Exception as e:
            print(f"  ERROR {pattern}: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_header("Phase 0.1.C-0: Compression Benchmark Suite")
    print(f"  Date: 2026-02-06")
    print(f"  Author: Claude11 (Coder)")
    print(f"  N_VALUES: {N_VALUES:,}  N_VALUES_SMALL: {N_VALUES_SMALL:,}")

    t0 = time.time()

    run_compression_ratio()
    run_encode_time()
    run_js_decode()
    run_dtype_boundary()
    run_zip_friendly()
    run_min_column_length()
    run_peak_memory()
    run_bit_coding()
    run_bit_coding_scale()
    run_bit_coding_js()

    elapsed = time.time() - t0

    # Write results
    output_path = BENCHMARK_DIR / "benchmark_results.json"
    results["metadata"]["elapsed_seconds"] = round(elapsed, 1)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print_header(f"COMPLETE — {elapsed:.1f}s elapsed")
    print(f"  Results: {output_path}")
