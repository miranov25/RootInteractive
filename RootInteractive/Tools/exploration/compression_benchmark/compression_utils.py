# RootInteractive/Tools/exploration/compression_benchmark/compression_utils.py
"""
Standalone compression utilities for Phase 0.1.C-0 benchmarking.

These are STANDALONE reimplementations for benchmarking purposes.
They do NOT modify or import from production code (compressArray.py).

Phase: 0.1.C-0 (Exploration & Benchmarking)
Date: 2026-02-06
Author: Claude11 (Coder)
"""

import numpy as np
import zlib
import base64
import time
import json


# =============================================================================
# QUANTIZATION (standalone copy of roundAbsolute logic)
# =============================================================================

def quantize_delta(data, delta, downgrade_type=True):
    """
    Standalone reimplementation of roundAbsolute for benchmarking.
    
    Replicates the production logic (compressArray.py lines 53-76)
    without importing from production code.
    
    Args:
        data: np.ndarray (float64)
        delta: float (quantization step)
        downgrade_type: bool (if True, downgrade to smallest int dtype)
    
    Returns:
        quantized: np.ndarray (int8, int16, or int32)
        metadata: dict with 'scale' and 'origin', or None
    
    NOTE: Does NOT handle ±Inf. NaN is mapped to -1 (current production
    behavior). Phase 0.1.C-0 benchmarks use NaN-only test data.
    
    >>> arr = np.array([0.0, 0.5, 1.0, np.nan, 1.5])
    >>> q, meta = quantize_delta(arr, 0.1)
    >>> meta['scale']
    0.1
    >>> q[3]  # NaN mapped to -1
    -1
    """
    dtype = data.dtype
    if dtype.kind not in ['f', 'c']:
        return data, None
    if delta == 0:
        raise ZeroDivisionError("delta must be non-zero")

    quantized = np.rint(data / delta)
    result = quantized * delta
    delta_mean = np.nanmean(data - result)

    if downgrade_type:
        df_min = np.nanmin(quantized)
        quantized = np.where(np.isnan(data), -1, quantized - df_min)
        range_size = np.max(quantized)
        origin = df_min * delta - delta_mean

        if range_size <= 0x7f:
            return quantized.astype(np.int8), {"scale": delta, "origin": origin}
        if range_size <= 0x7fff:
            return quantized.astype(np.int16), {"scale": delta, "origin": origin}
        return quantized.astype(np.int32), {"scale": delta, "origin": origin}

    result -= delta_mean
    return result.astype(dtype), None


# =============================================================================
# VARIANT 1: BASELINE (current pipeline)
# =============================================================================

def compress_baseline(data, delta):
    """
    Current production pipeline: delta → zip → base64.
    
    Returns:
        result: dict with 'payload' (base64 string), 'metadata', 'sizes'
    """
    quantized, meta = quantize_delta(data, delta)
    if meta is None:
        meta = {"scale": 1.0, "origin": 0.0}

    dtype_name = quantized.dtype.name
    raw_bytes = quantized.tobytes()
    compressed = zlib.compress(raw_bytes)
    encoded = base64.b64encode(compressed).decode("utf-8")

    metadata = {
        "dtype": dtype_name,
        "scale": meta["scale"],
        "origin": meta["origin"],
        "method": "direct",
    }

    return {
        "payload": encoded,
        "metadata": metadata,
        "sizes": {
            "raw_bytes": len(raw_bytes),
            "zipped_bytes": len(compressed),
            "base64_bytes": len(encoded),
            "metadata_bytes": len(json.dumps(metadata).encode()),
            "total_transfer": len(encoded) + len(json.dumps(metadata).encode()),
        }
    }


# =============================================================================
# VARIANT 2: CODE RE-ENABLED (numpy reimplementation)
# =============================================================================

def compress_code_reenabled(data, delta):
    """
    Simulate re-enabling the existing 'code' action for numerics.
    
    Pipeline: delta → code → zip → base64.
    
    NOTE (P1-C11-2): The production 'code' action uses pandas methods
    (Series.unique(), Series.map()) which fail on numpy arrays.
    This benchmark uses numpy equivalents (np.unique + np.searchsorted)
    as a fair comparison.
    
    The existing 'code' action uses int8 for <256 unique values and
    SIGNED int types. We replicate that behavior.
    """
    quantized, meta = quantize_delta(data, delta)
    if meta is None:
        meta = {"scale": 1.0, "origin": 0.0}

    # Numpy reimplementation of the code action (lines 203-215)
    unique_vals = np.unique(quantized)
    n_unique = len(unique_vals)

    # Build forward mapping: value → code
    codes = np.searchsorted(unique_vals, quantized)

    # Match production dtype selection (int8/int16/int32, SIGNED)
    if n_unique < 2**8:
        codes = codes.astype(np.int8)
    elif n_unique < 2**16:
        codes = codes.astype(np.int16)
    else:
        codes = codes.astype(np.int32)

    dtype_name = codes.dtype.name
    raw_bytes = codes.tobytes()
    compressed = zlib.compress(raw_bytes)
    encoded = base64.b64encode(compressed).decode("utf-8")

    # valueCode dict (matches production format)
    value_code = {int(i): float(v) for i, v in enumerate(unique_vals)}

    metadata = {
        "dtype": dtype_name,
        "scale": meta["scale"],
        "origin": meta["origin"],
        "method": "code",
        "valueCode": value_code,
        "skipCode": False,
    }

    return {
        "payload": encoded,
        "metadata": metadata,
        "sizes": {
            "raw_bytes": len(raw_bytes),
            "zipped_bytes": len(compressed),
            "base64_bytes": len(encoded),
            "metadata_bytes": len(json.dumps(metadata).encode()),
            "total_transfer": len(encoded) + len(json.dumps(metadata).encode()),
        }
    }


# =============================================================================
# VARIANT 3: CATEGORICAL (proposed Phase 0.1.C)
# =============================================================================

def compress_categorical(data, delta):
    """
    Proposed categorical pipeline: delta → categorical(uint8) → zip → base64.
    
    Uses pre-computed NaN mask (Phase 0.1.B pattern).
    Codebook stores decoded float values (§3.6).
    Sentinel codes at top of range (§3.2).
    """
    # Step 1: Extract NaN mask BEFORE quantization (Phase 0.1.B pattern)
    nan_mask = np.isnan(data)
    # No ±Inf in 0.1.C-0 benchmarks (Phase 0.1.B not yet merged)

    # Step 2: Quantize
    quantized, meta = quantize_delta(data, delta)
    if meta is None:
        meta = {"scale": 1.0, "origin": 0.0}

    # Step 3: Analyze cardinality (using masks, NOT np.isfinite)
    finite_quantized = quantized[~nan_mask]
    unique_vals = np.unique(finite_quantized)
    n_unique = len(unique_vals)

    sentinel_count = 3  # NaN, +Inf, -Inf
    n_codes = n_unique + sentinel_count

    # Check if categorical provides dtype downgrade
    if n_codes > 255 or np.dtype(np.uint8).itemsize >= quantized.dtype.itemsize:
        # No benefit — fall back to baseline
        return compress_baseline(data, delta)

    # Step 4: Categorical encode using pre-computed masks
    SENTINEL_NAN = n_codes - 3
    SENTINEL_PINF = n_codes - 2
    SENTINEL_NINF = n_codes - 1

    codes = np.zeros(len(quantized), dtype=np.uint8)
    finite_mask = ~nan_mask
    if np.any(finite_mask):
        codes[finite_mask] = np.searchsorted(unique_vals, quantized[finite_mask]).astype(np.uint8)
    codes[nan_mask] = SENTINEL_NAN

    # Codebook: decoded float values (§3.6)
    codebook = meta["origin"] + meta["scale"] * unique_vals.astype(np.float64)

    dtype_name = codes.dtype.name  # "uint8"
    raw_bytes = codes.tobytes()
    compressed = zlib.compress(raw_bytes)
    encoded = base64.b64encode(compressed).decode("utf-8")

    metadata = {
        "dtype": dtype_name,
        "method": "categorical",
        "codebook": codebook.tolist(),
        "n_unique": int(n_unique),
        "sentinel_nan": int(SENTINEL_NAN),
        "sentinel_pinf": int(SENTINEL_PINF),
        "sentinel_ninf": int(SENTINEL_NINF),
    }

    return {
        "payload": encoded,
        "metadata": metadata,
        "sizes": {
            "raw_bytes": len(raw_bytes),
            "zipped_bytes": len(compressed),
            "base64_bytes": len(encoded),
            "metadata_bytes": len(json.dumps(metadata).encode()),
            "total_transfer": len(encoded) + len(json.dumps(metadata).encode()),
        }
    }


# =============================================================================
# DECODE (Python-side, for roundtrip verification)
# =============================================================================

def decode_baseline(result):
    """Decode baseline compression result back to float64."""
    raw = base64.b64decode(result["payload"])
    decompressed = zlib.decompress(raw)
    meta = result["metadata"]
    arr = np.frombuffer(decompressed, dtype=meta["dtype"])
    if meta.get("method") == "direct":
        decoded = meta["origin"] + meta["scale"] * arr.astype(np.float64)
        # Restore NaN for sentinel value -1 in signed types
        if np.dtype(meta["dtype"]).kind == 'i':
            nan_sentinel = -1 - 0  # roundAbsolute maps NaN to -1 before subtracting dfMin
            # Actually: roundAbsolute does quantized-dfMin for finite, -1 for NaN
            # So the NaN value in the array is -1 (before the offset subtraction)
            # After: quantized = np.where(np.isnan(df), -1, quantized-dfMin)
            # So NaN is always -1 in the array
            decoded[arr == -1] = np.nan
        return decoded
    return arr.astype(np.float64)


def decode_categorical(result):
    """Decode categorical compression result back to float64."""
    raw = base64.b64decode(result["payload"])
    decompressed = zlib.decompress(raw)
    meta = result["metadata"]
    codes = np.frombuffer(decompressed, dtype=meta["dtype"])
    codebook = np.array(meta["codebook"])

    values = np.empty(len(codes), dtype=np.float64)
    sentinel_nan = meta["sentinel_nan"]
    sentinel_pinf = meta["sentinel_pinf"]
    sentinel_ninf = meta["sentinel_ninf"]

    for i, code in enumerate(codes):
        if code == sentinel_nan:
            values[i] = np.nan
        elif code == sentinel_pinf:
            values[i] = np.inf
        elif code == sentinel_ninf:
            values[i] = -np.inf
        else:
            values[i] = codebook[code]

    return values


def decode_categorical_vectorized(result):
    """Vectorized decode — closer to JS performance characteristics."""
    raw = base64.b64decode(result["payload"])
    decompressed = zlib.decompress(raw)
    meta = result["metadata"]
    codes = np.frombuffer(decompressed, dtype=meta["dtype"])
    codebook = np.array(meta["codebook"])

    sentinel_nan = meta["sentinel_nan"]
    sentinel_pinf = meta["sentinel_pinf"]
    sentinel_ninf = meta["sentinel_ninf"]

    # Vectorized lookup
    values = codebook[np.clip(codes, 0, len(codebook) - 1)]
    values[codes == sentinel_nan] = np.nan
    values[codes == sentinel_pinf] = np.inf
    values[codes == sentinel_ninf] = -np.inf

    return values


# =============================================================================
# TIMING UTILITY
# =============================================================================

def measure_time(func, *args, n_iter=10, warmup=2, **kwargs):
    """
    Measure execution time with warmup.
    
    Returns:
        dict with 'mean_ms', 'min_ms', 'max_ms', 'std_ms', 'n_iter'
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "mean_ms": np.mean(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "std_ms": np.std(times),
        "n_iter": n_iter,
    }


# =============================================================================
# VARIANT 4: BIT PACKING (proposed for bitmask/status columns)
# =============================================================================

def bitpack_array(data, bits_per_value):
    """
    Pack integer values into minimal bits.
    
    Args:
        data: np.ndarray of integers in [0, 2^bits_per_value - 1]
        bits_per_value: int (1, 2, 3, 4, 5, 6, 7, 8)
    
    Returns:
        packed: np.ndarray of uint8 (packed bytes)
        n_values: int (original length, needed for unpacking tail)
    
    For bits_per_value=1: 8 values per byte → 8x compression vs int8
    For bits_per_value=2: 4 values per byte → 4x compression vs int8
    For bits_per_value=4: 2 values per byte → 2x compression vs int8
    """
    n = len(data)
    total_bits = n * bits_per_value
    n_bytes = (total_bits + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)

    if bits_per_value == 1:
        # Optimized: pack 8 booleans per byte
        full_bytes = n // 8
        remainder = n % 8
        if full_bytes > 0:
            reshaped = data[:full_bytes * 8].reshape(full_bytes, 8).astype(np.uint8)
            for bit in range(8):
                packed[:full_bytes] |= reshaped[:, bit] << (7 - bit)
        for i in range(remainder):
            packed[full_bytes] |= int(data[full_bytes * 8 + i]) << (7 - i)
    elif bits_per_value == 2:
        # Pack 4 values per byte
        full_bytes = n // 4
        remainder = n % 4
        if full_bytes > 0:
            reshaped = data[:full_bytes * 4].reshape(full_bytes, 4).astype(np.uint8)
            packed[:full_bytes] = (
                (reshaped[:, 0] << 6) |
                (reshaped[:, 1] << 4) |
                (reshaped[:, 2] << 2) |
                reshaped[:, 3]
            )
        for i in range(remainder):
            packed[full_bytes] |= int(data[full_bytes * 4 + i]) << (6 - 2 * i)
    elif bits_per_value == 4:
        # Pack 2 values per byte (nibbles)
        full_bytes = n // 2
        remainder = n % 2
        if full_bytes > 0:
            reshaped = data[:full_bytes * 2].reshape(full_bytes, 2).astype(np.uint8)
            packed[:full_bytes] = (reshaped[:, 0] << 4) | reshaped[:, 1]
        if remainder:
            packed[full_bytes] = int(data[-1]) << 4
    else:
        # General case: bit-by-bit packing
        bit_pos = 0
        for i in range(n):
            val = int(data[i])
            for b in range(bits_per_value - 1, -1, -1):
                byte_idx = bit_pos // 8
                bit_idx = 7 - (bit_pos % 8)
                if val & (1 << b):
                    packed[byte_idx] |= (1 << bit_idx)
                bit_pos += 1

    return packed, n


def bitunpack_array(packed, n_values, bits_per_value):
    """Unpack bit-packed array back to integer values."""
    result = np.zeros(n_values, dtype=np.uint8)

    if bits_per_value == 1:
        full_bytes = n_values // 8
        remainder = n_values % 8
        if full_bytes > 0:
            for bit in range(8):
                result[bit::8][:full_bytes] = (packed[:full_bytes] >> (7 - bit)) & 1
        for i in range(remainder):
            result[full_bytes * 8 + i] = (packed[full_bytes] >> (7 - i)) & 1
    elif bits_per_value == 2:
        full_bytes = n_values // 4
        remainder = n_values % 4
        if full_bytes > 0:
            result[0::4][:full_bytes] = (packed[:full_bytes] >> 6) & 3
            result[1::4][:full_bytes] = (packed[:full_bytes] >> 4) & 3
            result[2::4][:full_bytes] = (packed[:full_bytes] >> 2) & 3
            result[3::4][:full_bytes] = packed[:full_bytes] & 3
        for i in range(remainder):
            result[full_bytes * 4 + i] = (packed[full_bytes] >> (6 - 2 * i)) & 3
    elif bits_per_value == 4:
        full_bytes = n_values // 2
        remainder = n_values % 2
        if full_bytes > 0:
            result[0::2][:full_bytes] = (packed[:full_bytes] >> 4) & 0xF
            result[1::2][:full_bytes] = packed[:full_bytes] & 0xF
        if remainder:
            result[-1] = (packed[full_bytes] >> 4) & 0xF
    else:
        bit_pos = 0
        for i in range(n_values):
            val = 0
            for b in range(bits_per_value - 1, -1, -1):
                byte_idx = bit_pos // 8
                bit_idx = 7 - (bit_pos % 8)
                if packed[byte_idx] & (1 << bit_idx):
                    val |= (1 << b)
                bit_pos += 1
            result[i] = val

    return result


def compress_bitpacked(data_int, bits_per_value):
    """
    Bit packing pipeline: bitpack → zip → base64.
    
    For bitmask/status columns where values need only 1-4 bits.
    No quantization step needed — data is already integer.
    
    Args:
        data_int: np.ndarray of integers
        bits_per_value: number of bits needed per value
    """
    packed, n_values = bitpack_array(data_int, bits_per_value)

    raw_bytes = packed.tobytes()
    compressed = zlib.compress(raw_bytes)
    encoded = base64.b64encode(compressed).decode("utf-8")

    metadata = {
        "dtype": "bitpacked",
        "method": "bitpack",
        "bits_per_value": bits_per_value,
        "n_values": n_values,
    }

    return {
        "payload": encoded,
        "metadata": metadata,
        "sizes": {
            "raw_bytes": len(raw_bytes),
            "zipped_bytes": len(compressed),
            "base64_bytes": len(encoded),
            "metadata_bytes": len(json.dumps(metadata).encode()),
            "total_transfer": len(encoded) + len(json.dumps(metadata).encode()),
        },
    }


def compress_int8_baseline(data_int):
    """Baseline: store as int8 → zip → base64 (current production for integer columns)."""
    arr = data_int.astype(np.int8)
    raw_bytes = arr.tobytes()
    compressed = zlib.compress(raw_bytes)
    encoded = base64.b64encode(compressed).decode("utf-8")

    metadata = {"dtype": "int8", "method": "direct_int"}

    return {
        "payload": encoded,
        "metadata": metadata,
        "sizes": {
            "raw_bytes": len(raw_bytes),
            "zipped_bytes": len(compressed),
            "base64_bytes": len(encoded),
            "metadata_bytes": len(json.dumps(metadata).encode()),
            "total_transfer": len(encoded) + len(json.dumps(metadata).encode()),
        },
    }


# =============================================================================
# BIT CODING DATA GENERATORS
# =============================================================================

def generate_bitmask_data(pattern, n=1_000_000, seed=42):
    """
    Generate bitmask/status test data.
    
    Args:
        pattern: str, one of:
            'boolean_balanced' - 50/50 true/false
            'boolean_sparse_1pct' - 1% true, 99% false
            'boolean_sparse_01pct' - 0.1% true
            'status_2bit' - 4 states (0-3) uniform
            'status_2bit_skewed' - 4 states, dominated by 0
            'status_3bit' - 8 states (0-7) uniform
            'status_4bit' - 16 states (0-15) uniform
            'detector_mask_8bit' - 8 detector flags combined
            'quality_flag' - 3 quality levels (good/warn/bad)
        n: number of values
        seed: random seed
    
    Returns:
        data: np.ndarray (uint8 or uint16)
        bits_needed: int (minimum bits to represent)
        info: dict (pattern metadata)
    """
    rng = np.random.default_rng(seed)

    if pattern == "boolean_balanced":
        data = rng.integers(0, 2, size=n, dtype=np.uint8)
        bits_needed = 1
        entropy = 1.0

    elif pattern == "boolean_sparse_1pct":
        data = np.zeros(n, dtype=np.uint8)
        idx = rng.choice(n, size=int(n * 0.01), replace=False)
        data[idx] = 1
        bits_needed = 1
        p = 0.01
        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    elif pattern == "boolean_sparse_01pct":
        data = np.zeros(n, dtype=np.uint8)
        idx = rng.choice(n, size=int(n * 0.001), replace=False)
        data[idx] = 1
        bits_needed = 1
        p = 0.001
        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    elif pattern == "status_2bit":
        data = rng.integers(0, 4, size=n, dtype=np.uint8)
        bits_needed = 2
        entropy = 2.0

    elif pattern == "status_2bit_skewed":
        # 80% state 0, 10% state 1, 7% state 2, 3% state 3
        probs = [0.80, 0.10, 0.07, 0.03]
        data = rng.choice(4, size=n, p=probs).astype(np.uint8)
        bits_needed = 2
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    elif pattern == "status_3bit":
        data = rng.integers(0, 8, size=n, dtype=np.uint8)
        bits_needed = 3
        entropy = 3.0

    elif pattern == "status_4bit":
        data = rng.integers(0, 16, size=n, dtype=np.uint8)
        bits_needed = 4
        entropy = 4.0

    elif pattern == "detector_mask_8bit":
        # 8 independent detector flags, each ~5% probability
        flags = rng.random((n, 8)) < 0.05
        data = np.zeros(n, dtype=np.uint8)
        for bit in range(8):
            data |= flags[:, bit].astype(np.uint8) << bit
        bits_needed = 8
        # Entropy of 8 independent Bernoulli(0.05) ~= 8 * H(0.05)
        p = 0.05
        entropy = 8 * (-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))

    elif pattern == "quality_flag":
        # good=0 (90%), warning=1 (8%), bad=2 (2%)
        probs = [0.90, 0.08, 0.02]
        data = rng.choice(3, size=n, p=probs).astype(np.uint8)
        bits_needed = 2
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    info = {
        "pattern": pattern,
        "n": n,
        "bits_needed": bits_needed,
        "n_unique": len(np.unique(data)),
        "entropy_bits": round(entropy, 3),
        "theoretical_min_bytes": int(np.ceil(n * entropy / 8)),
    }

    return data, bits_needed, info


# =============================================================================
# DATA GENERATORS
# =============================================================================

def generate_test_data(distribution, n=1_000_000, resolution_pct=5, seed=42,
                       nan_fraction=0.01):
    """
    Generate synthetic test data for benchmarking.
    
    Args:
        distribution: str ('gaussian', 'exponential', 'uniform')
        n: int (number of values)
        resolution_pct: int (resolution as % of characteristic scale)
        seed: int
        nan_fraction: float (fraction of NaN values)
    
    Returns:
        data: np.ndarray (float64, with NaN)
        delta: float (quantization step)
        info: dict (distribution parameters)
    """
    rng = np.random.default_rng(seed)

    if distribution == "gaussian":
        data = rng.normal(0, 1, n)
        sigma = 1.0
        delta = sigma * resolution_pct / 100
    elif distribution == "exponential":
        data = rng.exponential(1, n)
        sigma = 1.0  # std of exponential(1) = 1
        delta = sigma * resolution_pct / 100
    elif distribution == "uniform":
        data = rng.uniform(0, 10, n)
        data_range = 10.0
        delta = data_range * resolution_pct / 100
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Inject NaN (Phase 0.1.C-0: NaN only, no ±Inf)
    nan_indices = rng.choice(n, size=int(n * nan_fraction), replace=False)
    data[nan_indices] = np.nan

    # Compute expected properties
    quantized, meta = quantize_delta(data, delta)
    n_unique = len(np.unique(quantized[~np.isnan(data)]))
    range_size = int(np.max(quantized))

    info = {
        "distribution": distribution,
        "n": n,
        "resolution_pct": resolution_pct,
        "delta": delta,
        "nan_fraction": nan_fraction,
        "n_unique": n_unique,
        "range_size": range_size,
        "quantized_dtype": quantized.dtype.name,
    }

    return data, delta, info
