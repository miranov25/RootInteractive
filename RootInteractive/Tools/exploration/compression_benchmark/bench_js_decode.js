#!/usr/bin/env node
/**
 * Phase 0.1.C-0: JS Decode Benchmark
 * 
 * Measures JavaScript decode performance for three compression variants:
 * 1. Baseline (linear transform: origin + scale * value)
 * 2. Code re-enabled (codebook lookup via valueCode dict)
 * 3. Categorical (codebook array lookup with sentinel handling)
 * 
 * Also measures end-to-end: decode + downstream usage (sum, histogram fill).
 * 
 * Input: JSON from stdin with test configurations
 * Output: JSON to stdout with timing results
 * 
 * Author: Claude11 (Coder)
 * Phase: 0.1.C-0
 * Date: 2026-02-06
 */

const zlib = require('zlib');

// ============================================================================
// DECODE FUNCTIONS
// ============================================================================

function decodeBaseline(buffer, dtype, scale, origin) {
    let arr;
    if (dtype === 'int8') {
        arr = new Int8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    } else if (dtype === 'int16') {
        arr = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
    } else if (dtype === 'int32') {
        arr = new Int32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    } else if (dtype === 'uint8') {
        arr = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    } else {
        throw new Error('Unknown dtype: ' + dtype);
    }
    const values = new Float64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
        values[i] = origin + scale * arr[i];
    }
    return values;
}

function decodeCodeAction(buffer, dtype, valueCode) {
    let arr;
    if (dtype === 'int8') {
        arr = new Int8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    } else if (dtype === 'int16') {
        arr = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
    } else {
        throw new Error('Unknown dtype: ' + dtype);
    }
    const values = new Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
        values[i] = valueCode[arr[i]];
    }
    return values;
}

function decodeCategorical(buffer, dtype, codebook, sentinelNan, sentinelPinf, sentinelNinf) {
    let arr;
    if (dtype === 'uint8') {
        arr = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    } else if (dtype === 'uint16') {
        arr = new Uint16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
    } else {
        throw new Error('Unknown dtype: ' + dtype);
    }
    const values = new Float64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
        const code = arr[i];
        if (code === sentinelNan) {
            values[i] = NaN;
        } else if (code === sentinelPinf) {
            values[i] = Infinity;
        } else if (code === sentinelNinf) {
            values[i] = -Infinity;
        } else {
            values[i] = codebook[code];
        }
    }
    return values;
}

// ============================================================================
// DOWNSTREAM USAGE SIMULATION
// ============================================================================

function computeSum(values) {
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v === v) { // NaN check
            sum += v;
        }
    }
    return sum;
}

function fillHistogram(values, nbins, vmin, vmax) {
    const bins = new Int32Array(nbins);
    const step = (vmax - vmin) / nbins;
    for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v !== v) continue; // NaN
        const bin = Math.floor((v - vmin) / step);
        if (bin >= 0 && bin < nbins) {
            bins[bin]++;
        }
    }
    return bins;
}

// ============================================================================
// TIMING UTILITY
// ============================================================================

function measureTime(fn, nIter = 20, warmup = 5) {
    // Warmup
    for (let i = 0; i < warmup; i++) fn();

    const times = [];
    for (let i = 0; i < nIter; i++) {
        const t0 = performance.now();
        fn();
        const t1 = performance.now();
        times.push(t1 - t0);
    }
    times.sort((a, b) => a - b);
    return {
        mean_ms: times.reduce((a, b) => a + b) / times.length,
        median_ms: times[Math.floor(times.length / 2)],
        min_ms: times[0],
        max_ms: times[times.length - 1],
        p95_ms: times[Math.floor(times.length * 0.95)],
        n_iter: nIter,
    };
}

// ============================================================================
// MAIN
// ============================================================================

function runBenchmarks(config) {
    const results = {};

    for (const test of config.tests) {
        const label = test.label;
        const b64data = test.payload;
        const meta = test.metadata;

        // Decode base64
        const rawB64 = Buffer.from(b64data, 'base64');
        // Decompress
        const decompressed = zlib.inflateSync(rawB64);

        let decodeFn;
        let decodedRef;  // Reference decode for downstream tests

        if (meta.method === 'direct') {
            decodeFn = () => decodeBaseline(decompressed, meta.dtype, meta.scale, meta.origin);
        } else if (meta.method === 'code') {
            decodeFn = () => decodeCodeAction(decompressed, meta.dtype, meta.valueCode);
        } else if (meta.method === 'categorical') {
            decodeFn = () => decodeCategorical(
                decompressed, meta.dtype, meta.codebook,
                meta.sentinel_nan, meta.sentinel_pinf, meta.sentinel_ninf
            );
        } else {
            results[label] = { error: 'Unknown method: ' + meta.method };
            continue;
        }

        // Measure isolated decode time
        const decodeTime = measureTime(decodeFn);

        // Measure full pipeline: base64 + inflate + decode
        const fullPipelineFn = () => {
            const raw = Buffer.from(b64data, 'base64');
            const dec = zlib.inflateSync(raw);
            if (meta.method === 'direct') {
                return decodeBaseline(dec, meta.dtype, meta.scale, meta.origin);
            } else if (meta.method === 'code') {
                return decodeCodeAction(dec, meta.dtype, meta.valueCode);
            } else {
                return decodeCategorical(
                    dec, meta.dtype, meta.codebook,
                    meta.sentinel_nan, meta.sentinel_pinf, meta.sentinel_ninf
                );
            }
        };
        const fullTime = measureTime(fullPipelineFn);

        // Decode once for downstream tests
        decodedRef = decodeFn();
        const n = Array.isArray(decodedRef) ? decodedRef.length : decodedRef.length;

        // End-to-end: decode + sum
        const e2eSumFn = () => {
            const vals = decodeFn();
            return computeSum(vals);
        };
        const e2eSumTime = measureTime(e2eSumFn);

        // End-to-end: decode + histogram fill
        const e2eHistFn = () => {
            const vals = decodeFn();
            return fillHistogram(vals, 100, -5, 5);
        };
        const e2eHistTime = measureTime(e2eHistFn);

        results[label] = {
            n_values: n,
            decode_only: decodeTime,
            full_pipeline: fullTime,
            e2e_sum: e2eSumTime,
            e2e_histogram: e2eHistTime,
        };
    }

    return results;
}

// Read JSON from stdin
let inputData = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { inputData += chunk; });
process.stdin.on('end', () => {
    try {
        const config = JSON.parse(inputData);
        const results = runBenchmarks(config);
        process.stdout.write(JSON.stringify(results, null, 2));
    } catch (e) {
        process.stderr.write('Error: ' + e.message + '\n' + e.stack + '\n');
        process.exit(1);
    }
});
