/**
 * Phase 0.1.D — Convolution Benchmark: WASM vs JS
 *
 * Tests tight inner-loop performance where WASM should have clear advantage:
 *   - 1D convolution with kernel sizes 3, 5, 9, 21
 *   - N = 1K to 1M data points
 *   - Memory-local access pattern, no Math.* calls
 *
 * Requires: wasm_conv.wasm compiled from wasm_conv.cpp
 *
 * Usage: node bench_convolution.mjs <wasm_path>
 */
import fs from "fs";

// ============================================================
// Setup WASM
// ============================================================
const wasmPath = process.argv[2] || "wasm_conv.wasm";
const wasmBytes = fs.readFileSync(wasmPath);
const imports = {
    env: { emscripten_notify_memory_growth: () => {} },
    wasi_snapshot_preview1: {},
};
const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
const ex = instance.exports;
let memory = ex.memory;

function copyToWasm(arr, ptr) {
    new Float64Array(memory.buffer, ptr, arr.length).set(arr);
}
function readFromWasm(ptr, len) {
    return new Float64Array(memory.buffer.slice(ptr, ptr + len * 8));
}

// ============================================================
// JS reference: 1D convolution (same algorithm as C++)
// ============================================================
function js_convolve(data, kernel, out, n, klen) {
    const half = (klen - 1) >> 1;
    for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < klen; j++) {
            let idx = i - half + j;
            // Clamp to edges (mirror boundary)
            if (idx < 0) idx = 0;
            if (idx >= n) idx = n - 1;
            sum += data[idx] * kernel[j];
        }
        out[i] = sum;
    }
}

// Also test a naive no-boundary-check version for fair comparison
function js_convolve_inner(data, kernel, out, n, klen) {
    const half = (klen - 1) >> 1;
    // Boundary region
    for (let i = 0; i < half; i++) {
        let sum = 0;
        for (let j = 0; j < klen; j++) {
            let idx = i - half + j;
            if (idx < 0) idx = 0;
            sum += data[idx] * kernel[j];
        }
        out[i] = sum;
    }
    // Inner region (no boundary checks)
    for (let i = half; i < n - half; i++) {
        let sum = 0;
        for (let j = 0; j < klen; j++) {
            sum += data[i - half + j] * kernel[j];
        }
        out[i] = sum;
    }
    // Boundary region
    for (let i = n - half; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < klen; j++) {
            let idx = i - half + j;
            if (idx >= n) idx = n - 1;
            sum += data[idx] * kernel[j];
        }
        out[i] = sum;
    }
}

// ============================================================
// Gaussian kernel generator
// ============================================================
function makeGaussianKernel(size) {
    const kernel = new Float64Array(size);
    const sigma = size / 4;
    const half = (size - 1) / 2;
    let sum = 0;
    for (let i = 0; i < size; i++) {
        const x = i - half;
        kernel[i] = Math.exp(-0.5 * (x * x) / (sigma * sigma));
        sum += kernel[i];
    }
    // Normalize
    for (let i = 0; i < size; i++) kernel[i] /= sum;
    return kernel;
}

// ============================================================
// Benchmark harness
// ============================================================
function bench(fn, warmup = 100, iterations = 1000) {
    for (let i = 0; i < warmup; i++) fn();
    const times = [];
    for (let i = 0; i < iterations; i++) {
        const t0 = performance.now();
        fn();
        times.push(performance.now() - t0);
    }
    times.sort((a, b) => a - b);
    return times[Math.floor(times.length / 2)] * 1000; // median µs
}

function pad(s, w) { return String(s).padStart(w); }

function makeData(n) {
    const data = new Float64Array(n);
    for (let i = 0; i < n; i++) data[i] = Math.sin(i * 0.01) + 0.5 * Math.cos(i * 0.03);
    return data;
}

// ============================================================
// Invariance check: verify WASM == JS
// ============================================================
console.log("╔══════════════════════════════════════════════════════════════╗");
console.log("║     Phase 0.1.D — Convolution Benchmark: WASM vs JS       ║");
console.log("╚══════════════════════════════════════════════════════════════╝\n");

console.log("─── Invariance Check ───\n");

const KERNEL_SIZES = [3, 5, 9, 21];
const CHECK_N = 1000;
const checkData = makeData(CHECK_N);

for (const klen of KERNEL_SIZES) {
    const kernel = makeGaussianKernel(klen);
    const jsOut = new Float64Array(CHECK_N);
    js_convolve(checkData, kernel, jsOut, CHECK_N, klen);

    // WASM
    const dataPtr = ex.malloc(CHECK_N * 8);
    const kernPtr = ex.malloc(klen * 8);
    const outPtr = ex.malloc(CHECK_N * 8);
    copyToWasm(checkData, dataPtr);
    copyToWasm(kernel, kernPtr);
    ex.convolve1d(dataPtr, kernPtr, outPtr, CHECK_N, klen);
    const wasmOut = readFromWasm(outPtr, CHECK_N);
    ex.free(dataPtr); ex.free(kernPtr); ex.free(outPtr);

    // Compare
    let maxDiff = 0;
    for (let i = 0; i < CHECK_N; i++) {
        const d = Math.abs(jsOut[i] - wasmOut[i]);
        if (d > maxDiff) maxDiff = d;
    }
    const ok = maxDiff < 1e-12;
    console.log(`  ${ok ? "✅" : "❌"} kernel=${klen}: max_diff=${maxDiff.toExponential(2)}${ok ? "" : " FAIL"}`);
}

// ============================================================
// TC-BENCH-CONV: Convolution throughput
// ============================================================
const DATA_SIZES = [1000, 10000, 100000, 1000000];

console.log("\n─── Convolution Throughput (pre-allocated buffers) ───\n");

for (const klen of KERNEL_SIZES) {
    const kernel = makeGaussianKernel(klen);
    const ops = klen; // multiply-adds per output element

    console.log(`  Kernel size = ${klen} (${ops} ops/element):`);
    console.log(`  ${"N".padStart(10)} │ ${"JS (µs)".padStart(10)} │ ${"JS opt".padStart(10)} │ ${"WASM (µs)".padStart(10)} │ ${"W/JS".padStart(6)} │ ${"W/opt".padStart(6)} │ Winner`);
    console.log(`  ──────────┼────────────┼────────────┼────────────┼────────┼────────┼───────`);

    for (const N of DATA_SIZES) {
        const data = makeData(N);
        const jsOut = new Float64Array(N);
        const iters = Math.max(20, Math.floor(200000 / N));

        // JS naive
        const js_us = bench(() => js_convolve(data, kernel, jsOut, N, klen), 20, iters);

        // JS optimized (split boundary/inner)
        const jsopt_us = bench(() => js_convolve_inner(data, kernel, jsOut, N, klen), 20, iters);

        // WASM pre-allocated
        const dataPtr = ex.malloc(N * 8);
        const kernPtr = ex.malloc(klen * 8);
        const outPtr = ex.malloc(N * 8);
        copyToWasm(data, dataPtr);
        copyToWasm(kernel, kernPtr);

        const wasm_us = bench(() => {
            ex.convolve1d(dataPtr, kernPtr, outPtr, N, klen);
        }, 20, iters);

        ex.free(dataPtr); ex.free(kernPtr); ex.free(outPtr);

        const ratio_naive = (wasm_us / js_us).toFixed(2);
        const ratio_opt = (wasm_us / jsopt_us).toFixed(2);
        const winner = wasm_us < jsopt_us ? "WASM" : "JS";

        console.log(
            `  ${pad(N, 10)} │ ${pad(js_us.toFixed(1), 10)} │ ${pad(jsopt_us.toFixed(1), 10)} │ ${pad(wasm_us.toFixed(1), 10)} │ ${pad(ratio_naive, 6)} │ ${pad(ratio_opt, 6)} │ ${winner}`
        );
    }
    console.log("");
}

// ============================================================
// Summary
// ============================================================
console.log("─── Summary ───\n");
console.log("  W/JS:  WASM time / JS naive time (< 1.0 = WASM faster)");
console.log("  W/opt: WASM time / JS optimized time (< 1.0 = WASM faster)");
console.log("  JS opt: boundary-check hoisted out of inner loop");
console.log("  WASM: compute-only (data pre-copied, no malloc in hot loop)");
console.log("");
console.log("  Node.js v" + process.version.slice(1));
console.log("  WASM memory: " + (memory.buffer.byteLength / 1024 / 1024).toFixed(1) + " MB (after growth)");
