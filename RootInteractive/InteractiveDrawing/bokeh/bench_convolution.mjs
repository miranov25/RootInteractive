/**
 * Phase 0.1.D — Convolution Benchmark: WASM vs JS (1D, 2D, 3D)
 *
 * Tests tight inner-loop performance for histogram smoothing use cases:
 *   1D: signal smoothing, N = 1K..1M
 *   2D: 2D histogram smoothing, grid sizes 64² to 512²
 *   3D: 3D histogram smoothing, grid sizes 16³ to 64³
 *
 * Usage: node bench_convolution.mjs <wasm_path>
 */
import fs from "fs";

// ============================================================
// WASM Setup
// ============================================================
const wasmPath = process.argv[2] || "wasm_conv.wasm";
const wasmBytes = fs.readFileSync(wasmPath);
const imports = {
    env: { emscripten_notify_memory_growth: () => {} },
    wasi_snapshot_preview1: {},
};
const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
const ex = instance.exports;
const memory = ex.memory;

function copyToWasm(arr, ptr) {
    new Float64Array(memory.buffer, ptr, arr.length).set(arr);
}
function readFromWasm(ptr, len) {
    return new Float64Array(memory.buffer.slice(ptr, ptr + len * 8));
}

// ============================================================
// JS Reference Implementations
// ============================================================
function clamp(val, max) {
    if (val < 0) return 0;
    if (val >= max) return max - 1;
    return val;
}

function js_convolve1d(data, kernel, out, n, klen) {
    const half = (klen - 1) >> 1;
    // Boundary left
    for (let i = 0; i < half; i++) {
        let sum = 0;
        for (let j = 0; j < klen; j++) sum += data[clamp(i - half + j, n)] * kernel[j];
        out[i] = sum;
    }
    // Inner
    for (let i = half; i < n - half; i++) {
        let sum = 0;
        for (let j = 0; j < klen; j++) sum += data[i - half + j] * kernel[j];
        out[i] = sum;
    }
    // Boundary right
    for (let i = n - half; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < klen; j++) sum += data[clamp(i - half + j, n)] * kernel[j];
        out[i] = sum;
    }
}

function js_convolve2d(data, kernel, out, nx, ny, klen) {
    const half = (klen - 1) >> 1;
    for (let y = 0; y < ny; y++) {
        const y_inner = (y >= half && y < ny - half);
        for (let x = 0; x < nx; x++) {
            let sum = 0;
            if (y_inner && x >= half && x < nx - half) {
                for (let ky = 0; ky < klen; ky++) {
                    const row = (y - half + ky) * nx;
                    const krow = ky * klen;
                    for (let kx = 0; kx < klen; kx++) {
                        sum += data[row + (x - half + kx)] * kernel[krow + kx];
                    }
                }
            } else {
                for (let ky = 0; ky < klen; ky++) {
                    const iy = clamp(y - half + ky, ny);
                    const krow = ky * klen;
                    for (let kx = 0; kx < klen; kx++) {
                        sum += data[iy * nx + clamp(x - half + kx, nx)] * kernel[krow + kx];
                    }
                }
            }
            out[y * nx + x] = sum;
        }
    }
}

function js_convolve3d(data, kernel, out, nx, ny, nz, klen) {
    const half = (klen - 1) >> 1;
    const slice = ny * nx;
    for (let z = 0; z < nz; z++) {
        const z_inner = (z >= half && z < nz - half);
        for (let y = 0; y < ny; y++) {
            const y_inner = (y >= half && y < ny - half);
            for (let x = 0; x < nx; x++) {
                let sum = 0;
                if (z_inner && y_inner && x >= half && x < nx - half) {
                    for (let kz = 0; kz < klen; kz++) {
                        const iz = z - half + kz;
                        for (let ky = 0; ky < klen; ky++) {
                            const iy = y - half + ky;
                            const base_d = iz * slice + iy * nx + (x - half);
                            const base_k = (kz * klen + ky) * klen;
                            for (let kx = 0; kx < klen; kx++) {
                                sum += data[base_d + kx] * kernel[base_k + kx];
                            }
                        }
                    }
                } else {
                    for (let kz = 0; kz < klen; kz++) {
                        const iz = clamp(z - half + kz, nz);
                        for (let ky = 0; ky < klen; ky++) {
                            const iy = clamp(y - half + ky, ny);
                            const base_k = (kz * klen + ky) * klen;
                            for (let kx = 0; kx < klen; kx++) {
                                const ix = clamp(x - half + kx, nx);
                                sum += data[iz * slice + iy * nx + ix] * kernel[base_k + kx];
                            }
                        }
                    }
                }
                out[z * slice + y * nx + x] = sum;
            }
        }
    }
}

// ============================================================
// Kernel generators
// ============================================================
function makeGaussianKernel1D(size) {
    const k = new Float64Array(size);
    const sigma = size / 4, half = (size - 1) / 2;
    let sum = 0;
    for (let i = 0; i < size; i++) { const x = i - half; k[i] = Math.exp(-0.5 * x * x / (sigma * sigma)); sum += k[i]; }
    for (let i = 0; i < size; i++) k[i] /= sum;
    return k;
}

function makeGaussianKernel2D(size) {
    const k = new Float64Array(size * size);
    const sigma = size / 4, half = (size - 1) / 2;
    let sum = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const dx = x - half, dy = y - half;
            k[y * size + x] = Math.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma));
            sum += k[y * size + x];
        }
    }
    for (let i = 0; i < k.length; i++) k[i] /= sum;
    return k;
}

function makeGaussianKernel3D(size) {
    const k = new Float64Array(size * size * size);
    const sigma = size / 4, half = (size - 1) / 2;
    let sum = 0;
    for (let z = 0; z < size; z++) {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - half, dy = y - half, dz = z - half;
                const idx = (z * size + y) * size + x;
                k[idx] = Math.exp(-0.5 * (dx * dx + dy * dy + dz * dz) / (sigma * sigma));
                sum += k[idx];
            }
        }
    }
    for (let i = 0; i < k.length; i++) k[i] /= sum;
    return k;
}

// ============================================================
// Data generators
// ============================================================
function makeData1D(n) {
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = Math.sin(i * 0.01) + 0.5 * Math.cos(i * 0.03);
    return d;
}
function makeData2D(nx, ny) {
    const d = new Float64Array(nx * ny);
    for (let i = 0; i < d.length; i++) d[i] = Math.sin(i * 0.01) + 0.3 * Math.cos(i * 0.07);
    return d;
}
function makeData3D(nx, ny, nz) {
    const d = new Float64Array(nx * ny * nz);
    for (let i = 0; i < d.length; i++) d[i] = Math.sin(i * 0.01) + 0.3 * Math.cos(i * 0.07);
    return d;
}

// ============================================================
// Benchmark harness
// ============================================================
function bench(fn, warmup = 50, iterations = 200) {
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

// ============================================================
// Invariance checks
// ============================================================
function checkInvariance(jsOut, wasmOut, n, label) {
    let maxDiff = 0;
    for (let i = 0; i < n; i++) {
        const d = Math.abs(jsOut[i] - wasmOut[i]);
        if (d > maxDiff) maxDiff = d;
    }
    const ok = maxDiff < 1e-10;
    console.log(`  ${ok ? "✅" : "❌"} ${label}: max_diff=${maxDiff.toExponential(2)}${ok ? "" : " FAIL"}`);
    return ok;
}

// ============================================================
// Main
// ============================================================
console.log("╔══════════════════════════════════════════════════════════════╗");
console.log("║  Phase 0.1.D — Convolution Benchmark: WASM vs JS (1D/2D/3D)║");
console.log("╚══════════════════════════════════════════════════════════════╝\n");

const KERNEL_SIZES = [3, 5, 9];

// ─── Invariance ───
console.log("─── Invariance Check ───\n");

// 1D
for (const klen of KERNEL_SIZES) {
    const N = 1000;
    const data = makeData1D(N), kernel = makeGaussianKernel1D(klen);
    const jsOut = new Float64Array(N);
    js_convolve1d(data, kernel, jsOut, N, klen);
    const dp = ex.malloc(N * 8), kp = ex.malloc(klen * 8), op = ex.malloc(N * 8);
    copyToWasm(data, dp); copyToWasm(kernel, kp);
    ex.convolve1d(dp, kp, op, N, klen);
    const wasmOut = readFromWasm(op, N);
    ex.free(dp); ex.free(kp); ex.free(op);
    checkInvariance(jsOut, wasmOut, N, `1D k=${klen}`);
}

// 2D
for (const klen of KERNEL_SIZES) {
    const nx = 64, ny = 64, total = nx * ny;
    const data = makeData2D(nx, ny), kernel = makeGaussianKernel2D(klen);
    const jsOut = new Float64Array(total);
    js_convolve2d(data, kernel, jsOut, nx, ny, klen);
    const dp = ex.malloc(total * 8), kp = ex.malloc(klen * klen * 8), op = ex.malloc(total * 8);
    copyToWasm(data, dp); copyToWasm(kernel, kp);
    ex.convolve2d(dp, kp, op, nx, ny, klen);
    const wasmOut = readFromWasm(op, total);
    ex.free(dp); ex.free(kp); ex.free(op);
    checkInvariance(jsOut, wasmOut, total, `2D ${nx}×${ny} k=${klen}`);
}

// 3D
for (const klen of [3, 5]) {
    const nx = 16, ny = 16, nz = 16, total = nx * ny * nz;
    const data = makeData3D(nx, ny, nz), kernel = makeGaussianKernel3D(klen);
    const jsOut = new Float64Array(total);
    js_convolve3d(data, kernel, jsOut, nx, ny, nz, klen);
    const dp = ex.malloc(total * 8), kp = ex.malloc(klen ** 3 * 8), op = ex.malloc(total * 8);
    copyToWasm(data, dp); copyToWasm(kernel, kp);
    ex.convolve3d(dp, kp, op, nx, ny, nz, klen);
    const wasmOut = readFromWasm(op, total);
    ex.free(dp); ex.free(kp); ex.free(op);
    checkInvariance(jsOut, wasmOut, total, `3D ${nx}³ k=${klen}`);
}

// ─── 1D Benchmarks ───
console.log("\n─── 1D Convolution Throughput ───\n");

const SIZES_1D = [1000, 10000, 100000, 1000000];

for (const klen of KERNEL_SIZES) {
    const kernel = makeGaussianKernel1D(klen);
    console.log(`  Kernel = ${klen}:`);
    console.log(`  ${"N".padStart(10)} │ ${"JS (µs)".padStart(10)} │ ${"WASM (µs)".padStart(10)} │ ${"Ratio".padStart(6)} │ Winner`);
    console.log(`  ──────────┼────────────┼────────────┼────────┼───────`);

    for (const N of SIZES_1D) {
        const data = makeData1D(N);
        const jsOut = new Float64Array(N);
        const iters = Math.max(20, Math.floor(200000 / N));

        const js_us = bench(() => js_convolve1d(data, kernel, jsOut, N, klen), 20, iters);

        const dp = ex.malloc(N * 8), kp = ex.malloc(klen * 8), op = ex.malloc(N * 8);
        copyToWasm(data, dp); copyToWasm(kernel, kp);
        const wasm_us = bench(() => ex.convolve1d(dp, kp, op, N, klen), 20, iters);
        ex.free(dp); ex.free(kp); ex.free(op);

        const ratio = (wasm_us / js_us).toFixed(2);
        const winner = wasm_us < js_us ? "WASM" : "JS";
        console.log(`  ${pad(N, 10)} │ ${pad(js_us.toFixed(1), 10)} │ ${pad(wasm_us.toFixed(1), 10)} │ ${pad(ratio, 6)} │ ${winner}`);
    }
    console.log("");
}

// ─── 2D Benchmarks ───
console.log("─── 2D Convolution Throughput (histogram smoothing) ───\n");

const GRIDS_2D = [
    [64, 64],
    [128, 128],
    [256, 256],
    [512, 512],
];

for (const klen of KERNEL_SIZES) {
    const kernel = makeGaussianKernel2D(klen);
    console.log(`  Kernel = ${klen}×${klen} (${klen * klen} ops/element):`);
    console.log(`  ${"Grid".padStart(12)} │ ${"Total".padStart(8)} │ ${"JS (µs)".padStart(10)} │ ${"WASM (µs)".padStart(10)} │ ${"Ratio".padStart(6)} │ Winner`);
    console.log(`  ────────────┼──────────┼────────────┼────────────┼────────┼───────`);

    for (const [nx, ny] of GRIDS_2D) {
        const total = nx * ny;
        const data = makeData2D(nx, ny);
        const jsOut = new Float64Array(total);
        const iters = Math.max(10, Math.floor(50000 / total));

        const js_us = bench(() => js_convolve2d(data, kernel, jsOut, nx, ny, klen), 10, iters);

        const dp = ex.malloc(total * 8), kp = ex.malloc(klen * klen * 8), op = ex.malloc(total * 8);
        copyToWasm(data, dp); copyToWasm(kernel, kp);
        const wasm_us = bench(() => ex.convolve2d(dp, kp, op, nx, ny, klen), 10, iters);
        ex.free(dp); ex.free(kp); ex.free(op);

        const ratio = (wasm_us / js_us).toFixed(2);
        const winner = wasm_us < js_us ? "WASM" : "JS";
        console.log(`  ${pad(`${nx}×${ny}`, 12)} │ ${pad(total, 8)} │ ${pad(js_us.toFixed(1), 10)} │ ${pad(wasm_us.toFixed(1), 10)} │ ${pad(ratio, 6)} │ ${winner}`);
    }
    console.log("");
}

// ─── 3D Benchmarks ───
console.log("─── 3D Convolution Throughput (3D histogram smoothing) ───\n");

const GRIDS_3D = [
    [16, 16, 16],
    [32, 32, 32],
    [64, 64, 64],
];

for (const klen of [3, 5]) {
    const kernel = makeGaussianKernel3D(klen);
    console.log(`  Kernel = ${klen}³ (${klen ** 3} ops/element):`);
    console.log(`  ${"Grid".padStart(14)} │ ${"Total".padStart(8)} │ ${"JS (µs)".padStart(12)} │ ${"WASM (µs)".padStart(12)} │ ${"Ratio".padStart(6)} │ Winner`);
    console.log(`  ──────────────┼──────────┼──────────────┼──────────────┼────────┼───────`);

    for (const [nx, ny, nz] of GRIDS_3D) {
        const total = nx * ny * nz;
        const data = makeData3D(nx, ny, nz);
        const jsOut = new Float64Array(total);
        const iters = Math.max(5, Math.floor(10000 / Math.max(total / 1000, 1)));

        const js_us = bench(() => js_convolve3d(data, kernel, jsOut, nx, ny, nz, klen), 5, iters);

        const dp = ex.malloc(total * 8), kp = ex.malloc(klen ** 3 * 8), op = ex.malloc(total * 8);
        copyToWasm(data, dp); copyToWasm(kernel, kp);
        const wasm_us = bench(() => ex.convolve3d(dp, kp, op, nx, ny, nz, klen), 5, iters);
        ex.free(dp); ex.free(kp); ex.free(op);

        const ratio = (wasm_us / js_us).toFixed(2);
        const winner = wasm_us < js_us ? "WASM" : "JS";
        console.log(`  ${pad(`${nx}×${ny}×${nz}`, 14)} │ ${pad(total, 8)} │ ${pad(js_us.toFixed(1), 12)} │ ${pad(wasm_us.toFixed(1), 12)} │ ${pad(ratio, 6)} │ ${winner}`);
    }
    console.log("");
}

// ─── Summary ───
console.log("─── Summary ───\n");
console.log("  Ratio < 1.0 = WASM faster, > 1.0 = JS faster");
console.log("  All WASM benchmarks: compute-only (data pre-copied)");
console.log("  Node.js v" + process.version.slice(1));
console.log("  WASM memory: " + (memory.buffer.byteLength / 1024 / 1024).toFixed(1) + " MB (after growth)");
