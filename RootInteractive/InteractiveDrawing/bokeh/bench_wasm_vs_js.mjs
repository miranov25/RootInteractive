/**
 * Phase 0.1.D v1.1 — WASM vs JS Performance Benchmark
 *
 * TC-BENCH-01: Scalar call overhead (ns/call)
 * TC-BENCH-02: Vector throughput with malloc/free (µs/call)
 * TC-BENCH-03: Vector throughput with pre-allocated buffers (µs/call)
 * TC-BENCH-04: Memory copy fraction (% of total time)
 *
 * Usage: node bench_wasm_vs_js.mjs <wasm_path>
 *
 * Output: timing tables + crossover analysis for decision framework.
 * All benchmarks Node.js; browser may vary ±2-3×.
 */
import fs from "fs";

// ============================================================
// Setup
// ============================================================
const wasmPath = process.argv[2] || "functions.wasm";
const wasmBytes = fs.readFileSync(wasmPath);
const imports = {
    env: {
        emscripten_notify_memory_growth: () => {},
    },
    wasi_snapshot_preview1: {},
};
const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
const ex = instance.exports;
const memory = ex.memory;

const hasFree = typeof ex.free === 'function';
const hasResetHeap = typeof ex.reset_heap === 'function';

function copyToWasm(arr, ptr) {
    // Re-reference memory.buffer — it changes after memory growth
    new Float64Array(memory.buffer, ptr, arr.length).set(arr);
}
function readFromWasm(ptr, len) {
    return new Float64Array(memory.buffer.slice(ptr, ptr + len * 8));
}

// JS reference implementations (identical math)
const JS = {
    fun1: (a, b, c, out, n) => { for (let i = 0; i < n; i++) out[i] = (a[i] + b[i]) / c[i]; },
    fun2: (a, b, _c, out, n) => { for (let i = 0; i < n; i++) out[i] = a[i] * b[i] + 1.0; },
    fun3: (a, _b, _c, out, n) => { for (let i = 0; i < n; i++) out[i] = Math.sqrt(a[i] * a[i] + 1.0); },
    fun4: (a, b, c, out, n) => { for (let i = 0; i < n; i++) out[i] = a[i] > b[i] ? c[i] : -c[i]; },
    fun5: (a, b, c, out, n) => { for (let i = 0; i < n; i++) out[i] = Math.sin(a[i]) * Math.cos(b[i]) + Math.exp(-c[i]); },
};

// JS scalar versions
const JS_S = {
    fun1: (a, b, c) => (a + b) / c,
    fun2: (a, b) => a * b + 1.0,
    fun3: (a) => Math.sqrt(a * a + 1.0),
    fun4: (a, b, c) => a > b ? c : -c,
    fun5: (a, b, c) => Math.sin(a) * Math.cos(b) + Math.exp(-c),
};

// ============================================================
// Benchmark harness
// ============================================================
function bench(fn, warmup = 200, iterations = 5000) {
    for (let i = 0; i < warmup; i++) fn();
    const times = [];
    for (let i = 0; i < iterations; i++) {
        const t0 = performance.now();
        fn();
        times.push(performance.now() - t0);
    }
    times.sort((a, b) => a - b);
    const median_ms = times[Math.floor(times.length / 2)];
    return median_ms * 1e6; // return ns
}

function makeData(n) {
    const a = new Float64Array(n);
    const b = new Float64Array(n);
    const c = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        a[i] = (i + 1) * 0.1;
        b[i] = (i + 1) * 0.2;
        c[i] = (i + 1) * 0.3 + 0.1;
    }
    return { a, b, c };
}

function pad(s, w) { return String(s).padStart(w); }

// ============================================================
// Determine max safe N for vector tests
// ============================================================
// With ALLOW_MEMORY_GROWTH=1, WASM memory grows on demand via malloc.
// No need to cap N based on initial memory size.
const SIZES = [100, 1000, 10000, 100000, 1000000];

// ============================================================
// Function metadata
// ============================================================
const FUNCS = [
    { name: "fun1", tier: "Trivial", nArgs: 3, desc: "(a+b)/c" },
    { name: "fun2", tier: "Trivial", nArgs: 2, desc: "a*b+1" },
    { name: "fun3", tier: "Complex", nArgs: 1, desc: "sqrt(a²+1)" },
    { name: "fun4", tier: "Trivial", nArgs: 3, desc: "a>b?c:-c" },
    { name: "fun5", tier: "Complex", nArgs: 3, desc: "sin*cos+exp" },
];

// ============================================================
// TC-BENCH-01: Scalar Call Overhead
// ============================================================
console.log("╔══════════════════════════════════════════════════════════════╗");
console.log("║   Phase 0.1.D — WASM vs JS Performance Benchmark (v1.1)    ║");
console.log("╚══════════════════════════════════════════════════════════════╝\n");

console.log("─── TC-BENCH-01: Scalar Call Overhead (per invocation) ───\n");
console.log("Function       Tier     │   JS (ns) │ WASM (ns) │ Ratio  │ Winner");
console.log("────────────────────────┼───────────┼───────────┼────────┼───────");

for (const f of FUNCS) {
    const jsScalar = JS_S[f.name];
    const wasmScalar = ex[f.name];
    if (!wasmScalar) { console.log(`${f.name}: not exported`); continue; }

    let jsFn, wasmFn;
    if (f.nArgs === 1) {
        jsFn = () => jsScalar(1.5);
        wasmFn = () => wasmScalar(1.5);
    } else if (f.nArgs === 2) {
        jsFn = () => jsScalar(1.5, 2.5);
        wasmFn = () => wasmScalar(1.5, 2.5);
    } else {
        jsFn = () => jsScalar(1.5, 2.5, 3.5);
        wasmFn = () => wasmScalar(1.5, 2.5, 3.5);
    }

    const js_ns = bench(jsFn, 500, 50000);
    const wasm_ns = bench(wasmFn, 500, 50000);
    const ratio = (wasm_ns / js_ns).toFixed(2);
    const winner = wasm_ns < js_ns ? "WASM" : "JS";

    console.log(
        `${(f.name + " " + f.desc).padEnd(22)} ${f.tier.padEnd(8)}│ ${pad(js_ns.toFixed(0), 9)} │ ${pad(wasm_ns.toFixed(0), 9)} │ ${pad(ratio, 6)} │ ${winner}`
    );
}

// ============================================================
// TC-BENCH-02: Vector Throughput (with malloc/free per call)
// ============================================================
console.log(`  Initial WASM memory: ${memory.buffer.byteLength / 1024}KB (grows on demand)\n`);

console.log("\n─── TC-BENCH-02: Vector Throughput (malloc/free per call) ───\n");

for (const f of FUNCS) {
    const wasmVec = ex[f.name + "_v"];
    if (!wasmVec) continue;

    console.log(`  ${f.name} [${f.desc}] (${f.tier}):`);
    console.log(`  ${"N".padStart(8)} │ ${"JS (µs)".padStart(10)} │ ${"WASM (µs)".padStart(10)} │ Ratio  │ Winner`);
    console.log(`  ────────┼────────────┼────────────┼────────┼───────`);

    for (const N of SIZES) {
        const { a, b, c } = makeData(N);
        const out = new Float64Array(N);
        const iters = Math.max(200, Math.floor(2000000 / N));

        // JS vector
        const js_ns = bench(() => JS[f.name](a, b, c, out, N), 50, iters);

        // WASM vector with malloc/free
        const wasm_ns = bench(() => {
            if (hasResetHeap) ex.reset_heap();
            const ptrs = [];
            const nArgs = f.nArgs;
            const arrays = [a, b, c].slice(0, nArgs);
            for (const arr of arrays) {
                const p = ex.malloc(N * 8);
                copyToWasm(arr, p);
                ptrs.push(p);
            }
            const po = ex.malloc(N * 8);
            wasmVec(...ptrs, po, N);
            readFromWasm(po, N);
            if (hasFree && !hasResetHeap) {
                for (const p of ptrs) ex.free(p);
                ex.free(po);
            }
        }, 50, iters);

        const js_us = (js_ns / 1000).toFixed(2);
        const wasm_us = (wasm_ns / 1000).toFixed(2);
        const ratio = (wasm_ns / js_ns).toFixed(2);
        const winner = wasm_ns < js_ns ? "WASM" : "JS";
        console.log(`  ${pad(N, 8)} │ ${pad(js_us, 10)} │ ${pad(wasm_us, 10)} │ ${pad(ratio, 6)} │ ${winner}`);
    }
    console.log("");
}

// ============================================================
// TC-BENCH-03: Vector Throughput (pre-allocated buffers)
// ============================================================
console.log("─── TC-BENCH-03: Vector Throughput (pre-allocated buffers) ───\n");

for (const f of FUNCS) {
    const wasmVec = ex[f.name + "_v"];
    if (!wasmVec) continue;

    console.log(`  ${f.name} [${f.desc}] (${f.tier}):`);
    console.log(`  ${"N".padStart(8)} │ ${"JS (µs)".padStart(10)} │ ${"WASM (µs)".padStart(10)} │ Ratio  │ Winner`);
    console.log(`  ────────┼────────────┼────────────┼────────┼───────`);

    for (const N of SIZES) {
        const { a, b, c } = makeData(N);
        const out = new Float64Array(N);
        const iters = Math.max(200, Math.floor(2000000 / N));

        // JS vector (same as TC-BENCH-02)
        const js_ns = bench(() => JS[f.name](a, b, c, out, N), 50, iters);

        // Pre-allocate WASM buffers ONCE
        if (hasResetHeap) ex.reset_heap();
        const nArgs = f.nArgs;
        const arrays = [a, b, c].slice(0, nArgs);
        const ptrs = arrays.map(() => ex.malloc(N * 8));
        const po = ex.malloc(N * 8);

        // Benchmark: only copy + compute + read (no malloc/free in hot loop)
        const wasm_ns = bench(() => {
            for (let j = 0; j < nArgs; j++) {
                copyToWasm(arrays[j], ptrs[j]);
            }
            wasmVec(...ptrs, po, N);
            readFromWasm(po, N);
        }, 50, iters);

        // Cleanup
        if (hasFree && !hasResetHeap) {
            for (const p of ptrs) ex.free(p);
            ex.free(po);
        }

        const js_us = (js_ns / 1000).toFixed(2);
        const wasm_us = (wasm_ns / 1000).toFixed(2);
        const ratio = (wasm_ns / js_ns).toFixed(2);
        const winner = wasm_ns < js_ns ? "WASM" : "JS";
        console.log(`  ${pad(N, 8)} │ ${pad(js_us, 10)} │ ${pad(wasm_us, 10)} │ ${pad(ratio, 6)} │ ${winner}`);
    }
    console.log("");
}

// ============================================================
// TC-BENCH-04: Memory Copy Fraction
// ============================================================
console.log("─── TC-BENCH-04: Memory Copy Fraction (copy % of total WASM time) ───\n");
console.log(`  ${"Func".padEnd(8)} ${"N".padStart(8)} │ ${"Total (µs)".padStart(10)} │ ${"Compute (µs)".padStart(12)} │ Copy %`);
console.log(`  ────────────────┼────────────┼──────────────┼───────`);

for (const f of FUNCS) {
    const wasmVec = ex[f.name + "_v"];
    if (!wasmVec) continue;

    for (const N of SIZES.filter(n => n <= 10000)) {
        const { a, b, c } = makeData(N);
        const iters = Math.max(500, Math.floor(2000000 / N));
        const nArgs = f.nArgs;
        const arrays = [a, b, c].slice(0, nArgs);

        // Pre-allocate
        if (hasResetHeap) ex.reset_heap();
        const ptrs = arrays.map(() => ex.malloc(N * 8));
        const po = ex.malloc(N * 8);

        // Total: copy + compute + read
        const total_ns = bench(() => {
            for (let j = 0; j < nArgs; j++) copyToWasm(arrays[j], ptrs[j]);
            wasmVec(...ptrs, po, N);
            readFromWasm(po, N);
        }, 50, iters);

        // Compute-only: buffers already written, just call + read
        for (let j = 0; j < nArgs; j++) copyToWasm(arrays[j], ptrs[j]);
        const compute_ns = bench(() => {
            wasmVec(...ptrs, po, N);
            readFromWasm(po, N);
        }, 50, iters);

        if (hasFree && !hasResetHeap) {
            for (const p of ptrs) ex.free(p);
            ex.free(po);
        }

        const copy_pct = (100 * (total_ns - compute_ns) / total_ns).toFixed(1);
        console.log(
            `  ${f.name.padEnd(8)} ${pad(N, 8)} │ ${pad((total_ns / 1000).toFixed(2), 10)} │ ${pad((compute_ns / 1000).toFixed(2), 12)} │ ${pad(copy_pct, 5)}%`
        );
    }
}

// ============================================================
// Summary
// ============================================================
console.log("\n─── Summary ───\n");
console.log("  Ratio < 1.0 = WASM faster, > 1.0 = JS faster");
console.log("  TC-BENCH-02: pessimistic (malloc/free per call)");
console.log("  TC-BENCH-03: production-realistic (pre-allocated buffers)");
console.log("  TC-BENCH-04: shows where optimization effort should go");
console.log("");
console.log("  All benchmarks: Node.js v" + process.version.slice(1));
console.log("  WASM memory: " + (memory.buffer.byteLength / 1024 / 1024).toFixed(1) + " MB (after growth)");
console.log("  Browser performance may vary ±2-3× (V8 vs SpiderMonkey vs JSC)");
