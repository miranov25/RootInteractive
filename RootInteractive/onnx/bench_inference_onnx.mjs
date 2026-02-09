/**
 * Phase 0.1.E v1.1 — ONNX Performance Benchmarks
 *
 * Measures model load time and inference latency for all 7 models
 * at batch sizes N = 1 to 1M. Single-threaded (intraOpNumThreads: 1).
 *
 * Usage:
 *   node bench_onnx.mjs                    # Run all benchmarks
 *   node bench_onnx.mjs --model ridge      # Single model
 *   node bench_onnx.mjs --json             # Output as JSON
 *
 * Output: bench_results/bench_onnx_results.json
 */

import * as ort from 'onnxruntime-node';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = resolve(__dirname, 'models');
const RESULTS_DIR = resolve(__dirname, 'bench_results');

// ============================================================
// Configuration
// ============================================================

const MODELS = {
    ridge:    { file: 'ridge_4feat.onnx',    task: 'regression',      desc: 'Ridge(α=0.1) — linear, 4 weights + bias' },
    rf10:     { file: 'rf10_4feat.onnx',     task: 'regression',      desc: 'RandomForest(n_trees=10, max_depth=3) — tree ensemble' },
    rf50:     { file: 'rf50_4feat.onnx',     task: 'regression',      desc: 'RandomForest(n_trees=50, max_depth=5) — tree ensemble' },
    mlp:      { file: 'mlp_4feat.onnx',      task: 'regression',      desc: 'MLP(hidden=[100,50]) — 2-layer neural net' },
    logistic: { file: 'logistic_4feat.onnx', task: 'classification',  desc: 'LogisticRegression — linear classifier, 4 weights + bias' },
    rf_clf:   { file: 'rf_clf_4feat.onnx',   task: 'classification',  desc: 'RandomForestClassifier(n_trees=10, max_depth=3) — tree ensemble' },
    mlp_clf:  { file: 'mlp_clf_4feat.onnx',  task: 'classification',  desc: 'MLPClassifier(hidden=[50,25]) — 2-layer neural net' },
};

const N_FEATURES = 4;

// Adaptive iteration count (handoff prompt: fewer at large N)
const BATCH_CONFIG = [
    { N: 1,        warmup: 5,  iterations: 20 },
    { N: 1_000,    warmup: 5,  iterations: 20 },
    { N: 10_000,   warmup: 5,  iterations: 20 },
    { N: 100_000,  warmup: 3,  iterations: 10 },
    { N: 1_000_000, warmup: 2, iterations: 5 },
];

// Performance budget anchors (from Strategic Vision §4.3)
const BUDGET_INTERACTIVE_MS = 75;   // <0.3s / 4 calls
const BUDGET_BATCH_MS = 1000;       // <5s / 5 calls

// ============================================================
// Helpers
// ============================================================

function generateInput(N) {
    const data = new Float32Array(N * N_FEATURES);
    // Deterministic pseudo-random (simple LCG, same across runs)
    let seed = 12345;
    for (let i = 0; i < data.length; i++) {
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF;
        data[i] = (seed >>> 0) / 0xFFFFFFFF;
    }
    return data;
}

function median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function budgetLabel(medianMs) {
    if (medianMs < BUDGET_INTERACTIVE_MS) return '✅ Interactive';
    if (medianMs < BUDGET_BATCH_MS) return '⚠️ Batch only';
    return '❌ Too slow';
}

// ============================================================
// Benchmarks
// ============================================================

async function benchLoadTime(modelName, modelPath) {
    // Cold load
    const t0 = performance.now();
    const session1 = await ort.InferenceSession.create(modelPath, { intraOpNumThreads: 1 });
    const coldMs = performance.now() - t0;
    session1.release();

    // Warm load (cached)
    const t1 = performance.now();
    const session2 = await ort.InferenceSession.create(modelPath, { intraOpNumThreads: 1 });
    const warmMs = performance.now() - t1;
    session2.release();

    return { cold_ms: coldMs, warm_ms: warmMs };
}

async function benchInference(modelName, modelPath, batchConfig) {
    const { N, warmup, iterations } = batchConfig;
    const inputData = generateInput(N);

    const session = await ort.InferenceSession.create(modelPath, { intraOpNumThreads: 1 });
    const inputName = session.inputNames[0];
    const tensor = new ort.Tensor('float32', inputData, [N, N_FEATURES]);

    // Warmup
    for (let i = 0; i < warmup; i++) {
        await session.run({ [inputName]: tensor });
    }

    // Measure
    const times = [];
    for (let i = 0; i < iterations; i++) {
        const t0 = performance.now();
        await session.run({ [inputName]: tensor });
        times.push(performance.now() - t0);
    }

    session.release();

    const medMs = median(times);
    return {
        N,
        iterations,
        median_ms: medMs,
        mean_ms: mean(times),
        min_ms: Math.min(...times),
        max_ms: Math.max(...times),
        budget: budgetLabel(medMs),
    };
}

// ============================================================
// Main
// ============================================================

async function main() {
    const args = process.argv.slice(2);
    const singleModel = args.includes('--model') ? args[args.indexOf('--model') + 1] : null;
    const jsonOutput = args.includes('--json');

    mkdirSync(RESULTS_DIR, { recursive: true });

    const modelsToRun = singleModel ? { [singleModel]: MODELS[singleModel] } : MODELS;
    if (singleModel && !MODELS[singleModel]) {
        console.error(`Unknown model: ${singleModel}. Available: ${Object.keys(MODELS).join(', ')}`);
        process.exit(1);
    }

    const results = { timestamp: new Date().toISOString(), benchmarks: {} };

    console.log('Phase 0.1.E — ONNX Performance Benchmarks');
    console.log(`Models: ${Object.keys(modelsToRun).join(', ')}`);
    console.log(`Thread config: intraOpNumThreads=1 (single-threaded, browser simulation)`);
    console.log(`Budget: Interactive < ${BUDGET_INTERACTIVE_MS}ms, Batch < ${BUDGET_BATCH_MS}ms`);
    console.log('');

    // TC-BENCH-ONNX-01/02: Load time
    console.log('=== TC-BENCH-ONNX-01/02: Model Load Time ===');
    console.log(`${'Model'.padEnd(12)} | ${'Cold (ms)'.padStart(10)} | ${'Warm (ms)'.padStart(10)} | Description`);
    console.log('-'.repeat(90));

    for (const [name, config] of Object.entries(modelsToRun)) {
        const modelPath = resolve(MODELS_DIR, config.file);
        const loadResult = await benchLoadTime(name, modelPath);
        results.benchmarks[name] = { desc: config.desc, load: loadResult, inference: [] };
        console.log(`${name.padEnd(12)} | ${loadResult.cold_ms.toFixed(1).padStart(10)} | ${loadResult.warm_ms.toFixed(1).padStart(10)} | ${config.desc}`);
    }

    // TC-BENCH-ONNX-03/04: Inference latency
    console.log('');
    console.log('=== TC-BENCH-ONNX-03/04: Inference Latency ===');
    console.log(`${'Model'.padEnd(12)} | ${'N'.padStart(8)} | ${'Median (ms)'.padStart(12)} | ${'Mean (ms)'.padStart(10)} | ${'Min'.padStart(8)} | ${'Max'.padStart(8)} | Budget`);
    console.log('-'.repeat(85));

    for (const [name, config] of Object.entries(modelsToRun)) {
        const modelPath = resolve(MODELS_DIR, config.file);
        console.log(`# ${name}: ${config.desc}`);
        for (const batchConf of BATCH_CONFIG) {
            const inferResult = await benchInference(name, modelPath, batchConf);
            results.benchmarks[name].inference.push(inferResult);
            console.log(
                `${name.padEnd(12)} | ${String(inferResult.N).padStart(8)} | ` +
                `${inferResult.median_ms.toFixed(2).padStart(12)} | ` +
                `${inferResult.mean_ms.toFixed(2).padStart(10)} | ` +
                `${inferResult.min_ms.toFixed(2).padStart(8)} | ` +
                `${inferResult.max_ms.toFixed(2).padStart(8)} | ` +
                `${inferResult.budget}`
            );
        }
        console.log('');
    }

    // Save results
    const outPath = resolve(RESULTS_DIR, 'bench_onnx_results.json');
    writeFileSync(outPath, JSON.stringify(results, null, 2));
    console.log(`Results saved: ${outPath}`);
}

main().catch(err => {
    console.error('Benchmark error:', err.message);
    process.exit(1);
});
