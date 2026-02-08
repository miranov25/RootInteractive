/**
 * Phase 0.1.D v1.1 — WASM Cross-Backend Invariance Test Runner
 *
 * Receives JSON from Python with:
 *   - func/func_v: WASM function names (scalar/vector)
 *   - inputs_b64: base64-encoded Float64Arrays
 *   - ref_b64: base64-encoded reference (computed by numpy)
 *   - abs_tol, rel_tol: per-function tolerance
 *
 * Evaluates in WASM, compares with numpy reference.
 * Pattern follows Phase 0.1.B test_serialization_integration.mjs
 *
 * Usage:
 *   node test_wasm_cross_backend.mjs <wasm_path> <json_path> [--consistency]
 *
 * v1.1 changes:
 *   - Explicit free() after vector operations [Gemini1 P0-1]
 *   - --consistency flag: compare scalar vs vector results (TC-WASM-11)
 *   - Support for fun5 (5 functions total)
 */
import fs from "fs";

// ============================================================
// Helpers
// ============================================================
function decodeFloat64(b64) {
    const buf = Buffer.from(b64, 'base64');
    return new Float64Array(buf.buffer, buf.byteOffset, buf.length / 8);
}

function copyToWasm(memory, arr, ptr) {
    new Float64Array(memory.buffer, ptr, arr.length).set(arr);
}

function readFromWasm(memory, ptr, len) {
    return new Float64Array(memory.buffer.slice(ptr, ptr + len * 8));
}

function allclose(A, B, abs_tol, rel_tol) {
    if (A.length !== B.length) return { ok: false, msg: `length: ${A.length} vs ${B.length}` };
    for (let i = 0; i < A.length; i++) {
        const a = A[i], b = B[i];
        // NaN == NaN: both NaN → pass
        if (Number.isNaN(a) && Number.isNaN(b)) continue;
        // One NaN, one not → fail
        if (Number.isNaN(a) || Number.isNaN(b))
            return { ok: false, msg: `[${i}]: ${a} vs ${b} (NaN mismatch)` };
        // Inf: must match exactly (sign matters)
        if (!Number.isFinite(a) || !Number.isFinite(b)) {
            if (a !== b) return { ok: false, msg: `[${i}]: ${a} vs ${b} (Inf mismatch)` };
            continue;
        }
        // Finite: abs + rel tolerance
        const diff = Math.abs(a - b);
        if (diff > abs_tol && diff > rel_tol * Math.max(Math.abs(a), Math.abs(b)))
            return { ok: false, msg: `[${i}]: ${a} vs ${b}, diff=${diff}` };
    }
    return { ok: true };
}

// ============================================================
// Load WASM & test data
// ============================================================
const wasmPath = process.argv[2];
const jsonPath = process.argv[3];
const consistencyMode = process.argv.includes('--consistency');

const wasmBytes = fs.readFileSync(wasmPath);

// Emscripten imports: ALLOW_MEMORY_GROWTH requires emscripten_notify_memory_growth stub
const imports = {
    env: {
        emscripten_notify_memory_growth: () => {},
    },
    wasi_snapshot_preview1: {},
};
const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
const ex = instance.exports;
const memory = ex.memory;

// Check for malloc/free (emscripten provides these; WAT prototype may not)
const hasMalloc = typeof ex.malloc === 'function';
const hasFree = typeof ex.free === 'function';
const hasResetHeap = typeof ex.reset_heap === 'function'; // WAT prototype only

const testCases = JSON.parse(fs.readFileSync(jsonPath, "utf8"));

// ============================================================
// Execute tests
// ============================================================
let passed = 0, failed = 0;
const results = {}; // For consistency check: { "fun1": { scalar: [...], vector: [...] } }

for (const tc of testCases) {
    const { func, func_v, mode, fields, inputs_b64, ref_b64, length, abs_tol, rel_tol, description } = tc;

    // Decode reference (from numpy)
    const ref = decodeFloat64(ref_b64);

    // Decode inputs
    const inputs = {};
    for (const f of fields) {
        inputs[f] = decodeFloat64(inputs_b64[f]);
    }

    let result;
    const label = `${mode} ${func} [${description}]`;

    if (mode === "scalar") {
        // Call scalar WASM function per element
        const scalarFunc = ex[func];
        if (!scalarFunc) { console.log(`  ⚠️  ${label}: function not exported`); continue; }
        result = new Float64Array(length);
        for (let i = 0; i < length; i++) {
            const args = fields.map(f => inputs[f][i]);
            result[i] = scalarFunc(...args);
        }
    } else if (mode === "vector") {
        // Call vector WASM function with memory pointers
        const vectorFunc = ex[func_v];
        if (!vectorFunc) { console.log(`  ⚠️  ${label}: function not exported`); continue; }

        // Memory management: reset_heap (WAT) or malloc/free (emscripten)
        if (hasResetHeap) {
            ex.reset_heap();
        }

        const ptrs = [];
        for (const f of fields) {
            const ptr = ex.malloc(length * 8);
            copyToWasm(memory, inputs[f], ptr);
            ptrs.push(ptr);
        }
        const outPtr = ex.malloc(length * 8);

        vectorFunc(...ptrs, outPtr, length);
        result = readFromWasm(memory, outPtr, length);

        // v1.1: Explicit free() [Gemini1 P0-1]
        if (hasFree && !hasResetHeap) {
            for (const ptr of ptrs) ex.free(ptr);
            ex.free(outPtr);
        }
    }

    // Compare with numpy reference
    const check = allclose(result, ref, abs_tol, rel_tol);
    if (check.ok) {
        console.log(`  ✅ ${label}`);
        passed++;
    } else {
        console.log(`  ❌ ${label}: ${check.msg}`);
        failed++;
    }

    // Store for consistency check
    if (consistencyMode) {
        if (!results[func]) results[func] = {};
        results[func][mode] = result;
    }
}

// ============================================================
// Consistency check: scalar == vector (TC-WASM-11)
// ============================================================
if (consistencyMode) {
    console.log("\n--- Scalar vs Vector Consistency ---");
    for (const [func, modes] of Object.entries(results)) {
        if (!modes.scalar || !modes.vector) {
            console.log(`  ⚠️  ${func}: missing ${modes.scalar ? 'vector' : 'scalar'} result`);
            continue;
        }
        // Exact match required (same WASM code, same inputs)
        const check = allclose(modes.scalar, modes.vector, 0, 0);
        if (check.ok) {
            console.log(`  ✅ ${func}: scalar == vector (bit-identical)`);
            passed++;
        } else {
            console.log(`  ❌ ${func}: scalar != vector: ${check.msg}`);
            failed++;
        }
    }
}

// ============================================================
// Summary
// ============================================================
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
