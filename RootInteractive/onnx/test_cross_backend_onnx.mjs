/**
 * Phase 0.1.E v1.1 — Node.js ONNX Cross-Backend Runner
 *
 * Loads an ONNX model via onnxruntime-node, runs inference on test vectors,
 * and writes results to JSON for comparison with Python ORT.
 *
 * Usage:
 *   node test_onnx_cross_backend.mjs config.json
 *
 * Config JSON:
 *   { model_path, input_b64, N, n_features, output_path, task }
 *
 * Output JSON:
 *   { outputs: {name: base64_float32}, N, output_names }
 */

import * as ort from 'onnxruntime-node';
import { readFileSync, writeFileSync } from 'fs';
import { resolve } from 'path';

// ============================================================
// Safe base64 → Float32Array decoding (v1.1, GPT10 P0)
// Handles non-zero byteOffset from Buffer.from()
// ============================================================

function decodeFloat32(b64string, expectedLength) {
    const buf = Buffer.from(b64string, 'base64');
    // Safe slice — handles non-zero byteOffset
    const arrayBuf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    const result = new Float32Array(arrayBuf);
    if (result.length !== expectedLength) {
        throw new Error(`Expected ${expectedLength} float32 values, got ${result.length}`);
    }
    return result;
}

// ============================================================
// ONNX Inference
// ============================================================

async function runInference(modelPath, inputData, N, nFeatures) {
    // Single-threaded to simulate browser (v1.1, Gemini1 P1)
    const sessionOptions = { intraOpNumThreads: 1 };
    const session = await ort.InferenceSession.create(modelPath, sessionOptions);

    // Dynamic input/output names (v1.1, Claude20 P1-3)
    const inputName = session.inputNames[0];
    const outputNames = session.outputNames;

    // Input tensor — float32, shape [N, nFeatures]
    const tensor = new ort.Tensor('float32', inputData, [N, nFeatures]);
    const results = await session.run({ [inputName]: tensor });

    // Extract outputs
    const outputs = {};
    for (const name of outputNames) {
        const data = results[name].data;
        // Handle BigInt64Array (classification labels)
        if (data instanceof BigInt64Array) {
            // Convert to regular numbers, then to Float32Array for uniform base64 encoding
            const numbers = Array.from(data).map(x => Number(x));
            outputs[name] = new Float32Array(numbers);
        } else if (data instanceof Float32Array) {
            outputs[name] = data;
        } else if (data instanceof Float64Array) {
            outputs[name] = new Float32Array(data);
        } else {
            outputs[name] = new Float32Array(Array.from(data));
        }
    }

    // Session cleanup (v1.1, Claude3 P1-2)
    session.release();

    return { outputs, outputNames };
}

// ============================================================
// Main
// ============================================================

async function main() {
    const configPath = process.argv[2];
    if (!configPath) {
        console.error('Usage: node test_onnx_cross_backend.mjs config.json');
        process.exit(1);
    }

    const config = JSON.parse(readFileSync(configPath, 'utf-8'));
    const modelPath = resolve(config.model_path);
    const N = config.N;
    const nFeatures = config.n_features || 4;

    // Decode input
    const inputFloat32 = decodeFloat32(config.input_b64, N * nFeatures);

    // Run inference
    const { outputs, outputNames } = await runInference(modelPath, inputFloat32, N, nFeatures);

    // Encode outputs as base64
    const outputsB64 = {};
    for (const [name, data] of Object.entries(outputs)) {
        outputsB64[name] = Buffer.from(data.buffer, data.byteOffset, data.byteLength).toString('base64');
    }

    // Write result
    const result = {
        outputs: outputsB64,
        N: N,
        output_names: outputNames,
        onnxruntime_version: ort.env ? 'onnxruntime-node' : 'unknown',
    };

    if (config.output_path) {
        writeFileSync(config.output_path, JSON.stringify(result, null, 2));
    } else {
        console.log(JSON.stringify(result));
    }
}

main().catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
});
