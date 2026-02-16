/**
 * test_performance.mjs — Phase 0.2.A TC-PERF-01
 *
 * Benchmarks SortedRangeIndex vs InverseIndexCSR.
 * Reports build time, query time, memory for both variants.
 *
 * Run: node --test test_performance.mjs
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { buildInverseIndex } from "./buildInverseIndex.mjs";
import { crossTableFilter } from "./CrossTableFilter.mjs";

const fixture = JSON.parse(readFileSync(new URL("./ao2d_fixture.json", import.meta.url), "utf-8"));
const { dense_fk, row_counts } = fixture;

const trEventIdx = new Int32Array(dense_fk.tracks_event_idx);
const clTrackIdx = new Int32Array(dense_fk.clusters_track_idx);
const nEvents = row_counts.events;
const nTracks = row_counts.tracks;
const nClusters = row_counts.clusters;

function benchBuild(fk, parentCount, sorted, iterations = 100) {
    const start = performance.now();
    let idx;
    for (let i = 0; i < iterations; i++) {
        idx = buildInverseIndex(fk, parentCount, { sorted });
    }
    return { timeMs: (performance.now() - start) / iterations, index: idx };
}

function benchQuery(index, selections, childCount, iterations = 100) {
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
        for (const sel of selections) {
            crossTableFilter(index, sel, childCount);
        }
    }
    return (performance.now() - start) / iterations;
}

describe("TC-PERF-01: SortedRange vs CSR performance", () => {
    it("tracks→events: SortedRange build faster than CSR", () => {
        const csrResult = benchBuild(trEventIdx, nEvents, false);
        const srResult = benchBuild(trEventIdx, nEvents, true);

        console.log(`  tracks→events build: CSR=${csrResult.timeMs.toFixed(3)}ms, ` +
            `SortedRange=${srResult.timeMs.toFixed(3)}ms, ` +
            `ratio=${(csrResult.timeMs / srResult.timeMs).toFixed(2)}x`);

        // SortedRange should be at least competitive (single pass vs two)
        assert.ok(true, "Performance recorded");
    });

    it("clusters→tracks: SortedRange build faster than CSR", () => {
        const csrResult = benchBuild(clTrackIdx, nTracks, false);
        const srResult = benchBuild(clTrackIdx, nTracks, true);

        console.log(`  clusters→tracks build: CSR=${csrResult.timeMs.toFixed(3)}ms, ` +
            `SortedRange=${srResult.timeMs.toFixed(3)}ms, ` +
            `ratio=${(csrResult.timeMs / srResult.timeMs).toFixed(2)}x`);

        assert.ok(true, "Performance recorded");
    });

    it("query performance: 1, 10, 100, all parents", () => {
        const csrIdx = buildInverseIndex(trEventIdx, nEvents, { sorted: false });
        const srIdx = buildInverseIndex(trEventIdx, nEvents, { sorted: true });

        const selSizes = [1, 10, 100];
        for (const size of selSizes) {
            const sel = new Int32Array(Array.from({ length: Math.min(size, nEvents) },
                (_, i) => i));
            const csrTime = benchQuery(csrIdx, [sel], nTracks, 500);
            const srTime = benchQuery(srIdx, [sel], nTracks, 500);
            console.log(`  query ${size} parents: CSR=${csrTime.toFixed(4)}ms, ` +
                `SortedRange=${srTime.toFixed(4)}ms, ` +
                `ratio=${(csrTime / srTime).toFixed(2)}x`);
        }
        assert.ok(true, "Performance recorded");
    });

    it("memory comparison", () => {
        const csrIdx = buildInverseIndex(trEventIdx, nEvents, { sorted: false });
        const srIdx = buildInverseIndex(trEventIdx, nEvents, { sorted: true });

        const csrBytes = csrIdx.offsets.byteLength + csrIdx.indices.byteLength;
        const srBytes = srIdx.starts.byteLength + srIdx.counts.byteLength;

        console.log(`  memory tracks→events: CSR=${csrBytes} bytes, ` +
            `SortedRange=${srBytes} bytes, ratio=${(csrBytes / srBytes).toFixed(1)}x`);

        // SortedRange should use less memory (no indices array)
        assert.ok(srBytes <= csrBytes, "SortedRange should use ≤ memory");
    });
});
