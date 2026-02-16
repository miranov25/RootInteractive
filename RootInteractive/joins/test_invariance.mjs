/**
 * test_invariance.mjs — Phase 0.2.A Cross-Backend Invariance Tests (MERGE GATE)
 *
 * Loads ao2d_fixture.json (Python oracle) and validates that JS
 * buildInverseIndex + CrossTableFilter produce identical results.
 *
 * If any test fails, Phase 0.2.A cannot merge.
 *
 * Run: node --test test_invariance.mjs
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { buildInverseIndex, childrenOf } from "./buildInverseIndex.mjs";
import { crossTableFilter, maskToIndices } from "./CrossTableFilter.mjs";

// Load fixture
const fixture = JSON.parse(readFileSync(new URL("./ao2d_fixture.json", import.meta.url), "utf-8"));
const { tables, dense_fk, oracle, row_counts } = fixture;

// Convert dense FK arrays to Int32Array
const trEventIdx = new Int32Array(dense_fk.tracks_event_idx);
const clTrackIdx = new Int32Array(dense_fk.clusters_track_idx);
const clEventIdx = new Int32Array(dense_fk.clusters_event_idx);
const mcTrackIdx = new Int32Array(tables.mc_labels.mc_track_idx);

const nEvents = row_counts.events;
const nTracks = row_counts.tracks;
const nClusters = row_counts.clusters;
const nReco = row_counts.reco_tracks;
const nLabels = row_counts.mc_labels;

// Helper: compare sorted arrays
function assertSortedArraysEqual(actual, expected, msg) {
    const a = Array.from(actual).sort((x, y) => x - y);
    const e = Array.from(expected).sort((x, y) => x - y);
    assert.deepStrictEqual(a, e, msg);
}

// Helper: compare masks
function assertMasksEqual(actual, expected, msg) {
    assert.equal(actual.length, expected.length, `${msg}: length mismatch`);
    for (let i = 0; i < actual.length; i++) {
        assert.equal(actual[i], expected[i], `${msg}: mismatch at index ${i}`);
    }
}

// ================================================================
// TC-INV-01: InverseIndexCSR — tracks_event_idx → events
// ================================================================
describe("TC-INV-01: InverseIndexCSR basic correctness", () => {
    const csrIndex = buildInverseIndex(trEventIdx, nEvents, { sorted: false });
    const oracleData = oracle["TC-INV-01"];

    it("total children matches oracle", () => {
        let total = 0;
        for (let j = 0; j < nEvents; j++) {
            total += csrIndex.offsets[j + 1] - csrIndex.offsets[j];
        }
        assert.equal(total, oracleData.total_children);
    });

    it("children_of(j) matches oracle for every parent", () => {
        for (let j = 0; j < nEvents; j++) {
            const jsChildren = Array.from(childrenOf(csrIndex, j));
            const pyChildren = oracleData.children_per_parent[String(j)];
            assertSortedArraysEqual(jsChildren, pyChildren,
                `event ${j} children mismatch`);
        }
    });

    it("offsets match oracle", () => {
        assert.deepStrictEqual(
            Array.from(csrIndex.offsets),
            oracleData.offsets,
        );
    });
});

// ================================================================
// TC-INV-02: SortedRangeIndex — tracks_event_idx (sorted)
// ================================================================
describe("TC-INV-02: SortedRangeIndex correctness", () => {
    const sortedIndex = buildInverseIndex(trEventIdx, nEvents, { sorted: true });
    const oracleData = oracle["TC-INV-02"];

    it("starts match oracle", () => {
        assert.deepStrictEqual(
            Array.from(sortedIndex.starts),
            oracleData.starts,
        );
    });

    it("counts match oracle", () => {
        assert.deepStrictEqual(
            Array.from(sortedIndex.counts),
            oracleData.counts,
        );
    });

    it("children_of(j) matches oracle for every parent", () => {
        for (let j = 0; j < nEvents; j++) {
            const jsChildren = Array.from(childrenOf(sortedIndex, j));
            const pyChildren = oracleData.children_per_parent[String(j)];
            assertSortedArraysEqual(jsChildren, pyChildren,
                `event ${j} children mismatch (SortedRange)`);
        }
    });
});

// ================================================================
// TC-INV-03: CSR vs SortedRange invariance
// ================================================================
describe("TC-INV-03: CSR vs SortedRange produce identical child sets", () => {
    const csrIndex = buildInverseIndex(trEventIdx, nEvents, { sorted: false });
    const sortedIndex = buildInverseIndex(trEventIdx, nEvents, { sorted: true });
    const canonical = oracle["TC-INV-03"].canonical_children;

    it("both variants match Python oracle for every parent", () => {
        for (let j = 0; j < nEvents; j++) {
            const csrChildren = Array.from(childrenOf(csrIndex, j)).sort((a, b) => a - b);
            const sortedChildren = Array.from(childrenOf(sortedIndex, j)).sort((a, b) => a - b);
            const pyChildren = canonical[String(j)];

            assert.deepStrictEqual(csrChildren, pyChildren,
                `event ${j}: CSR doesn't match oracle`);
            assert.deepStrictEqual(sortedChildren, pyChildren,
                `event ${j}: SortedRange doesn't match oracle`);
        }
    });
});

// ================================================================
// TC-INV-04: Sentinel handling (efficiency holes)
// ================================================================
describe("TC-INV-04: Sentinel handling — efficiency holes", () => {
    const csrIndex = buildInverseIndex(mcTrackIdx, nTracks, { sorted: false });
    const oracleData = oracle["TC-INV-04"];

    it("number of matched MC tracks matches oracle", () => {
        // Count parents with >0 children
        let matched = 0;
        for (let j = 0; j < nTracks; j++) {
            if (csrIndex.offsets[j + 1] > csrIndex.offsets[j]) {
                matched++;
            }
        }
        assert.equal(matched, oracleData.n_matched);
    });

    it("number of unmatched MC tracks matches oracle", () => {
        let unmatched = 0;
        for (let j = 0; j < nTracks; j++) {
            if (csrIndex.offsets[j + 1] === csrIndex.offsets[j]) {
                unmatched++;
            }
        }
        assert.equal(unmatched, oracleData.n_unmatched);
    });

    it("unmatched track indices match oracle", () => {
        const jsUnmatched = [];
        for (let j = 0; j < nTracks; j++) {
            if (csrIndex.offsets[j + 1] === csrIndex.offsets[j]) {
                jsUnmatched.push(j);
            }
        }
        assert.deepStrictEqual(jsUnmatched, oracleData.unmatched_mc_tracks);
    });
});

// ================================================================
// TC-INV-05: Empty parent (0 children)
// ================================================================
describe("TC-INV-05: Empty parent handling", () => {
    const csrIndex = buildInverseIndex(mcTrackIdx, nTracks, { sorted: false });
    const oracleData = oracle["TC-INV-05"];

    it("empty tracks count matches oracle", () => {
        let emptyCount = 0;
        for (let j = 0; j < nTracks; j++) {
            if (csrIndex.offsets[j + 1] === csrIndex.offsets[j]) {
                emptyCount++;
            }
        }
        assert.equal(emptyCount, oracleData.n_empty_tracks_in_mc_labels);
    });

    it("children_of(empty_parent) returns empty array", () => {
        const emptyTracks = oracleData.empty_tracks_in_mc_labels;
        for (const j of emptyTracks) {
            const children = childrenOf(csrIndex, j);
            assert.equal(children.length, 0, `track ${j} should have 0 children`);
        }
    });
});

// ================================================================
// TC-CTF-01: Single parent selection — event 5
// ================================================================
describe("TC-CTF-01: Single parent selection", () => {
    const csrIndex = buildInverseIndex(trEventIdx, nEvents, { sorted: false });
    const oracleData = oracle["TC-CTF-01"];

    it("mask matches oracle", () => {
        const mask = crossTableFilter(csrIndex, new Int32Array([5]), nTracks);
        assertMasksEqual(mask, oracleData.expected_mask, "CTF-01 mask");
    });

    it("selected track indices match oracle", () => {
        const mask = crossTableFilter(csrIndex, new Int32Array([5]), nTracks);
        const indices = Array.from(maskToIndices(mask));
        assert.deepStrictEqual(indices, oracleData.expected_track_indices);
    });
});

// ================================================================
// TC-CTF-02: Range selection — events 0–9
// ================================================================
describe("TC-CTF-02: Range selection", () => {
    const csrIndex = buildInverseIndex(trEventIdx, nEvents, { sorted: false });
    const oracleData = oracle["TC-CTF-02"];

    it("mask matches oracle", () => {
        const sel = new Int32Array(oracleData.selected_events);
        const mask = crossTableFilter(csrIndex, sel, nTracks);
        assertMasksEqual(mask, oracleData.expected_mask, "CTF-02 mask");
    });

    it("selected track indices match oracle", () => {
        const sel = new Int32Array(oracleData.selected_events);
        const mask = crossTableFilter(csrIndex, sel, nTracks);
        const indices = Array.from(maskToIndices(mask));
        assert.deepStrictEqual(indices, oracleData.expected_track_indices);
    });
});

// ================================================================
// TC-CTF-03: Select all → all tracks
// ================================================================
describe("TC-CTF-03: Select all parents", () => {
    const csrIndex = buildInverseIndex(trEventIdx, nEvents, { sorted: false });
    const oracleData = oracle["TC-CTF-03"];

    it("all tracks selected", () => {
        const sel = new Int32Array(Array.from({ length: nEvents }, (_, i) => i));
        const mask = crossTableFilter(csrIndex, sel, nTracks);
        const nSelected = mask.reduce((s, v) => s + v, 0);
        assert.equal(nSelected, oracleData.n_selected_tracks);
    });

    it("mask matches oracle", () => {
        const sel = new Int32Array(Array.from({ length: nEvents }, (_, i) => i));
        const mask = crossTableFilter(csrIndex, sel, nTracks);
        assertMasksEqual(mask, oracleData.expected_mask, "CTF-03 mask");
    });
});

// ================================================================
// TC-CTF-04: Select none → empty mask (JS-only, no oracle needed)
// ================================================================
describe("TC-CTF-04: Select none", () => {
    const csrIndex = buildInverseIndex(trEventIdx, nEvents, { sorted: false });

    it("empty selection produces all-zero mask", () => {
        const mask = crossTableFilter(csrIndex, new Int32Array(0), nTracks);
        assert.equal(mask.length, nTracks);
        assert.equal(mask.reduce((s, v) => s + v, 0), 0);
    });
});

// ================================================================
// TC-CTF-05: Cascade — event 5 → tracks → clusters
// ================================================================
describe("TC-CTF-05: 3-level cascade", () => {
    const invTE = buildInverseIndex(trEventIdx, nEvents, { sorted: false });
    const invCT = buildInverseIndex(clTrackIdx, nTracks, { sorted: false });
    const oracleData = oracle["TC-CTF-05"];

    it("intermediate tracks match oracle", () => {
        const maskTracks = crossTableFilter(invTE, new Int32Array([5]), nTracks);
        const trackIndices = Array.from(maskToIndices(maskTracks));
        assert.deepStrictEqual(trackIndices, oracleData.intermediate_track_indices);
    });

    it("cascade cluster mask matches oracle", () => {
        const maskTracks = crossTableFilter(invTE, new Int32Array([5]), nTracks);
        const selectedTracks = maskToIndices(maskTracks);
        const maskClusters = crossTableFilter(invCT, selectedTracks, nClusters);
        assertMasksEqual(maskClusters, oracleData.expected_cluster_mask, "CTF-05 clusters");
    });

    it("cascade cluster count matches oracle", () => {
        const maskTracks = crossTableFilter(invTE, new Int32Array([5]), nTracks);
        const selectedTracks = maskToIndices(maskTracks);
        const maskClusters = crossTableFilter(invCT, selectedTracks, nClusters);
        const nCls = maskClusters.reduce((s, v) => s + v, 0);
        assert.equal(nCls, oracleData.n_selected_clusters);
    });
});

// ================================================================
// TC-CTF-06: M:N via junction (mc_labels)
// ================================================================
describe("TC-CTF-06: M:N via junction table", () => {
    // mc_labels.mc_track_idx → tracks (inverse: tracks → mc_label rows)
    const invML = buildInverseIndex(mcTrackIdx, nTracks, { sorted: false });
    const oracleData = oracle["TC-CTF-06"];

    it("label mask matches oracle", () => {
        const sel = new Int32Array(oracleData.selected_mc_tracks);
        const mask = crossTableFilter(invML, sel, nLabels);
        assertMasksEqual(mask, oracleData.expected_label_mask, "CTF-06 label mask");
    });

    it("reached reco_track_ids match oracle", () => {
        const sel = new Int32Array(oracleData.selected_mc_tracks);
        const mask = crossTableFilter(invML, sel, nLabels);
        const survivingIdx = Array.from(maskToIndices(mask));
        const recoIds = survivingIdx.map(i => tables.mc_labels.reco_track_id[i]);
        assert.deepStrictEqual(recoIds.sort((a, b) => a - b),
            oracleData.reached_reco_track_ids);
    });
});

// ================================================================
// TC-CTF-08: Round-trip invariance
// ================================================================
describe("TC-CTF-08: Round-trip invariance", () => {
    const csrIndex = buildInverseIndex(trEventIdx, nEvents, { sorted: false });
    const oracleData = oracle["TC-CTF-08"];

    it("select all events → all tracks", () => {
        const sel = new Int32Array(Array.from({ length: nEvents }, (_, i) => i));
        const mask = crossTableFilter(csrIndex, sel, nTracks);
        const nSelected = mask.reduce((s, v) => s + v, 0);
        assert.equal(nSelected, oracleData.n_total_tracks);
        assert.equal(oracleData.all_selected, true);
    });

    it("gather event_id from filtered tracks matches selection", () => {
        const sel = new Int32Array([5, 10, 15]);
        const mask = crossTableFilter(csrIndex, sel, nTracks);
        const selectedTrackIdx = maskToIndices(mask);
        // Gather event_id for each selected track via dense FK
        const eventIds = new Set();
        for (const ti of selectedTrackIdx) {
            eventIds.add(trEventIdx[ti]);
        }
        // All gathered event_ids must be in the selection
        for (const eid of eventIds) {
            assert.ok(sel.includes(eid),
                `track's event_id ${eid} not in selection`);
        }
    });
});
