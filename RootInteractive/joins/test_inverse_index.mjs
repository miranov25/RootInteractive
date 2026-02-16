/**
 * test_inverse_index.mjs — Phase 0.2.A
 *
 * JS-internal inverse index tests. These validate JS consistency
 * without requiring the Python oracle.
 *
 * Run: node --test test_inverse_index.mjs
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { buildInverseIndex, childrenOf } from "./buildInverseIndex.mjs";

describe("buildInverseIndex — type validation", () => {
    it("throws TypeError for non-Int32Array", () => {
        assert.throws(
            () => buildInverseIndex([1, 2, 3], 5),
            { name: "TypeError" },
        );
        assert.throws(
            () => buildInverseIndex(new Float64Array([1, 2]), 5),
            { name: "TypeError" },
        );
    });

    it("throws RangeError for negative parentCount", () => {
        assert.throws(
            () => buildInverseIndex(new Int32Array([0]), -1),
            { name: "RangeError" },
        );
    });
});

describe("buildInverseIndex — small examples", () => {
    // 3 parents, 6 children: [0,0,1,1,1,2]
    const fk = new Int32Array([0, 0, 1, 1, 1, 2]);

    it("CSR: correct children per parent", () => {
        const idx = buildInverseIndex(fk, 3, { sorted: false });
        assert.equal(idx.type, "csr");
        assert.deepStrictEqual(Array.from(childrenOf(idx, 0)), [0, 1]);
        assert.deepStrictEqual(Array.from(childrenOf(idx, 1)), [2, 3, 4]);
        assert.deepStrictEqual(Array.from(childrenOf(idx, 2)), [5]);
    });

    it("SortedRange: correct children per parent", () => {
        const idx = buildInverseIndex(fk, 3, { sorted: true });
        assert.equal(idx.type, "sorted_range");
        assert.deepStrictEqual(Array.from(childrenOf(idx, 0)), [0, 1]);
        assert.deepStrictEqual(Array.from(childrenOf(idx, 1)), [2, 3, 4]);
        assert.deepStrictEqual(Array.from(childrenOf(idx, 2)), [5]);
    });

    it("CSR and SortedRange give same children", () => {
        const csr = buildInverseIndex(fk, 3, { sorted: false });
        const sr = buildInverseIndex(fk, 3, { sorted: true });
        for (let j = 0; j < 3; j++) {
            const cc = Array.from(childrenOf(csr, j)).sort((a, b) => a - b);
            const sc = Array.from(childrenOf(sr, j)).sort((a, b) => a - b);
            assert.deepStrictEqual(cc, sc, `parent ${j}`);
        }
    });
});

describe("buildInverseIndex — sentinel handling", () => {
    // Sentinels: -1, and value >= parentCount
    const fk = new Int32Array([-1, 0, 1, -1, 2, 99]);

    it("CSR: sentinels excluded", () => {
        const idx = buildInverseIndex(fk, 3, { sorted: false });
        assert.deepStrictEqual(Array.from(childrenOf(idx, 0)), [1]);
        assert.deepStrictEqual(Array.from(childrenOf(idx, 1)), [2]);
        assert.deepStrictEqual(Array.from(childrenOf(idx, 2)), [4]);
        // Total valid = 3 (out of 6)
        const total = idx.offsets[3] - idx.offsets[0];
        assert.equal(total, 3);
    });
});

describe("buildInverseIndex — empty cases", () => {
    it("empty fkArray", () => {
        const idx = buildInverseIndex(new Int32Array(0), 5);
        for (let j = 0; j < 5; j++) {
            assert.equal(childrenOf(idx, j).length, 0);
        }
    });

    it("zero parentCount", () => {
        const idx = buildInverseIndex(new Int32Array(0), 0);
        assert.equal(idx.parentCount, 0);
    });

    it("all sentinels", () => {
        const fk = new Int32Array([-1, -1, -1]);
        const idx = buildInverseIndex(fk, 3);
        for (let j = 0; j < 3; j++) {
            assert.equal(childrenOf(idx, j).length, 0);
        }
    });

    it("parent with 0 children among non-empty parents", () => {
        const fk = new Int32Array([0, 0, 2, 2]);
        const idx = buildInverseIndex(fk, 3);
        assert.deepStrictEqual(Array.from(childrenOf(idx, 0)), [0, 1]);
        assert.equal(childrenOf(idx, 1).length, 0);  // empty!
        assert.deepStrictEqual(Array.from(childrenOf(idx, 2)), [2, 3]);
    });
});

describe("buildInverseIndex — sortedness assertion", () => {
    it("SortedRange throws on unsorted input", () => {
        const fk = new Int32Array([0, 2, 1, 1]);
        assert.throws(
            () => buildInverseIndex(fk, 3, { sorted: true }),
            /not sorted/,
        );
    });

    it("SortedRange accepts non-decreasing with gaps", () => {
        const fk = new Int32Array([0, 0, 2, 2]);
        const idx = buildInverseIndex(fk, 4, { sorted: true });
        assert.equal(idx.type, "sorted_range");
    });
});

describe("childrenOf — out of range parent", () => {
    it("returns empty for negative parent", () => {
        const fk = new Int32Array([0, 1]);
        const idx = buildInverseIndex(fk, 2);
        assert.equal(childrenOf(idx, -1).length, 0);
    });

    it("returns empty for parent >= parentCount", () => {
        const fk = new Int32Array([0, 1]);
        const idx = buildInverseIndex(fk, 2);
        assert.equal(childrenOf(idx, 5).length, 0);
    });
});
