/**
 * test_cross_table_filter.mjs — Phase 0.2.A
 *
 * JS-internal CrossTableFilter tests using small synthetic data.
 *
 * Run: node --test test_cross_table_filter.mjs
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { buildInverseIndex } from "./buildInverseIndex.mjs";
import { crossTableFilter, maskToIndices } from "./CrossTableFilter.mjs";

describe("crossTableFilter — basic", () => {
    // 3 parents, 6 children: [0,0,1,1,1,2]
    const fk = new Int32Array([0, 0, 1, 1, 1, 2]);
    const idx = buildInverseIndex(fk, 3);

    it("single parent selection", () => {
        const mask = crossTableFilter(idx, new Int32Array([1]), 6);
        assert.deepStrictEqual(Array.from(mask), [0, 0, 1, 1, 1, 0]);
    });

    it("multiple parent selection", () => {
        const mask = crossTableFilter(idx, new Int32Array([0, 2]), 6);
        assert.deepStrictEqual(Array.from(mask), [1, 1, 0, 0, 0, 1]);
    });

    it("select all", () => {
        const mask = crossTableFilter(idx, new Int32Array([0, 1, 2]), 6);
        assert.deepStrictEqual(Array.from(mask), [1, 1, 1, 1, 1, 1]);
    });

    it("select none", () => {
        const mask = crossTableFilter(idx, new Int32Array(0), 6);
        assert.deepStrictEqual(Array.from(mask), [0, 0, 0, 0, 0, 0]);
    });
});

describe("crossTableFilter — SortedRange variant", () => {
    const fk = new Int32Array([0, 0, 1, 1, 1, 2]);
    const idx = buildInverseIndex(fk, 3, { sorted: true });

    it("single parent via SortedRange", () => {
        const mask = crossTableFilter(idx, new Int32Array([1]), 6);
        assert.deepStrictEqual(Array.from(mask), [0, 0, 1, 1, 1, 0]);
    });

    it("CSR and SortedRange produce same mask", () => {
        const csr = buildInverseIndex(fk, 3, { sorted: false });
        const sr = buildInverseIndex(fk, 3, { sorted: true });
        const sel = new Int32Array([0, 2]);
        const maskCSR = crossTableFilter(csr, sel, 6);
        const maskSR = crossTableFilter(sr, sel, 6);
        assert.deepStrictEqual(Array.from(maskCSR), Array.from(maskSR));
    });
});

describe("crossTableFilter — sentinel exclusion", () => {
    // child 0 and 3 are sentinels (-1)
    const fk = new Int32Array([-1, 0, 1, -1, 2]);
    const idx = buildInverseIndex(fk, 3);

    it("sentinels not included when selecting all", () => {
        const mask = crossTableFilter(idx, new Int32Array([0, 1, 2]), 5);
        // Only children 1, 2, 4 are valid
        assert.deepStrictEqual(Array.from(mask), [0, 1, 1, 0, 1]);
    });
});

describe("crossTableFilter — cascade", () => {
    // level 0 → level 1: [0,0,1,1]
    // level 1 → level 2: [0,0,1,1,2,2,3,3]
    const fk01 = new Int32Array([0, 0, 1, 1]);
    const fk12 = new Int32Array([0, 0, 1, 1, 2, 2, 3, 3]);
    const inv01 = buildInverseIndex(fk01, 2);
    const inv12 = buildInverseIndex(fk12, 4);

    it("cascade: select parent 0 → level 1 → level 2", () => {
        // Parent 0 → children [0, 1]
        const mask1 = crossTableFilter(inv01, new Int32Array([0]), 4);
        assert.deepStrictEqual(Array.from(mask1), [1, 1, 0, 0]);

        // Children [0, 1] → grandchildren
        const sel1 = maskToIndices(mask1);
        const mask2 = crossTableFilter(inv12, sel1, 8);
        assert.deepStrictEqual(Array.from(mask2), [1, 1, 1, 1, 0, 0, 0, 0]);
    });
});

describe("maskToIndices", () => {
    it("extracts correct indices", () => {
        const mask = new Uint8Array([0, 1, 0, 1, 1, 0]);
        const indices = maskToIndices(mask);
        assert.deepStrictEqual(Array.from(indices), [1, 3, 4]);
    });

    it("empty mask → empty indices", () => {
        const mask = new Uint8Array([0, 0, 0]);
        assert.equal(maskToIndices(mask).length, 0);
    });

    it("full mask → all indices", () => {
        const mask = new Uint8Array([1, 1, 1]);
        assert.deepStrictEqual(Array.from(maskToIndices(mask)), [0, 1, 2]);
    });
});
