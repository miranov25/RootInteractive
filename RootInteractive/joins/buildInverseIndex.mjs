/**
 * buildInverseIndex.mjs — Phase 0.2.A
 *
 * Builds inverse index from a foreign key column: parent → child row indices.
 *
 * Two variants:
 *   InverseIndexCSR      — for unsorted FK columns (always correct)
 *   SortedRangeIndex     — for sorted FK columns (faster, cache-friendly)
 *
 * @module buildInverseIndex
 */

/**
 * Build an inverse index from a foreign key array.
 *
 * @param {Int32Array} fkArray - Foreign key column (child → parent row index).
 *   Values in [0, parentCount-1] are valid. Values < 0 or >= parentCount are
 *   excluded (sentinel handling).
 * @param {number} parentCount - Number of rows in the parent table.
 * @param {object} [options]
 * @param {boolean} [options.sorted=false] - If true, fkArray must be
 *   non-decreasing. Caller's responsibility. Enables SortedRangeIndex.
 *   Throws if violated (P1-1 from review).
 * @returns {InverseIndexCSR | SortedRangeIndex}
 */
export function buildInverseIndex(fkArray, parentCount, options = {}) {
    if (!(fkArray instanceof Int32Array)) {
        throw new TypeError("fkArray must be Int32Array");
    }
    if (parentCount < 0 || !Number.isInteger(parentCount)) {
        throw new RangeError("parentCount must be a non-negative integer");
    }

    const sorted = options.sorted === true;

    if (sorted) {
        return _buildSortedRangeIndex(fkArray, parentCount);
    } else {
        return _buildCSR(fkArray, parentCount);
    }
}

/**
 * @typedef {object} InverseIndexCSR
 * @property {"csr"} type
 * @property {Int32Array} offsets - Length parentCount + 1. Parent j's children
 *   are at indices[offsets[j] .. offsets[j+1]).
 * @property {Int32Array} indices - Child row indices, grouped by parent.
 * @property {number} parentCount
 * @property {number} childCount - Length of original fkArray.
 */

/**
 * Build CSR inverse index (unsorted FK).
 * Two passes: count, prefix-sum, fill.
 * O(N_children) time, O(N_parent + N_valid_children) space.
 */
function _buildCSR(fkArray, parentCount) {
    const n = fkArray.length;

    // Pass 1: count valid children per parent
    const counts = new Int32Array(parentCount);
    let validCount = 0;
    for (let i = 0; i < n; i++) {
        const v = fkArray[i];
        if (v >= 0 && v < parentCount) {
            counts[v]++;
            validCount++;
        }
    }

    // Prefix sum → offsets
    const offsets = new Int32Array(parentCount + 1);
    for (let j = 0; j < parentCount; j++) {
        offsets[j + 1] = offsets[j] + counts[j];
    }

    // Pass 2: fill indices
    const indices = new Int32Array(validCount);
    const pos = new Int32Array(parentCount);
    for (let j = 0; j < parentCount; j++) {
        pos[j] = offsets[j];
    }
    for (let i = 0; i < n; i++) {
        const v = fkArray[i];
        if (v >= 0 && v < parentCount) {
            indices[pos[v]++] = i;
        }
    }

    return {
        type: "csr",
        offsets,
        indices,
        parentCount,
        childCount: n,
    };
}

/**
 * @typedef {object} SortedRangeIndex
 * @property {"sorted_range"} type
 * @property {Int32Array} starts - Length parentCount. Parent j's children
 *   start at starts[j] in the child table.
 * @property {Int32Array} counts - Length parentCount. Parent j has counts[j] children.
 * @property {number} parentCount
 * @property {number} childCount
 */

/**
 * Build SortedRangeIndex (sorted FK).
 * Single pass. Asserts non-decreasing order (P1-1 from review).
 * O(N_children) time, O(N_parent) space.
 */
function _buildSortedRangeIndex(fkArray, parentCount) {
    const n = fkArray.length;
    const starts = new Int32Array(parentCount);
    const counts = new Int32Array(parentCount);

    if (n === 0) {
        return { type: "sorted_range", starts, counts, parentCount, childCount: 0 };
    }

    // Assert non-decreasing (P1-1: throw on violation)
    let prevValid = -1;
    for (let i = 0; i < n; i++) {
        const v = fkArray[i];
        if (v >= 0 && v < parentCount) {
            if (v < prevValid) {
                throw new Error(
                    `SortedRangeIndex: fkArray not sorted at index ${i} ` +
                    `(value ${v} < previous valid ${prevValid})`
                );
            }
            prevValid = v;
        }
    }

    // Single pass: scan for contiguous runs
    let i = 0;
    while (i < n) {
        const v = fkArray[i];
        if (v < 0 || v >= parentCount) {
            i++;
            continue;
        }
        const start = i;
        while (i < n && fkArray[i] === v) {
            i++;
        }
        starts[v] = start;
        counts[v] = i - start;
    }

    return {
        type: "sorted_range",
        starts,
        counts,
        parentCount,
        childCount: n,
    };
}

/**
 * Get children of a parent from an inverse index.
 *
 * @param {InverseIndexCSR | SortedRangeIndex} index
 * @param {number} parentIdx
 * @returns {Int32Array} Child row indices (sorted ascending for CSR,
 *   contiguous range for SortedRange).
 */
export function childrenOf(index, parentIdx) {
    if (parentIdx < 0 || parentIdx >= index.parentCount) {
        return new Int32Array(0);
    }
    if (index.type === "csr") {
        const start = index.offsets[parentIdx];
        const end = index.offsets[parentIdx + 1];
        // Return sorted copy (CSR fill order may not be sorted for unsorted FK)
        const children = index.indices.slice(start, end);
        children.sort();
        return children;
    } else {
        // sorted_range: contiguous range
        const s = index.starts[parentIdx];
        const c = index.counts[parentIdx];
        const result = new Int32Array(c);
        for (let i = 0; i < c; i++) {
            result[i] = s + i;
        }
        return result;
    }
}
