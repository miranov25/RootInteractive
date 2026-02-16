/**
 * CrossTableFilter.mjs — Phase 0.2.A
 *
 * Computes child include-mask from parent selection using an inverse index.
 *
 * @module CrossTableFilter
 */

/**
 * Compute child include-mask from parent selection.
 *
 * @param {InverseIndexCSR | SortedRangeIndex} inverseIndex - Prebuilt inverse index.
 * @param {Int32Array | number[]} selectedParents - Indices of selected parent rows.
 * @param {number} childCount - Total number of child rows (mask length).
 * @returns {Uint8Array} Include-mask: 1 = keep, 0 = exclude.
 */
export function crossTableFilter(inverseIndex, selectedParents, childCount) {
    const mask = new Uint8Array(childCount);

    if (inverseIndex.type === "csr") {
        const { offsets, indices } = inverseIndex;
        for (let k = 0; k < selectedParents.length; k++) {
            const p = selectedParents[k];
            if (p < 0 || p >= inverseIndex.parentCount) continue;
            const start = offsets[p];
            const end = offsets[p + 1];
            for (let j = start; j < end; j++) {
                mask[indices[j]] = 1;
            }
        }
    } else {
        // sorted_range
        const { starts, counts } = inverseIndex;
        for (let k = 0; k < selectedParents.length; k++) {
            const p = selectedParents[k];
            if (p < 0 || p >= inverseIndex.parentCount) continue;
            const s = starts[p];
            const c = counts[p];
            for (let j = s; j < s + c; j++) {
                mask[j] = 1;
            }
        }
    }

    return mask;
}

/**
 * Extract selected indices from a mask.
 *
 * @param {Uint8Array} mask - Include-mask (1 = selected).
 * @returns {Int32Array} Sorted indices where mask[i] == 1.
 */
export function maskToIndices(mask) {
    let count = 0;
    for (let i = 0; i < mask.length; i++) {
        if (mask[i]) count++;
    }
    const result = new Int32Array(count);
    let k = 0;
    for (let i = 0; i < mask.length; i++) {
        if (mask[i]) result[k++] = i;
    }
    return result;
}
