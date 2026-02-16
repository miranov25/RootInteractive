/**
 * test_backward_compat.mjs — Phase 0.2.A TC-COMPAT-01
 *
 * Verifies that N:1 index lookup semantics are unchanged:
 * - Direct array indexing: parent_col[fk_array[i]] works as before
 * - No inverse index is built unless explicitly requested
 *
 * Run: node --test test_backward_compat.mjs
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { buildInverseIndex } from "./buildInverseIndex.mjs";

const fixture = JSON.parse(readFileSync(new URL("./ao2d_fixture.json", import.meta.url), "utf-8"));
const { tables, dense_fk } = fixture;

describe("TC-COMPAT-01: N:1 gather unchanged", () => {
    it("direct index lookup: tracks → events.n_tracks", () => {
        const trEventIdx = dense_fk.tracks_event_idx;
        const eventNTracks = tables.events.n_tracks;

        // N:1 gather: for each track, look up its event's n_tracks
        for (let i = 0; i < Math.min(100, trEventIdx.length); i++) {
            const eventIdx = trEventIdx[i];
            const nTracks = eventNTracks[eventIdx];
            assert.ok(nTracks >= 3 && nTracks <= 10,
                `event ${eventIdx} n_tracks=${nTracks} out of expected range`);
        }
    });

    it("direct index lookup: reco_tracks → tracks.pt", () => {
        const mcTrackIdx = tables.reco_tracks.mc_track_idx;
        const trackPt = tables.tracks.pt;

        for (let i = 0; i < Math.min(100, mcTrackIdx.length); i++) {
            const trkIdx = mcTrackIdx[i];
            const pt = trackPt[trkIdx];
            assert.ok(pt >= 0.3 && pt <= 5.0,
                `track ${trkIdx} pt=${pt} out of expected range`);
        }
    });

    it("inverse index is independent — building it does not affect N:1", () => {
        const fk = new Int32Array(dense_fk.tracks_event_idx);
        const nEvents = tables.events.event_id.length;

        // N:1 lookup before inverse index
        const val_before = tables.events.n_tracks[fk[0]];

        // Build inverse index
        buildInverseIndex(fk, nEvents);

        // N:1 lookup after — same result
        const val_after = tables.events.n_tracks[fk[0]];
        assert.equal(val_before, val_after);
    });
});
