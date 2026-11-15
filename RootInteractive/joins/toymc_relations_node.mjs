/**
 * PHASE_5_1 Node-first integration tests for RI's EXISTING inverse-index / cross-table-filter code.
 * Run explicitly with RI_TOYMC_FIXTURE pointing to generated JSON; NOT part of test_*.mjs glob.
 * The Python oracle is constructed separately. This file does NOT test CDSJoin.ts.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { buildInverseIndex, childrenOf } from "./buildInverseIndex.mjs";
import { crossTableFilter, maskToIndices } from "./CrossTableFilter.mjs";

const expectedMd5 = "10a9546d2b1c2632c1496d1be267197d";
const fixturePath = process.env.RI_TOYMC_FIXTURE;
if (!fixturePath) throw new Error("RI_TOYMC_FIXTURE unset: run generate_toymc_node_fixture.py first");
const d = JSON.parse(readFileSync(fixturePath, "utf8"));
const { tables: t, oracle: o, selections: s, metadata: meta } = d;
const names = ["collisions", "tracks", "v0s", "cascades"];
const id = a => Array.from(a);
const where = mask => id(maskToIndices(mask));
const fk = (table, field) => Int32Array.from(t[table][field]);
const idx = (table, field, parent) => buildInverseIndex(fk(table, field), meta.row_counts[parent]);
const childScan = (ref, row) => ref.flatMap((p, j) => p === row ? [j] : []);

function gather(values, reference) {
  return reference.map(i => i >= 0 && i < values.length ? values[i] : null);
}
function expectedSelection(which) { return s[which].expected; }
function allPaths(selected) {
  // Selection is AND over ALL constituent paths, not an OR over V0 branches.
  const sel = new Int32Array(selected);
  const trIdx = idx("tracks", "collisionIdx", "collisions");
  const tr = crossTableFilter(trIdx, sel, meta.row_counts.tracks);
  const trRows = maskToIndices(tr);
  const pos = crossTableFilter(idx("v0s", "posTrackIdx", "tracks"), trRows, meta.row_counts.v0s);
  const neg = crossTableFilter(idx("v0s", "negTrackIdx", "tracks"), trRows, meta.row_counts.v0s);
  const v0Direct = crossTableFilter(idx("v0s", "collisionIdx", "collisions"), sel, meta.row_counts.v0s);
  const v0 = Uint8Array.from(pos, (p, i) => p && neg[i] && v0Direct[i] ? 1 : 0);
  const v0Rows = maskToIndices(v0);
  const v0C = crossTableFilter(idx("cascades", "v0Idx", "v0s"), v0Rows, meta.row_counts.cascades);
  const bachC = crossTableFilter(idx("cascades", "bachelorTrackIdx", "tracks"), trRows, meta.row_counts.cascades);
  const caDirect = crossTableFilter(idx("cascades", "collisionIdx", "collisions"), sel, meta.row_counts.cascades);
  const ca = Uint8Array.from(v0C, (v, i) => v && bachC[i] && caDirect[i] ? 1 : 0);
  return { collisions: id(sel).sort((a, b) => a - b), tracks: where(tr),
           v0s: where(v0), cascades: where(ca) };
}

test("TMC-N0: source identity, row counts, schema, row-zero and non-vacuous branches", () => {
  assert.equal(meta.fixture_version, "0.1");
  assert.equal(meta.generator_md5, expectedMd5);
  assert.equal(meta.seed, 42);
  assert.equal(meta.n_collisions, 10);
  for (const name of names) {
    assert.equal(t[name].rowId.length, meta.row_counts[name]);
    assert.deepEqual(t[name].rowId, Array.from({ length: meta.row_counts[name] }, (_, i) => i));
    for (const [key, column] of Object.entries(t[name])) {
      assert.equal(column.length, meta.row_counts[name], `${name}.${key}`);
    }
  }
  assert.ok(t.tracks.collisionIdx.some(x => x === 0));
  assert.equal(t.v0s.rowId[0], 0); // row zero exists; cascade reuse is not guaranteed
  assert.ok(t.cascades.bachelorTrackIdx.some(x => x === 0));
  assert.ok(meta.row_counts.cascades > 0);
  assert.equal(t.collisions.nTracks.at(-1), 0);  // reserved empty collision
});

test("TMC-N1: five forward gathers agree with Python numerical oracle", () => {
  const checks = [
    ["tracks.collisionVz", gather(t.collisions.vz, t.tracks.collisionIdx)],
    ["v0s.posPt", gather(t.tracks.pt, t.v0s.posTrackIdx)],
    ["v0s.negPt", gather(t.tracks.pt, t.v0s.negTrackIdx)],
    ["cascades.v0Mass", gather(t.v0s.mass, t.cascades.v0Idx)],
    ["cascades.bachelorPt", gather(t.tracks.pt, t.cascades.bachelorTrackIdx)],
  ];
  for (const [key, result] of checks) {
    assert.deepEqual(result, o[key], key);
    assert.equal(result.length, meta.row_counts[key.split(".")[0]]);
  }
});

test("TMC-N2: recursive V0 and cascade collision paths match independent oracle", () => {
  const posCollision = gather(t.tracks.collisionIdx, t.v0s.posTrackIdx);
  const negCollision = gather(t.tracks.collisionIdx, t.v0s.negTrackIdx);
  assert.deepEqual(posCollision, o["v0s.posCollisionIdx"]);
  assert.deepEqual(negCollision, o["v0s.negCollisionIdx"]);
  for (const [key, value] of [
    ["cascades.collisionViaV0Pos", gather(posCollision, t.cascades.v0Idx)],
    ["cascades.collisionViaV0Neg", gather(negCollision, t.cascades.v0Idx)],
    ["cascades.collisionViaBachelor", gather(t.tracks.collisionIdx, t.cascades.bachelorTrackIdx)],
  ]) assert.deepEqual(value, o[key], key);
  if (!meta.orphans) for (const x of ["cascades.collisionViaV0Pos", "cascades.collisionViaV0Neg",
                                   "cascades.collisionViaBachelor"]) {
    assert.deepEqual(o[x], t.cascades.collisionIdx);
  }
});

test("TMC-N3: inverse-index CSR reproduces explicit child scan including shared references", () => {
  for (const [child, field, parent] of [
    ["tracks", "collisionIdx", "collisions"],
    ["v0s", "posTrackIdx", "tracks"], ["v0s", "negTrackIdx", "tracks"],
    ["cascades", "v0Idx", "v0s"], ["cascades", "bachelorTrackIdx", "tracks"],
  ]) {
    const ref = t[child][field], inverted = idx(child, field, parent);
    let count = 0;
    for (let p = 0; p < meta.row_counts[parent]; p++) {
      const got = id(childrenOf(inverted, p));
      const expected = childScan(ref, p);
      assert.deepEqual(got, expected, `${child}.${field} parent ${p}`);
      count += got.length;
    }
    assert.equal(count, ref.filter(p => p >= 0 && p < meta.row_counts[parent]).length);
  }
});

test("TMC-N4: multi-parent AND propagation agrees with Python truth", () => {
  for (const [name, item] of Object.entries(s)) {
    assert.deepEqual(allPaths(item.input), expectedSelection(name), name);
  }
  const empty = allPaths([meta.n_collisions - 1]);
  assert.equal(empty.tracks.length + empty.v0s.length + empty.cascades.length, 0);
});

test("TMC-N5: orphan -1 never accesses last parent or appears in inverse indices", () => {
  if (!meta.orphans) return;
  assert.ok(t.tracks.collisionIdx.includes(-1));
  assert.ok(t.v0s.posTrackIdx.includes(-1));
  assert.ok(t.cascades.bachelorTrackIdx.includes(-1));
  assert.equal(o["tracks.collisionVz"].at(-1), null);
  for (const name of Object.keys(s)) assert.deepEqual(allPaths(s[name].input), s[name].expected);
  const inv = idx("tracks", "collisionIdx", "collisions");
  const allChildren = new Set();
  for (let p = 0; p < meta.n_collisions; p++) for (const c of childrenOf(inv, p)) allChildren.add(c);
  assert.ok(!allChildren.has(t.tracks.rowId.at(-1)));
});

test("TMC-N6: negative controls catch dropped children, OR instead of AND, and shifted FK", () => {
  const ref = t.tracks.collisionIdx;
  const expected = s.collision_0.expected.tracks;
  assert.ok(expected.length > 0);
  const shifted = [...ref];
  shifted[expected[0]] = meta.n_collisions - 1;
  const wrongMask = crossTableFilter(buildInverseIndex(Int32Array.from(shifted), meta.n_collisions),
                                    new Int32Array([0]), shifted.length);
  assert.notDeepEqual(where(wrongMask), expected, "shifted-track-index mutant must be detected");
  const good = allPaths([0]);
  assert.deepEqual(good, s.collision_0.expected);
  const tMask = crossTableFilter(idx("tracks", "collisionIdx", "collisions"), new Int32Array([0]), t.tracks.rowId.length);
  const negMask = crossTableFilter(idx("v0s", "negTrackIdx", "tracks"), maskToIndices(tMask), t.v0s.rowId.length);
  const posMask = crossTableFilter(idx("v0s", "posTrackIdx", "tracks"), maskToIndices(tMask), t.v0s.rowId.length);
  const orphanIndex = t.v0s.posTrackIdx.findIndex(x => x < 0);
  if (orphanIndex >= 0) {
    const wrongOr = where(Uint8Array.from(posMask, (p, i) => p || negMask[i] ? 1 : 0));
    assert.notDeepEqual(wrongOr, good.v0s, "one-daughter OR mutant must be detected");
  }
});
