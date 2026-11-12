"""PHASE_4_6 D1p2 — test-only production CDSCompress safety net.

Candidate by GPT28.RI; requires receiving-HEAD reconciliation and pinned Bokeh
3.3.2 + Node/jsdom 24.1.3. Only independent Python controls were run here.
Node tests deserialize real bokehDrawSA.fromArray documents and exercise the
actual CDSCompress / HistogramCDS TypeScript models. No product files changed.

D1p2 deliberately complements (does not duplicate) the historical
SerializationUtils/compressArray codec round-trip tests. Missing update
invalidation is a test failure, NOT a pre-approved XFAIL.

Integration (Claude20.RI, 2026-11-11). In plain words: CDSCompress is the table that is sent to the page compressed;
it decodes a column the first time the column is read and keeps the decoded array (its cache, `freshColumns`).
Target node_transport. Changes to GPT28's v0.1a, all test-only:
- The Architect decided (2026-11-11) that replacing the data of a live CDSCompress is a supported operation, so
  the replacement test is a known defect, TR-D1, with an exact signature, a strict partner (GPT28's test, unchanged
  expectation) and a healthy control:
    TR-D1  after `inputData` is replaced, every column decoded before the replacement keeps its OLD values (the
           cache is never invalidated; CDSCompress.ts listens only to `enableDithering`), every column not decoded
           before comes out NEW, so the table can mix old and new columns; the histogram keeps the old bins.
- Review F4: the wrong-length test also rejects page and console errors and checks the cache after the refusal.
  That check found a second defect:
    TR-D2  a column refused for its wrong length is cached before the length check (CDSCompress.ts get_column), so
           the SECOND read returns the 7-entry column without any error.
- Review F4: a cold-cache test shows the decoding is lazy (nothing decoded at load in Node, reading x decodes x).
- Review F3: narrower feature ids instead of ENC.compression.roundtrip, which stays planned: lossless decoding is
  not the whole compression pipeline (no lossy codes here).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect (integration, Claude20.RI)."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


STRICT = dict(strict=True, raises=KnownDefect)
LENGTH_ERROR = "Corrupted length of column w: 7 expected: 8"

LOSSLESS = [(r".*", [("zip", 0), ("base64", 0)])]
EDGES = np.array([0., 1., 2., 3., 4.])
# Within the gate RangeSlider bounds [-1.5, 2.0]; one of its 30 legal steps.
# No fixture gate values lie in this interval.
EMPTY_RANGE = (0.25, 0.25 + 3.5 / 30)


def _frame():
    return pd.DataFrame({
        "x": np.array([.25, .75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75], dtype=np.float64),
        "y": np.array([-0., .25, np.nan, 1.5, np.inf, -np.inf, .5, -1.5], dtype=np.float64),
        "w": np.array([1., 2., 3., 4., 5., 6., 7., 8.], dtype=np.float64),
        "gate": np.array([-1.5, -.5, .5, 1.5, -1., 0., 1., 2.], dtype=np.float64),
    })


def _num_equal(got, expected):
    """Event-level exact oracle, including nonfinite location and signed zero."""
    a, b = np.asarray(got, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_array_equal(np.isposinf(a), np.isposinf(b))
    np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))
    np.testing.assert_array_equal(np.signbit(a[a == 0]), np.signbit(b[b == 0]))
    np.testing.assert_array_equal(a[np.isfinite(a)], b[np.isfinite(b)])


def _hist_ref(df, cut=None):
    selected = df if cut is None else df.loc[df.gate.between(*cut)]
    return np.histogram(selected.x.to_numpy(), bins=EDGES, weights=selected.w.to_numpy())[0]


def _document(df, target: Path, *, with_filter=False, corrupted_weight_payload=None, source_only=False):
    """Use the public page builder and then optionally corrupt a disposable document.

    A distinct builder invocation supplies replacement compressed inputData; no
    synthetic JS model, fake decoder or production-source modification is used.
    """
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa

    figures = [
        [["bin_center"], ["bin_count"], {"source": "h"}],
        [["x"], ["y"]],
    ]
    widgets = [["range", ["gate", -1.5, 2.]]] if with_filter else []
    shown = sa.show
    sa.show = lambda *a, **k: None
    try:
        page = sa.bokehDrawSA.fromArray(
            df, None, figures, widgets,
            sourceArray=[{"name": "h", "variables": ["x"], "nbins": 4,
                          "range": [0, 4], "weights": "w"}],
            arrayCompression=[(pat, list(acts)) for pat, acts in LOSSLESS],
            layout=[[0, 1]], widgetLayout=[list(range(len(widgets)))],
        )
    finally:
        sa.show = shown

    sources = [m for m in page.pAll.references()
               if type(m).__name__ == "CDSCompress" and getattr(m, "name", None) == "cdsOrig"]
    assert len(sources) == 1, f"expected exactly one public CDSCompress, found {len(sources)}"
    source = sources[0]
    assert set(("x", "y", "w")).issubset(source.inputData), sorted(source.inputData)
    assert source.inputData["x"]["dtype"] == "float64"
    if corrupted_weight_payload is not None:
        source.inputData = {**source.inputData, "w": corrupted_weight_payload}

    doc = Document()
    doc.add_root(source if source_only else page.pAll)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return target, json.loads(json.dumps(source.inputData))


def _open(session, path):
    reply = session.request("new_doc", path=str(path))
    assert reply.get("ok"), reply
    assert not reply.get("page_errors") and not reply.get("console_errors"), reply


def _eval(session, js, arg=None):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    reply = session.request("eval", code=js, arg=arg)
    assert reply.get("ok") and not reply.get("page_errors") and not reply.get("console_errors"), reply
    return decode(reply["result"])


READ = r"""() => {
  const src=window.__riH.model('cdsOrig',['CDSCompress']);
  if (!src) throw Error(window.__riH.why('cdsOrig',['CDSCompress']));
  const x=src.get_column('x'), xAgain=src.get_column('x');
  const y=src.get_column('y'), w=src.get_column('w');
  const h=window.__riH.model('h',['HistogramCDS']);
  if (!h) throw Error(window.__riH.why('h',['HistogramCDS']));
  return {
    type:src.type, n:src.get_length(), keys:Object.keys(src.inputData).sort(),
    x:Array.from(x), y:Array.from(y), w:Array.from(w),
    cacheIdentity:x===xAgain,
    cached:Array.from(src.freshColumns).sort(),
    bins:Array.from(h.get_column('bin_count'))
  };
}"""

# Integration: the histogram alone (it decodes x and w, never y), the decoded-column cache, and a replacement step
# that reports the cache before and after it.
BINS_ONLY = r"""() => {
  const src=window.__riH.model('cdsOrig',['CDSCompress']);
  if (!src) throw Error(window.__riH.why('cdsOrig',['CDSCompress']));
  const h=window.__riH.model('h',['HistogramCDS']);
  if (!h) throw Error(window.__riH.why('h',['HistogramCDS']));
  return {bins:Array.from(h.get_column('bin_count')), cached:Array.from(src.freshColumns).sort()};
}"""

REPLACE = r"""(inputData) => {
  const m=window.__riH.model('cdsOrig',['CDSCompress']);
  if (!m) throw Error('missing production CDSCompress');
  const before=Array.from(m.freshColumns).sort();
  m.inputData = inputData;
  m.change.emit();
  return {updated:true, cachedBefore:before, cachedAfter:Array.from(m.freshColumns).sort(),
          keys:Object.keys(m.inputData).sort()};
}"""


def _equal(got, expected) -> bool:
    try:
        _num_equal(got, expected)
        return True
    except AssertionError:
        return False


def _is_tr_d1(replaced, observed, old, new, decoded_before) -> bool:
    """TR-D1, exact: the replacement leaves the cache as it was (non-empty); every column decoded before it keeps the
    old values, every other column is new; the histogram (x and w) keeps the old bins; and the replacement really
    changed a decoded column and the histogram, so 'old' and 'new' are distinguishable."""
    decoded = sorted(decoded_before)
    return (bool(decoded)
            and replaced["cachedBefore"] == replaced["cachedAfter"] == decoded
            and all(_equal(observed[c], old[c]) for c in decoded)
            and all(_equal(observed[c], new[c]) for c in ("x", "y", "w") if c not in decoded)
            and any(not _equal(old[c], new[c]) for c in decoded)
            and _equal(observed["bins"], _hist_ref(old))
            and not _equal(_hist_ref(old), _hist_ref(new)))


def _replacement(d):
    replacement = d.copy()
    replacement["x"] = d.x.to_numpy()[::-1].copy()
    replacement["w"] = d.w.to_numpy() * 2
    return replacement


@pytest.mark.headless_unit
def test_d1p2_transport_oracle_nonfinite_signed_zero_and_mutations():
    d = _frame()
    _num_equal(d.y, d.y)
    _num_equal(_hist_ref(d), [3., 7., 11., 15.])
    altered = d.y.to_numpy().copy(); altered[0] = +0.
    with pytest.raises(AssertionError):
        _num_equal(altered, d.y)
    altered = d.y.to_numpy().copy(); altered[2] = 0.
    with pytest.raises(AssertionError):
        _num_equal(altered, d.y)
    altered = d.y.to_numpy().copy(); altered[4] = -np.inf
    with pytest.raises(AssertionError):
        _num_equal(altered, d.y)
    wrong = _hist_ref(d).copy(); wrong[2] += 1
    with pytest.raises(AssertionError):
        _num_equal(wrong, _hist_ref(d))


@pytest.mark.headless_unit
def test_d1p2_transport_oracle_filter_and_restore():
    d = _frame()
    np.testing.assert_array_equal(_hist_ref(d, (-1., 1.)), [2., 3., 11., 7.])
    assert -1.5 <= EMPTY_RANGE[0] < EMPTY_RANGE[1] <= 2.0
    assert not d.gate.between(*EMPTY_RANGE).any(), "empty-range probe selected fixture rows"
    np.testing.assert_array_equal(_hist_ref(d, EMPTY_RANGE), np.zeros(4))
    np.testing.assert_array_equal(_hist_ref(d, (-1.5, 2.)), _hist_ref(d))
    with pytest.raises(AssertionError):
        _num_equal(_hist_ref(d, (-1., 1.)), _hist_ref(d))


@pytest.mark.headless_unit
def test_d1p2_tr_d1_signature_is_not_vacuous():
    """Integration: the TR-D1 detector accepts exactly the defect state and rejects the correct one and a mix-up."""
    d = _frame()
    r = _replacement(d)
    r["y"] = d.y.to_numpy()[::-1].copy()
    keep = {"cachedBefore": ["w", "x"], "cachedAfter": ["w", "x"]}
    defect = {"x": d.x, "w": d.w, "y": r.y, "bins": _hist_ref(d)}
    assert _is_tr_d1(keep, defect, d, r, ["w", "x"])
    assert not _is_tr_d1(keep, {"x": r.x, "w": r.w, "y": r.y, "bins": _hist_ref(r)}, d, r, ["w", "x"])   # fixed
    assert not _is_tr_d1(keep, {**defect, "y": d.y}, d, r, ["w", "x"])                                      # y old too
    assert not _is_tr_d1({"cachedBefore": ["w", "x"], "cachedAfter": []}, defect, d, r, ["w", "x"])         # cache dropped
    assert not _is_tr_d1(keep, defect, d, d, ["w", "x"])                                                    # no change


@pytest.mark.node_transport
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("ENC.cds_compress.page_decode")
def test_d1p2_real_cdscompress_lazy_decode_and_cache(node_session, headless_out):
    """Lossless decoding and cache reuse (review F4: this test shows the reuse; the lazy decoding is the next test)."""
    d = _frame()
    path, _ = _document(d, headless_out / "d1p2_lazy.json")
    _open(node_session, path)
    got = _eval(node_session, READ)
    assert got["type"] == "CDSCompress"
    assert got["n"] == len(d)
    assert set(("x", "y", "w")).issubset(got["keys"])
    assert got["cacheIdentity"] is True
    for col in ("x", "y", "w"):
        _num_equal(got[col], d[col])
    assert set(("x", "y", "w")).issubset(got["cached"])
    _num_equal(got["bins"], _hist_ref(d))


@pytest.mark.node_transport
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("ENC.cds_compress.page_decode")
def test_d1p2_real_cdscompress_decodes_a_column_only_when_it_is_read(node_session, headless_out):
    """Integration (review F4): nothing is decoded at load in Node (nothing draws); reading x decodes x alone, exactly."""
    d = _frame()
    path, _ = _document(d, headless_out / "d1p2_cold.json")
    _open(node_session, path)
    got = _eval(node_session, r"""() => {
      const m=window.__riH.model('cdsOrig',['CDSCompress']);
      if (!m) throw Error(window.__riH.why('cdsOrig',['CDSCompress']));
      const atLoad=Array.from(m.freshColumns).sort();
      const x=m.get_column('x');
      return {atLoad, afterX:Array.from(m.freshColumns).sort(), x:Array.from(x)};
    }""")
    assert got["atLoad"] == [], got
    assert got["afterX"] == ["x"], got
    _num_equal(got["x"], d.x)


@pytest.mark.node_transport
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("ENC.cds_compress.page_decode")
def test_d1p2_real_compressed_source_histogram_filter_restoration(node_session, headless_out):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import SET_JS, SETTLE_ROUNDS
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    d = _frame()
    path, _ = _document(d, headless_out / "d1p2_filtered.json", with_filter=True)
    _open(node_session, path)
    _num_equal(_eval(node_session, READ)["bins"], _hist_ref(d))
    probe = [["h", "bin_count", ["HistogramCDS"]]]
    for rng, expected in [([-1., 1.], _hist_ref(d, (-1., 1.))),
                          (list(EMPTY_RANGE), _hist_ref(d, EMPTY_RANGE)),
                          ([-1.5, 2.], _hist_ref(d))]:
        reply = node_session.request("eval", code=SET_JS, arg=["gate", rng, probe, SETTLE_ROUNDS])
        assert reply.get("ok") and not reply.get("page_errors") and not reply.get("console_errors"), reply
        settled = decode(reply["result"])
        assert settled.get("settled") is True and not settled.get("error"), settled
        # Prove the real Bokeh widget accepted the requested in-bounds state.
        applied = _eval(node_session, r"""() => {
          const s=window.__riH.slider('gate');
          if (!s) throw Error('gate RangeSlider missing');
          return {value:Array.from(s.value), start:s.start, end:s.end};
        }""")
        assert applied["start"] <= rng[0] <= rng[1] <= applied["end"], applied
        np.testing.assert_allclose(applied["value"], rng, atol=1e-12, rtol=0,
                                   err_msg="gate RangeSlider rejected requested state")
        _num_equal(_eval(node_session, READ)["bins"], expected)


@pytest.mark.node_transport
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("ENC.cds_compress.data_replace")
@pytest.mark.xfail(**STRICT, reason="TR-D1: decoded columns keep their old values after inputData is replaced")
def test_d1p2_real_compressed_source_inputdata_replacement_invalidates_cache(node_session, headless_out):
    """Deliberately red if CDSCompress retains stale decoded columns after replacement.

    An inputData property update is the public Bokeh model operation. This test
    asserts a proposed update contract; classify a failure before strict XFAIL.
    Integration: the Architect classified it (supported operation), so it is TR-D1's strict partner; the
    expectation below is GPT28's, unchanged.
    """
    d = _frame()
    replacement = _replacement(d)
    original_path, _ = _document(d, headless_out / "d1p2_original.json")
    _, payload = _document(replacement, headless_out / "d1p2_replacement.json")
    _open(node_session, original_path)
    _num_equal(_eval(node_session, READ)["bins"], _hist_ref(d))
    result = _eval(node_session, REPLACE, payload)
    assert result["updated"] is True
    observed = _eval(node_session, READ)
    known_defect(_is_tr_d1(result, observed, d, replacement, ["w", "x", "y"]),
                 "TR-D1: x, y and w were decoded before the replacement and keep their old values")
    _num_equal(observed["x"], replacement.x)
    _num_equal(observed["w"], replacement.w)
    _num_equal(observed["bins"], _hist_ref(replacement))


@pytest.mark.node_transport
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("ENC.cds_compress.data_replace")
def test_signature_tr_d1_decoded_columns_stay_old_undecoded_columns_come_out_new(node_session, headless_out):
    """TR-D1, exact: the histogram decodes x and w (not y); after the replacement x and w are still the old columns,
    y (decoded only afterwards) is the new one, and the histogram keeps the old bins: the table mixes old and new."""
    d = _frame()
    replacement = _replacement(d)
    replacement["y"] = d.y.to_numpy()[::-1].copy()
    original_path, _ = _document(d, headless_out / "d1p2_sig_original.json")
    _, payload = _document(replacement, headless_out / "d1p2_sig_replacement.json")
    _open(node_session, original_path)
    first = _eval(node_session, BINS_ONLY)
    _num_equal(first["bins"], _hist_ref(d))
    assert first["cached"] == ["w", "x"], first
    result = _eval(node_session, REPLACE, payload)
    observed = _eval(node_session, READ)
    assert _is_tr_d1(result, observed, d, replacement, ["w", "x"]), (result, observed)


@pytest.mark.node_transport
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("ENC.cds_compress.data_replace")
def test_control_tr_d1_replacement_before_any_read_gives_the_new_table(node_session, headless_out):
    """TR-D1's control: replaced before any column is decoded, every column and the histogram are the new ones."""
    d = _frame()
    replacement = _replacement(d)
    replacement["y"] = d.y.to_numpy()[::-1].copy()
    original_path, _ = _document(d, headless_out / "d1p2_ctl_original.json")
    _, payload = _document(replacement, headless_out / "d1p2_ctl_replacement.json")
    _open(node_session, original_path)
    result = _eval(node_session, REPLACE, payload)
    assert result["cachedBefore"] == result["cachedAfter"] == [], result
    observed = _eval(node_session, READ)
    for col in ("x", "y", "w"):
        _num_equal(observed[col], replacement[col])
    _num_equal(observed["bins"], _hist_ref(replacement))


# Integration (review F4): the wrong-length page, read twice. x has 8 rows, the corrupted w has 7.
READ_TWICE = r"""() => {
  const m=window.__riH.model('cdsOrig',['CDSCompress']);
  if (!m) throw Error('missing production CDSCompress');
  const x=m.get_column('x');
  const tries=[];
  for (let i=0; i<2; i++) {
    try { const w=m.get_column('w'); tries.push({rejected:false, length:w.length}); }
    catch(e) { tries.push({rejected:true, error:String(e.message)}); }
  }
  const xAgain=m.get_column('x');
  return {before:x.length, tries, cached:Array.from(m.freshColumns).sort(), n:m.get_length(),
          xSame:x===xAgain, xAgainLength:xAgain.length};
}"""


def _is_tr_d2(got) -> bool:
    """TR-D2, exact: the first read of w is refused with the length error, the second returns 7 entries without
    error, and w is in the cache; the length stays 8."""
    return (got["before"] == 8 and got["n"] == 8
            and got["tries"] == [{"rejected": True, "error": LENGTH_ERROR}, {"rejected": False, "length": 7}]
            and got["cached"] == ["w", "x"])


@pytest.mark.node_transport
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("ENC.cds_compress.length_guard")
def test_d1p2_real_compressed_source_rejects_mismatched_column_length(node_session, headless_out):
    """Corrupt only a disposable public-page JSON; require the real model's guard."""
    d = _frame()
    shortened = d.iloc[:-1].reset_index(drop=True)
    _, payload = _document(shortened, headless_out / "d1p2_short.json")
    corrupt, _ = _document(d, headless_out / "d1p2_corrupt.json",
                           corrupted_weight_payload=payload["w"], source_only=True)
    _open(node_session, corrupt)
    # Deliberately decode x FIRST (8 rows), then invalid w (7 rows).
    reply = node_session.request("eval", code=r"""() => {
      const m=window.__riH.model('cdsOrig',['CDSCompress']);
      if (!m) throw Error('missing production CDSCompress');
      const x=m.get_column('x');
      try { m.get_column('w'); return {rejected:false, before:x.length}; }
      catch(e) { return {rejected:true, before:x.length, error:String(e.message)}; }
    }""")
    assert reply.get("ok"), reply
    assert not reply.get("page_errors") and not reply.get("console_errors"), reply      # integration, review F4
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    got = decode(reply["result"])
    assert got["before"] == 8
    assert got["rejected"] is True, got
    assert got["error"] == LENGTH_ERROR, got


def _corrupt_page(headless_out, stem):
    d = _frame()
    shortened = d.iloc[:-1].reset_index(drop=True)
    _, payload = _document(shortened, headless_out / f"{stem}_short.json")
    corrupt, _ = _document(d, headless_out / f"{stem}_corrupt.json",
                           corrupted_weight_payload=payload["w"], source_only=True)
    return corrupt


@pytest.mark.node_transport
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("ENC.cds_compress.length_guard")
def test_signature_tr_d2_a_refused_column_is_returned_on_the_second_read(node_session, headless_out):
    """TR-D2, exact (found by the review-F4 cache check): refused once, then returned with 7 entries, no error."""
    _open(node_session, _corrupt_page(headless_out, "d1p2_trd2_sig"))
    got = _eval(node_session, READ_TWICE)
    assert _is_tr_d2(got), got


@pytest.mark.node_transport
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("ENC.cds_compress.length_guard")
def test_control_tr_d2_a_column_of_the_right_length_is_cached_and_reused(node_session, headless_out):
    """TR-D2's control, on the same corrupted page: x (8 rows) is decoded once and the same array comes back."""
    _open(node_session, _corrupt_page(headless_out, "d1p2_trd2_ctl"))
    got = _eval(node_session, READ_TWICE)
    assert got["before"] == got["xAgainLength"] == got["n"] == 8, got
    assert got["xSame"] is True, got
    assert "x" in got["cached"], got


@pytest.mark.node_transport
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("ENC.cds_compress.length_guard")
@pytest.mark.xfail(**STRICT, reason="TR-D2: a column refused for its length is returned on the second read")
def test_strict_a_column_of_the_wrong_length_is_refused_on_every_read(node_session, headless_out):
    """The guard's contract: every read of the corrupted w is refused with the same error, and w is not cached."""
    _open(node_session, _corrupt_page(headless_out, "d1p2_trd2_strict"))
    got = _eval(node_session, READ_TWICE)
    known_defect(_is_tr_d2(got), "TR-D2: the second read returns the 7-entry w without error")
    assert got["tries"] == [{"rejected": True, "error": LENGTH_ERROR}] * 2, got
    assert "w" not in got["cached"], got
