"""The Node model harness itself (tier T2; proposal v0.3 §3, stage 4.2c-1, acceptance items A-C).

A  the stored model bundle: key (erratum V-4), inventory of the 20 model classes with a negative control;
B  the three rules, each shown to hold and shown to be necessary: realm, construction path, settle rule (V-3);
   the delayed-update control (two timer rounds) and the never-ready control, as in the browser harness;
C  errors in the page and console errors are ERROR outcomes, also when raised while the harness starts (v0.2,
   GPT24); the harness exits cleanly with code 0;
D  isolation between documents (v0.2, Claude3 and GPT24): a global set by one scenario does not exist in the next;
E  the 4.2c-1 review notes (stage 4.2c-2a): the limits of isolation and the fresh window (A-1), controls on recorded
   values (A-2), the late-update control (A-6).
"""
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import data as data_mod
from RootInteractive.InteractiveDrawing.bokeh.headless.engine import node_bundle
from RootInteractive.InteractiveDrawing.bokeh.headless.engine import page as page_mod
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import NodeSession
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import SET_JS, SETTLE_ROUNDS, _probes, run
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
    Labels, Page, Scenario, TableSpec, Tables, audit, auto, hist, js_action, js_value, load, range_, set_)
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios.known_defects import B1_TWO_FILTERS

pytestmark = [pytest.mark.backend("node"), pytest.mark.layer("integration")]


def _small(id_, steps, checks, widget=False):
    return Scenario(id=id_, data=Tables(TableSpec(1000, 3, {"x": "uniform(0,1)", "a": "normal(0,1)"})),
                    page=Page(histograms=(hist("h1", ["x"], 10, [0, 1]),), widgets=(range_("a"),) if widget else ()),
                    steps=steps, checks=checks, labels=Labels(family="harness", technique="node harness"))


# ---------------------------------------------------------------------------------------------- A: bundle, inventory
def test_bundle_key_follows_bokeh_sources_and_model_list():
    base = node_bundle.cache_key("3.3.2", {"A.ts": "x", "B.ts": "y"}, ["M1", "M2"])
    assert base == node_bundle.cache_key("3.3.2", {"B.ts": "y", "A.ts": "x"}, ["M2", "M1"])      # order does not matter
    assert base != node_bundle.cache_key("3.3.4", {"A.ts": "x", "B.ts": "y"}, ["M1", "M2"])      # Bokeh version
    assert base != node_bundle.cache_key("3.3.2", {"A.ts": "x ", "B.ts": "y"}, ["M1", "M2"])     # a changed source
    assert base != node_bundle.cache_key("3.3.2", {"A.ts": "x", "B.ts": "y"}, ["M1", "M2", "M3"])  # an added import


def test_inventory_is_complete_and_the_check_can_fail(node_session):
    assert node_session.missing_models == [], f"models not registered in Node: {node_session.missing_models}"
    got = node_session.request("registered", names=["HistogramCDS", "NoSuchModel"])["result"]
    assert got == ["HistogramCDS"]                                   # negative control: an absent class is reported
    assert len(node_bundle.EXPECTED_MODELS) == 20


# ---------------------------------------------------------------------------------------------- B: the three rules
def test_realm_and_construction_rules(node_session):
    r = node_session.request("selftest")["result"]
    assert r["node_realm"].startswith("rejected"), r              # rule 1 is necessary: Node-realm data is refused
    assert r["window_realm"] == "accepted", r                     # ... and sufficient
    # rule 2 is necessary: with a VALID source, direct construction fails for the construction-order reason itself
    assert "setting 'length'" in r["direct_histo_nd"], r


def test_settle_rule_waits_for_a_timer_round(node_session, node_bundle_root):
    """Erratum V-3: after a widget change the selection changes only after a setTimeout(0) round; one microtask
    reads the old value. The engine's settle rule (microtask, then timer rounds until two equal snapshots) sees it."""
    out = node_bundle_root / "unit"
    out.mkdir(exist_ok=True)
    df = data_mod.generate(B1_TWO_FILTERS.data.main)
    doc = page_mod.build_document(B1_TWO_FILTERS, df, out / "b1_document.json")
    sel = "() => window.__riH.selected('h2', ['HistogramCDS', 'HistoNdCDS'], 20000).count"
    node_session.request("new_doc", path=str(doc))
    before = node_session.request("eval", code=sel)["result"]
    one_microtask = node_session.request("eval", code="""async () => { window.__riH.slider('a').value = [-1, 1];
        await Promise.resolve(); return window.__riH.selected('h2', ['HistogramCDS', 'HistoNdCDS'], 20000).count; }""")
    node_session.request("new_doc", path=str(doc))                 # fresh document for the settled read
    settled = node_session.request("eval", code=SET_JS, arg=["a", [-1, 1], _probes(B1_TWO_FILTERS), SETTLE_ROUNDS])
    after = node_session.request("eval", code=sel)["result"]
    assert one_microtask["result"] == before == 18039                # stale after one microtask
    assert settled["ok"] and settled["result"]["settled"] and after == 12361 and after != before


def test_delayed_update_completes_by_its_condition(node_session, headless_out):
    sc = _small("harness/node_delayed_update_two_rounds",
                (load(), js_action("window.__d = 0; setTimeout(() => setTimeout(() => { window.__d = 1; }, 0), 0);",
                                   until="window.__d === 1")),
                (js_value("window.__d", 1, steps=(1,)),))
    res = run(node_session, sc, headless_out, test_ref=__file__)
    assert res.ok, res.text()


def test_condition_never_met_is_an_error(node_session, headless_out):
    sc = _small("harness/node_condition_never_met",
                (load(), js_action("window.__e = 0;", until="window.__e === 1", timeout_s=0.3)),
                (js_value("window.__e", 1, steps=(1,)),))
    res = run(node_session, sc, headless_out, test_ref=__file__)
    step1 = [o for o in res.outcomes if o.step == 1]
    assert not res.ok and [o.outcome for o in step1] == ["ERROR", "SKIP"] and "completion condition not met" in step1[0].detail


# ---------------------------------------------------------------------------------------------- C: errors, exit
def test_page_error_is_an_error(node_session, headless_out):
    sc = _small("harness/node_page_error",
                (load(), js_action("setTimeout(() => { throw new Error('boom in a timer'); }, 0); window.__t = 1;",
                                   until="window.__t === 1"), set_("a", [-1, 1])),
                (auto(),), widget=True)
    res = run(node_session, sc, headless_out, test_ref=__file__)
    pe = [o for o in res.outcomes if o.check == "page errors"]
    assert not res.ok and pe and "boom in a timer" in pe[0].detail, res.text()


def test_console_error_not_allowed_is_an_error(node_session, headless_out):
    sc = _small("harness/node_console_error_policy",
                (load(), js_action("console.error('an unexpected page error');", until="true"), set_("a", [-1, 1])),
                (auto(),), widget=True)
    res = run(node_session, sc, headless_out, test_ref=__file__)
    pol = [o for o in res.outcomes if o.check == "console error"]
    assert not res.ok and [o.outcome for o in pol] == ["ERROR"] and "an unexpected page error" in pol[0].detail


def test_harness_exits_cleanly(node_bundle_root):
    s = NodeSession(node_bundle_root)
    assert s.missing_models == []
    assert s.close() == 0 and s.proc.poll() == 0                  # no hang (jsdom keeps timers alive otherwise)


def test_startup_error_fails_every_run_of_the_session(node_bundle_root, headless_out, tmp_path):
    """GPT24's counterexample: an unexpected console error while the harness loads must not leave a run OK."""
    bad = tmp_path / "startup_error.js"
    bad.write_text("console.error('unexpected error at start-up');")
    s = NodeSession(node_bundle_root, extra_files=[bad])
    try:
        sc = _small("harness/node_startup_error", (load(),), (auto(),))
        res = run(s, sc, headless_out / "controls", test_ref=__file__)
        pol = [o for o in res.outcomes if o.check == "console error"]
        assert not res.ok and pol and "unexpected error at start-up" in pol[0].detail, res.text()
    finally:
        s.close()


# ---------------------------------------------------------------------------------------------- D: isolation
def test_globals_do_not_leak_between_documents(node_session, headless_out):
    first = _small("harness/node_isolation_sets_a_global",
                   (load(), js_action("window.__leak = 42; setTimeout(() => {}, 60000);", until="window.__leak === 42")),
                   (js_value("window.__leak", 42, steps=(1,)),))
    second = _small("harness/node_isolation_reads_the_global",
                    (load(), js_action("window.__seen = typeof window.__leak;", until="true")),
                    (js_value("window.__seen", "undefined", steps=(1,)),))
    r1 = run(node_session, first, headless_out, test_ref=__file__)
    r2 = run(node_session, second, headless_out, test_ref=__file__)
    assert r1.ok and r2.ok, r1.text() + r2.text()
    assert "__leak" in r2.isolation["globals_removed"] and r2.isolation["leftover_timers_cleared"] >= 1, r2.isolation


# ---------------------------------------------------------------------------------------------- E: 4.2c-1 review notes
# A-1: the limits of per-document isolation (Claude10's probe cases as controls)
def test_non_removable_global_is_an_error_in_the_next_document(node_bundle_root, headless_out):
    """Own session: a non-removable global cannot be removed, so it poisons every later document of its session
    (which is exactly what the ERROR reports)."""
    s = NodeSession(node_bundle_root)
    try:
        first = _small("harness/node_nc_sets", (load(), js_action(
            "Object.defineProperty(window, '__nc', {value: 42, configurable: false, writable: true});", until="true")),
            (js_value("window.__nc", 42, steps=(1,)),))
        second = _small("harness/node_nc_next", (load(),), (auto(),))
        out = headless_out / "controls"
        run(s, first, out, test_ref=__file__)
        r = run(s, second, out, test_ref=__file__)
        iso = [o for o in r.outcomes if o.check == "isolation"]
        assert not r.ok and iso and "__nc" in iso[0].detail, r.text()
    finally:
        s.close()


def test_fresh_window_isolates_everything(node_bundle_root, headless_out):
    """With a fresh window per document (on by default for fault sessions) no state survives: not a non-removable
    global, not a changed built-in (Date.now), not work queued through a MessageChannel."""
    s = NodeSession(node_bundle_root, fresh_window=True)
    out = headless_out / "controls"
    try:
        a = _small("harness/node_fresh_sets", (load(), js_action(
            "Object.defineProperty(window, '__nc', {value: 42, configurable: false}); Date.now = () => 0;", until="true")),
            (js_value("Date.now()", 0, steps=(1,)),))
        b = _small("harness/node_fresh_reads", (load(), js_action("window.__seen = [typeof window.__nc, Date.now() > 1e12];",
                                                                      until="true")),
                   (js_value("window.__seen", ["undefined", True], steps=(1,)),))
        assert run(s, a, out, test_ref=__file__).ok
        r = run(s, b, out, test_ref=__file__)
        assert r.ok and r.isolation["fresh_window"], r.text()
    finally:
        s.close()


def test_message_channel_work_does_not_reach_the_next_document(node_session, headless_out):
    queue = _small("harness/node_mc_queues", (load(), js_action(
        "const t0 = performance.now(); const ch = new MessageChannel();"
        "ch.port1.onmessage = () => { if (performance.now() - t0 < 800) ch.port2.postMessage(0);"
        " else { window.__late = 1; } }; ch.port2.postMessage(0);", until="true")), (auto(),))
    nxt = _small("harness/node_mc_next", (load(), js_action("", until="window.__late === 1", timeout_s=1.5)),
                 (js_value("window.__late", 1, steps=(1,)),))
    out = headless_out / "controls"
    run(node_session, queue, out, test_ref=__file__)
    r = run(node_session, nxt, out, test_ref=__file__)
    step1 = [o for o in r.outcomes if o.step == 1]
    assert step1 and step1[0].outcome == "ERROR" and "completion condition not met" in step1[0].detail, r.text()
    assert r.isolation["channels_closed"] >= 1, r.isolation


# A-2: controls on RECORDED values (not on the helper functions)
def test_recorded_snapshot_keeps_nan_and_infinity_apart(node_session, headless_out):
    sc = _small("harness/node_recorded_special_values", (load(), js_action(
        "const m = window.__riH.model('h1', ['HistogramCDS']); m.get_column = () => [NaN, Infinity, -Infinity, -0, 0];",
        until="true")), (js_value("1", 1, steps=(1,)),))
    r = run(node_session, sc, headless_out / "controls", test_ref=__file__)
    rec = r.probe_values["1"]
    assert '"NaN"' in rec and '"Infinity"' in rec and '"-Infinity"' in rec and '"-0"' in rec, rec


def test_recorded_mask_fingerprint_is_the_live_mask(node_session, headless_out):
    sc = _small("harness/node_recorded_mask", (load(), set_("a", [-1, 1])), (auto(),), widget=True)
    r = run(node_session, sc, headless_out / "controls", test_ref=__file__)
    recorded = r.probe_values["1:selected"]
    live = node_session.request("eval", code="""() => { const f = window.__riH.model('h1', ['HistogramCDS']).filter;
        const v = Array.from(f.v_compute()).slice(0, 1000); const flipped = v.slice(); flipped[0] = flipped[0] ? 0 : 1;
        return [window.__riH.maskHash(v), window.__riH.maskHash(flipped)]; }""")["result"]
    assert recorded["mask_fnv"] == live[0] and live[1] != live[0], (recorded, live)
    assert recorded["mask_fnv"].endswith(":1000") and len(recorded["mask_fnv"]) == 21       # 64-bit + row count


# A-6: a model that reacts two timer rounds late — the settle rule accepts the stale state, the audit catches it
def test_late_update_is_caught_by_the_audit(node_session, headless_out):
    sc = _small("harness/node_late_update_two_rounds",
                (load(), js_action("setTimeout(() => setTimeout(() => { window.__riH.slider('a').value = [-1, 1]; }, 0), 0);",
                                   until="true")),
                (auto(), audit(0.3, steps=(1,))), widget=True)
    r = run(node_session, sc, headless_out / "controls", test_ref=__file__)
    aud = [o for o in r.outcomes if o.check.startswith("audit")]
    assert aud and aud[0].outcome == "ERROR" and not r.ok, r.text()
