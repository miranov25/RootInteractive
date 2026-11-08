// Node model harness of the RootInteractive headless tests (tier T2; proposal v0.3 §3, stage 4.2c-1).
//
// Loads, ONCE per process, into a jsdom window: pako, Bokeh's bundles, RootInteractive's compiled models (Bokeh's own
// bundle_all_models()), and the engine's helper JavaScript (the same window.__riH the browser engine injects). Then it
// answers requests, one JSON object per line on stdin, one JSON reply per line on stdout:
//   {"id": n, "op": "new_doc", "path": "<document JSON file>"}   build a FRESH document through Bokeh's deserializer
//   {"id": n, "op": "eval", "code": "<function source>", "arg": <json>}  call a function inside the window (awaited)
//   {"id": n, "op": "wait_for", "code": "<function source>", "timeout_ms": t}  poll until the function returns truthy
//   {"id": n, "op": "sleep", "ms": t}                            let the window's timers run for t ms
//   {"id": n, "op": "selftest"}                                  the realm and construction rules (proposal §3.3)
//   {"id": n, "op": "batch_docs", "docs": ["<document JSON>", ...], "code": "<function source>"}
//        for each document: the same isolation as new_doc, a fresh document, then the function; returns all results
//        (mass cases by template substitution, proposal v0.3 erratum V-1; stage 4.2c-2a)
//   {"id": n, "op": "new_window"}                                a completely fresh window (bundle reloaded), used
//                                                                 before every document of a fault session (4.2c-2a)
//   {"id": n, "op": "quit"}                                      exit (process.exit; jsdom keeps timers alive)
// Every reply carries the page errors and console errors collected since the previous reply.
//
// Rules (proposal v0.3 §3.3):
//   1. realm: everything handed to Bokeh is created inside the window (arguments are parsed with window.eval);
//   2. construction: models only through Document.from_json, from documents the Python builder produced;
//   3. asynchronous callbacks: no fixed waits here; the engine's settle rule (helper JavaScript) decides completion.
// Isolation (4.2c-1 v0.2): the window lives for the whole process, so every new_doc first clears the timers the previous
// document left pending and removes every global added since the harness finished loading (the fault execution
// counter __riFault, globals set by a scenario's js_action, ...), and reports what it removed. Since 4.2c-2a it also
// closes every MessageChannel a page opened after the harness loaded (work queued there could otherwise run inside the
// next document). Documented limits: a page that changes a built-in (e.g. Date.now) or makes a global non-removable
// is not repaired here — a non-removable global is reported (and is an ERROR in the run), and fault sessions avoid all
// of it by building a fresh window per document (op new_window).
// Stand-ins (they concern drawing only; parity with the browser is their standing control): Node's MessageChannel,
// and an inert canvas context that accepts every call and draws nothing (Bokeh touches a canvas while loading).
import fs from "node:fs";
import { createRequire } from "node:module";
import readline from "node:readline";
import jsdomPkg from "jsdom";

const { JSDOM, VirtualConsole } = jsdomPkg;
const [helperPath, ...bundleFiles] = process.argv.slice(2);
const pageErrors = [], consoleErrors = [];

function inert() {
  const f = function () { return p; };
  const p = new Proxy(f, { get(_, k) { if (k === Symbol.toPrimitive) return () => 0; if (k === "then") return undefined;
                                        if (k === "length") return 0; return p; },
                           apply() { return p; }, set() { return true; }, construct() { return p; } });
  return p;
}

const pendingTimers = new Map();                                  // id -> clear function, for isolation
const channels = new Set();                                       // MessageChannels opened after the baseline
let trackChannels = false;
const vc = new VirtualConsole();
vc.on("jsdomError", e => pageErrors.push(String(e && (e.message || e)).slice(0, 300)));
vc.on("error", (...a) => consoleErrors.push(a.map(String).join(" ").slice(0, 300)));
process.on("unhandledRejection", e => pageErrors.push("unhandled rejection: " + String(e && (e.message || e)).slice(0, 280)));
process.on("uncaughtException", e => pageErrors.push("uncaught exception: " + String(e && (e.message || e)).slice(0, 280)));

const t0 = Date.now();
const scripts = bundleFiles.concat([helperPath]).map(f => `<script>${fs.readFileSync(f, "utf8").replace(/<\/script>/g, "<\\/script>")}</script>`);
const html = `<!DOCTYPE html><html><head></head><body>${scripts.join("\n")}</body></html>`;
let dom, w, Document, baseline;

function makeWindow() {
  if (dom) { try { dom.window.close(); } catch (e) { /* already closed */ } }
  pendingTimers.clear(); channels.clear(); trackChannels = false;
  dom = new JSDOM(html, {
    runScripts: "dangerously", virtualConsole: vc,
    beforeParse(win) {
      win.MessageChannel = class extends MessageChannel {             // stand-in: Node's MessageChannel, tracked
        constructor() { super(); if (trackChannels) channels.add(this); }
      };
      win.HTMLCanvasElement.prototype.getContext = () => inert();
      const [st, ct, si, ci] = [win.setTimeout, win.clearTimeout, win.setInterval, win.clearInterval];
      win.setTimeout = (fn, ms, ...a) => {                          // non-function callbacks pass through (4.2c-2a)
        const id = typeof fn === "function" ? st.call(win, (...b) => { pendingTimers.delete(id); fn(...b); }, ms, ...a)
                                            : st.call(win, fn, ms, ...a);
        pendingTimers.set(id, () => ct.call(win, id)); return id; };
      win.clearTimeout = id => { pendingTimers.delete(id); ct.call(win, id); };
      win.setInterval = (fn, ms, ...a) => { const id = si.call(win, fn, ms, ...a); pendingTimers.set(id, () => ci.call(win, id)); return id; };
      win.clearInterval = id => { pendingTimers.delete(id); ci.call(win, id); };
    },
  });
  w = dom.window;
  Document = w.Bokeh.require("document").Document;
  baseline = new Set(Object.getOwnPropertyNames(w));                // the window as the harness left it
  trackChannels = true;
}
makeWindow();
const loadMs = Date.now() - t0;

// numbers travel as base64 Float64 (NaN, +-Inf, -0 survive); other values as JSON
function enc(x) {
  if (x === null || x === undefined) return null;
  if (typeof x === "number") return Number.isFinite(x) && !Object.is(x, -0) ? x : { __num: Object.is(x, -0) ? "-0" : String(x) };
  if (ArrayBuffer.isView(x) || (Array.isArray(x) && x.length > 0 && x.every(v => typeof v === "number"))) {
    const f = Float64Array.from(x);
    return { __f64: Buffer.from(f.buffer).toString("base64") };
  }
  if (Array.isArray(x)) return x.map(enc);
  if (typeof x === "object") { const o = {}; for (const k of Object.keys(x)) o[k] = enc(x[k]); return o; }
  return x;
}
const inWindow = v => w.eval("(" + JSON.stringify(v === undefined ? null : v) + ")");     // rule 1: window realm
const fn = code => w.eval("(" + code + ")");

function freshDocument(text) {                                     // rule 2 + isolation, one place for both ops
  const leftoverTimers = pendingTimers.size;
  for (const clear of pendingTimers.values()) clear();
  pendingTimers.clear();
  const closedChannels = channels.size;
  for (const ch of channels) { try { ch.port1.close(); ch.port2.close(); } catch (e) { /* closed */ } }
  channels.clear();
  const removed = Object.getOwnPropertyNames(w).filter(k => !baseline.has(k));
  for (const k of removed) { try { delete w[k]; } catch (e) { /* not configurable */ } }
  const notRemoved = removed.filter(k => Object.prototype.hasOwnProperty.call(w, k));
  w.Bokeh.documents.length = 0;
  const doc = Document.from_json(w.JSON.parse(text));                // the window's JSON: window-realm objects
  w.Bokeh.documents.push(doc);
  return { models: doc._all_models.size, leftover_timers: leftoverTimers, removed_globals: removed,
           not_removed: notRemoved, closed_channels: closedChannels };
}

async function handle(req) {
  switch (req.op) {
    case "new_doc":
      return freshDocument(fs.readFileSync(req.path, "utf8"));
    case "batch_docs": {
      const f = fn(req.code), out = [];
      for (const text of req.docs) {
        const iso = freshDocument(text);
        let r = f();
        if (r && typeof r.then === "function") r = await r;
        out.push({ result: enc(r), isolation: { timers: iso.leftover_timers, channels: iso.closed_channels,
                                                not_removed: iso.not_removed } });
      }
      return out;
    }
    case "eval": {
      let r = fn(req.code)(inWindow(req.arg));
      if (r && typeof r.then === "function") r = await r;
      return enc(r);
    }
    case "wait_for": {
      const f = fn(req.code), t1 = Date.now();
      while (!(await f())) {
        if (Date.now() - t1 > req.timeout_ms) throw new Error(`condition not met within ${req.timeout_ms} ms`);
        await new Promise(r => w.setTimeout(r, 5));
      }
      return true;
    }
    case "sleep":
      await new Promise(r => w.setTimeout(r, req.ms));
      return true;
    case "new_window": {                                           // fault sessions: nothing survives (4.2c-2a)
      const t1 = Date.now();
      makeWindow();
      return { load_ms: Date.now() - t1 };
    }
    case "registered":                                               // inventory check (Python passes the names)
      return req.names.filter(n => { try { return !!w.Bokeh.Models.get(n); } catch (e) { return false; } });
    case "selftest": {
      const CDS = w.Bokeh.Models.get("ColumnDataSource"), HN = w.Bokeh.Models.get("HistoNdCDS");
      const out = {};
      try { new CDS({ data: { x: [1, 2] } }); out.node_realm = "accepted"; } catch (e) { out.node_realm = "rejected: " + String(e.message).slice(0, 120); }
      try { new CDS(w.eval("({data: {x: [1, 2]}})")); out.window_realm = "accepted"; } catch (e) { out.window_realm = "rejected: " + String(e.message).slice(0, 120); }
      try {                                                          // a VALID window-realm source (4.2c-1 v0.2)
        const src = new CDS(w.eval("({data: {x: [0.1, 0.5], y: [0.2, 0.7]}})"));
        const attrs = w.eval("({sample_variables: ['x', 'y'], nbins: [2, 2], range: [[0, 1], [0, 1]]})");
        attrs.source = src;
        new HN(attrs);
        out.direct_histo_nd = "constructed";
      } catch (e) { out.direct_histo_nd = "failed: " + String(e.message).slice(0, 160); }
      return out;
    }
    case "quit":
      reply({ id: req.id, ok: true, result: true });
      process.exit(0);
    default:
      throw new Error("unknown op " + req.op);
  }
}

function reply(o) {
  o.page_errors = pageErrors.splice(0);
  o.console_errors = consoleErrors.splice(0);
  process.stdout.write(JSON.stringify(o) + "\n");
}

let jsdomVersion = "unknown";
try { jsdomVersion = createRequire(import.meta.url)("jsdom/package.json").version; } catch (e) { /* export rules */ }
reply({ id: 0, ok: true, result: { ready: true, load_ms: loadMs, bokeh: w.Bokeh.version, node: process.version,
                                   jsdom: jsdomVersion } });

const rl = readline.createInterface({ input: process.stdin });
let queue = Promise.resolve();
rl.on("line", line => {
  queue = queue.then(async () => {
    let req;
    try { req = JSON.parse(line); } catch (e) { reply({ id: -1, ok: false, error: "bad request: " + e.message }); return; }
    try { reply({ id: req.id, ok: true, result: await handle(req) }); }
    catch (e) { reply({ id: req.id, ok: false, error: String(e && (e.message || e)).slice(0, 400) }); }
  });
});
rl.on("close", () => queue.then(() => process.exit(0)));
