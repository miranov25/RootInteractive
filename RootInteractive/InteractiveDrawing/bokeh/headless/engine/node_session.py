"""Python side of the Node model harness (tier T2; proposal v0.3 §3; stage 4.2c-1).

`NodeSession` starts one Node.js process (engine/node/runner.mjs) that loads Bokeh and RootInteractive's compiled
models into jsdom once; `NodePage` offers the four page operations the engine's step loop uses — goto (here: build a
fresh document from a page document through Bokeh's deserializer), wait_for_function, evaluate, wait_for_timeout — so
the same steps, probes, oracles, signatures and reports run against Node and against Chromium.

A fault (engine/faults.py) is applied to a disposable copy of the model bundle when the session starts; that copy is
never the stored healthy bundle. Numbers come back as base64 Float64 arrays, so NaN, +-Inf and -0 survive.
"""
import base64
import json
import os
import platform
import tempfile
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import faults as faults_mod
from . import node_bundle

RUNNER = Path(__file__).resolve().parent / "node" / "runner.mjs"
JSDOM_PIN = "24.1.3"                                         # proposal v0.3 D-26; checked when required (conftest)


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file and a rename, so a parallel process never reads half a file (4.2c-1 v0.2)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


class NodeUnavailable(RuntimeError):
    """Node.js or jsdom missing, or the harness could not start."""


INVARIANT_PREFIX = "RI invariant violated: "        # engine/node/runner.mjs checkInvariants, __riH.invariants


class InvariantViolated(AssertionError):
    """A page broke a read-only invariant of the page helper (PHASE_5 Batch 1, review C1): an intersection filter
    selected a position past its table's last row, or an unweighted histogram counted more entries than its table has
    rows. Also raised when the check could not read a current result, or failed itself (Batches 2A/2B v0.2, review
    F3). Raised by NodeSession.request for every request, so no test can read past it."""


def decode(x: Any) -> Any:
    if isinstance(x, dict):
        if "__f64" in x:
            return np.frombuffer(base64.b64decode(x["__f64"]), dtype=np.float64).tolist()
        if "__num" in x:
            return {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf"), "-0": -0.0}[x["__num"]]
        return {k: decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [decode(v) for v in x]
    return x


class NodeSession:
    backend = "node"

    def __init__(self, out_root: Path, fault: Optional[faults_mod.Fault] = None, timeout_s: float = 120.0,
                 extra_files: Sequence[Path] = (), fresh_window: Optional[bool] = None):
        """`extra_files`: scripts loaded after the helper — for tests of the start-up error policy only.
        `fresh_window`: build a completely fresh window before every document (default: on for fault sessions, so
        a fault's execution proof can never be touched by an earlier document — 4.2c-1 review note A-1)."""
        self.fresh_window = (fault is not None) if fresh_window is None else fresh_window
        from .runner import HELPER_JS, REPO_ROOT, _git          # local import: runner imports this module too
        out_root = Path(out_root)
        healthy = node_bundle.model_bundle(out_root)
        models, self.fault = healthy, fault
        if fault is not None:                                    # disposable copy; never shares the stored bundle
            mutated = faults_mod.apply(healthy.read_text(encoding="utf-8"), fault)
            models = healthy.parent / f"fault_{fault.id}_{healthy.name}"
            _write_atomic(models, mutated)
        helper = healthy.parent / "helper.js"
        _write_atomic(helper, HELPER_JS)
        files = node_bundle.bundle_files(out_root, models) + [Path(f) for f in extra_files]
        try:
            self.proc = subprocess.Popen(["node", str(RUNNER), str(helper)] + [str(f) for f in files],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         text=True, bufsize=1)
        except FileNotFoundError as exc:
            raise NodeUnavailable(f"node not found: {exc}")
        self._lines: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._next = 0
        ready = self._read(timeout_s)
        if not ready.get("ok"):
            raise NodeUnavailable(f"Node harness did not start: {ready}")
        info = ready["result"]
        self.load_ms = info["load_ms"]
        reg = self.request("registered", names=list(node_bundle.EXPECTED_MODELS))
        # errors raised while the harness loaded are part of every run of this session (4.2c-1 v0.2, GPT24)
        self.startup_page_errors = ready.get("page_errors", []) + reg.get("page_errors", [])
        self.startup_console_errors = ready.get("console_errors", []) + reg.get("console_errors", [])
        self.missing_models = sorted(set(node_bundle.EXPECTED_MODELS) - set(reg["result"]))
        self.versions = {
            "backend": "node (jsdom)", "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "bokeh": info["bokeh"], "node": info["node"], "jsdom": info["jsdom"],
            "machine": platform.machine(),
            "rootinteractive": f"{_git('rev-parse', '--short', 'HEAD')} ({_git('rev-parse', '--abbrev-ref', 'HEAD')})",
            "model_bundle_sha256": node_bundle.sha256_of(models)[:16],
            "model_bundle_key": node_bundle.current_key()[:16] + (f" + fault {fault.id}" if fault else ""),
        }
        self.exit_code: Optional[int] = None
        if os.environ.get("RI_HEADLESS_REQUIRED") == "1" and info["jsdom"] != JSDOM_PIN:   # note A-4: every session
            self.close()
            raise NodeUnavailable(f"jsdom {info['jsdom']} installed, {JSDOM_PIN} required (proposal v0.3 D-26) — run: "
                                  f"npm install   (in RootInteractive/InteractiveDrawing/bokeh)")

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put("")                                      # end of stream

    def _read(self, timeout_s: float) -> dict:
        try:
            line = self._lines.get(timeout=timeout_s)
        except queue.Empty:
            raise NodeUnavailable(f"no answer from the Node harness within {timeout_s} s")
        if not line:
            err = self.proc.stderr.read()[-800:] if self.proc.stderr else ""
            raise NodeUnavailable(f"the Node harness ended: {err}")
        return json.loads(line)

    def request(self, op: str, timeout_s: float = 60.0, **kw) -> dict:
        self._next += 1
        self.proc.stdin.write(json.dumps({"id": self._next, "op": op, **kw}) + "\n")
        self.proc.stdin.flush()
        reply = self._read(timeout_s)
        if reply.get("id") != self._next:
            raise NodeUnavailable(f"protocol out of step: sent {self._next}, got {reply.get('id')}")
        self.last_unchecked = reply.get("unchecked", [])   # review F3: what the general check could not verify
        broken = [e for e in reply.get("page_errors", []) if str(e).startswith(INVARIANT_PREFIX)]
        if broken:            # PHASE_5 Batch 1 (review C1): never silently accepted, whatever the test reads next
            raise InvariantViolated("; ".join(broken))
        return reply

    def close(self) -> int:
        """Ask the process to exit and wait; the exit code must be 0 (the harness never hangs)."""
        if self.exit_code is not None:
            return self.exit_code
        try:
            self.proc.stdin.write(json.dumps({"id": self._next + 1, "op": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.exit_code = self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
            self.exit_code = self.proc.wait(timeout=10)
        return self.exit_code

    def new_page(self, res) -> "NodePage":
        res.page_errors.extend(f"at harness start-up: {e}" for e in self.startup_page_errors)
        res.console_errors.extend(f"at harness start-up: {e}" for e in self.startup_console_errors)
        return NodePage(self, res)


class NodePage:
    """The page operations the engine's step loop uses, against one fresh document in the Node harness."""

    # no views in Node: the document is complete when Document.from_json has returned (see runner._drive)
    load_condition = "() => window.Bokeh && Bokeh.documents.length > 0"

    def __init__(self, session: NodeSession, res):
        self.session, self.res = session, res

    def _call(self, op: str, timeout_s: float = 60.0, **kw) -> Any:
        r = self.session.request(op, timeout_s=timeout_s, **kw)
        self.res.page_errors.extend(r.get("page_errors", []))
        self.res.console_errors.extend(r.get("console_errors", []))
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "Node harness error"))
        return decode(r.get("result"))

    def goto(self, target, timeout: Optional[int] = None) -> None:
        fresh = self._call("new_window", timeout_s=60) if self.session.fresh_window else None
        r = self._call("new_doc", path=str(target), timeout_s=(timeout or 60_000) / 1000)
        self.res.isolation = {"fresh_window": bool(fresh), "leftover_timers_cleared": r["leftover_timers"],
                              "channels_closed": r["closed_channels"], "globals_removed": r["removed_globals"],
                              "globals_not_removable": r["not_removed"]}
        if r["not_removed"]:                                     # note A-1: state this document inherits
            self.res.add(-1, "isolation", "isolation", "ERROR", "an earlier document left non-removable globals: "
                         + ", ".join(r["not_removed"]) + " (use a fresh window, NodeSession(fresh_window=True))",
                         assertion=False)

    def wait_for_function(self, code: str, timeout: int = 20_000) -> None:
        self._call("wait_for", code=code, timeout_ms=timeout, timeout_s=timeout / 1000 + 10)

    def evaluate(self, code: str, arg: Any = None) -> Any:
        return self._call("eval", code=code, arg=arg)

    def wait_for_timeout(self, ms: int) -> None:
        self._call("sleep", ms=ms, timeout_s=ms / 1000 + 10)
