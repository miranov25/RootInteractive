"""PHASE_4_4 W2 - RootInteractive's own WASM classes, CustomWasmModule and CustomWasmFunc. Tests only.

What they are. RootInteractive's real path from a table column to compiled C++ code (the binaries tested in W0 are only
demo and benchmark examples):
  CustomWasmModule (.py/.ts)  holds a WASM binary as base64 text; in the browser it decodes and instantiates it.
  CustomWasmFunc   (.py/.ts)  a column function like CustomJSNAryFunction: copy the input columns into WASM memory,
                              call one C function of the module, return the result.

What is tested. A RootInteractive model reaches the browser only after Bokeh's compiler has compiled its TypeScript, with
the function a page build calls (bokeh.util.compiler.bundle_models). Invariant: every RootInteractive model compiles with
the pinned Bokeh. Healthy control: CustomJSNAryFunction, a column function of the same kind, compiles in the same call.
Each class is compiled on its own, in a fresh Python process (Bokeh caches compiled bundles per process and keeps a
global model registry; the test process stays untouched).

Findings (provisional ids; exact signatures below - the strict checks accept only KnownDefect, raised only when the
compiler reports exactly these errors):
  W-D1  CustomWasmModule.ts does not compile, 9 errors. The namespace is misspelt `CustomWasmMosule` (line 4), so
        CustomWasmModule.Attrs and .Props do not exist (TS2702 at lines 12, 15, 18, 25, with TS4022, TS4031, TS4063),
        and then neither does the property `wasmBytes` (TS2339, line 32); TS7031 at line 25.
  W-D2  CustomWasmFunc.ts does not compile, 3 errors: `p.Array(String)` - core/properties has no `Array` (TS2339,
        line 34); the nullable function name used as an index (TS2538, line 46); the nullable dtype returned as a
        string (TS2322, line 94).
On a compile error Bokeh's bundle_models prints the errors and calls sys.exit(1): a page that included either class
would end the Python process building it (a script or a Jupyter kernel). Consequence for the plan: the run-time
hypotheses W-H2..W-H8 of the proposal cannot be tested on production code - they need a version that compiles,
which is a code change for the human developer.
W-9 fact: no RootInteractive module imports these classes (only CustomWasmFunc.py imports CustomWasmModule), and the
drawing API's function types are JavaScript (default), "onnx" and "linearFit" - no WASM. No page contains them today.
"""
import functools
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

BOKEH = Path(__file__).resolve().parent
PACKAGE = BOKEH.parents[1]                             # RootInteractive/
REPO = PACKAGE.parent


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


STRICT = dict(strict=True, raises=KnownDefect)

# (file, line, column, TypeScript error code), as Bokeh 3.3.2's compiler reports them
W_D1 = sorted([("CustomWasmModule.ts", 12, 43, 2702), ("CustomWasmModule.ts", 12, 43, 4022),
               ("CustomWasmModule.ts", 15, 15, 2702), ("CustomWasmModule.ts", 15, 15, 4031),
               ("CustomWasmModule.ts", 18, 31, 2702), ("CustomWasmModule.ts", 18, 31, 4063),
               ("CustomWasmModule.ts", 25, 17, 2702), ("CustomWasmModule.ts", 25, 43, 7031),
               ("CustomWasmModule.ts", 32, 44, 2339)])
W_D2 = sorted([("CustomWasmFunc.ts", 34, 18, 2339), ("CustomWasmFunc.ts", 46, 38, 2538),
               ("CustomWasmFunc.ts", 94, 82, 2322)])

COMPILE = r"""
import importlib, sys
from bokeh.util.compiler import bundle_models
name = sys.argv[1]
cls = getattr(importlib.import_module("RootInteractive.InteractiveDrawing.bokeh." + name), name)
js = bundle_models([cls])
print("COMPILED", len(js or ""))
"""


@functools.lru_cache(maxsize=None)
def _compile(name: str) -> dict:
    """Compile one RootInteractive model as a page build would; a broken harness fails the test, never a KnownDefect."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO)] + [p for p in [os.environ.get("PYTHONPATH")] if p])}
    r = subprocess.run([sys.executable, "-c", COMPILE, name], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=300)
    text = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + "\n" + r.stderr)
    errors = sorted((Path(f).name, int(line), int(col), int(code))
                    for f, line, col, code in re.findall(r"(\S+\.ts):(\d+):(\d+) - error TS(\d+)", text))
    compiled = r.returncode == 0 and re.search(r"^COMPILED [1-9]\d*$", r.stdout, re.M) is not None
    failed = r.returncode == 1 and "Compilation failed:" in r.stderr
    if not (compiled or failed):
        pytest.fail(f"the compile harness itself failed for {name} (exit {r.returncode}): {text[-1500:]}")
    return {"compiled": compiled, "errors": errors, "text": text}


def _is_signature(result: dict, signature: list) -> bool:
    return not result["compiled"] and result["errors"] == signature


# ===================================================================================== control
@pytest.mark.feature("WASM.integration")
def test_w2_control_a_column_function_of_the_same_kind_compiles():
    r = _compile("CustomJSNAryFunction")
    assert r["compiled"] and r["errors"] == [], r["text"][-2000:]


# ===================================================================================== W-D1 CustomWasmModule
@pytest.mark.feature("WASM.integration")
def test_signature_w_d1_custom_wasm_module_does_not_compile():
    r = _compile("CustomWasmModule")
    assert _is_signature(r, W_D1), r["text"][-3000:]


@pytest.mark.feature("WASM.integration")
@pytest.mark.xfail(**STRICT, reason="W-D1: CustomWasmModule.ts does not compile")
def test_strict_custom_wasm_module_compiles():
    r = _compile("CustomWasmModule")
    known_defect(_is_signature(r, W_D1), "W-D1: CustomWasmModule.ts does not compile (9 TypeScript errors)")
    assert r["compiled"], r["text"][-3000:]


# ===================================================================================== W-D2 CustomWasmFunc
@pytest.mark.feature("WASM.integration")
def test_signature_w_d2_custom_wasm_func_does_not_compile():
    r = _compile("CustomWasmFunc")
    assert _is_signature(r, W_D2), r["text"][-3000:]


@pytest.mark.feature("WASM.integration")
@pytest.mark.xfail(**STRICT, reason="W-D2: CustomWasmFunc.ts does not compile")
def test_strict_custom_wasm_func_compiles():
    r = _compile("CustomWasmFunc")
    known_defect(_is_signature(r, W_D2), "W-D2: CustomWasmFunc.ts does not compile (3 TypeScript errors)")
    assert r["compiled"], r["text"][-3000:]


# ===================================================================================== W-9 reachability (fact)
IMPORT = re.compile(r"^\s*(?:from\s+[\w.]*\b(\w+)\s+import|import\s+[\w.]*\b(\w+))", re.M)


def _importers(class_module: str) -> set:
    """RootInteractive source files (tests excluded) that import the module named `class_module`."""
    out = set()
    for p in PACKAGE.rglob("*.py"):
        if p.name.startswith("test_") or p.name == "conftest.py" or "tests" in p.parts:
            continue
        mods = {a or b for a, b in IMPORT.findall(p.read_text(encoding="utf-8", errors="replace"))}
        if class_module in mods:
            out.add(p.relative_to(PACKAGE).as_posix())
    return out


@pytest.mark.feature("WASM.integration")
def test_w9_fact_no_rootinteractive_module_uses_the_wasm_classes():
    assert _importers("CustomWasmFunc") == set()
    assert _importers("CustomWasmModule") == {"InteractiveDrawing/bokeh/CustomWasmFunc.py"}
    types = set(re.findall(r'transformType == "(\w+)"', (BOKEH / "bokehTools.py").read_text(encoding="utf-8")))
    assert types == {"onnx", "linearFit"}, types            # anything else becomes a JavaScript function


# ===================================================================================== the checks can fail
def test_the_signature_check_is_exact():
    sig = W_D2
    assert _is_signature({"compiled": False, "errors": list(sig)}, sig)
    assert not _is_signature({"compiled": False, "errors": sig[:-1]}, sig)                       # one error fewer
    assert not _is_signature({"compiled": False, "errors": sorted(sig + [("x.ts", 1, 1, 1)])}, sig)  # one more
    assert not _is_signature({"compiled": True, "errors": []}, sig)                                  # it compiles


def test_the_import_scan_finds_a_known_import():
    assert "InteractiveDrawing/bokeh/bokehTools.py" in _importers("OrtFunction")
