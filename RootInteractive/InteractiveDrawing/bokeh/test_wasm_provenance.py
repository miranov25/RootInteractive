"""PHASE_4_4 W0 - the committed WASM binaries: provenance record (W-1) and interface (W-2). Tests only.

WASM (WebAssembly) is compiled code that the browser or Node runs; RootInteractive's C++ numerical functions reach the
client as two committed binaries built with emscripten from wasm/src: wasm/build/functions.wasm (the demo functions
fun1..fun5, each as a single-value and an array version) and wasm/build/wasm_conv.wasm (convolve1d/2d/3d).

W-1  Provenance. The two binaries, their two C++ sources and the Makefile are exactly the recorded files (size and
     SHA-256), so a binary can never be silently out of step with its source: changing either needs a deliberate
     update of RECORD. bokeh/functions.wasm, which the existing WASM tests load, is the built binary itself (a symbolic
     link), not a copy that could differ.
     NOT tested here: a byte-identical rebuild. The binaries were built on 2026-02-07 (Phase 0.1.D, commits b384f8d,
     fd73f7a, b1bad4d) with an emscripten version the Makefile pins as 3.1.51; alma2 has emscripten 7.0.0, whose output
     differs by about 60 bytes and exports one more runtime function (setThrew). Stage W1 checks instead that a 7.0.0
     rebuild computes the same values bit for bit.
W-2  Interface. Each binary's exported functions are exactly its declaration - the EXPORTED_FUNCTIONS list of its build
     rule (the Makefile for functions.wasm; for wasm_conv.wasm, which has no Makefile rule, the build command written in
     the comment of wasm_conv.cpp) - plus emscripten's own runtime exports; it imports exactly
     env.emscripten_notify_memory_growth; every exported function takes as many arguments as its C++ definition has
     parameters; every C function of the source is exported. Plus one recorded fact that stage W2 builds on: such a
     binary cannot be instantiated with an EMPTY import object, which is what RootInteractive's loader
     (CustomWasmModule.ts) passes (hypothesis W-H1).
Oracles: the C++ source and the Makefile, read as text independently of the binary; the binary is read by Node's
WebAssembly API (the same engine the browser tier uses).
"""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

BOKEH = Path(__file__).resolve().parent
WASM = BOKEH.parents[1] / "wasm"                       # RootInteractive/wasm

# ------------------------------------------------------------------------------------------- the provenance record
# Relative to RootInteractive/wasm: (bytes, sha256). Recorded in PHASE_4_4 W0 from the committed files.
RECORD = {
    "build/functions.wasm": (17449, "c3bce1d112f01a6562f9a65b21ec8ad1168c4bab17d0cd7ee8c283f2428fc394"),
    "build/wasm_conv.wasm": (9763, "0848051598321f86ad686b28744c49aef8258815c31a670de2f66532e6e33aa3"),
    "src/wasm_functions.cpp": (2922, "dc0cd221d08f27bd04a0a34bd9383db1156dbaeb3bd67ce043f99790b8e62d3c"),
    "src/wasm_conv.cpp": (5653, "94b19d18f3d42e7dd305ab0dc452735ec59d34563bcbbd18bc391abd508debc2"),
    "Makefile": (832, "7f7835ccf8ca54df5ecbb86864e57c662ee585926c2bd687e8a72abd591bf4da"),
}

# binary -> (its C++ source, the file holding its build command)
BINARIES = {
    "functions.wasm": ("src/wasm_functions.cpp", "Makefile"),
    "wasm_conv.wasm": ("src/wasm_conv.cpp", "src/wasm_conv.cpp"),
}
RUNTIME_EXPORTS = {                                    # emscripten's own exports of a STANDALONE_WASM --no-entry build
    "memory": "memory", "__indirect_function_table": "table", "_initialize": "function",
    "_emscripten_stack_restore": "function", "emscripten_stack_get_current": "function",
}
IMPORTS = [["env", "emscripten_notify_memory_growth", "function"]]   # required by ALLOW_MEMORY_GROWTH
LIBC_PARAMETERS = {"malloc": 1, "free": 1}             # standard C: void* malloc(size_t), void free(void*)

INSPECT = r"""
import fs from 'fs';
const bytes = fs.readFileSync(process.argv[1]);
const mod = new WebAssembly.Module(bytes);
const imports = WebAssembly.Module.imports(mod).map(i => [i.module, i.name, i.kind]);
const exports = WebAssembly.Module.exports(mod).map(e => [e.name, e.kind]);
let empty = null;                                      // RootInteractive's loader passes {} (CustomWasmModule.ts)
try { await WebAssembly.instantiate(bytes, {}); } catch (e) { empty = e.constructor.name + ": " + e.message; }
const stub = {};                                       // a no-op for every imported function
for (const [m, n, k] of imports) if (k === "function") { stub[m] = stub[m] || {}; stub[m][n] = () => {}; }
const { instance } = await WebAssembly.instantiate(bytes, stub);
const arity = {};                                      // a JS function's length = the WASM function's parameter count
for (const [n, k] of exports) if (k === "function") arity[n] = instance.exports[n].length;
process.stdout.write(JSON.stringify({ imports, exports, arity, empty_import_error: empty }));
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_mismatch(path: Path, size: int, sha: str) -> str:
    """'' if the file is exactly the recorded one, otherwise what differs."""
    if not path.is_file():
        return f"{path} is missing"
    got_size, got_sha = path.stat().st_size, _sha256(path)
    if (got_size, got_sha) == (size, sha):
        return ""
    return f"{path}: recorded {size} bytes sha256 {sha}, found {got_size} bytes sha256 {got_sha}"


def _inspect(path: Path) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.fail("node is required to read a WASM binary (Node's WebAssembly API) and was not found in PATH")
    r = subprocess.run([node, "--input-type=module", "-e", INSPECT, str(path)], capture_output=True, text=True,
                       timeout=60)
    if r.returncode != 0:
        pytest.fail(f"node could not read {path}: {r.stderr[:800]}")
    return json.loads(r.stdout)


def _declared_exports(build_text: str) -> list:
    """The function names of the EXPORTED_FUNCTIONS list in a build command, without emscripten's leading '_'."""
    joined = build_text.replace("\\\n", " ")           # make's line continuations
    m = re.search(r"""EXPORTED_FUNCTIONS="\[([^\]]*)\]\"""", joined)
    if m is None:
        pytest.fail("no EXPORTED_FUNCTIONS list found in the build command")
    return re.findall(r"'_(\w+)'", m.group(1))


def _cpp_parameters(source: str) -> dict:
    """{function: number of parameters} for every non-static function defined at the start of a line."""
    out = {}
    for m in re.finditer(r"^(?:double|float|void|int)\s+(\w+)\s*\(([^)]*)\)\s*\{", source, re.M):
        out[m.group(1)] = len([p for p in m.group(2).split(",") if p.strip() and p.strip() != "void"])
    return out


# ===================================================================================== W-1 provenance
@pytest.mark.feature("WASM.compile")
@pytest.mark.parametrize("rel", sorted(RECORD))
def test_w1_file_is_the_recorded_one(rel):
    """W-1: binary, source and Makefile are exactly the recorded files; any change needs a deliberate record update."""
    size, sha = RECORD[rel]
    problem = _record_mismatch(WASM / rel, size, sha)
    assert problem == "", (problem + "\nA binary and its source change together: rebuild, re-run the WASM tests, then "
                           "update RECORD in this file in the same commit.")


@pytest.mark.feature("WASM.compile")
def test_w1_bokeh_functions_wasm_is_the_built_binary():
    """W-1: bokeh/functions.wasm, which test_wasm_invariance.py loads, is a link to wasm/build/functions.wasm."""
    link = BOKEH / "functions.wasm"
    assert link.is_symlink(), f"{link} is not a symbolic link: it could be a copy that differs from the built binary"
    assert link.resolve() == (WASM / "build" / "functions.wasm").resolve()


# ===================================================================================== W-2 interface
@pytest.mark.feature("WASM.compile")
@pytest.mark.parametrize("binary", sorted(BINARIES))
def test_w2_exports_are_the_declared_functions_plus_the_runtime(binary):
    _, build_file = BINARIES[binary]
    declared = _declared_exports((WASM / build_file).read_text())
    want = {**{n: "function" for n in declared}, **RUNTIME_EXPORTS}
    got = dict(map(tuple, _inspect(WASM / "build" / binary)["exports"]))
    assert got == want, (f"missing: {sorted(set(want) - set(got))}, unexpected: {sorted(set(got) - set(want))}, "
                         f"wrong kind: {sorted(n for n in set(got) & set(want) if got[n] != want[n])}")


@pytest.mark.feature("WASM.compile")
@pytest.mark.parametrize("binary", sorted(BINARIES))
def test_w2_imports_are_exactly_the_memory_growth_notification(binary):
    assert _inspect(WASM / "build" / binary)["imports"] == IMPORTS


@pytest.mark.feature("WASM.compile")
@pytest.mark.parametrize("binary", sorted(BINARIES))
def test_w2_every_function_has_the_cpp_parameter_count_and_every_cpp_function_is_exported(binary):
    source_file, build_file = BINARIES[binary]
    params = _cpp_parameters((WASM / source_file).read_text())
    declared = _declared_exports((WASM / build_file).read_text())
    assert params, f"no function definitions parsed from {source_file}"
    assert sorted(set(params) - set(declared)) == [], "C functions of the source that the build does not export"
    expected = {n: params.get(n, LIBC_PARAMETERS.get(n)) for n in declared}
    assert None not in expected.values(), f"declared but defined nowhere: {[n for n, v in expected.items() if v is None]}"
    arity = _inspect(WASM / "build" / binary)["arity"]
    assert {n: arity.get(n) for n in declared} == expected


@pytest.mark.feature("WASM.compile")
@pytest.mark.parametrize("binary", sorted(BINARIES))
def test_w2_fact_an_empty_import_object_cannot_instantiate_the_binary(binary):
    """The fact behind hypothesis W-H1: CustomWasmModule.ts instantiates with {} - this binary refuses that, naming
    the missing 'env' module. (Whether RootInteractive's loader then fails is tested through the production class in
    stage W2.)"""
    error = _inspect(WASM / "build" / binary)["empty_import_error"]
    assert error is not None and "env" in error, error


# ===================================================================================== the checks can fail
def _tiny_module(with_import: bool) -> bytes:
    """A hand-assembled module: f(a: f64, b: f64) -> a + b, optionally importing env.g."""
    b = bytearray(b"\x00asm\x01\x00\x00\x00")
    b += bytes([1, 7, 1, 0x60, 2, 0x7C, 0x7C, 1, 0x7C])                              # type: (f64, f64) -> f64
    if with_import:
        b += bytes([2, 9, 1, 3]) + b"env" + bytes([1]) + b"g" + bytes([0, 0])         # import env.g, type 0
    b += bytes([3, 2, 1, 0])                                                           # one function of type 0
    b += bytes([7, 5, 1, 1]) + b"f" + bytes([0, 1 if with_import else 0])              # export "f"
    b += bytes([10, 9, 1, 7, 0, 0x20, 0, 0x20, 1, 0xA0, 0x0B])                       # local.get 0, 1; f64.add
    return bytes(b)


def test_the_inspector_reads_a_known_module_and_the_empty_import_fact_can_be_false(tmp_path):
    """Control: the Node inspector reports what a hand-made module really has, and the W-H1 fact is not built in -
    a module without imports does instantiate with {}."""
    plain, importing = tmp_path / "plain.wasm", tmp_path / "importing.wasm"
    plain.write_bytes(_tiny_module(False))
    importing.write_bytes(_tiny_module(True))
    p, i = _inspect(plain), _inspect(importing)
    assert p == {"imports": [], "exports": [["f", "function"]], "arity": {"f": 2}, "empty_import_error": None}
    assert i["imports"] == [["env", "g", "function"]] and i["arity"] == {"f": 2}
    assert i["empty_import_error"] is not None and "env" in i["empty_import_error"]


def test_the_record_check_catches_one_changed_byte(tmp_path):
    original = tmp_path / "x.wasm"
    original.write_bytes(_tiny_module(False))
    size, sha = original.stat().st_size, _sha256(original)
    assert _record_mismatch(original, size, sha) == ""
    changed = bytearray(original.read_bytes())
    changed[-2] ^= 0x01
    original.write_bytes(bytes(changed))
    assert "found" in _record_mismatch(original, size, sha)
    assert "missing" in _record_mismatch(tmp_path / "absent.wasm", size, sha)


def test_the_declaration_and_signature_parsers_read_the_build_text():
    """Control on literal text: a make continuation inside the list, a two-line signature, a static helper skipped."""
    make = "EMFLAGS = -O2 -s EXPORTED_FUNCTIONS=\"['_a','_b',\\\n          '_c_v','_malloc']\" -s X=1\n"
    assert _declared_exports(make) == ["a", "b", "c_v", "malloc"]
    src = ("static inline int clamp(int v, int m) {\n}\n"
           "double a(double x, double y) {\n}\n"
           "void c_v(const double* p, double* out,\n          int n) {\n}\n"
           "int none(void) {\n}\n")
    assert _cpp_parameters(src) == {"a": 2, "c_v": 3, "none": 0}
