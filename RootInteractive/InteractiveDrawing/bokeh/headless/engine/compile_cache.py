"""A disk cache for Bokeh's compilation of RootInteractive's TypeScript models, for the tests (prototype v0.1;
plan RI_Compile_Cache_For_Tests_Plan_v0_1_Claude10_RI_20261121).

Bokeh compiles every model, and every local file a model imports, with `bokeh.util.compiler.nodejs_compile`: one Node
process per file. Bokeh keeps the result only in memory, so every new Python process (every test group, every xdist
worker, every capture child) compiles the same unchanged files again. This module wraps `nodejs_compile` and stores
each compiled file on disk.

The key holds everything that can change the compiled result:
  - a format version of this cache;
  - the Bokeh version and the SHA-256 of Bokeh's compiler.js (it contains the TypeScript compiler and its settings);
  - the Node version;
  - the language and the file name;
  - the full text of the file;
  - the text of every source file in the file's folder: Bokeh compiles a TypeScript file as a program with type
    checking and reads the files it imports from disk, so a neighbour can change the result (a type error certainly).
An entry holds the key, the compiled code and deps, and the SHA-256 of those; one that does not verify is never used.
A compile error is never stored. Entries are written to a temporary file and renamed, so parallel processes never
read half a file. The folder must belong to the user and be writable by nobody else: cached JavaScript runs in
every page.

Switches: RI_COMPILE_CACHE=0 turns it off; RI_COMPILE_CACHE_DIR chooses the folder (default
$RI_HEADLESS_BUNDLE/compile_cache, else headless/_out/compile_cache).
Command line: python -m RootInteractive.InteractiveDrawing.bokeh.headless.engine.compile_cache warm|prune|stats
"""
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

FORMAT = "ri-compile-cache-1"
SOURCE_SUFFIXES = (".ts", ".tsx", ".js", ".mjs", ".less", ".css")
PRUNE_DAYS = 14
HERE = Path(__file__).resolve().parents[1]                       # headless/

STATS = {"hits": 0, "misses": 0, "compile_s": 0.0, "stored": 0, "rejected": 0}
_state = {"original": None, "folder": None, "fixed": None}


def default_folder() -> Path:
    if os.environ.get("RI_COMPILE_CACHE_DIR"):
        return Path(os.environ["RI_COMPILE_CACHE_DIR"])
    root = os.environ.get("RI_HEADLESS_BUNDLE") or os.environ.get("RI_HEADLESS_OUT") or str(HERE / "_out")
    return Path(root) / "compile_cache"


def _warn(message: str) -> None:
    print(f"[ri compile cache] {message}", file=sys.stderr)


def _sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def _fixed_inputs() -> dict:
    """Bokeh version, compiler.js and Node version: the same for the whole process."""
    if _state["fixed"] is None:
        import bokeh
        from bokeh.util import compiler
        js = Path(compiler.bokehjs_dir) / "js" / "compiler.js"
        _state["fixed"] = {"bokeh": bokeh.__version__, "compiler_js": _sha(js.read_bytes()),
                           "node": compiler.nodejs_version() or "none"}
    return _state["fixed"]


def folder_sources(file: Optional[str]) -> list:
    """[(name, sha256 of text)] of every source file in the folder of `file`; [] for code without a file."""
    if not file:
        return []
    folder = Path(file).resolve().parent
    if not folder.is_dir():
        return []
    return sorted((p.name, _sha(p.read_bytes())) for p in folder.iterdir()
                  if p.is_file() and p.suffix in SOURCE_SUFFIXES)


def cache_key(code: str, lang: str, file: Optional[str], fixed: dict, neighbours: list) -> str:
    """A pure function of every input that can change the compiled result (unit-tested)."""
    return _sha(json.dumps({"format": FORMAT, **fixed, "lang": lang, "file": file and str(Path(file).resolve()),
                            "code": code, "folder": neighbours}, sort_keys=True))


def _payload_sha(code, deps) -> str:
    return _sha(json.dumps({"code": code, "deps": deps}, sort_keys=True))


def trusted(folder: Path) -> bool:
    """The folder belongs to this user and nobody else may write to it."""
    try:
        st = folder.stat()
    except OSError:
        return False
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        return False
    return not (st.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def _read(path: Path, key: str):
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("key") == key and entry.get("sha256") == _payload_sha(entry.get("code"), entry.get("deps")):
            return entry
    except FileNotFoundError:
        return None
    except (OSError, ValueError, AttributeError):
        pass
    STATS["rejected"] += 1
    _warn(f"a damaged entry is not used and is compiled again: {path.name}")
    return None


def _write(path: Path, key: str, result) -> None:
    entry = {"key": key, "code": result.get("code"), "deps": result.get("deps")}
    entry["sha256"] = _payload_sha(entry["code"], entry["deps"])
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem[:16], suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(entry, fh)
        os.replace(tmp, path)
        STATS["stored"] += 1
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _touch(path: Path) -> None:
    """Renew the entry's time at most once a day, so prune() keeps what is used."""
    try:
        if time.time() - path.stat().st_mtime > 86400:
            os.utime(path)
    except OSError:
        pass


def cached_compile(code: str, lang: str = "javascript", file: Optional[str] = None):
    original, folder = _state["original"], _state["folder"]
    key = cache_key(code, lang, file, _fixed_inputs(), folder_sources(file))
    path = folder / f"{key}.json"
    entry = _read(path, key)
    if entry is not None:
        STATS["hits"] += 1
        _touch(path)
        from bokeh.util.compiler import AttrDict
        out = {"code": entry["code"]}
        if entry.get("deps") is not None:
            out["deps"] = entry["deps"]
        return AttrDict(out)
    t0 = time.perf_counter()
    result = original(code, lang=lang, file=file)
    STATS["compile_s"] += time.perf_counter() - t0
    STATS["misses"] += 1
    if "error" not in result:
        _write(path, key, result)
    return result


def install(folder: Optional[Path] = None) -> bool:
    """Wrap bokeh.util.compiler.nodejs_compile (idempotent). False, and nothing changed, if switched off or refused."""
    if os.environ.get("RI_COMPILE_CACHE", "1") == "0":
        return False
    from bokeh.util import compiler
    if _state["original"] is not None:
        return True
    folder = Path(folder) if folder else default_folder()
    try:
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        _warn(f"not used: cannot create {folder}: {exc}")
        return False
    if not trusted(folder):
        _warn(f"not used: {folder} belongs to another user or others may write to it")
        return False
    _state["original"], _state["folder"] = compiler.nodejs_compile, folder
    compiler.nodejs_compile = cached_compile
    return True


def uninstall() -> None:
    if _state["original"] is not None:
        from bokeh.util import compiler
        compiler.nodejs_compile = _state["original"]
        _state["original"] = _state["folder"] = None


def prune(folder: Optional[Path] = None, days: float = PRUNE_DAYS) -> int:
    folder = Path(folder) if folder else default_folder()
    limit, removed = time.time() - days * 86400, 0
    for p in folder.glob("*.json"):
        try:
            if p.stat().st_mtime < limit:
                p.unlink()
                removed += 1
        except OSError:
            pass
    for p in folder.glob("*.tmp"):                   # a temporary file older than an hour was left by a killed process
        try:
            if p.stat().st_mtime < time.time() - 3600:
                p.unlink()
        except OSError:
            pass
    return removed


def warm() -> dict:
    """Compile every RootInteractive model once, so the workers of a run find a full cache."""
    if not install():
        return dict(STATS, installed=False)
    import RootInteractive.InteractiveDrawing.bokeh.bokehTools  # noqa: F401  registers the models
    from bokeh.util.compiler import bundle_all_models
    t0 = time.perf_counter()
    bundle_all_models()
    return dict(STATS, installed=True, seconds=round(time.perf_counter() - t0, 1), folder=str(_state["folder"]))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if what == "warm":
        result = warm()
        result["pruned"] = prune()
        print(f"[ri compile cache] warm-up: {json.dumps(result)}")
    elif what == "prune":
        print(f"[ri compile cache] pruned {prune()} entries")
    else:
        f = default_folder()
        print(f"[ri compile cache] {f}: {len(list(f.glob('*.json')))} entries")
