"""The model bundle for the Node model harness (proposal v0.3 §3.1, erratum V-4; stage 4.2c-1).

The Node harness loads, into jsdom: the pinned `pako` (engine/vendor), Bokeh's own bundles from the installation
(`bokeh.min.js`, `bokeh-widgets.min.js`, `bokeh-tables.min.js`), and RootInteractive's compiled models as produced by
Bokeh's public `bundle_all_models()` — the same compiled code a page receives.

`bundle_all_models()` compiles every model (about 30-60 s). The finished bundle is therefore stored as one file under
the engine's git-ignored output folder, keyed by:
  - the Bokeh version,
  - the content of every RootInteractive `.ts` source (models and the helpers they import by relative path),
  - the list of model names being bundled (an added import changes no `.ts` file but changes the bundle).
It is written to a temporary file and renamed, so parallel processes never read half a file.

After loading, the Node runner checks the inventory: every class of EXPECTED_MODELS must be registered; a missing
one is an ERROR, never a silent gap. `CustomWasmFunc` and `CustomWasmModule` are not bundled because the page builder
does not import them; their functions are tested in tier T1 and WASM inside a page in tier T3 (erratum V-4).
"""
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

from . import vendor

BOKEH_DIR = Path(__file__).resolve().parents[2]                 # RootInteractive/InteractiveDrawing/bokeh
EXPECTED_MODELS = (
    "BokehVisJSGraph3D", "CDSAlias", "CDSCompress", "CDSJoin", "CDSStack", "ClientLinearFitter", "ColumnFilter",
    "ConcatenatedString", "CustomJSNAryFunction", "DownsamplerCDS", "HistoNdCDS", "HistoNdProfile", "HistoStatsCDS",
    "HistogramCDS", "LazyIntersectionFilter", "LazyTabs", "MultiSelectFilter", "OrtFunction", "RIFilter", "RangeFilter",
)
BOKEH_BUNDLES = ("bokeh.min.js", "bokeh-widgets.min.js", "bokeh-tables.min.js")


def bokeh_js_dir() -> Path:
    import bokeh
    return Path(bokeh.__file__).resolve().parent / "server" / "static" / "js"


def _custom_model_names() -> List[str]:
    """Names of the custom models Bokeh would bundle now (cheap: no compilation)."""
    import RootInteractive.InteractiveDrawing.bokeh.bokehTools  # noqa: F401  registers the models
    from bokeh.util.compiler import _get_custom_models
    return sorted(m.name for m in (_get_custom_models(None) or {}).values())


def cache_key(bokeh_version: str, sources: Dict[str, str], model_names: Sequence[str]) -> str:
    """Pure function of the three inputs of erratum V-4 (unit-tested)."""
    h = hashlib.sha256()
    h.update(f"bokeh {bokeh_version}\n".encode())
    for name in sorted(sources):
        h.update(f"source {name} {hashlib.sha256(sources[name].encode()).hexdigest()}\n".encode())
    for m in sorted(model_names):
        h.update(f"model {m}\n".encode())
    return h.hexdigest()


def current_key() -> str:
    import bokeh
    sources = {p.name: p.read_text(encoding="utf-8") for p in sorted(BOKEH_DIR.glob("*.ts"))}
    return cache_key(bokeh.__version__, sources, _custom_model_names())


def model_bundle(out_root: Path) -> Path:
    """The stored RootInteractive model bundle for the current key; built (atomically) if absent."""
    key = current_key()
    folder = Path(out_root) / "node_bundle"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"ri_models_{key[:16]}.js"
    if target.exists():
        return target
    from bokeh.util.compiler import bundle_all_models
    js = bundle_all_models() or ""
    if not js:
        raise RuntimeError("bundle_all_models() returned nothing: no RootInteractive model registered")
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(js)
    os.replace(tmp, target)                                            # atomic on the same file system
    return target


def bundle_files(out_root: Path, models: Path = None) -> List[Path]:
    """The files the Node runner loads, in order: pako, Bokeh's bundles, RootInteractive's models."""
    pako = [p.main for p in vendor.PINS if p.package == "pako"][0]
    return [pako] + [bokeh_js_dir() / b for b in BOKEH_BUNDLES] + [models or model_bundle(out_root)]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
