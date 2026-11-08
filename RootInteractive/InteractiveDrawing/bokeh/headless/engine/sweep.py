"""Mass cases for the Node model harness by template substitution (proposal v0.3 erratum V-1; stage 4.2c-2a).

One page document per configuration is built by the real Python builder (page.build_document); for every case only
its data are replaced — re-encoded with RootInteractive's own compression (Tools.compressArray.compressCDSPipe) and
Bokeh's serializer — and Node builds a FRESH document from it (batch_docs). No Python page build per case.

Measured (2026-11-08, coder): exact only for pages WITHOUT widgets — a range widget's start, end, step and value are
computed from the data at build time, so pages with widgets (filter families, load-time behaviour) are document cases
built by the Python builder per case. `content()` compares two documents model by model with ids and definition order
ignored, and the toolbar's tools as a set (Bokeh merges them in no fixed order, also between two fresh builds of the
same data); `equivalent()` is the gate that a substituted document equals a fresh build for the same data.
"""
import copy
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd

from . import page as page_mod
from .scenario import Scenario


def _cds_nodes(o, out: list) -> list:
    if isinstance(o, dict):
        if o.get("type") == "object" and o.get("name") == "CDSCompress":
            out.append(o)
        for v in o.values():
            _cds_nodes(v, out)
    elif isinstance(o, list):
        for v in o:
            _cds_nodes(v, out)
    return out


def content(doc: dict) -> Counter:
    """Every model definition with nested definitions and references reduced to the referenced model's type: two
    documents are equal for Bokeh if these multisets are equal (ids and definition order do not matter)."""
    defs = {}

    def collect(o):
        if isinstance(o, dict):
            if o.get("type") == "object" and "attributes" in o:
                defs[o["id"]] = o
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)
    collect(doc)

    def flat(o):
        if isinstance(o, dict):
            if "id" in o and (o.get("type") == "object" or set(o) == {"id"}):
                return "REF:" + defs.get(o["id"], {}).get("name", "?")
            return {k: flat(v) for k, v in o.items()}
        if isinstance(o, list):
            return [flat(v) for v in o]
        return o
    out = Counter()
    for d in defs.values():
        attrs = flat(d["attributes"])
        if d.get("name") == "Toolbar" and isinstance(attrs.get("tools"), list):
            attrs["tools"] = sorted(attrs["tools"])          # Bokeh merges toolbar tools in no fixed order (UI only)
        out[d.get("name", "") + json.dumps(attrs, sort_keys=True)] += 1
    return out


class Template:
    """A page document built once; `document(df)` returns the JSON text of the same page with `df`'s data."""

    def __init__(self, scenario: Scenario, df0: pd.DataFrame, path: Path):
        if scenario.page.widgets:
            raise ValueError("template substitution is exact only for pages without widgets (widget limits are "
                             "computed from the data at build time): use document cases")
        self.scenario = scenario
        self.doc = json.loads(page_mod.build_document(scenario, df0, path).read_text(encoding="utf-8"))
        nodes = _cds_nodes(self.doc, [])
        if len(nodes) != 1:
            raise ValueError(f"expected one CDSCompress source in the page, found {len(nodes)}")
        self.columns = [k for k, _ in nodes[0]["attributes"]["inputData"]["entries"]]
        self.compression = [(pat, list(a)) for pat, a in scenario.transport.array_compression]
        # the template serialised once, split around the two values that change per case (no deep copy per case)
        marked = copy.deepcopy(self.doc)
        node = _cds_nodes(marked, [])[0]
        node["attributes"]["inputData"], node["attributes"]["sizeMap"] = "@@RI_INPUT@@", "@@RI_SIZE@@"
        text = json.dumps(marked)
        a, rest = text.split('"@@RI_INPUT@@"')
        b, c = rest.split('"@@RI_SIZE@@"')
        self._parts = (a, b, c)

    def document(self, df: pd.DataFrame) -> str:
        from bokeh.core.serialization import Serializer
        from RootInteractive.Tools.compressArray import compressCDSPipe
        data, size_map = compressCDSPipe(df[self.columns], self.compression, 0)
        enc = Serializer().encode
        a, b, c = self._parts
        return a + json.dumps(enc(data)) + b + json.dumps(enc(size_map)) + c

    def equivalent(self, df: pd.DataFrame, path: Path) -> bool:
        """The gate of erratum V-1: the substituted document equals a fresh Python build for the same data."""
        fresh = json.loads(page_mod.build_document(self.scenario, df, path).read_text(encoding="utf-8"))
        return content(json.loads(self.document(df))) == content(fresh)


def run_batch(session, docs: Sequence[str], code: str, chunk: int = 200) -> Tuple[List, Dict[str, float]]:
    """Run `code` on a fresh document for each JSON text in `docs`; returns the decoded results and timings."""
    from .node_session import decode
    out, t0 = [], time.perf_counter()
    for i in range(0, len(docs), chunk):
        r = session.request("batch_docs", docs=list(docs[i:i + chunk]), code=code, timeout_s=600)
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        if r.get("page_errors") or r.get("console_errors"):
            raise RuntimeError(f"errors during the batch: {r.get('page_errors')} {r.get('console_errors')}")
        for item in r["result"]:
            if item["isolation"]["not_removed"]:
                raise RuntimeError(f"isolation: non-removable globals {item['isolation']['not_removed']}")
            out.append(decode(item["result"]))
    return out, {"node_s": time.perf_counter() - t0, "cases": len(docs)}


def substitute_all(template: Template, frames: Sequence[pd.DataFrame]) -> Tuple[List[str], float]:
    t0 = time.perf_counter()
    docs = [template.document(df) for df in frames]
    return docs, time.perf_counter() - t0
