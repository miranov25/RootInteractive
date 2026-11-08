"""Mass cases for the Node model harness by template substitution (proposal v0.3 erratum V-1; stage 4.2c-2a).

One page document per configuration is built by the real Python builder (page.build_document); for every case only
its data are replaced — re-encoded with RootInteractive's own compression (Tools.compressArray.compressCDSPipe) and
Bokeh's serializer — and Node builds a FRESH document from it (batch_docs). No Python page build per case.

Measured (2026-11-08, coder): exact only for pages WITHOUT widgets — a range widget's start, end, step and value are
computed from the data at build time, so pages with widgets (filter families, load-time behaviour) are document cases
built by the Python builder per case. `content()` compares two documents as model graphs with ids and definition order
ignored, the toolbar's tools as a set (Bokeh merges them in no fixed order, also between two fresh builds), and which
model each reference points to kept; `equivalent()` is the gate that a substituted document equals a fresh build.
The toolbar's and the tool proxies' tool lists are compared as sets: Bokeh merges the tools of several plots in no
fixed order, also between two fresh builds of the same data (measured: 1-5 of 12 builds differ otherwise).
"""
import copy
import hashlib
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


UNORDERED = (("Toolbar", "tools"), ("ToolProxy", "tools"))     # lists Bokeh fills in no fixed order (UI tools only)


def content(doc: dict, rounds: int = 8, unordered=UNORDERED) -> Counter:
    """The documents' model graph as a multiset of labels that ignore generated ids, definition order and the
    toolbar's and tool proxies' tool order, but keep WHICH model each reference points to (4.2c-2a review, GPT23/GPT24: references
    reduced to their type could not see two plots with swapped data sources). Each model starts from its type and
    attributes with references blanked; in each round its label absorbs the labels of the models it references, at
    their attribute positions (the toolbar's tools as a set). Two documents are equal for Bokeh if the multisets are."""
    defs, neigh = {}, {}

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

    def blank(o, path, refs, unordered):
        if isinstance(o, dict):
            if "id" in o and (o.get("type") == "object" or set(o) == {"id"}):
                refs.append(("*" + path if unordered else path, o["id"]))
                return "@"
            return {k: blank(v, f"{path}/{k}", refs, unordered) for k, v in o.items()}
        if isinstance(o, list):
            return [blank(v, path if unordered else f"{path}[{i}]", refs, unordered) for i, v in enumerate(o)]
        return o

    label = {}
    for mid, d in defs.items():
        refs, attrs = [], {}
        for k, v in d["attributes"].items():
            attrs[k] = blank(v, k, refs, (d.get("name"), k) in unordered)
        neigh[mid] = refs
        label[mid] = d.get("name", "") + json.dumps(attrs, sort_keys=True)
    for _ in range(rounds):
        new = {}
        for mid in defs:
            ordered = [(p, label.get(r, "?")) for p, r in neigh[mid] if not p.startswith("*")]
            as_set = sorted((p, label.get(r, "?")) for p, r in neigh[mid] if p.startswith("*"))
            new[mid] = hashlib.sha256(json.dumps([label[mid], ordered, as_set]).encode()).hexdigest()
        label = new
    return Counter(label.values())


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
    """Run `code` on a fresh document for each JSON text in `docs`; returns the decoded results and timings.
    Refuses a session that had errors while starting (the start-up policy, as for single documents), and requires
    exactly one result per document, per chunk and in total (4.2c-2a review: a short reply was silently accepted)."""
    from .node_session import decode
    if getattr(session, "startup_page_errors", None) or getattr(session, "startup_console_errors", None):
        raise RuntimeError(f"errors at harness start-up: {session.startup_page_errors} {session.startup_console_errors}")
    out, t0 = [], time.perf_counter()
    for i in range(0, len(docs), chunk):
        part = list(docs[i:i + chunk])
        r = session.request("batch_docs", docs=part, code=code, timeout_s=600)
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        if r.get("page_errors") or r.get("console_errors"):
            raise RuntimeError(f"errors during the batch: {r.get('page_errors')} {r.get('console_errors')}")
        if len(r["result"]) != len(part):
            raise RuntimeError(f"batch starting at document {i}: {len(part)} documents sent, "
                               f"{len(r['result'])} results returned")
        for item in r["result"]:
            if item["isolation"]["not_removed"]:
                raise RuntimeError(f"isolation: non-removable globals {item['isolation']['not_removed']}")
            out.append(decode(item["result"]))
    if len(out) != len(docs):
        raise RuntimeError(f"{len(docs)} documents, {len(out)} results")
    return out, {"node_s": time.perf_counter() - t0, "cases": len(docs)}


def substitute_all(template: Template, frames: Sequence[pd.DataFrame]) -> Tuple[List[str], float]:
    t0 = time.perf_counter()
    docs = [template.document(df) for df in frames]
    return docs, time.perf_counter() - t0
