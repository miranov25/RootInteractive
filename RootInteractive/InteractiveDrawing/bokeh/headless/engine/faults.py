"""Deliberate faults (proposal v0.2 §12; stage 4.2b): reversible errors injected into a disposable copy of a
generated page — never into the repository — to prove that the checks catch the kind of error they are meant for.

Rules:
- each fault is an exact search-and-replace on the compiled code inside the page, and it must match **exactly once**;
  zero or several matches is an ERROR ("fault not applied"), never a silent run without the fault;
- the replacement increments `window.__riFault` when the mutated code runs, so the report proves the mutated path
  was executed (a fault on a path the page never runs would prove nothing);
- every fault is judged against a **healthy control**: the same scenario without the fault must pass first; a fault
  on a path already broken by a known defect is "inconclusive", not "detected".

The targets are compiled TypeScript of RootInteractive 4e8d783 as Bokeh 3.3.x emits it; a new Bokeh version that
changes the text makes the fault ERROR ("not applied") rather than pass.
"""
from dataclasses import dataclass

COUNTER = "window.__riFault = (window.__riFault || 0) + 1"


class FaultNotApplied(RuntimeError):
    pass


@dataclass(frozen=True)
class Fault:
    id: str
    description: str
    source: str            # the TypeScript line the compiled text comes from
    search: str
    replace: str


UPDATE_NEVER_TRIGGERED = Fault(
    id="update_never_triggered",
    description="a range widget's move no longer reaches its filter: RangeFilter does not connect its range change",
    source="RangeFilter.ts 53",
    search="this.connect(this.properties.range.change, this.mark_dirty_widget);",
    replace=f"{COUNTER};",
)

ND_BIN_SHIFTED = Fault(
    id="nd_bin_shifted",
    description="the N-D histogram puts every in-range value one bin higher on each axis",
    source="HistoNdCDS.ts 343",
    search="bins[i] += ((val * scale - origin) | 0) * stride;",
    replace=f"bins[i] += (((val * scale - origin) | 0) + ({COUNTER}, 1)) * stride;",
)

FAULTS = {f.id: f for f in (UPDATE_NEVER_TRIGGERED, ND_BIN_SHIFTED)}


def apply(text: str, fault: Fault) -> str:
    n = text.count(fault.search)
    if n != 1:
        raise FaultNotApplied(f"fault {fault.id!r}: target text found {n} times (must be exactly once); "
                              f"source {fault.source}")
    return text.replace(fault.search, fault.replace)
