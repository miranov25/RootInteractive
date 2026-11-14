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

Defect faults (PHASE_5 Batch 1; B8 and B10 in Batch 2A). A fixed defect is put back into the disposable copy by a fault, so that the engine's
classification can still be proven on it: a known defect explained exactly is XFAIL, and an unrelated fault on top of
it is FAIL ("a different problem"), never XFAIL. A composite fault (`compose`) applies several faults, each exactly
once; the outcomes of the run show that each part acted (the defect's exact signature, and the unrelated failure).
"""
from dataclasses import dataclass
from typing import Tuple

COUNTER = "window.__riFault = (window.__riFault || 0) + 1"


class FaultNotApplied(RuntimeError):
    pass


@dataclass(frozen=True)
class Fault:
    id: str
    description: str
    source: str            # the TypeScript line the compiled text comes from
    search: str = ""
    replace: str = ""
    parts: Tuple["Fault", ...] = ()   # a composite fault: every part applied, each exactly once


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
    source="HistoNdCDS.ts 343 (the line as fixed for B10 in PHASE_5 Batch 2A)",
    search="bins[i] += (bin < nbins ? bin : nbins - 1) * stride;",
    replace=f"bins[i] += (((val * scale - origin) | 0) + ({COUNTER}, 1)) * stride;",
)

FAULTS = {f.id: f for f in (UPDATE_NEVER_TRIGGERED, ND_BIN_SHIFTED)}

# ---- defect faults: a fixed defect put back (PHASE_5 Batch 1); the text searched is the fixed line ----
B1_REINTRODUCED = Fault(
    id="b1_reintroduced",
    description="B1 put back: once one filter has changed, a later filter's counts are not updated (the old `||=`)",
    source="LazyIntersectionFilter.ts 197 (fixed in PHASE_5 Batch 1)",
    search="this._changed_values = from_bits_8(this.old_values[x], values_bits, counts) || this._changed_values;",
    replace=f"this._changed_values = ({COUNTER}, this._changed_values || "
            f"from_bits_8(this.old_values[x], values_bits, counts));",
)

B2_REINTRODUCED = Fault(
    id="b2_reintroduced",
    description="B2 put back: the 1-D histogram does not apply the selection when the page loads",
    source="HistogramCDS.ts 50 (fixed in PHASE_5 Batch 1)",
    search="this.change_selection(); // PHASE_5 Batch 1 (B2)",
    replace=f"{COUNTER}; // PHASE_5 Batch 1 (B2)",
)

# the findings C1 and C2 of the review of the Batch 1 fix v0.1, put back (falsifiers of their tests, Batch 1 v0.2)
PHANTOM_ROWS = Fault(
    id="phantom_rows",
    description="review C1 put back: the room after the table's last row in the packed selection is not cleared",
    source="LazyIntersectionFilter.ts v_compute (fixed in PHASE_5 Batch 1 v0.2)",
    search="for (let i = n_rows; i < new_vector.length; i++) {",
    replace=f"for (let i = n_rows; ({COUNTER}, false); i++) {{",
)

FILTER_CACHE_WRITTEN = Fault(
    id="filter_cache_written",
    description="review C2 put back: with 256 or more filters, disabling a filter sets every bit of its cached array",
    source="LazyIntersectionFilter.ts update_counts (fixed in PHASE_5 Batch 1 v0.2)",
    search="this.old_values[x] = null; // PHASE_5 Batch 1 v0.2 (review C2)",
    replace=f"{COUNTER}; old_values.fill(-1); // PHASE_5 Batch 1 v0.2 (review C2)",
)

# ---- PHASE_5 Batch 2A: B8 and B10 put back (the text searched is the fixed line) ----
B8_REINTRODUCED = Fault(
    id="b8_reintroduced",
    description="B8 put back: a 1-D histogram counts NaN (a missing value) in bin 0 - getbin's lower test is `val < min`",
    source="HistogramCDS.ts getbin (fixed in PHASE_5 Batch 2A)",
    search="if (!(val >= this._range_min))",
    replace=f"if (val < this._range_min || ({COUNTER}, false))",
)

B10_REINTRODUCED = Fault(
    id="b10_reintroduced",
    description="B10 put back: a value one rounding step below the maximum computes as bin nbins - dropped by a 1-D "
                "histogram, moved to bin 0 of the next row of bins (or dropped) by an N-D histogram",
    source="HistogramCDS.ts getbin and HistoNdCDS.ts compute_bins (fixed in PHASE_5 Batch 2A)",
    parts=(Fault(id="b10_reintroduced_1d", description="B10 put back in the 1-D histogram",
                 source="HistogramCDS.ts getbin (fixed in PHASE_5 Batch 2A)",
                 search="return bin < this._nbins ? bin : this._nbins - 1;", replace=f"return ({COUNTER}, bin);"),
           Fault(id="b10_reintroduced_nd", description="B10 put back in the N-D histogram",
                 source="HistoNdCDS.ts compute_bins (fixed in PHASE_5 Batch 2A)",
                 search="bins[i] += (bin < nbins ? bin : nbins - 1) * stride;",
                 replace=f"bins[i] += ({COUNTER}, bin) * stride;")),
)

DEFECT_FAULTS = {f.id: f for f in (B1_REINTRODUCED, B2_REINTRODUCED, PHANTOM_ROWS, FILTER_CACHE_WRITTEN,
                                   B8_REINTRODUCED, B10_REINTRODUCED)}


def compose(*faults: Fault) -> Fault:
    """One fault made of several; each part must match exactly once. The id joins the parts' ids with '+'."""
    flat = tuple(p for f in faults for p in (f.parts or (f,)))
    return Fault(id="+".join(p.id for p in flat), description="; ".join(p.description for p in flat),
                 source="; ".join(p.source for p in flat), parts=flat)


def apply(text: str, fault: Fault) -> str:
    for part in (fault.parts or (fault,)):
        n = text.count(part.search) if part.search else 0
        if n != 1:
            raise FaultNotApplied(f"fault {part.id!r}: target text found {n} times (must be exactly once); "
                                  f"source {part.source}")
        text = text.replace(part.search, part.replace)
    return text
