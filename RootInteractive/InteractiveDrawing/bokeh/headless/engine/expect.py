"""Expected failures with machine-checked signatures, outcome classification and per-feature aggregation
(proposal v0.2 §4 `expect`, §8; stage 4.2b).

A known RootInteractive defect is declared on the check and the steps where it shows, with a *signature*: a predicate
over the page value and the oracle value that describes **how** the page is wrong. Classification of one check:

    check passes, no expectation          -> PASS
    check fails,  no expectation          -> FAIL
    check fails,  expectation, signature holds      -> XFAIL  (the known defect, exactly as described)
    check fails,  expectation, signature does not hold -> FAIL (a different problem on top of the defect)
    check passes, expectation              -> XPASS  (strict: the defect is gone or the expectation is stale;
                                                      the scenario fails until the expectation is removed)

A step that cannot complete is an ERROR and never an XFAIL. Signatures receive a context of independently computed
quantities (e.g. the histogram without any selection, the selection a known defect produces).

Rules for signatures (stage 4.2b v0.2, after the review of v0.1):
- use the **most specific** signature the known mechanism allows: the page must equal what the defect produces
  (`EqualsSelection`, `EqualsUnselected`, `ExcessInCells`). A direction-only `Mismatch` is allowed only while the
  mechanism is not yet known; its report line says "loose signature". (v0.1's B1 used `Mismatch` and let a page that
  ignored every widget move pass as B1.)
- a signature compares with the **tolerance of the check it classifies** (counts exactly; weighted sums within the
  check's absolute and relative tolerance), through the same comparison helper as the check;
- a page value that is not finite never matches a signature (only `Error` signatures concern messages, not values);
- one expectation per check and step (validated in scenario.py).
"""
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from .compare import compare


# ------------------------------------------------------------------------------------------------- signatures
Tol = Tuple[float, float]                       # (absolute, relative) tolerance of the classified check


def _equal(a, b, tol: Tol) -> bool:
    """Equality as the checks define it (engine/compare.py): shapes first, then cell by cell within the tolerance."""
    return compare(a, b, kind="float", abs_tol=tol[0], rel_tol=tol[1]).ok


@dataclass(frozen=True)
class Mismatch:
    """Loose signature, only while a defect's mechanism is unknown. `direction`: "page > oracle" (every cell that
    differs beyond the tolerance is above the oracle) or "page < oracle"; `cells`: "any" (at least one cell differs)
    or "all" (every cell with a non-zero oracle differs)."""
    direction: str
    cells: str = "any"

    def holds(self, page: np.ndarray, oracle: np.ndarray, ctx: dict, tol: Tol = (0.0, 0.0)) -> Tuple[bool, str]:
        if page.shape != oracle.shape:
            return False, f"shape {page.shape} vs {oracle.shape}"
        diff = page - oracle
        differing = np.abs(diff) > tol[0] + tol[1] * np.abs(oracle)
        if self.direction == "page > oracle":
            wrong = int((differing & (diff < 0)).sum())
        elif self.direction == "page < oracle":
            wrong = int((differing & (diff > 0)).sum())
        else:
            raise ValueError(f"unknown direction {self.direction!r}")
        if wrong:
            return False, f"{wrong} cell(s) deviate in the other direction"
        if self.cells == "any" and not differing.any():
            return False, "no cell differs"
        if self.cells == "all" and not differing[oracle != 0].all():
            return False, "not every non-empty cell differs"
        return True, f"loose signature (direction only): {self.direction} in {int(differing.sum())} of {diff.size} cells"


@dataclass(frozen=True)
class EqualsUnselected:
    """The page shows the histogram computed without any selection (all rows), i.e. it ignores the filter."""

    def holds(self, page: np.ndarray, oracle: np.ndarray, ctx: dict, tol: Tol = (0.0, 0.0)) -> Tuple[bool, str]:
        un = ctx.get("unselected")
        if un is None:
            return False, "no unselected reference available for this check"
        un = np.asarray(un, dtype=np.float64)
        if _equal(un, oracle, tol):
            return False, "the selection keeps every row, so the defect cannot be distinguished"
        if not _equal(page, un, tol):
            return False, "page differs from the histogram of all rows"
        return True, f"page equals the histogram of all rows ({float(un.sum()):g}), selection keeps {float(oracle.sum()):g}"


@dataclass(frozen=True)
class EqualsSelection:
    """The page equals the oracle's result for the selection a known defect produces — `rule` names a function of
    oracles/filters.py SELECTION_RULES, computed by the runner from the data and the widget values (now and at load),
    never from the page. The defect's selection must differ from the correct one, else the defect cannot show."""
    rule: str

    def holds(self, page: np.ndarray, oracle: np.ndarray, ctx: dict, tol: Tol = (0.0, 0.0)) -> Tuple[bool, str]:
        ref = ctx.get(f"selection:{self.rule}")
        if ref is None:
            return False, f"no reference for selection rule {self.rule!r}"
        ref = np.asarray(ref, dtype=np.float64)
        if _equal(ref, oracle, tol):
            return False, f"the {self.rule} selection equals the correct one here, so the defect cannot show"
        if not _equal(page, ref, tol):
            return False, f"page differs from the {self.rule} selection"
        return True, f"page equals the {self.rule} selection exactly ({float(ref.sum()):g} vs correct {float(oracle.sum()):g})"


@dataclass(frozen=True)
class ExcessInCells:
    """`page - oracle` equals `amount` in the listed cells and 0 elsewhere, within the check's tolerance. `amount` is a
    number or the name of a context quantity (e.g. "nan_rows": selected rows — or their summed weights — whose
    histogram variable is NaN)."""
    cells: Tuple[int, ...]
    amount: Union[str, float]

    def holds(self, page: np.ndarray, oracle: np.ndarray, ctx: dict, tol: Tol = (0.0, 0.0)) -> Tuple[bool, str]:
        amount = ctx.get(self.amount) if isinstance(self.amount, str) else self.amount
        if amount is None:
            return False, f"context quantity {self.amount!r} not available"
        if not amount:
            return False, f"{self.amount} is 0, so the defect cannot show"
        expected = oracle.copy()
        expected[list(self.cells)] += amount
        if not _equal(page, expected, tol):
            return False, f"page - oracle is not {amount:g} in cell(s) {list(self.cells)} and 0 elsewhere"
        return True, f"cell(s) {list(self.cells)} exceed the oracle by {amount:g} ({self.amount})"


@dataclass(frozen=True)
class Error:
    """For checks that end in an error message (page errors, builder errors): the message matches `regex`."""
    regex: str

    def holds(self, page, oracle, ctx: dict, tol: Tol = (0.0, 0.0)) -> Tuple[bool, str]:
        msg = ctx.get("error", "")
        ok = bool(re.search(self.regex, msg))
        return ok, f"error {'matches' if ok else 'does not match'} {self.regex!r}"


Signature = Union[Mismatch, EqualsUnselected, EqualsSelection, ExcessInCells, Error]


@dataclass(frozen=True)
class ExpectedFailure:
    """A known defect on one check (exact check name as in the report, e.g. "h1.bin_count") at the listed steps."""
    bug: str
    check: str
    steps: Tuple[int, ...]
    signature: Signature
    note: str = ""


def expected_failure(bug: str, check: str, steps: Iterable[int], signature: Signature, note: str = "") -> ExpectedFailure:
    return ExpectedFailure(bug, check, tuple(steps), signature, note)


# ------------------------------------------------------------------------------------------------- classification
def classify(check: str, step: int, ok: bool, page, oracle, ctx: dict,
             expectations: Sequence[ExpectedFailure], tol: Tol = (0.0, 0.0)) -> Tuple[str, str]:
    """Outcome and a short reason for one executed check; `tol` is the tolerance the check itself used."""
    exp = [e for e in expectations if e.check == check and step in e.steps]
    if not exp:
        return ("PASS", "") if ok else ("FAIL", "")
    e = exp[0]                                   # validation guarantees at most one per check and step
    if ok:
        return "XPASS", f"expected failure {e.bug} did not occur: the defect is gone or the expectation is stale"
    page_a = np.asarray(page, dtype=np.float64)
    if not isinstance(e.signature, Error) and not np.isfinite(page_a).all():
        return "FAIL", (f"fails, but not as known defect {e.bug} describes (non-finite page values, which no "
                        f"value signature describes): a different problem")
    holds, why = e.signature.holds(page_a, np.asarray(oracle, dtype=np.float64), ctx, tol)
    if holds:
        return "XFAIL", f"known defect {e.bug}: {why}"
    return "FAIL", f"fails, but not as known defect {e.bug} describes ({why}): a different problem"


# ------------------------------------------------------------------------------------------------- aggregation
def feature_status(outcomes) -> Dict[str, str]:
    """Per feature, in this order (proposal §8, §10.3):
    Broken      — any FAIL, ERROR or XPASS among its checks;
    Unverified  — a required check of it was skipped (SKIP: its step was never reached), or none was executed;
    Known Issue — its failures are all matched XFAIL;
    Working     — all its checks PASS.
    `outcomes` are report.CheckOutcome objects; outcomes without a feature (prerequisites) are ignored."""
    by: Dict[str, list] = {}
    for o in outcomes:
        if o.feature:
            by.setdefault(o.feature, []).append(o.outcome)
    status = {}
    for f, outs in by.items():
        if any(x in ("FAIL", "ERROR", "XPASS") for x in outs):
            status[f] = "Broken"
        elif any(x == "SKIP" for x in outs) or not any(x in ("PASS", "XFAIL") for x in outs):
            status[f] = "Unverified"
        elif any(x == "XFAIL" for x in outs):
            status[f] = "Known Issue"
        else:
            status[f] = "Working"
    return status
