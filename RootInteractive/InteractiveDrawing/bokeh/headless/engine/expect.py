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
quantities (e.g. the histogram without any selection, the number of selected rows whose variable is NaN).
"""
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np


# ------------------------------------------------------------------------------------------------- signatures
@dataclass(frozen=True)
class Mismatch:
    """`direction`: "page > oracle" (every differing cell is above the oracle, page never below it) or
    "page < oracle"; `cells`: "any" (at least one cell differs) or "all" (every cell with a non-zero oracle differs)."""
    direction: str
    cells: str = "any"

    def holds(self, page: np.ndarray, oracle: np.ndarray, ctx: dict) -> Tuple[bool, str]:
        if page.shape != oracle.shape:
            return False, f"shape {page.shape} vs {oracle.shape}"
        diff = page - oracle
        if self.direction == "page > oracle":
            ok_dir, bad = bool((diff >= 0).all()), int((diff < 0).sum())
        elif self.direction == "page < oracle":
            ok_dir, bad = bool((diff <= 0).all()), int((diff > 0).sum())
        else:
            raise ValueError(f"unknown direction {self.direction!r}")
        if not ok_dir:
            return False, f"{bad} cell(s) deviate in the other direction"
        differing = diff != 0
        if self.cells == "any" and not differing.any():
            return False, "no cell differs"
        if self.cells == "all" and not differing[oracle != 0].all():
            return False, "not every non-empty cell differs"
        return True, f"{self.direction} in {int(differing.sum())} of {diff.size} cells"


@dataclass(frozen=True)
class EqualsUnselected:
    """The page shows the histogram computed without any selection (all rows), i.e. it ignores the filter."""

    def holds(self, page: np.ndarray, oracle: np.ndarray, ctx: dict) -> Tuple[bool, str]:
        un = ctx.get("unselected")
        if un is None:
            return False, "no unselected reference available for this check"
        un = np.asarray(un, dtype=np.float64)
        if un.shape != page.shape or not np.array_equal(page, un):
            return False, "page differs from the histogram of all rows"
        if np.array_equal(un, oracle):
            return False, "the selection keeps every row, so the defect cannot be distinguished"
        return True, f"page equals the histogram of all rows ({int(un.sum())}), selection keeps {int(oracle.sum())}"


@dataclass(frozen=True)
class ExcessInCells:
    """`page - oracle` equals `amount` in the listed cells and 0 elsewhere. `amount` is a number or the name of a
    context quantity (e.g. "nan_rows": selected rows whose histogram variable is NaN)."""
    cells: Tuple[int, ...]
    amount: Union[str, float]

    def holds(self, page: np.ndarray, oracle: np.ndarray, ctx: dict) -> Tuple[bool, str]:
        amount = ctx.get(self.amount) if isinstance(self.amount, str) else self.amount
        if amount is None:
            return False, f"context quantity {self.amount!r} not available"
        if not amount:
            return False, f"{self.amount} is 0, so the defect cannot show"
        expected = np.zeros_like(oracle)
        expected[list(self.cells)] = amount
        if page.shape != oracle.shape or not np.array_equal(page - oracle, expected):
            return False, f"page - oracle is not {amount} in cell(s) {list(self.cells)} and 0 elsewhere"
        return True, f"cell(s) {list(self.cells)} exceed the oracle by exactly {amount:g} ({self.amount})"


@dataclass(frozen=True)
class Error:
    """For checks that end in an error message (page errors, builder errors): the message matches `regex`."""
    regex: str

    def holds(self, page, oracle, ctx: dict) -> Tuple[bool, str]:
        msg = ctx.get("error", "")
        return (bool(re.search(self.regex, msg)), f"error {'matches' if re.search(self.regex, msg) else 'does not match'} {self.regex!r}")


Signature = Union[Mismatch, EqualsUnselected, ExcessInCells, Error]


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
             expectations: Sequence[ExpectedFailure]) -> Tuple[str, str]:
    """Outcome and a short reason for one executed check."""
    exp = [e for e in expectations if e.check == check and step in e.steps]
    if not exp:
        return ("PASS", "") if ok else ("FAIL", "")
    e = exp[0]
    if ok:
        return "XPASS", f"expected failure {e.bug} did not occur: the defect is gone or the expectation is stale"
    holds, why = e.signature.holds(np.asarray(page, dtype=np.float64), np.asarray(oracle, dtype=np.float64), ctx)
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
