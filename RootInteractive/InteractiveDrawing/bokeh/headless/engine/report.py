"""Reports (proposal v0.2 §8): one outcome per scenario, step and check, written as JSON and as text next to the
page and the saved case. Outcomes: PASS, FAIL, ERROR, XFAIL (a declared known defect whose signature holds), XPASS
(an expected failure that did not occur — strict, fails the scenario), SKIP (a declared check on a step never reached).
A scenario is OK only if every outcome is PASS or XFAIL **and** at least one declared assertion (a check the scenario
asked for) was executed; prerequisites such as the transport check do not count (v0.3, GPT24 P1-2). Each outcome
carries the feature it verifies; `feature_status` aggregates them (engine/expect.py).

    python -m RootInteractive.InteractiveDrawing.bokeh.headless.engine.report outcomes <output folder>
prints one normalised line per scenario, step and check ("scenario | step | check | outcome") plus each scenario's
case content hash, so that two platforms' runs can be compared with `diff`.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .expect import feature_status


@dataclass
class CheckOutcome:
    step: int
    step_name: str
    check: str
    outcome: str            # PASS | FAIL | ERROR
    detail: str
    feature: str = ""
    assertion: bool = True          # False for prerequisites (transport); they never make a scenario OK alone


@dataclass
class ScenarioResult:
    id: str
    outdir: str
    outcomes: List[CheckOutcome] = field(default_factory=list)
    page_errors: List[str] = field(default_factory=list)
    console_errors: List[str] = field(default_factory=list)
    requests_served_locally: List[str] = field(default_factory=list)
    requests_blocked: List[str] = field(default_factory=list)
    versions: Dict[str, str] = field(default_factory=dict)
    case_sha256: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    completion: List[dict] = field(default_factory=list)
    reproduce: str = ""
    fault: str = ""                                   # id of the deliberate fault this run carries, if any
    fault_executions: int = 0                         # how often the mutated code ran (must be > 0)
    unchecked_models: List[str] = field(default_factory=list)    # completeness: verifiable models nobody checked
    console_allowed: List[str] = field(default_factory=list)     # console errors on the allowed list, with reason

    def add(self, step: int, step_name: str, check: str, outcome: str, detail: str, feature: str = "",
            assertion: bool = True) -> None:
        self.outcomes.append(CheckOutcome(step, step_name, check, outcome, detail, feature, assertion))

    @property
    def assertions_executed(self) -> int:
        return sum(1 for o in self.outcomes if o.assertion and o.outcome in ("PASS", "FAIL", "XFAIL", "XPASS"))

    @property
    def ok(self) -> bool:
        return self.assertions_executed > 0 and all(o.outcome in ("PASS", "XFAIL") for o in self.outcomes)

    def features(self) -> Dict[str, str]:
        return feature_status(self.outcomes)

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for o in self.outcomes:
            c[o.outcome] = c.get(o.outcome, 0) + 1
        return c

    def text(self) -> str:
        lines = [f"scenario {self.id}: {'PASS' if self.ok else 'FAIL'} {self.counts()}; "
                 f"declared assertions executed: {self.assertions_executed}"]
        for o in self.outcomes:
            lines.append(f"  [{o.outcome:5s}] step {o.step} ({o.step_name}) {o.check}: {o.detail}")
        if self.assertions_executed == 0:
            lines.append("  no declared assertion was executed — this is a failure")
        lines.append(f"  features: {self.features()}")
        for m in self.unchecked_models:
            lines.append(f"  not checked (exercised only): {m}")
        for e in self.console_allowed:
            lines.append(f"  console error, allowed: {e}")
        if self.fault:
            lines.append(f"  fault: {self.fault}, mutated code executed {self.fault_executions} times")
        for e in self.page_errors:
            lines.append(f"  page error: {e}")
        for r in self.requests_blocked:
            lines.append(f"  blocked request: {r}")
        lines.append(f"  versions: {self.versions}")
        lines.append(f"  case sha256: {self.case_sha256}; output: {self.outdir}")
        lines.append(f"  reproduce: {self.reproduce}")
        return "\n".join(lines)

    def write(self) -> None:
        out = Path(self.outdir)
        (out / "report.json").write_text(json.dumps({**asdict(self), "features": self.features()}, indent=1, default=str))
        (out / "report.txt").write_text(self.text() + "\n")


def outcomes(folder) -> list:
    """Normalised outcome lines of every report.json below `folder`, sorted."""
    lines = []
    for f in sorted(Path(folder).rglob("report.json")):
        r = json.loads(f.read_text())
        lines.append(f"{r['id']} | case | content sha256 {r['case_sha256']}")
        for o in r["outcomes"]:
            lines.append(f"{r['id']} | step {o['step']} | {o['check']} | {o['outcome']}")
    return sorted(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != "outcomes":
        sys.exit("usage: python -m RootInteractive.InteractiveDrawing.bokeh.headless.engine.report outcomes <folder>")
    print("\n".join(outcomes(sys.argv[2])))
