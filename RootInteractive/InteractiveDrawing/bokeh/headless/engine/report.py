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
    backend: str = "browser"                          # "browser" (tier T3) or "node" (tier T2, proposal v0.3)
    probe_values: Dict[str, object] = field(default_factory=dict)  # per step: probed models' columns, selected rows
    isolation: Dict[str, object] = field(default_factory=dict)     # Node: what the previous document left behind

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
        """Write report.json / report.txt. A scenario run again into the same folder (one worker, two tests running
        the same scenario) replaces an IDENTICAL earlier report but never a differing one: that one is kept as
        report.1.json, report.2.json, … so a FAIL cannot vanish behind a later PASS (stage 4.2c-2b review, GPT24)."""
        out = Path(self.outdir)
        data = {**asdict(self), "features": self.features()}
        target, k = out / "report.json", 0
        while target.exists() and _block(json.loads(target.read_text())) != _block(data):
            k += 1
            target = out / f"report.{k}.json"
        target.write_text(json.dumps(data, indent=1, default=str))
        target.with_suffix(".txt").write_text(self.text() + "\n")


def _block(r: dict) -> tuple:
    """The normalised outcome lines of one report (what reviewers compare across machines)."""
    block = [f"{r['id']} | case | content sha256 {r['case_sha256']}"]
    block += [f"{r['id']} | step {o['step']} | {o['check']} | {o['outcome']}" for o in r["outcomes"]]
    return tuple(block)


def _reports(folder):
    return sorted(Path(folder).rglob("report*.json"))


def outcomes(folder) -> list:
    """Normalised outcome lines of every report.json below `folder`, sorted. Identical reports of the same scenario
    count once (stage 4.2c-2b): with parallel workers a scenario run by several tests (e.g. a healthy control) lands
    in several workers' folders, and the list must not depend on how tests were spread over workers. Repeated runs
    of a scenario that DIFFER all stay in the list."""
    blocks = {_block(json.loads(f.read_text())) for f in _reports(folder)}
    return sorted(line for block in blocks for line in block)


def required(folder, manifest, full: bool = False) -> list:
    """Scenario ids listed in `manifest` (one per line, # comments) that have no report below `folder` — an empty list
    means every required scenario ran (stage 4.2c-2b; the required-case manifest of proposal v0.3 item I)."""
    present = {json.loads(f.read_text())["id"] for f in _reports(folder)}
    wanted, section = [], "fast"
    for line in Path(manifest).read_text().splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]                     # [fast]: every run; [full]: only runs of the full suite
        elif line and not line.startswith("#") and (section == "fast" or full):
            wanted.append(line)
    return [w for w in wanted if w not in present]


def timing_table(folders) -> str:
    """One comparison table of several headless runs (stage 4.2c-2b v0.2): per run its workers, wall time, sum of test
    times, speed-up against the first run, efficiency (speed-up / workers) and the browser compile seconds summed over
    all workers. Each folder is a headless output root holding timings.jsonl (written at the end of the session)."""
    rows = []
    for f in folders:
        times = [json.loads(l) for l in (Path(f) / "timings.jsonl").read_text().splitlines()]
        starts = [t["start"] for t in times if t.get("start") is not None]
        stops = [t["stop"] for t in times if t.get("stop") is not None]
        wall = max(stops) - min(starts)
        compile_s = 0.0
        for r in _reports(f):
            d = json.loads(r.read_text())
            b = d.get("timings", {}).get("build_s")
            if d.get("backend", "browser") == "browser" and b is not None and b > 3:
                compile_s += b
        rows.append((Path(f).as_posix(), len({t["worker"] for t in times}), len(times), wall,
                     sum(t["duration"] for t in times), compile_s))
    base = rows[0][3]
    out = ["| run | workers | tests | wall s | sum of test times s | speed-up vs first | efficiency | browser compile s (all workers) |",
           "|---|---|---|---|---|---|---|---|"]
    for name, w, n, wall, total, comp in rows:
        out.append(f"| {name} | {w} | {n} | {wall:.1f} | {total:.1f} | {base / wall:.2f} | {base / wall / w:.2f} | {comp:.1f} |")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "outcomes":
        print("\n".join(outcomes(sys.argv[2])))
    elif len(sys.argv) in (4, 5) and sys.argv[1] == "required":
        missing = required(sys.argv[2], sys.argv[3], full=sys.argv[4:] == ["--full"])
        print("required headless scenarios: " + ("all present" if not missing else "MISSING: " + ", ".join(missing)))
        sys.exit(1 if missing else 0)
    elif len(sys.argv) >= 3 and sys.argv[1] == "timing-table":
        print(timing_table(sys.argv[2:]))
    else:
        sys.exit("usage: python -m ...engine.report outcomes <folder> | required <folder> <manifest> [--full] | "
                 "timing-table <folder> [<folder> ...]")
