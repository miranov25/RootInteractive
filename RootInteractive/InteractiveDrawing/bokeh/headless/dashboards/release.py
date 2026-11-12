"""The release check of the oracle test dashboards (PHASE_4_7, Claude20.RI; item G3 of GPT24's Cycle A review).

In plain words: every oracle test dashboard writes `dashboard_health.json`, with one status per user action:
- PASS: the page equals the oracle;
- KNOWN_ISSUE_RED: a known RootInteractive bug shows exactly its signature;
- FAIL: anything else.
The dashboard itself is PASS only when every action is PASS, otherwise RED.

A strict expected failure keeps a pytest run green while a known bug is present, so pytest alone cannot tell a
release that a user workflow is broken. This check can. It reads the health files of one run and passes only when:
- every required dashboard has exactly one health file, with every required action;
- every action is PASS, or KNOWN_ISSUE_RED for a known bug the Architect has accepted for release (waivers.json:
  the bug, the decision, the date);
- every accepted bug is still seen. A waiver for a bug that no longer shows must be removed.
Anything missing, FAIL, NOT_RUN or unknown fails. The check fails closed: no file, no pass.

    python -m RootInteractive.InteractiveDrawing.bokeh.headless.dashboards.release RUN_DIR [WAIVERS.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

HERE = Path(__file__).resolve().parent
WAIVERS = HERE / "waivers.json"
REQUIRED: Dict[str, List[str]] = {"H01": ["C1", "C2", "C3", "C4", "C5", "C6"]}
STATUSES = ("PASS", "KNOWN_ISSUE_RED", "FAIL")


def load_waivers(path: Path = WAIVERS) -> dict:
    data = json.loads(Path(path).read_text())
    waivers = data.get("waivers")
    if not isinstance(waivers, dict):
        raise ValueError(f"{path}: 'waivers' must be an object")
    for bug, w in waivers.items():
        missing = [k for k in ("decision", "date", "by") if not (isinstance(w, dict) and w.get(k))]
        if missing:
            raise ValueError(f"{path}: waiver {bug} lacks {', '.join(missing)}")
    return waivers


def check(health_files: Iterable[Path], waivers: dict, required: Dict[str, List[str]] = REQUIRED) -> List[str]:
    problems: List[str] = []
    by_scenario: Dict[str, list] = {}
    for f in health_files:
        try:
            h = json.loads(Path(f).read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"{f}: unreadable health file ({exc})")
            continue
        by_scenario.setdefault(h.get("scenario"), []).append((f, h))
    seen_bugs = set()
    for scenario, actions in required.items():
        files = by_scenario.get(scenario, [])
        if len(files) != 1:
            problems.append(f"{scenario}: {len(files)} health files, expected exactly one")
            continue
        f, h = files[0]
        got = h.get("actions") or {}
        bugs = h.get("known_defects") or {}
        for cid in actions:
            status = got.get(cid, "NOT_RUN")
            if status == "PASS":
                continue
            if status == "KNOWN_ISSUE_RED":
                bug = bugs.get(cid)
                seen_bugs.add(bug)
                if bug not in waivers:
                    problems.append(f"{scenario} {cid}: known bug {bug} - not accepted for release "
                                    f"(no waiver in waivers.json)")
                continue
            problems.append(f"{scenario} {cid}: {status}")
        extra = sorted(set(got) - set(actions))
        if extra:
            problems.append(f"{scenario}: actions not in the release list: {extra}")
        unknown = sorted({s for s in got.values() if s not in STATUSES})
        if unknown:
            problems.append(f"{scenario}: unknown statuses {unknown}")
        should = "PASS" if got and all(got.get(c) == "PASS" for c in actions) else "RED"
        if h.get("dashboard") != should:
            problems.append(f"{scenario}: dashboard status {h.get('dashboard')!r}, but its actions make it {should}")
    for scenario in sorted(set(by_scenario) - set(required), key=str):
        problems.append(f"{scenario}: a health file for a dashboard the release check does not know")
    for bug in sorted(set(waivers) - seen_bugs):
        problems.append(f"waiver {bug}: no dashboard shows this bug any more - remove the waiver")
    return problems


def main(argv: List[str]) -> int:
    run_dir = Path(argv[0])
    waivers = load_waivers(Path(argv[1]) if len(argv) > 1 else WAIVERS)
    files = sorted(run_dir.rglob("dashboard_health.json"))
    problems = check(files, waivers)
    print(f"release check of the oracle test dashboards: {len(files)} health file(s) under {run_dir}")
    for p in problems:
        print("  PROBLEM:", p)
    print("RELEASE CHECK:", "PASS" if not problems else f"FAIL ({len(problems)} problem(s))")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
