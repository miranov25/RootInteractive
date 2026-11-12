"""The release check of the oracle test dashboards (PHASE_4_7, Claude20.RI; GPT24's Cycle A review item G3, revised
after GPT25's summary of 2026-11-12, items H-01 and H-02).

In plain words: every oracle test dashboard writes `dashboard_health.json`, with one status per user action:
- PASS: the page equals the oracle;
- KNOWN_ISSUE_RED: a known RootInteractive bug shows exactly its signature;
- FAIL: anything else.
The dashboard itself is PASS only when every action is PASS, otherwise RED.

A strict expected failure keeps a pytest run green while a known bug is present, so pytest alone cannot tell a
release that a user workflow is broken. This check can. It reads the health files of one run and passes only when:
- the run is the one asked for: every required dashboard has exactly one health file, written by the same run (run
  identity), by the current test code (code hash), for the required dashboard function, table (rows, seed and the
  table's hash, regenerated here independently) and page (manifest.json and page.html next to it, matching);
- every required action is present and PASS, or KNOWN_ISSUE_RED for a known bug the Architect has accepted for
  release (waivers.json). A waiver names its exact place - scenario, checkpoint and bug - its signature (for
  example the lost rows), its decision, date, author and expiry date. A bug at another checkpoint, an expired waiver
  or a different signature is not accepted;
- every waiver is still used. A waiver for a bug that no longer shows must be removed.
Anything missing, FAIL, NOT_RUN or unknown fails. The check fails closed: no file, no pass. An accepted bug does not
make the dashboard healthy: its health file stays RED, and the report lists the accepted bugs.

    python -m RootInteractive.InteractiveDrawing.bokeh.headless.dashboards.release RUN_DIR --run-id ID [--waivers FILE]

A release check always names the run it certifies (GPT25's fix-check summary, A1): without a run identity it fails.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
WAIVERS = HERE / "waivers.json"
# the code whose result a health file reports: a health file written by other code is not this release's evidence
CODE_FILES = ("dashboards/capture.py", "dashboards/oracle.py", "dashboards/compare.py",
              "test_headless_oracle_dashboards.py")
REQUIRED: Dict[str, dict] = {
    "H01": {"function": "testBokehClientHistogramOnlyHisto", "rows": 50000, "seed": 4711,
            "actions": ["C1", "C2", "C3", "C4", "C5", "C6"]},
}
STATUSES = ("PASS", "KNOWN_ISSUE_RED", "FAIL")
WAIVER_KEYS = ("scenario", "checkpoint", "bug", "signature", "decision", "date", "by", "expires")


def code_sha256(root: Path = HERE.parent) -> str:
    h = hashlib.sha256()
    for rel in CODE_FILES:
        h.update(rel.encode() + b"\0" + (root / rel).read_bytes() + b"\0")
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _table_sha256(rows: int, seed: int) -> str:
    from . import oracle                                     # independent regeneration, never the page's own bytes
    return oracle.table_sha256(oracle.regenerate_table(rows, seed))


def load_waivers(path: Path = WAIVERS) -> list:
    data = json.loads(Path(path).read_text())
    waivers = data.get("waivers")
    if not isinstance(waivers, list):
        raise ValueError(f"{path}: 'waivers' must be a list")
    for i, w in enumerate(waivers):
        missing = [k for k in WAIVER_KEYS if not (isinstance(w, dict) and w.get(k) not in (None, "", {}))]
        if missing:
            raise ValueError(f"{path}: waiver {i} lacks {', '.join(missing)}")
        if not re.match(r"^DEC-\d\d+$", w["decision"]):
            raise ValueError(f"{path}: waiver {i}: decision {w['decision']!r} is not a DEC number")
        for k in ("date", "expires"):
            _dt.date.fromisoformat(w[k])
        if not isinstance(w["signature"], dict):
            raise ValueError(f"{path}: waiver {i}: signature must be an object")
    return waivers


def check(health_files: Iterable[Path], waivers: list, required: Dict[str, dict] = REQUIRED,
          run_id: Optional[str] = None, code: Optional[str] = None, today: Optional[_dt.date] = None,
          accepted: Optional[list] = None) -> List[str]:
    """The problems of one run (empty = pass). `accepted`, when given, receives the known bugs accepted by a waiver."""
    today = today or _dt.date.today()
    code = code or code_sha256()
    problems: List[str] = []
    if not run_id:
        problems.append("no run identity given: a release check must name the run it certifies (--run-id)")
    by_scenario: Dict[str, list] = {}
    for f in health_files:
        try:
            h = json.loads(Path(f).read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"{f}: unreadable health file ({exc})")
            continue
        by_scenario.setdefault(h.get("scenario"), []).append((Path(f), h))
    used = set()
    for scenario, spec in required.items():
        files = by_scenario.get(scenario, [])
        if len(files) != 1:
            problems.append(f"{scenario}: {len(files)} health files, expected exactly one")
            continue
        f, h = files[0]
        # identity: which dashboard, which table, which page, which code, which run
        for key in ("function", "rows", "seed"):
            if h.get(key) != spec[key]:
                problems.append(f"{scenario}: {key} {h.get(key)!r}, the release requires {spec[key]!r}")
        if h.get("table_sha256") != _table_sha256(spec["rows"], spec["seed"]):
            problems.append(f"{scenario}: the table hash is not the hash of the table regenerated from "
                            f"({spec['rows']} rows, seed {spec['seed']})")
        if h.get("code_sha256") != code:
            problems.append(f"{scenario}: written by other test code (code hash {str(h.get('code_sha256'))[:12]}, "
                            f"current {code[:12]})")
        if run_id and h.get("run_id") != run_id:
            problems.append(f"{scenario}: written by run {h.get('run_id')!r}, not by this release run {run_id!r}")
        man = f.parent / "manifest.json"
        if not man.is_file():
            problems.append(f"{scenario}: no manifest.json next to the health file")
        else:
            m = json.loads(man.read_text())
            for key in ("function", "rows", "seed", "table_sha256", "page_sha256"):
                if m.get(key) != h.get(key):
                    problems.append(f"{scenario}: manifest {key} {m.get(key)!r} differs from the health file's "
                                    f"{h.get(key)!r}")
            page = f.parent / "page.html"
            if not page.is_file() or _sha256_file(page) != m.get("page_sha256"):
                problems.append(f"{scenario}: page.html is missing or is not the page of the manifest")
        # the actions
        got = h.get("actions") or {}
        bugs = h.get("known_defects") or {}
        detail = h.get("known_defect_detail") or {}
        for cid in spec["actions"]:
            status = got.get(cid, "NOT_RUN")
            if status == "PASS":
                continue
            if status != "KNOWN_ISSUE_RED":
                problems.append(f"{scenario} {cid}: {status}")
                continue
            bug = bugs.get(cid)
            match = [i for i, w in enumerate(waivers)
                     if (w["scenario"], w["checkpoint"], w["bug"]) == (scenario, cid, bug)]
            if not match:
                problems.append(f"{scenario} {cid}: known bug {bug} - not accepted for release "
                                f"(no waiver for {scenario} {cid} {bug} in waivers.json)")
                continue
            w = waivers[match[0]]
            used.add(match[0])
            if _dt.date.fromisoformat(w["expires"]) < today:
                problems.append(f"{scenario} {cid}: the waiver for {bug} expired on {w['expires']}")
            elif any(detail.get(cid, {}).get(k) != v for k, v in w["signature"].items()):
                problems.append(f"{scenario} {cid}: {bug} shows {detail.get(cid)}, not the accepted signature "
                                f"{w['signature']}")
            elif accepted is not None:
                accepted.append(f"{scenario} {cid}: {bug} accepted by {w['decision']} ({w['by']}, {w['date']}, "
                                f"until {w['expires']})")
        extra = sorted(set(got) - set(spec["actions"]))
        if extra:
            problems.append(f"{scenario}: actions not in the release list: {extra}")
        unknown = sorted({s for s in got.values() if s not in STATUSES})
        if unknown:
            problems.append(f"{scenario}: unknown statuses {unknown}")
        should = "PASS" if got and all(got.get(c) == "PASS" for c in spec["actions"]) else "RED"
        if h.get("dashboard") != should:
            problems.append(f"{scenario}: dashboard status {h.get('dashboard')!r}, but its actions make it {should}")
    for scenario in sorted(set(by_scenario) - set(required), key=str):
        problems.append(f"{scenario}: a health file for a dashboard the release check does not know")
    for i, w in enumerate(waivers):
        if i not in used:
            problems.append(f"waiver {w['scenario']} {w['checkpoint']} {w['bug']}: not seen at that place in this "
                            f"run - remove or correct the waiver")
    return problems


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="release check of the oracle test dashboards")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--waivers", type=Path, default=WAIVERS)
    ap.add_argument("--run-id", required=True, help="the run identity recorded in the health files of the run")
    a = ap.parse_args(argv)
    waivers = load_waivers(a.waivers)
    files = sorted(a.run_dir.rglob("dashboard_health.json"))
    accepted: list = []
    problems = check(files, waivers, run_id=a.run_id, accepted=accepted)
    print(f"release check of the oracle test dashboards: {len(files)} health file(s) under {a.run_dir}; "
          f"code {code_sha256()[:12]}; waivers {len(waivers)}")
    for p in accepted:
        print("  ACCEPTED (dashboard stays RED):", p)
    for p in problems:
        print("  PROBLEM:", p)
    print("RELEASE CHECK:", "PASS" if not problems else f"FAIL ({len(problems)} problem(s))")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
