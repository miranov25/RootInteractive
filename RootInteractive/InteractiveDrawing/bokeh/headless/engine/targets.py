"""The collection contract of the headless targets (stage 4.2c-2c, specification v0.1 §5).

A target is a pytest marker registered in the headless conftest (TARGETS). This module checks a collection — a mapping
from each test id to the target markers it carries — against the registered targets: every test belongs to at least one
target, every target selects at least one test. Overlaps (a test in several targets) are allowed and reported.
A misspelled marker never reaches this check: `--strict-markers` (pytest.ini) makes it a collection error.
"""


def problems(collected: dict, targets) -> list:
    """Plain-language problems of a collection: tests without a target, and targets that select nothing."""
    out = [f"test without a target marker: {t}" for t, marks in sorted(collected.items()) if not marks]
    used = set().union(*collected.values()) if collected else set()
    out += [f"target selects no test: {name}" for name in targets if name not in used]
    return out


def overlaps(collected: dict) -> dict:
    """Tests carrying more than one target marker, with their targets."""
    return {t: sorted(m) for t, m in sorted(collected.items()) if len(m) > 1}
