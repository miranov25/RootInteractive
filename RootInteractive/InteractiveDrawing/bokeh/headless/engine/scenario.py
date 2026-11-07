"""The scenario contract (proposal v0.2 §4) — the subset delivered in stage 4.2a.

A scenario is one test written as data: a generated table, a page declaration, the steps, the checks after each
step, and labels. Validation runs before any browser starts; a malformed scenario raises ScenarioError with a
readable message, never a silent skip.

Stage 4.2a supports: one main table; 1-D and N-D histograms (optionally weighted by a column, or by a parameter
whose options create a stack of histograms); range widgets; aliases (names, numbers, + - * /, parentheses);
profiles `<histogram>_<axis>` shown in a figure; lossless transport; steps `load`, `set_` (range widget, "full"
read from the page) and `js_action` (only with a stated completion condition); checks `auto` (automatic oracle),
`audit` (re-read after a delay), `js_value`, and `responds` (a probed value changes on a move and returns on the
move back). Every scenario declares at least one check, and every step selector names an existing step (v0.3).
Expected failures, raw `fromArray` pages and further constructs arrive in later stages.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

LOSSLESS = ((".*", (("code", 0), ("zip", 0), ("base64", 0))),)


class ScenarioError(ValueError):
    """A scenario that cannot be run as written."""


# ----------------------------------------------------------------------------------------------- data and page
@dataclass(frozen=True)
class TableSpec:
    n: int
    seed: int
    columns: Dict[str, str]


@dataclass(frozen=True)
class Tables:
    main: TableSpec


@dataclass(frozen=True)
class Hist:
    name: str
    variables: Tuple[str, ...]
    nbins: Tuple[int, ...]
    range: Tuple[Tuple[float, float], ...]
    weights: Optional[str] = None          # a column, or the name of a Parameter (one histogram per option)

    def source(self) -> dict:
        d = {"name": self.name, "variables": list(self.variables),
             "nbins": self.nbins[0] if len(self.nbins) == 1 else list(self.nbins),
             "range": list(self.range[0]) if len(self.range) == 1 else [list(r) for r in self.range]}
        if self.weights is not None:
            d["weights"] = self.weights
        return d


def hist(name: str, variables: Sequence[str], nbins: Union[int, Sequence[int]], range: Sequence,
         weights: Optional[str] = None) -> Hist:
    variables = tuple(variables)
    nbins = (int(nbins),) if isinstance(nbins, int) else tuple(int(b) for b in nbins)
    rng = (tuple(float(x) for x in range),) if not isinstance(range[0], (list, tuple)) else \
        tuple(tuple(float(x) for x in r) for r in range)
    return Hist(name, variables, nbins, rng, weights)


@dataclass(frozen=True)
class RangeWidget:
    column: str
    limits: Optional[Tuple[float, float]] = None       # None = RootInteractive's default limits


def range_(column: str, limits: Optional[Tuple[float, float]] = None) -> RangeWidget:
    return RangeWidget(column, None if limits is None else (float(limits[0]), float(limits[1])))


@dataclass(frozen=True)
class Parameter:
    name: str
    value: Tuple[str, ...]
    options: Tuple[str, ...]


@dataclass(frozen=True)
class Page:
    histograms: Tuple[Hist, ...] = ()
    widgets: Tuple[RangeWidget, ...] = ()
    aliases: Tuple[Tuple[str, str], ...] = ()
    parameters: Tuple[Parameter, ...] = ()
    profiles: Tuple[str, ...] = ()                     # profile sources shown in a figure, e.g. "h_1"


@dataclass(frozen=True)
class Transport:
    array_compression: Tuple = LOSSLESS


# ----------------------------------------------------------------------------------------------- steps
@dataclass(frozen=True)
class Load:
    def describe(self) -> str:
        return "load"


@dataclass(frozen=True)
class SetRange:
    column: str
    value: Union[str, Tuple[float, float]]             # (low, high) or "full" = the widget's start/end on the page

    def describe(self) -> str:
        return f"set {self.column} -> {self.value}"


@dataclass(frozen=True)
class JsAction:
    script: str
    until: Optional[str] = None                        # the stated completion condition (a JS expression)
    timeout_s: float = 5.0

    def describe(self) -> str:
        return f"js action (until {self.until})"


def load() -> Load:
    return Load()


def set_(column: str, value) -> SetRange:
    return SetRange(column, value if value == "full" else (float(value[0]), float(value[1])))


def js_action(script: str, until: Optional[str] = None, timeout_s: float = 5.0) -> JsAction:
    return JsAction(script, until, timeout_s)


# ----------------------------------------------------------------------------------------------- checks
@dataclass(frozen=True)
class Auto:
    """Automatic oracle for every supported model of the page, at every step (or the listed steps)."""
    steps: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class Audit:
    """After the completion condition: wait, read again; any difference is an ERROR naming the step."""
    delay_s: float = 1.0
    steps: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class JsValue:
    expr: str
    expected: Any
    steps: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class Responds:
    """Metamorphic check (proposal §13): the probed column differs from the reference step after the move, and equals
    it again after the move back. Needs no oracle of the statistic itself — it shows the model exists, is computed and
    reacts. Evaluated at the last of the three steps."""
    model: str
    column: str
    moved: int
    restored: int
    reference: int = 0
    rel_tol: float = 1e-12

    @property
    def steps(self) -> Tuple[int, ...]:
        return (self.reference, self.moved, self.restored)


def responds(model: str, column: str, moved: int, restored: int, reference: int = 0) -> Responds:
    return Responds(model, column, moved, restored, reference)


def auto(steps=None) -> Auto:
    return Auto(None if steps is None else tuple(steps))


def audit(delay_s: float = 1.0, steps=None) -> Audit:
    return Audit(delay_s, None if steps is None else tuple(steps))


def js_value(expr: str, expected: Any, steps=None) -> JsValue:
    return JsValue(expr, expected, None if steps is None else tuple(steps))


# ----------------------------------------------------------------------------------------------- scenario
@dataclass(frozen=True)
class Labels:
    family: str
    models: Tuple[str, ...] = ()
    features: Tuple[str, ...] = ()
    technique: str = ""
    priority: str = "p1"


@dataclass(frozen=True)
class Scenario:
    id: str
    data: Tables
    page: Page
    steps: Tuple[Any, ...]
    checks: Tuple[Any, ...]
    labels: Labels
    transport: Transport = field(default_factory=Transport)
    expect: Tuple[Any, ...] = ()

    def validate(self) -> "Scenario":
        where = f"scenario {self.id!r}"
        if not self.id or "/" not in self.id:
            raise ScenarioError(f"{where}: the id must start with its family, e.g. 'histograms/…'")
        if self.expect:
            raise ScenarioError(f"{where}: expected failures arrive in stage 4.2b")
        if tuple(self.transport.array_compression) != LOSSLESS:
            raise ScenarioError(f"{where}: stage 4.2a supports only lossless transport {LOSSLESS}")
        if not self.steps or not isinstance(self.steps[0], Load):
            raise ScenarioError(f"{where}: the first step must be load()")
        if not self.checks:
            raise ScenarioError(f"{where}: no check declared; a scenario must assert something (the transport "
                                f"check alone is a prerequisite, not an assertion)")
        columns = set(self.data.main.columns)
        aliases = {a for a, _ in self.page.aliases}
        params = {p.name: p for p in self.page.parameters}
        for a, expr in self.page.aliases:
            if not re.fullmatch(r"[A-Za-z_0-9.\s+\-*/()]+", expr) or "**" in expr or "//" in expr:
                raise ScenarioError(f"{where}: alias {a!r} = {expr!r} uses more than names, numbers, + - * / and "
                                    f"parentheses (the oracle's alias subset until rulebook RB-6)")
            for t in re.findall(r"[A-Za-z_]\w*", expr):
                if t not in columns:
                    raise ScenarioError(f"{where}: alias {a!r} uses {t!r}, which is not a column of the main table")
        hists = {h.name: h for h in self.page.histograms}
        for prof in self.page.profiles:
            m = re.fullmatch(r"(.+)_(\d+)", prof)
            if not m or m.group(1) not in hists or len(hists[m.group(1)].variables) < 2 or \
                    int(m.group(2)) >= len(hists[m.group(1)].variables):
                raise ScenarioError(f"{where}: profile {prof!r} is not a projection '<histogram>_<axis>' of a declared "
                                    f"N-D histogram")
        widgets = {w.column for w in self.page.widgets}
        for w in self.page.widgets:
            if w.column not in columns:
                raise ScenarioError(f"{where}: widget on unknown column {w.column!r}")
        for h in self.page.histograms:
            if not (len(h.variables) == len(h.nbins) == len(h.range)):
                raise ScenarioError(f"{where}: histogram {h.name!r} needs one nbins and one range per variable")
            for v in h.variables:
                if v not in columns and v not in aliases:
                    raise ScenarioError(f"{where}: histogram {h.name!r} uses unknown variable {v!r}")
            if h.weights is not None and h.weights not in columns and h.weights not in params:
                raise ScenarioError(f"{where}: histogram {h.name!r} uses unknown weights {h.weights!r}")
            if h.weights in params and any(o not in columns for o in params[h.weights].options):
                raise ScenarioError(f"{where}: parameter {h.weights!r} has an option that is not a column")
        for i, s in enumerate(self.steps):
            if isinstance(s, SetRange) and s.column not in widgets:
                raise ScenarioError(f"{where}: step {i} sets {s.column!r}, but the page has no widget on it")
            if isinstance(s, JsAction) and not s.until:
                raise ScenarioError(f"{where}: step {i} is a JavaScript action without a stated completion "
                                    f"condition (until=…); the engine does not guess when it has finished")
            if i > 0 and isinstance(s, Load):
                raise ScenarioError(f"{where}: load() only as the first step")
        n = len(self.steps)
        for c in self.checks:
            if isinstance(c, Auto) and not self.page.histograms:
                raise ScenarioError(f"{where}: auto() needs at least one supported model on the page")
            if not isinstance(c, (Auto, Audit, JsValue, Responds)):
                raise ScenarioError(f"{where}: unsupported check {type(c).__name__} in stage 4.2a")
            if c.steps is not None and any(not (0 <= k < n) for k in c.steps):
                raise ScenarioError(f"{where}: {type(c).__name__} names step(s) {c.steps}, but the scenario has steps "
                                    f"0 … {n - 1}; a check on a step that never runs would never assert anything")
            if isinstance(c, Responds):
                if len(set(c.steps)) != 3:
                    raise ScenarioError(f"{where}: responds() needs three different steps (reference, moved, restored)")
                models = {name for name in hists} | set(self.page.profiles)
                models |= {f"{h.name}[{o}]" for h in self.page.histograms if h.weights in params
                           for o in params[h.weights].options}
                if c.model not in models:
                    raise ScenarioError(f"{where}: responds() probes {c.model!r}, which the page declaration does not "
                                        f"create")
        return self
