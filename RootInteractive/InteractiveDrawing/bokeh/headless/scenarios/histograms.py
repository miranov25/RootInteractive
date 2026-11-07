"""Stage 4.2a scenarios: written as data; adding one needs no engine change.

1. `histograms/plain_1d_2d_one_range_two_moves` — the plain scenario of the 4.2a acceptance: a 1-D and a 2-D
   histogram, one range widget, two moves. The widget column has no missing values, so the 1-D histogram's known
   load-state defect (B2, B2') cannot show here; that defect gets its own scenario in stage 4.2b.
2. `selection/rb1_nan_excluded_at_full_bounds` — the measured evidence for rule RB-1.2 (erratum E-3): with 10 %
   missing values in the widget column, the N-D histogram at load must exclude exactly those rows.
3. `chain/deep_widget_filter_alias_stack_profile` — the deepest chain the engine supports in 4.2a: range widget
   -> RangeFilter -> LazyIntersectionFilter -> CDSAlias (alias xa = x*2) -> two weighted HistoNdCDS (a parameter
   with two weight options) -> CDSStack -> HistoNdProfile. Checked by the automatic oracle and by an audit that
   re-reads every probed model one second after the completion condition was met. Since v0.3 also `responds`:
   the profile exists, has values, changes on the move and returns exactly on the move back (GPT24 P1-3). The
   profile's values themselves are compared with an oracle only from stage 4.4 (rulebook RB-3).

Regression scenarios written by the reviewers of stage 4.2a without touching the engine (v0.3):
4. `histograms/claude3_3d_widget_with_explicit_limits` (Claude3) — widget limits narrower than the data; the load
   check must not apply (it applies only at RootInteractive's default limits).
5. `histograms/claude10_two_ranges_alternating_moves` (Claude10) — two range widgets, seven alternating moves,
   audit after every step.
6. `histograms/gpt24_empty_selection_shifted_range` (GPT24) — a selection with zero rows, then recovery.
"""
from ..engine.scenario import (Labels, Page, Parameter, Scenario, TableSpec, Tables, audit, auto, hist, load,
                               range_, responds, set_)

PLAIN = Scenario(
    id="histograms/plain_1d_2d_one_range_two_moves",
    data=Tables(main=TableSpec(n=20_000, seed=7, columns={"x": "uniform(0,1)", "y": "normal(0,1)",
                                                          "a": "normal(0,1)"})),
    page=Page(histograms=(hist("h1", ["x"], 10, [0, 1]),
                          hist("h2", ["x", "y"], [10, 4], [[0, 1], [-3, 3]])),
              widgets=(range_("a"),)),
    steps=(load(), set_("a", [-1, 1]), set_("a", "full")),
    checks=(auto(),),
    labels=Labels(family="histograms", models=("HistogramCDS", "HistoNdCDS", "RangeFilter"),
                  verified=("HistogramCDS", "HistoNdCDS", "RangeFilter"),
                  features=("HIST.histogram_1d", "HIST.histogram_nd"), technique="differential, state sequence",
                  priority="p0"),
)

RB1_NAN = Scenario(
    id="selection/rb1_nan_excluded_at_full_bounds",
    data=Tables(main=TableSpec(n=20_000, seed=11, columns={"x": "uniform(0,1)", "y": "normal(0,1)",
                                                           "a": "normal(0,1) + 10% NaN"})),
    page=Page(histograms=(hist("h2", ["x", "y"], [10, 4], [[0, 1], [-3, 3]]),),
              widgets=(range_("a"),)),
    steps=(load(), set_("a", [-1, 1]), set_("a", "full")),
    checks=(auto(),),
    labels=Labels(family="selection", models=("RangeFilter", "HistoNdCDS"), verified=("RangeFilter", "HistoNdCDS"),
                  features=("HIST.histogram_nd",), technique="boundary value (missing values at full bounds)",
                  priority="p0"),
)

DEEP_CHAIN = Scenario(
    id="chain/deep_widget_filter_alias_stack_profile",
    data=Tables(main=TableSpec(n=20_000, seed=13, columns={"x": "uniform(0,1)", "a": "normal(0,1) + 10% NaN",
                                                           "w1": "uniform(0,1)", "w2": "uniform(0,2)"})),
    page=Page(histograms=(hist("h", ["xa", "a"], [10, 4], [[0, 2], [-3, 3]], weights="weights"),),
              widgets=(range_("a"),),
              aliases=(("xa", "x*2"),),
              parameters=(Parameter("weights", ("w1", "w2"), ("w1", "w2")),),
              profiles=("h_1",)),
    steps=(load(), set_("a", [-1, 1]), set_("a", "full")),
    checks=(auto(), audit(delay_s=1.0), responds("h_1", "mean", moved=1, restored=2)),
    labels=Labels(family="chain", models=("RangeFilter", "LazyIntersectionFilter", "CDSAlias", "HistoNdCDS",
                                          "CDSStack", "HistoNdProfile"),
                  # the stack is exercised only; the profile is verified by an explicit relation (responds), its
                  # values by an oracle only from stage 4.4 (RB-3)
                  verified=("RangeFilter", "LazyIntersectionFilter", "CDSAlias", "HistoNdCDS", "HistoNdProfile"),
                  features=("HIST.histogram_nd",), technique="state sequence, completion audit", priority="p0"),
)

C3_LIMITS = Scenario(
    id="histograms/claude3_3d_widget_with_explicit_limits",
    data=Tables(main=TableSpec(n=20_000, seed=21, columns={"x": "uniform(0,1)", "y": "normal(0,1)",
                                                           "z": "uniform(-1,1)", "a": "normal(0,1)"})),
    page=Page(histograms=(hist("h3", ["x", "y", "z"], [5, 4, 3], [[0, 1], [-3, 3], [-1, 1]]),),
              widgets=(range_("a", (-1, 1)),)),
    steps=(load(), set_("a", [-0.5, 0.5]), set_("a", "full")),
    checks=(auto(),),
    labels=Labels(family="histograms", models=("HistoNdCDS", "RangeFilter"),
                  verified=("HistoNdCDS", "RangeFilter"),
                  features=("HIST.histogram_nd",),
                  technique="state sequence, explicit widget limits (reviewer Claude3)", priority="p1"),
)

C10_TWO_WIDGETS = Scenario(
    id="histograms/claude10_two_ranges_alternating_moves",
    data=Tables(main=TableSpec(n=30_000, seed=21, columns={"x": "uniform(0,1)", "y": "normal(0,1)",
                                                           "a": "normal(0,1)", "b": "normal(0,1)"})),
    page=Page(histograms=(hist("h1", ["x"], 10, [0, 1]),
                          hist("h2", ["x", "y"], [10, 4], [[0, 1], [-3, 3]])),
              widgets=(range_("a"), range_("b"))),
    steps=(load(), set_("a", [-1, 1]), set_("b", [0, 2]), set_("a", "full"), set_("b", "full"),
           set_("b", [-2, -0.5]), set_("a", [0.5, 3])),
    checks=(auto(), audit(delay_s=0.5)),
    labels=Labels(family="histograms",
                  models=("HistogramCDS", "HistoNdCDS", "RangeFilter", "LazyIntersectionFilter"),
                  verified=("HistogramCDS", "HistoNdCDS", "RangeFilter", "LazyIntersectionFilter"),
                  features=("HIST.histogram_1d", "HIST.histogram_nd"),
                  technique="state sequence, intersection (reviewer Claude10)", priority="p1"),
)

GPT24_EMPTY = Scenario(
    id="histograms/gpt24_empty_selection_shifted_range",
    data=Tables(main=TableSpec(n=2048, seed=24, columns={"x": "uniform(-2,3)", "y": "uniform(-1,1)",
                                                         "a": "normal(0,1)"})),
    page=Page(histograms=(hist("h1", ["x"], 6, [-1, 2]),
                          hist("h2", ["x", "y"], [6, 4], [[-1, 2], [-1, 1]])),
              widgets=(range_("a"),)),
    steps=(load(), set_("a", [0, 0]), set_("a", "full")),
    checks=(auto(),),
    labels=Labels(family="histograms", models=("HistogramCDS", "HistoNdCDS", "RangeFilter"),
                  verified=("HistogramCDS", "HistoNdCDS", "RangeFilter"),
                  technique="empty selection, shifted range, recovery (reviewer GPT24)", priority="p1"),
)

SCENARIOS = (PLAIN, RB1_NAN, DEEP_CHAIN, C3_LIMITS, C10_TWO_WIDGETS, GPT24_EMPTY)
