"""Stage 4.2a scenarios: written as data; adding one needs no engine change.

1. `histograms/plain_1d_2d_one_range_two_moves` — the plain scenario of the 4.2a acceptance: a 1-D and a 2-D
   histogram, one range widget, two moves. The widget column has no missing values, so the 1-D histogram's known
   load-state defect (B2, B2') cannot show here; that defect gets its own scenario in stage 4.2b.
2. `selection/rb1_nan_excluded_at_full_bounds` — the measured evidence for rule RB-1.2 (erratum E-3): with 10 %
   missing values in the widget column, the N-D histogram at load must exclude exactly those rows.
3. `chain/deep_widget_filter_alias_stack_profile` — the deepest chain the engine supports in 4.2a: range widget
   -> RangeFilter -> LazyIntersectionFilter -> CDSAlias (alias xa = x*2) -> two weighted HistoNdCDS (a parameter
   with two weight options) -> CDSStack -> HistoNdProfile. Checked by the automatic oracle and by an audit that
   re-reads every probed model one second after the completion condition was met.
"""
from ..engine.scenario import (Labels, Page, Parameter, Scenario, TableSpec, Tables, audit, auto, hist, load,
                               range_, set_)

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
    labels=Labels(family="selection", models=("RangeFilter", "HistoNdCDS"), features=("HIST.histogram_nd",),
                  technique="boundary value (missing values at full bounds)", priority="p0"),
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
    checks=(auto(), audit(delay_s=1.0)),
    labels=Labels(family="chain", models=("RangeFilter", "LazyIntersectionFilter", "CDSAlias", "HistoNdCDS",
                                          "CDSStack", "HistoNdProfile"),
                  features=("HIST.histogram_nd",), technique="state sequence, completion audit", priority="p0"),
)

SCENARIOS = (PLAIN, RB1_NAN, DEEP_CHAIN)
