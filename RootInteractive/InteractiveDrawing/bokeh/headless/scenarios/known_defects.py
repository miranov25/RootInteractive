"""Known RootInteractive defects as strict expected failures (stage 4.2b). Each scenario must FAIL today, at the
declared checks and steps, exactly as its signature describes; every other check of the scenario must PASS. When a
later fix removes a defect, its expectation turns into XPASS, which fails the run until the expectation is removed
(proposal D-13: a fix declares in advance which scenarios it flips).

B1  — with several range filters, later filters are not fully applied (LazyIntersectionFilter.ts 191/193/208/210,
      `||=`). Exact mechanism, measured (4.2b review): the first filter works; every later filter keeps the rows it
      excluded at load and otherwise follows its widget (oracles/filters.py `b1_selection`). The signature requires the
      page to equal that selection exactly, so another overcount (e.g. a lost widget update) is a normal failure.
B2  — a 1-D histogram ignores the selection at page load (HistogramCDS.ts `initialize`). Bug report v0.2.
B8  — a 1-D histogram counts NaN values in bin 0 (HistogramCDS.ts `getbin`: (NaN*scale + origin)|0 = 0).
      Bug report, P0, decision D-12.
"""
from ..engine.expect import EqualsSelection, EqualsUnselected, ExcessInCells, expected_failure
from ..engine.scenario import LOAD_CHECK, Labels, Page, Scenario, TableSpec, Tables, auto, hist, load, range_, set_

B1_RULE = "b1_later_filters_keep_rows_excluded_at_load"

B1_TWO_FILTERS = Scenario(
    id="known_defects/b1_two_filters_missing_values",
    data=Tables(main=TableSpec(n=20_000, seed=23, columns={"x": "uniform(0,1)", "y": "normal(0,1)",
                                                           "a": "normal(0,1) + 10% NaN",
                                                           "b": "normal(0,1) + 10% NaN"})),
    page=Page(histograms=(hist("h2", ["x", "y"], [10, 2], [[0, 1], [-5, 5]]),),
              widgets=(range_("a"), range_("b"))),
    steps=(load(), set_("a", [-1, 1]), set_("b", [-1, 1])),
    checks=(auto(),),
    expect=(
        expected_failure("B1", "h2.bin_count", (0, 1, 2), EqualsSelection(B1_RULE),
                         note="the rows the second filter excluded at load stay selected, at load and after "
                              "both moves"),
        expected_failure("B1", LOAD_CHECK, (0,), EqualsSelection(B1_RULE),
                         note="at load the page selects exactly the rows of the first filter"),
    ),
    labels=Labels(family="known_defects", models=("RangeFilter", "LazyIntersectionFilter", "HistoNdCDS"),
                  verified=("RangeFilter", "LazyIntersectionFilter", "HistoNdCDS"),
                  features=("FILTER.intersection",), technique="state sequence, known defect B1", priority="p0"),
)

B2_LOAD_SELECTION = Scenario(
    id="known_defects/b2_1d_histogram_ignores_selection_at_load",
    data=Tables(main=TableSpec(n=20_000, seed=31, columns={"x": "uniform(0,1)", "y": "normal(0,1)",
                                                           "a": "normal(0,1)"})),
    page=Page(histograms=(hist("h1", ["x"], 10, [0, 1]),
                          hist("h2", ["x", "y"], [10, 4], [[0, 1], [-3, 3]])),
              widgets=(range_("a", (-1, 1)),)),           # limits narrower than the data: rows excluded at load
    steps=(load(), set_("a", [-0.5, 0.5]), set_("a", [-1, 1])),
    checks=(auto(),),
    expect=(
        expected_failure("B2", "h1.bin_count", (0,), EqualsUnselected(),
                         note="at load the 1-D histogram shows all rows; after a move it follows the selection; "
                              "the 2-D histogram follows it from load on"),
    ),
    labels=Labels(family="known_defects", models=("RangeFilter", "HistogramCDS", "HistoNdCDS"),
                  verified=("RangeFilter", "HistogramCDS", "HistoNdCDS"),
                  features=("HIST.histogram_1d",), technique="state sequence, known defect B2", priority="p0"),
)

B8_NAN_IN_BIN0 = Scenario(
    id="known_defects/b8_1d_histogram_counts_nan_in_bin0",
    data=Tables(main=TableSpec(n=20_000, seed=37, columns={"x": "uniform(0,1) + 10% NaN", "y": "normal(0,1)",
                                                           "a": "normal(0,1)"})),
    page=Page(histograms=(hist("h1", ["x"], 10, [0, 1]),
                          hist("h2", ["x", "y"], [10, 4], [[0, 1], [-3, 3]])),
              widgets=(range_("a"),)),
    steps=(load(), set_("a", [-1, 1]), set_("a", "full")),
    checks=(auto(),),
    expect=(
        expected_failure("B8", "h1.bin_count", (0, 1, 2), ExcessInCells((0,), "nan_rows"),
                         note="bin 0 exceeds the oracle by exactly the selected rows whose x is NaN; the 2-D "
                              "histogram excludes them correctly"),
    ),
    labels=Labels(family="known_defects", models=("RangeFilter", "HistogramCDS", "HistoNdCDS"),
                  verified=("RangeFilter", "HistogramCDS", "HistoNdCDS"),
                  features=("HIST.histogram_1d",), technique="boundary value (missing values), known defect B8",
                  priority="p0"),
)

# ---- regression scenarios written by the reviewers of stage 4.2b without touching the engine (credited by name) ----
C3_B1_THREE_FILTERS = Scenario(
    id="known_defects/claude3_b1_three_filters_missing_values",
    data=Tables(main=TableSpec(n=20_000, seed=43, columns={"x": "uniform(0,1)", "y": "normal(0,1)",
                                                           "a": "normal(0,1) + 10% NaN", "b": "normal(0,1) + 10% NaN",
                                                           "c": "normal(0,1) + 10% NaN"})),
    page=Page(histograms=(hist("h2", ["x", "y"], [10, 2], [[0, 1], [-5, 5]]),),
              widgets=(range_("a"), range_("b"), range_("c"))),
    steps=(load(), set_("a", [-1, 1]), set_("b", [-1, 1]), set_("c", [-1, 1])),
    checks=(auto(),),
    expect=(expected_failure("B1", "h2.bin_count", (0, 1, 2, 3), EqualsSelection(B1_RULE)),
            expected_failure("B1", LOAD_CHECK, (0,), EqualsSelection(B1_RULE))),
    labels=Labels(family="known_defects", models=("RangeFilter", "LazyIntersectionFilter", "HistoNdCDS"),
                  verified=("RangeFilter", "LazyIntersectionFilter", "HistoNdCDS"), features=("FILTER.intersection",),
                  technique="known defect B1, three filters (reviewer Claude3)", priority="p1"),
)

C3_B8_WEIGHTED = Scenario(
    id="known_defects/claude3_b8_weighted_1d_histogram",
    data=Tables(main=TableSpec(n=20_000, seed=41, columns={"x": "uniform(0,1) + 10% NaN", "y": "normal(0,1)",
                                                           "a": "normal(0,1)", "w": "uniform(0.5,2)"})),
    page=Page(histograms=(hist("h1", ["x"], 10, [0, 1], weights="w"),
                          hist("h2", ["x", "y"], [10, 4], [[0, 1], [-3, 3]], weights="w")),
              widgets=(range_("a"),)),
    steps=(load(), set_("a", [-1, 1]), set_("a", "full")),
    checks=(auto(),),
    expect=(expected_failure("B8", "h1.bin_count", (0, 1, 2), ExcessInCells((0,), "nan_rows"),
                             note="bin 0 exceeds the oracle by the summed weights of the selected NaN rows"),),
    labels=Labels(family="known_defects", models=("RangeFilter", "HistogramCDS", "HistoNdCDS"),
                  verified=("RangeFilter", "HistogramCDS", "HistoNdCDS"), features=("HIST.histogram_1d",),
                  technique="known defect B8, weighted (reviewer Claude3)", priority="p1"),
)

C10_B8_WEIGHTED = Scenario(
    id="known_defects/claude10_b8_weighted_1d_histogram_nan_in_bin0",
    data=Tables(main=TableSpec(n=20_000, seed=41, columns={"x": "uniform(0,1) + 10% NaN", "y": "normal(0,1)",
                                                           "a": "normal(0,1)", "w": "uniform(0,1)"})),
    page=Page(histograms=(hist("h1w", ["x"], 10, [0, 1], weights="w"),
                          hist("h2", ["x", "y"], [10, 4], [[0, 1], [-3, 3]])),
              widgets=(range_("a"),)),
    steps=(load(), set_("a", [-1, 1]), set_("a", "full")),
    checks=(auto(),),
    expect=(expected_failure("B8", "h1w.bin_count", (0, 1, 2), ExcessInCells((0,), "nan_rows"),
                             note="weighted variant: bin 0 exceeds the oracle by the weight sum of the selected "
                                  "NaN rows"),),
    labels=Labels(family="known_defects", models=("RangeFilter", "HistogramCDS", "HistoNdCDS"),
                  verified=("RangeFilter", "HistogramCDS", "HistoNdCDS"), features=("HIST.histogram_1d",),
                  technique="known defect B8, weighted (reviewer Claude10)", priority="p1"),
)

SCENARIOS = (B1_TWO_FILTERS, B2_LOAD_SELECTION, B8_NAN_IN_BIN0, C3_B1_THREE_FILTERS, C3_B8_WEIGHTED, C10_B8_WEIGHTED)
