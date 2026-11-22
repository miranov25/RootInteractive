"""PHASE_5_2 stage 1b: the line's series and legend, finished (FR-RI-1 items 2 and 6).

In plain words.
- The compiler gives its groups as the weights of a profile and colours them with its own fixed colour mapper: the
  figure option "color" = factor_cmap("weights", its palette, all its groups). Its markers keep their colour whatever
  is selected. At stage 1 the lines took RootInteractive's positional colours instead (the k-th selected weight, the
  k-th Category10 colour), so a line and its markers could differ. Now each line takes its weight's colour in the
  figure's own mapper.
- Groups told apart by any other legend_field column would be joined by one line (stage 1 drew that silently). They
  are refused by name; the groups go in as weights.
- line_style legend_label gives a single line a legend label of its own (FR-RI-1 item 6). The series of a weighted
  profile take their weights' labels, so there it is refused by name.
- Before the fix (commit 1 of stage 1b, the tests only) each of these failed with stage 1's exact result: strict
  expected failures (KnownDefect). Since the fix they are regressions that name it.
Author: Claude10.RI, 2026-11-22.
"""
from __future__ import annotations

import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_phase5_2_lines import (
    _build_error, _from_array, _glyphs, _node, _of, _open, _page, _set, _state, _table, _validation_errors)


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect or missing feature."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


def regression(condition: bool, message: str) -> None:
    """After the fix, the exact signature of a fixed defect is a regression: an ordinary failure that names it."""
    if condition:
        raise AssertionError("REGRESSION, a fixed defect is back: " + message)


STRICT = dict(strict=True, raises=KnownDefect)
STAGE_1_TODAY = pytest.mark.xfail(**STRICT, reason="FR-RI-1 items 2 and 6, as left by stage 1")
UNIT = pytest.mark.headless_unit
OWN_PALETTE = ["#d62728", "#2ca02c", "#9467bd"]          # the compiler's own colours of w0, w1, w2 (not Category10's order)
CATEGORY10 = ["#1f77b4", "#ff7f0e", "#2ca02c"]           # RootInteractive's positional colours of the 1st, 2nd, 3rd selected


def _own_colours(kind="scatter+line"):
    from bokeh.transform import factor_cmap
    return [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": kind, "filter": "entries >= 3",
                                         "color": factor_cmap("weights", OWN_PALETTE, ["w0", "w1", "w2"])}]


# ------------------------------------------------------------------ the compiler's fixed colours
@_node
@pytest.mark.feature("DISPLAY.line")
@pytest.mark.feature("HIST.profile")
@pytest.mark.parametrize("kind", ["scatter+line", "line"])
@STAGE_1_TODAY
def test_each_line_keeps_its_weights_colour_from_the_figures_own_colour_mapper(node_session, tmp_path, kind):
    """The figure gives color = factor_cmap("weights", own palette, all weights). Through three selections each shown
    line has its weight's own colour, equal to its markers' colour (with markers)."""
    _open(node_session, _page(tmp_path, f"own_{kind}", _table(), [_own_colours(kind)], weighted=True))
    for selection in (["w0", "w1"], ["w2", "w0"], ["w1"]):
        if selection != ["w0", "w1"]:
            _set(node_session, "weights", selection)
        s = _state(node_session)
        shown = {r["weights"][0]: r["color"] for r in _of(s, "Line") if r["visible"]}
        positional = {w: CATEGORY10[k] for k, w in enumerate(selection)}
        known_defect(shown == positional and shown != {w: OWN_PALETTE[int(w[1])] for w in selection},
                     f"stage 1: the lines take the positional colours {positional}, not the figure's own")
        assert shown == {w: OWN_PALETTE[int(w[1])] for w in selection}, (selection, shown)
        if kind == "scatter+line":
            (markers,) = _of(s, "Scatter")
            assert all(markers["palette"][w] == c for w, c in shown.items()), (markers["palette"], shown)


# ------------------------------------------------------------------ groups of another column
@UNIT
@pytest.mark.feature("DISPLAY.line")
@STAGE_1_TODAY
def test_groups_of_another_legend_field_column_are_refused_not_joined():
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "scatter+line", "legend_field": "entries"}]
    error = _build_error(_table(), [figure])
    if error is None:
        known_defect(_glyphs(_from_array(_table(), [figure])) == ["Line", "Scatter"],
                     "stage 1: one line joins the groups of legend_field 'entries'")
    assert error is not None and error[0] == "ValueError", error
    assert "legend_field 'entries'" in error[1] and "weights" in error[1], error


@UNIT
@pytest.mark.feature("DISPLAY.line")
def test_control_the_weights_legend_field_on_a_weighted_profile_is_accepted():
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "scatter+line", "legend_field": "weights"}]
    page = _from_array(_table(), [figure], weighted=True)
    assert _glyphs(page) == ["Line", "Line", "Line", "Scatter"] and _validation_errors(page) == []


# ------------------------------------------------------------------ a line's own legend label
def _legend_labels(page):
    from bokeh.models import Legend
    out = []
    for legend in (m for m in page.pAll.references() if isinstance(m, Legend)):
        for item in legend.items:
            label = item.label
            out.append((label.get("value") if isinstance(label, dict) else getattr(label, "value", label),
                        sorted(type(r.glyph).__name__ for r in item.renderers)))
    return out


@UNIT
@pytest.mark.feature("DISPLAY.line")
@pytest.mark.feature("DISPLAY.legend")
@pytest.mark.parametrize("kind,want", [("line", [("profile fit", ["Line"])]),
                                       ("scatter+line", [("mean B vs A", ["Scatter"]), ("profile fit", ["Line"])])])
@STAGE_1_TODAY
def test_a_single_line_has_its_own_legend_label(kind, want):
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": kind,
                                           "line_style": {"legend_label": "profile fit"}}]
    error = _build_error(_table(), [figure])
    known_defect(error is not None and error[0] == "ValueError" and "unknown key 'legend_label'" in error[1],
                 "stage 1: line_style has no legend_label")
    assert error is None, error
    page = _from_array(_table(), [figure])
    labels = _legend_labels(page)
    assert sorted(labels) == sorted(want), labels
    assert _validation_errors(page) == []


@UNIT
@pytest.mark.feature("DISPLAY.line")
@STAGE_1_TODAY
def test_a_legend_label_of_ones_own_on_weight_series_is_refused():
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "line",
                                           "line_style": {"legend_label": "fit"}}]
    error = _build_error(_table(), [figure], weighted=True)
    known_defect(error is not None and "unknown key 'legend_label'" in error[1], "stage 1: line_style has no legend_label")
    assert error is not None and error[0] == "ValueError" and "legend_label" in error[1] and "weight" in error[1], error
