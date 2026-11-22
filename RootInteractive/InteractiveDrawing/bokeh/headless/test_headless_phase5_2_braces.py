"""PHASE_5_2 stage 2: text with braces in titles and axis titles, shown as written (FR-RI-4).

In plain words.
- A figure's title, its axis titles and its legend text are templates: every {name} is replaced by a parameter's value
  (which then follows its widget), else by the name's AxisTitle from the table's metadata, else by the name itself.
- At the snapshot every {...} was replaced, so a title could not show a brace, and TeX lost its braces:
  "$$p_{T,max}$$" was shown as "$$p_T,max$$".
- The rules now (design D2 (a) of the PHASE_5_2 proposal, as in Python's str.format, plus TeX kept as written):
  - {name}: replaced as before;
  - {{ and }}: one written brace each;
  - TeX ($$...$$, \\(...\\), \\[...\\]): kept exactly as written, braces included;
  - a brace without a partner: kept as written.
- One visible change for text that worked before: "a}}b" now shows "a}b" (a doubled brace is one brace).
- Before the fix (commit 1, the tests only) the cases that need the new rules failed with the snapshot's exact
  result, computed by the snapshot's rule (_old_rule): strict expected failures (KnownDefect). Since the fix (commit 2)
  that result is a regression that names it.
Controls (pass before and after): the parametric title follows its slider, RI's own "{transform} label" axis title
follows its select, and plain text is unchanged.
Author: Claude10.RI, 2026-11-22.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest


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
FR_RI_4_TODAY = pytest.mark.xfail(**STRICT, reason="FR-RI-4: every {...} is replaced; TeX loses its braces")
UNIT = pytest.mark.headless_unit


def _node(f):
    """The target, backend and layer marks of a Node test. The feature marks are written out on each test, because
    scripts/generate_capability_matrix.py reads only the literal form @pytest.mark.feature("ID")."""
    for m in reversed([pytest.mark.node_display, pytest.mark.backend("node"), pytest.mark.layer("integration")]):
        f = m(f)
    return f


# ------------------------------------------------------------------ the snapshot's rule (the signature)
def _old_rule(template, params=None, titles=None):
    """What the snapshot showed: re.split at \\{(.*?)\\}, every captured name replaced (parameter value, AxisTitle,
    else the name)."""
    params, titles = params or {}, titles or {}
    parts = re.split(r"\{(.*?)\}", template)
    for i in range(1, len(parts), 2):
        name = parts[i]
        value = str(params[name]) if name in params else name
        parts[i] = titles.get(value, value)
    return "".join(parts)


# ------------------------------------------------------------------ data and pages (the public builder)
def _table() -> pd.DataFrame:
    from RootInteractive.Tools.pandaTools import initMetadata
    rng = np.random.default_rng(5)
    d = pd.DataFrame({"A": rng.random(200) * 4, "B": rng.random(200) * 4 + 1})
    initMetadata(d)
    d.meta.metaData["B.AxisTitle"] = "B title"
    return d


PARAMS = [{"name": "nBins", "value": 10, "range": [5, 50], "step": 1},
          {"name": "transformY", "value": "sqrt", "options": [None, "sqrt"]}]
WIDGETS = [["slider", ["nBins"]], ["select", ["transformY"]]]


def _from_array(figures, histograms=()):
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    before = sa.show
    sa.show = lambda *a, **k: None
    try:
        return sa.bokehDrawSA.fromArray(_table(), None, list(figures), [list(w) for w in WIDGETS],
                                        parameterArray=[dict(p) for p in PARAMS], histogramArray=list(histograms),
                                        layout=[list(range(len(figures)))], widgetLayout=[list(range(len(WIDGETS)))])
    finally:
        sa.show = before


def _shown(page):
    """The texts the figures show: [(title, x axis title, y axis title)] in figure order."""
    from bokeh.models import Plot
    plots = [m for m in page.pAll.references() if isinstance(m, Plot)]
    plots.sort(key=lambda p: p.id)
    return [(p.title.text, p.xaxis[0].axis_label, p.yaxis[0].axis_label) for p in plots]


# The cases of the feature request's table, with what the page shows now. "old" is what the snapshot showed.
CASES = [
    ("p_T {GeV}", "p_T GeV"),                                 # a name that is no parameter: replaced, as before
    ("p_T {{GeV}}", "p_T {GeV}"),                             # a written brace
    ("bins {nBins}", "bins 10"),                              # the parametric title
    ("{{nBins}}", "{nBins}"),                                 # written braces around a parameter's name: no value
    ("f(x) = {{x}}^2", "f(x) = {x}^2"),
    ("$$p_{T,max}$$", "$$p_{T,max}$$"),                       # TeX as written
    (r"$$\frac{a}{b}$$", r"$$\frac{a}{b}$$"),
    ("a{b", "a{b"), ("a}b", "a}b"),                            # unpaired: as written
    ("a}}b", "a}b"),                                          # the one change for text that worked before
    ("{{GeV}}", "{GeV}"),
    ("{{{B}}}", "{B title}"),                                 # a written brace around a name (B's AxisTitle)
    ("$$p_{T}$$ in {nBins} bins, {{raw}}", "$$p_{T}$$ in 10 bins, {raw}"),
]
OLD = {"nBins": 10}
CHANGED = [t for t, want in CASES if _old_rule(t, OLD, {"B": "B title"}) != want]


@UNIT
def test_the_cases_that_change_are_exactly_these():
    """The signature is not vacuous: the snapshot's rule gives a different text for exactly these cases."""
    assert CHANGED == ["$$p_{T,max}$$", r"$$\frac{a}{b}$$", "a}}b", "{{{B}}}", "$$p_{T}$$ in {nBins} bins, {{raw}}"]
    assert _old_rule("$$p_{T,max}$$") == "$$p_T,max$$" and _old_rule("{{GeV}}") == "{GeV}"


def _check_case(shown, template, want):
    old = _old_rule(template, OLD, {"B": "B title"})
    if old != want:
        known_defect(shown == old, f"FR-RI-4: {template!r} is shown as {old!r}")
    assert shown == want, (template, shown, want)


@UNIT
@pytest.mark.feature("DISPLAY.text_braces")
@pytest.mark.parametrize("template,want", [c for c in CASES if c[0] not in CHANGED])
def test_control_titles_that_do_not_change(template, want):
    """Plain text, single names, written braces that the snapshot showed right by accident: the same before and after."""
    page = _from_array([[["A"], ["B"], {"plotTitle": template, "xAxisTitle": template, "yAxisTitle": template}]])
    for shown in _shown(page)[0]:
        _check_case(shown, template, want)


@UNIT
@pytest.mark.feature("DISPLAY.text_braces")
@pytest.mark.parametrize("template,want", [c for c in CASES if c[0] in CHANGED])
@FR_RI_4_TODAY
def test_titles_and_axis_titles_show_braces_and_tex_as_written(template, want):
    page = _from_array([[["A"], ["B"], {"plotTitle": template, "xAxisTitle": template, "yAxisTitle": template}]])
    for shown in _shown(page)[0]:
        _check_case(shown, template, want)


@UNIT
@pytest.mark.feature("DISPLAY.text_braces")
@FR_RI_4_TODAY
def test_a_histogram_figure_shows_braces_and_tex_as_written():
    title = "$$p_{T}$$ {{GeV}}"
    page = _from_array([[["A"], ["hA"], {"xAxisTitle": title, "plotTitle": title}]],
                       histograms=[{"name": "hA", "variables": ["A"], "nbins": 8, "range": [0, 4]}])
    plot_title, x_title, _ = _shown(page)[0]
    for shown in (plot_title, x_title):
        known_defect(shown == _old_rule(title), f"FR-RI-4: {title!r} is shown as {_old_rule(title)!r}")
        assert shown == "$$p_{T}$$ {GeV}", shown


@UNIT
@pytest.mark.feature("DISPLAY.text_braces")
def test_control_the_default_titles_are_unchanged():
    """RI writes its own titles with braces: "{B}" (B's AxisTitle), "{transformY} {B}" for a transformed axis, the
    histogram's "{A}". They show as before."""
    page = _from_array([[["A"], ["B"]], [["A"], ["B"], {"y_transform": "transformY"}], [["A"], ["hA"]]],
                       histograms=[{"name": "hA", "variables": ["A"], "nbins": 8, "range": [0, 4]}])
    shown = _shown(page)
    assert shown[0] == ("B title vs A", "A", "B title"), shown[0]
    assert shown[1] == ("sqrt B title vs A", "A", "sqrt B title"), shown[1]
    assert shown[2][1] == "A", shown[2]


# ------------------------------------------------------------------ Node: the texts follow their widgets
STATE = r"""() => {
  const plots = window.__riH.models().filter(m => m.type === 'Figure' || m.type === 'Plot');
  plots.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const axis = (side) => side.find(m => /Axis$/.test(m.type));
  return plots.map(p => [p.title.text, axis(p.below).axis_label, axis(p.left).axis_label]);
}"""

SET = r"""async (a) => {
  const w = window.__riH.models().filter(m => 'value' in m && m.js_property_callbacks && (m.name === a.name || (m.name == null && m.title === a.name)));
  if (w.length !== 1) throw Error('widget identity: ' + a.name + ': ' + w.length);
  w[0].value = a.value;
  await new Promise(resolve => setTimeout(resolve, 0));
  return true;
}"""


def _page(tmp_path, tag, figures, histograms=()):
    from bokeh.document import Document
    doc = Document()
    doc.add_root(_from_array(figures, histograms).pAll)
    out = tmp_path / f"braces_{tag}.json"
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


def _open(session, path):
    r = session.request("new_doc", path=str(path))
    assert r.get("ok") and not r.get("page_errors") and not r.get("console_errors"), r


def _eval(session, code, arg=None):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = session.request("eval", code=code, arg=arg)
    assert r.get("ok") and not r.get("page_errors") and not r.get("console_errors"), r
    return decode(r["result"])


MIXED = "$$p_{T}$$ in {nBins} bins, {{raw}}"


@_node
@pytest.mark.feature("DISPLAY.text_braces")
@pytest.mark.feature("DISPLAY.axis_label")
@FR_RI_4_TODAY
def test_a_parametric_title_with_tex_and_braces_follows_its_slider(node_session, tmp_path):
    _open(node_session, _page(tmp_path, "mixed", [[["A"], ["B"], {"plotTitle": MIXED, "xAxisTitle": MIXED}]]))
    for n in (10, 20, 7):
        if n != 10:
            _eval(node_session, SET, {"name": "nBins", "value": n})
        title, x_title, _ = _eval(node_session, STATE)[0]
        old = _old_rule(MIXED, {"nBins": n})
        for shown in (title, x_title):
            known_defect(shown == old, f"FR-RI-4: {MIXED!r} is shown as {old!r}")
            assert shown == f"$$p_{{T}}$$ in {n} bins, {{raw}}", shown


@_node
@pytest.mark.feature("DISPLAY.text_braces")
@pytest.mark.feature("DISPLAY.axis_label")
def test_control_the_transform_axis_title_follows_its_select(node_session, tmp_path):
    _open(node_session, _page(tmp_path, "transform", [[["A"], ["B"], {"y_transform": "transformY"}]]))
    assert _eval(node_session, STATE)[0][2] == "sqrt B title"
    _eval(node_session, SET, {"name": "transformY", "value": "None"})
    assert _eval(node_session, STATE)[0][2] in ("None B title", " B title"), _eval(node_session, STATE)


# ------------------------------------------------------------------ Chromium: the page with TeX and braces
TEX_TITLE, AXIS_TITLE = "$$p_{T,max}$$", "bins {nBins}, {{raw}}"


@pytest.mark.browser_display
@pytest.mark.feature("DISPLAY.text_braces")
@FR_RI_4_TODAY
def test_browser_a_tex_title_and_a_braced_axis_title_are_drawn_and_follow_the_slider(headless_session, tmp_path):
    """Chromium: the TeX title keeps its braces and is drawn by Bokeh's MathJax (the page becomes idle, with no
    console error); the axis title with a written brace follows the slider, and the plot is redrawn.
    Bokeh 3.3.2 draws TeX only when the whole text is TeX: a title mixing text and $$...$$ is never drawn and the page
    never becomes idle, with or without this change (measured; the compiler writes such a title as one TeX text with
    \\text{...})."""
    import hashlib
    from bokeh.io import save
    from bokeh.resources import INLINE
    from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_display import _open_browser_page
    p = tmp_path / "braces_browser.html"
    save(_from_array([[["A"], ["B"], {"plotTitle": TEX_TITLE, "xAxisTitle": AXIS_TITLE}]]).pAll, filename=str(p),
         resources=INLINE, title="braces")
    ctx, page, errors = _open_browser_page(headless_session, p)
    try:
        state = page.evaluate(STATE)[0]
        known_defect(state[0] == _old_rule(TEX_TITLE), f"FR-RI-4: {TEX_TITLE!r} is shown as {_old_rule(TEX_TITLE)!r}")
        assert state[:2] == [TEX_TITLE, "bins 10, {raw}"], state
        canvas = page.locator("canvas").first
        before = hashlib.sha256(canvas.screenshot()).hexdigest()
        page.evaluate(SET, {"name": "nBins", "value": 30})
        page.wait_for_function("() => Bokeh.documents[0].is_idle")
        page.wait_for_timeout(300)
        assert page.evaluate(STATE)[0][:2] == [TEX_TITLE, "bins 30, {raw}"]
        assert hashlib.sha256(canvas.screenshot()).hexdigest() != before, "the plot was not redrawn"
        (tmp_path / "braces_browser.png").write_bytes(canvas.screenshot())
        assert not errors, errors
    finally:
        ctx.close()
