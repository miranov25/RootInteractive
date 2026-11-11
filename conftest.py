# RootInteractive/conftest.py (REPO ROOT)
"""
Root-level pytest configuration for RootInteractive.

Fixes:
1. Registers custom markers globally
2. Replaces bokeh.io.show() with save() - creates HTML without opening browser
3. Keeps output_file() working normally

Phase: 0.1.A
Date: 2026-02-03
"""

import pytest
import os
import warnings


# =============================================================================
# REPLACE BOKEH SHOW() WITH SAVE() — Create HTML without opening browser
# =============================================================================

def pytest_configure(config):
    """
    Register ALL custom markers and configure bokeh for testing.
    """
    # -------------------------------------------------------------------------
    # Set environment variable to disable browser opening (backup)
    # -------------------------------------------------------------------------
    os.environ['BOKEH_BROWSER'] = 'none'
    
    # -------------------------------------------------------------------------
    # Replace show() with save() so HTML files are created but browser doesn't open
    # P0-4 FIX: Use warnings.warn() instead of silent pass
    # -------------------------------------------------------------------------
    try:
        import bokeh.io
        from bokeh.io import save as bokeh_save
        
        def show_as_save(obj=None, browser=None, new=None, notebook_handle=False, notebook_url="localhost:8888", **kwargs):
            """Replace show() with save() - creates HTML file without opening browser."""
            if obj is not None:
                try:
                    bokeh_save(obj)
                except Exception as e:
                    # P0-4 FIX: Do NOT silently swallow - warn so errors are visible
                    warnings.warn(f"Bokeh save() failed: {e}", RuntimeWarning)
        
        bokeh.io.show = show_as_save
        
        # Also patch bokeh.plotting.show
        try:
            import bokeh.plotting
            bokeh.plotting.show = show_as_save
        except ImportError:
            pass
            
    except ImportError:
        pass  # bokeh not installed
    
    # -------------------------------------------------------------------------
    # Feature tracking markers (CAPABILITY_MATRIX)
    # -------------------------------------------------------------------------
    config.addinivalue_line(
        "markers",
        "feature(name): Feature ID from FEATURE_TAXONOMY (e.g., 'DSL.arithmetic_expr')"
    )
    
    # -------------------------------------------------------------------------
    # Backend markers
    # -------------------------------------------------------------------------
    config.addinivalue_line(
        "markers",
        "backend(name): Backend required (python, node, browser)"
    )
    
    # -------------------------------------------------------------------------
    # Layer markers
    # -------------------------------------------------------------------------
    config.addinivalue_line(
        "markers",
        "layer(name): Test layer (unit, integration, invariance, vector)"
    )
    
    # -------------------------------------------------------------------------
    # Priority markers
    # -------------------------------------------------------------------------
    config.addinivalue_line(
        "markers",
        "p0: Priority 0 - critical/blocking"
    )
    config.addinivalue_line(
        "markers",
        "p1: Priority 1 - important/required"
    )
    config.addinivalue_line(
        "markers",
        "p2: Priority 2 - nice to have"
    )
    
    # -------------------------------------------------------------------------
    # Execution markers
    # -------------------------------------------------------------------------
    config.addinivalue_line(
        "markers",
        "cross_backend: Test requires both Python and JavaScript execution"
    )
    config.addinivalue_line(
        "markers",
        "slow: Mark test as slow-running"
    )
    config.addinivalue_line(
        "markers",
        "regression(pr): Regression test for a specific PR"
    )
    config.addinivalue_line(
        "markers",
        "gui: Test creates GUI/visualization (may be slow)"
    )
    config.addinivalue_line(
        "markers",
        "unittest: Unit test marker (legacy)"
    )

    # -------------------------------------------------------------------------
    # Target markers (PHASE_4 stage 4.2c-2c): ./run_tests.sh --target <name> == pytest <headless folder> -m <name>
    # -------------------------------------------------------------------------
    config.addinivalue_line(
        "markers",
        "headless_unit: target: Python-only headless tests: oracles, comparison, expectations, packaging, session end, targets"
    )
    config.addinivalue_line(
        "markers",
        "node_harness: target: the Node model harness and its fault controls"
    )
    config.addinivalue_line(
        "markers",
        "node_scenarios: target: every computing scenario of 4.2a/4.2b in Node alone (B1, B2, B8 by their exact descriptions)"
    )
    config.addinivalue_line(
        "markers",
        "node_histograms: target: histogram families at sample size, rule and B10 controls, template, batch and reduced-document checks"
    )
    config.addinivalue_line(
        "markers",
        "node_histograms_full: target: histogram families at full size (3 x 3 400 generated tables)"
    )
    config.addinivalue_line(
        "markers",
        "node_filters: target: B1 row-count sweep at sample row counts, 2 and 3 filters in every move order, 20 000 / 20 001 rows"
    )
    config.addinivalue_line(
        "markers",
        "node_filters_full: target: B1 row-count sweep 1-200 rows, 4 and 5 filters in every move order"
    )
    # node_joins has its own call (R-6: it was glued to the string above)
    config.addinivalue_line(
        "markers",
        "node_joins: target: production CDSJoin in Node - joins and recursive joins, oracle and invariant tests (PHASE_4_3)"
    )
    config.addinivalue_line(
        "markers",
        "node_aliases: target: production aliases in Node - expressions, functions, parameters, gathers, change propagation (PHASE_4_5 A1)"
    )
    config.addinivalue_line(
        "markers",
        "node_widgets: target: interactive widgets in Node - every widget type changes its filter or parameter, histograms and profiles follow (PHASE_4_5 A3)"
    )
    config.addinivalue_line(
        "markers",
        "browser_widgets: target: interactive widgets in Chromium - real keyboard, click and select input on every widget type (PHASE_4_5 A3)"
    )
    config.addinivalue_line(
        "markers",
        "node_a4: target: three linked tables V0 -> track -> event on one real page in Node, every state against GPT24's shared reference (PHASE_4_5 A4)"
    )
    config.addinivalue_line(
        "markers",
        "browser_a4: target: the same A4 page in Chromium - real multi-select, keyboard and select input, painted V0 rows, browser = Node (PHASE_4_5 A4)"
    )
    config.addinivalue_line(
        "markers",
        "node_stats: target: statistics computed in the page in Node - statistics table, linear fit, integrals, weighted unbinned quantiles, stacked histograms (PHASE_4_6 C1)"
    )
    config.addinivalue_line(
        "markers",
        "node_display: target: widgets and display in Node - text query, downsampler, lazy tabs, parametric axis labels (PHASE_4_6 C2)"
    )
    config.addinivalue_line(
        "markers",
        "browser_display: target: widgets and display in Chromium - typed text query, tab click, axis label after a real widget change (PHASE_4_6 C2)"
    )
    config.addinivalue_line(
        "markers",
        "node_transport: target: the compressed table CDSCompress in a real page in Node - lossless decoding, lazy cache, histogram under a filter, data replacement, wrong-length guard (PHASE_4_6 D1p2)"
    )
    config.addinivalue_line(
        "markers",
        "browser_smoke: target: the plain page in Chromium and its browser = Node parity pair"
    )
    config.addinivalue_line(
        "markers",
        "browser_scenarios: target: every scenario in Chromium"
    )
    config.addinivalue_line(
        "markers",
        "browser_parity: target: browser = Node parity for every scenario, and the parity controls"
    )
    config.addinivalue_line(
        "markers",
        "browser_harness: target: the browser engine's harness and fault controls"
    )
