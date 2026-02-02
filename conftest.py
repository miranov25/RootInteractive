# RootInteractive/conftest.py (REPO ROOT)
"""
Root-level pytest configuration for RootInteractive.

Fixes:
1. Registers custom markers globally
2. Replaces bokeh.io.show() with save() - creates HTML without opening browser
3. Keeps output_file() working normally

Phase: 0.1.A
Date: 2026-02-02
"""

import pytest
import os


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
    # -------------------------------------------------------------------------
    try:
        import bokeh.io
        from bokeh.io import save as bokeh_save
        
        def show_as_save(obj=None, browser=None, new=None, notebook_handle=False, notebook_url="localhost:8888", **kwargs):
            """Replace show() with save() - creates HTML file without opening browser."""
            if obj is not None:
                try:
                    bokeh_save(obj)
                except Exception:
                    pass  # Ignore save errors silently
        
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
