# RootInteractive/InteractiveDrawing/bokeh/conftest.py
"""
Bokeh-specific pytest configuration and fixtures.

Markers are registered at repo level (../../../conftest.py).
This file provides bokeh-specific fixtures and feature ID validation.

Phase: 0.1.A
Date: 2026-02-02
"""

import pytest
import sys
from pathlib import Path


# =============================================================================
# PATH SETUP
# =============================================================================

_this_dir = Path(__file__).parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))


# =============================================================================
# FEATURE ID VALIDATION (Lazy Import)
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Validate feature IDs against taxonomy.
    
    Uses LAZY IMPORT to avoid loading taxonomy for every pytest run.
    Only validates when tests actually use @pytest.mark.feature.
    
    This catches typos like @pytest.mark.feature("DSL.arithmetc_expr")
    that would silently create untested features.
    """
    # Quick check: do any tests use feature markers?
    needs_validation = False
    for item in items:
        if list(item.iter_markers(name="feature")):
            needs_validation = True
            break
    
    if not needs_validation:
        return
    
    # Lazy import taxonomy
    FEATURE_TAXONOMY = None
    FEATURE_ALIASES = None
    
    try:
        from feature_taxonomy import FEATURE_TAXONOMY, FEATURE_ALIASES
    except ImportError:
        # Taxonomy not available - skip validation
        # This allows tests to run even without feature_taxonomy.py
        return
    
    if FEATURE_TAXONOMY is None:
        return
    
    # Validate feature IDs
    invalid_features = []
    for item in items:
        for marker in item.iter_markers(name="feature"):
            if marker.args:
                feature_id = marker.args[0]
                # Check both direct and aliased
                if feature_id not in FEATURE_TAXONOMY:
                    if FEATURE_ALIASES is None or feature_id not in FEATURE_ALIASES:
                        invalid_features.append((item.nodeid, feature_id))
    
    if invalid_features:
        msg = "\n\nInvalid feature IDs found:\n"
        for nodeid, feature_id in invalid_features[:10]:
            msg += f"  - '{feature_id}' in {nodeid}\n"
        if len(invalid_features) > 10:
            msg += f"  ... and {len(invalid_features) - 10} more\n"
        msg += "\nValid feature IDs are defined in feature_taxonomy.py\n"
        msg += f"Available categories: {', '.join(sorted(set(k.split('.')[0] for k in FEATURE_TAXONOMY)))}\n"
        
        # Show similar feature IDs (typo detection)
        for nodeid, feature_id in invalid_features[:3]:
            similar = [k for k in FEATURE_TAXONOMY if feature_id.split('.')[-1] in k]
            if similar:
                msg += f"\nDid you mean: {', '.join(similar[:3])}?\n"
        
        pytest.fail(msg, pytrace=False)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def node_available():
    """Check if Node.js is available for cross-backend tests."""
    import subprocess
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture
def cwd_bokeh():
    """Return path to bokeh directory."""
    return Path(__file__).parent


@pytest.fixture
def test_data_dir():
    """Return path to test data directory (if exists)."""
    data_dir = Path(__file__).parent / "test_data"
    if data_dir.exists():
        return data_dir
    return None
