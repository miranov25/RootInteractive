# RootInteractive/onnx/conftest.py
"""
ONNX-specific pytest configuration.

Registers custom markers and validates test prerequisites.

Phase: 0.1.E
"""

import pytest
import subprocess
import sys
from pathlib import Path


# =============================================================================
# MARKER REGISTRATION
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "feature(id): RootInteractive feature ID")
    config.addinivalue_line("markers", "bench: Benchmark test (slow, excluded from CI)")


# =============================================================================
# NODE.JS AVAILABILITY CHECK
# =============================================================================

_node_available = None

def check_node_available():
    """Check if Node.js is available."""
    global _node_available
    if _node_available is None:
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True, text=True, timeout=5
            )
            _node_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            _node_available = False
    return _node_available


_onnxruntime_node_available = None

def check_onnxruntime_node():
    """Check if onnxruntime-node is installed."""
    global _onnxruntime_node_available
    if _onnxruntime_node_available is None:
        if not check_node_available():
            _onnxruntime_node_available = False
        else:
            try:
                result = subprocess.run(
                    ["node", "-e", "require('onnxruntime-node')"],
                    capture_output=True, text=True, timeout=10,
                    cwd=str(Path(__file__).parent)
                )
                _onnxruntime_node_available = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                _onnxruntime_node_available = False
    return _onnxruntime_node_available


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def node_available():
    """Check if Node.js + onnxruntime-node are available."""
    return check_onnxruntime_node()


# =============================================================================
# REFERENCE DATA CHECK
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """Check that reference data exists before running tests."""
    ref_dir = Path(__file__).parent / "reference_data"
    models_dir = Path(__file__).parent / "models"

    if not ref_dir.exists() or not models_dir.exists():
        skip_msg = pytest.mark.skip(
            reason="Reference data missing. Run: python generate_models.py"
        )
        for item in items:
            item.add_marker(skip_msg)
