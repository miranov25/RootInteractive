# RootInteractive/InteractiveDrawing/bokeh/feature_taxonomy.py
"""
Feature taxonomy for RootInteractive cross-backend testing.
Based on RDataFrameDSL pattern (Phase 13.6.B).

Usage:
    @pytest.mark.feature("DSL.arithmetic_expr")
    from feature_taxonomy import FEATURE_TAXONOMY

IMPORTANT: Feature IDs are IMMUTABLE. If renaming needed, use FEATURE_ALIASES.

Phase: 0.1.A
Date: 2026-02-02
Version: 1.0
"""

FEATURE_TAXONOMY_VERSION = "1.0"

# =============================================================================
# FEATURE ALIASES (for backward compatibility if renaming needed)
# =============================================================================
# Format: {"old_id": "new_id"}
# Usage: If feature ID must change, add mapping here instead of breaking tests
FEATURE_ALIASES = {}


# =============================================================================
# FEATURE TAXONOMY
# =============================================================================

FEATURE_TAXONOMY = {
    # =========================================================================
    # DSL Operations
    # =========================================================================
    "DSL.arithmetic_expr": {
        "name": "Arithmetic expressions",
        "description": "Basic arithmetic expressions (a + b * c)",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "unit",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },
    "DSL.math_functions": {
        "name": "Math functions",
        "description": "Math functions (arctan2, sqrt, sin, cos, etc.)",
        "priority": "P1",
        "backends": ["python", "node"],
        "layer": "unit",
        "proof": None,
        "planned": True,
    },
    "DSL.gather_operation": {
        "name": "Cross-table gather",
        "description": "Cross-table gather (dfB.a[i] syntax)",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "integration",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },
    "DSL.custom_js_func": {
        "name": "CustomJS function execution",
        "description": "CustomJS function execution in browser/Node",
        "priority": "P0",
        "backends": ["node", "browser"],
        "layer": "integration",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },

    # =========================================================================
    # Encoding / Data Transfer
    # =========================================================================
    "ENC.base64.float64": {
        "name": "Float64Array encoding",
        "description": "Base64 Float64Array encoding/decoding",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "unit",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },
    "ENC.base64.int32": {
        "name": "Int32Array encoding",
        "description": "Base64 Int32Array encoding/decoding",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "unit",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },
    "ENC.compression.relative": {
        "name": "Relative compression",
        "description": "Relative compression round-trip",
        "priority": "P1",
        "backends": ["python", "node"],
        "layer": "integration",
        "proof": None,
        "planned": True,
    },
    "ENC.compression.zip": {
        "name": "ZIP compression",
        "description": "ZIP compression round-trip",
        "priority": "P2",
        "backends": ["python", "node"],
        "layer": "integration",
        "proof": None,
        "planned": True,
    },

    # =========================================================================
    # Join Operations
    # =========================================================================
    "JOIN.cdsjoin.basic": {
        "name": "Basic CDSJoin",
        "description": "Basic CDSJoin index join",
        "priority": "P0",
        "backends": ["browser"],
        "layer": "integration",
        "proof": "test_ClientSideJoin.py::test_join",
    },
    "JOIN.cdsjoin.index0": {
        "name": "CDSJoin index-0 regression",
        "description": "CDSJoin with index 0 (regression test for x>=0 fix)",
        "priority": "P0",
        "backends": ["browser"],
        "layer": "integration",
        "proof": "test_ClientSideJoin.py::test_join",
        "regression": "PR-371",
    },
    "JOIN.cdsjoin.outer": {
        "name": "CDSJoin outer join",
        "description": "CDSJoin outer join with null handling",
        "priority": "P1",
        "backends": ["browser"],
        "layer": "integration",
        "proof": None,
        "planned": True,
    },
    "JOIN.cross_table": {
        "name": "Multi-CDS cross-table",
        "description": "Multi-CDS function with cross-table access",
        "priority": "P0",
        "backends": ["python", "node"],
        "layer": "integration",
        "proof": "test_dsl_customjs.py::test_compileVarName",
    },

    # =========================================================================
    # Histogram Operations
    # =========================================================================
    "HIST.histogram_1d": {
        "name": "1D histogram",
        "description": "1D histogram binning (HistogramCDS)",
        "priority": "P1",
        "backends": ["browser"],
        "layer": "integration",
        "proof": None,
        "planned": True,
    },
    "HIST.histogram_nd": {
        "name": "N-D histogram",
        "description": "N-dimensional histogram (HistoNdCDS)",
        "priority": "P2",
        "backends": ["browser"],
        "layer": "integration",
        "proof": None,
        "planned": True,
    },

    # =========================================================================
    # Alias Operations
    # =========================================================================
    "ALIAS.cdsalias": {
        "name": "CDSAlias",
        "description": "CDSAlias column aliasing",
        "priority": "P1",
        "backends": ["browser"],
        "layer": "integration",
        "proof": "test_Alias.py",
        "planned": True,
    },
}


# =============================================================================
# KNOWN LIMITATIONS
# =============================================================================

KNOWN_LIMITATIONS = {
    # Add limitations as they are discovered
    # Format follows RDataFrameDSL pattern:
    # "L1": {
    #     "name": "Limitation name",
    #     "status": "⚠️ Partial",
    #     "description": "Description of the limitation",
    #     "workaround": "How to work around it",
    #     "resolution": None,  # or description of fix
    # },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_feature_ids():
    """Return sorted list of all feature IDs."""
    return sorted(FEATURE_TAXONOMY.keys())


def get_features_by_category(prefix: str) -> dict:
    """Get all features with a given prefix (e.g., 'DSL', 'ENC')."""
    return {k: v for k, v in FEATURE_TAXONOMY.items() if k.startswith(prefix + ".")}


def get_features_by_priority(priority: str) -> dict:
    """Get all features with a given priority."""
    return {k: v for k, v in FEATURE_TAXONOMY.items() if v.get("priority") == priority}


def get_all_categories() -> list:
    """Get list of all category prefixes."""
    categories = set()
    for feature_id in FEATURE_TAXONOMY:
        if "." in feature_id:
            categories.add(feature_id.split(".")[0])
    return sorted(categories)


def validate_feature_id(feature_id: str) -> bool:
    """Check if feature ID is valid (including aliases)."""
    if feature_id in FEATURE_TAXONOMY:
        return True
    if feature_id in FEATURE_ALIASES:
        return True
    return False


def resolve_feature_id(feature_id: str) -> str:
    """Resolve alias to canonical feature ID."""
    return FEATURE_ALIASES.get(feature_id, feature_id)


def get_planned_features() -> dict:
    """Get all features marked as planned."""
    return {k: v for k, v in FEATURE_TAXONOMY.items() if v.get("planned")}


def get_implemented_features() -> dict:
    """Get all features not marked as planned."""
    return {k: v for k, v in FEATURE_TAXONOMY.items() if not v.get("planned")}
