"""SCIP support: public API over the adapter registry.

This module is the stable surface for "is SCIP working?" style questions,
backed by the pluggable `LanguageAdapter` registry in `adapter.py`. Callers
outside the SCIP subpackage (handlers.py, generate.py, web/server.py,
tests) keep using `scip_available()`, `get_scip_status()`, and
`reset_scip_state()` — the three symbols imported via `_try_import_scip`.

Importing this module also imports `descry.scip.adapters`, which registers
the built-in adapters (rust-analyzer, scip-typescript, scip-python) at load
time.
"""

from __future__ import annotations

import logging
import os

import descry.scip.adapters  # noqa: F401 — side-effect: registers built-in adapters
from descry.scip.adapter import (
    ADAPTERS,
    available_adapters,
    indexer_available,
    indexer_status,
    reset_registry_state,
)

logger = logging.getLogger(__name__)


def rust_analyzer_available() -> bool:
    """Back-compat shim: True iff the rust-analyzer adapter's binary is present."""
    return indexer_available("rust")


def scip_typescript_available() -> bool:
    """Back-compat shim: True iff the scip-typescript adapter's binary is present."""
    return indexer_available("typescript")


def scip_python_available() -> bool:
    """Back-compat shim: True iff the scip-python adapter's binary is present.

    scip-python is distributed via npm (`@sourcegraph/scip-python`) and
    installs a `scip-python` binary on PATH.
    """
    return indexer_available("python")


def scip_available() -> bool:
    """True if at least one SCIP indexer is available (respects DESCRY_NO_SCIP)."""
    if os.environ.get("DESCRY_NO_SCIP"):
        return False
    return len(available_adapters()) > 0


def get_scip_status() -> dict:
    """Return SCIP status for diagnostics.

    The `indexers` dict enumerates every registered adapter, keyed by the
    indexer's binary name (e.g. "rust-analyzer", "scip-typescript"). Legacy
    top-level keys `rust_analyzer_path` and `rust_analyzer_version` are
    preserved for back-compat with callers that read them directly.
    """
    indexers: dict[str, dict] = {}
    for adapter in ADAPTERS.values():
        info = indexer_status(adapter.name)
        indexers[adapter.binary] = {
            "available": bool(info.get("available")),
            "path": info.get("path"),
            "version": info.get("version"),
        }

    rust_info = indexer_status("rust") if "rust" in ADAPTERS else {}

    return {
        "available": scip_available(),
        "disabled_by_env": bool(os.environ.get("DESCRY_NO_SCIP")),
        "indexers": indexers,
        # Legacy back-compat fields:
        "rust_analyzer_path": rust_info.get("path"),
        "rust_analyzer_version": rust_info.get("version"),
    }


def reset_scip_state() -> None:
    """Reset cached indexer-binary probe results (for tests)."""
    reset_registry_state()
