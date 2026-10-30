"""SCIP availability and status."""

from __future__ import annotations

import descry.scip.adapters  # noqa: F401 — side-effect: registers built-in adapters
from descry.scip.adapter import (
    ADAPTERS,
    available_adapters,
    indexer_status,
    scip_disabled_by_env,
)


def scip_available() -> bool:
    """True if at least one SCIP indexer is available (respects DESCRY_NO_SCIP)."""
    if scip_disabled_by_env():
        return False
    return len(available_adapters()) > 0


def get_scip_status() -> dict:
    """Return SCIP status for diagnostics.

    The `indexers` dict enumerates every registered adapter, keyed by the
    indexer's binary name (e.g. "rust-analyzer", "scip-typescript").
    """
    indexers: dict[str, dict] = {}
    for adapter in ADAPTERS.values():
        info = indexer_status(adapter.name)
        indexers[adapter.binary] = {
            "available": bool(info.get("available")),
            "path": info.get("path"),
            "version": info.get("version"),
        }

    return {
        "available": scip_available(),
        "disabled_by_env": scip_disabled_by_env(),
        "indexers": indexers,
    }
