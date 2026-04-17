"""SCIP support with auto-discovery for multiple languages.

Provides functions to detect SCIP indexer availability and manage
SCIP-based symbol resolution for improved descry call resolution.

SCIP (Source Code Index Protocol) is a format from Sourcegraph that
provides type-aware symbol information from language servers.

Supported indexers:
- rust-analyzer: Rust
- scip-typescript: TypeScript, JavaScript
- scip-python: Python
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import TYPE_CHECKING

from descry._env import safe_env

if TYPE_CHECKING:
    from typing import Dict

logger = logging.getLogger(__name__)

# Global state (lazy-initialized)
_indexer_cache: Dict[str, dict] = {}


def _check_indexer(name: str, command: str, version_flag: str = "--version") -> dict:
    """Check if an indexer is available.

    Args:
        name: Indexer name for logging
        command: Command to check for
        version_flag: Flag to get version info

    Returns:
        Dict with available, path, version keys
    """
    result = {"available": False, "path": None, "version": None}

    path = shutil.which(command)
    if path:
        try:
            proc = subprocess.run(
                [path, version_flag],
                capture_output=True,
                text=True,
                timeout=10,
                env=safe_env(),
            )
            if proc.returncode == 0:
                result["available"] = True
                result["path"] = path
                result["version"] = proc.stdout.strip() or proc.stderr.strip()
                logger.debug(f"SCIP: {name} available at {path}")
            else:
                logger.debug(f"SCIP: {name} found but failed to run")
        except subprocess.TimeoutExpired:
            logger.debug(f"SCIP: {name} version check timed out")
        except (OSError, FileNotFoundError) as e:
            logger.debug(f"SCIP: {name} check failed: {e}")
    else:
        logger.debug(f"SCIP: {name} not found in PATH")

    return result


def rust_analyzer_available() -> bool:
    """Check if rust-analyzer is available for Rust SCIP generation."""
    if os.environ.get("DESCRY_NO_SCIP"):
        return False

    if "rust-analyzer" not in _indexer_cache:
        _indexer_cache["rust-analyzer"] = _check_indexer(
            "rust-analyzer", "rust-analyzer", "--version"
        )
    return _indexer_cache["rust-analyzer"]["available"]


def scip_typescript_available() -> bool:
    """Check if scip-typescript is available for TypeScript/JavaScript SCIP generation."""
    if os.environ.get("DESCRY_NO_SCIP"):
        return False

    if "scip-typescript" not in _indexer_cache:
        _indexer_cache["scip-typescript"] = _check_indexer(
            "scip-typescript", "scip-typescript", "--version"
        )
    return _indexer_cache["scip-typescript"]["available"]


def scip_python_available() -> bool:
    """Check if scip-python is available for Python SCIP generation.

    scip-python is a Sourcegraph-maintained indexer distributed via npm
    (@sourcegraph/scip-python) that installs a `scip-python` binary.
    """
    if os.environ.get("DESCRY_NO_SCIP"):
        return False

    if "scip-python" not in _indexer_cache:
        _indexer_cache["scip-python"] = _check_indexer(
            "scip-python", "scip-python", "--version"
        )
    return _indexer_cache["scip-python"]["available"]


def scip_available() -> bool:
    """Check if any SCIP indexer is available (for backwards compatibility).

    Returns:
        True if at least one SCIP indexer is available.
    """
    if os.environ.get("DESCRY_NO_SCIP"):
        return False

    return (
        rust_analyzer_available()
        or scip_typescript_available()
        or scip_python_available()
    )


def get_scip_status() -> dict:
    """Return SCIP status for diagnostics.

    Returns:
        Dictionary with indexer availability information.
    """
    # Force availability checks
    _ = rust_analyzer_available()
    _ = scip_typescript_available()
    _ = scip_python_available()

    rust_info = _indexer_cache.get("rust-analyzer", {})
    ts_info = _indexer_cache.get("scip-typescript", {})
    py_info = _indexer_cache.get("scip-python", {})

    return {
        "available": scip_available(),
        "disabled_by_env": bool(os.environ.get("DESCRY_NO_SCIP")),
        "indexers": {
            "rust-analyzer": {
                "available": rust_info.get("available", False),
                "path": rust_info.get("path"),
                "version": rust_info.get("version"),
            },
            "scip-typescript": {
                "available": ts_info.get("available", False),
                "path": ts_info.get("path"),
                "version": ts_info.get("version"),
            },
            "scip-python": {
                "available": py_info.get("available", False),
                "path": py_info.get("path"),
                "version": py_info.get("version"),
            },
        },
        # Backwards compatibility
        "rust_analyzer_path": rust_info.get("path"),
        "rust_analyzer_version": rust_info.get("version"),
    }


def reset_scip_state():
    """Reset cached SCIP state. Useful for testing."""
    global _indexer_cache
    _indexer_cache = {}
