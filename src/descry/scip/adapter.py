"""SCIP language adapter Protocol and registry.

Each supported SCIP indexer is represented by a `LanguageAdapter` that knows:
- how to discover projects of its language under a root
- how to build the subprocess command that invokes its indexer
- how to parse its own SCIP symbol descriptors

Adapters register themselves at import time via `register()`. All registry
consumers (support.py, cache.py, parser.py) iterate `ADAPTERS` rather than
hardcode per-language branches, so adding a new SCIP language is one new
adapter module plus an import in `adapters/__init__.py`.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredProject:
    """A concrete project discovered by an adapter's `discover()` method."""

    name: str
    """Project identifier, also used as the `.scip` filename stem."""

    root: Path
    """Directory to run the indexer in (subprocess cwd)."""

    language: str
    """Owning adapter's `name` (e.g. ``"rust"``, ``"typescript"``)."""


@dataclass(frozen=True)
class CommandSpec:
    """Subprocess invocation spec returned by `build_command()`."""

    argv: list[str]
    """Fully-validated command line — no shell expansion."""

    cwd: Path
    """Working directory for the subprocess."""

    env_extras: dict[str, str] = field(default_factory=dict)
    """Additional env vars merged after `safe_env()`."""

    output_mode: str = "direct"
    """"direct": indexer writes to the requested out_path.
    "rename": indexer writes to its default location (``index.scip`` in cwd)
    and the shared runner moves the file into place after success. Used for
    indexers that don't expose an --output flag (scip-go contingency)."""


@dataclass(frozen=True)
class AdapterConfig:
    """Per-language config forwarded into `build_command()`."""

    toolchain: str | None = None
    """e.g. ``"1.92.0"`` for Rust rustup toolchain, JDK version for Java."""

    extra_args: tuple[str, ...] = ()
    """Validated positional/flag extras appended to argv."""

    options: dict[str, str] = field(default_factory=dict)
    """Adapter-specific knobs (e.g. ``build_tool`` for Java, ``module_name`` for Go)."""


@runtime_checkable
class LanguageAdapter(Protocol):
    """Contract each SCIP language integration implements."""

    name: str
    """Language identifier used across descry (``"rust"``, ``"go"``, ...)."""

    scheme: str
    """Single-token SCIP scheme this adapter owns (``"rust-analyzer"``,
    ``"scip-typescript"``, ``"scip-python"``, ``"scip-java"``, ``"scip-go"``)."""

    binary: str
    """Command name probed via ``shutil.which()``."""

    extensions: tuple[str, ...]
    """Source file extensions this adapter owns (e.g. ``(".java", ".kt", ".scala")``)."""

    def discover(
        self, root: Path, excluded_dirs: set[str]
    ) -> list[DiscoveredProject]: ...

    def build_command(
        self,
        project: DiscoveredProject,
        out_path: Path,
        config: AdapterConfig,
    ) -> CommandSpec: ...

    def parse_descriptors(self, raw: str) -> list[str]:
        """Parse raw SCIP descriptor string into name components.

        Example: ``"state/AppState#new()."`` -> ``["AppState", "new"]``.
        """
        ...


ADAPTERS: dict[str, LanguageAdapter] = {}
"""Module-level registry of installed adapters, keyed by ``adapter.name``."""

_binary_cache: dict[str, dict] = {}
"""Cache of binary probe results, keyed by adapter.name."""


def register(adapter: LanguageAdapter) -> None:
    """Register an adapter in the global registry. Idempotent on `name`."""
    ADAPTERS[adapter.name] = adapter


def reset_registry_state() -> None:
    """Clear cached binary probe results (for tests)."""
    _binary_cache.clear()


def _probe_binary(adapter: LanguageAdapter) -> dict:
    """Probe an adapter's binary and cache the result."""
    if adapter.name in _binary_cache:
        return _binary_cache[adapter.name]

    import subprocess

    from descry._env import safe_env

    result = {"available": False, "path": None, "version": None}
    path = shutil.which(adapter.binary)
    if path:
        try:
            proc = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=safe_env(),
            )
            if proc.returncode == 0:
                result["available"] = True
                result["path"] = path
                result["version"] = proc.stdout.strip() or proc.stderr.strip()
                logger.debug(f"SCIP: {adapter.binary} available at {path}")
            else:
                logger.debug(f"SCIP: {adapter.binary} found but failed to run")
        except subprocess.TimeoutExpired:
            logger.debug(f"SCIP: {adapter.binary} version check timed out")
        except (OSError, FileNotFoundError) as e:
            logger.debug(f"SCIP: {adapter.binary} check failed: {e}")
    else:
        logger.debug(f"SCIP: {adapter.binary} not found in PATH")

    _binary_cache[adapter.name] = result
    return result


def indexer_status(lang_name: str) -> dict:
    """Return the cached binary-probe status dict for `lang_name`."""
    adapter = ADAPTERS.get(lang_name)
    if adapter is None:
        return {"available": False, "path": None, "version": None}
    return _probe_binary(adapter)


def indexer_available(lang_name: str) -> bool:
    """True if the adapter for `lang_name` is registered and its binary is on PATH."""
    if os.environ.get("DESCRY_NO_SCIP"):
        return False
    adapter = ADAPTERS.get(lang_name)
    if adapter is None:
        return False
    return bool(_probe_binary(adapter).get("available"))


def available_adapters() -> list[LanguageAdapter]:
    """Return adapters whose binaries are present (respecting DESCRY_NO_SCIP)."""
    if os.environ.get("DESCRY_NO_SCIP"):
        return []
    return [a for a in ADAPTERS.values() if _probe_binary(a).get("available")]


def adapter_for_scheme(scheme: str) -> LanguageAdapter | None:
    """Return the adapter whose SCIP scheme token matches `scheme`."""
    for adapter in ADAPTERS.values():
        if adapter.scheme == scheme:
            return adapter
    return None


def adapter_for_extension(ext: str) -> LanguageAdapter | None:
    """Return the adapter that owns source file extension `ext`.

    Used by `ScipIndex.resolve` to pick the right stats bucket based on the
    source file being resolved.
    """
    for adapter in ADAPTERS.values():
        if ext in adapter.extensions:
            return adapter
    return None
