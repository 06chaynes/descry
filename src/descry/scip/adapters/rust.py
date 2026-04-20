"""Rust adapter backed by rust-analyzer's built-in SCIP emitter."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from descry.scip.adapter import (
    AdapterConfig,
    CommandSpec,
    DiscoveredProject,
    register,
)

logger = logging.getLogger(__name__)


_RUST_DESCRIPTOR_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)(\([^)]*\)|[#./\[\]])?")


def _parse_workspace_members(root_cargo: Path) -> list[Path]:
    """Parse ``[workspace] members = [...]`` entries from ``root_cargo``.

    Glob patterns like ``"crates/*"`` are expanded against the workspace
    root. Returns absolute paths; invalid/missing entries are silently
    dropped. Uses ``tomllib`` (3.11+); on parse error returns an empty
    list so discovery falls through to the single-level glob fallback.
    """
    try:
        import tomllib

        with open(root_cargo, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.debug(f"Cargo.toml parse failed at {root_cargo}: {e}")
        return []

    workspace = data.get("workspace") or {}
    raw_members = workspace.get("members") or []
    if not isinstance(raw_members, list):
        return []

    root = root_cargo.parent
    result: list[Path] = []
    for member in raw_members:
        if not isinstance(member, str) or not member:
            continue
        if "*" in member or "?" in member or "[" in member:
            try:
                result.extend(p for p in root.glob(member) if p.is_dir())
            except (OSError, ValueError):
                pass
        else:
            candidate = (root / member).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                continue
            if candidate.is_dir():
                result.append(candidate)
    return result


class RustAdapter:
    """rust-analyzer scip — Rust symbols via Cargo workspace analysis."""

    name = "rust"
    scheme = "rust-analyzer"
    binary = "rust-analyzer"
    extensions = (".rs",)

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per workspace-level Rust crate.

        A crate is any directory containing ``Cargo.toml`` plus a ``src/``
        directory. The workspace root is reported as each project's
        ``root`` (rust-analyzer shares analysis cache across crates in one
        workspace), and the crate's relative path from root is the
        project's ``name`` (e.g. ``"crates/cargo-util"``).

        Discovery:

        1. Parses ``[workspace] members = [...]`` from root ``Cargo.toml``
           (canonical Cargo workspace layout — used by cargo, coreutils,
           most serious multi-crate repos). Glob patterns like
           ``"crates/*"`` are expanded.
        2. Falls back to top-level ``*/Cargo.toml`` globbing for simple
           workspaces that don't declare ``[workspace] members`` (e.g.,
           tokio — each subdir is an independent crate).
        3. If neither matches but root has ``src/``, index root as a
           single-crate project (small libraries).
        """
        root_cargo = root / "Cargo.toml"
        if not root_cargo.exists():
            return []

        projects: list[DiscoveredProject] = []
        seen_paths: set[Path] = set()

        for member_dir in _parse_workspace_members(root_cargo):
            if member_dir.name.startswith("."):
                continue
            if any(
                part in excluded_dirs for part in member_dir.relative_to(root).parts
            ):
                continue
            if not (member_dir / "Cargo.toml").exists():
                continue
            if not (member_dir / "src").exists():
                continue
            if member_dir in seen_paths:
                continue
            seen_paths.add(member_dir)
            rel = member_dir.relative_to(root)
            projects.append(
                DiscoveredProject(
                    name=str(rel),
                    root=root,
                    language=self.name,
                )
            )

        if not projects:
            for cargo_toml in root.glob("*/Cargo.toml"):
                crate_dir = cargo_toml.parent
                if crate_dir.name.startswith("."):
                    continue
                if crate_dir.name in excluded_dirs:
                    continue
                if not (crate_dir / "src").exists():
                    continue
                if crate_dir in seen_paths:
                    continue
                seen_paths.add(crate_dir)
                projects.append(
                    DiscoveredProject(
                        name=crate_dir.name,
                        root=root,
                        language=self.name,
                    )
                )

        if not projects and (root / "src").exists():
            projects.append(
                DiscoveredProject(
                    name=root.name,
                    root=root,
                    language=self.name,
                )
            )

        projects.sort(key=lambda p: p.name)
        return projects

    def build_command(
        self,
        project: DiscoveredProject,
        out_path: Path,
        config: AdapterConfig,
    ) -> CommandSpec:
        """Build the `rust-analyzer scip` command.

        Runs from the workspace root (`project.root`) with the crate path
        passed as the positional argument, so rust-analyzer can share its
        workspace analysis cache across invocations.
        """
        crate_path = project.root / project.name
        argv: list[str] = []
        if config.toolchain:
            argv.extend(["rustup", "run", config.toolchain])
        argv.extend(
            [
                self.binary,
                "scip",
                str(crate_path),
                "--output",
                str(out_path),
            ]
        )
        argv.extend(config.extra_args)
        return CommandSpec(argv=argv, cwd=project.root)

    def parse_descriptors(self, raw: str) -> list[str]:
        """Parse Rust-flavored SCIP descriptors into name components.

        Suffix meanings:
        - ``/`` — namespace/module (skipped; already in file path)
        - ``#`` — type (struct/enum/trait) — included
        - ``.`` — term (constant/static) — included
        - ``()`` — method/function — included
        - ``[]`` / ``[impl]`` — type parameters / impl blocks — the ``impl``
          keyword is skipped but the bracketed type name is included

        Examples:
            ``state/AppState#new().`` -> ``["AppState", "new"]``
            ``database/migrations/run_migrations().`` -> ``["run_migrations"]``
            ``impl#[AppState]new().`` -> ``["AppState", "new"]``
        """
        names: list[str] = []
        for match in _RUST_DESCRIPTOR_PATTERN.finditer(raw):
            name = match.group(1)
            suffix = match.group(2) or ""
            if not name:
                continue
            if suffix == "/":
                continue
            if name == "impl":
                continue
            if name == "tests" and suffix == "/":
                continue
            names.append(name)
        return names


register(RustAdapter())
