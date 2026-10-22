"""Rust adapter backed by rust-analyzer's built-in SCIP emitter."""

from __future__ import annotations

import logging
import re
import tomllib
from pathlib import Path

from descry.scip.adapter import (
    AdapterConfig,
    CommandSpec,
    DiscoveredProject,
    project_dir,
    register,
)

logger = logging.getLogger(__name__)


# `impl#[Type][Trait]`: the trait an `impl` block is for, without its
# arguments. A name that is no identifier is written between backticks.
_RUST_TRAIT_IMPL = re.compile(r"impl#\[(?:`[^`]*`|[^\]`])*\]\[`?([^\]`<]+)")
# A symbol's package and its descriptors
_RUST_SYMBOL = re.compile(r"\S+ \S+ (\S+) \S+ (.+)", re.DOTALL)
_RUST_DESCRIPTOR_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)(\([^)]*\)|[#./\[\]])?")


def crate_dir(workspace_root: Path, name: str) -> Path:
    """Directory of the crate that ``discover()`` reported as ``name``."""
    return project_dir(workspace_root, name, "Cargo.toml", "src")


def _load_manifest(cargo_toml: Path) -> dict:
    """The parsed ``cargo_toml``, or an empty table where it cannot be read."""
    try:
        with open(cargo_toml, "rb") as f:
            return tomllib.load(f)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        logger.debug(f"Cargo.toml parse failed at {cargo_toml}: {e}")
        return {}


def _parse_workspace_members(root_cargo: Path) -> list[Path]:
    """Parse ``[workspace] members = [...]`` entries from ``root_cargo``.

    Glob patterns like ``"crates/*"`` are expanded against the workspace
    root. Returns absolute paths; invalid/missing entries and entries
    outside the root are silently dropped. A manifest that cannot be
    parsed gives an empty list, so discovery falls through to the
    single-level glob fallback.
    """
    data = _load_manifest(root_cargo)
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
                result.extend(
                    p
                    for p in root.glob(member)
                    if p.is_dir() and p.resolve().is_relative_to(root.resolve())
                )
            except (OSError, ValueError, NotImplementedError):
                # pathlib raises NotImplementedError for an absolute pattern.
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
        3. The root is a crate of its own, named after the root directory,
           or ``.`` where a member is named so, where its ``Cargo.toml``
           declares a ``[package]`` and it has ``src/``, or where nothing
           else was found and it has ``src/``.
           ``rust-analyzer scip`` indexes only the crate at the path it is
           given, so a root package beside its members needs its own run.
        """
        root_cargo = root / "Cargo.toml"
        if not root_cargo.is_file():
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
            if not (member_dir / "Cargo.toml").is_file():
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
                if crate_dir.name.startswith(".") or not cargo_toml.is_file():
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

        root_package = "package" in _load_manifest(root_cargo)
        if (root_package or not projects) and (root / "src").exists():
            # Named after the root directory, or `.` where a member already
            # holds that name; `crate_dir` resolves either to the root.
            taken = any(p.name == root.name for p in projects)
            projects.append(
                DiscoveredProject(
                    name="." if taken else root.name,
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
        crate_path = crate_dir(project.root, project.name)
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

    def _packages(self, root: Path | None) -> set[str]:
        """Cargo package names of the crates ``discover()`` finds in ``root``."""
        names = set()
        for project in self.discover(root, set()) if root else ():
            manifest = crate_dir(root, project.name) / "Cargo.toml"
            if not manifest.is_file():
                continue
            try:
                with open(manifest, "rb") as f:
                    names.add(tomllib.load(f)["package"]["name"])
            except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError):
                pass
        return names

    def leaving(self, defined, root):
        """A test of whether a call of a symbol no index defines cannot
        reach the project: it names a function, a method of a concrete type
        (`impl#[String]len().`), or a method of a trait (`Display#fmt().`)
        that no `impl` among ``defined``, the symbols the indexes define, is
        for. A symbol of a crate of ``root`` that has no index does not
        leave; the `impl`s of such a crate are not seen."""
        found = map(_RUST_SYMBOL.fullmatch, defined)
        known = [match.groups() for match in found if match]
        found = (_RUST_TRAIT_IMPL.search(descriptors) for _, descriptors in known)
        implemented = {impl[1].rpartition("::")[2] for impl in found if impl}
        unindexed = self._packages(root) - {package for package, _ in known}

        def leaves(symbol: str) -> bool:
            match = _RUST_SYMBOL.fullmatch(symbol)
            if not match or match[1] in unindexed or not match[2].endswith(")."):
                return False
            descriptors = match[2]
            if "impl#[" in descriptors or "#" not in descriptors:
                return True
            trait = descriptors.rpartition("#")[0].rpartition("/")[2]
            return trait not in implemented

        return leaves

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
