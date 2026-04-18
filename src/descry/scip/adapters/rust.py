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


class RustAdapter:
    """rust-analyzer scip — Rust symbols via Cargo workspace analysis."""

    name = "rust"
    scheme = "rust-analyzer"
    binary = "rust-analyzer"
    extensions = (".rs",)

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per workspace-level Rust crate.

        A crate is a top-level subdirectory containing `Cargo.toml` plus a
        `src/` directory. The workspace root is reported as each project's
        `root` (rust-analyzer shares analysis cache across crates under one
        workspace), and the crate directory name is the project's `name`.
        """
        root_cargo = root / "Cargo.toml"
        if not root_cargo.exists():
            return []

        projects: list[DiscoveredProject] = []
        for cargo_toml in root.glob("*/Cargo.toml"):
            crate_dir = cargo_toml.parent
            if crate_dir.name.startswith("."):
                continue
            if crate_dir.name in excluded_dirs:
                continue
            if not (crate_dir / "src").exists():
                continue
            projects.append(
                DiscoveredProject(
                    name=crate_dir.name,
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
