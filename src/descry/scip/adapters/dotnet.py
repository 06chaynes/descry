""".NET adapter backed by scip-dotnet (Sourcegraph).

scip-dotnet is a .NET tool built on Roslyn that indexes C# and VB
projects via MSBuild. Install:

    dotnet tool install --global scip-dotnet

Invocation: ``scip-dotnet index`` at the project or solution root.
The tool reads ``.sln`` / ``.csproj`` / ``.vbproj`` and can do its own
``dotnet restore``. Supports ``--output`` so we use
``output_mode="direct"``.

scip-dotnet v0.2.13 targets ``net9.0`` — on a system with a newer SDK
(.NET 10+) the adapter sets ``DOTNET_ROLL_FORWARD=LatestMajor`` so the
tool runs against the installed runtime. ``DOTNET_ROOT`` is surfaced
too so ``safe_env()`` filtering doesn't drop it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from descry.scip.adapter import (
    AdapterConfig,
    CommandSpec,
    DiscoveredProject,
    register,
)
from descry.scip.adapters.typescript import parse_backtick_descriptors

logger = logging.getLogger(__name__)


_DOTNET_MARKERS = (".sln", ".csproj", ".vbproj", ".fsproj")


def _find_solution_or_project(pkg_dir: Path) -> Path | None:
    """Prefer .sln at the directory; else first .csproj / .vbproj."""
    for name in pkg_dir.glob("*.sln"):
        return name
    for name in pkg_dir.glob("*.csproj"):
        return name
    for name in pkg_dir.glob("*.vbproj"):
        return name
    for name in pkg_dir.glob("*.fsproj"):
        return name
    return None


def _has_dotnet_sources(pkg_dir: Path) -> bool:
    for ext in ("*.cs", "*.vb"):
        if any(pkg_dir.rglob(ext)):
            return True
    return False


class DotnetAdapter:
    """scip-dotnet — C#/VB symbols via Roslyn."""

    name = "dotnet"
    scheme = "scip-dotnet"
    binary = "scip-dotnet"
    extensions = (".cs", ".vb")

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Discover .NET projects.

        Order of preference:
        1. Solution-level project at root (`root/*.sln`).
        2. Top-level subdirectories containing a .sln/.csproj/.vbproj.
        3. Root as a single project if it carries a .csproj/.vbproj.
        """
        seen: set[str] = set()
        projects: list[DiscoveredProject] = []

        # 1. Root-level solution — treat the whole repo as one project.
        root_sln = _find_solution_or_project(root)
        if root_sln and _has_dotnet_sources(root):
            projects.append(
                DiscoveredProject(
                    name=root.name,
                    root=root,
                    language=self.name,
                )
            )
            return projects

        # 2. Per-subdir projects.
        for marker in _DOTNET_MARKERS:
            for match in root.glob(f"*/*{marker}"):
                pkg_dir = match.parent
                if pkg_dir.name.startswith("."):
                    continue
                if pkg_dir.name in excluded_dirs:
                    continue
                if pkg_dir.name in seen:
                    continue
                if not _has_dotnet_sources(pkg_dir):
                    continue
                seen.add(pkg_dir.name)
                projects.append(
                    DiscoveredProject(
                        name=pkg_dir.name,
                        root=pkg_dir,
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
        """Build the ``scip-dotnet index --output <out>`` command.

        Always sets ``DOTNET_ROLL_FORWARD=LatestMajor`` so the tool
        runs against whatever SDK is installed (scip-dotnet 0.2.13
        targets net9.0 but descry users may have net10+ installed).
        """
        argv: list[str] = [self.binary, "index", "--output", str(out_path)]
        argv.extend(config.extra_args)

        env_extras: dict[str, str] = {"DOTNET_ROLL_FORWARD": "LatestMajor"}
        dotnet_root = os.environ.get("DOTNET_ROOT")
        if dotnet_root:
            env_extras["DOTNET_ROOT"] = dotnet_root

        return CommandSpec(
            argv=argv,
            cwd=project.root,
            env_extras=env_extras,
            output_mode="direct",
        )

    def parse_descriptors(self, raw: str) -> list[str]:
        r"""scip-dotnet emits backtick-wrapped descriptors.

        Example from the README:
        ``scip-dotnet nuget . . Main/Expressions#TargetType#\`.ctor\`()``
        """
        return parse_backtick_descriptors(raw)


register(DotnetAdapter())
