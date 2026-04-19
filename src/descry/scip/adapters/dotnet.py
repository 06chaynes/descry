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


_DOTNET_MARKERS = (".sln", ".slnx", ".csproj", ".vbproj", ".fsproj")


def _find_solution_or_project(pkg_dir: Path) -> Path | None:
    """Prefer a solution file, then a project file.

    Checks ``.sln`` first (legacy MSBuild), then ``.slnx`` (modern XML
    solution format used by dotnet/aspnetcore and other newer repos),
    then per-language project files.
    """
    for ext in (".sln", ".slnx", ".csproj", ".vbproj", ".fsproj"):
        for name in pkg_dir.glob(f"*{ext}"):
            return name
    return None


def _has_dotnet_sources(pkg_dir: Path) -> bool:
    for ext in ("*.cs", "*.vb"):
        if any(pkg_dir.rglob(ext)):
            return True
    return False


def _find_net9_dotnet_root() -> str | None:
    """Return the path of a DOTNET_ROOT that has a net9.0 runtime, or None.

    scip-dotnet 0.2.13 targets net9.0 and won't roll forward to net10+.
    When the system default SDK is net10 we need to point the tool at a
    side-installed net9 runtime. Check the conventional locations in
    priority order — Homebrew's ``dotnet@9`` keg-only formula, the
    official ``dotnet-install.sh`` path, then a user-scoped install.
    """
    candidates = [
        "/opt/homebrew/opt/dotnet@9/libexec",
        "/usr/local/opt/dotnet@9/libexec",
        str(Path.home() / ".dotnet"),
        str(Path.home() / ".dotnet-9"),
    ]
    for candidate in candidates:
        shared = Path(candidate) / "shared" / "Microsoft.NETCore.App"
        if shared.is_dir() and any(
            p.name.startswith("9.") for p in shared.iterdir() if p.is_dir()
        ):
            return candidate
    return None


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

        scip-dotnet 0.2.13 targets ``net9.0``. On a system with only
        .NET 10+ installed the tool crashes at startup (exit 131)
        because ``LatestMajor`` roll-forward doesn't cover the net9 →
        net10 major-version gap. To make the adapter self-healing on
        systems with net10+ as the default SDK, we look for a
        side-installed net9 runtime in common locations (Homebrew's
        ``dotnet@9``, the official dotnet-install.sh path, or an
        explicit ``DOTNET_ROOT``) and point scip-dotnet at it when
        found.
        """
        argv: list[str] = [self.binary, "index", "--output", str(out_path)]
        argv.extend(config.extra_args)

        env_extras: dict[str, str] = {"DOTNET_ROLL_FORWARD": "LatestMajor"}

        dotnet_root = os.environ.get("DOTNET_ROOT") or _find_net9_dotnet_root()
        if dotnet_root:
            env_extras["DOTNET_ROOT"] = dotnet_root
            # Prepend its bin dir to PATH so the `dotnet` binary used by
            # scip-dotnet resolves to the net9-capable SDK.
            dotnet_bin = Path(dotnet_root) / "bin"
            if dotnet_bin.exists():
                existing_path = os.environ.get("PATH", "")
                env_extras["PATH"] = f"{dotnet_bin}:{existing_path}"

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
