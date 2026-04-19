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


def _list_installed_sdks(dotnet_root: str) -> list[str]:
    """Return SDK versions discovered under ``dotnet_root/sdk``."""
    sdk_dir = Path(dotnet_root) / "sdk"
    if not sdk_dir.is_dir():
        return []
    return [p.name for p in sdk_dir.iterdir() if p.is_dir()]


def _global_json_satisfiable(project_root: Path, dotnet_root: str | None) -> str | None:
    """Check whether ``project_root/global.json`` pins an SDK that the
    resolved ``dotnet_root`` actually ships.

    Returns a human-readable reason string if the pin is **not**
    satisfiable (so the caller can skip scip-dotnet cleanly), or None
    if either there's no global.json, no pin, or the pin resolves.

    scip-dotnet runs ``dotnet restore`` under the hood; when the
    pinned SDK is missing and ``rollForward`` disallows moves, every
    project fails to restore and the scip index is useless.
    """
    gj = project_root / "global.json"
    if not gj.exists():
        return None
    try:
        import json as _json

        data = _json.loads(gj.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, ValueError):
        return None
    sdk = data.get("sdk") if isinstance(data, dict) else None
    if not isinstance(sdk, dict):
        return None
    pin = sdk.get("version")
    if not isinstance(pin, str) or not pin:
        return None

    # rollForward "latestFeature" / "latestMajor" / "latestMinor" /
    # "latestPatch" all allow graceful fallback; only disable≈"disable"
    # or default behavior with a missing SDK is a hard block.
    roll_forward = (sdk.get("rollForward") or "").lower()
    allow_prerelease = bool(sdk.get("allowPrerelease", False))

    installed = _list_installed_sdks(dotnet_root) if dotnet_root else []
    if not installed:
        # Nothing to compare against; let scip-dotnet surface the real
        # error at runtime.
        return None

    if pin in installed:
        return None

    # Same major.minor band?
    pin_prefix = ".".join(pin.split(".")[:2])
    matching_band = [v for v in installed if v.startswith(pin_prefix + ".")]
    if matching_band and roll_forward in (
        "latestfeature",
        "latestpatch",
        "latestminor",
        "latestmajor",
    ):
        return None
    # Any version with allowPrerelease + latestMajor
    if roll_forward == "latestmajor" and allow_prerelease and installed:
        return None

    return (
        f"global.json pins SDK {pin!r} (rollForward={roll_forward or 'default'}, "
        f"allowPrerelease={allow_prerelease}); installed SDKs at {dotnet_root} "
        f"are {sorted(installed)!r}. scip-dotnet's 'dotnet restore' will fail "
        f"for every project under this pin."
    )


def _find_net9_dotnet_root() -> str | None:
    """Return the path of a DOTNET_ROOT that has a net9.0 runtime, or None.

    scip-dotnet 0.2.13 targets net9.0 and won't roll forward to net10+.
    When the system default SDK is net10 we need to point the tool at a
    side-installed net9 runtime. Check the conventional locations in
    priority order — Homebrew's ``dotnet@9`` keg-only formula, the
    official ``dotnet-install.sh`` path, then a user-scoped install.
    """
    # ``~/.dotnet`` comes first: when users have both net9 and net10
    # side-by-side there (the dotnet-install.sh default layout), scip-
    # dotnet gets access to BOTH the net9 runtime it needs AND the
    # net10+ SDK modern projects pin via ``global.json``. The keg-only
    # Homebrew ``dotnet@9`` path is a fallback — it carries only net9,
    # so projects with ``global.json`` pinning net10 fail there with
    # "compatible SDK was not found".
    candidates = [
        str(Path.home() / ".dotnet"),
        str(Path.home() / ".dotnet-9"),
        "/opt/homebrew/opt/dotnet@9/libexec",
        "/usr/local/opt/dotnet@9/libexec",
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
        dotnet_root = os.environ.get("DOTNET_ROOT") or _find_net9_dotnet_root()

        # Pre-check: if global.json pins an SDK we don't have installed
        # and rollForward won't bail us out, skip with a clear reason.
        # This prevents confusing scip-dotnet error spew and avoids
        # wasting a `dotnet restore` round-trip that will fail anyway.
        reason = _global_json_satisfiable(project.root, dotnet_root)
        if reason is not None:
            logger.warning(
                f"SCIP: skipping scip-dotnet for {project.name}: {reason}. "
                f"Falling back to regex-only resolution for this project."
            )
            raise RuntimeError(f"scip-dotnet incompatibility: {reason}")

        argv: list[str] = [self.binary, "index", "--output", str(out_path)]
        argv.extend(config.extra_args)

        env_extras: dict[str, str] = {"DOTNET_ROLL_FORWARD": "LatestMajor"}
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
