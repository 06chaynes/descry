""".NET adapter backed by scip-dotnet (Sourcegraph).

scip-dotnet is a .NET tool built on Roslyn that indexes C# and VB
projects via MSBuild. Install:

    dotnet tool install --global scip-dotnet

Invocation: ``scip-dotnet index`` at the project's directory, which
indexes every ``.sln`` / ``.csproj`` / ``.vbproj`` there, else
``scip-dotnet index <each .slnx>``. The tool can do its own ``dotnet
restore``. Supports ``--output`` so we use ``output_mode="direct"``.

scip-dotnet v0.2.13 targets ``net9.0`` — on a system with a newer SDK
(.NET 10+) the adapter sets ``DOTNET_ROLL_FORWARD=LatestMajor`` so the
tool runs against the installed runtime, and ``DOTNET_ROOT`` so it
finds one.
"""

from __future__ import annotations

import os
from pathlib import Path

from descry.scip.adapter import (
    AdapterConfig,
    CommandSpec,
    DiscoveredProject,
    register,
    subdir_projects,
)
from descry.scip.adapters.typescript import parse_backtick_descriptors

_DOTNET_MARKERS = (".sln", ".slnx", ".csproj", ".vbproj", ".fsproj")


def _has_marker(pkg_dir: Path, markers=_DOTNET_MARKERS) -> bool:
    """Whether ``pkg_dir`` holds a file with one of the extensions ``markers``."""
    return any(any(pkg_dir.glob(f"*{marker}")) for marker in markers)


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
    if not gj.is_file():
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
    if roll_forward == "latestmajor" and allow_prerelease:
        return None

    return (
        f"global.json pins SDK {pin!r} (rollForward={roll_forward or 'default'}, "
        f"allowPrerelease={allow_prerelease}); installed SDKs at {dotnet_root} "
        f"are {sorted(installed)!r}. scip-dotnet's 'dotnet restore' will fail "
        f"for every project under this pin."
    )


def _find_net9_dotnet_root() -> str | None:
    """The first of these locations holding a 9.x runtime, or None."""
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
        1. The root, where it holds a solution or project file.
        2. Top-level subdirectories holding one.
        """
        if _has_marker(root) and _has_dotnet_sources(root):
            return [DiscoveredProject(name=root.name, root=root)]
        patterns = tuple(f"*/*{marker}" for marker in _DOTNET_MARKERS)
        return subdir_projects(root, excluded_dirs, patterns, _has_dotnet_sources)

    def build_command(
        self,
        project: DiscoveredProject,
        out_path: Path,
        config: AdapterConfig,
    ) -> CommandSpec:
        """Build the ``scip-dotnet index --output <out>`` command.

        It sets ``DOTNET_ROLL_FORWARD=LatestMajor``, and ``DOTNET_ROOT``,
        with its ``bin`` first on ``PATH``, to ``$DOTNET_ROOT`` or else a
        side-installed net9 runtime where one is found: scip-dotnet
        0.2.13 targets ``net9.0``, which ``LatestMajor`` does not roll
        forward to net10+. Raises where ``global.json`` pins an SDK that
        is not installed.
        """
        dotnet_root = os.environ.get("DOTNET_ROOT") or _find_net9_dotnet_root()

        # Pre-check: if global.json pins an SDK we don't have installed
        # and rollForward won't bail us out, skip with a clear reason.
        # This prevents confusing scip-dotnet error spew and avoids
        # wasting a `dotnet restore` round-trip that will fail anyway.
        reason = _global_json_satisfiable(project.root, dotnet_root)
        if reason is not None:
            raise RuntimeError(reason)

        argv: list[str] = [self.binary, "index", "--output", str(out_path)]
        # It indexes every `.sln` and project file it finds itself, and no
        # `.slnx`, which it is given where it finds none of those.
        if not _has_marker(project.root, (".sln", ".csproj", ".vbproj")):
            argv += sorted(p.name for p in project.root.glob("*.slnx"))

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
        )

    def parse_descriptors(self, raw: str) -> list[str]:
        r"""scip-dotnet emits backtick-wrapped descriptors.

        Example from the README:
        ``scip-dotnet nuget . . Main/Expressions#TargetType#\`.ctor\`()``
        """
        return parse_backtick_descriptors(raw)


register(DotnetAdapter())
