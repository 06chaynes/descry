"""Dart adapter backed by scip-dart (Workiva).

scip-dart is distributed as a Dart pub package. Install:

    dart pub global activate scip_dart

That installs the binary into ``$HOME/.pub-cache/bin/scip_dart``. If
that directory isn't on PATH, symlink into your usual local bin:

    ln -s "$HOME/.pub-cache/bin/scip_dart" "$HOME/.local/bin/scip-dart"

scip-dart expects a populated ``.dart_tool/package_config.json`` — run
``dart pub get`` (or ``flutter pub get`` for Flutter projects) at the
project root before indexing, otherwise scip-dart exits with
``ERROR: Unable to locate packageConfig``.

Invocation: ``scip-dart ./`` from the package root. No ``--output``
flag — writes ``index.scip`` to cwd, so the adapter uses
``output_mode="rename"`` to move the emitted file into the descry
cache atomically.
"""

from __future__ import annotations

import logging
from pathlib import Path

from descry.scip.adapter import (
    AdapterConfig,
    CommandSpec,
    DiscoveredProject,
    register,
)
from descry.scip.adapters.typescript import parse_backtick_descriptors

logger = logging.getLogger(__name__)


def _has_dart_sources(pkg_dir: Path) -> bool:
    return any(pkg_dir.rglob("*.dart"))


class DartAdapter:
    """scip-dart — Dart / Flutter symbols via the Dart analyzer."""

    name = "dart"
    scheme = "scip-dart"
    binary = "scip-dart"
    extensions = (".dart",)

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per Dart package.

        A Dart package is a directory containing ``pubspec.yaml``.
        Discovery precedence:
        1. Root-level ``pubspec.yaml`` → single project (standard Dart
           and Flutter app layout).
        2. Per-subdir packages (monorepo / melos workspace layout with
           ``packages/*/pubspec.yaml``).
        """
        if (root / "pubspec.yaml").exists() and _has_dart_sources(root):
            return [
                DiscoveredProject(
                    name=root.name,
                    root=root,
                    language=self.name,
                )
            ]

        seen: set[str] = set()
        projects: list[DiscoveredProject] = []

        for marker in root.glob("*/pubspec.yaml"):
            pkg_dir = marker.parent
            if pkg_dir.name.startswith("."):
                continue
            if pkg_dir.name in excluded_dirs:
                continue
            if pkg_dir.name in seen:
                continue
            if not _has_dart_sources(pkg_dir):
                continue
            seen.add(pkg_dir.name)
            projects.append(
                DiscoveredProject(
                    name=pkg_dir.name,
                    root=pkg_dir,
                    language=self.name,
                )
            )

        # Melos / workspace layout: packages/<name>/pubspec.yaml.
        if not projects:
            for marker in root.glob("packages/*/pubspec.yaml"):
                pkg_dir = marker.parent
                if pkg_dir.name.startswith("."):
                    continue
                if pkg_dir.name in excluded_dirs:
                    continue
                if pkg_dir.name in seen:
                    continue
                if not _has_dart_sources(pkg_dir):
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
        """scip-dart takes the package root as a positional arg and
        writes ``index.scip`` to cwd.

        The shared runner (``output_mode="rename"``) moves that file
        into ``out_path`` after success. If ``dart pub get`` hasn't run
        scip-dart exits non-zero with a clear error; the runner reports
        the exit code and moves on to other adapters.
        """
        argv: list[str] = [self.binary, "./"]
        argv.extend(config.extra_args)
        return CommandSpec(argv=argv, cwd=project.root, output_mode="rename")

    def parse_descriptors(self, raw: str) -> list[str]:
        """scip-dart descriptor format isn't publicly documented — the
        shared backtick helper handles both backtick-wrapped and
        path-style forms, which covers the two common SCIP conventions.
        Verify on a real index during smoke tests and swap if needed.
        """
        return parse_backtick_descriptors(raw)


register(DartAdapter())
