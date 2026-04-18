"""PHP adapter backed by scip-php (davidrjenni/scip-php).

Install globally via Composer:

    composer global require davidrjenni/scip-php

(As of v0.0.2 the package's ``google/protobuf`` dep hits a known
Composer security advisory; use ``composer global config audit.ignore
'[\"PKSA-tcfz-w4fm-hhk9\"]' --json`` to allow the install for
development use.)

Invocation: ``scip-php`` at the project root. Requires
``composer.json``/``composer.lock``/``vendor/`` populated. No
``--output`` flag, so the adapter uses ``output_mode="rename"`` to
move the emitted ``index.scip`` into the descry cache.
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


def _has_php_sources(pkg_dir: Path) -> bool:
    return any(pkg_dir.rglob("*.php"))


class PhpAdapter:
    """scip-php — PHP symbols via static analysis."""

    name = "php"
    scheme = "scip-php"
    binary = "scip-php"
    extensions = (".php",)

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per top-level Composer package.

        A PHP project is a directory containing ``composer.json``. The
        adapter does not require ``vendor/`` during discovery because
        that's a runtime prerequisite for scip-php, not for descry's
        file walk. scip-php will surface its own error if vendor is
        missing at index time.
        """
        seen: set[str] = set()
        projects: list[DiscoveredProject] = []

        for marker in root.glob("*/composer.json"):
            pkg_dir = marker.parent
            if pkg_dir.name.startswith("."):
                continue
            if pkg_dir.name in excluded_dirs:
                continue
            if pkg_dir.name in seen:
                continue
            if not _has_php_sources(pkg_dir):
                continue
            seen.add(pkg_dir.name)
            projects.append(
                DiscoveredProject(
                    name=pkg_dir.name,
                    root=pkg_dir,
                    language=self.name,
                )
            )

        if not projects:
            if (root / "composer.json").exists() and _has_php_sources(root):
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
        """scip-php has no subcommands or --output flag — just run it."""
        argv: list[str] = [self.binary]
        argv.extend(config.extra_args)
        return CommandSpec(argv=argv, cwd=project.root, output_mode="rename")

    def parse_descriptors(self, raw: str) -> list[str]:
        """Delegate to the backtick parser.

        scip-php's descriptor format is not publicly documented. The
        shared backtick helper handles both backtick-wrapped and
        path-style descriptors, so it's a safe default; verify on a
        real index during the smoke test and swap if needed.
        """
        return parse_backtick_descriptors(raw)


register(PhpAdapter())
