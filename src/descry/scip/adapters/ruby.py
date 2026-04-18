"""Ruby adapter backed by scip-ruby (Sourcegraph).

scip-ruby builds on Sorbet. Files annotated with ``# typed: <level>``
yield the highest-quality resolution; files with ``# typed: false`` (or
no typed comment) are indexed best-effort.

Install (direct binary; the gem path also works but requires a
Gemfile.lock entry):

    ARCH="$(uname -m)" OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
    curl -L "https://github.com/sourcegraph/scip-ruby/releases/latest/download/scip-ruby-$ARCH-$OS" \\
         -o scip-ruby && chmod +x scip-ruby

Invocation: run at project root. Without a ``sorbet/config`` file
scip-ruby still needs a positional ``.`` argument to scope the walk,
so the adapter always passes it.

scip-ruby writes ``index.scip`` to cwd (no ``--output`` flag), so the
adapter uses ``output_mode="rename"`` to move it into the cache dir.
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


def _has_ruby_sources(pkg_dir: Path) -> bool:
    for ext in ("*.rb", "*.rake", "*.gemspec"):
        if any(pkg_dir.rglob(ext)):
            return True
    return False


class RubyAdapter:
    """scip-ruby — Ruby symbols via Sorbet."""

    name = "ruby"
    scheme = "scip-ruby"
    binary = "scip-ruby"
    extensions = (".rb", ".rake", ".gemspec")

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per top-level Ruby project.

        A Ruby project is any top-level subdirectory containing a
        ``Gemfile`` plus at least one ``.rb`` source. The root itself
        becomes the single project when it carries a Gemfile with no
        competing subdirectory projects.
        """
        seen: set[str] = set()
        projects: list[DiscoveredProject] = []

        for marker in root.glob("*/Gemfile"):
            pkg_dir = marker.parent
            if pkg_dir.name.startswith("."):
                continue
            if pkg_dir.name in excluded_dirs:
                continue
            if pkg_dir.name in seen:
                continue
            if not _has_ruby_sources(pkg_dir):
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
            if (root / "Gemfile").exists() and _has_ruby_sources(root):
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
        """Build the ``scip-ruby .`` command.

        scip-ruby writes ``index.scip`` in cwd; the shared runner renames
        it to the requested ``out_path`` via ``output_mode="rename"``.
        The trailing ``"."`` explicitly scopes the walk to the project
        root so scip-ruby behaves consistently whether or not
        ``sorbet/config`` is present.
        """
        argv: list[str] = [self.binary, "."]
        argv.extend(config.extra_args)
        return CommandSpec(argv=argv, cwd=project.root, output_mode="rename")

    def parse_descriptors(self, raw: str) -> list[str]:
        """Parse scip-ruby descriptor strings into name components.

        scip-ruby emits the same backtick-wrapped file-path format used
        by scip-typescript / scip-python / scip-go, so the shared helper
        handles it correctly. Verify with a real fixture; swap to the
        path-style (Rust) helper if scip-ruby changes formats.
        """
        return parse_backtick_descriptors(raw)


register(RubyAdapter())
