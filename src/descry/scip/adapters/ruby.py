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

from pathlib import Path

from descry.scip.adapter import (
    AdapterConfig,
    CommandSpec,
    DiscoveredProject,
    register,
    subdir_projects,
)
from descry.scip.adapters.typescript import parse_backtick_descriptors


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

        A root with a ``Gemfile`` and ``.rb`` sources is the single
        project. A root with none is a workspace whose top-level
        subdirectories, each with a ``Gemfile`` and ``.rb`` sources, are
        the projects.
        """
        if (root / "Gemfile").exists():
            # scip-ruby run at a root with its Gemfile indexes the tree.
            if not _has_ruby_sources(root):
                return []
            return [DiscoveredProject(name=root.name, root=root)]
        return subdir_projects(root, excluded_dirs, ("*/Gemfile",), _has_ruby_sources)

    def build_command(
        self,
        project: DiscoveredProject,
        _out_path: Path,
        config: AdapterConfig,
    ) -> CommandSpec:
        """Build the ``scip-ruby .`` command.

        scip-ruby writes ``index.scip`` in cwd; the shared runner renames
        it to the requested ``out_path`` via ``output_mode="rename"``.
        The trailing ``"."`` explicitly scopes the walk to the project
        root so scip-ruby behaves consistently whether or not
        ``sorbet/config`` is present.

        ``-q`` / ``--no-error-count`` / ``--silence-dev-message`` let
        scip-ruby tolerate Sorbet non-critical errors that would
        otherwise exit non-zero before writing any index — on
        Sorbet-unannotated Ruby code (the common case), scip-ruby v1
        aborts at the first typecheck error without this flag
        combination.
        """
        argv: list[str] = [
            self.binary,
            "-q",
            "--no-error-count",
            "--silence-dev-message",
            ".",
        ]
        return CommandSpec(argv=argv, cwd=project.root, output_mode="rename")

    def parse_descriptors(self, raw: str) -> list[str]:
        """Parse scip-ruby descriptor strings into name components.

        scip-ruby emits the same backtick-wrapped file-path format used
        by scip-typescript / scip-python / scip-go, so the shared helper
        handles it.
        """
        return parse_backtick_descriptors(raw)


register(RubyAdapter())
