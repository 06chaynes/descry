"""Go adapter backed by scip-go (Sourcegraph).

scip-go is a Go-toolchain-based indexer installed via::

    go install github.com/scip-code/scip-go/cmd/scip-go@latest

It uses the Go ``go list`` tooling to walk the module's package graph and
emits a single ``.scip`` index file. The ``--output`` flag is supported,
so descry uses ``output_mode="direct"``.

SCIP symbol format (path-descriptor style, same as Rust and Java):

    scip-go gomod <module> <version> <descriptors>

Examples:
    scip-go gomod github.com/example/mylib v0.1.0 pkg/http/Server#Handle().
    scip-go gomod k8s.io/kubernetes v1.29.0
        pkg/kubelet/Kubelet#Run().
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


def _has_go_sources(pkg_dir: Path) -> bool:
    """True if `pkg_dir` contains any ``.go`` file outside ``vendor/``."""
    return any("vendor" not in f.parts for f in pkg_dir.rglob("*.go"))


class GoAdapter:
    """scip-go — Go module symbols via the Go toolchain."""

    name = "go"
    scheme = "scip-go"
    binary = "scip-go"
    extensions = (".go",)

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per Go module.

        A module is any directory containing ``go.mod``. Multi-module
        monorepos (one ``go.mod`` per subdirectory) are supported. The
        root is a project beside them where it has a ``go.mod``.
        ``go.work`` multi-module workspaces are out of scope; each
        ``go.mod`` is indexed independently.
        """
        projects: list[DiscoveredProject] = []
        if (root / "go.mod").exists() and _has_go_sources(root):
            projects.append(DiscoveredProject(name=root.name, root=root))
        projects += subdir_projects(
            root,
            excluded_dirs,
            ("*/go.mod",),
            _has_go_sources,
            frozenset(p.name for p in projects),
        )
        return sorted(projects, key=lambda p: p.name)

    def build_command(
        self,
        project: DiscoveredProject,
        out_path: Path,
        config: AdapterConfig,
    ) -> CommandSpec:
        """Build the ``scip-go --output <out>`` command.

        scip-go is invoked from the module root and discovers packages via
        ``go list``.
        """
        argv: list[str] = [self.binary, "--output", str(out_path)]

        return CommandSpec(argv=argv, cwd=project.root)

    def parse_descriptors(self, raw: str) -> list[str]:
        """Parse scip-go descriptor strings into name components.

        scip-go wraps domain-style package paths in backticks (same wire
        format as scip-typescript and scip-python):

            `k8s.io/kubernetes/pkg/kubelet`/Kubelet#Run().
            `github.com/example/svc/internal/http`/Server#Handle().

        The path components inside backticks are redundant (already in
        the file path); we skip to the symbol portion after the final
        closing backtick.

        Examples:
            ``\\`k8s.io/.../kubelet\\`/Kubelet#Run().`` -> ``["Kubelet", "Run"]``
            ``pkg/http/Server#Handle().`` (no backticks) -> ``["Server", "Handle"]``
        """
        return parse_backtick_descriptors(raw)


register(GoAdapter())
