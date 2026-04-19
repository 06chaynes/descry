"""Go adapter backed by scip-go (Sourcegraph).

scip-go is a Go-toolchain-based indexer installed via::

    go install github.com/sourcegraph/scip-go/cmd/scip-go@latest

It uses the Go ``go list`` tooling to walk the module's package graph and
emits a single ``.scip`` index file. The ``--output`` flag is supported
(verified via ``scip-go --help`` as of 0.1.26), so descry uses
``output_mode="direct"``.

SCIP symbol format (path-descriptor style, same as Rust and Java):

    scip-go gomod <module> <version> <descriptors>

Examples:
    scip-go gomod github.com/example/mylib v0.1.0 pkg/http/Server#Handle().
    scip-go gomod k8s.io/kubernetes v1.29.0
        pkg/kubelet/Kubelet#Run().
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


def _has_go_sources(pkg_dir: Path) -> bool:
    """True if `pkg_dir` contains any ``.go`` file outside ``vendor/``."""
    for go_file in pkg_dir.rglob("*.go"):
        if "vendor" in go_file.parts:
            continue
        return True
    return False


class GoAdapter:
    """scip-go — Go module symbols via the Go toolchain."""

    name = "go"
    scheme = "scip-go"
    binary = "scip-go"
    extensions = (".go",)

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per Go module.

        A module is any directory containing ``go.mod``. Multi-module
        monorepos (one ``go.mod`` per subdirectory) are supported.
        **Root is always included as a project if it has a ``go.mod``**
        — many repos (prometheus, etc.) have a large root-level module
        plus small helper modules under subdirs; the prior "only
        include root when no subdir modules exist" rule silently
        skipped those root modules, killing resolution rates on the
        largest codebases. ``go.work`` multi-module workspaces are out
        of scope; each ``go.mod`` is indexed independently.
        """
        seen: set[str] = set()
        projects: list[DiscoveredProject] = []

        if (root / "go.mod").exists() and _has_go_sources(root):
            seen.add(root.name)
            projects.append(
                DiscoveredProject(
                    name=root.name,
                    root=root,
                    language=self.name,
                )
            )

        for marker in root.glob("*/go.mod"):
            pkg_dir = marker.parent
            if pkg_dir.name.startswith("."):
                continue
            if pkg_dir.name in excluded_dirs:
                continue
            if pkg_dir.name in seen:
                continue
            if not _has_go_sources(pkg_dir):
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
        """Build the ``scip-go --output <out>`` command.

        scip-go is invoked from the module root and discovers packages via
        ``go list``. Optional overrides are surfaced via
        ``config.options``: ``module_name``, ``module_version``,
        ``go_version`` — each maps to the corresponding ``scip-go``
        flag. ``config.extra_args`` are appended verbatim after any
        validated options.
        """
        argv: list[str] = [self.binary, "--output", str(out_path)]

        opts = config.options or {}
        if "module_name" in opts:
            argv.extend(["--module-name", opts["module_name"]])
        if "module_version" in opts:
            argv.extend(["--module-version", opts["module_version"]])
        if "go_version" in opts:
            argv.extend(["--go-version", opts["go_version"]])

        argv.extend(config.extra_args)

        return CommandSpec(argv=argv, cwd=project.root, output_mode="direct")

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
