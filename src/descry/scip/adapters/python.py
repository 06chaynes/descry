"""Python adapter backed by scip-python (npm `@sourcegraph/scip-python`)."""

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


def _py_in_scope(root: Path, path: Path, excluded_dirs: set[str]) -> bool:
    """True if `path` is under `root` and not within an excluded/hidden dir."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    for part in rel.parts[:-1]:
        if part.startswith(".") or part in excluded_dirs:
            return False
    return True


class PythonAdapter:
    """scip-python — Python symbols, including type-aware call resolution."""

    name = "python"
    scheme = "scip-python"
    binary = "scip-python"
    extensions = (".py",)

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Two discovery modes:

        1. Monorepo: each top-level subdirectory with its own
           `pyproject.toml` / `setup.py` and at least one `.py` file.
        2. Single-package root: `root` itself has a packaging marker and
           Python files in scope. Yields a synthetic project whose name is
           the root directory's basename (so `.scip` filenames don't
           collide across projects).
        """
        seen: set[str] = set()
        projects: list[DiscoveredProject] = []

        for marker in root.glob("*/pyproject.toml"):
            pkg_dir = marker.parent
            if pkg_dir.name.startswith(".") or pkg_dir.name in excluded_dirs:
                continue
            if not any(pkg_dir.rglob("*.py")):
                continue
            if pkg_dir.name in seen:
                continue
            seen.add(pkg_dir.name)
            projects.append(
                DiscoveredProject(
                    name=pkg_dir.name,
                    root=pkg_dir,
                    language=self.name,
                )
            )

        for marker in root.glob("*/setup.py"):
            pkg_dir = marker.parent
            if pkg_dir.name.startswith(".") or pkg_dir.name in excluded_dirs:
                continue
            if pkg_dir.name in seen:
                continue
            if not any(pkg_dir.rglob("*.py")):
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
            root_marker_exists = (
                (root / "pyproject.toml").exists()
                or (root / "setup.py").exists()
                or (root / "setup.cfg").exists()
            )
            if root_marker_exists and any(
                p for p in root.rglob("*.py") if _py_in_scope(root, p, excluded_dirs)
            ):
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
        """Build the `scip-python index` command.

        scip-python accepts `--project-name`, `--output`, and `--cwd`;
        the absolute `--cwd` prevents a leading-dash filename smuggling
        attack via `os.chdir`. No shell.
        """
        argv: list[str] = [
            self.binary,
            "index",
            "--project-name",
            project.name,
            "--output",
            str(out_path),
            "--cwd",
            str(project.root),
        ]
        argv.extend(config.extra_args)
        return CommandSpec(argv=argv, cwd=project.root)

    def parse_descriptors(self, raw: str) -> list[str]:
        # scip-python uses the same backtick-wrapped file-path format as
        # scip-typescript, so we share the parser.
        return parse_backtick_descriptors(raw)


register(PythonAdapter())
