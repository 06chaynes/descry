"""Python adapter backed by scip-python (npm `@sourcegraph/scip-python`)."""

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

# A Python environment with no packages installed, in scip-python's format
_EMPTY_ENVIRONMENT = Path(__file__).parent / "_empty_environment.json"


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

        1. Single-package root: `root` itself has a packaging marker and
           Python files in scope; scip-python run there indexes the tree
           under it. Yields a synthetic project whose name is the root
           directory's basename (so `.scip` filenames don't collide
           across projects).
        2. Monorepo, a root with no marker: each top-level subdirectory
           with its own `pyproject.toml` / `setup.py` and at least one
           `.py` file.
        """
        root_marker_exists = (
            (root / "pyproject.toml").exists()
            or (root / "setup.py").exists()
            or (root / "setup.cfg").exists()
        )
        if root_marker_exists:
            in_scope = (
                p for p in root.rglob("*.py") if _py_in_scope(root, p, excluded_dirs)
            )
            if not any(in_scope):
                return []
            return [DiscoveredProject(name=root.name, root=root)]
        return subdir_projects(
            root,
            excluded_dirs,
            ("*/pyproject.toml", "*/setup.py"),
            lambda pkg_dir: any(pkg_dir.rglob("*.py")),
        )

    def build_command(
        self,
        project: DiscoveredProject,
        out_path: Path,
        config: AdapterConfig,
    ) -> CommandSpec:
        """Build the `scip-python index` command.

        scip-python accepts `--project-name`, `--output`, and `--cwd`;
        the absolute `--cwd` prevents a leading-dash filename smuggling
        attack via `os.chdir`. No shell. An empty `--environment` stands
        in for the `pip list` it otherwise runs, which fails where no pip
        is on PATH and only names the symbols of other packages; a
        `--project-version` for the git revision it otherwise reads,
        without which it fails outside a repository.
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
            "--environment",
            str(_EMPTY_ENVIRONMENT),
            "--project-version",
            "0",
        ]
        return CommandSpec(argv=argv, cwd=project.root)

    def parse_descriptors(self, raw: str) -> list[str]:
        # scip-python uses the same backtick-wrapped file-path format as
        # scip-typescript, so we share the parser.
        return parse_backtick_descriptors(raw)

    def leaving(self, defined, root, implemented):
        """A test of whether a call of a symbol no index defines cannot
        reach the project: a function or method of a package the indexes
        define nothing of (`python-stdlib`'s `pathlib/Path#resolve().`) that
        no symbol of theirs overrides, ``implemented`` naming what they
        override. ``root`` is not read."""
        own = {symbol.split(" ")[2] for symbol in defined if symbol.count(" ") >= 4}

        def leaves(symbol: str) -> bool:
            parts = symbol.split(" ")
            return (
                len(parts) >= 5
                and parts[2] not in own
                and symbol.endswith(").")
                and symbol not in implemented
            )

        return leaves


register(PythonAdapter())
