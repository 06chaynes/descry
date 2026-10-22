"""C/C++ adapter backed by scip-clang (Sourcegraph).

Install (direct binary; no Homebrew tap):

    curl -L "https://github.com/sourcegraph/scip-clang/releases/latest/download/scip-clang-arm64-darwin" \\
         -o scip-clang && chmod +x scip-clang

scip-clang needs a Clang-format compilation database
(``compile_commands.json``). Generate it however your build system
supports:

- CMake:  ``cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON``
- Meson (Ninja backend): generated automatically into ``build/``
- Bazel: hedronvision/bazel-compile-commands-extractor
- Make:  ``bear -- make``

The adapter looks for the compdb at (in order):
1. ``{project.root}/compile_commands.json`` (in-place).
2. ``{project.root}/build/compile_commands.json`` (conventional
   out-of-tree build dir).

A project with neither is not indexed: the adapter raises, and the
runner logs why.
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


_C_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx", ".cu")
_C_HEADERS = (".h", ".hh", ".hpp", ".hxx")


def _has_c_sources(pkg_dir: Path) -> bool:
    for pattern in (*_C_EXTENSIONS, *_C_HEADERS):
        if any(pkg_dir.rglob(f"*{pattern}")):
            return True
    return False


# A build file that makes a directory a project
_MARKERS = ("CMakeLists.txt", "Makefile", "meson.build", "compile_commands.json")


def _find_compdb(project_root: Path) -> Path | None:
    """Locate compile_commands.json with precedence rules from the adapter docstring."""
    for rel in ("compile_commands.json", "build/compile_commands.json"):
        candidate = project_root / rel
        if candidate.exists():
            return candidate
    return None


class ClangAdapter:
    """scip-clang — C/C++/CUDA symbols via Clang frontend.

    scip-clang emits its SCIP scheme as ``cxx`` (not ``scip-clang``):
    ``cxx . . $ listCreate(a153265b2bd52385).``
    The ``$`` is scip-clang's "project-symbol" marker, the trailing
    parenthesized token is the signature hash, and the terminator
    ``.`` is the standard SCIP Term/Method suffix.
    """

    name = "clang"
    scheme = "cxx"
    binary = "scip-clang"
    extensions = (".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx")

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per top-level C/C++ project.

        A root with a compile database of its own, there or in ``build/``,
        where a CMake build writes it, is one project: its subdirectories'
        build files are parts of it, as `add_subdirectory` and recursive
        make make them, and scip-clang run from one would miss the root's
        entries. Else the subdirectories with a build file and a compile
        database of their own are the projects; where none has one, a root
        with a build file (CMakeLists.txt, Makefile, meson.build) is the
        project, else each subdirectory with one is.
        """
        if _find_compdb(root) is None:
            parts = self._parts(root, excluded_dirs)
            compiled = [p for p in parts if _find_compdb(p.root)]
            if compiled or not any((root / m).exists() for m in _MARKERS):
                return compiled or parts
        if not _has_c_sources(root):
            return []
        return [DiscoveredProject(name=root.name, root=root, language=self.name)]

    def _parts(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """The subdirectories of ``root`` with a build file and C/C++
        sources of their own."""
        seen: set[str] = set()
        projects: list[DiscoveredProject] = []
        for marker_name in _MARKERS:
            for marker in root.glob(f"*/{marker_name}"):
                pkg_dir = marker.parent
                if pkg_dir.name.startswith("."):
                    continue
                if pkg_dir.name in excluded_dirs:
                    continue
                if pkg_dir.name in seen:
                    continue
                if not _has_c_sources(pkg_dir):
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
        """Build the scip-clang command with --compdb-path and
        --index-output-path. A project without a compile database is not
        indexed: this raises, and the shared runner logs why.
        """
        compdb = _find_compdb(project.root)
        if compdb is None:
            raise FileNotFoundError(
                f"no compile_commands.json in {project.root} or its build/"
            )
        argv: list[str] = [
            self.binary,
            f"--compdb-path={compdb}",
            f"--index-output-path={out_path}",
        ]

        return CommandSpec(argv=argv, cwd=project.root, output_mode="direct")

    def parse_descriptors(self, raw: str) -> list[str]:
        """scip-clang descriptors follow the same path-suffix format
        other SCIP indexers use. The shared backtick helper handles
        both backtick-wrapped and plain path-style forms.
        """
        return parse_backtick_descriptors(raw)


register(ClangAdapter())
