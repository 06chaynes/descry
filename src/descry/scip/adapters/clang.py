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
1. ``config.options["compdb_path"]`` (user override).
2. ``{project.root}/compile_commands.json`` (in-place).
3. ``{project.root}/build/compile_commands.json`` (conventional
   out-of-tree build dir).

If none is found the adapter still returns a CommandSpec but logs a
warning; scip-clang will fail with a clear message pointing at the
missing compdb.
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


def _find_compdb(project_root: Path, config: AdapterConfig) -> Path | None:
    """Locate compile_commands.json with precedence rules from the adapter docstring."""
    override = config.options.get("compdb_path") if config.options else None
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.exists():
            return candidate
        logger.warning(f"scip-clang: configured compdb_path {override!r} not found")
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

        Discovery precedence:
        1. **Root compile_commands.json takes priority.** If the
           workspace has a ``compile_commands.json`` at the root, treat
           the whole repo as a single project — this is how Bear-backed
           Makefile builds and top-level CMake builds emit their compdb,
           and running scip-clang from subdirs would miss the entries.
        2. Per-subdir projects (each with its own
           CMakeLists.txt/Makefile/meson.build/compile_commands.json).
        3. Root as single project if root has a build marker.
        """
        markers = (
            "CMakeLists.txt",
            "Makefile",
            "meson.build",
            "compile_commands.json",
        )

        # Priority 1: root-level compile_commands.json → single project.
        if (root / "compile_commands.json").exists() and _has_c_sources(root):
            return [
                DiscoveredProject(
                    name=root.name,
                    root=root,
                    language=self.name,
                )
            ]

        seen: set[str] = set()
        projects: list[DiscoveredProject] = []

        for marker_name in markers:
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

        if not projects:
            root_has_marker = any((root / m).exists() for m in markers)
            if root_has_marker and _has_c_sources(root):
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
        """Build the scip-clang command with --compdb-path and
        --index-output-path.

        If no compile_commands.json is present, the adapter still
        returns a command pointing at the default path so scip-clang
        can surface its own "missing compdb" error; the shared runner
        reports the exit code and moves on.
        """
        compdb = _find_compdb(project.root, config)
        compdb_arg = (
            str(compdb) if compdb else str(project.root / "compile_commands.json")
        )

        argv: list[str] = [
            self.binary,
            f"--compdb-path={compdb_arg}",
            f"--index-output-path={out_path}",
        ]
        argv.extend(config.extra_args)

        return CommandSpec(argv=argv, cwd=project.root, output_mode="direct")

    def parse_descriptors(self, raw: str) -> list[str]:
        """scip-clang descriptors follow the same path-suffix format
        other SCIP indexers use. The shared backtick helper handles
        both backtick-wrapped and plain path-style forms.
        """
        return parse_backtick_descriptors(raw)


register(ClangAdapter())
