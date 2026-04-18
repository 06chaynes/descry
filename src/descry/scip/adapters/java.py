"""Java/Kotlin/Scala adapter backed by scip-java (Sourcegraph).

scip-java is a JVM-based indexer installed via coursier:

    coursier bootstrap --standalone -o scip-java \\
        com.sourcegraph:scip-java_2.13:0.12.3 \\
        --main com.sourcegraph.scip_java.ScipJava

It autodetects Maven (pom.xml), Gradle (build.gradle / build.gradle.kts), and
sbt (build.sbt) natively. Covers Java, Kotlin, and Scala in one pass — so
this adapter owns ``.java``, ``.kt``, and ``.scala`` even though descry's
baseline regex parser only handles Java source.

SCIP symbol format (path-descriptor scheme, not backticks):

    semanticdb maven <maven-coords> <version> <descriptors>

scip-java emits the SCIP ``scheme`` token as ``semanticdb`` (not
``scip-java``) because it uses Scala's SemanticDB format internally and
converts to SCIP. The ``manager`` slot is ``maven`` regardless of whether
the build system is Maven, Gradle, or sbt.

Examples:
    semanticdb maven maven/org.apache.kafka/kafka-streams 4.4.0-SNAPSHOT
        org/apache/kafka/streams/state/WindowStore#put().
    semanticdb maven . . com/example/Foo#InnerBar#method().
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from descry.scip.adapter import (
    AdapterConfig,
    CommandSpec,
    DiscoveredProject,
    register,
)

logger = logging.getLogger(__name__)


_JAVA_DESCRIPTOR_PATTERN = re.compile(
    r"([a-zA-Z_$][a-zA-Z0-9_$]*)(\([^)]*\)|[#./\[\]])?"
)

_JAVA_BUILD_MARKERS = (
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "build.sbt",
)


_COMPAT_INIT_SCRIPT_PATH = Path(__file__).parent / "_java_compat.init.gradle"


def _compat_init_script() -> Path | None:
    """Path to the shipped Gradle compat init-script, or None if missing.

    Shipped as part of the descry package so it travels with installs;
    returns None defensively so adapters still work even if the file is
    missing (e.g. a broken zip / packaging).
    """
    if _COMPAT_INIT_SCRIPT_PATH.exists():
        return _COMPAT_INIT_SCRIPT_PATH
    return None


def _has_jvm_sources(pkg_dir: Path) -> bool:
    """Quick check: does this directory contain any .java/.kt/.scala files?"""
    for ext in ("*.java", "*.kt", "*.scala"):
        if any(pkg_dir.rglob(ext)):
            return True
    return False


class JavaAdapter:
    """scip-java — Java (+ Kotlin, Scala via the same indexer)."""

    name = "java"
    scheme = "semanticdb"
    binary = "scip-java"
    extensions = (".java", ".kt", ".scala")

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per top-level JVM build directory.

        A JVM "project" in descry's sense is any top-level subdirectory
        containing one of the recognized build markers (Maven, Gradle, or
        sbt) plus at least one JVM source file. The root itself is
        detected as a single project when it carries a build marker with
        no subdirectory competing for the name.
        """
        seen: set[str] = set()
        projects: list[DiscoveredProject] = []

        for marker_name in _JAVA_BUILD_MARKERS:
            for marker in root.glob(f"*/{marker_name}"):
                pkg_dir = marker.parent
                if pkg_dir.name.startswith("."):
                    continue
                if pkg_dir.name in excluded_dirs:
                    continue
                if pkg_dir.name in seen:
                    continue
                if not _has_jvm_sources(pkg_dir):
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
            root_has_marker = any((root / m).exists() for m in _JAVA_BUILD_MARKERS)
            if root_has_marker and _has_jvm_sources(root):
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
        """Build the ``scip-java index --output <out>`` command.

        scip-java autodetects Maven/Gradle/sbt from the project directory.
        For Gradle projects, we always pass a shipped init-script that
        strips ``-Werror`` from JavaCompile tasks — this makes descry
        work out of the box on conservative Java codebases (Kafka, many
        Apache projects) whose builds treat any warning as fatal. The
        init-script is a no-op for projects that don't set ``-Werror``.

        When the init-script is applied we must also specify the default
        scip-java Gradle tasks (``clean scipPrintDependencies
        scipCompileAll``) because ``--`` replaces scip-java's default
        task list.

        Additional user extras from ``config.extra_args`` are appended
        AFTER the compat flags, so users can override or add to the
        default tasks.
        """
        argv: list[str] = [self.binary, "index", "--output", str(out_path)]

        init_script = _compat_init_script()
        pass_through: list[str] = []
        if init_script is not None:
            pass_through.extend(
                [
                    f"--init-script={init_script}",
                    "clean",
                    "scipPrintDependencies",
                    "scipCompileAll",
                ]
            )
        pass_through.extend(config.extra_args)

        if pass_through:
            argv.append("--")
            argv.extend(pass_through)

        env_extras: dict[str, str] = {}
        # scip-java honors JVM_VERSION; the AdapterConfig.toolchain slot is
        # the natural home for that override when set.
        if config.toolchain:
            env_extras["JVM_VERSION"] = config.toolchain

        return CommandSpec(argv=argv, cwd=project.root, env_extras=env_extras)

    def parse_descriptors(self, raw: str) -> list[str]:
        """Parse scip-java path-descriptor strings into name components.

        scip-java uses the same suffix conventions as Rust:
            ``/``   namespace/package separator — skip (already in file path)
            ``#``   type (class, interface, enum, record) — include
            ``.``   term (field, constant, enum value) — include
            ``()``  method signature — include
            ``[]``  type parameters — included (bracket is the suffix, not
                    the name), harmless pass-through

        Examples:
            ``org/example/Foo#bar().`` -> ``["Foo", "bar"]``
            ``org/example/Foo#Inner#method().`` -> ``["Foo", "Inner", "method"]``
            ``org/example/Color#RED.`` -> ``["Color", "RED"]``
        """
        names: list[str] = []
        for match in _JAVA_DESCRIPTOR_PATTERN.finditer(raw):
            name = match.group(1)
            suffix = match.group(2) or ""
            if not name:
                continue
            if suffix == "/":
                continue
            names.append(name)
        return names


register(JavaAdapter())
