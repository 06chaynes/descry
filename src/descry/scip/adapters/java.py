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

# The regular files by which scip-java 0.12.3 detects each build tool it
# supports. sbt is also detected by a `project/build.properties` holding the
# key `sbt.version`; `_SBT_VERSION_KEY` finds it where a line starts with it,
# which is every form but an escaped key or a continued line. A `\u` escape
# not followed by four hex digits outside a comment, `_MALFORMED_ESCAPE`,
# makes scip-java abort reading the file. Text mode turns CR and CRLF into
# LF, the other line ends the properties format accepts.
_BUILD_TOOL_MARKERS = {
    "SCIP": ("scip-java.json", "lsif-java.json"),
    "Maven": ("pom.xml",),
    "Gradle": ("settings.gradle", "gradlew", "build.gradle", "build.gradle.kts"),
    "sbt": ("build.sbt",),
    "mill": ("build.sc",),
    "Bazel": ("WORKSPACE",),
}
_SBT_VERSION_KEY = re.compile(r"^[ \t\f]*sbt\.version(?:[ \t\f=:]|$)", re.MULTILINE)
_MALFORMED_ESCAPE = re.compile(
    r"^(?![ \t\f]*[#!]).*?(?<!\\)(?:\\\\)*\\u(?![0-9a-fA-F]{4})", re.MULTILINE
)


def _build_tools(directory: Path) -> set[str] | None:
    """The build tools scip-java detects in `directory`, or None where it
    aborts reading a `project/build.properties` it cannot read or parse.

    Run there without ``--build-tool``, as descry runs it, scip-java builds
    with the one it detects and refuses a directory where it detects none
    ("No build tool detected") or more than one ("Multiple build tools
    detected").
    """
    tools = {
        tool
        for tool, markers in _BUILD_TOOL_MARKERS.items()
        if any((directory / m).is_file() for m in markers)
    }
    properties = directory / "project" / "build.properties"
    if properties.is_file():
        try:
            text = properties.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        if _MALFORMED_ESCAPE.search(text):
            return None
        if _SBT_VERSION_KEY.search(text):
            tools.add("sbt")
    return tools


_GRADLE_SETTINGS = ("settings.gradle", "settings.gradle.kts")


def _runs_gradle(directory: Path) -> bool:
    """Whether scip-java run in `directory` builds it with Gradle."""
    return _build_tools(directory) == {"Gradle"}


def _scip_java_runs(directory: Path) -> bool:
    """Whether scip-java run in `directory` builds it, with one tool."""
    tools = _build_tools(directory)
    return tools is not None and len(tools) == 1


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


# Marker for the Kotlin DSL precompiled-script-plugin pattern that breaks
# scip-java. Both ``plugins { `kotlin-dsl` }`` (Kotlin DSL) and
# ``apply plugin: 'kotlin-dsl'`` (Groovy DSL) variants count.
_KOTLIN_DSL_RE = re.compile(
    r"""(?:
        ['"`]kotlin-dsl['"`]
        | id\s*\(\s*['"]org\.gradle\.kotlin\.kotlin-dsl['"]
        | id\s+['"]org\.gradle\.kotlin\.kotlin-dsl['"]
    )""",
    re.VERBOSE,
)


def _detect_kotlin_dsl_buildSrc(project_root: Path) -> str | None:
    """Return a one-line reason if scip-java will fail on this project.

    scip-java's init-script applies ``SemanticdbGradlePlugin`` to every
    project including ``:buildSrc`` and any ``includeBuild`` participants.
    If one of those projects applies the ``kotlin-dsl`` Gradle plugin,
    Gradle 8.x raises ``Querying the mapped value of map(flatmap(provider(
    task 'generatePrecompiledScriptPluginAccessors'))) before task ... has
    completed is not supported`` during *task creation* — there's no flag
    that defers this past configuration phase, so scip-java can't proceed.

    Observed on:
      * ``Kotlin/kotlinx.coroutines`` — ``buildSrc/build.gradle.kts``
        with ``plugins { `kotlin-dsl` }``.
      * ``ktorio/ktor`` — ``build-settings-logic/build.gradle.kts`` (an
        ``includeBuild`` participant) with ``kotlin-dsl``.

    We probe the well-known locations that Gradle treats as included
    builds with their own configuration:

      * ``buildSrc/build.gradle{,.kts}``
      * Any ``<dir>/build.gradle{,.kts}`` where ``<dir>`` is referenced
        via ``includeBuild`` in ``settings.gradle{,.kts}``.

    Returns the path that triggered detection, or ``None`` if safe.
    """
    candidates: list[Path] = []

    # buildSrc is implicitly included by Gradle.
    for ext in ("build.gradle.kts", "build.gradle"):
        bs = project_root / "buildSrc" / ext
        if bs.is_file():
            candidates.append(bs)

    # includeBuild('<path>') in settings.gradle{,.kts} — best-effort regex
    # parse, only used to decide which extra build files to probe.
    include_build_re = re.compile(r"""includeBuild\s*\(?\s*['"]([^'"]+)['"]""")
    for settings_name in ("settings.gradle.kts", "settings.gradle"):
        settings = project_root / settings_name
        if not settings.is_file():
            continue
        try:
            text = settings.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for sub_path in include_build_re.findall(text):
            sub_dir = (project_root / sub_path).resolve()
            try:
                sub_dir.relative_to(project_root.resolve())
            except ValueError:
                # includeBuild outside project root — out of our scope.
                continue
            for ext in ("build.gradle.kts", "build.gradle"):
                bs = sub_dir / ext
                if bs.is_file():
                    candidates.append(bs)

    for build_file in candidates:
        try:
            text = build_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _KOTLIN_DSL_RE.search(text):
            return str(build_file.relative_to(project_root))
    return None


class JavaAdapter:
    """scip-java — Java (+ Kotlin, Scala via the same indexer)."""

    name = "java"
    scheme = "semanticdb"
    binary = "scip-java"
    extensions = (".java", ".kt", ".scala")

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per top-level JVM build directory.

        A JVM "project" in descry's sense is any top-level subdirectory
        where scip-java detects exactly one build tool, and so runs, holding
        at least one JVM source file. The root itself is a single project
        on the same terms when its build aggregates the subdirectories' —
        a Gradle settings file, a Maven `pom.xml` with `<modules>` — or
        when no subdirectory is one. Indexing a subdirectory of an
        aggregating build alone fails where it depends on its parent's
        configuration (a Maven module's parent POM, a Gradle ``buildSrc/``,
        observed on spring-framework).
        """
        runs = _scip_java_runs(root) and _has_jvm_sources(root)
        settings = any((root / name).is_file() for name in _GRADLE_SETTINGS)
        pom = root / "pom.xml"
        modules = pom.is_file() and "<modules>" in pom.read_text(errors="replace")
        aggregates = settings if _runs_gradle(root) else modules
        if runs and aggregates:
            return [DiscoveredProject(name=root.name, root=root, language=self.name)]

        projects: list[DiscoveredProject] = []
        for pkg_dir in sorted(root.iterdir()):
            if not pkg_dir.is_dir() or pkg_dir.name.startswith("."):
                continue
            if pkg_dir.name in excluded_dirs:
                continue
            if not _scip_java_runs(pkg_dir) or not _has_jvm_sources(pkg_dir):
                continue
            projects.append(
                DiscoveredProject(
                    name=pkg_dir.name,
                    root=pkg_dir,
                    language=self.name,
                )
            )

        if not projects and runs:
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
        """
        # The init-script and the default task list are Gradle arguments:
        # scip-java forwards everything after `--` to the build tool it
        # detects, and on a Maven (or sbt) project those flags reach `mvn`
        # (or `sbt`), which rejects them and writes no index.
        gradle = _runs_gradle(project.root)

        # Pre-flight: scip-java's SemanticdbGradlePlugin applies to
        # allprojects (incl. :buildSrc and includeBuild participants).
        # On Kotlin DSL projects with precompiled-script-plugins, that
        # query-during-config combination is unsupported by Gradle 8.x
        # and there is no flag (incl. -x / --exclude-task) that defers
        # it past task creation. Skip with a clear log so the user
        # sees what's happening instead of an opaque scip-java stderr.
        # The plugin runs inside Gradle, so only a Gradle build is skipped.
        kotlin_dsl_marker = (
            _detect_kotlin_dsl_buildSrc(project.root) if gradle else None
        )
        if kotlin_dsl_marker is not None:
            logger.warning(
                f"SCIP: skipping scip-java for {project.name}: "
                f"{kotlin_dsl_marker} applies the `kotlin-dsl` Gradle plugin, "
                f"which scip-java's SemanticdbGradlePlugin cannot index "
                f"alongside Gradle 8.x precompiled-script-plugin accessors "
                f"(upstream sourcegraph/scip-java limitation). "
                f"Falling back to regex-only resolution for this project."
            )
            raise RuntimeError(
                "scip-java incompatibility: kotlin-dsl precompiled-script-plugins "
                f"in {kotlin_dsl_marker}"
            )

        argv: list[str] = [self.binary, "index", "--output", str(out_path)]

        init_script = _compat_init_script() if gradle else None
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

        if pass_through:
            argv.append("--")
            argv.extend(pass_through)

        return CommandSpec(argv=argv, cwd=project.root)

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
