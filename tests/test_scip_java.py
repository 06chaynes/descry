"""Tests for the scip-java adapter: discovery, command building, and
SCIP symbol resolution through the shared ScipIndex dispatch path."""

from __future__ import annotations

import os

import pytest

from descry.scip.adapter import AdapterConfig, DiscoveredProject
from descry.scip.adapters.java import JavaAdapter
from descry.scip.parser import ScipIndex


class TestJavaAdapterDiscovery:
    def test_single_maven_project_at_root(self, tmp_path):
        (tmp_path / "pom.xml").write_text("<project/>")
        (tmp_path / "src" / "main" / "java" / "com" / "ex").mkdir(parents=True)
        (tmp_path / "src" / "main" / "java" / "com" / "ex" / "Foo.java").write_text(
            "package com.ex; public class Foo {}"
        )

        projects = JavaAdapter().discover(tmp_path, set())

        assert len(projects) == 1
        assert projects[0].name == tmp_path.name
        assert projects[0].root == tmp_path
        assert projects[0].language == "java"

    def test_gradle_monorepo(self, tmp_path):
        for name in ("alpha", "beta"):
            pkg = tmp_path / name
            pkg.mkdir()
            (pkg / "build.gradle").write_text("// gradle config")
            src = pkg / "src" / "main" / "java"
            src.mkdir(parents=True)
            (src / f"{name.capitalize()}.java").write_text(
                f"public class {name.capitalize()} {{}}"
            )

        projects = JavaAdapter().discover(tmp_path, set())

        names = sorted(p.name for p in projects)
        assert names == ["alpha", "beta"]
        for p in projects:
            assert p.language == "java"
            assert p.root.name == p.name

    def test_sbt_build(self, tmp_path):
        pkg = tmp_path / "scala-svc"
        pkg.mkdir()
        (pkg / "build.sbt").write_text('name := "scala-svc"')
        (pkg / "Main.scala").write_text("object Main { def main: Unit = () }")

        projects = JavaAdapter().discover(tmp_path, set())

        assert [p.name for p in projects] == ["scala-svc"]

    def test_kotlin_gradle_kts_detected(self, tmp_path):
        pkg = tmp_path / "kotlin-svc"
        pkg.mkdir()
        (pkg / "build.gradle.kts").write_text('plugins { kotlin("jvm") }')
        (pkg / "Main.kt").write_text('fun main() = println("hi")')

        projects = JavaAdapter().discover(tmp_path, set())

        assert [p.name for p in projects] == ["kotlin-svc"]

    def test_multi_project_root_without_a_build_file(self, tmp_path):
        """A root holding `settings.gradle(.kts)` and `gradlew` but no build
        file of its own is one Gradle build, not a project per subproject."""
        for settings in ("settings.gradle", "settings.gradle.kts"):
            root = tmp_path / settings
            app = root / "app"
            app.mkdir(parents=True)
            (root / settings).write_text("include 'app'\n")
            (root / "gradlew").write_text("#!/bin/sh\n")
            (app / "build.gradle").write_text("// app build\n")
            (app / "Main.java").write_text("class Main {}")

            projects = JavaAdapter().discover(root, set())

            assert [p.root for p in projects] == [root], settings

    def test_root_with_another_build_tool_yields_its_subprojects(self, tmp_path):
        """scip-java refuses a directory where it detects another build tool
        beside Gradle, so such a root is not taken as one Gradle build."""
        others = {
            "scip-java.json": "{}",
            "lsif-java.json": "{}",
            "pom.xml": "<project/>",
            "build.sbt": "",
            "build.sc": "",
            "WORKSPACE": "",
            "project/build.properties": "# sbt\nsbt.version=1.9.9\n",
        }
        for other, text in others.items():
            for extra in ((), ("gradlew",), ("build.gradle",)):
                root = tmp_path / other.replace("/", "_") / "-".join(("root", *extra))
                app = root / "app"
                app.mkdir(parents=True)
                for name in ("settings.gradle", *extra):
                    (root / name).write_text("// marker\n")
                (root / other).parent.mkdir(exist_ok=True)
                (root / other).write_text(text)
                (app / "build.gradle").write_text("// app build\n")
                (app / "Main.java").write_text("class Main {}")

                projects = JavaAdapter().discover(root, set())

                assert [p.root for p in projects] == [app], (other, extra)

    def test_build_properties_without_sbt_version_is_no_sbt_build(self, tmp_path):
        app = tmp_path / "app"
        app.mkdir()
        (tmp_path / "settings.gradle").write_text("include 'app'\n")
        (tmp_path / "project").mkdir()
        (tmp_path / "project" / "build.properties").write_bytes(
            b"# sbt.version=1.9.9 caf\xe9\nsbt.versions=1\n"
        )
        (app / "build.gradle").write_text("// app build\n")
        (app / "Main.java").write_text("class Main {}")

        projects = JavaAdapter().discover(tmp_path, set())

        assert [p.root for p in projects] == [tmp_path]

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
    def test_unreadable_build_properties_yields_the_subprojects(self, tmp_path):
        """scip-java aborts on a `project/build.properties` it cannot read, so
        the root is not taken as one Gradle build."""
        app = tmp_path / "app"
        app.mkdir()
        (tmp_path / "settings.gradle").write_text("include 'app'\n")
        (tmp_path / "project").mkdir()
        properties = tmp_path / "project" / "build.properties"
        properties.write_text("# no key\n")
        properties.chmod(0)
        (app / "build.gradle").write_text("// app build\n")
        (app / "Main.java").write_text("class Main {}")

        try:
            projects = JavaAdapter().discover(tmp_path, set())
        finally:
            properties.chmod(0o644)

        assert [p.root for p in projects] == [app]

    def test_malformed_escape_in_build_properties_yields_the_subprojects(
        self, tmp_path
    ):
        """scip-java aborts on a `\\u` escape not followed by four hex digits
        outside a comment, so the root is not taken as one Gradle build; an
        escaped backslash, a comment and a whole escape abort nothing."""
        app = tmp_path / "app"
        app.mkdir()
        (tmp_path / "settings.gradle").write_text("include 'app'\n")
        (tmp_path / "project").mkdir()
        properties = tmp_path / "project" / "build.properties"
        (app / "build.gradle").write_text("// app build\n")
        (app / "Main.java").write_text("class Main {}")

        properties.write_text("# \\uzz\nname=\\\\u12 \\u00e9\n")
        assert [p.root for p in JavaAdapter().discover(tmp_path, set())] == [tmp_path]
        properties.write_text("name=\\u12\n")
        assert [p.root for p in JavaAdapter().discover(tmp_path, set())] == [app]

    def test_a_directory_is_a_project_where_scip_java_runs(self, tmp_path):
        """scip-java, given no `--build-tool`, builds with the one tool it
        detects and refuses none or more than one; discovery follows it."""
        layouts = {
            ("pom.xml",): True,
            ("build.gradle",): True,
            ("build.sbt",): True,
            ("build.sc",): True,
            ("WORKSPACE",): True,
            ("scip-java.json",): True,
            ("settings.gradle.kts",): False,
            ("pom.xml", "build.gradle"): False,
            ("build.sbt", "gradlew"): False,
            ("WORKSPACE", "settings.gradle"): False,
        }
        for markers, runs in layouts.items():
            for nested in (False, True):
                root = tmp_path / "-".join(markers) / str(nested)
                base = root / "app" if nested else root
                base.mkdir(parents=True)
                for marker in markers:
                    (base / marker).write_text("x\n")
                (base / "Main.java").write_text("class Main {}")

                projects = JavaAdapter().discover(root, set())

                assert [p.root for p in projects] == ([base] if runs else []), (
                    markers,
                    nested,
                )

    def test_a_directory_named_like_a_marker_is_no_marker(self, tmp_path):
        """scip-java detects a build tool by a regular file, so beside a
        directory named `pom.xml` it runs Gradle and takes Gradle arguments."""
        (tmp_path / "pom.xml").mkdir()
        (tmp_path / "build.gradle").write_text("// gradle\n")
        (tmp_path / "Main.java").write_text("class Main {}")

        (project,) = JavaAdapter().discover(tmp_path, set())
        spec = JavaAdapter().build_command(
            project, tmp_path / "out.scip", AdapterConfig()
        )

        assert "--" in spec.argv

    def test_a_directory_named_settings_gradle_is_no_settings_file(self, tmp_path):
        """scip-java and Gradle read a settings file, so a directory of that
        name does not make the root one multi-project build."""
        app = tmp_path / "app"
        app.mkdir()
        (tmp_path / "settings.gradle").mkdir()
        (tmp_path / "gradlew").write_text("#!/bin/sh\n")
        (app / "build.gradle").write_text("// app build\n")
        (app / "Main.java").write_text("class Main {}")

        projects = JavaAdapter().discover(tmp_path, set())

        assert [p.root for p in projects] == [app]

    def test_a_maven_root_with_modules_is_one_project(self, tmp_path):
        """A root POM that lists modules builds them: the root is the one
        project, not each module."""
        (tmp_path / "pom.xml").write_text(
            "<project><modules><module>core</module></modules></project>"
        )
        core = tmp_path / "core"
        core.mkdir()
        (core / "pom.xml").write_text("<project></project>")
        (core / "Main.java").write_text("class Main {}")

        projects = JavaAdapter().discover(tmp_path, set())

        assert [p.root for p in projects] == [tmp_path]

    def test_modules_are_read_from_the_pom_itself(self, tmp_path):
        """Modules in a comment or only in a profile are not built by the
        root, so each is a project of its own."""
        core = tmp_path / "core"
        core.mkdir()
        (core / "pom.xml").write_text("<project></project>")
        (core / "Main.java").write_text("class Main {}")
        pom = tmp_path / "pom.xml"
        pom.write_text(
            "<project><!-- <modules><module>core</module></modules> -->"
            "<profiles><profile><modules><module>core</module></modules>"
            "</profile></profiles></project>"
        )
        assert [p.root for p in JavaAdapter().discover(tmp_path, set())] == [core]
        pom.write_text(
            '<project xmlns="http://maven.apache.org/POM/4.0.0">'
            "<modules><module>core</module></modules></project>"
        )
        assert [p.root for p in JavaAdapter().discover(tmp_path, set())] == [tmp_path]

    def test_wrapper_alone_marks_the_root(self, tmp_path):
        (tmp_path / "gradlew").write_text("#!/bin/sh\n")
        (tmp_path / "Main.java").write_text("class Main {}")

        projects = JavaAdapter().discover(tmp_path, set())

        assert [p.root for p in projects] == [tmp_path]

    def test_excluded_directory_ignored(self, tmp_path):
        pkg = tmp_path / "target"
        pkg.mkdir()
        (pkg / "pom.xml").write_text("<project/>")
        (pkg / "Foo.java").write_text("class Foo {}")

        projects = JavaAdapter().discover(tmp_path, {"target"})

        assert projects == []

    def test_hidden_directory_ignored(self, tmp_path):
        pkg = tmp_path / ".cache"
        pkg.mkdir()
        (pkg / "pom.xml").write_text("<project/>")
        (pkg / "Foo.java").write_text("class Foo {}")

        projects = JavaAdapter().discover(tmp_path, set())

        assert projects == []

    def test_no_jvm_sources_skipped(self, tmp_path):
        pkg = tmp_path / "empty"
        pkg.mkdir()
        (pkg / "pom.xml").write_text("<project/>")
        # Only a README, no java/kt/scala files.
        (pkg / "README.md").write_text("empty")

        projects = JavaAdapter().discover(tmp_path, set())

        assert projects == []


class TestJavaAdapterBuildCommand:
    def test_basic_command(self, tmp_path):
        project = DiscoveredProject(
            name="myapp", root=tmp_path / "myapp", language="java"
        )
        spec = JavaAdapter().build_command(
            project, tmp_path / "myapp.scip", AdapterConfig()
        )

        assert spec.argv[0] == "scip-java"
        assert spec.argv[1] == "index"
        assert "--output" in spec.argv
        assert str(tmp_path / "myapp.scip") in spec.argv
        assert spec.cwd == tmp_path / "myapp"
        assert spec.env_extras == {}

    def test_compat_init_script_included_by_default(self, tmp_path):
        """Descry ships a Gradle init-script that strips -Werror from
        JavaCompile tasks so scip-java works out of the box on Kafka-style
        conservative builds. Verify it gets added on a Gradle project."""
        (tmp_path / "build.gradle").write_text("// gradle")
        project = DiscoveredProject(name="myapp", root=tmp_path, language="java")
        spec = JavaAdapter().build_command(
            project, tmp_path / "out.scip", AdapterConfig()
        )

        init_flags = [a for a in spec.argv if a.startswith("--init-script=")]
        assert len(init_flags) == 1
        assert init_flags[0].endswith("_java_compat.init.gradle")
        # Must also include the scip-java default Gradle tasks since `--`
        # replaces scip-java's implicit task list.
        assert "clean" in spec.argv
        assert "scipPrintDependencies" in spec.argv
        assert "scipCompileAll" in spec.argv

    def test_no_gradle_args_on_a_maven_project(self, tmp_path):
        """A Maven project gets no Gradle arguments."""
        (tmp_path / "pom.xml").write_text("<project/>")
        project = DiscoveredProject(name="myapp", root=tmp_path, language="java")
        spec = JavaAdapter().build_command(
            project, tmp_path / "out.scip", AdapterConfig()
        )

        assert "--" not in spec.argv

    def test_no_gradle_args_on_an_sbt_project(self, tmp_path):
        (tmp_path / "build.sbt").write_text('scalaVersion := "2.13.14"')
        project = DiscoveredProject(name="myapp", root=tmp_path, language="java")
        spec = JavaAdapter().build_command(
            project, tmp_path / "out.scip", AdapterConfig()
        )

        assert "--" not in spec.argv

    def test_gradle_args_follow_scip_java_detection(self, tmp_path):
        """The init-script is passed for each marker scip-java's
        GradleBuildTool accepts, and not where a `pom.xml` beside it makes
        scip-java refuse the directory."""
        for marker in (
            "settings.gradle",
            "gradlew",
            "build.gradle",
            "build.gradle.kts",
        ):
            for pom in (False, True):
                root = tmp_path / marker / str(pom)
                root.mkdir(parents=True)
                (root / marker).write_text("// marker\n")
                if pom:
                    (root / "pom.xml").write_text("<project/>")
                project = DiscoveredProject(name="myapp", root=root, language="java")
                spec = JavaAdapter().build_command(
                    project, root / "out.scip", AdapterConfig()
                )

                assert ("--" in spec.argv) is not pom, (marker, pom)


class TestJavaAdapterKotlinDslPreflight:
    """scip-java's SemanticdbGradlePlugin can't index Kotlin DSL builds
    that use precompiled-script-plugins (kotlin-dsl plugin). The pre-flight
    check raises with a clear reason instead of letting scip-java fail
    with an opaque 1100-byte stderr dump.
    """

    def test_kotlin_dsl_in_buildSrc_kts_raises(self, tmp_path):
        # buildSrc/build.gradle.kts with `kotlin-dsl` (kotlinx.coroutines pattern)
        (tmp_path / "build.gradle.kts").write_text("// root build\n")
        (tmp_path / "buildSrc").mkdir()
        (tmp_path / "buildSrc" / "build.gradle.kts").write_text(
            "plugins {\n    `kotlin-dsl`\n}\n"
        )
        project = DiscoveredProject(name="kx", root=tmp_path, language="java")

        with pytest.raises(RuntimeError, match="kotlin-dsl"):
            JavaAdapter().build_command(project, tmp_path / "out.scip", AdapterConfig())

    def test_kotlin_dsl_via_includeBuild_raises(self, tmp_path):
        # ktor pattern: includeBuild('build-settings-logic') in settings.gradle.kts
        # and that included build applies kotlin-dsl
        (tmp_path / "gradlew").write_text("#!/bin/sh\n")
        (tmp_path / "settings.gradle.kts").write_text(
            'rootProject.name = "ktor"\nincludeBuild("build-settings-logic")\n'
        )
        (tmp_path / "build-settings-logic").mkdir()
        (tmp_path / "build-settings-logic" / "build.gradle.kts").write_text(
            'plugins { id("org.gradle.kotlin.kotlin-dsl") }\n'
        )
        project = DiscoveredProject(name="ktor", root=tmp_path, language="java")

        with pytest.raises(RuntimeError, match="kotlin-dsl"):
            JavaAdapter().build_command(project, tmp_path / "out.scip", AdapterConfig())

    def test_kotlin_dsl_groovy_dsl_buildSrc_raises(self, tmp_path):
        # The Groovy-DSL `apply plugin: 'kotlin-dsl'` variant
        (tmp_path / "build.gradle").write_text("// root build\n")
        (tmp_path / "buildSrc").mkdir()
        (tmp_path / "buildSrc" / "build.gradle").write_text(
            "apply plugin: 'kotlin-dsl'\n"
        )
        project = DiscoveredProject(name="weird", root=tmp_path, language="java")

        with pytest.raises(RuntimeError, match="kotlin-dsl"):
            JavaAdapter().build_command(project, tmp_path / "out.scip", AdapterConfig())

    def test_plain_buildSrc_does_not_raise(self, tmp_path):
        # Java buildSrc without kotlin-dsl is fine
        (tmp_path / "buildSrc").mkdir()
        (tmp_path / "buildSrc" / "build.gradle.kts").write_text(
            "plugins {\n    java\n}\n"
        )
        project = DiscoveredProject(name="ok", root=tmp_path, language="java")
        spec = JavaAdapter().build_command(
            project, tmp_path / "out.scip", AdapterConfig()
        )
        assert spec.argv[0] == "scip-java"

    def test_kotlin_dsl_buildSrc_on_a_maven_project_is_not_skipped(self, tmp_path):
        # The plugin runs inside Gradle and a Maven project gets no
        # init-script, so a Gradle build script beside `pom.xml` is no
        # reason to lose SCIP for it.
        (tmp_path / "pom.xml").write_text("<project/>")
        (tmp_path / "buildSrc").mkdir()
        (tmp_path / "buildSrc" / "build.gradle.kts").write_text(
            "plugins {\n    `kotlin-dsl`\n}\n"
        )
        project = DiscoveredProject(name="maven", root=tmp_path, language="java")
        spec = JavaAdapter().build_command(
            project, tmp_path / "out.scip", AdapterConfig()
        )

        assert "--" not in spec.argv

    def test_no_buildSrc_does_not_raise(self, tmp_path):
        # Plain Maven project — no buildSrc at all
        (tmp_path / "pom.xml").write_text("<project></project>")
        project = DiscoveredProject(name="maven", root=tmp_path, language="java")
        spec = JavaAdapter().build_command(
            project, tmp_path / "out.scip", AdapterConfig()
        )
        assert spec.argv[0] == "scip-java"

    def test_includeBuild_outside_root_ignored(self, tmp_path):
        # Defensive: a settings.gradle.kts that includeBuild()s a path
        # outside the project root must not be probed.
        (tmp_path / "settings.gradle.kts").write_text(
            'includeBuild("../somewhere-else")\n'
        )
        project = DiscoveredProject(name="ok", root=tmp_path, language="java")
        spec = JavaAdapter().build_command(
            project, tmp_path / "out.scip", AdapterConfig()
        )
        assert spec.argv[0] == "scip-java"


class TestJavaAdapterDescriptorParsing:
    def test_simple_method(self):
        adapter = JavaAdapter()
        assert adapter.parse_descriptors("org/example/Foo#bar().") == ["Foo", "bar"]

    def test_nested_inner_class_method(self):
        adapter = JavaAdapter()
        assert adapter.parse_descriptors("org/example/Foo#Inner#method().") == [
            "Foo",
            "Inner",
            "method",
        ]

    def test_enum_constant(self):
        adapter = JavaAdapter()
        assert adapter.parse_descriptors("com/ex/Color#RED.") == ["Color", "RED"]

    def test_method_with_param_types(self):
        adapter = JavaAdapter()
        result = adapter.parse_descriptors(
            "org/apache/kafka/clients/producer/KafkaProducer#send()."
        )
        assert result == ["KafkaProducer", "send"]

    def test_field(self):
        adapter = JavaAdapter()
        assert adapter.parse_descriptors("com/ex/Config#DEFAULT_TIMEOUT.") == [
            "Config",
            "DEFAULT_TIMEOUT",
        ]


class TestScipJavaResolution:
    """ScipIndex._extract_name must dispatch scip-java schemes through the
    JavaAdapter's parse_descriptors, not the fallback Rust-style parser.
    """

    def test_extract_name_for_java_method(self):
        idx = ScipIndex([])
        sym = (
            "semanticdb maven maven/org.apache.kafka/kafka-clients 4.4.0 "
            "org/apache/kafka/clients/producer/KafkaProducer#send()."
        )
        assert idx._extract_name(sym) == "send"

    def test_extract_name_for_inner_class(self):
        idx = ScipIndex([])
        sym = "semanticdb maven . . com/example/Foo#Inner#method()."
        assert idx._extract_name(sym) == "method"

    def test_extract_name_for_enum_constant(self):
        idx = ScipIndex([])
        sym = "semanticdb maven . . com/example/Color#RED."
        assert idx._extract_name(sym) == "RED"

    def test_local_symbol_returns_none(self):
        idx = ScipIndex([])
        assert idx._extract_name("local foo") is None

    def test_java_stats_bucket_exists(self):
        """Registry-driven _resolution_stats should include a "java" bucket
        now that JavaAdapter is registered."""
        idx = ScipIndex([])
        assert "java" in idx._resolution_stats
        assert idx._resolution_stats["java"] == {"attempted": 0, "resolved": 0}


class TestScipJavaHealthStatus:
    def test_java_listed_in_scip_status(self):
        from descry.scip import support

        support.reset_scip_state()
        status = support.get_scip_status()
        assert "scip-java" in status["indexers"]
