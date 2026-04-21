"""Tests for the scip-java adapter: discovery, command building, and
SCIP symbol resolution through the shared ScipIndex dispatch path."""

from __future__ import annotations

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

    def test_extra_args_forwarded_after_separator(self, tmp_path):
        project = DiscoveredProject(name="myapp", root=tmp_path, language="java")
        config = AdapterConfig(extra_args=("--batch-mode", "-DskipTests"))
        spec = JavaAdapter().build_command(project, tmp_path / "out.scip", config)

        assert "--" in spec.argv
        # Extras are appended AFTER the shipped compat flags
        # (--init-script=..., clean, scipPrintDependencies, scipCompileAll).
        assert spec.argv[-2:] == ["--batch-mode", "-DskipTests"]

    def test_compat_init_script_included_by_default(self, tmp_path):
        """Descry ships a Gradle init-script that strips -Werror from
        JavaCompile tasks so scip-java works out of the box on Kafka-style
        conservative builds. Verify it gets added."""
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

    def test_toolchain_sets_jvm_version_env(self, tmp_path):
        project = DiscoveredProject(name="myapp", root=tmp_path, language="java")
        config = AdapterConfig(toolchain="17")
        spec = JavaAdapter().build_command(project, tmp_path / "out.scip", config)

        assert spec.env_extras == {"JVM_VERSION": "17"}


class TestJavaAdapterKotlinDslPreflight:
    """scip-java's SemanticdbGradlePlugin can't index Kotlin DSL builds
    that use precompiled-script-plugins (kotlin-dsl plugin). The pre-flight
    check raises with a clear reason instead of letting scip-java fail
    with an opaque 1100-byte stderr dump.
    """

    def test_kotlin_dsl_in_buildSrc_kts_raises(self, tmp_path):
        # buildSrc/build.gradle.kts with `kotlin-dsl` (kotlinx.coroutines pattern)
        (tmp_path / "buildSrc").mkdir()
        (tmp_path / "buildSrc" / "build.gradle.kts").write_text(
            "plugins {\n    `kotlin-dsl`\n}\n"
        )
        project = DiscoveredProject(name="kx", root=tmp_path, language="java")

        import pytest

        with pytest.raises(RuntimeError, match="kotlin-dsl"):
            JavaAdapter().build_command(project, tmp_path / "out.scip", AdapterConfig())

    def test_kotlin_dsl_via_includeBuild_raises(self, tmp_path):
        # ktor pattern: includeBuild('build-settings-logic') in settings.gradle.kts
        # and that included build applies kotlin-dsl
        (tmp_path / "settings.gradle.kts").write_text(
            'rootProject.name = "ktor"\nincludeBuild("build-settings-logic")\n'
        )
        (tmp_path / "build-settings-logic").mkdir()
        (tmp_path / "build-settings-logic" / "build.gradle.kts").write_text(
            'plugins { id("org.gradle.kotlin.kotlin-dsl") }\n'
        )
        project = DiscoveredProject(name="ktor", root=tmp_path, language="java")

        import pytest

        with pytest.raises(RuntimeError, match="kotlin-dsl"):
            JavaAdapter().build_command(project, tmp_path / "out.scip", AdapterConfig())

    def test_kotlin_dsl_groovy_dsl_buildSrc_raises(self, tmp_path):
        # The Groovy-DSL `apply plugin: 'kotlin-dsl'` variant
        (tmp_path / "buildSrc").mkdir()
        (tmp_path / "buildSrc" / "build.gradle").write_text(
            "apply plugin: 'kotlin-dsl'\n"
        )
        project = DiscoveredProject(name="weird", root=tmp_path, language="java")

        import pytest

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
