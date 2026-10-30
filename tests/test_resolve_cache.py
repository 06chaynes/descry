"""Re-resolving only the files a change can affect: after each change the
graph is the one a full build gives, and the files no change reaches keep
their resolution."""

import logging
import textwrap

import pytest

from descry._parse_cache import ParseCache
from descry.generate import CodeGraphBuilder


def _graph(root, cache_dir, caplog):
    cache = ParseCache(cache_dir, key_path=cache_dir.parent / "key")
    builder = CodeGraphBuilder(root, parse_cache=cache)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="descry.generate"):
        builder.process_directory()
        builder.resolve_references()
    kept = [r.message for r in caplog.records if r.message.startswith("Reused")]
    return builder, int(kept[0].split()[4]) if kept else 0


def _write(root, files):
    for name, text in files.items():
        path = root / name
        if text is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")


UNTOUCHED = {"z.py": "def alone():\n    return 1\n\n\ndef calls():\n    alone()\n"}


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            {
                "a.py": "from b import B\n\ndef run(b: B):\n    b.go()\n",
                "b.py": "class B:\n    pass\n",
            },
            {"b.py": "class B:\n    def go(self):\n        pass\n"},
            id="a-method-another-file-calls-is-added",
        ),
        pytest.param(
            {
                "a.py": "from b import helper\n\ndef run():\n    helper()\n",
                "b.py": "def helper():\n    pass\n",
            },
            {"b.py": "def other():\n    pass\n"},
            id="a-definition-another-file-calls-is-removed",
        ),
        pytest.param(
            {"a.py": "def run(x):\n    x.shared()\n"},
            {"c.py": "class C:\n    def shared(self):\n        pass\n"},
            id="a-new-file-defines-a-name-another-file-calls",
        ),
        pytest.param(
            {
                "a.py": "from b import make\n\ndef run():\n    make().go()\n",
                "b.py": "class P:\n    def go(self):\n        pass\n\n\nclass Q:\n"
                "    def go(self):\n        pass\n\n\ndef make() -> P:\n    return P()\n",
            },
            {
                "b.py": "class P:\n    def go(self):\n        pass\n\n\nclass Q:\n"
                "    def go(self):\n        pass\n\n\ndef make() -> Q:\n    return Q()\n"
            },
            id="a-return-type-another-file-chains-through-changes",
        ),
        pytest.param(
            {
                "a.ts": "import { f } from './b';\nexport function run() { f(); }\n",
                "b.ts": "export { f } from './c';\n",
                "c.ts": "export function f() {}\n",
            },
            {
                "b.ts": "export { f } from './d';\n",
                "d.ts": "export function f() {}\n",
            },
            id="a-re-export-another-module-imports-through-moves",
        ),
    ],
)
def test_a_change_re_resolves_what_it_reaches(tmp_path, caplog, before, after):
    root = tmp_path / "p"
    _write(root, {**UNTOUCHED, **before})
    _graph(root, tmp_path / "c", caplog)
    _write(root, after)
    again, kept = _graph(root, tmp_path / "c", caplog)
    full, _ = _graph(root, tmp_path / "fresh", caplog)
    assert again.nodes == full.nodes
    assert again.edges == full.edges
    assert kept >= 1


@pytest.mark.parametrize(
    "files",
    [
        {"a.py": "from z import alone\n\nalone()\n"},
        {
            "a.ts": "import { f } from './b';\nexport function run() { f(); }\n",
            "b.ts": "export function f() {}\n",
        },
        {
            "a.cs": "using B;\nnamespace A { class Run { void Go() { new Helper().Help(); } } }\n",
            "b.cs": "namespace B { public class Helper { public void Help() {} } }\n",
        },
        {
            "a.java": "package a;\nimport b.Helper;\nclass Run { void go() { new Helper().help(); } }\n",
            "b/Helper.java": "package b;\npublic class Helper { public void help() {} }\n",
        },
        {
            "src/main.rs": "mod b;\nfn main() { b::help(); }\n",
            "src/b.rs": "pub fn help() {}\n",
            "Cargo.toml": '[package]\nname = "p"\n',
        },
        {
            "a.go": "package main\nfunc main() { help() }\n",
            "b.go": "package main\nfunc help() {}\n",
        },
        {
            "a.rb": "require_relative 'b'\nHelper.new.help\n",
            "b.rb": "class Helper\n  def help; end\nend\n",
        },
        {
            "a.php": "<?php\nfunction run() { (new Helper())->help(); }\n",
            "b.php": "<?php\nclass Helper { function help() {} }\n",
        },
        {
            "a.c": '#include "b.h"\nint main(void) { help(); return 0; }\n',
            "b.h": "#ifndef B_H\n#define B_H\nvoid help(void);\n#endif\n",
            "b.c": "void help(void) {}\n",
        },
        {
            "a.cpp": '#include "b.hpp"\nint main() { Helper().help(); }\n',
            "b.hpp": "#pragma once\nstruct Helper { void help() {} };\n",
        },
        {
            "lib/a.dart": "import 'b.dart';\nvoid main() { Helper().help(); }\n",
            "lib/b.dart": "class Helper { void help() {} }\n",
        },
    ],
    ids=[
        "python",
        "typescript",
        "csharp",
        "java",
        "rust",
        "go",
        "ruby",
        "php",
        "c",
        "cpp",
        "dart",
    ],
)
def test_an_unchanged_tree_keeps_every_file_s_resolution(tmp_path, caplog, files):
    root = tmp_path / "p"
    _write(root, {**UNTOUCHED, **files})
    first, _ = _graph(root, tmp_path / "c", caplog)
    again, kept = _graph(root, tmp_path / "c", caplog)
    assert kept == 1 + sum(not name.endswith(".toml") for name in files)
    assert again.edges == first.edges


def test_with_scip_every_file_is_resolved_again(tmp_path, caplog):
    root = tmp_path / "p"
    _write(root, UNTOUCHED)
    cache = ParseCache(tmp_path / "c", key_path=tmp_path / "key")
    for _ in range(2):
        builder = CodeGraphBuilder(root, parse_cache=cache)
        builder.process_directory()
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="descry.generate"):
            builder.resolve_references(scip_index=_NoScip())
        assert not any(r.message.startswith("Reused") for r in caplog.records)


class _NoScip:
    """A SCIP index that names nothing."""

    def definition_at(self, *_args, **_kwargs):
        return None

    def outside_at(self, *_args):
        return False

    def resolve(self, *_args):
        return None


def test_moving_a_definition_s_lines_keeps_its_callers_resolution(tmp_path, caplog):
    """A caller reads what a definition is, not where it stands: an edit
    that only moves the definition's lines re-resolves its own file alone."""
    root = tmp_path / "p"
    _write(
        root,
        {
            "a.py": "from b import helper\n\ndef run():\n    helper()\n",
            "b.py": "def helper():\n    pass\n",
        },
    )
    _graph(root, tmp_path / "c", caplog)
    _write(root, {"b.py": "import os\n\n\ndef helper():\n    pass\n"})
    again, kept = _graph(root, tmp_path / "c", caplog)
    full, _ = _graph(root, tmp_path / "fresh", caplog)
    assert again.edges == full.edges
    assert kept == 1


@pytest.mark.parametrize("flag", ["made", "removed"])
def test_a_file_an_msbuild_condition_probes_re_resolves(tmp_path, caplog, flag):
    """`Exists()` asks the directory, which no table holds: making or
    removing the file it names re-resolves the C# that reads it."""
    root = tmp_path / "p"
    files = {
        "Directory.Build.props": "<Project><ItemGroup>"
        '<Using Include="Lib" Condition="Exists(\'flag.txt\')" />'
        "</ItemGroup></Project>\n",
        "app/app.csproj": '<Project Sdk="Microsoft.NET.Sdk"></Project>\n',
        "app/Run.cs": "namespace App { class Run { void Go() { new Helper().Help(); } } }\n",
        "lib/Helper.cs": "namespace Lib { public class Helper { public void Help() {} } }\n",
    }
    _write(root, {**files, **({"flag.txt": "x\n"} if flag == "removed" else {})})
    first, _ = _graph(root, tmp_path / "c", caplog)
    _write(root, {"flag.txt": "x\n" if flag == "made" else None})
    again, _ = _graph(root, tmp_path / "c", caplog)
    full, _ = _graph(root, tmp_path / "fresh", caplog)
    assert again.edges == full.edges != first.edges
