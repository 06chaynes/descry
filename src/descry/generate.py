"""Generate codebase knowledge graph with optional SCIP-based type-aware resolution."""

import contextlib
import functools
import hashlib
import json
import logging
import os
import posixpath
import re
import signal
import stat
import threading
import tomllib
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from descry._files import create_sibling
from descry._parse_cache import ParseCache, _decode, _encode
from descry._parsing import _split_callee, logger
from descry._reads import Reads, print_of, record, whole_print
from descry._streams import write_utf8
from descry.proto_parser import ProtoParser
from descry.python_parser import PythonParser
from descry.query import file_of
from descry.resolver import _EXTERNAL, _JSTS_EXTENSIONS, CallResolver, _ancestors
from descry.rust_parser import RustParser
from descry.ts_parser import TSParser

# Configure logging for consistent error reporting: other libraries' at
# WARNING, descry's own at the level DESCRY_LOG_LEVEL gives its logger, so
# the HTTP requests a model's download makes are not shown as progress.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


_RUBY_EXTENSIONS = (".rb", ".rake", ".gemspec")


_DOTNET_EXTENSIONS = (".cs", ".vb")


_CLANG_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx")


# Manifests read for the package names call resolution needs. They do not
# become graph nodes.
_MANIFESTS = ("go.mod", "Cargo.toml", "package.json", "pubspec.yaml")
# The .NET project files that say which namespaces a project's C# files have
# without a `using`
_DOTNET_PROJECT_FILES = (".csproj", ".props", ".targets")


def _msbuild_file(content):
    """``[SDK, entries]`` of an MSBuild file, targets left out (what a
    target sets, it sets when it runs): the SDK its `<Project Sdk>` or
    `<Sdk Name>` names, and see `_msbuild_entries`. No entries for a file
    that is no XML, or declares a DTD, which MSBuild never does."""
    try:
        if "<!DOCTYPE" in content:
            raise ElementTree.ParseError
        project = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return [None, []]
    named = [project.get("Sdk")]
    named += [e.get("Name") for e in project.iter() if _msbuild_tag(e) == "Sdk"]
    sdk = next((n.split("/")[0].strip() for n in named if n), None)
    return [sdk, _msbuild_entries(project, [])]


def _msbuild_tag(element):
    """The name of an MSBuild element, without its XML namespace."""
    return element.tag.rpartition("}")[2] if isinstance(element.tag, str) else ""


def _msbuild_entries(element, around):
    """In document order, each property the MSBuild ``element`` sets,
    ``["property", name, value, conditions]``, each `<Using>` item it
    includes or removes, ``["Include" or "Remove", name, [alias, static],
    conditions]``, and each file it imports, ``["import", path,
    conditions]``: ``conditions``, which must all hold, are those of the
    element, of its groups and ``around``; a `<Choose>` gives its first
    `<When>` whose condition holds, else its `<Otherwise>`."""
    entries = []
    for part in element:
        kind = _msbuild_tag(part)
        held = around + [c for c in [part.get("Condition")] if c is not None]
        if kind == "Import":
            entries.append(["import", part.get("Project", ""), held])
        elif kind == "ImportGroup":
            entries += _msbuild_entries(part, held)
        elif kind == "Choose":
            passed = []
            for branch in part:
                if _msbuild_tag(branch) not in ("When", "Otherwise"):
                    continue
                test = [c for c in [branch.get("Condition")] if c]
                ruled_out = [f"!({c})" for c in passed]
                entries += _msbuild_entries(branch, held + ruled_out + test)
                passed += test
        elif kind == "PropertyGroup":
            for found in part:
                test = [c for c in [found.get("Condition")] if c is not None]
                value = (found.text or "").strip()
                entries.append(["property", _msbuild_tag(found), value, held + test])
        elif kind == "ItemGroup":
            for found in part:
                if _msbuild_tag(found) != "Using":
                    continue
                # Metadata as attributes or as child elements
                metadata = {_msbuild_tag(c): (c.text or "").strip() for c in found}
                metadata.update(found.attrib)
                static = metadata.get("Static", "").lower() == "true"
                how = [metadata.get("Alias", ""), static]
                test = [c for c in [metadata.get("Condition")] if c is not None]
                for item in ("Include", "Remove"):
                    for name in metadata.get(item, "").split(";"):
                        if name.strip():
                            entries.append([item, name.strip(), how, held + test])
    return entries


_GO_MODULE_RE = re.compile(r"^module\s+(\S+)", re.MULTILINE)


_PUBSPEC_NAME_RE = re.compile(r"^name:\s*['\"]?([\w$]+)", re.MULTILINE)


_CARGO_PACKAGE_RE = re.compile(
    r'^\[package\][^\[]*?^name\s*=\s*"([^"]+)"', re.MULTILINE | re.DOTALL
)


def _cargo_renames(content):
    """``(own, shared)`` of the Cargo manifest ``content``: the crate each
    dependency key its code writes names, `tui = { package = "helix-tui" }`
    as `helix_tui` and None for one `workspace = true` takes from the
    workspace, and the renames of its `[workspace.dependencies]`."""
    try:
        manifest = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return {}, {}

    def renames(tables):
        return {
            key.replace("-", "_"): spec.get("package", "").replace("-", "_") or None
            for table in tables
            for kind in ("dependencies", "dev-dependencies", "build-dependencies")
            for key, spec in table.get(kind, {}).items()
            if isinstance(spec, dict) and ("package" in spec or spec.get("workspace"))
        }

    targets = manifest.get("target", {}).values()
    shared = renames([manifest.get("workspace", {})])
    return renames([manifest, *targets]), {k: v for k, v in shared.items() if v}


# A TypeScript or JavaScript project file, read for its import path aliases
_TSCONFIG_RE = re.compile(r"[jt]sconfig(?:\.[\w.-]+)?\.json")


# What a tsconfig may hold that JSON may not, next to the strings to keep:
# comments, then commas before a closing bracket.
_JSON_STRING = r'"(?:[^"\\]|\\.)*"'


_JSONC_COMMENT_RE = re.compile(_JSON_STRING + r"|//[^\n]*|/\*.*?\*/", re.DOTALL)


_JSONC_COMMA_RE = re.compile(_JSON_STRING + r"|,(?=\s*[}\]])")


def _loads_jsonc(text: str):
    """Parse JSON that may have comments and trailing commas."""
    for extra in (_JSONC_COMMENT_RE, _JSONC_COMMA_RE):
        text = extra.sub(lambda m: m.group() if m.group()[0] == '"' else "", text)
    return json.loads(text)


# Extensions the indexer has a parser for. Besides these it opens only the
# manifests in _MANIFESTS and the project files _TSCONFIG_RE names.
_INDEXED_EXTENSIONS = (
    ".py",
    ".rs",
    ".proto",
    ".java",
    ".go",
    *_RUBY_EXTENSIONS,
    ".php",
    *_DOTNET_EXTENSIONS,
    *_CLANG_EXTENSIONS,
    ".dart",
    ".svelte",
    *_JSTS_EXTENSIONS,
)


# Try to import SCIP support for type-aware call resolution
try:
    from descry.scip.cache import ScipCacheManager
    from descry.scip.parser import ScipIndex
    from descry.scip.support import get_scip_status, scip_available

    SCIP_SUPPORT_LOADED = True
except ImportError:
    SCIP_SUPPORT_LOADED = False
    logger.debug("SCIP support not available")


def definition_spans(nodes):
    """(file, name) -> ``(first line, last line, node ID)`` of each
    definition among ``nodes``."""
    spans = {}
    for node in nodes:
        metadata = node["metadata"]
        if first := metadata.get("lineno"):
            key = (node["id"].split("::")[0][5:], metadata.get("name"))
            last = metadata.get("end_lineno") or first
            spans.setdefault(key, []).append((first, last, node["id"]))
    return spans


def defined_at(spans, path, line, name):
    """The node of the innermost definition of ``name`` in the file ``path``
    whose lines hold ``line``, where an indexer records it: a template,
    decorator or annotation can start a definition lines before its name.
    None where two of the same lines do."""
    held = [span for span in spans.get((path, name), ()) if span[0] <= line <= span[1]]
    innermost = [s for s in held if s[:2] == max(held, key=lambda s: (s[0], -s[1]))[:2]]
    return innermost[0][2] if len(innermost) == 1 else None


def scip_target(spans, found, name):
    """The node of the definition of ``name`` that SCIP found, ``(node ID,
    file, lines)``: the one whose lines hold each of its lines, and for a
    definition at one line SCIP's own ID where none does."""
    node_id, path, lines = found
    held = {defined_at(spans, path, line, name) for line in lines}
    if len(lines) == 1:
        return held.pop() or node_id
    return held.pop() if len(held) == 1 else None


class _OverTime(BaseException):
    """The block `_time_limit` runs has used up its time. No `Exception`, so
    that no handler inside the block takes it for a failure of its own."""


@contextlib.contextmanager
def _time_limit(seconds):
    """Raise `_OverTime` in the block once it has run for ``seconds``. There
    is no limit for 0, off the main thread, or on a platform without
    interval timers."""
    main = threading.current_thread() is threading.main_thread()
    if not (seconds and main and hasattr(signal, "setitimer")):
        yield
        return

    def over(_signum, _frame):
        raise _OverTime

    previous = signal.signal(signal.SIGALRM, over)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class _Numbered(list):
    """A list numbered from ``base``: the edges of one file, whose indexes
    the origin paths that name a call by them are, as no other file's."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def __len__(self):
        return self.base + super().__len__()

    def __bool__(self):
        return super().__len__() > 0

    def _at(self, index):
        return index - self.base if index is not None and index >= 0 else index

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = slice(self._at(index.start), self._at(index.stop), index.step)
            return list.__getitem__(self, index)
        return list.__getitem__(self, self._at(index))


@functools.lru_cache(maxsize=65536)
def _file_language(file_part):
    """The language of the file ``file_part``, by its extension."""
    if file_part.endswith(".rs"):
        return "rust"
    elif file_part.endswith((*_JSTS_EXTENSIONS, ".svelte")):
        return "typescript"
    elif file_part.endswith(".py"):
        return "python"
    elif file_part.endswith(".proto"):
        return "proto"
    elif file_part.endswith(".java"):
        return "java"
    elif file_part.endswith(".go"):
        return "go"
    elif file_part.endswith(_RUBY_EXTENSIONS):
        return "ruby"
    elif file_part.endswith(".php"):
        return "php"
    elif file_part.endswith(_DOTNET_EXTENSIONS):
        return "dotnet"
    elif file_part.endswith(_CLANG_EXTENSIONS):
        return "clang"
    elif file_part.endswith(".dart"):
        return "dart"
    return "unknown"


def _merged(into, held):
    """Add the container ``held`` to the container ``into`` of its kind: a
    list's entries after, a set's in, a dict's by key, merging the
    containers two values under one key are."""
    if isinstance(into, list):
        into.extend(held)
    elif isinstance(into, set):
        into.update(held)
    else:
        for key, value in held.items():
            if key in into and isinstance(value, (dict, list, set)):
                _merged(into[key], value)
            else:
                into[key] = value


_SLOT_SIZE = 10**7
_SLOTS = 2**62 // _SLOT_SIZE


# What a builder derives from what parsing a file adds, or holds of its own
_DERIVED_STATE = {
    "node_registry",
    "nodes_by_id",
    "excluded_dirs",
    "edge_index",
    "slots",
    "parse_entries",
}

# The counts of how the calls were resolved
_STATS = ("scip", "regex", "self_type", "unresolved", "external")

# What resolution reads of a builder that no other file's resolution changes
_UNRECORDED = {
    "nodes",
    "edges",
    "edge_index",
    "slots",
    "excluded_dirs",
    "parse_entries",
}

# What a builder holds of its own, not what parsing a file adds
_OWN_STATE = _DERIVED_STATE | {"nodes", "edges"}


class CodeGraphBuilder:
    def __init__(
        self,
        root_dir,
        excluded_dirs=None,
        rel_base=None,
        max_file_bytes=0,
        max_parse_seconds=0,
        respect_gitignore=False,
        parse_cache=None,
    ):
        self.root_dir = Path(root_dir).resolve()
        # The `ParseCache` files' parses are reused from, if any, and file
        # ID -> the entry of each file parsed whole, which names its parse
        self.parse_cache = parse_cache
        self.parse_entries = {}
        # Larger source files are skipped; 0 walks files of any size.
        self.max_file_bytes = max_file_bytes
        # A file is parsed for no longer than this; 0 sets no limit.
        self.max_parse_seconds = max_parse_seconds
        # Whether the files git ignores are left out.
        self.respect_gitignore = respect_gitignore
        # What node-ID paths are relative to: the project root, which is
        # above root_dir when only a subtree is indexed.
        self.rel_base = Path(rel_base).resolve() if rel_base else self.root_dir
        self.excluded_dirs = excluded_dirs or {
            "target",
            "node_modules",
            "dist",
            "build",
            "docs",
            "coverage",
            "demo-output",
            ".beads",
        }
        self.serial_base = 0  # see `serial`
        # The number an origin path names a call by -> its edge's index
        self.edge_index = {}
        self.slots = set()  # the files' slots of numbers, see `_shell`
        self.nodes = []
        self.edges = []
        self.node_registry = set()
        self.nodes_by_id = {}
        # For CallResolver: file ID -> {local name: import target},
        # go.mod directory -> module path, Cargo package name -> directory,
        # and the npm packages the project defines and depends on.
        self.imports = {}
        # Python file ID -> {name: "import" where an import rebinds the
        # file's definition of it, "either" where which binds is not told}
        self.import_binding = {}
        # Python file ID that writes `__all__` -> the names it lists, None
        # where they are not written out
        self.py_all = {}
        self.go_modules = {}
        self.rust_crates = {}
        self.cargo_renames = {}  # manifest directory -> see `_cargo_renames`
        self.js_packages = {}  # name -> (directory, `exports` of package.json)
        self.js_entries = {}  # directory -> its package.json's main, types, typings
        self.js_imports = {}  # directory -> its package.json's `imports`
        self.js_module_folders = (
            set()
        )  # directories whose package.json is `type: module`
        self.module = False  # the file being parsed is an ES module by its package
        self.js_dependencies = set()
        # tsconfig path -> (configs it extends, `baseUrl`, `paths`,
        # `customConditions`, `moduleResolution`, `module`), as written
        self.ts_configs = {}
        # JS/TS file or object ID -> [(name exported or held, "*" for each
        # name of what it names or "=" for the file's whole, module specifier
        # or "" for a name written in the file, the name there, the scope
        # the name is read in)]
        self.reexports = {}
        # JS files that write CommonJS exports, whose `require` gives those
        self.commonjs = set()
        # JS/TS file ID -> {local name: the name its module exports it as},
        # for an import that renames what it binds, ``default`` for one of
        # the default and ``=`` for one of the module whole
        self.import_names = {}
        # JS/TS file ID -> each name a `require` binds -> the line it does on
        self.required_names = {}
        # JS/TS file ID -> {name an `import()` binds: its line}
        self.dynamic_names = {}
        # directory of a package.json -> the `typescript` version it depends on
        self.ts_versions = {}
        # JS/TS constant ID -> the ID of the constant its value, a name,
        # names where the constant is declared
        self.js_aliases = {}
        # (defining file or class ID, fixture name) -> (function ID, origin
        # path of the value the fixture returns or yields, if it has one,
        # whether it yields it)
        self.fixtures = {}
        # Names declared as type aliases, which are not graph nodes, and for
        # TypeScript ``(file ID, name) -> (type parameters, text)`` of each
        # written on one line
        self.type_aliases = set()
        self.ts_aliases = {}
        # C/C++ file ID -> the function names a prototype in it declares:
        # the definition may sit in any file linked in.
        self.declares = {}
        # C/C++ function, by its ID without the file, -> ``[defaulted, most,
        # types, scope]`` of each declaration of it without a body: the
        # positions of the parameters it gives defaults, the arguments it
        # takes at most (None for any number), their types, and the ID it is
        # declared in
        self.prototypes = {}
        # Type or variant ID -> {field name or position: its declared type};
        # Python module or class ID -> {variable: origin paths of its values},
        # function ID -> {`()`: origin paths of what it returns}
        self.fields = {}
        # Type ID -> the standard traits it derives, as its language's
        # standard-library data names them
        self.derived = {}
        self.rust_aliases = {}  # Rust file ID -> names of the aliases it declares
        # Rust file ID -> {name: crate} of each `extern crate crate as name`
        self.rust_externs = {}
        # (Rust type ID, trait its impl block implements, name) -> the type
        # that block's associated type of that name is
        self.rust_associated = {}
        # (Rust module ID, alias name) -> {(its type parameters, the type it
        # names)}, one for each declaration that differs, as `#[cfg]` writes
        self.rust_alias_types = {}
        # Rust macro name -> (file ID, the functions its body writes) of each
        # `macro_rules!` of the name
        self.rust_macros = {}
        # (Rust scope ID, name) -> the paths the `use`s of the scope that bind
        # the name to different items give, one for each `#[cfg]`
        self.rust_use_alternatives = {}
        # Rust item ID -> its visibility where less than `pub` (see
        # `_rust_visibility`), and (scope ID, name) -> that of a `use`
        self.rust_visibility = {}
        self.rust_use_visibility = {}
        # ID of a Rust impl block's owner -> the macros called among its items
        self.rust_macro_uses = {}
        # Rust file ID -> (ID, `#[path]` or None) of each `mod name;` in it
        self.rust_mods = {}
        self.dart_packages = {}  # Dart package name -> its directory
        # Dart file ID -> the (`import`, `export` or `part`, URIs, prefix,
        # names shown, names hidden) of each of its directives
        self.dart_directives = {}
        # C/C++ alias name -> [(ID of the scope that declares it, its type)]
        self.c_aliases = {}
        # C++ class ID -> name -> the base its `using` declaration names
        self.c_using = {}
        # C++ class ID -> the names of the member functions it declares
        self.c_declared = {}
        # C++ class template, by its ID without the file, -> the default
        # argument of each parameter, "" for none, that each declaration of
        # it without a body gives
        self.c_defaults = {}
        self.js_bound = {}  # JS/TS file ID -> the names its declarations bind
        # JS/TS files that are modules: what they declare is named in them
        self.js_modules = set()
        # JS/TS or Python class ID -> {attribute its methods assign to the
        # instance, `this.name = ...` or `self.name = ...`: the values, the
        # text written in JS/TS, the origin paths in Python}
        self.assigned = {}
        # Python class ID -> {name its body binds last by an assignment: the
        # name of the function called on its `def` that is the value, if any}
        self.py_class_assigned = {}
        # Function ID -> its signatures as written, one for each overload:
        # the bound of each type parameter, the parameters, the return type
        # and, by argument position, the declared types of the parameters
        # of the function it is passed there
        self.signatures = {}
        # `[N]`, which an origin path names an array literal by -> the origin
        # path of each of its elements, for an object literal those of its
        # properties by name; None where they are not known
        self.literals = {}
        # `{N}` -> (an object or a tuple type a parameter is declared, its
        # file's ID)
        self.parameter_types = {}
        # `%N` -> (the origin paths of the values assigned to a Dart variable
        # promoted to a type, None for one not known, that type's origin
        # path, the origin path of what the variable was before)
        self.narrowings = {}
        # The literal a function returns -> the function, in whose scope the
        # names in it are read
        self.literal_scopes = {}
        # Directory -> the `global using` directives of its C# files, which
        # hold for every file below it, as ``imports`` has a file's own
        self.dotnet_global_usings = {}
        # .NET project, `Directory.Build.props` or `.targets` file -> its
        # SDK and its properties and `<Using>` items (see `_msbuild_file`)
        self.dotnet_projects = {}
        # Directory -> `dotnet_usings` entries of each `global using
        # static` in a C# file of it, which hold for its project
        self.dotnet_global_statics = {}
        # C/C++ file ID -> {macro name: (line, whether a `#define` or an
        # `#undef`) of each of its directives, in order}
        self.c_macro_lines = {}
        self.c_once = set()  # C/C++ files an include guard reads once
        # C# file ID -> (the path a `using` names, the namespace whose body
        # it is written in, "" for the file's, whether it sits in an `#if`
        # arm, whether `static` names a type, whether `global::` roots the
        # path, the alias it gives the path or None, the first and last line
        # of the namespace declaration it is written in or None) of each
        # `using`
        self.dotnet_usings = {}

    def serial(self, table):
        """A number for the next entry of ``table`` that no other file's
        entries have: this file's base and the entries before."""
        return self.serial_base + len(table)

    def _empty(self):
        """A builder of this one's settings, empty."""
        return CodeGraphBuilder(
            self.root_dir,
            self.excluded_dirs,
            self.rel_base,
            self.max_file_bytes,
            self.max_parse_seconds,
            self.respect_gitignore,
        )

    def _shell(self, rel_path):
        """An empty builder that the file ``rel_path`` is parsed into on its
        own."""
        shell = self._empty()
        # A slot of its own, of ten million numbers below 2**62
        slot = int.from_bytes(hashlib.sha256(rel_path.encode()).digest()[:8], "big")
        slot %= _SLOTS
        while slot in self.slots:
            slot = (slot + 1) % _SLOTS
        self.slots.add(slot)
        shell.serial_base = slot * _SLOT_SIZE
        shell.edges = _Numbered(shell.serial_base)
        return shell

    def _state(self):
        """What parsing a file into this builder gave, as `_restored` takes
        it back."""
        state = {
            name: held
            for name, held in vars(self).items()
            if isinstance(held, (dict, list, set)) and name not in _DERIVED_STATE
        }
        state["edges"] = list(list.__iter__(self.edges))
        state["serial_base"] = self.serial_base
        return state

    def _restored(self, state):
        """A builder holding ``state``, what `_state` gave, unless its slot
        is another file's already or it holds what no builder does."""
        shell = self._empty()
        base = state.pop("serial_base", None)
        if not isinstance(base, int) or base // _SLOT_SIZE in self.slots:
            return None
        shell.serial_base, shell.edges = base, _Numbered(base)
        list.extend(shell.edges, state.pop("edges", ()))
        for name, held in state.items():
            if name in _DERIVED_STATE or type(getattr(shell, name, None)) is not type(
                held
            ):
                return None
            setattr(shell, name, held)
        shell.node_registry = {node["id"] for node in shell.nodes}
        shell.nodes_by_id = {node["id"]: node for node in shell.nodes}
        self.slots.add(base // _SLOT_SIZE)
        return shell

    def _merge(self, shell):
        """Add what one file parsed into ``shell`` gives, as parsing it into
        this builder after the files before would."""
        for node in shell.nodes:
            if node["id"] not in self.node_registry:
                self.nodes.append(node)
                self.node_registry.add(node["id"])
                self.nodes_by_id[node["id"]] = node
        for at in range(list.__len__(shell.edges)):
            self.edge_index[shell.edges.base + at] = len(self.edges) + at
        self.edges.extend(list.__iter__(shell.edges))
        for name, held in vars(shell).items():
            if name not in _OWN_STATE and isinstance(held, (dict, list, set)):
                _merged(getattr(self, name), held)

    def add_node(self, node_id, node_type, **metadata):
        if node_id not in self.node_registry:
            node = {"id": node_id, "type": node_type, "metadata": metadata}
            self.nodes.append(node)
            self.node_registry.add(node_id)
            self.nodes_by_id[node_id] = node

    def add_edge(self, source, target, relation, metadata=None):
        edge = {"source": source, "target": target, "relation": relation}
        if metadata:
            edge["metadata"] = metadata
        self.edges.append(edge)

    def js_module(self, path):
        """Whether the package.json nearest above the file ``path`` makes
        its JavaScript an ES module, `"type": "module"`."""
        folders = (above or "." for above in _ancestors(posixpath.dirname(path)))
        nearest = next((f for f in folders if f in self.js_entries), None)
        return nearest in self.js_module_folders

    def get_rel_path(self, path):
        # Always use forward-slash separators in node IDs so the graph is
        # cross-platform (node IDs generated on Windows remain readable and
        # queryable on Linux/macOS, and vice versa).
        return path.relative_to(self.rel_base).as_posix()

    def _git_ignored(self) -> set[str]:
        """Paths under the root that git ignores, relative to it: files, and
        directories ignored whole. Empty outside a git repository, without
        git, and where ignored files are to be indexed."""
        if not self.respect_gitignore:
            return set()
        from descry.git_history import GitError, GitHistoryAnalyzer

        listing = "ls-files -z --others --ignored --exclude-standard --directory"
        try:
            listed = GitHistoryAnalyzer(self.root_dir)._run_git(listing.split())
        except (GitError, OSError):
            return set()
        return {path.rstrip("/") for path in listed.split("\0") if path}

    def process_directory(self):
        ignored = self._git_ignored()

        def kept(root, names):
            """``names`` in directory ``root`` without those git ignores."""
            under = Path(root).relative_to(self.root_dir)
            return [name for name in names if (under / name).as_posix() not in ignored]

        def contained(path):
            """Whether ``path`` resolves inside the project root, so a symlink
            out of it is not followed; one that escapes is logged."""
            try:
                real = path.resolve(strict=False)
            except OSError:
                return False
            if real == self.root_dir or self.root_dir in real.parents:
                return True
            logger.warning("Skipping %s — escapes project root via symlink", real)
            return False

        for root, dirs, files in os.walk(self.root_dir, followlinks=False):
            dirs.sort()
            # A directory's manifests first: they tell how its files parse.
            files.sort(key=lambda name: (name not in _MANIFESTS, name))
            # Filter by name first (excluded dirs, dotfiles, the build output
            # of a .NET or an Eclipse project), and by what git ignores.
            # `bin` holds sources in Dart and Rust.
            built = any(
                f.endswith((".csproj", ".vbproj", ".fsproj")) or f == ".classpath"
                for f in files
            )
            dirs[:] = [
                d
                for d in kept(root, dirs)
                if not d.startswith(".")
                and d not in self.excluded_dirs
                and not (built and d == "bin")
            ]
            dirs[:] = [d for d in dirs if contained(Path(root) / d)]
            for file in kept(root, files):
                if not (
                    file.endswith(_INDEXED_EXTENSIONS)
                    or file in _MANIFESTS
                    or file.endswith(_DOTNET_PROJECT_FILES)
                    or _TSCONFIG_RE.fullmatch(file)
                ):
                    continue
                file_path = Path(root) / file
                if not contained(file_path):
                    continue
                rel_path = self.get_rel_path(file_path)
                # Each file is read whole, so one over the limit is skipped
                # before it is opened, as is anything but a regular file: a
                # FIFO would hold the read open and a device never end it.
                try:
                    st = file_path.stat()
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                size = st.st_size
                if self.max_file_bytes and size > self.max_file_bytes:
                    logger.warning(
                        "Skipping %s — %.1f MiB exceeds [code_files] max_file_mb",
                        rel_path,
                        size / (1024 * 1024),
                    )
                    continue
                # O_NOFOLLOW on the final open so a post-check symlink swap
                # can't redirect us outside the project root, and O_NONBLOCK
                # so one to a FIFO doesn't wait. Mirrors the api_source
                # hardening in web/server.py.
                try:
                    flags = (
                        os.O_RDONLY
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_NONBLOCK", 0)
                    )
                    fd = os.open(str(file_path), flags)
                    with os.fdopen(fd, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except OSError as e:
                    logger.warning(f"Cannot read {rel_path}: {e.strerror}")
                    continue
                except ValueError as e:
                    logger.warning(f"Error reading {rel_path}: {e}")
                    continue
                # A file one parser chokes on, or does not finish in time,
                # must not abort the whole index.
                module = rel_path.endswith(".js") and self.js_module(rel_path)
                cached = self.parse_cache and self.parse_cache.load(
                    rel_path, content, module
                )
                try:
                    shell = cached and self._restored(cached)
                except Exception:  # noqa: BLE001 — an entry is never trusted to restore
                    shell = None
                if shell:
                    self.parse_entries[f"FILE:{rel_path}"] = self.parse_cache.entry(
                        rel_path, content, module
                    )
                    self._merge(shell)
                    continue
                shell, parsed = self._shell(rel_path), False
                shell.module = module
                try:
                    with _time_limit(self.max_parse_seconds):
                        shell._parse(file, file_path, rel_path, content)
                    parsed = True
                except _OverTime:
                    logger.warning(
                        "Stopped parsing %s after %g s ([code_files] "
                        "max_parse_seconds); it is indexed as far as it got",
                        rel_path,
                        self.max_parse_seconds,
                    )
                except Exception as e:  # noqa: BLE001 — boundary over every parser
                    logger.warning(f"Failed to parse {rel_path}: {e!r}")
                # A parse cut short is not kept: a later one may finish.
                if parsed and self.parse_cache:
                    self.parse_cache.store(rel_path, content, module, shell._state())
                    self.parse_entries[f"FILE:{rel_path}"] = self.parse_cache.entry(
                        rel_path, content, module
                    )
                self._merge(shell)
        if self.parse_cache:
            self.parse_cache.prune(whole=self.root_dir == self.rel_base)

    def _parse(self, file, file_path, rel_path, content):
        """Parse ``content``, the text of ``file``, with its language's parser."""
        if file.endswith(".py"):
            PythonParser(self).parse(file_path, rel_path, content)
        elif file.endswith(".rs"):
            RustParser(self).parse(file_path, rel_path, content)
        elif file.endswith(".proto"):
            ProtoParser(self).parse(file_path, rel_path, content)
        elif file.endswith(".java"):
            from descry.java_parser import JavaParser

            JavaParser(self).parse(file_path, rel_path, content)
        elif file.endswith(".go"):
            from descry.go_parser import GoParser

            GoParser(self).parse(file_path, rel_path, content)
        elif file == "go.mod":
            if match := _GO_MODULE_RE.search(content):
                folder = rel_path.rpartition("/")[0]
                self.go_modules[folder] = match.group(1)
        elif file == "Cargo.toml":
            folder = rel_path.rpartition("/")[0]
            if match := _CARGO_PACKAGE_RE.search(content):
                self.rust_crates[match.group(1).replace("-", "_")] = folder
            self.cargo_renames[folder] = _cargo_renames(content)
        elif file.endswith(_DOTNET_PROJECT_FILES):
            self.dotnet_projects[rel_path] = _msbuild_file(content)
        elif file == "pubspec.yaml":
            if match := _PUBSPEC_NAME_RE.search(content):
                folder = rel_path.rpartition("/")[0]
                self.dart_packages[match.group(1)] = folder
        elif file == "package.json":
            manifest = json.loads(content)
            folder = rel_path.rpartition("/")[0]
            exports = manifest.get("exports")
            self.js_packages[manifest.get("name")] = (folder, exports)
            self.js_entries[folder or "."] = [
                manifest[k] for k in ("main", "types", "typings") if k in manifest
            ]
            if isinstance(manifest.get("imports"), dict):
                self.js_imports[folder or "."] = manifest["imports"]
            if manifest.get("type") == "module":
                self.js_module_folders.add(folder or ".")
            for key, value in manifest.items():
                if key.endswith("ependencies") and isinstance(value, dict):
                    self.js_dependencies.update(value)
                    if isinstance(value.get("typescript"), str):
                        self.ts_versions[folder or "."] = value["typescript"]
        elif _TSCONFIG_RE.fullmatch(file):
            config = _loads_jsonc(content)
            options = config.get("compilerOptions") or {}
            extends = config.get("extends")
            if isinstance(extends, str):
                extends = [extends]
            base, paths = options.get("baseUrl"), options.get("paths")
            self.ts_configs[rel_path] = (
                [e for e in extends or () if isinstance(e, str)],
                base if isinstance(base, str) else None,
                {
                    str(pattern): [t for t in targets if isinstance(t, str)]
                    for pattern, targets in paths.items()
                    if isinstance(targets, list)
                }
                if isinstance(paths, dict)
                else None,
                [c for c in conditions if isinstance(c, str)]
                if isinstance(conditions := options.get("customConditions"), list)
                else None,
                *(
                    value.lower() if isinstance(value, str) else None
                    for value in (
                        options.get("moduleResolution"),
                        options.get("module"),
                    )
                ),
            )
        elif file.endswith(_RUBY_EXTENSIONS):
            from descry.ruby_parser import RubyParser

            RubyParser(self).parse(file_path, rel_path, content)
        elif file.endswith(".php"):
            from descry.php_parser import PhpParser

            PhpParser(self).parse(file_path, rel_path, content)
        elif file.endswith(_DOTNET_EXTENSIONS):
            from descry.dotnet_parser import DotnetParser

            DotnetParser(self).parse(file_path, rel_path, content)
        elif file.endswith(_CLANG_EXTENSIONS):
            from descry.clang_parser import ClangParser

            ClangParser(self).parse(file_path, rel_path, content)
        elif file.endswith(".dart"):
            from descry.dart_parser import DartParser

            DartParser(self).parse(file_path, rel_path, content)
        elif file.endswith(_JSTS_EXTENSIONS):
            TSParser(self).parse(file_path, rel_path, content)
        elif file.endswith(".svelte"):
            # Iterate all <script> blocks and shift each block's
            # parsed linenos by its offset within the .svelte file.
            pos = 0
            while True:
                start_tag = content.find("<script", pos)
                if start_tag == -1:
                    break
                gt = content.find(">", start_tag)
                if gt == -1:
                    break
                close = content.find("</script>", gt)
                if close == -1:
                    break
                body = content[gt + 1 : close]
                # Lines before the script body (the character after '>').
                line_offset = content.count("\n", 0, gt + 1)
                before = len(self.nodes)
                edges_before = len(self.edges)
                TSParser(self).parse(file_path, rel_path, body, line_offset=line_offset)
                # Shift newly-added node and edge linenos by the offset.
                for node in self.nodes[before:]:
                    meta = node.get("metadata", {})
                    for k in ("lineno", "end_lineno"):
                        v = meta.get(k)
                        if isinstance(v, int):
                            meta[k] = v + line_offset
                for edge in self.edges[edges_before:]:
                    if "lineno" in edge.get("metadata", {}):
                        edge["metadata"]["lineno"] += line_offset
                pos = close + len("</script>")

    def _get_language(self, node_id: str) -> str:
        """The language of the node ``node_id``, by its file's extension."""
        # Node ID format: FILE:path/to/file.ext::Symbol
        if not node_id.startswith("FILE:"):
            return "unknown"
        return _file_language(node_id.split("::")[0][5:])

    def resolve_references(self, scip_index=None):
        """Resolve REF: targets to actual node IDs.

        A CALLS reference goes to SCIP first (type-aware) and then to
        CallResolver, which pins it to a definition, leaves it unresolved,
        or drops it as external. A base type is the one CallResolver finds
        its name to stand for where the subtype is declared. Other
        references are matched by name within the source's language:
        qualified name (Type::method), then same file, then any match.

        Args:
            scip_index: Optional ScipIndex for type-aware resolution
        """
        call_resolver = CallResolver(self)
        spans = definition_spans(self.nodes)
        # (file, line, name) -> how many calls of the name the line makes
        named_calls = Counter(
            (
                e["source"].split("::")[0][5:],
                e.get("metadata", {}).get("lineno"),
                _split_callee(e["target"][4:])[1],
            )
            for e in self.edges
            if e["relation"] == "CALLS" and e["target"].startswith("REF:")
        )

        # With kept parses, what each file's calls resolve to is kept with
        # the keys of the tables it read, and reused while its parse and
        # those keys hold what they held. SCIP's indexes change every run.
        cache = self.parse_cache if scip_index is None and self.parse_entries else None
        reads = cache and Reads()
        if reads:
            record(
                call_resolver, "resolver.", reads, CallResolver.UNRECORDED, {"nodes"}
            )
            record(self, "builder.", reads, _UNRECORDED)
        kept = self._kept_resolutions(cache, reads)

        edges_to_remove = set()
        file_stats, file_reads = {}, {}
        resolving = None  # the file whose calls are being resolved
        for idx, edge in enumerate(self.edges):
            target = edge["target"]
            if (file_id := edge["source"].split("::")[0]) != resolving:
                resolving = file_id
                call_resolver.begin_file()
                stats = file_stats[file_id] = dict.fromkeys(_STATS, 0)
                if reads:
                    file_reads[file_id] = reads.start()
            if file_id in kept:
                continue
            if target.startswith("REF:"):
                ref_name = target.replace("REF:", "")
                source_id = edge["source"]
                source_lang = self._get_language(source_id)

                resolved = None
                resolution_source = None
                is_call = edge["relation"] == "CALLS"
                metadata = edge.get("metadata", {})
                origin = metadata.pop("origin", None)
                column = metadata.pop("col", None)
                within = metadata.pop("within", ())
                arguments = metadata.pop("args", None)
                lambdas = metadata.pop("lambdas", None)
                call_resolver.call_guards = metadata.get("guards")

                # A Rust trait written with a path is the type the path
                # leads to; one the path does not place is read by its name.
                placed = None
                if not is_call and source_lang == "rust" and origin:
                    placed, origin = call_resolver.type_at(origin, source_id), None
                if placed == _EXTERNAL:
                    edges_to_remove.add(idx)
                    continue
                if placed:
                    resolved, resolution_source = placed, "regex"
                # The parser writes a PHP type by its full name, which no
                # other type of that name answers to.
                elif not is_call and source_lang == "php":
                    resolved = call_resolver.type_in_scope(ref_name, source_id)
                    if not resolved:
                        continue
                    resolution_source = "regex"

                # Strategy 0: SCIP lookup (highest accuracy), for each
                # language with a SCIP adapter.
                if scip_index and source_lang not in ("proto", "unknown"):
                    source_file = file_of(source_id)
                    lineno = metadata.get("lineno", 0)
                    if is_call:
                        # A call is taken from SCIP only where SCIP saw it; a
                        # name match is left to CallResolver.
                        named = (source_file, lineno, _split_callee(ref_name)[1])
                        found = scip_index.definition_at(
                            ref_name,
                            source_file,
                            lineno,
                            column,
                            alone=named_calls[named] == 1,
                            within=within,
                        )
                        scip_resolved = found and scip_target(
                            spans, found, _split_callee(ref_name)[1]
                        )
                        # An indexer that names a method that cannot take
                        # what the call passes did not bind the call.
                        if scip_resolved in self.node_registry and not (
                            call_resolver.takes(
                                scip_resolved, arguments, ref_name, origin
                            )
                        ):
                            scip_resolved = None
                        # What SCIP saw go to code outside the project is
                        # no call of a project definition of that name.
                        if not found and scip_index.outside_at(
                            ref_name, source_file, lineno
                        ):
                            resolution_source = _EXTERNAL
                    else:
                        scip_resolved = scip_index.resolve(
                            ref_name, source_file, lineno, column
                        )
                    # SCIP spells a symbol's path its own way; an ID that is
                    # no node of this graph would leave the edge dangling.
                    if scip_resolved in self.node_registry:
                        # Validate cross-crate resolutions to prevent false positives
                        source_crate = (
                            source_file.split("/")[0] if "/" in source_file else ""
                        )
                        resolved_file = scip_resolved.split("::")[0].replace(
                            "FILE:", ""
                        )
                        target_crate = (
                            resolved_file.split("/")[0] if "/" in resolved_file else ""
                        )

                        # If cross-crate, verify ref_name looks related to the target
                        # This prevents spurious matches like EnvFilter -> RequestFilter
                        if (
                            source_crate
                            and target_crate
                            and source_crate != target_crate
                        ):
                            target_name = scip_resolved.rpartition("::")[2]
                            # Only accept if ref_name contains the target name or vice versa
                            if target_name and target_name.lower() in ref_name.lower():
                                resolved = scip_resolved
                                resolution_source = "scip"
                        else:
                            # Same crate or can't determine - accept the resolution
                            resolved = scip_resolved
                            resolution_source = "scip"

                if is_call:
                    if not resolved and resolution_source != _EXTERNAL:
                        site = metadata.get("site", {})
                        # Where the call is written: its line, and its column
                        # in characters (the last of SCIP's encodings)
                        at = metadata.get("lineno")
                        chars = (
                            column[-1] if isinstance(column, list | tuple) else column
                        )
                        written_at = at and (at, chars or 0)
                        resolved, resolution_source = call_resolver.resolve(
                            ref_name,
                            source_id,
                            origin,
                            arguments,
                            lambdas=lambdas,
                            passed=site.get("arguments"),
                            borrows=site.get("borrows"),
                            receiver=site.get("receiver"),
                            on=site.get("on"),
                            in_block=site.get("block", False),
                            shadowed=site.get("shadowed", False),
                            throws=site.get("throws", False),
                            line=written_at,
                        )
                    call_resolver.keep(idx, resolved)
                    if resolution_source == _EXTERNAL:
                        edges_to_remove.add(idx)
                        stats["external"] += 1
                        continue
                elif not resolved and edge["relation"] == "INHERITS":
                    resolved = call_resolver.base_type(idx)
                    if not resolved:
                        # A base the standard library's data names where it
                        # is written is from outside; any other the project
                        # does not place stays open.
                        if call_resolver.outside_base(idx):
                            edges_to_remove.add(idx)
                        continue
                    resolution_source = "regex"

                if resolved:
                    edge["target"] = resolved
                    edge["metadata"] = metadata
                    metadata["resolution_source"] = resolution_source
                    if is_call:
                        edge["relation"] = "CALLS_RESOLVED"
                        stats[resolution_source] += 1
                elif is_call:
                    stats["unresolved"] += 1

        if reads:
            reads.stop()
        # Each file's edges without the non-project calls filtered, a kept
        # file's as they were kept
        by_file = {}
        for idx, edge in enumerate(self.edges):
            held = by_file.setdefault(edge["source"].split("::")[0], [])
            if idx not in edges_to_remove:
                held.append(edge)
        for file_id, held in kept.items():
            if file_id in by_file:
                by_file[file_id], file_stats[file_id] = held["edges"], held["stats"]
        self.edges = [edge for edges in by_file.values() for edge in edges]
        # What a Ruby maker call gives where a project constant shadows the
        # maker is no type or member; a constant assigned it stays one.
        unmade = {
            node["id"]
            for node in self.nodes
            if node["type"] != "Constant"
            and any(
                node["id"] == u or node["id"].startswith(f"{u}::")
                for u in call_resolver.unmade
            )
        }
        if unmade:
            self.nodes = [node for node in self.nodes if node["id"] not in unmade]
            self.node_registry -= unmade
            self.edges = [
                edge
                for edge in self.edges
                if edge["source"] not in unmade and edge["target"] not in unmade
            ]
        if edges_to_remove:
            logger.info(
                f"Filtered {len(edges_to_remove)} non-project edges during resolution"
            )
        if kept:
            logger.info(f"Reused the resolution of {len(kept)} unchanged files")
        if cache:
            self._keep_resolutions(cache, reads, by_file, file_stats, file_reads, kept)

        stats = {key: sum(held[key] for held in file_stats.values()) for key in _STATS}
        total = (
            stats["scip"] + stats["regex"] + stats["self_type"] + stats["unresolved"]
        )
        if total > 0:
            scip_pct = 100 * stats["scip"] / total
            regex_pct = 100 * stats["regex"] / total
            self_type_pct = 100 * stats["self_type"] / total
            resolved_total = stats["scip"] + stats["regex"] + stats["self_type"]
            resolved_pct = 100 * resolved_total / total
            logger.info(
                f"CALLS Resolution: {resolved_pct:.0f}% of {total} project calls "
                f"(SCIP: {stats['scip']}/{scip_pct:.0f}%, "
                f"Regex: {stats['regex']}/{regex_pct:.0f}%, "
                f"Self: {stats['self_type']}/{self_type_pct:.0f}%, "
                f"Unresolved: {stats['unresolved']}); "
                f"{stats['external']} external calls not counted"
            )

    def _kept_resolutions(self, cache, reads):
        """File ID -> what `_keep_resolutions` kept of the file, its keys and
        tables read whole as sets, for each file whose parse and every key
        its calls read hold what they held."""
        held = cache and cache.load_resolution()
        if not held:
            return {}
        keys = [_decode(k) for k in held["keys"]]
        changed = {
            at
            for at, (key, kept) in enumerate(zip(keys, held["prints"]))
            if kept is None or print_of(reads, *key) != kept
        }
        whole = {
            t
            for t, kept in held["wholes"].items()
            if kept is None or whole_print(reads, t) != kept
        }
        found = {}
        for file_id, kept in held["files"].items():
            if (
                self.parse_entries.get(file_id) == kept["entry"]
                and changed.isdisjoint(kept["keys"])
                and whole.isdisjoint(kept["whole"])
            ):
                kept["keys"] = {keys[at] for at in kept["keys"]}
                kept["whole"] = set(kept["whole"])
                found[file_id] = kept
        return found

    def _keep_resolutions(self, cache, reads, by_file, file_stats, file_reads, kept):
        """Keep, for each file of a complete parse, its edges as resolved,
        their counts and the keys its calls read, each key once with what
        it holds now."""
        positions = {}  # (table, key) -> where it is kept
        files = {}
        for file_id, edges in by_file.items():
            if file_id in kept:
                keys, whole = kept[file_id]["keys"], kept[file_id]["whole"]
            elif file_id in self.parse_entries:
                keys, whole = file_reads[file_id]
            else:
                continue
            files[file_id] = {
                "entry": self.parse_entries[file_id],
                "keys": sorted(positions.setdefault(k, len(positions)) for k in keys),
                "whole": sorted(whole),
                "edges": edges,
                "stats": file_stats[file_id],
            }
        tables = set().union(*(held["whole"] for held in files.values()))
        try:
            keys = [_encode(k) for k in positions]
        except TypeError:
            return
        cache.store_resolution(
            {
                "files": files,
                "keys": keys,
                "prints": [print_of(reads, *k) for k in positions],
                "wholes": {t: whole_print(reads, t) for t in tables},
            }
        )

    def export(self, output_path, scip_index=None):
        """Export the graph to JSON.

        Args:
            output_path: Path to write the graph JSON
            scip_index: Optional ScipIndex for type-aware resolution
        """
        from descry._graph import CURRENT_SCHEMA, count_callers

        self.resolve_references(scip_index=scip_index)
        count_callers(self.nodes, self.edges)
        # Atomic write: stream to a sibling tmp file then os.replace. A
        # concurrent reader (MCP prewarm, web `/api/health`, another CLI
        # invocation) that stats the destination mid-write would otherwise
        # observe a partial JSON file and raise `json.JSONDecodeError`.
        # os.replace is atomic on POSIX and on Windows >=3.3.
        output_path = Path(output_path)
        fd, tmp_path = create_sibling(output_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schema_version": CURRENT_SCHEMA,
                        "nodes": self.nodes,
                        "edges": self.edges,
                    },
                    f,
                    indent=2,
                )
            os.replace(tmp_path, output_path)
        except Exception:
            # Leave the previous good graph in place if anything fails mid-write.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        logger.info(
            "Graph exported to %s. Stats: %d nodes, %d edges",
            output_path,
            len(self.nodes),
            len(self.edges),
        )


def main():
    """CLI entry point for descry-generate."""
    import argparse

    write_utf8()

    parser = argparse.ArgumentParser(description="Generate codebase knowledge graph")
    parser.add_argument(
        "path",
        nargs="?",
        help="Directory to index (default: the current directory, or the "
        "project root with --project-root)",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root, when PATH is a directory inside it: config, cache "
        "and node-ID paths come from the root (default: PATH)",
    )
    parser.add_argument(
        "--no-scip",
        action="store_true",
        help="Disable SCIP resolution (use regex only)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    # Progress is logged at INFO unless the environment or -v sets a level.
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ["DESCRY_LOG_LEVEL"] = "DEBUG"
    else:
        os.environ.setdefault("DESCRY_LOG_LEVEL", "INFO")

    target = args.path or args.project_root or "."

    from descry.handlers import DescryConfig

    if args.no_scip:
        os.environ["DESCRY_NO_SCIP"] = "1"

    project_root = Path(args.project_root or target).resolve()
    if not Path(target).resolve().is_relative_to(project_root):
        parser.error(f"{target!r} is outside the project root {project_root}")
    # The root comes from the arguments, not from the working directory.
    config = DescryConfig.from_env(project_root)
    if error := config.cache_dir_error():
        parser.error(error)

    cache_dir = config.cache_dir
    if cache_dir.exists() and not cache_dir.is_dir():
        parser.error(f"{cache_dir} is a file, not a directory; the cache needs one")
    graph_path = cache_dir / "codebase_graph.json"
    if graph_path.exists() and not graph_path.is_file():
        parser.error(f"{graph_path} is a directory, not the graph file")
    cache_dir.mkdir(parents=True, exist_ok=True)

    builder = CodeGraphBuilder(
        target,
        excluded_dirs=config.excluded_dirs,
        rel_base=project_root,
        max_file_bytes=config.max_file_bytes,
        max_parse_seconds=config.max_parse_seconds,
        respect_gitignore=config.respect_gitignore,
        parse_cache=ParseCache(
            cache_dir / "parse", config.max_parse_seconds, project=project_root
        ),
    )
    builder.process_directory()

    scip_index = None
    if config.enable_scip and SCIP_SUPPORT_LOADED and scip_available():
        try:
            scip_status = get_scip_status()
            indexers = scip_status.get("indexers", {})
            enabled_indexers = [
                name for name, info in indexers.items() if info.get("available")
            ]
            logger.info(f"SCIP: Enabled ({', '.join(enabled_indexers) or 'none'})")

            cache_manager = ScipCacheManager(
                project_root,
                config.excluded_dirs,
                scip_extra_args=config.scip_extra_args,
                scip_skip_crates=config.scip_skip_crates,
                scip_toolchain=config.scip_rust_toolchain,
                scip_timeout_minutes=config.scip_timeout_minutes,
                cache_dir=config.cache_dir,
            )
            scip_paths = cache_manager.update_all()
            scip_files = list(scip_paths.values())

            if scip_files:
                scip_index = ScipIndex(
                    scip_files, cache_manager.project_dirs(scip_paths), project_root
                )
                stats = scip_index.get_stats()
                logger.info(
                    f"SCIP: Loaded {stats['definitions']} definitions, "
                    f"{stats['unique_names']} unique names from {len(scip_files)} files"
                )
        except Exception as e:  # noqa: BLE001 — boundary over external SCIP indexers; indexing continues regex-only
            logger.warning(f"SCIP: Failed to load ({e}), using regex only")
    elif not config.enable_scip:
        logger.info("SCIP: Disabled")
    elif SCIP_SUPPORT_LOADED:
        logger.info(
            "SCIP: Unavailable (no indexers found — install any of: "
            "rust-analyzer, scip-typescript, scip-python, scip-java, "
            "scip-go, scip-ruby, scip-php, scip-dotnet, scip-clang, "
            "scip-dart)"
        )

    builder.export(str(graph_path), scip_index=scip_index)

    if config.enable_embeddings:
        try:
            from descry.embeddings import EMBEDDINGS_AVAILABLE, SemanticSearcher

            if EMBEDDINGS_AVAILABLE:
                logger.info("Generating embeddings for semantic search...")
                searcher = SemanticSearcher(
                    str(graph_path),
                    cache_dir=str(config.cache_dir),
                    force_rebuild=True,
                    model_name=config.embedding_model,
                    source_root=str(config.project_root),
                )
                logger.info(
                    f"Embeddings generated: {len(searcher.nodes)} nodes indexed"
                )
            else:
                logger.debug(
                    "Embeddings: sentence-transformers not available, skipping"
                )
        except ImportError:
            logger.debug("Embeddings: module not available, skipping")
        except Exception as e:  # noqa: BLE001 — boundary over sentence-transformers/torch; indexing continues without embeddings
            logger.warning(f"Embeddings: Failed to generate ({e})")


if __name__ == "__main__":
    main()
