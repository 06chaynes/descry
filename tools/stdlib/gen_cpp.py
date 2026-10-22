"""Generate cpp.json from Clang's own AST of the standard headers.

usage: gen_cpp.py INCLUDE_DIR OUTPUT

INCLUDE_DIR is the directory Clang's C++ standard headers stand at — the
`libstdc++` include root GCC names, `/usr/include/c++/N`. One translation
unit including the public headers is compiled and its AST dumped as JSON;
every declaration `std` puts in them is walked: the records and class
templates, with the names of their parameters, their bases, their methods
and the member typedefs each declares for itself, the namespace-scope
functions, the variables (`cout`, `endl` is a function), and the type
aliases, including those of the implementation namespaces — those are how
the public spellings a header only writes in an internal spelling
(`__detail::__unique_ptr_t<_Tp>`) are read back as the names they stand
for (`unique_ptr<_Tp>`). No list of names is kept: what is public is what
the AST puts in `std` without a name that starts with `_`, the
implementation namespace `std::__cxx11` of `basic_string` & co. being the
one exception, its contents being the public ones.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import dump_ast, main, new_type, record

# The public headers the translation unit written for the dump includes.
HEADERS = (
    "memory",
    "vector",
    "string",
    "string_view",
    "map",
    "unordered_map",
    "set",
    "unordered_set",
    "deque",
    "list",
    "forward_list",
    "queue",
    "stack",
    "array",
    "span",
    "optional",
    "variant",
    "utility",
    "tuple",
    "functional",
    "algorithm",
    "iterator",
    "iostream",
    "sstream",
    "fstream",
    "istream",
    "ostream",
    "iomanip",
    "numeric",
    "random",
    "chrono",
    "filesystem",
    "regex",
    "thread",
    "mutex",
    "shared_mutex",
    "atomic",
    "condition_variable",
    "future",
    "exception",
    "stdexcept",
    "type_traits",
    "limits",
    "new",
    "initializer_list",
    "bitset",
    "complex",
    "valarray",
    "ratio",
    "bit",
    "concepts",
    "ranges",
    "format",
    "expected",
    "source_location",
    "any",
    "system_error",
    "cstring",
    "cstdlib",
    "cstdio",
    "cctype",
    "cerrno",
    "cassert",
    "cstdint",
    "ctime",
    "climits",
    "cwchar",
    "cstddef",
    "cmath",
    "cfenv",
    "cfloat",
    "cinttypes",
    "clocale",
    "csetjmp",
    "csignal",
    "cstdarg",
    "cuchar",
    "cwctype",
    "codecvt",
    "locale",
    "typeinfo",
    "typeindex",
    "scoped_allocator",
    "execution",
    "barrier",
    "latch",
    "semaphore",
    "syncstream",
    "spanstream",
    "print",
    "mdspan",
    "generator",
    "stop_token",
)

# The namespaces of `std` whose contents are the implementation's, not the
# library's API — any starting `_`, except `__cxx11`, the inline namespace
# the string types are declared in. `__gnu_cxx` is an extension namespace
# entirely, not `std` at all, and is walked under no names.


def _public(name):
    """Whether the name is one of the library's API: a `_`/`__` prefix
    marks a name as the implementation's; operators are spelled `operator`."""
    if not name:
        return False
    if name.startswith("operator"):
        return True
    return not name.startswith("_")


_ABI_TAG = re.compile(r"_+[A-Za-z]?\w*\d+")


def _namespace_public(path):
    """Whether declarations inside the namespace path — `std`, `std::chrono`,
    and the like — are the library's: every namespace after `std` that is
    an implementation namespace (a leading `_`) rules them out, save the
    ABI-tag namespaces (`__cxx11`, `__2824` — digits end them), whose
    contents are the library's declarations under their public names."""
    return not any(
        part.startswith("_") and not _ABI_TAG.fullmatch(part) for part in path[1:]
    )


def _substitute(text, params, arguments):
    """``text`` with the parameter names ``params`` written as
    ``arguments``."""
    for param, argument in zip(params, arguments):
        if param:
            text = re.sub(rf"\b{re.escape(param)}\b", argument, text)
    return text


def _parts(text):
    """The comma-separated parts of ``text``'s brackets."""
    out, depth, start = [], 0, 0
    for at, char in enumerate(text):
        depth += char in "<([{"
        depth -= char in ">)]}"
        if char == "," and depth == 0:
            out.append(text[start:at])
            start = at + 1
    out.append(text[start:])
    return [part.strip() for part in out if part.strip()]


def _rewrite(head, aliases, public):
    """One rewrite of the type spelling ``head`` through ``aliases`` —
    `(name -> (parameters, target))` — or None where ``head`` already
    names a public type of the data or no rewrite applies."""
    if not head.startswith("typename"):
        base_of = head.partition("<")[0].strip()
        if "::" not in base_of and base_of in public:
            return None

    def alias_of(name):
        """``(params, target)`` of the alias ``name`` names, by its full
        spelling, without `std::` first, by its last segment, or — the
        member typedefs of implementation classes are read too — any
        record's member of that name."""
        short = re.sub(r"^std::", "", name)
        found = (
            aliases.get(name) or aliases.get(short) or aliases.get(name.split("::")[-1])
        )
        if found is None and name.startswith("_"):
            found = next(
                (v for k, v in aliases.items() if k.endswith(f"::{name}")),
                None,
            )
        return found

    match = re.match(r"^(typename\s+|class\s+)?([\w:]+)(<(.+)>)?::(\w+)$", head)
    if match:
        # `Owner<args>::member` — a member typedef of `Owner`.
        _, owner, _, argtext, member = match.groups()
        short = owner.split("::")[-1]
        entry = (
            aliases.get(f"{owner}::{member}")
            or aliases.get(f"{re.sub('^std::', '', owner)}::{member}")
            or aliases.get(f"{short}::{member}")
        )
        if (
            member in ("type", "value")
            and f"{short}_t" in aliases
            or (member == "type" and f"{short}_t" in public)
        ):
            # `X<T>::type` is the public `X_t<T>` where that alias exists.
            return f"{short}_t<{argtext}>" if argtext else f"{short}_t"
        if entry:
            params, target = entry
            return _substitute(target, params, _parts(argtext) if argtext else [])
        # `Owner` may itself be an alias the member is read through.
        owner_entry = alias_of(owner)
        if owner_entry:
            params, target = owner_entry
            substituted = _substitute(
                target, params, _parts(argtext) if argtext else []
            )
            return f"{substituted}::{member}"
        return None
    match = re.match(r"^(typename\s+|class\s+)?([\w:]+)(<.*>)$", head)
    if match:
        _, base, args = match.groups()
        entry = alias_of(base)
        if entry:
            params, target = entry
            return _substitute(target, params, _parts(args[1:-1]))
        return None
    base = re.sub(r"^(typename\s+|class\s+)", "", head).strip()
    if re.fullmatch(r"_[A-Z_]\w*", base):
        # A bare type parameter (`_Tp`, `_Alloc`) is not an alias.
        return None
    entry = alias_of(base)
    if entry:
        return entry[1]
    return None


def _returned(written, aliases, public, depth=0):
    """The type a Clang `type` string returns, before its `(...)`; the
    spelling the header writes is rewritten through ``aliases`` — the map
    the typedefs of `std`, its records and its implementation namespaces
    declare — to a fixed point, stopping where the spelling names a public
    name (``public``). The aliases are also followed all the way: where
    they end in a bare type-parameter name (`_Tp` — `move`, `forward`,
    `addressof` write their return with one), that is the answer."""
    if depth > 3:
        return None
    head = written.split("(", 1)[0].strip()
    head = re.sub(r"^(?:const|volatile)\s+", "", head)
    head = re.sub(r"\s*[&*]+\s*$", "", head)
    head = re.sub(r"^(typename|class)\s+", "", head)
    deep = head
    for _ in range(10):
        rewritten = _rewrite(deep, aliases, frozenset())
        if rewritten is None:
            break
        deep = re.sub(r"^(typename|class)\s+", "", rewritten.strip())
        deep = re.sub(r"\s*[&*]+\s*$", "", deep)
    if re.fullmatch(r"_\w+", deep or ""):
        return deep
    for _ in range(10):
        rewritten = _rewrite(head, aliases, public)
        if rewritten is None:
            break
        head = re.sub(r"^(typename|class)\s+", "", rewritten.strip())
        head = re.sub(r"\s*[&*]+\s*$", "", head)
    match = re.match(r"^([\w:]+)(<.*>)$", head)
    if match:
        parts = [
            _returned(f"{part} ()", aliases, public, depth + 1) or part
            for part in _parts(match[2][1:-1])
        ]
        head = f"{match[1]}<{', '.join(parts)}>"
    leaf = head.split("<")[0].split("::")[-1]
    if "(" in head or leaf.startswith("__"):
        # A function spelling or an implementation name is no public type.
        return None
    return head or None


def _cpp_type(params=()):
    """The shape a C++ type's data takes: its parameters, bases, methods,
    statics and the member typedefs (`value_type`) it declares."""
    held = new_type(params)
    held["aliases"] = {}
    return held


class _Build:
    def __init__(self):
        # key -> type entry (params, bases, methods, statics, aliases)
        self.types = {}
        # impl-name and `Type::name` -> the spelling the typedef stands for
        self.aliases = {}
        self.functions = {}
        self.values = set()
        # `std` itself namespaces `chrono`, `filesystem`, `this_thread`…
        self.namespaces = set()
        # leaf -> the spelling a `std` scope typedef stands for
        self.scope_aliases = {}
        # leaf names public to the data: collected as types are walked
        self.public = set()


def _members(record_node, held, build, path, public):
    """Read ``record_node``'s methods, statics and member typedefs into
    ``held``."""
    for member in record_node.get("inner", ()):
        kind = member.get("kind")
        name = member.get("name", "")
        if kind in ("TypedefDecl", "TypeAliasDecl") and name:
            # The class's own member typedefs: `vector`'s `value_type` is
            # `_Tp`, `map`'s `pair<const _Key, _Tp>` — read per class.
            if _public(name):
                held["aliases"][name] = (
                    member.get("type", {}).get("qualType", "") or None
                )
            continue
        if kind not in (
            "CXXMethodDecl",
            "CXXConstructorDecl",
            "CXXConversionDecl",
            "FunctionTemplateDecl",
            "VarDecl",
        ) or not _public(name):
            continue
        if kind == "VarDecl":
            if member.get("storageClass") == "static":
                record(held["statics"], name, member.get("type", {}).get("qualType"))
            continue
        if kind == "FunctionTemplateDecl":
            inner = next(
                (
                    i
                    for i in member.get("inner", ())
                    if i.get("kind") in ("CXXMethodDecl", "CXXConstructorDecl")
                ),
                None,
            )
            if inner is None:
                continue
            member = inner
            name = member.get("name", "")
            if not _public(name):
                continue
        if member.get("kind") == "CXXConstructorDecl":
            continue
        if member.get("kind") == "CXXConversionDecl":
            name = f"operator {member.get('conversionType', {}).get('qualType', '').split('(')[0]}"
        written = member.get("type", {}).get("qualType", "")
        static = member.get("storageClass") == "static"
        returns = _returned(written, build.aliases, public)
        seen = set()
        while (
            returns
            and re.fullmatch(r"\w+", returns)
            and returns in (held.get("aliases") or {})
            and returns not in seen
        ):
            # A return spelled `reference` is the class's own typedef.
            seen.add(returns)
            returns = (
                _returned(f"{held['aliases'][returns]} ()", build.aliases, public)
                or returns
            )
        held["statics" if static else "methods"].setdefault(name, []).append(returns)


def _walk(node, path, build):
    """Walk declarations of the AST under ``path``: records and templates
    of `std` go into the types table with their members, typedefs and
    aliases; functions into the functions table; variables the headers
    declare into the values table. Names and namespaces of the
    implementation are skipped but for their aliases — those read the
    spelling a public name is written in."""
    kind = node.get("kind")
    name = node.get("name", "")
    inner = node.get("inner", ())
    if kind == "LinkageSpecDecl":
        # `extern "C"` wraps the C-compatible names of `std`.
        for item in inner:
            _walk(item, path, build)
        return
    if kind == "NamespaceDecl":
        next_path = [*path, name]
        if not _namespace_public(next_path):
            # The implementation namespaces' aliases are collected in the
            # first pass; its declarations are not the library's API.
            return
        if name:
            build.namespaces.add("::".join(next_path[1:]))
        for item in inner:
            _walk(item, next_path, build)
        return
    if not _public(name):
        return
    if kind == "ClassTemplateDecl":
        params = [
            i.get("name", "") for i in inner if i.get("kind") == "TemplateTypeParmDecl"
        ]
        records = [i for i in inner if i.get("kind") == "CXXRecordDecl"]
        record_node = max(records, key=lambda i: len(i.get("inner", ())), default=node)
        key = f"std::{name}`{len(params)}"
        held = build.types.get(key) or _cpp_type(params)
        if any(params):
            held["params"] = params
        for base in (
            b["type"]["qualType"]
            for b in record_node.get("bases", ())
            if b.get("type", {}).get("qualType")
        ):
            if base not in held["bases"]:
                held["bases"].append(base)
        _members(record_node, held, build, path + [name], build.public)
        build.types[key] = held
    elif kind == "CXXRecordDecl" and node.get("tagUsed") in ("class", "struct"):
        key = f"std::{name}`0"
        held = build.types.get(key) or _cpp_type()
        for base in (
            b["type"]["qualType"]
            for b in node.get("bases", ())
            if b.get("type", {}).get("qualType")
        ):
            if base not in held["bases"]:
                held["bases"].append(base)
        _members(node, held, build, path + [name], build.public)
        build.types[key] = held
    elif kind == "EnumDecl":
        build.types.setdefault(f"std::{name}`0", _cpp_type())
    elif kind in ("TypedefDecl", "TypeAliasDecl", "TypeAliasTemplateDecl"):
        # `std::ostream`, `std::size_t` and the like name types — a
        # typedef anywhere public reaches `scope`.
        build.scope_aliases[name] = _alias_target(node)
        return
    elif kind in ("FunctionDecl", "FunctionTemplateDecl"):
        if "<" in name:
            # Deduction guides and the like are no callables of the API.
            return
        written = node.get("type", {}).get("qualType", "")
        if not written:
            fn = next((i for i in inner if i.get("kind") == "FunctionDecl"), None)
            written = (fn or {}).get("type", {}).get("qualType", "")
        build.functions.setdefault(f"std::{name}", []).append(
            _returned(written, build.aliases, build.public)
        )
    elif kind == "VarDecl" and node.get("storageClass") not in ("static",):
        build.values.add(name)


def _alias_target(node):
    """The spelling a typedef or alias-decl stands for, out of the
    declaration itself or the `TypeAliasDecl` a `TypeAliasTemplateDecl`
    holds."""
    if node.get("kind") == "TypeAliasTemplateDecl":
        inner = next(
            (i for i in node.get("inner", ()) if i.get("kind") == "TypeAliasDecl"),
            None,
        )
        return (inner or {}).get("type", {}).get("qualType") or None
    return node.get("type", {}).get("qualType") or None


def _version(compiler, origin):
    """`clang++ A.B.C, libstdc++ N, -std=c++23` — the toolchain the data
    was read from."""
    run = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True
    )
    compiler_version = re.search(r"version (\d+\.\d+\.\d+)", run.stdout)
    libstdcxx = ""
    run = subprocess.run(
        [compiler, "-x", "c++", "-std=c++23", "-E", "-dM", "-"],
        input="#include <version>\n",
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"#define _GLIBCXX_RELEASE (\d+)", run.stdout)
    if match:
        libstdcxx = f"libstdc++ {match[1]}"
    if not libstdcxx:
        match = re.search(r"c\+\+/(\d+)", str(origin))
        libstdcxx = f"libstdc++ {match[1]}" if match else "libstdc++"
    return (
        f"{compiler} {compiler_version[1]}, {libstdcxx}, -std=c++23"
        if compiler_version
        else f"{compiler}, {libstdcxx}, -std=c++23"
    )


def _params_of(node):
    """The type-parameter names a template declares."""
    return tuple(
        i.get("name", "")
        for i in node.get("inner", ())
        if i.get("kind") == "TemplateTypeParmDecl"
    )


def _collect_aliases(node, build, context=("", ())):
    """Fill ``build.aliases`` with every typedef and type alias of the
    tree, `Owner::member` keys for the members a record declares and bare
    names for namespace-scope ones — each value the parameters the
    spelling's owner declares and the spelling it stands for. Aliases are
    read before the members that may be written with them, so the walk is
    a first pass."""
    kind = node.get("kind")
    name = node.get("name", "")
    owner_name, owner_params = context
    key = f"{owner_name}::{name}" if owner_name else name
    if kind == "TypeAliasTemplateDecl":
        inner = next(
            (i for i in node.get("inner", ()) if i.get("kind") == "TypeAliasDecl"),
            None,
        )
        target = (inner or {}).get("type", {}).get("qualType")
        if target and not target.startswith("("):
            # A lambda or an unnamed spelling is no type the data names.
            build.aliases.setdefault(key, (_params_of(node), target))
            if name.startswith("__"):
                # An implementation spelling may be written unqualified.
                build.aliases.setdefault(name, (_params_of(node), target))
        return
    if kind in ("TypedefDecl", "TypeAliasDecl"):
        target = _alias_target(node)
        if target and not target.startswith("("):
            build.aliases.setdefault(key, (owner_params, target))
            if name.startswith("__"):
                build.aliases.setdefault(name, (owner_params, target))
        return
    if kind == "NamespaceDecl":
        context = (name or owner_name, ())
    elif kind == "ClassTemplateDecl":
        context = (name or owner_name, _params_of(node))
    elif kind in (
        "CXXRecordDecl",
        "ClassTemplatePartialSpecializationDecl",
        "ClassTemplateSpecializationDecl",
    ):
        context = (name or owner_name, owner_params)
    for item in node.get("inner", ()):
        _collect_aliases(item, build, context)


def build(origin):
    held = _Build()
    with tempfile.TemporaryDirectory() as tmp:
        tree = dump_ast(
            ["clang++", "-x", "c++", "-std=c++23"], Path(tmp) / "dump.cc", HEADERS
        )
    _collect_aliases(tree, held)
    # Aliases `std` declares of public names are themselves public names.
    held.public = {
        k.split("::")[-1] for k in held.aliases if not k.split("::")[-1].startswith("_")
    }

    def walk_top(node):
        for item in node.get("inner", ()):
            if item.get("kind") == "NamespaceDecl" and item.get("name") == "std":
                for child in item.get("inner", ()):
                    _walk(child, ["std"], held)
            else:
                walk_top(item)

    walk_top(tree)
    held.public |= {key.split("::", 1)[1].rpartition("`")[0] for key in held.types}
    _finish_types(held.types)
    types = dict(sorted(held.types.items()))
    functions = {
        name: _pick(returns) for name, returns in sorted(held.functions.items())
    }
    scope = {}
    for key in types:
        scope[key.split("::", 1)[1].rpartition("`")[0]] = key
    for name in functions:
        scope[name.split("::", 1)[1]] = name
    for name in held.values:
        scope.setdefault(name, f"std::{name}")
    for name in held.namespaces:
        scope.setdefault(name.split("::")[-1], f"std::{name.split('::')[-1]}")
    for name, target in held.scope_aliases.items():
        # `std::ostream` is the typedef `basic_ostream<char>` — its data
        # key where it is one.
        normed = (
            _returned(f"{target} ()", held.aliases, held.public) if target else None
        ) or target
        spelled_name = normed or name
        leaf = spelled_name.split("<")[0].split("::")[-1]
        arity = str(spelled_name.count(",") + 1) if "<" in spelled_name else None
        key = next(
            (
                k
                for k in types
                if k.split("::", 1)[1].rpartition("`")[0] == leaf
                and (arity is None or k.rpartition("`")[2] == arity)
            ),
            None,
        ) or next(
            iter(
                sorted(
                    (
                        k
                        for k in types
                        if k.split("::", 1)[1].rpartition("`")[0] == leaf
                    ),
                    key=lambda k: int(k.rpartition("`")[2] or 0),
                    reverse=True,
                )
            ),
            None,
        )
        scope.setdefault(name, key or f"std::{name}")
    return _version("clang++", origin), types, functions, scope, None, None


def _pick(returns):
    """The return a name of several overloads is recorded with: the
    one most of them give, of the spelling that names a type
    parameter where any does — None where they genuinely disagree."""
    spelled = [r for r in returns if r]
    if not spelled:
        return None
    bases = {}
    for r in spelled:
        base = r.split("<", 1)[0].split("::")[-1]
        bases.setdefault(base, []).append(r)
    group = max(bases.values(), key=len)
    if len(group) == 1:
        # Every overload spells a different return: the bare-parameter
        # answer is the identity-like one (`move`, `forward`), else
        # the first the headers declare.
        bare = [r for r in spelled if re.fullmatch(r"_\w+", r)]
        return bare[0] if bare else spelled[0]
    options = [r for r in group if re.search(r"\b_\w+\b", r)] or group
    # Where one spelling names only the type's own parameters and public
    # names, it is the overload whose return the data should give.
    clean = [
        r for r in options if not re.search(r"\b(?:__|lambda|decltype|auto|void\b)", r)
    ]
    return min(set(clean or options))


def _finish_types(types):
    """Collapse each method's per-overload returns into the one the data
    records."""
    for entry in types.values():
        for table in ("methods", "statics"):
            entry[table] = {
                name: _pick(returns) for name, returns in entry[table].items()
            }


if __name__ == "__main__":
    main(build)
