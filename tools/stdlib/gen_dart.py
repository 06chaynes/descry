"""Generate dart.json from the Dart SDK's `lib` sources.

usage: gen_dart.py SDK OUTPUT

SDK is a checkout of `dart-lang/sdk` at a tag; the `sdk/lib` libraries
(`core`, `async`, `collection`, `convert`, `io`, `math`, `typed_data` and
the rest) hold the `dart:` declarations every program calls. The sources
are read by a reader for the declaration syntax they use; nothing of the
SDK is run.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import new_type, record, split

# A comment or a string literal, whichever starts first
_LEXEME = re.compile(
    r"//[^\n]*|/\*.*?\*/"
    r"|(?<![\w$])r?(?:'''.*?'''|\"\"\".*?\"\"\")"
    r"|(?<![\w$])r(?:'[^'\n]*'|\"[^\"\n]*\")"
    r"|'(?:[^'\\\n$]|\\.|\$\{[^{}]*\}|\$)*'"
    r"|\"(?:[^\"\\\n$]|\\.|\$\{[^{}]*\}|\$)*\"",
    re.DOTALL,
)
_TYPE_HEAD = (
    r"^\s*(?:abstract\s+|base\s+|final\s+|sealed\s+|interface\s+)*"
    r"(class|mixin\s+class|mixin|enum|extension\s+type|extension)\s+"
)
_TYPE_DECL = re.compile(
    _TYPE_HEAD + r"(?:name\s+)?([A-Za-z_$][\w$]*)?\s*"
    r"(?:<([A-Za-z_][\w\s,]*(?:extends\s+[^{}]*)?)>)?\s*"
    r"(.*?)\s*{\s*}?\s*$"
)
# Words before a name that are no return type
_NO_TYPE = "new|as|with|class|implements|typedef|get|set|return|await|throw"
_MEMBER = re.compile(
    r"^\s*(?:@[\w$.]+(?![\w$.])(?:\([^)]*\))?\s*)*"
    r"((?:(?:static|factory|abstract|covariant|external|late|final|const)\s+)*)"
    rf"(?:(?!(?:{_NO_TYPE})\b)([A-Za-z_][\w$.<>,? ]*?|operator|void|dynamic)\s+)?"
    r"(get\s+|set\s+|operator\s*)?(?!Function\b)([A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)?|operator\s*[<>=!+\-*/%&|^~\[\]]+|[\w$]+)"
    r"\s*(?:(?:<[\w$,\s]*>)?\s*\((.*)\))?\s*(?:native\s*)?(=(?!=)|:|;|\{|$)"
)
_FUNC_PARAM = re.compile(
    r"(?:required\s+|covariant\s+)?"
    r"([A-Za-z_][\w$.<>,? ]*?\s+Function\s*(?:\([^)]*\))?\??)\s+([\w$]+)"
)


def _lines(text):
    """The lines of ``text``, with a parameter list and the header of a type
    declaration each on one line."""
    held, parens, header = "", 0, False
    for line in text.splitlines():
        if not held:
            header = bool(re.match(_TYPE_HEAD, line))
        held = f"{held} {line.strip()}".lstrip()
        parens += line.count("(") - line.count(")")
        if parens > 0 or header and not re.search(r"[{;]", held):
            continue
        yield held
        held, parens = "", 0


def _written(type_, params):
    """The type ``type_`` as the data writes it: ``List<E>`` with the
    generics Dart writes, ``dynamic``/``void``/``Function``-typed returns
    as None."""
    type_ = re.sub(r"\s+", " ", type_.strip())
    if not type_ or type_ in ("void", "dynamic", "Object?", "Null"):
        return None
    type_ = type_.rstrip("?").strip()
    if "Function" in type_:
        return None
    match = re.match(r"[A-Za-z_][\w$.]*", type_)
    if not match:
        return None
    name = match[0]
    if name == "bool" or name == "int" or name == "double" or name == "num":
        return name
    rest = type_[len(name) :].strip()
    if rest.startswith("<"):
        depth = 0
        for i, c in enumerate(rest):
            depth += (c == "<") - (c == ">")
            if depth == 0:
                inner = rest[1:i]
                return f"{name}<{', '.join(split(inner, ',', '<', '>'))}>"
        return name
    return name


def _bases(heritage, params):
    """The type names ``extends/with/implements/on`` clauses write."""
    found = []
    for clause in ("extends", "with", "implements", "on"):
        match = re.search(
            rf"\b{clause}\b\s+([^{{]*?)(?:\b(?:extends|with|implements|on)\b|$)",
            heritage,
        )
        if match:
            for base in split(match[1], ",", "<", ">"):
                if base.strip().lstrip("<>").startswith("_"):
                    continue
                held = _written(base.strip(), params)
                if held:
                    found.append(held)
    return found


_NESTED_FUNC = re.compile(r"([A-Za-z_][\w$.<>,?]*)\s+([\w$]+)\s*\(([^()]*)\)")

_STATEMENTS = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "return",
        "assert",
        "new",
        "this",
        "super",
        "case",
    }
)


def _callback(sig, params):
    """``{"params", "returns"}`` of a parameter whose type is a function
    type — `T Function(U) name`, or nested-declaration `R name(U arg)` —
    where ``sig`` is the member's parameter list."""
    found = _FUNC_PARAM.search(sig)
    if found:
        takes = found[1]
        inside = takes.partition("(")[2].rpartition(")")[0]
        if not inside:
            return None
        held = [
            re.match(r"[A-Za-z_][\w$.<>,?]*", a.strip())[0].strip("<>?")
            for a in split(inside, ",", "<", ">")
            if a.strip()
        ]
        ret = re.match(r"([A-Za-z_][\w$.<>,?]*)\s+Function", takes)
        ret = _written(ret[1], params) if ret else None
        return {"params": held, "returns": ret if ret in params else None}
    # `void action(E element)`: the function type's signature, nested.
    found = _NESTED_FUNC.search(sig)
    if not found:
        return None
    ret_type, _, takes = found.groups()
    held = []
    for arg in split(takes, ",", "<", ">"):
        arg = arg.strip().rstrip("?")
        name = re.match(r"[A-Za-z_][\w$.<>,?]*", arg)
        if name:
            held.append(name[0])
    ret = _written(ret_type, params)
    return {"params": held, "returns": ret if ret in params else None}


def build(origin):
    origin = Path(origin) / "sdk" / "lib"
    if not origin.is_dir():
        origin = Path(origin)
    types, callbacks, functions = {}, {}, {}
    skip_dirs = {"_internal", "internal", "native", "js", "wasm", "web"}
    files = [
        p
        for p in sorted(origin.rglob("*.dart"))
        if not set(p.relative_to(origin).parts) & skip_dirs and p.is_file()
    ]
    for path in files:
        text = path.read_text("utf-8", errors="replace")
        # A string literal may hold braces, parentheses or `//`.
        text = _LEXEME.sub(lambda found: "" if found[0][0] == "/" else "''", text)
        current = None
        params = []
        depth = 0
        for line in _lines(text):
            stripped = line.strip()
            if not stripped:
                continue
            at_depth = depth
            depth += line.count("{") - line.count("}")
            if current is not None and depth <= 0:
                current, params = None, []
            decl = _TYPE_DECL.match(stripped)
            if decl:
                _, name, type_params, heritage = decl.groups()
                if name is None:
                    # An unnamed `extension on T` is members of `T`.
                    held = re.search(r"\bon\b\s+([A-Za-z_][\w$.<>,]*)", heritage or "")
                    name = held[1].partition("<")[0] if held else None
                if not name:
                    continue
                if name.startswith("_"):
                    current = "_skipped"
                    continue
                name = name.removeprefix("$")
                params = [
                    p.strip().split(" extends ")[0]
                    for p in (type_params or "").split(",")
                    if p.strip()
                ]
                current = name
                entry = types.setdefault(name, new_type(params))
                for base in _bases(heritage or "", params):
                    if base not in entry["bases"]:
                        entry["bases"].append(base)
                continue
            if current is None:
                member = _MEMBER.match(stripped)
                # `Type name(...)`, or `name(...)` with a body: a `name (` is
                # a clause, and `name(...);` alone a call.
                if (
                    at_depth == 0
                    and member
                    and member[5] is not None
                    and not member[4].startswith("_")
                    and (
                        member[2]
                        or stripped.startswith(("(", "<"), member.end(4))
                        and not stripped.endswith(";")
                    )
                ):
                    functions[member[4]] = _written(member[2] or "", ())
                continue
            if current == "_skipped" or at_depth != 1:
                continue
            if re.match(r"set\s", stripped):
                continue
            member = _MEMBER.match(stripped)
            if not member or not member[4]:
                continue
            modifiers, ret_type, accessor, name, sig, end = member.groups()
            constructs = (
                "factory" in modifiers
                or name == current
                or name.startswith(current + ".")
            )
            # Without a type: a constructor with its parameters, or a field
            # with a modifier. Any other is a bare name, a statement, or a
            # call a value goes on with.
            kept = sig is not None if constructs else modifiers and sig is None
            if ret_type is None and not accessor and not kept:
                continue
            if (ret_type or "").endswith(".") or name in _STATEMENTS:
                continue
            if end == ":" and not constructs:
                # Only a constructor goes on with `:`; the rest of a
                # conditional expression does too.
                continue
            gparams = re.search(r"<([\w$,\s]+)>\s*\(", stripped)
            own = params + (
                [p.strip() for p in gparams[1].split(",")] if gparams else []
            )
            static = "static" in modifiers
            if name.rpartition(".")[2].startswith("_"):
                continue
            table = types[current]["statics" if static else "methods"]
            if accessor == "get" or (
                accessor is None and sig is None and name != "operator"
            ):
                record(table, name, _written(ret_type or "", params))
                continue
            if accessor == "set":
                continue
            if name == "operator":
                continue
            if constructs:
                record(
                    types[current]["statics"],
                    name.split(".")[-1] if "." in name else "new",
                    current,
                )
                continue
            ret = _written(ret_type or "", own)
            record(table, name, ret)
            held = _callback(sig or "", own)
            if held:
                callbacks.setdefault(current, {}).setdefault(name, {})["0"] = held
    for name, entry in types.items():
        entry["bases"] = [
            base
            for base in entry["bases"]
            if re.match(r"[A-Za-z_][\w$]*", base)[0] != name
            and re.match(r"[A-Za-z_][\w$]*", base)[0] in types
        ]
    scope = {}
    return (
        "the Dart 3.7.0 SDK `sdk/lib` libraries",
        types,
        functions,
        scope,
        None,
        callbacks,
    )


if __name__ == "__main__":
    from common import main

    main(build)
