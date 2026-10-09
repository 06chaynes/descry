"""Generate ruby.json from the RBS declarations of the `rbs` gem.

usage: gen_ruby.py RBS OUTPUT

RBS is the `ruby/rbs` repository's checkout at a tag: its `core/*.rbs`
declarations are the classes and modules every Ruby program has, its
`stdlib/*/0/*.rbs` those a `require` brings. The files are read by a
reader for the declaration syntax they use; nothing of the repository
is run.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import new_type, record, split

_DECL = re.compile(
    r"^(class|module|interface|struct)\s+([A-Z]\w*(?:::[A-Z]\w*)*)\s*"
    r"(?:\[([^\]]*)\])?\s*(?:<\s*([A-Z]\w*(?:::[\w:\[\],* ]*)?))?"
)
_MEMBER = re.compile(
    r"^(def|include|extend|prepend|attr_reader|attr_writer|attr_accessor|alias)\b"
)
_DEF = re.compile(r"^def\s+(self\.)?([^\s:]+)\s*:\s*(.*)$")
_TYPE_HEAD = re.compile(
    r"([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*(\[|\(|\{|$|[^[\](){}<>\w]*)"
)
_BASE = re.compile(r"^(include|extend|prepend)\s+(.+?)\s*$")
_ALIAS = re.compile(r"^alias\s+([^\s]+)\s+([^\s]+)\s*$")
_ATTR = re.compile(r"^(attr_reader|attr_writer|attr_accessor)\s+([^\s:]+)\s*:\s*(.*)$")

# A `def` that ends one signature is followed by another `def`, a member
# or the `end` of the type; an overload continues its line.
_OVERLOAD = re.compile(r"^\s*\|")


def _types(text):
    """The types ``text`` writes, as written, one where a union writes
    several (the resolver cannot pick between them and leaves the
    name)."""
    text = re.sub(r"\{.*?\}", "{}", text, flags=re.DOTALL)
    parts = split(text, "|", "(", ")")
    return [p.strip() for p in parts]


def _written(type_, _params):
    """The type ``type_`` as the data writes it: ``Array[Integer]`` with
    the brackets Ruby writes; ``untyped``/``void``/``self``/a union or a
    callable as None."""
    type_ = type_.strip()
    if not type_ or type_ in ("void", "untyped", "nil", "self", "instance", "class"):
        return None
    type_ = type_.removeprefix("^").strip()
    if "|" in type_ or "{" in type_:
        return None
    if type_.endswith("?"):
        # `T?` is `T` or `nil`.
        return _written(type_[:-1], _params)
    name = re.match(r"[A-Za-z_:][\w:.]*", type_)
    if not name:
        return None
    name = name[0].removeprefix("::")
    if name == "bool":
        return "bool"
    if name == "int":
        return "Integer"
    if name == "string":
        return "String"
    rest = type_[len(name[0]) :] if name[0].startswith("::") else type_[len(name) :]
    args = []
    if rest.startswith("["):
        depth = 0
        for i, c in enumerate(rest):
            depth += (c == "[") - (c == "]")
            if depth == 0:
                inner = rest[1:i]
                args = [a.strip() for a in split(inner, ",", "[", "]")]
                break
    return f"{name}[{', '.join(args)}]" if args else name


def _return(signature, params):
    """The return type of the ``(args) -> type`` signature ``signature``,
    or of its last `->`."""
    arrow = signature.rfind("->")
    if arrow < 0:
        return None
    return _written(signature[arrow + 2 :], params)


def _block(signature, params):
    """``{"params", "returns"}`` of the ``{ (args) -> type }`` block the
    signature writes, where one is given; its parameter types and the
    parameter of the signature's own that stands for its result."""
    found = re.search(r"\{\s*\(([^)]*)\)\s*->\s*([^}]*)\}", signature)
    if not found:
        return None
    takes = [
        t.strip().lstrip("?*") for t in split(found[1], ",", "(", ")") if t.strip()
    ]
    takes = [
        re.match(r"[A-Za-z_:][\w:?.]*", t)[0]
        if re.match(r"[A-Za-z_:][\w:?.]*", t)
        else t
        for t in takes
    ]
    ret = _written(found[2], params)
    return {
        "params": takes,
        "returns": ret if ret in params else None,
    }


def _collect(path, types, callbacks, functions):
    """Read one `.rbs` file's declarations into ``types``."""
    current = statics = None
    params = []
    method = returns = block = None
    bases = []

    def flush():
        nonlocal method, returns, block
        if current is None or method is None:
            method = returns = block = None
            return
        table = types[current]["statics" if statics else "methods"]
        record(table, method, returns)
        if block:
            callbacks.setdefault(current, {}).setdefault(method, {})["0"] = block
        method = returns = block = None

    for line in path.read_text("utf-8").splitlines():
        line = re.sub(r"#.*", "", line).rstrip()
        if not line.strip():
            continue
        head = _DECL.match(line.strip())
        if head and not line.startswith(" "):
            flush()
            kind, name, type_params, base = head.groups()
            name = name.split("::")[-1] if "::" not in name else name
            if kind == "interface":
                # An interface is what it mixes in may have; it has no
                # members of its own to call.
                current = None
                continue
            current = name
            params = [p.strip() for p in (type_params or "").split(",") if p.strip()]
            types.setdefault(current, new_type(params))
            if base:
                held = _written(base, params)
                if held:
                    bases.append(held)
            continue
        if current is None:
            continue
        if line.strip() == "end" and not line.startswith("  "):
            flush()
            types[current]["bases"].extend(
                b for b in bases if b not in types[current]["bases"]
            )
            current, bases = None, []
            continue
        member = _MEMBER.match(line.strip())
        if member:
            flush()
            defined = _DEF.match(line.strip())
            base = _BASE.match(line.strip())
            attr = _ATTR.match(line.strip())
            alias = _ALIAS.match(line.strip())
            if defined:
                statics_name, method, signature = defined.groups()
                statics = bool(statics_name)
                returns = _return(signature, params)
                block = _block(signature, params)
            elif base:
                held = _written(base[2], params)
                if held:
                    bases.append(held)
            elif attr:
                for name_ in re.split(r"\s*,\s*", attr[2].strip()):
                    held = _written(attr[3], params)
                    record(types[current]["methods"], name_, held)
                    if attr[1] != "attr_reader":
                        record(types[current]["methods"], f"{name_}=", None)
            elif alias:
                # `alias size length` gives `size` what `length` returns.
                held = types[current]["methods"].get(alias[2])
                record(types[current]["methods"], alias[1], held)
            continue
        if _OVERLOAD.match(line) and method is not None:
            signature = line.strip().lstrip("|").strip()
            other = _return(signature, params)
            if other != returns:
                returns = None
            extra = _block(signature, params)
            block = block or extra
            continue
    flush()
    if current is not None:
        types[current]["bases"].extend(
            b for b in bases if b not in types[current]["bases"]
        )


def build(origin):
    origin = Path(origin)
    types, callbacks, functions = {}, {}, {}
    for path in sorted((origin / "core").glob("*.rbs")):
        _collect(path, types, callbacks, functions)
    for path in sorted(origin.glob("stdlib/**/*.rbs")):
        _collect(path, types, callbacks, functions)
    for name, entry in types.items():
        entry["bases"] = [
            base
            for base in entry["bases"]
            if base.partition("[")[0].rpartition("::")[2] != name
            and base.partition("[")[0] in types
        ]
    scope = {}
    return (
        "the core and stdlib declarations of rbs 3.9.0",
        types,
        functions,
        scope,
        None,
        callbacks,
    )


if __name__ == "__main__":
    from common import main

    main(build)
