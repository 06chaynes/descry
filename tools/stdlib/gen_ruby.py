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

from common import closing, lineage, main, new_type, record, split

_DECL = re.compile(
    r"^(class|module|interface|struct)\s+([A-Z]\w*(?:::[A-Z]\w*)*)\s*"
    r"(?:\[([^\]]*)\])?\s*(?:<\s*([A-Z]\w*(?:::[\w:\[\],* ]*)?))?"
)
_MEMBER = re.compile(
    r"^(def|include|extend|prepend|attr_reader|attr_writer|attr_accessor|alias)\b"
)
# `self.` a singleton method, `self?.` a module function: one both ways.
_DEF = re.compile(r"^def\s+(self\??\.)?([^\s:]+)\s*:\s*(.*)$")
_BASE = re.compile(r"^(include|extend|prepend)\s+(.+?)\s*$")
_ALIAS = re.compile(r"^alias\s+(self\.)?([^\s]+)\s+(?:self\.)?([^\s]+)\s*$")
_ATTR = re.compile(r"^(attr_reader|attr_writer|attr_accessor)\s+([^\s:]+)\s*:\s*(.*)$")

# A `def` that ends one signature is followed by another `def`, a member
# or the `end` of the type; an overload continues its line.
_OVERLOAD = re.compile(r"^\s*\|")


def _written(type_):
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
        return _written(type_[:-1])
    found = re.match(r"[A-Za-z_:][\w:.]*", type_)
    if not found:
        return None
    name = found[0].removeprefix("::")
    if name == "bool":
        return "bool"
    if name == "int":
        return "Integer"
    if name == "string":
        return "String"
    rest = type_[found.end() :]
    end = closing(rest, 0, "[", "]") if rest.startswith("[") else None
    args = split(rest[1 : end - 1], ",", "[", "]") if end else []
    return f"{name}[{', '.join(args)}]" if args else name


def _return(signature):
    """The return type of the ``(args) -> type`` signature ``signature``,
    or of its last `->`."""
    arrow = signature.rfind("->")
    if arrow < 0:
        return None
    return _written(signature[arrow + 2 :])


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
    ret = _written(found[2])
    return {
        "params": takes,
        "returns": ret if ret in params else None,
    }


def _collect(path, types, callbacks, signatures):
    """Read one `.rbs` file's declarations into ``types``, and each
    method's signatures into ``signatures`` as ``(type, how it is
    defined: None, "self." or "self?.", method, signature)``."""
    current = statics = None
    params = []
    method = returns = block = None
    bases = []

    def flush():
        """Record the method read so far, and the bases, of the current type."""
        nonlocal method, returns, block
        if current is not None:
            held = types[current]["bases"]
            held.extend(b for b in bases if b not in held)
        if current is not None and method is not None:
            tables = {None: ("methods",), "self.": ("statics",)}.get(
                statics, ("statics", "methods")
            )
            for table in tables:
                record(types[current][table], method, returns)
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
            if kind == "interface":
                # An interface is what it mixes in may have; it has no
                # members of its own to call.
                current = None
                continue
            current = name
            params = [p.strip() for p in (type_params or "").split(",") if p.strip()]
            types.setdefault(current, new_type(params))
            if base:
                held = _written(base)
                if held:
                    bases.append(held)
            continue
        if current is None:
            continue
        if line.strip() == "end" and not line.startswith("  "):
            flush()
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
                statics, method, signature = defined.groups()
                returns = _return(signature)
                block = _block(signature, params)
                signatures.append((current, statics, method, signature))
            elif base:
                held = _written(base[2])
                if held:
                    bases.append(held)
            elif attr:
                for name_ in re.split(r"\s*,\s*", attr[2].strip()):
                    held = _written(attr[3])
                    record(types[current]["methods"], name_, held)
                    if attr[1] != "attr_reader":
                        record(types[current]["methods"], f"{name_}=", None)
            elif alias:
                # `alias size length` gives `size` what `length` returns.
                table = types[current]["statics" if alias[1] else "methods"]
                record(table, alias[2], table.get(alias[3]))
            continue
        if _OVERLOAD.match(line) and method is not None:
            signature = line.strip().lstrip("|").strip()
            other = _return(signature)
            if other != returns:
                returns = None
            extra = _block(signature, params)
            block = block or extra
            signatures.append((current, statics, method, signature))
            continue
    flush()


def _makers(signatures, types):
    """`Type.method` -> what the call makes of the type it makes: `Class.new`
    and `Module.new` make one, its instances being modules, the `Class`
    their `initialize` is first given its base; a singleton method whose
    block runs as the type it makes (`Struct.new`, `Data.define`) gives it a
    member for each symbol it is passed, writable where the type has
    `[]=`."""
    makers = {}
    for owner, how, method, signature in signatures:
        if (
            method == "initialize"
            and not how
            and "Module" in lineage(owner, types, "[", {})
        ):
            first = re.match(r"\(\s*\??(\w+)", signature)
            made = {"base": True} if first and first[1] == "Class" else {}
            if owner == "Module" or "Class" not in lineage(owner, types, "[", {}):
                made["module"] = True
            makers.setdefault(f"{owner}.new", {}).update(made)
        elif how and (runs := re.search(r"\[self:\s*([^\]]+)\]", signature)):
            returned = re.search(r"->\s*(\w+)\s*$", signature)
            generic = re.match(rf"\[\s*{re.escape(runs[1])}\b", signature)
            made = runs[1] == f"singleton({owner})" or (
                generic and returned and returned[1] == runs[1]
            )
            if made and re.search(r"\*\s*(?:interned|Symbol)\b", signature):
                writable = "[]=" in types.get(owner, {}).get("methods", {})
                makers[f"{owner}.{method}"] = {
                    "fields": "accessor" if writable else "reader"
                }
    return makers


def build(origin):
    origin = Path(origin)
    types, callbacks, signatures = {}, {}, []
    for path in sorted((origin / "core").glob("*.rbs")):
        _collect(path, types, callbacks, signatures)
    for path in sorted(origin.glob("stdlib/**/*.rbs")):
        _collect(path, types, callbacks, signatures)
    for name, entry in types.items():
        entry["bases"] = [
            base
            for base in entry["bases"]
            if base.partition("[")[0].rpartition("::")[2] != name
            and base.partition("[")[0] in types
        ]
    scope = {}
    return {
        "source": "the core and stdlib declarations of rbs 3.9.0",
        "types": types,
        "functions": {},
        "scope": scope,
        "callbacks": callbacks,
        # The methods that run their block in the object they are called on
        "evaluators": sorted(
            {m for _, _, m, signature in signatures if "[self: self]" in signature}
        ),
        "makers": _makers(signatures, types),
    }


if __name__ == "__main__":
    main(build)
