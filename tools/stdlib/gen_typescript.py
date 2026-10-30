"""Generate typescript.json from the TypeScript compiler's library declarations.

usage: gen_typescript.py TYPESCRIPT.tgz OUTPUT

TYPESCRIPT.tgz is the `typescript` package as the npm registry serves it;
its `lib/lib.*.d.ts` are read in place, nothing of it is unpacked or run.
"""

import json
import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import closing, lineage, main, new_type, record
from common import split as cut

TOKEN = re.compile(
    r"""//[^\n]*|/\*.*?\*/|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`"""
    r"""|[A-Za-z_$][\w$]*|\d[\w.]*|=>|\.\.\.|\S""",
    re.DOTALL,
)
# The libraries read, each with every library it references: ECMAScript
# and the browser's globals.
LIBRARIES = ("esnext", "dom")
REFERENCE = re.compile(r'^/// <reference lib="([^"]+)" />', re.MULTILINE)
NAME = re.compile(r"[A-Za-z_$][\w$]*")
OPEN, CLOSE = "([{<", ")]}>"
MODIFIERS = ("readonly", "static", "abstract", "get", "set")
# What can follow the name of a member; before one of these a modifier's
# word is the name itself.
AFTER_NAME = ("(", "<", ":", "?", "")
# The keyword of each primitive type and the interface that has its methods.
PRIMITIVES = {
    "string": "String",
    "number": "Number",
    "boolean": "Boolean",
    "bigint": "BigInt",
    "symbol": "Symbol",
    "object": "Object",
}


def split(toks, separators):
    """`toks` cut at each of `separators` that is outside any bracket."""
    parts, depth = [[]], 0
    for tok in toks:
        if depth == 0 and tok[0] in separators:
            parts.append([])
        else:
            depth += (tok[0] in OPEN) - (tok[0] in CLOSE)
            parts[-1].append(tok)
    return [part for part in parts if part]


def arguments(text):
    """The type arguments of `text` where it is one type written with them,
    `Name<A, B>`; None for any other."""
    name, _, rest = text.partition("<")
    if not NAME.fullmatch(name.replace(".", "")) or "=>" in rest:
        return None
    depth = 1
    for at, char in enumerate(rest):
        depth += (char == "<") - (char == ">")
        if depth == 0:
            return cut(rest[:at], ",", OPEN, CLOSE) if at == len(rest) - 1 else None
    return None


def join(one, other):
    """What two signatures both return: the type they agree on, `unknown`
    standing for each type argument they do not; None where they return
    different types, or nothing."""
    if one == other or None in (one, other):
        return one if one == other else None
    if one.endswith("[]") and other.endswith("[]"):
        return f"{join(one[:-2], other[:-2]) or 'unknown'}[]"
    held, others = arguments(one), arguments(other)
    same = one.partition("<")[0] == other.partition("<")[0]
    if not held or not others or not same or len(held) != len(others):
        return None
    held = ", ".join(join(*pair) or "unknown" for pair in zip(held, others))
    return f"{one.partition('<')[0]}<{held}>"


def annotation(toks):
    """The tokens of the type that the parameter `toks` is declared with."""
    depth = 0
    for at, tok in enumerate(toks):
        if tok[0] == ":" and depth == 0:
            return toks[at + 1 :]
        depth += (tok[0] in OPEN) - (tok[0] in CLOSE)
    return []


def callable_type(toks):
    """`(params, returns)` of the function type that `toks`, the type of a
    parameter, are, whether or not it may be null or undefined instead: the
    tokens of each of its parameters and those of what it returns."""
    words = [tok[0] for tok in toks] + [""]
    if words[0] == "(":
        end = closing(words, 0, OPEN, CLOSE)
        if words[end] == "=>":
            return split(toks[1 : end - 1], ","), toks[end + 1 :]
        if not words[end]:
            return callable_type(toks[1:-1])
    options = [o for o in split(toks, "|") if o[0][0] not in ("null", "undefined")]
    if len(options) == 1 and len(options[0]) < len(toks):
        return callable_type(options[0])
    return None


class Library:
    """The declarations of every file read, merged by qualified name."""

    def __init__(self):
        self.types = {}
        self.functions = {}
        self.scope = {}
        # Names are resolved once every file is read: a declaration may
        # refer to one that a later file makes.
        self.declared = set()
        self.heritage = []
        self.signatures = []
        self.variables = []
        self.callbacks = {}
        # (type, method) of each signature that puts what it is passed on
        # the object it is passed first, and hands that object back
        self.writes = set()
        # global value -> the type it is declared with
        self.value_types = {}

    def declare(self, name, params=()):
        self.declared.add(name)
        return self.types.setdefault(name, new_type(params))

    def resolve(self, namespace, name):
        """`name` as written inside `namespace`, qualified."""
        scopes = namespace.split(".")[:-1]
        head = name.split(".")[0]
        for depth in range(len(scopes), 0, -1):
            prefix = ".".join(scopes[:depth]) + "."
            if prefix + head in self.declared:
                return prefix + name
        return name

    def reference(self, namespace, toks):
        """The type `toks` name when they are one type reference."""
        words = [tok[0] for tok in toks]
        size = 1
        while words[size : size + 1] == ["."]:
            size += 2
        if size < len(words) and (
            words[size] != "<" or closing(words, size, OPEN, CLOSE) < len(words)
        ):
            return None
        return self.resolve(namespace, "".join(words[:size]))

    def spell(self, namespace, toks, bounds=()):
        """A type as the source writes it, None for `void`. Each type
        parameter in `bounds` is written as the type it has there."""
        out = []
        for n, tok in enumerate(toks):
            word = tok[0]
            if n and tok.start() > toks[n - 1].end():
                out.append(" ")
            qualified = n and toks[n - 1][0] == "."
            if NAME.fullmatch(word) and not qualified:
                word = self.resolve(namespace, word)
            out.append(bounds[word] if word in bounds and not qualified else word)
        return None if out == ["void"] else "".join(out)

    def finish(self):
        for kind, namespace, toks in self.heritage:
            for part in split(toks, ("extends", "implements", ",")):
                base = self.spell(namespace, part)
                if (
                    self.reference(namespace, part) in self.types
                    and base not in kind["bases"]
                ):
                    kind["bases"].append(base)
        owners = {id(kind["methods"]): name for name, kind in self.types.items()}
        for table, name, namespace, generics, params, returns in self.signatures:
            generics = split(generics, ",")
            # A type parameter that extends one type is written as that type.
            bounds = {
                part[0][0]: self.resolve(namespace, part[2][0])
                for part in generics
                if [tok[0] for tok in part[1:4:2]] in (["extends"], ["extends", "="])
            }
            returned = self.spell(namespace, returns, bounds)
            table[name] = join(table[name], returned) if name in table else returned
            first = annotation(params[0]) if params else []
            gives = split(returns, "&")[0] if returns else []
            # A further parameter, required, of what is put on the object:
            # no primitive, which no member is made of
            put = [
                held
                for param in params[1:]
                if (held := annotation(param))
                and "?" not in [tok[0] for tok in param[: len(param) - len(held)]]
                and held[0][0] not in PRIMITIVES
            ]
            if (
                put
                and len(first) == 1
                and first[0][0] in [part[0][0] for part in generics]
                and [tok[0] for tok in gives] == [first[0][0]]
            ):
                # `assign<T, U>(target: T, source: U): T & U`
                self.writes.add((owners.get(id(table)), name))
            passed = self.callbacks.setdefault(owners.get(id(table)), {})
            for position, param in enumerate(params):
                taken = callable_type(annotation(param))
                if not taken:
                    continue
                types = [
                    self.spell(namespace, annotation(each), bounds) or "unknown"
                    for each in taken[0]
                    if each[0][0] not in ("this", "...")
                ]
                # The type parameter of the signature that the function returns
                result = split(taken[1], "|")[0]
                result = result[0][0] if len(result) == 1 else None
                if result not in [part[0][0] for part in generics]:
                    result = None
                known = passed.setdefault(name, {}).setdefault(
                    str(position), {"params": types, "returns": result}
                )
                known["params"] = [
                    one if one == other else "unknown"
                    for one, other in zip(known["params"], types)
                ]
                if known["returns"] != result:
                    known["returns"] = None
        self.callbacks = {
            key: held for key, held in self.callbacks.items() if key and held
        }
        for keyword, name in PRIMITIVES.items():
            self.types[keyword] = new_type()
            self.types[keyword]["bases"].append(name)
        for name, namespace, toks in self.variables:
            target = name
            if toks:
                operands = [
                    self.reference(namespace, part)
                    for part in split(toks, "&")
                    if part[0][0] != "typeof"
                ]
                if len(operands) != 1 or operands[0] not in self.types:
                    continue
                target = operands[0]
                if not namespace:
                    self.value_types[name] = target
            if toks and name in self.types:
                # The variable is the value of the type that shares its
                # name: what can be called on it are the type's statics.
                members = {}
                for each in lineage(target, self.types, "<", {}):
                    members = self.types[each]["methods"] | members
                for method, returns in members.items():
                    record(self.types[name]["statics"], method, returns)
                target = name
            if not namespace:
                self.scope[name] = target
        # A call signature is kept under the empty name until here: that of
        # a type's own variable is the function of the type's name.
        for name, kind in self.types.items():
            kind["methods"].pop("", None)
            if "" in kind["statics"]:
                self.functions[name] = kind["statics"].pop("")


class Reader:
    """Reads the declarations of one library file into a `Library`."""

    def __init__(self, library, name, text):
        self.library = library
        self.name = name
        self.toks = [
            tok for tok in TOKEN.finditer(text) if not tok[0].startswith(("//", "/*"))
        ]
        self.at = 0

    def peek(self):
        return self.toks[self.at][0]

    def take(self):
        self.at += 1
        return self.toks[self.at - 1][0]

    def span(self, stops):
        """The tokens up to the first of `stops` outside any bracket, or to
        the bracket that closes the enclosing one."""
        start, depth = self.at, 0
        while depth or not (self.peek() in stops or self.peek() in CLOSE):
            depth += (self.peek() in OPEN) - (self.peek() in CLOSE)
            self.at += 1
        return self.toks[start : self.at]

    def brackets(self):
        """The tokens inside the bracket pair that starts here."""
        self.at += 1
        inner = self.span(())
        self.at += 1
        return inner

    def is_module(self):
        """A file with a top-level `export` declares globals only inside
        `declare global`."""
        depth = 0
        for tok in self.toks:
            if depth == 0 and tok[0] == "export":
                return True
            depth += (tok[0] == "{") - (tok[0] == "}")
        return False

    def statements(self, namespace):
        """Read declarations up to the end of the enclosing block; a
        `namespace` of None is a module's own scope, which is passed over."""
        library = self.library
        while self.at < len(self.toks) and self.peek() != "}":
            word = self.take()
            if word in ("declare", "export", "abstract", ";"):
                continue
            if word == "global":
                self.take()
                self.statements("")
                self.take()
            elif namespace is None:
                self.at -= 1
                self.span(";{")
                if self.peek() == "{":
                    self.brackets()
                else:
                    self.at += 1
            elif word in ("namespace", "module"):
                name = namespace + self.take()
                library.declare(name)
                if not namespace:
                    library.scope[name] = name
                self.take()
                self.statements(name + ".")
                self.take()
            elif word in ("interface", "class"):
                kind = library.declare(namespace + self.take(), self.params())
                library.heritage.append((kind, namespace, self.span("{")))
                self.members(namespace, kind["methods"], kind["statics"])
            elif word == "type":
                library.declared.add(namespace + self.take())
                self.span(";")
            elif word == "function":
                name = self.take()
                table = library.functions
                if namespace:
                    table = library.types[namespace[:-1]]["statics"]
                else:
                    library.scope[name] = name
                self.signature(namespace, table, name, self.span(";"))
            elif word in ("var", "let", "const"):
                name = namespace + self.take()
                self.take()
                toks = ()
                if self.peek() == "{":
                    statics = library.declare(name)["statics"]
                    self.members(namespace, statics, statics)
                else:
                    library.declared.add(name)
                    toks = self.span(";")
                library.variables.append((name, namespace, toks))
            else:
                raise ValueError(f"{self.name}: unexpected {word!r}")

    def params(self):
        if self.peek() != "<":
            return []
        return [part[0][0] for part in split(self.brackets(), ",")]

    def members(self, namespace, methods, statics):
        """Read the methods of a `{ ... }` body into `methods`, those marked
        `static` into `statics`."""
        self.take()
        while self.peek() != "}":
            toks = self.span(";,")
            if self.peek() != "}":
                self.at += 1
            words = [tok[0] for tok in toks] + [""]
            at, table, accessor = 0, methods, False
            while words[at] in MODIFIERS and words[at + 1] not in AFTER_NAME:
                table = statics if words[at] == "static" else table
                accessor |= words[at] in ("get", "set")
                at += 1
            if accessor or words[at] == "new" and words[at + 1] in ("(", "<"):
                continue
            name = ""
            if words[at] == "[":
                end = closing(words, at, OPEN, CLOSE)
                name, at = "".join(words[at:end]), end
            elif words[at] not in ("(", "<"):
                name, at = words[at], at + 1
            self.signature(namespace, table, name, toks[at:])
        self.take()

    def signature(self, namespace, table, name, toks):
        """Note the return type of `toks`, a method or function from after
        its name, when they are one."""
        words = [tok[0] for tok in toks] + [""]
        at = int(words[0] == "?")
        generics = []
        if words[at] == "<":
            end = closing(words, at, OPEN, CLOSE)
            generics, at = toks[at + 1 : end - 1], end
        if words[at] != "(":
            return
        end = closing(words, at, OPEN, CLOSE)
        if words[end] != ":":
            raise ValueError(f"{self.name}: {name} has no return type")
        params = split(toks[at + 1 : end - 1], ",")
        self.library.signatures.append(
            (table, name, namespace, generics, params, toks[end + 1 :])
        )


def build(archive):
    with tarfile.open(archive) as tar:
        files = {
            member.name: tar.extractfile(member).read().decode("utf-8")
            for member in tar
            if member.name == "package/package.json"
            or member.name.startswith("package/lib/lib.")
            and member.name.endswith(".d.ts")
        }
    library = Library()
    pending, seen = list(LIBRARIES), set()
    while pending:
        name = pending.pop(0)
        if name in seen:
            continue
        seen.add(name)
        text = files[f"package/lib/lib.{name}.d.ts"]
        pending += REFERENCE.findall(text)
        reader = Reader(library, name, text)
        reader.statements(None if reader.is_module() else "")
    library.finish()
    version = json.loads(files["package/package.json"])["version"]
    libraries = " and ".join(f"lib.{name}.d.ts" for name in LIBRARIES)
    source = (
        f"typescript {version} (npm), {libraries} with the libraries they reference"
    )
    return {
        "source": source,
        "types": library.types,
        "functions": library.functions,
        "scope": library.scope,
        "callbacks": library.callbacks,
        # The functions of a global value that put members on the object
        # passed first: `Object.assign`, `Object.defineProperty`
        "writers": sorted(
            f"{name}.{method}"
            for name, declared in library.value_types.items()
            for owner, method in library.writes
            if owner == declared
        ),
    }


if __name__ == "__main__":
    main(build)
