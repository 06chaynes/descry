"""Generate python.json from typeshed's standard-library stubs.

usage: gen_python.py TYPESHED.tar.gz OUTPUT

TYPESHED.tar.gz is GitHub's archive of one typeshed commit; it is read in
place, not unpacked.
"""

import ast
import copy
import sys
import tarfile
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent))

from common import main, new_type

# The kinds of binding that define a name where they stand.
OWN = ("class", "function", "value")
TYPE_VARIABLES = ("typing.TypeVar", "typing.ParamSpec", "typing.TypeVarTuple")
PARAMETERS = ("typing.Generic", "typing.Protocol")
# What a decorator makes of the function under it: one that is called on
# the class, one that is not called at all, or the function unchanged. Any
# other decorator leaves the return type unknown.
STATIC = ("builtins.staticmethod", "builtins.classmethod")
# Static or class methods without a decorator (Python Language Reference,
# "Basic customization" and "Emulating generic types").
IMPLICITLY_STATIC = ("__new__", "__init_subclass__", "__class_getitem__")
ATTRIBUTE = (
    "builtins.property",
    "types.DynamicClassAttribute",
    "functools.cached_property",
)
ACCESSORS = ("setter", "getter", "deleter")
UNCHANGED = (
    "typing.overload",
    "typing.final",
    "abc.abstractmethod",
    "warnings.deprecated",
    "typing_extensions.deprecated",
)
UNIONS = ("typing.Optional", "typing.Union")
# What says how a variable is used, around the type it holds.
QUALIFIERS = ("typing.Final", "typing.ClassVar")
# The names `typing` has for classes of other modules, which the stubs
# declare without saying what they stand for (PEP 585).
RENAMED = {
    "typing.ChainMap": "collections.ChainMap",
    "typing.Counter": "collections.Counter",
    "typing.DefaultDict": "collections.defaultdict",
    "typing.Deque": "collections.deque",
    "typing.Dict": "builtins.dict",
    "typing.FrozenSet": "builtins.frozenset",
    "typing.List": "builtins.list",
    "typing.OrderedDict": "collections.OrderedDict",
    "typing.Set": "builtins.set",
    "typing.Tuple": "builtins.tuple",
    "typing.Type": "builtins.type",
}
# What a union says nothing with: no value has a method through them.
EMPTY = ("None", "typing.Any")


def statements(body):
    """The statements of `body` on the newest Python, on every platform."""
    for node in body:
        if isinstance(node, ast.If):
            holds = truth(node.test)
            if holds is not False:
                yield from statements(node.body)
            if holds is not True:
                yield from statements(node.orelse)
        else:
            yield node


def truth(test):
    """Whether `test` holds on the newest Python; None when the platform
    decides."""
    if isinstance(test, ast.BoolOp):
        values = [truth(value) for value in test.values]
        decisive = isinstance(test.op, ast.Or)
        if decisive in values:
            return decisive
        return None if None in values else not decisive
    if isinstance(test, ast.Compare) and ast.unparse(test.left) == "sys.version_info":
        return isinstance(test.ops[0], ast.GtE | ast.Gt)
    return None


def names(node):
    """The names `node` uses, in the order it writes them."""
    if isinstance(node, ast.Name):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from names(child)


def alternatives(node):
    """The types the union `node` is of, in the order it writes them."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return alternatives(node.left) + alternatives(node.right)
    if isinstance(node, ast.Subscript) and ast.unparse(node.value) in UNIONS:
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return [found for item in items for found in alternatives(item)]
    return [node]


def lineage(key, types, scope):
    """The type `key` and the types it has the methods of, nearest first."""
    found = [key]
    for name in found:
        for base in types[name]["bases"]:
            head = base.partition("[")[0]
            if scope.get(head, head) not in found:
                found.append(scope.get(head, head))
    return found


def merged(returns, params, types, scope):
    """What a function returns, where `returns` has what each of its
    overloads does: what they agree on, also where they differ in `None`
    and in a type variable of their own in a union (`dict.get` gives `_VT |
    None`); one type with `typing.Any` for the arguments they give it
    differently; for several types of the data the nearest type all of
    them are, `object` at the least; else None. A union of several types is
    as many overloads."""
    if None in returns:
        return None
    found = [alternatives(ast.parse(text, mode="eval").body) for text in returns]
    written = [[ast.unparse(node) for node in nodes] for nodes in found]
    optional = " | None" * any("None" in texts for texts in written)
    plain = [
        [
            node
            for node, text in zip(nodes, texts)
            if text not in EMPTY
            and (not text.isidentifier() or text in params or text in scope)
        ]
        for nodes, texts in zip(found, written)
    ]
    single = len(set(returns)) == 1
    if single and len(plain[0]) < 2:
        return returns[0]
    if not all(plain):
        return returns[0] if single else None
    shared = {ast.unparse(nodes[0]) for nodes in plain}
    if all(len(nodes) == 1 for nodes in plain) and len(shared) == 1:
        return shared.pop() + optional
    heads = {
        ast.unparse(node.value if isinstance(node, ast.Subscript) else node)
        for nodes in plain
        for node in nodes
    }
    given = [
        [ast.unparse(item) for item in getattr(node.slice, "elts", [node.slice])]
        for nodes in plain
        for node in nodes
        if isinstance(node, ast.Subscript)
    ]
    counted = sum(map(len, plain))
    if len(heads) == 1 and len(given) == counted and len(set(map(len, given))) == 1:
        items = [
            column[0] if len(set(column)) == 1 else "typing.Any"
            for column in zip(*given)
        ]
        return f"{heads.pop()}[{', '.join(items)}]{optional}"
    keys = sorted(scope.get(head, head) for head in heads)
    if all(key in types for key in keys):
        lines = [lineage(key, types, scope) for key in keys]
        common = [key for key in lines[0] if all(key in line for line in lines)]
        key = common[0] if common else "builtins.object"
        name = key.rpartition(".")[2] if key in scope.values() else key
        items = ", ".join(["typing.Any"] * len(types[key]["params"]))
        return name + f"[{items}]" * bool(items) + optional
    return returns[0] if single else None


class Found(NamedTuple):
    """A definition: a "module", a "class" or "function" with the nodes
    that define it, a "value" with the expression assigned to it."""

    kind: str
    module: str
    name: str = ""
    nodes: object = None

    @property
    def path(self):
        return f"{self.module}.{self.name}"


class Module:
    """What the top level of one stub binds."""

    def __init__(self, name, is_package, tree):
        self.package = name if is_package else name.rpartition(".")[0]
        self.names = {}
        self.typed = {}
        self.stars = []
        self.listed = None
        self.reexported = set()
        for node in statements(tree.body):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                if self.names.get(node.name, ("",))[0] != kind:
                    self.names[node.name] = (kind, [])
                self.names[node.name][1].append(node)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    head = alias.name.partition(".")[0]
                    self.names[alias.asname or head] = (
                        "module",
                        alias.asname and alias.name or head,
                    )
            elif isinstance(node, ast.ImportFrom):
                self.imports(node)
            elif isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
                target = (
                    node.targets[0] if isinstance(node, ast.Assign) else node.target
                )
                if not isinstance(target, ast.Name):
                    continue
                alias = isinstance(node, ast.Assign) or (
                    isinstance(node, ast.AnnAssign)
                    and ast.unparse(node.annotation).endswith("TypeAlias")
                )
                if target.id == "__all__":
                    self.listed = (self.listed or set()) | {
                        item.value for item in node.value.elts
                    }
                else:
                    self.names[target.id] = ("value", node.value if alias else None)
                    if not alias and isinstance(node, ast.AnnAssign):
                        self.typed[target.id] = node.annotation

    def imports(self, node):
        source = node.module
        if node.level:
            package = self.package.split(".")
            package = package[: len(package) - node.level + 1]
            source = ".".join(package + [node.module] * bool(node.module))
        for alias in node.names:
            if alias.name == "*":
                self.stars.append(source)
                continue
            self.names[alias.asname or alias.name] = ("import", source, alias.name)
            if alias.asname == alias.name:
                self.reexported.add(alias.name)


class Stubs:
    """Every module of the stubs, and what a name written in one refers to."""

    def __init__(self, modules):
        self.modules = modules
        self.scope = {}
        self.exported = {}
        # owner -> function -> argument -> what each overload says of the
        # callable passed there: (the types of its parameters, its result)
        self.callbacks = {}

    def find(self, module, name, seen=()):
        """What `name` is in `module`, followed through imports and plain
        renames to its definition."""
        if (module, name) in seen or module not in self.modules:
            return None
        seen = (*seen, (module, name))
        here = self.modules[module]
        bound = here.names.get(name)
        if bound is None:
            for star in here.stars:
                if name in self.starred(star):
                    return self.find(star, name, seen)
            return None
        if bound[0] == "module":
            return Found("module", bound[1])
        if bound[0] == "import":
            return self.member(bound[1], bound[2], seen)
        # `typing_extensions` repeats much of `typing`; the name is `typing`'s.
        if module == "typing_extensions" and (same := self.find("typing", name, seen)):
            return same
        if isinstance(bound[1], ast.Name | ast.Attribute):
            return self.target(module, (), bound[1], seen)
        return Found(bound[0], module, name, bound[1])

    def member(self, module, name, seen=()):
        submodule = f"{module}.{name}"
        return self.find(module, name, seen) or (
            Found("module", submodule) if submodule in self.modules else None
        )

    def inner(self, outer, name):
        """The class `name` nested in the class `outer`."""
        nodes = [
            item
            for node in outer.nodes
            for item in statements(node.body)
            if isinstance(item, ast.ClassDef) and item.name == name
        ]
        return (
            Found("class", outer.module, f"{outer.name}.{name}", nodes)
            if nodes
            else None
        )

    def target(self, module, classes, expr, seen=()):
        """What the dotted name `expr` refers to where it is written, in
        `module` inside `classes`."""
        if isinstance(expr, ast.Name):
            for outer in reversed(classes):
                if nested := self.inner(outer, expr.id):
                    return nested
            return self.find(module, expr.id, seen) or self.find(
                "builtins", expr.id, seen
            )
        if isinstance(expr, ast.Attribute):
            base = self.target(module, classes, expr.value, seen)
            if base and base.kind == "module":
                return self.member(base.module, expr.attr, seen)
            if base and base.kind == "class":
                return self.inner(base, expr.attr)
        return None

    def listing(self, module):
        """The module's `__all__`, None when it has none."""
        here = self.modules[module]
        if bound := here.names.get("__all__"):
            return (self.listing(bound[1]) or set()) | (here.listed or set())
        return here.listed

    def starred(self, module):
        """The names `from module import *` brings."""
        if module not in self.modules:
            return set()
        listing = self.listing(module)
        if listing is None:
            return {name for name in self.exports(module) if not name.startswith("_")}
        return listing

    def exports(self, module):
        """The names of `module` that are its own or that it hands on."""
        if module not in self.exported:
            here = self.modules[module]
            out = {name for name, bound in here.names.items() if bound[0] in OWN}
            out |= here.reexported | (self.listing(module) or set())
            for star in here.stars:
                out |= self.starred(star)
            self.exported[module] = out
        return self.exported[module]

    def is_variable(self, found):
        if found.kind != "value" or not isinstance(found.nodes, ast.Call):
            return False
        maker = self.target(found.module, (), found.nodes.func)
        return bool(maker) and maker.path in TYPE_VARIABLES

    def written(self, found):
        """How a return type names `found`."""
        if self.is_variable(found):
            return found.name
        if found.path == "typing.LiteralString":
            return "str"
        return found.name if self.scope.get(found.name) == found.path else found.path

    def spell(self, module, classes, node):
        """The type `node` as the data writes it, None for `None`."""
        speller = Speller(self, module, classes)
        text = ast.unparse(speller.visit(copy.deepcopy(node)))
        return None if text == "None" else text

    def held(self, module, classes, node):
        """The type a variable annotated `node` holds, None where the
        annotation names none."""
        qualified = isinstance(node, ast.Subscript) and node.value
        if qualified and self.spell(module, classes, qualified) in QUALIFIERS:
            node = node.slice
        text = self.spell(module, classes, node)
        return None if text in QUALIFIERS else text

    def returns(self, module, classes, node):
        """What calling the function `node` gives, None for nothing."""
        text = "typing.Any"
        if node.returns:
            text = self.spell(module, classes, node.returns)
        if isinstance(node, ast.AsyncFunctionDef):
            return f"typing.Coroutine[typing.Any, typing.Any, {text}]"
        return text

    def signature(self, module, classes, node, methods, statics):
        table, known = statics if node.name in IMPLICITLY_STATIC else methods, True
        bound, name = bool(classes), node.name
        for decorator in node.decorator_list:
            maker = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(maker, ast.Attribute) and maker.attr in ACCESSORS:
                return
            found = self.target(module, classes, maker)
            role = found and found.path
            if role in STATIC:
                table = statics
                bound = role != "builtins.staticmethod"
            elif role in ATTRIBUTE:
                name = f".{name}"
            elif role not in UNCHANGED:
                known = False
        # An index is asked for, never a slice.
        sliced = node.name == "__getitem__" and any(
            ast.unparse(arg.annotation).partition("[")[0] == "slice"
            for arg in node.args.posonlyargs + node.args.args
            if arg.annotation
        )
        if not known:
            table.setdefault(name, []).append(None)
        elif not sliced:
            returns = self.returns(module, classes, node)
            table.setdefault(name, []).append(returns)
            owner = classes[-1].path if classes else module
            self.callables(module, classes, node, bound, owner)

    def callables(self, module, classes, node, bound, owner):
        """Note the callables that the function `node` of `owner` takes, by
        their position after the receiver and, unless they are
        positional-only, by their name."""
        args = node.args
        positional = (args.posonlyargs + args.args)[bound:]
        named = args.args[max(bound - len(args.posonlyargs), 0) :] + args.kwonlyargs
        keys = [(str(at), arg) for at, arg in enumerate(positional)]
        for key, arg in keys + [(arg.arg, arg) for arg in named]:
            for found in alternatives(arg.annotation) if arg.annotation else ():
                held = isinstance(found, ast.Subscript) and found.slice
                if (
                    not isinstance(held, ast.Tuple)
                    or len(held.elts) != 2
                    or not isinstance(held.elts[0], ast.List)
                    or not held.elts[0].elts
                    or self.spell(module, classes, found.value) != "typing.Callable"
                ):
                    continue
                params, returns = held.elts[0].elts, held.elts[1]
                taken = tuple(self.spell(module, classes, it) for it in params)
                gives = self.spell(module, classes, returns)
                table = self.callbacks.setdefault(owner, {}).setdefault(node.name, {})
                table.setdefault(key, []).append((taken, gives))

    def base(self, module, classes, node, parent):
        """The base class `parent` with the arguments `node` gives it."""
        if not isinstance(node, ast.Subscript):
            return parent.path
        arguments = Speller(self, module, classes).visit(copy.deepcopy(node.slice))
        return ast.unparse(ast.Subscript(ast.Name(parent.path), arguments))

    def describe(self, types, found, outer=()):
        """Add the class `found`, nested in the classes `outer`, to `types`."""
        declared, used, bases = [], [], []
        for base in (base for node in found.nodes for base in node.bases):
            head = base.value if isinstance(base, ast.Subscript) else base
            parent = self.target(found.module, outer, head)
            variables = [
                name.id
                for name in names(base)
                if (it := self.target(found.module, outer, name))
                and self.is_variable(it)
            ]
            if parent and parent.path in PARAMETERS:
                declared += variables
            else:
                used += variables
            if parent and parent.kind == "class":
                bases.append(self.base(found.module, outer, base, parent))
        kind = types[found.path] = new_type(dict.fromkeys(declared or used))
        kind["bases"] = list(dict.fromkeys(bases))
        classes = (*outer, found)
        nested = set()
        for item in (item for node in found.nodes for item in statements(node.body)):
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                self.signature(
                    found.module, classes, item, kind["methods"], kind["statics"]
                )
            elif isinstance(item, ast.ClassDef):
                nested.add(item.name)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                held = self.held(found.module, classes, item.annotation)
                kind["methods"].setdefault(f".{item.target.id}", []).append(held)
            elif isinstance(item, ast.Assign) and isinstance(item.value, ast.Name):
                for table in kind["methods"], kind["statics"]:
                    if item.value.id in table:
                        table[item.targets[0].id] = table[item.value.id]
        for name in sorted(nested):
            self.describe(types, self.inner(found, name), classes)


class Speller(ast.NodeTransformer):
    """Rewrites a type so that each name in it is the one the data uses,
    with the type aliases it goes through written out."""

    def __init__(self, stubs, module, classes, expanding=()):
        self.stubs = stubs
        self.module = module
        self.classes = classes
        self.expanding = expanding

    def visit_Name(self, node, expand=True):
        found = self.stubs.target(self.module, self.classes, node)
        if found is None or found.kind == "module":
            return self.generic_visit(node)
        alias = found.kind == "value" and found.nodes
        if alias and not isinstance(found.nodes, ast.Call) and expand:
            if found in self.expanding:
                return ast.Name(found.path)
            inner = Speller(self.stubs, found.module, (), (*self.expanding, found))
            return inner.visit(copy.deepcopy(found.nodes))
        return ast.Name(self.stubs.written(found))

    visit_Attribute = visit_Name

    def visit_Subscript(self, node):
        # An alias that takes arguments is left as the name it is.
        if isinstance(node.value, ast.Name | ast.Attribute):
            node.value = self.visit_Name(node.value, expand=False)
        else:
            node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)
        return node


def build(archive):
    with tarfile.open(archive) as tar:
        commit = tar.pax_headers["comment"]
        files = {
            member.name.split("/", 2)[2]: tar.extractfile(member).read().decode("utf-8")
            for member in tar
            if member.isfile() and member.name.split("/")[1] == "stdlib"
        }
    # `name: 3.0-3.11` in VERSIONS is a module the newest Python no longer has.
    removed = {
        line.partition(":")[0]
        for line in files["VERSIONS"].splitlines()
        if line.partition("#")[0].strip()[-1:].isdigit()
    }
    modules = {}
    for path in sorted(files):
        parts = path.removesuffix(".pyi").split("/")
        is_package = parts[-1] == "__init__"
        parts = parts[: len(parts) - is_package]
        name = ".".join(parts)
        gone = any(name == old or name.startswith(f"{old}.") for old in removed)
        if path.endswith(".pyi") and not gone:
            modules[name] = Module(name, is_package, ast.parse(files[path]))
    stubs = Stubs(modules)

    # A class marked `@type_check_only` is not there when the program runs.
    for name, bound in modules["builtins"].names.items():
        if bound[0] not in ("class", "function") or name.startswith("_"):
            continue
        marks = [ast.unparse(mark) for mark in bound[1][0].decorator_list]
        if "type_check_only" not in marks:
            stubs.scope[name] = f"builtins.{name}"

    types, functions = {}, {}
    for module, here in modules.items():
        own = {}
        for name, bound in here.names.items():
            if bound[0] == "class":
                stubs.describe(types, Found("class", module, name, bound[1]))
            elif bound[0] == "function":
                for node in bound[1]:
                    stubs.signature(module, (), node, own, own)
            elif name in here.typed:
                held = stubs.held(module, (), here.typed[name])
                functions[f".{module}.{name}"] = [held]
        functions |= {f"{module}.{name}": returns for name, returns in own.items()}
    tables = [((), functions)]
    for kind in types.values():
        tables += [(kind["params"], kind["methods"]), (kind["params"], kind["statics"])]
    for params, table in tables:
        for name, returns in table.items():
            table[name] = merged(returns, params, types, stubs.scope)
    callbacks = {}
    for owner, name, key, said in (
        (owner, name, key, said)
        for owner, named in stubs.callbacks.items()
        for name, keyed in named.items()
        for key, said in keyed.items()
    ):
        params = types[owner]["params"] if owner in types else ()
        taken = [
            merged(list(column), params, types, stubs.scope)
            for column in zip(*(taken for taken, _ in said))
        ]
        if len({len(taken) for taken, _ in said}) == 1 and all(taken):
            gives = {gives for _, gives in said}
            returns = gives.pop() if len(gives) == 1 else "typing.Any"
            callbacks.setdefault(owner, {}).setdefault(name, {})[key] = {
                "params": taken,
                "returns": returns or "None",
            }
    # A class or function under another module's name: the class as a type
    # with the one it is as its base.
    renamed = dict(RENAMED)
    for module in modules:
        for name in sorted(stubs.exports(module)):
            found = stubs.find(module, name)
            alias = f"{module}.{name}"
            if not found or found.path == alias or alias in renamed:
                continue
            if found.kind == "function":
                functions[alias] = functions[found.path]
            elif f".{found.path}" in functions:
                functions[f".{alias}"] = functions[f".{found.path}"]
            elif found.kind == "class":
                renamed[alias] = found.path
    for alias, path in renamed.items():
        functions.pop(f".{alias}", None)
        params = types[path]["params"]
        types[alias] = new_type(params)
        types[alias]["bases"] = [path + f"[{', '.join(params)}]" * bool(params)]
    return f"typeshed {commit}, stdlib", types, functions, stubs.scope, None, callbacks


if __name__ == "__main__":
    main(build)
