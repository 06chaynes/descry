"""Generate rust.json from rustdoc's JSON for the Rust standard library.

usage: gen_rust.py JSON_DIR OUTPUT

JSON_DIR holds `core.json`, `alloc.json` and `std.json`, the `rust-docs-json`
component of one nightly toolchain. Every type, trait, alias and function a
path from one of the three crate roots reaches is written, but for the
functions of `core::arch` and `std::arch`, one for each instruction of each
processor:

- A type is keyed by its shortest path in `std`, failing that in `alloc`,
  failing that in `core`; a primitive by its name (`str`), or by how it is
  written where it has none (`[T]`, `[T; N]`). Each other path it is
  importable by is a type with the same `params` and the keyed type as its
  one base, as is a `type` alias with the type it stands for.
- `params` are the type and const parameters; lifetimes are left out
  everywhere. A parameter of an impl or of a method that no parameter of
  the type stands for is written `_`.
- `bases` are the traits a type implements, written with their type
  arguments and associated types (`Iterator<Item = &T>`), and the type it
  dereferences to; an array has the methods of a slice. They are in the
  order a method is looked for: the traits of the prelude, what the type
  dereferences to, then the traits that are in scope only after a `use`.
  A blanket impl is a base of every type when it bounds nothing, and a
  base of the trait it is bounded by otherwise.
- A type parameter bounded as a callable, `F: FnMut(A) -> B`, stands for
  what the callable returns, and so does `B`: `Option::map` returns
  `Option<F>` and `Map<I, F>` is an `Iterator<Item = F>`. `callbacks` has
  the argument such a method takes the callable as, with `A`. `variants`
  has what the variants of an enum hold.
- An impl for `&T` counts as one for `T` where `T` has none of that trait.
  A trait without methods is no base, nor is one a pointer implements only
  where what it points at does (`Box<I>: Iterator`): dereferencing finds
  those methods.
- `scope` is the prelude of the newest edition and the primitives that have
  a name: a type among them is written by its name there, any other by its
  key. A derive macro of the prelude is in `scope` too, as the trait it
  implements (`Debug`).
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import main, new_type, record

# In the order a type's canonical path is looked for.
CRATES = ("std", "alloc", "core")
TYPE_KINDS = ("struct", "enum", "union", "trait", "type_alias")
# The primitive types that have no name: how they are written, and their
# parameters.
UNNAMED = {
    "slice": ("[T]", ["T"]),
    "array": ("[T; N]", ["T", "N"]),
    "tuple": ("(T,)", []),
    "unit": ("()", []),
    "never": ("!", []),
    "reference": ("&T", ["T"]),
    "pointer": ("*const T", ["T"]),
    "fn": ("fn()", []),
}
DEREF = ("trait", ("core", "ops", "deref", "Deref"))
_NO_GENERICS = {"params": [], "where_predicates": []}


def _identity(entry):
    """What a `paths` entry names, the same from whichever crate refers to it."""
    if entry["kind"] == "primitive":
        return ("primitive", (entry["path"][-1],))
    return (entry["kind"], tuple(entry["path"]))


def _bounds(generics):
    """``(name, trait)`` for each bound `generics` puts on a type parameter."""
    predicates = [
        (parameter["name"], parameter["kind"]["type"]["bounds"])
        for parameter in generics["params"]
        if "type" in parameter["kind"]
    ]
    for predicate in generics["where_predicates"]:
        bound = predicate.get("bound_predicate")
        if bound and "generic" in bound["type"]:
            predicates.append((bound["type"]["generic"], bound["bounds"]))
    for name, bounds in predicates:
        for bound in bounds:
            if bound.get("trait_bound", {}).get("modifier", "maybe") != "maybe":
                yield name, bound["trait_bound"]["trait"]


def _bounded(generics):
    """Names of the type parameters `generics` bounds by a trait."""
    return {(name, trait["id"]) for name, trait in _bounds(generics)}


def _callables(generics):
    """Name of each type parameter `generics` bounds as a callable, `F:
    FnMut(A) -> B`, with the callable's parameter types and return type."""
    called = {}
    for name, trait in _bounds(generics):
        signature = (trait["args"] or {}).get("parenthesized")
        if signature:
            called[name] = (signature["inputs"], signature["output"])
    return called


def _agreed(written):
    """What every impl writes at one place, `_` where they differ."""
    return written[0] if len(set(written)) == 1 else "_"


class Generator:
    def __init__(self, directory):
        self.crates = {
            name: json.loads((directory / f"{name}.json").read_text("utf-8"))
            for name in CRATES
        }
        self.defined = {}  # identity -> (crate, item) of what a crate defines
        for name, crate in self.crates.items():
            for item_id, entry in crate["paths"].items():
                if entry["crate_id"] == 0 and item_id in crate["index"]:
                    item = crate["index"][item_id]
                    self.defined.setdefault(_identity(entry), (name, item))
        self.paths = {}  # identity -> the paths it is importable by
        for name, crate in self.crates.items():
            self._walk(name, crate["index"][str(crate["root"])], [name], ())
        self.keys = {
            identity: min(
                paths,
                key=lambda p: (CRATES.index(p.split("::")[0]), p.count("::"), p),
            )
            for identity, paths in self.paths.items()
        }
        self.keys.update(
            (identity, UNNAMED.get(identity[1][0], identity[1])[0])
            for identity in self.defined
            if identity[0] == "primitive"
        )
        prelude = max(
            path.rpartition("::")[0]
            for paths in self.paths.values()
            for path in paths
            if re.match(r"std::prelude::rust_\d+::\w+$", path)
        )
        self.scope = {
            path.rpartition("::")[2]: self.keys[identity]
            for identity, paths in self.paths.items()
            for path in paths
            if identity[0] in TYPE_KINDS and path.rpartition("::")[0] == prelude
        }
        self.scope.update(
            (path.rpartition("::")[2], self.keys[identity])
            for identity, paths in self.paths.items()
            for path in paths
            if identity[0] == "primitive"
        )
        self.short = {key: name for name, key in self.scope.items()}
        # A derive macro of the prelude is named as the trait it implements,
        # which has its path.
        traits = {
            path: self.keys[identity]
            for identity, paths in self.paths.items()
            for path in paths
            if identity[0] == "trait"
        }
        self.scope.update(
            (path.rpartition("::")[2], traits[shared])
            for identity, paths in self.paths.items()
            for path in paths
            for shared in paths
            if identity[0] == "proc_derive"
            and path.rpartition("::")[0] == prelude
            and shared in traits
        )
        self.types = {}  # key -> entry, its bases still to be merged
        self.variants = {}
        self.callbacks = {}
        # key -> [(trait, arguments, associated types, bounded parameters, borrow)]
        self.impls = {}

    def _reference(self, crate, item_id):
        entry = self.crates[crate]["paths"].get(str(item_id))
        return entry and _identity(entry)

    def _walk(self, crate, module, path, inside):
        """Record the path of every public item below ``module``, following
        re-exports into the crate that defines what they name. ``inside``
        are the modules being walked, which a re-export may name again."""
        index = self.crates[crate]["index"]
        for item_id in module["inner"]["module"]["items"]:
            item = index[str(item_id)]
            kind = next(iter(item["inner"]))
            if item["visibility"] != "public" or kind == "primitive":
                continue
            name, glob = item["name"], False
            if kind == "use":
                use = item["inner"]["use"]
                name, glob = use["name"], use["is_glob"]
                if use["id"] is None:
                    identity = ("primitive", (name,))
                else:
                    identity = self._reference(crate, use["id"])
            else:
                identity = self._reference(crate, item_id)
            if identity not in self.defined:
                continue
            if identity[0] == "module":
                if identity not in inside:
                    below = path if glob else [*path, name]
                    self._walk(*self.defined[identity], below, (*inside, identity))
            elif not glob:
                self.paths.setdefault(identity, set()).add("::".join([*path, name]))

    def _named(self, crate, path, names):
        """A path to a type or trait with its arguments, `_` for one that
        cannot be imported."""
        key = self.keys.get(self._reference(crate, path["id"]))
        if key is None:
            return "_"
        text = self.short.get(key, key)
        arguments = path.get("args") or {}
        if "parenthesized" in arguments:
            inputs = arguments["parenthesized"]["inputs"]
            output = arguments["parenthesized"]["output"]
            text += f"({', '.join(self._type(crate, t, names) for t in inputs)})"
            return text + (f" -> {self._type(crate, output, names)}" if output else "")
        written = self._arguments(crate, path, names)
        for constraint in arguments.get("angle_bracketed", {}).get("constraints", ()):
            bound = constraint["binding"].get("equality", {}).get("type")
            if bound:
                written.append(
                    f"{constraint['name']} = {self._type(crate, bound, names)}"
                )
        return f"{text}<{', '.join(written)}>" if written else text

    def _arguments(self, crate, path, names):
        """The type and const arguments of a path, as written."""
        arguments = (path.get("args") or {}).get("angle_bracketed", {}).get("args", ())
        return [
            self._type(crate, argument["type"], names) if "type" in argument else "_"
            for argument in arguments
            if "lifetime" not in argument
        ]

    def _type(self, crate, node, names):
        """A rustdoc type as Rust writes it. ``names`` maps the generic
        parameters that can be named to how they are written."""
        kind, inner = next(iter(node.items()))
        if kind == "generic":
            return names.get(inner, "_")
        if kind == "primitive":
            return "!" if inner == "never" else inner
        if kind == "resolved_path":
            return self._named(crate, inner, names)
        if kind == "borrowed_ref":
            target = self._type(crate, inner["type"], names)
            return f"&mut {target}" if inner["is_mutable"] else f"&{target}"
        if kind == "raw_pointer":
            target = self._type(crate, inner["type"], names)
            return f"*mut {target}" if inner["is_mutable"] else f"*const {target}"
        if kind == "slice":
            return f"[{self._type(crate, inner, names)}]"
        if kind == "array":
            length = names.get(inner["len"], "_")
            return f"[{self._type(crate, inner['type'], names)}; {length}]"
        if kind == "tuple":
            parts = [self._type(crate, part, names) for part in inner]
            return f"({', '.join(parts)}{',' if len(parts) == 1 else ''})"
        if kind == "qualified_path":
            owner = self._type(crate, inner["self_type"], names)
            named = owner != "_" and owner.isidentifier()
            return f"{owner}::{inner['name']}" if named else "_"
        if kind in ("dyn_trait", "impl_trait"):
            bounds = inner["traits"] if kind == "dyn_trait" else inner
            traits = [b.get("trait_bound", b).get("trait") for b in bounds]
            written = [self._named(crate, t, names) for t in traits if t]
            return f"{kind[:-6]} {' + '.join(written)}"
        if kind == "function_pointer":
            inputs = [self._type(crate, t, names) for _, t in inner["sig"]["inputs"]]
            output = self._returns(crate, inner["sig"], names)
            return f"fn({', '.join(inputs)})" + (f" -> {output}" if output else "")
        if kind == "pat":
            return self._type(crate, inner["type"], names)
        return "_"

    def _returns(self, crate, signature, names):
        output = signature["output"] and self._type(crate, signature["output"], names)
        return None if output in (None, "()", "!") else output

    def _entry(self, identity):
        """The entry of the type or trait ``identity``, None for one that
        cannot be imported."""
        key = self.keys.get(identity)
        if key is None or identity[0] not in (*TYPE_KINDS, "primitive"):
            return None
        if key not in self.types:
            item = self.defined[identity][1]
            generics = next(iter(item["inner"].values())).get("generics")
            params = UNNAMED.get(identity[1][0], (key, []))[1]
            if generics:
                params = [
                    parameter["name"]
                    for parameter in generics["params"]
                    if "lifetime" not in parameter["kind"]
                ]
            self.types[key] = new_type(params)
            self.impls[key] = []
        return self.types[key]

    def _add_functions(self, crate, key, item_ids, names, public_only):
        index = self.crates[crate]["index"]
        for item_id in item_ids:
            item = index[str(item_id)]
            if "function" not in item["inner"]:
                continue
            if public_only and item["visibility"] != "public":
                continue
            signature = item["inner"]["function"]["sig"]
            called = _callables(item["inner"]["function"]["generics"])
            # A callable and what it returns are both written as the callable.
            named = dict(names)
            for name, (_, output) in called.items():
                named[name] = name
                if output and output.get("generic", "Self") not in names:
                    named[output["generic"]] = name
            returns = self._returns(crate, signature, named)
            inputs = [node for name, node in signature["inputs"] if name != "self"]
            takes = {}
            for position, node in enumerate(inputs):
                while "borrowed_ref" in node:
                    node = node["borrowed_ref"]["type"]
                if node.get("generic") in called:
                    params, _ = called[node["generic"]]
                    takes[str(position)] = {
                        "params": [self._type(crate, p, named) for p in params],
                        "returns": node["generic"],
                    }
            takes_self = len(inputs) < len(signature["inputs"])
            table = self.types[key]["methods" if takes_self else "statics"]
            # Impls for different type arguments may each define the name.
            record(table, item["name"], returns)
            if takes:
                record(self.callbacks.setdefault(key, {}), item["name"], takes)

    def _implementor(self, crate, node):
        """``(identity, type arguments, borrow)`` of the type an impl is
        for: ``("generic", name)`` as the identity of a blanket impl, and
        0, 1 or 2 for an impl for the type, for `&` and for `&mut` of it."""
        borrow = 0
        while "borrowed_ref" in node:
            borrow = max(borrow, 2 if node["borrowed_ref"]["is_mutable"] else 1)
            node = node["borrowed_ref"]["type"]
        identity, arguments = self._implementor_type(crate, node)
        if borrow and identity and identity[0] == "generic":
            identity = None
        return identity, arguments, borrow

    def _implementor_type(self, crate, node):
        kind, inner = next(iter(node.items()))
        if kind == "generic":
            return ("generic", inner), []
        if kind == "resolved_path":
            arguments = (inner["args"] or {}).get("angle_bracketed", {}).get("args", ())
            return self._reference(crate, inner["id"]), [
                argument.get("type") or {"generic": argument["const"]["expr"]}
                for argument in arguments
                if "lifetime" not in argument
            ]
        if kind == "dyn_trait":
            return self._reference(crate, inner["traits"][0]["trait"]["id"]), []
        if kind == "array":
            return ("primitive", ("array",)), [inner["type"], {"generic": inner["len"]}]
        if kind in ("slice", "raw_pointer"):
            target = inner if kind == "slice" else inner["type"]
            name = "slice" if kind == "slice" else "pointer"
            return ("primitive", (name,)), [target]
        name = {
            "primitive": inner,
            "tuple": "tuple" if inner else "unit",
            "function_pointer": "fn",
        }.get(kind)
        return (name and ("primitive", (name,))), []

    def _add_trait(self, crate, identity, item):
        entry = self._entry(identity)
        trait = item["inner"]["trait"]
        names = {name: name for name in ("Self", *entry["params"])}
        self._add_functions(crate, self.keys[identity], trait["items"], names, False)
        for bound in trait["bounds"]:
            if "trait_bound" in bound:
                path = bound["trait_bound"]["trait"]
                self.impls[self.keys[identity]].append(
                    (
                        self._reference(crate, path["id"]),
                        self._arguments(crate, path, names),
                        {},
                        set(),
                        0,
                    )
                )

    def _add_impl(self, crate, impl):
        identity, arguments, borrow = self._implementor(crate, impl["for"])
        index = self.crates[crate]["index"]
        bounded = _bounded(impl["generics"])
        if identity and identity[0] == "generic":
            names = {"Self": "Self", identity[1]: "Self"}
            bounds = {
                self.keys.get(self._reference(crate, trait))
                for name, trait in bounded
                if name == identity[1]
            }
            # A blanket impl is a base of the one trait it is bounded by.
            keys = list(bounds)
            if not bounds:
                keys = [
                    self.keys[other]
                    for other in self.defined
                    if other[0] in ("struct", "enum", "union", "primitive")
                    and other in self.keys
                ]
            elif len(bounds) > 1 or keys[0] not in self.types:
                return
        else:
            entry = self._entry(identity)
            if entry is None:
                return
            names = {"Self": "Self"}
            for parameter, argument in zip(entry["params"], arguments):
                while "borrowed_ref" in argument:
                    argument = argument["borrowed_ref"]["type"]
                if "generic" in argument:
                    names.setdefault(argument["generic"], parameter)
            keys = [self.keys[identity]]
            for name, (_, output) in _callables(impl["generics"]).items():
                if name in names and output and "generic" in output:
                    names.setdefault(output["generic"], names[name])
            if impl["trait"] is None:
                self._add_functions(crate, keys[0], impl["items"], names, True)
                return
        associated = {
            index[str(item_id)]["name"]: self._type(
                crate, index[str(item_id)]["inner"]["assoc_type"]["type"], names
            )
            for item_id in impl["items"]
            if index[str(item_id)]["inner"].get("assoc_type", {}).get("type")
        }
        record = (
            self._reference(crate, impl["trait"]["id"]),
            self._arguments(crate, impl["trait"], names),
            associated,
            {names[name] for name, _ in bounded if name in names},
            borrow,
        )
        for key in keys:
            self.impls[key].append(record)

    def _bases(self, key, own_bounds):
        """The merged bases of the entry ``key``: each trait once, an
        argument its impls differ in as `_`, and what the type dereferences
        to, between the traits of the prelude and the others."""
        target = next(
            (
                associated.get("Target")
                for trait, _, associated, _, _ in self.impls[key]
                if trait == DEREF
            ),
            None,
        )
        merged = {}
        for trait, arguments, associated, bounded, borrow in self.impls[key]:
            entry = self._entry(trait)
            # Without methods of its own a trait adds none.
            if entry is None or not (entry["methods"] or entry["statics"]):
                continue
            # A pointer has the trait only where what it points at has it,
            # and that one's methods are reached by dereferencing.
            if target in bounded - own_bounds:
                continue
            merged.setdefault(self.keys[trait], []).append(
                (borrow, arguments, associated)
            )
        bases = {True: [], False: []}
        for name, impls in sorted(merged.items()):
            # The impls for the type itself, else those for a reference to it.
            least = min(impl[0] for impl in impls)
            impls = [impl[1:] for impl in impls if impl[0] == least]
            written = [
                _agreed(position) for position in zip(*(a for a, _ in impls))
            ] + [
                f"{assoc} = {_agreed([types.get(assoc) for _, types in impls])}"
                for assoc in sorted(set().union(*(types for _, types in impls)))
            ]
            text = self.short.get(name, name)
            bases[name in self.short].append(
                f"{text}<{', '.join(written)}>" if written else text
            )
        if key == "[T; N]":
            target = "[T]"
        if target and target != "_":
            bases[True].append(target.removeprefix("&mut ").removeprefix("&"))
        return bases[True] + bases[False]

    def _held(self, crate, kind, names):
        """The types of the values a variant of ``kind`` holds: a list for
        `V(A, B)`, by name for `V { a: A }`."""
        index = self.crates[crate]["index"]
        if "tuple" in kind:
            fields = [index.get(str(field_id)) for field_id in kind["tuple"]]
            return [
                self._type(crate, f["inner"]["struct_field"], names) if f else "_"
                for f in fields
            ]
        if "struct" in kind:
            fields = [index[str(field_id)] for field_id in kind["struct"]["fields"]]
            return {
                f["name"]: self._type(crate, f["inner"]["struct_field"], names)
                for f in fields
            }
        return None

    def build(self):
        own_bounds = {}  # key -> the parameters the type's declaration bounds
        aliases = {}
        for identity, (crate, item) in sorted(self.defined.items()):
            entry = self._entry(identity)
            if entry is None:
                continue
            key = self.keys[identity]
            names = {name: name for name in entry["params"]}
            inner = next(iter(item["inner"].values()))
            own_bounds[key] = {
                name for name, _ in _bounded(inner.get("generics") or _NO_GENERICS)
            }
            if identity[0] == "trait":
                self._add_trait(crate, identity, item)
            elif identity[0] == "type_alias":
                aliases[key] = self._type(crate, inner["type"], names)
            for variant_id in inner.get("variants", ()):
                variant = self.crates[crate]["index"][str(variant_id)]
                held = self._held(crate, variant["inner"]["variant"]["kind"], names)
                if held:
                    self.variants.setdefault(key, {})[variant["name"]] = held
        for crate, data in self.crates.items():
            for item in data["index"].values():
                impl = isinstance(item["inner"], dict) and item["inner"].get("impl")
                # rustdoc repeats a blanket impl for each type it applies to
                # in the crate that defines the type only.
                if (
                    impl
                    and not impl["is_synthetic"]
                    and not impl["is_negative"]
                    and not impl["blanket_impl"]
                ):
                    self._add_impl(crate, impl)
        for key, entry in self.types.items():
            entry["bases"] = self._bases(key, own_bounds.get(key, set()))
            if aliases.get(key, "_") != "_":
                entry["bases"] = [aliases[key]]
        functions = {}
        for identity, paths in self.paths.items():
            key = self.keys[identity]
            if identity[0] == "function":
                crate, item = self.defined[identity]
                signature = item["inner"]["function"]["sig"]
                returns = self._returns(crate, signature, {})
                functions.update(
                    (path, returns) for path in paths if path.split("::")[1] != "arch"
                )
            elif key in self.types:
                params = self.types[key]["params"]
                text = self.short.get(key, key)
                target = f"{text}<{', '.join(params)}>" if params else text
                for path in paths - {key}:
                    self.types[path] = new_type(params)
                    self.types[path]["bases"] = [target]
        versions = {
            " ".join(crate["crate_version"].split()) for crate in self.crates.values()
        }
        formats = {crate["format_version"] for crate in self.crates.values()}
        source = (
            f"rustdoc JSON (format {', '.join(map(str, sorted(formats)))}) of "
            f"core, alloc and std {', '.join(sorted(versions))}"
        )
        callbacks = {
            key: {name: takes for name, takes in methods.items() if takes}
            for key, methods in self.callbacks.items()
        }
        return source, self.types, functions, self.scope, self.variants, callbacks


def build(directory):
    return Generator(directory).build()


if __name__ == "__main__":
    main(build)
