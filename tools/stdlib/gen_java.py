"""Generate java.json from the `java.base` module of a JDK.

usage: gen_java.py JDK_HOME OUTPUT

Reads the module's exported packages with the JDK's own `jimage` and
`javap`; nothing of the JDK is loaded or run besides those two tools.
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import main, new_type, record, split

EXPORT = re.compile(r"^  exports ([\w.]+);$", re.MULTILINE)
CLASS_FILE = re.compile(r"^\s+(\S+)\.class$", re.MULTILINE)
RELEASE = re.compile(r'^(\w+)="(.*)"$', re.MULTILINE)
# `Entry=class java/util/Map$Entry of class java/util/Map` in `InnerClasses`.
NESTED = re.compile(r"// (\w+)=class (\S+) of class (\S+)$")
TYPE_ARGUMENTS = re.compile(r"<[^<>]*>")
NAME = re.compile(r"(?<![\w$.])[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*")
ROOT = "java.lang.Object"


def words(text, separator=" "):
    """`text` cut at each `separator` outside its angle brackets."""
    return split(text, separator, "<", ">")


def binary_name(text):
    """The class a type of javap's names; javap writes the inner class of
    a generic one as `Outer<E>.Inner`."""
    text = text.replace(">.", ">$")
    while "<" in text:
        text = TYPE_ARGUMENTS.sub("", text)
    return text


class Module:
    """The classes javap has printed, by binary name."""

    def __init__(self, jdk):
        self.jdk = jdk
        self.classes = {}
        self.public = set()
        self.nested = {}
        self.interfaces = set()
        # class -> (name, type parameters, parameter types, return type,
        # whether it is abstract) of each of its instance methods
        self.signatures = {}

    def run(self, tool, *arguments):
        command = [self.jdk / "bin" / tool, *arguments]
        return subprocess.run(
            command, capture_output=True, text=True, check=True
        ).stdout

    def read(self, names):
        """Add the public and protected methods of the classes `names`.

        Only `javap -v` prints a method's flags, which tell the bridge
        methods the compiler adds from the ones the source declares, and
        only with `-p` does it list every nested class."""
        output = self.run("javap", "--module", "java.base", "-p", "-v", *names)
        name = kind = member = None
        heading = inside = False
        for line in output.splitlines():
            if line.startswith("Classfile "):
                heading = True
            elif heading and not line.startswith(" "):
                heading = False
                name, kind = self.declare(words(line))
            elif line in ("{", "}"):
                inside = line == "{"
            elif inside and line.startswith("  ") and not line.startswith("   "):
                member = line
            elif inside and member and line.startswith("    flags: "):
                head, call, tail = member.partition("(")
                *rest, returns, method = [""] + words(head)
                accessible = "ACC_PUBLIC" in line or "ACC_PROTECTED" in line
                if (
                    call
                    and method != name
                    and accessible
                    and "ACC_SYNTHETIC" not in line
                ):
                    table = kind["statics" if "ACC_STATIC" in line else "methods"]
                    record(table, method, None if returns == "void" else returns)
                    if "ACC_STATIC" not in line:
                        generic = rest[-1][1:-1] if rest[-1].startswith("<") else ""
                        self.signatures.setdefault(name, []).append(
                            (
                                method,
                                [words(part)[0] for part in words(generic, ",")],
                                words(tail.rpartition(")")[0], ","),
                                returns,
                                "ACC_ABSTRACT" in line,
                            )
                        )
                member = None
            elif not inside and (found := NESTED.search(line)):
                simple, inner, outer = found.groups()
                self.nested[inner.replace("/", ".")] = outer.replace("/", "."), simple

    def declare(self, heading):
        at = next(n for n, word in enumerate(heading) if word in ("class", "interface"))
        name, _, params = heading[at + 1].partition("<")
        kind = self.classes[name] = new_type(
            words(part)[0] for part in words(params[:-1], ",")
        )
        kind["bases"] = [
            base
            for word in heading[at + 2 :]
            if word not in ("extends", "implements")
            for base in words(word, ",")
        ]
        if "public" in heading[:at]:
            self.public.add(name)
        if heading[at] == "interface":
            self.interfaces.add(name)
        return name, kind

    def callable(self, name):
        """``(name, parameter types, return type)`` of the one abstract
        method the interface ``name`` has, its own or a base's, that
        `Object` does not implement; None for any other class."""
        if name not in self.interfaces:
            return None
        own = self.signatures.get(name, ())
        found = {
            (method, len(takes)): (method, takes, returns)
            for method, _, takes, returns, abstract in own
            if abstract and method not in self.classes[ROOT]["methods"]
        }
        declared = {(method, len(takes)) for method, _, takes, *_ in own}
        for base in self.classes[name]["bases"]:
            inherited = self.callable(binary_name(base))
            if inherited and (inherited[0], len(inherited[1])) not in declared:
                found[inherited[0], len(inherited[1])] = self.given(base, inherited)
        return next(iter(found.values())) if len(found) == 1 else None

    def given(self, written, signature):
        """``signature``, what ``callable`` gives for the interface of the
        type ``written``, with the type arguments of ``written`` in place
        of the interface's parameters. A wildcard is its bound."""
        head, _, arguments = written.partition("<")
        passed = [
            re.sub(r"^\? (?:super|extends) ", "", argument)
            for argument in words(arguments[:-1], ",")
        ]
        values = dict(zip(self.classes[binary_name(head)]["params"], passed))
        method, takes, returns = signature
        return (
            method,
            [NAME.sub(lambda found: values.get(found[0], found[0]), t) for t in takes],
            NAME.sub(lambda found: values.get(found[0], found[0]), returns),
        )

    def exported(self, name):
        """Whether the class and every class it is nested in are public."""
        outer = self.nested.get(name, [None])[0]
        return name in self.public and (outer is None or self.exported(outer))

    def spell(self, name):
        """The name the source gives the class of binary name `name`."""
        if name in self.nested:
            outer, simple = self.nested[name]
            return f"{self.spell(outer)}.{simple}"
        return name


def build(jdk):
    module = Module(jdk)
    exports = EXPORT.findall(
        module.run("javap", "--module", "java.base", "module-info")
    )
    listing = module.run(
        "jimage", "list", "--include", "regex:/java.base/.*", jdk / "lib" / "modules"
    )
    names = [name.replace("/", ".") for name in CLASS_FILE.findall(listing)]
    module.read(name for name in names if name.rpartition(".")[0] in exports)

    # A type of an exported package is public along with every class it is
    # nested in; what such a type extends is kept whatever its access, for
    # the methods it hands down.
    kept = [name for name in module.classes if module.exported(name)]
    for name in kept:
        kind = module.classes[name]
        kind["bases"] = [base for base in kind["bases"] if binary_name(base) != ROOT]
        for base in map(binary_name, kind["bases"]):
            if base not in module.classes:
                module.read([base])
            if base not in kept:
                kept.append(base)

    scope = {
        name.removeprefix("java.lang."): name
        for name in kept
        if module.exported(name) and re.fullmatch(r"java\.lang\.\w+", name)
    }
    bare = {name: simple for simple, name in scope.items()}

    def spell(found):
        name = module.spell(found[0])
        return bare.get(name, name)

    # The callables each method takes, by argument position; overloads that
    # take different ones in a position leave none there.
    callbacks = {}
    for name in kept:
        taken = {}
        for method, generic, takes, _, _ in module.signatures.get(name, ()):
            for position, written in enumerate(takes):
                signature = module.callable(binary_name(written))
                if signature:
                    _, params, returns = module.given(written, signature)
                    record(
                        taken.setdefault(method, {}),
                        str(position),
                        {
                            "params": [NAME.sub(spell, param) for param in params],
                            "returns": returns if returns in generic else None,
                        },
                    )
        taken = {
            method: agreed
            for method, passed in taken.items()
            if (agreed := {at: it for at, it in passed.items() if it})
        }
        if taken:
            callbacks[module.spell(name)] = taken

    types = {}
    for name in kept:
        kind = module.classes[name]
        for at, base in enumerate(kind["bases"]):
            head, bracket, arguments = base.partition("<")
            kind["bases"][at] = (
                module.spell(head) + bracket + NAME.sub(spell, arguments)
            )
        for table in kind["methods"], kind["statics"]:
            for method, returns in table.items():
                table[method] = returns and NAME.sub(spell, returns)
        types[module.spell(name)] = kind

    release = dict(RELEASE.findall((jdk / "release").read_text(encoding="utf-8")))
    source = (
        f"java.base {release['JAVA_RUNTIME_VERSION']} ({release['IMPLEMENTOR']}),"
        " exported packages, read with javap"
    )
    # The parameters of a functional interface's one abstract method
    functional = {
        module.spell(name): len(signature[1])
        for name in kept
        if module.exported(name) and (signature := module.callable(name))
    }
    return source, types, {}, scope, None, callbacks, None, functional


if __name__ == "__main__":
    main(build)
