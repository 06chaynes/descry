"""Generate go.json from the Go project's API listing.

usage: gen_go.py GO_DIR OUTPUT

GO_DIR holds, at their paths in the Go repository at one release,
`api/go1.txt`, `api/go1.1.txt` and so on, the exported API each release
added with a declaration a line, and `src/builtin/builtin.go`, the
predeclared identifiers.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import main, new_type, record, split

# `pkg net/http (linux-386), method (*Client) Get(string) (*Response, error) #123`
LINE = re.compile(r"pkg (\S+)(?: \([\w-]+\))?, (func|method|type) (.*?)(?: #\d+)?")
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
NAME = re.compile(r"(?<![\w.$])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?")
MAJOR_VERSION = re.compile(r"v\d+")
BRACKETS = "([{", ")]}"

# `builtin.go` declares the predeclared identifiers as if they were a
# package, and says of the types that only stand for others:
STAND_IN = re.compile(
    r"^// (\w+) is here for the purposes of documentation only", re.MULTILINE
)
PREDECLARED_TYPE = re.compile(r"^type (\w+) (= )?(\S+)", re.MULTILINE)
PREDECLARED_METHODS = re.compile(
    r"^type (\w+) interface \{\n\t((?:.*\n)+?)\}", re.MULTILINE
)
PREDECLARED_FUNCTION = re.compile(r"^func (.+)$", re.MULTILINE)
# The types Go writes without a name: a slice or an array, a map, a channel
# and the several results of a function.
UNNAMED = ("[]", "map", "chan", "()")


def after(text, at):
    """The index after the bracket that closes the one at `at`."""
    depth = 0
    while True:
        depth += (text[at] in BRACKETS[0]) - (text[at] in BRACKETS[1])
        at += 1
        if depth == 0:
            return at


def signature(text):
    """The name and the results of `Name[$0 any](int) (string, error)`."""
    name = IDENTIFIER.match(text)[0]
    at = len(name)
    if text[at] == "[":
        at = after(text, at)
    return name, text[after(text, at) :].strip() or None


def package_name(path):
    """The name a package is used under, which the listing writes before
    the types of other packages."""
    *front, last = path.split("/")
    return front[-1] if MAJOR_VERSION.fullmatch(last) else last


def predeclared(text):
    """The types, functions and scope `builtin.go` declares. A result is
    kept when every type it names is one of them, not a stand-in or a type
    parameter."""
    stand_ins = set(STAND_IN.findall(text))
    types = {
        name: new_type()
        for name, _, _ in PREDECLARED_TYPE.findall(text)
        if name not in stand_ins
    }

    def returns(result):
        known = result and all(name in types for name in IDENTIFIER.findall(result))
        return result if known else None

    for name, alias, target in PREDECLARED_TYPE.findall(text):
        if alias and target in types:
            types[name]["bases"] = [target]
    for name, body in PREDECLARED_METHODS.findall(text):
        for method, result in map(signature, body.split("\n\t")):
            types[name]["methods"][method] = returns(result)
    functions = {
        name: returns(result)
        for name, result in map(signature, PREDECLARED_FUNCTION.findall(text))
    }
    return types, functions, {name: name for name in (*types, *functions)}


def build(directory):
    files = sorted(
        (directory / "api").glob("go1*.txt"),
        key=lambda path: [int(part) for part in path.stem[2:].split(".")],
    )
    builtin = (directory / "src" / "builtin" / "builtin.go").read_text(encoding="utf-8")
    types, functions, scope = predeclared(builtin)
    types.update((name, new_type()) for name in UNNAMED)
    # (package, type) -> the release that added it; a name is resolved
    # among the types that existed when the line using it was written.
    added = {}
    named = {}
    results, embedded = [], []
    for release, path in enumerate(files):
        for line in path.read_text(encoding="utf-8").splitlines():
            found = LINE.fullmatch(line)
            if not found or "//deprecated" in line:
                continue
            package, kind, text = found.groups()
            named.setdefault(package_name(package), set()).add(package)
            if kind == "func":
                name, returns = signature(text)
                results.append(
                    (functions, f"{package}.{name}", package, release, returns)
                )
            elif kind == "method":
                end = after(text, 0)
                receiver = text[1 : end - 1].lstrip("*").partition("[")[0]
                table = types.setdefault(f"{package}.{receiver}", new_type())["methods"]
                name, returns = signature(text[end + 1 :])
                results.append((table, name, package, release, returns))
            else:
                name = IDENTIFIER.match(text)[0]
                this = types.setdefault(f"{package}.{name}", new_type())
                added.setdefault((package, name), release)
                rest = text[len(name) :]
                if rest.startswith("["):
                    end = after(rest, 0)
                    this["params"] = [
                        part.split()[0]
                        for part in split(rest[1 : end - 1], ",", *BRACKETS)
                    ]
                    rest = rest[end:]
                shape, _, member = rest.strip().partition(", ")
                if shape == "interface" and "(" in member:
                    name, returns = signature(member)
                    results.append((this["methods"], name, package, release, returns))
                elif shape == "struct" and member.startswith("embedded "):
                    embedded.append(
                        (this, package, release, member.removeprefix("embedded "))
                    )
                elif shape.startswith("= "):
                    embedded.append((this, package, release, shape[2:]))

    def qualified(package, release, text):
        """`text` with each type named by its package's import path."""

        def spell(found):
            head, dot, name = found[0].rpartition(".")
            if not dot:
                return f"{package}.{name}" if (package, name) in added else name
            paths = [
                path
                for path in named.get(head, ())
                if added.get((path, name), release + 1) <= release
            ]
            return f"{paths[0]}.{name}" if len(paths) == 1 else found[0]

        return NAME.sub(spell, text)

    for table, name, package, release, returns in results:
        record(table, name, returns and qualified(package, release, returns))
    for this, package, release, text in embedded:
        base = qualified(package, release, text.lstrip("*").partition("[")[0])
        if base in types and base not in this["bases"]:
            this["bases"].append(base)

    newest = files[-1].stem.removeprefix("go")
    source = (
        f"Go {newest}, api/{files[0].name} to api/{files[-1].name} and"
        " src/builtin/builtin.go"
    )
    return source, types, functions, scope


if __name__ == "__main__":
    main(build)
