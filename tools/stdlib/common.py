"""What the gen_*.py generators share."""

import json
import re
import subprocess
import sys
from pathlib import Path


def new_type(params=()) -> dict:
    return {"params": list(params), "bases": [], "methods": {}, "statics": {}}


def split(text: str, separator: str, opening: str, closing: str) -> list[str]:
    """`text` cut at each `separator` outside its brackets, without the
    empty pieces."""
    out, depth, start = [], 0, 0
    for at, char in enumerate(text + separator):
        depth += (char in opening) - (char in closing)
        if char == separator and depth == 0:
            out.append(text[start:at].strip())
            start = at + 1
    return [piece for piece in out if piece]


def closing(text, at: int, opening: str, closes: str) -> int | None:
    """The index after the bracket that closes the one at `at` in `text`, a
    string or a list of tokens; None where none does."""
    depth = 0
    for end in range(at, len(text)):
        depth += (text[end] in opening) - (text[end] in closes)
        if depth == 0:
            return end + 1
    return None


def lineage(name: str, types: dict, opening: str, scope: dict) -> list[str]:
    """The type `name` and the types it derives from, nearest first: each
    base without the type arguments written from `opening`, under the key
    `scope` gives it where it gives one."""
    found = [name]
    for each in found:
        for base in types.get(each, {}).get("bases", ()):
            head = base.partition(opening)[0]
            head = scope.get(head, head)
            if head not in found:
                found.append(head)
    return found


def record(table: dict, name: str, returns: str | None) -> None:
    """Add one signature of `name`; signatures that disagree leave None."""
    if table.setdefault(name, returns) != returns:
        table[name] = None


def dump_ast(command: list[str], source: Path, headers, declarations=False) -> dict:
    """The AST the compiler `command` prints for a unit, written to
    `source`, that includes `headers`. Read one top-level declaration at a
    time; with `declarations` without what a generator of declarations
    never reads: ids, source locations and expressions."""
    source.write_text(
        "".join(f"#include <{header}>\n" for header in headers) + "int main(){}\n"
    )
    errors = source.with_suffix(".err")
    tree, lines = None, []
    with (
        errors.open("w") as stderr,
        subprocess.Popen(
            [*command, "-fsyntax-only", "-Xclang", "-ast-dump=json", str(source)],
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        ) as run,
    ):
        # The unit's declarations are the elements of its `inner`, each
        # printed from a line indented four spaces to one that closes it.
        for line in run.stdout:
            if tree is None:
                if line == '  "inner": [\n':
                    tree = {"kind": "TranslationUnitDecl", "inner": []}
                continue
            lines.append(line)
            if line.rstrip(",\n") == "    }":
                hook = _declaration if declarations else None
                text = "".join(lines).rstrip().removesuffix(",")
                tree["inner"].append(json.loads(text, object_hook=hook))
                lines = []
    if run.returncode or tree is None:
        raise RuntimeError(
            f"{command[0]} -ast-dump failed: {errors.read_text()[:2000]}"
        )
    return tree


def returned(written: str) -> str:
    """The type a Clang `type` string returns, before its `(...)`, without a
    leading `const`/`volatile` or a trailing `&`/`*`."""
    head = written.split("(", 1)[0].strip()
    head = re.sub(r"^(?:const|volatile)\s+", "", head)
    return re.sub(r"\s*[&*]+\s*$", "", head)


def version(compiler: str, language: str, standard: str, header: str, library) -> str:
    """`compiler A.B.C, <library>, -std=<standard>`: the toolchain the data
    was read from. ``library`` names the library from the macros `compiler`
    defines for a `language` unit that includes `header`."""
    run = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True
    )
    found = re.search(r"version (\d+\.\d+\.\d+)", run.stdout)
    macros = subprocess.run(
        [compiler, "-x", language, f"-std={standard}", "-E", "-dM", "-"],
        input=f"#include <{header}>\n",
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    named = f"{compiler} {found[1]}" if found else compiler
    return f"{named}, {library(macros)}, -std={standard}"


def _declaration(node: dict) -> dict:
    """`node` without its id, its source locations and the expressions it
    holds."""
    for key in ("id", "loc", "range"):
        node.pop(key, None)
    if "inner" in node:
        node["inner"] = [
            i for i in node["inner"] if not i.get("kind", "").endswith("Expr")
        ]
    return node


KEYS = (
    "source",
    "types",
    "functions",
    "scope",
    "variants",
    "callbacks",
    "by_argument",
    "lang",
    "same",
    "extensions",
    "functional",
    "modules",
    "ref_structs",
    "qualifiers",
    "attributes",
    "context_managers",
    "pointers",
    "writers",
    "makers",
    "awaited",
    "evaluators",
    "implicit",
    "implicit_usings",
    "platforms",
)


def main(build) -> None:
    """Run `build` on the source the command line names and write what it
    returns, the value of each of `KEYS` it has, to the file named next."""
    origin, output = map(Path, sys.argv[1:])
    data = {key: held for key, held in build(origin).items() if held is not None}
    if unknown := set(data) - set(KEYS):
        raise KeyError(f"not keys of the data: {sorted(unknown)}")
    text = json.dumps(data, indent=1, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")
