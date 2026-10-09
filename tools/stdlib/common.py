"""What the gen_*.py generators share."""

import json
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


def record(table: dict, name: str, returns: str | None) -> None:
    """Add one signature of `name`; signatures that disagree leave None."""
    if table.setdefault(name, returns) != returns:
        table[name] = None


def dump_ast(command: list[str], source: Path, headers) -> dict:
    """The AST the compiler `command` prints for a unit, written to
    `source`, that includes `headers`."""
    source.write_text(
        "".join(f"#include <{header}>\n" for header in headers) + "int main(){}\n"
    )
    run = subprocess.run(
        [
            *command,
            "-fsyntax-only",
            "-Xclang",
            "-ast-dump=json",
            str(source),
            "-o",
            "/dev/null",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode:
        raise RuntimeError(f"{command[0]} -ast-dump failed: {run.stderr[:2000]}")
    return json.loads(run.stdout)


KEYS = ("source", "types", "functions", "scope", "variants", "callbacks")


def main(build) -> None:
    """Run `build` on the source the command line names and write what it
    returns, the values of the first of `KEYS` in their order and None for
    one it has not, to the file named next."""
    origin, output = map(Path, sys.argv[1:])
    data = {key: held for key, held in zip(KEYS, build(origin)) if held is not None}
    text = json.dumps(data, indent=1, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")
