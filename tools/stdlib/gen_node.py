"""Generate node.json from the module declarations of `@types/node`.

usage: gen_node.py NODE.tgz OUTPUT

NODE.tgz is the `@types/node` package as the npm registry serves it; the
names its `.d.ts` files declare with `declare module` are read in place,
nothing of it is unpacked or run.
"""

import json
import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import main

MODULE = re.compile(r"""^declare module ["']([^"']+)["']""", re.MULTILINE)


def build(archive):
    with tarfile.open(archive) as tar:
        files = {
            member.name: tar.extractfile(member).read().decode("utf-8")
            for member in tar
            if member.isfile() and member.name.endswith((".d.ts", "/package.json"))
        }
    modules = sorted(
        {
            name
            for path, text in files.items()
            if path.endswith(".d.ts")
            for name in MODULE.findall(text)
        }
    )
    # The package's own, at the top of the one directory it is packed in
    manifest = next(
        path
        for path in files
        if path.count("/") == 1 and path.endswith("/package.json")
    )
    version = json.loads(files[manifest])["version"]
    source = f"@types/node {version} (npm), the modules its declarations name"
    return {"source": source, "modules": modules}


if __name__ == "__main__":
    main(build)
