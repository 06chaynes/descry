"""Generate c.json from Clang's own AST of the standard C headers.

usage: gen_c.py INCLUDE_DIR OUTPUT

INCLUDE_DIR is the directory Clang's C standard headers stand at — the
system include root, `/usr/include`. One translation unit including the
public headers is compiled and its AST dumped as JSON; the function
declarations the headers hold are read out of it — every `FunctionDecl`
whose location is inside ``INCLUDE_DIR`` (or its system neighbours) and
whose name is not the implementation's (a `_` prefix), with the return
type the declaration writes. No list of names is kept.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import dump_ast, main

# The public headers the translation unit written for the dump includes.
HEADERS = (
    "stdio.h",
    "stdlib.h",
    "string.h",
    "strings.h",
    "ctype.h",
    "errno.h",
    "math.h",
    "time.h",
    "stdint.h",
    "stdbool.h",
    "stddef.h",
    "limits.h",
    "float.h",
    "assert.h",
    "signal.h",
    "setjmp.h",
    "stdarg.h",
    "locale.h",
    "wchar.h",
    "wctype.h",
    "inttypes.h",
    "fcntl.h",
    "unistd.h",
    "sys/types.h",
    "sys/stat.h",
    "sys/wait.h",
    "sys/mman.h",
    "dirent.h",
    "pthread.h",
    "dlfcn.h",
    "netdb.h",
    "sys/socket.h",
    "netinet/in.h",
    "arpa/inet.h",
    "poll.h",
    "sys/select.h",
    "sys/epoll.h",
    "termios.h",
    "grp.h",
    "pwd.h",
    "ftw.h",
    "glob.h",
    "fnmatch.h",
    "regex.h",
    "search.h",
    "spawn.h",
    "sched.h",
    "semaphore.h",
    "sys/ipc.h",
    "sys/shm.h",
    "sys/sem.h",
    "sys/msg.h",
    "sys/resource.h",
    "sys/time.h",
    "sys/times.h",
    "sys/uio.h",
    "sys/un.h",
    "syslog.h",
    "utime.h",
    "getopt.h",
    "execinfo.h",
    "malloc.h",
    "alloca.h",
    "endian.h",
    "byteswap.h",
    "error.h",
    "fenv.h",
    "complex.h",
    "stdalign.h",
    "stdatomic.h",
    "stdnoreturn.h",
    "threads.h",
    "uchar.h",
    "fmtmsg.h",
    "iconv.h",
    "langinfo.h",
    "libgen.h",
    "monetary.h",
    "nl_types.h",
    "wordexp.h",
    "tar.h",
)


def _spot_file(place):
    """The file a location object names — Clang's JSON writes `file`
    only where the file differs from the declaration before, so absence
    does not say a decl is not in a header."""
    text = place.get("file")
    if not isinstance(text, str):
        return None
    return str(Path(text).resolve()) if ".." in text else text


def _file_of(node):
    """The file the declaration ``node`` stands in, where its locations
    name it."""
    for place in (
        node.get("loc", {}),
        node.get("range", {}).get("begin", {}),
        node.get("loc", {}).get("spellingLoc", {}),
        node.get("loc", {}).get("expansionLoc", {}),
    ):
        if isinstance(place, dict):
            text = _spot_file(place)
            if text:
                return text
    return None


def _returned(written):
    """The type a Clang `type` string returns, before its `(...)`."""
    head = written.split("(", 1)[0].strip()
    head = re.sub(r"^(?:const|volatile)\s+", "", head)
    head = re.sub(r"\s*[&*]+\s*$", "", head)
    return head or None


def _walk(node, build, headers_dir):
    """Walk the AST: every `FunctionDecl` declared in a system header of a
    name that is not the implementation's (a `_` prefix) is a library
    function, and so is a `VarDecl` — `errno` is a `*__errno_location()`
    macro, `stdin` a variable."""
    text = _file_of(node)
    if text is not None:
        build.file = text
    kind = node.get("kind")
    name = node.get("name", "")
    if kind == "FunctionDecl":
        in_headers = build.file.startswith(("/usr/include", headers_dir))
        if name and not name.startswith("_") and in_headers:
            returns = _returned(node.get("type", {}).get("qualType", ""))
            if name not in build.functions:
                build.functions[name] = returns
            elif returns is not None and build.functions[name] not in (
                None,
                returns,
            ):
                build.functions[name] = None
        return
    if kind == "VarDecl":
        if (
            name
            and not name.startswith("_")
            and build.file.startswith(("/usr/include", headers_dir))
        ):
            build.values.add(name)
        return
    for item in node.get("inner", ()):
        _walk(item, build, headers_dir)


def _version(compiler, origin):
    """`clang A.B.C, glibc <root>, -std=c23` — the toolchain the data was
    read from."""
    run = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True
    )
    compiler_version = re.search(r"version (\d+\.\d+\.\d+)", run.stdout)
    run = subprocess.run(
        [compiler, "-std=c23", "-E", "-dM", "-x", "c", "-"],
        input="#include <features.h>\n",
        capture_output=True,
        text=True,
        check=False,
    )
    minor = re.search(r"#define __GLIBC_MINOR__ (\d+)", run.stdout)
    major = re.search(r"#define __GLIBC__ (\d+)", run.stdout)
    libc_version = f"glibc {major[1]}.{minor[1]}" if major and minor else "glibc"
    return (
        f"{compiler} {compiler_version[1]}, {libc_version}, -std=c23"
        if compiler_version
        else f"{compiler}, {libc_version}, -std=c23"
    )


class _Build:
    def __init__(self):
        self.functions = {}
        self.values = set()
        self.file = ""


def build(origin):
    held = _Build()
    headers = [
        h
        for h in HEADERS
        if (Path(origin) / h).exists() or Path("/usr/include", h).exists()
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tree = dump_ast(["clang", "-std=c23"], Path(tmp) / "dump.c", headers)
    _walk(tree, held, str(Path(origin).resolve()))
    functions = dict(sorted(held.functions.items()))
    scope = {name: name for name in functions}
    for name in held.values:
        scope.setdefault(name, name)
    return _version("clang", origin), {}, functions, scope, None, None


if __name__ == "__main__":
    main(build)
