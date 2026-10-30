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
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import dump_ast, main, record, returned, version

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
    if text is None:
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
        text = _spot_file(place)
        if text:
            return text
    return None


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
    if kind in ("FunctionDecl", "VarDecl"):
        public = name and not name.startswith("_")
        if not public or not build.file.startswith(("/usr/include", headers_dir)):
            return
        if kind == "VarDecl":
            build.values.add(name)
        else:
            written = node.get("type", {}).get("qualType", "")
            record(build.functions, name, returned(written) or None)
        return
    for item in node.get("inner", ()):
        _walk(item, build, headers_dir)


def _glibc(macros):
    """`glibc A.B` as the macros of `features.h` give it."""
    minor = re.search(r"#define __GLIBC_MINOR__ (\d+)", macros)
    major = re.search(r"#define __GLIBC__ (\d+)", macros)
    return f"glibc {major[1]}.{minor[1]}" if major and minor else "glibc"


def _version(compiler):
    """`clang A.B.C, glibc A.B, -std=c23` — the toolchain the data was read
    from."""
    return version(compiler, "c", "c23", "features.h", _glibc)


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
    scope = {name: name for name in held.functions}
    for name in held.values:
        scope.setdefault(name, name)
    return {
        "source": _version("clang"),
        "types": {},
        "functions": held.functions,
        "scope": scope,
    }


if __name__ == "__main__":
    main(build)
