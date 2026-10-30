"""Writing files into a directory the indexed repository may have shipped."""

import fnmatch
import glob
import os
import re
import secrets
import stat
import time
from pathlib import Path

# A temporary nothing has written to for this long belongs to a writer that
# was killed before it could replace its target or remove the temporary.
_ORPHANED_AFTER = 3600


def create_sibling(path: Path) -> tuple[int, Path]:
    """Create a new file beside `path` to write it atomically, then replace it.

    The name is random and the file is created with `O_EXCL`, so neither a
    leftover nor a symlink a repository put at a name it could guess is
    opened or followed. The mode is the one `open(path, "w")` gives.
    Temporaries for `path` that are orphaned are removed first.
    Returns the open descriptor and the file's path.
    """
    remove_orphans(path.parent, glob.escape(path.name))
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.open(tmp, flags, 0o666), tmp


def open_regular(path: Path, mode: str = "rb", **kwargs):
    """`path` opened for reading if it is a regular file: never through a
    symlink at its name, nor a FIFO or a device, which a repository can ship
    where descry keeps files and a read of which would never end. OSError
    for anything else."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    # Without O_NOFOLLOW (Windows), the name must be a regular file before
    # the open and the very file opened after it.
    named = None if hasattr(os, "O_NOFOLLOW") else os.lstat(path)
    f = os.fdopen(os.open(path, flags), mode, **kwargs)
    opened = os.fstat(f.fileno())
    if not stat.S_ISREG(opened.st_mode) or (
        named and not (stat.S_ISREG(named.st_mode) and os.path.samestat(named, opened))
    ):
        f.close()
        raise OSError(f"{path} is not a regular file")
    return f


def remove_orphans(directory: Path, names: str) -> None:
    """Remove the temporaries `create_sibling` made in `directory` for targets
    whose names match the glob `names`, that nothing has written to for
    `_ORPHANED_AFTER` seconds."""
    cutoff = time.time() - _ORPHANED_AFTER
    for tmp in directory.glob(f".{names}.*.tmp"):
        target, _, token = tmp.name[1:-4].rpartition(".")
        try:
            st = tmp.lstat()
            if (
                re.fullmatch(r"[0-9a-f]{16}", token)
                and fnmatch.fnmatchcase(target, names)
                and not stat.S_ISDIR(st.st_mode)
                and st.st_mtime < cutoff
            ):
                tmp.unlink()
        except OSError:
            continue
