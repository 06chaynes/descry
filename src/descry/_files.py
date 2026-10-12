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
