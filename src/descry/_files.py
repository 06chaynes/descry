"""Writing files into a directory the indexed repository may have shipped."""

import os
import secrets
from pathlib import Path


def create_sibling(path: Path) -> tuple[int, Path]:
    """Create a new file beside `path` to write it atomically, then replace it.

    The name is random and the file is created with `O_EXCL`, so neither a
    leftover nor a symlink a repository put at a name it could guess is
    opened or followed. The mode is the one `open(path, "w")` gives.
    Returns the open descriptor and the file's path.
    """
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.open(tmp, flags, 0o666), tmp
