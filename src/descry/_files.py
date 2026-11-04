"""Writing files into a directory the indexed repository may have shipped."""

import contextlib
import fnmatch
import glob
import os
import re
import secrets
import stat
import time
from pathlib import Path

try:
    import fcntl as _fcntl  # Unix only
except ImportError:
    _fcntl = None

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
    return os.fdopen(_open_regular_fd(path, os.O_RDONLY), mode, **kwargs)


def _open_regular_fd(path: Path, flags: int) -> int:
    """`os.open(path, flags)` if the file is regular: never through a symlink
    at its name, nor a FIFO or a device. OSError for anything else."""
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    # Without O_NOFOLLOW (Windows), the name must be a regular file, or
    # missing, before the open and the very file opened after it.
    named = None
    if not hasattr(os, "O_NOFOLLOW"):
        with contextlib.suppress(FileNotFoundError):
            named = os.lstat(path)
        if named and not stat.S_ISREG(named.st_mode):
            raise OSError(f"{path} is not a regular file")
    fd = os.open(path, flags, 0o666)
    opened = os.fstat(fd)
    if not stat.S_ISREG(opened.st_mode) or (
        named and not os.path.samestat(named, opened)
    ):
        os.close(fd)
        raise OSError(f"{path} is not a regular file")
    return fd


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


@contextlib.contextmanager
def file_lock(lock_path: Path, timeout: float = 600.0, create: bool = True):
    """Cross-process advisory lock; with `create` False a missing lock file
    raises FileNotFoundError instead of being created.

    Uses fcntl.flock on Unix and msvcrt.locking on Windows. Both are held on
    an open handle, so the OS releases them when the holder exits.

    The lock file is never unlinked. Removing it would break mutual
    exclusion: a waiter blocked on the old inode and a newcomer that
    recreates the path hold locks on two different inodes and both enter.
    An empty lock file is cheap; correctness is not.

    The cache directory may have come with the repository, so the lock file
    is opened without following a symlink or waiting on a FIFO, never
    truncated, and refused unless it is a regular file.
    """
    if _fcntl is not None:
        poll_seconds = 0.05

        def lock(f):
            _fcntl.flock(f, _fcntl.LOCK_EX | _fcntl.LOCK_NB)

        def unlock(f):
            _fcntl.flock(f, _fcntl.LOCK_UN)

    else:
        # Windows fallback: lock the first byte of the (always empty) lock file.
        import msvcrt

        poll_seconds = 0.1

        def lock(f):
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

        def unlock(f):
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

    fd = _open_regular_fd(
        lock_path, os.O_WRONLY | os.O_APPEND | (os.O_CREAT if create else 0)
    )
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        # Poll non-blocking so the timeout is honoured; a blocking lock would
        # wait forever regardless of the argument.
        deadline = time.monotonic() + timeout
        while True:
            try:
                lock(f)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for the lock at {lock_path}"
                    ) from None
                time.sleep(poll_seconds)
        try:
            yield
        finally:
            unlock(f)


def lock_held(lock_path: Path) -> bool:
    """Whether another holder has `lock_path`'s `file_lock`; False where
    the lock file is missing or cannot be opened, which it never creates."""
    try:
        with file_lock(lock_path, timeout=0, create=False):
            return False
    except TimeoutError:
        return True
    except OSError:
        return False
