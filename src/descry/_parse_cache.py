"""Each file's parse, kept as plain data and reused while the file and
descry's own code stay the same.

An entry is `<cache>/parse/<place>-<key>.json`: `place` names the file by its
path, `key` the path, its content and descry's code. It is JSON, never
pickle, since a repository can ship its cache: a value JSON cannot write
plainly is written tagged, `{"t": [...]}` a tuple, `{"s": [...]}` a set,
`{"f": [...]}` a frozenset and `{"d": [[key, value], ...]}` a dict whose keys
are not all strings or that would read as a tag. Its first line is an HMAC
of its name and the rest under a key of the user's, outside every
repository: an entry descry did not write is read as none, and the file
parsed again.
"""

import hashlib
import hmac
import json
import os
import sys
from functools import cache
from pathlib import Path

from descry._files import create_sibling, open_regular

_TAGS = {"t": tuple, "s": set, "f": frozenset}


def _encode(value):
    """``value`` as JSON can write it; TypeError for a value it cannot."""
    if value is None or isinstance(value, (str, int, float)):
        return value
    if isinstance(value, list):
        return [_encode(v) for v in value]
    for tag, kind in _TAGS.items():
        if type(value) is kind:
            return {tag: [_encode(v) for v in value]}
    if isinstance(value, dict):
        plain = all(isinstance(k, str) for k in value)
        if plain and not (len(value) == 1 and next(iter(value)) in {*_TAGS, "d"}):
            return {k: _encode(v) for k, v in value.items()}
        return {"d": [[_encode(k), _encode(v)] for k, v in value.items()]}
    raise TypeError(type(value).__name__)


def _decode(value):
    """The value `_encode` wrote as ``value``."""
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        ((tag, held),) = value.items()
        if tag in _TAGS and isinstance(held, list):
            return _TAGS[tag](_decode(v) for v in held)
        if tag == "d" and isinstance(held, list):
            return {_decode(k): _decode(v) for k, v in held}
    return {k: _decode(v) for k, v in value.items()}


@cache
def _code_fingerprint():
    """A digest of descry's code and data, and of the Python whose `ast`
    reads Python source, which every entry is of."""
    digest = hashlib.sha256(sys.version.encode())
    package = Path(__file__).parent
    for path in sorted([*package.glob("*.py"), *package.glob("stdlib/*.json")]):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _key_path():
    """Where the user's key is kept: their state directory's `descry`. A
    relative directory in the environment is ignored, as XDG has it."""
    base = os.environ.get("LOCALAPPDATA" if os.name == "nt" else "XDG_STATE_HOME")
    if not base or not Path(base).is_absolute():
        base = Path.home() / ".local" / "state"
    return Path(base) / "descry" / "parse.key"


def _key(path):
    """The key at ``path``, made there if there is none; None where it
    cannot be read or made."""
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))
    except FileExistsError:
        pass
    except OSError:
        return None
    try:
        held = path.read_bytes()
    except OSError:
        return None
    return held if len(held) == 32 else None


def signing_key(project=None, key_path=None):
    """The key at ``key_path``, by default the user's, that descry signs
    what it keeps with; None where it would be inside ``project``, which
    could have shipped it, or cannot be read or made."""
    key_path = Path(key_path or _key_path()).resolve()
    if project is not None and key_path.is_relative_to(Path(project).resolve()):
        return None
    return _key(key_path)


def _digest(*parts):
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:32]


class ParseCache:
    """The entries under ``directory`` for a parse of the ``settings`` that
    change what it gives, signed with the key at ``key_path``. None are
    read or kept where the key would be inside ``project``, which could
    have shipped it."""

    def __init__(self, directory, *settings, key_path=None, project=None):
        self.directory = Path(directory)
        self.settings = "\0".join(map(str, settings))
        self.key = signing_key(project, key_path)
        self.walked = set()  # the places of the files walked this run
        self.current = set()  # the names of their entries now

    def _mac(self, name, text):
        return hmac.new(self.key, f"{name}\0{text}".encode(), "sha256").hexdigest()

    def _entry(self, rel_path, content, module):
        place = _digest(rel_path)
        key = _digest(
            _code_fingerprint(), self.settings, rel_path, content, str(module)
        )
        name = f"{place}-{key}.json"
        self.walked.add(place)
        self.current.add(name)
        return self.directory / name

    def entry(self, rel_path, content, module):
        """The name of the entry of the file ``rel_path`` holding
        ``content``, an ES module by its package where ``module``, which
        names that parse of it."""
        return self._entry(rel_path, content, module).name

    def load(self, rel_path, content, module):
        """What parsing ``content``, the file ``rel_path``, gave, as `store`
        was given it; None where no entry descry wrote holds it."""
        return self._read(self._entry(rel_path, content, module))

    def store(self, rel_path, content, module, state):
        """Keep ``state``, what parsing the file gave, unless JSON cannot
        write it."""
        self._write(self._entry(rel_path, content, module), state)

    def load_resolution(self):
        """What `store_resolution` kept, None where descry kept nothing."""
        return self._read(self.directory / "resolution.json", plain=True)

    def store_resolution(self, held):
        """Keep ``held``, what resolving the files' calls gave, as plain JSON:
        lists for tuples, and no sets."""
        self._write(self.directory / "resolution.json", held, plain=True)

    def _read(self, path, plain=False):
        if self.key is None:
            return None
        try:
            with open_regular(path, "r", encoding="utf-8") as f:
                mac, _, text = f.read().partition("\n")
            if not hmac.compare_digest(mac, self._mac(path.name, text)):
                return None
            held = json.loads(text)
            return held if plain else _decode(held)
        except (OSError, ValueError, TypeError):
            return None

    def _write(self, path, value, plain=False):
        if self.key is None:
            return
        try:
            text = json.dumps(value if plain else _encode(value))
        except TypeError:
            return
        text = f"{self._mac(path.name, text)}\n{text}"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, tmp = create_sibling(path)
        except OSError:
            return
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)

    def prune(self, whole):
        """Remove the entries of the files walked this run that hold another
        content or code than theirs now, and with ``whole`` those of files
        not walked."""
        for path in self.directory.glob("*-*.json") if self.directory.is_dir() else ():
            place = path.name.partition("-")[0]
            if path.name not in self.current and (whole or place in self.walked):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    # A shipped entry that is a directory cannot be removed;
                    # it is ignored like any other unparsable one.
                    continue
