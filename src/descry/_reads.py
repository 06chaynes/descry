"""Which keys of which tables resolving one file's calls reads, and whether
they hold what they held when another index read them."""

import hashlib
import json

from descry._parse_cache import _encode


class Reads:
    """The keys, and the tables read whole, of the file being resolved; None
    while no file is."""

    def __init__(self):
        self.keys = self.whole = None
        self.tables = {}  # name -> the recorded table
        # Tables no read of which is noted, each file taken to read whole
        self.always = set()

    def start(self):
        """Begin a file: returns its (keys, tables read whole)."""
        self.keys, self.whole = set(), set(self.always)
        return self.keys, self.whole

    def stop(self):
        self.keys = self.whole = None


class _Table:
    """Noting a read of a recorded table: of the key ``key``, or of all of
    it."""

    __slots__ = ()

    def _key(self, key):
        if self._reads.keys is not None:
            self._reads.keys.add((self._name, key))

    def _all(self):
        if self._reads.whole is not None:
            self._reads.whole.add(self._name)


class _Dict(_Table, dict):
    __slots__ = ("_name", "_reads")

    def __getitem__(self, key):
        self._key(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._key(key)
        return dict.get(self, key, default)

    def __contains__(self, key):
        self._key(key)
        return dict.__contains__(self, key)

    def __iter__(self):
        self._all()
        return dict.__iter__(self)

    def __len__(self):
        self._all()
        return dict.__len__(self)

    def keys(self):
        self._all()
        return dict.keys(self)

    def values(self):
        self._all()
        return dict.values(self)

    def items(self):
        self._all()
        return dict.items(self)


class _Nodes(_Dict):
    """A table of nodes: looking one up notes the key (table, (node ID,)),
    whether it is there; what is read of the node is noted by the node."""

    __slots__ = ()

    def _key(self, key):
        if self._reads.keys is not None:
            self._reads.keys.add((self._name, (key,)))


class _Fields(_Dict):
    """A node of a table of nodes, or a mapping it holds: each read of it
    notes the key (table, (node ID, *path to what is read)), a read of all
    of it the path closed by "*", whether it is empty by "?". Nothing
    writes to it."""

    # `_noted`: each field's key, made once; a mapping's is whether it is
    # empty, as the mapping notes what is read of it.
    __slots__ = ("_noted", "_path")

    def _key(self, step):
        if self._reads.keys is not None:
            self._reads.keys.add(
                self._noted.get(step) or (self._name, (*self._path, step))
            )

    def __bool__(self):
        self._key("?")
        return dict.__len__(self) > 0

    def _refuse(self, *args, **kwargs):
        raise TypeError("a recorded node is read only")

    __setitem__ = __delitem__ = __ior__ = _refuse
    pop = popitem = setdefault = update = clear = _refuse


def _whole(name):
    """dict's ``name``, which reads all of a mapping, noting so."""

    def read(self, *args):
        self._key("*")
        return getattr(dict, name)(self, *args)

    return read


for _read_all in (
    "__iter__",
    "__len__",
    "__eq__",
    "__or__",
    "keys",
    "values",
    "items",
    "copy",
):
    setattr(_Fields, _read_all, _whole(_read_all))


def _fields(value, name, reads, path):
    """``value``, a node or what it holds at ``path``, with each mapping in
    it one that notes what is read of it."""
    if type(value) is not dict:
        return value
    held = _Fields(
        {key: _fields(item, name, reads, (*path, key)) for key, item in value.items()}
    )
    held._name, held._reads, held._path = name, reads, path
    held._noted = {
        key: (name, (*path, key, "?") if type(item) is _Fields else (*path, key))
        for key, item in dict.items(held)
    }
    return held


class Probe:
    """A question asked of something no table holds, the project's
    directory: called with a key, it answers ``ask(key)``, noting the key as
    read where a recorded owner holds it, so a kept resolution is checked
    against the answer asked again."""

    def __init__(self, ask):
        self.ask = ask
        self._name = self._reads = None

    def __call__(self, key):
        if self._reads is not None and self._reads.keys is not None:
            self._reads.keys.add((self._name, key))
        return self.ask(key)


class _Set(_Table, set):
    __slots__ = ("_name", "_reads")

    def __contains__(self, key):
        self._key(key)
        return set.__contains__(self, key)

    def __iter__(self):
        self._all()
        return set.__iter__(self)

    def __len__(self):
        self._all()
        return set.__len__(self)


def record(owner, prefix, reads, skip, nodes=()):
    """Put tables that note what is read of them in place of each dict and
    set attribute of ``owner`` but those named in ``skip``, by the name
    ``prefix`` and the attribute's; those named in ``nodes`` are tables of
    nodes, each of whose fields is read apart."""
    for name, held in list(vars(owner).items()):
        if isinstance(held, Probe):
            held._name, held._reads = prefix + name, reads
            reads.tables[prefix + name] = held
        elif name in nodes:
            table = _Nodes(
                {
                    key: _fields(node, prefix + name, reads, (key,))
                    for key, node in held.items()
                }
            )
            table._name, table._reads = prefix + name, reads
            reads.tables[prefix + name] = table
            setattr(owner, name, table)
        elif name not in skip:
            setattr(owner, name, recorded(held, prefix + name, reads))


def recorded(held, name, reads):
    """``held`` as a table that notes what is read of it, if it is a dict or
    a set; else ``held``, a list or a tuple of which every file is taken to
    read whole."""
    kind = {dict: _Dict, set: _Set}.get(type(held))
    if kind is None:
        if isinstance(held, (list, tuple)):
            reads.tables[name] = held
            reads.always.add(name)
        return held
    table = kind(held)
    table._name, table._reads = name, reads
    reads.tables[name] = table
    return table


def _canonical(value):
    """``value`` with each set's members in one order, which a set's own
    order is not from one run to the next."""
    if isinstance(value, (set, frozenset)):
        return {type(value).__name__: sorted((_canonical(v) for v in value), key=repr)}
    if isinstance(value, (list, tuple)):
        return type(value)(_canonical(v) for v in value)
    if isinstance(value, dict):
        return {"dict": [[_canonical(k), _canonical(v)] for k, v in dict.items(value)]}
    return value


def _digest(value):
    try:
        text = json.dumps(_encode(_canonical(value)))
    except TypeError:
        return None
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def print_of(reads, name, key):
    """What the key ``key`` of the table ``name`` holds now, as a digest:
    of its value, or "absent"; None where it cannot be told."""
    table = reads.tables.get(name)
    if table is None:
        return None
    if isinstance(table, Probe):
        return _digest(table.ask(key))
    if isinstance(table, _Nodes):
        # A node is there or not; what is read of it is what its path leads
        # to, all of it for "*", whether it is empty for "?".
        *path, key = key
        for step in path:
            if not isinstance(table, dict) or not dict.__contains__(table, step):
                return "absent"
            table = dict.__getitem__(table, step)
        if key == "*":
            return _digest(table)
        if key == "?":
            if not isinstance(table, dict):
                return _digest(table)
            return "held" if dict.__len__(table) else "empty"
        if not path:
            return "in" if dict.__contains__(table, key) else "absent"
    if isinstance(table, set):
        return "in" if set.__contains__(table, key) else "absent"
    if not dict.__contains__(table, key):
        return "absent"
    return _digest(dict.__getitem__(table, key))


def whole_print(reads, name):
    """What the table ``name`` holds now, as a digest; None where it cannot
    be told."""
    table = reads.tables.get(name)
    if table is None:
        return None
    if isinstance(table, set):
        return _digest(set(set.__iter__(table)))
    return _digest(dict(dict.items(table)) if isinstance(table, dict) else table)
