"""Generate php.json from JetBrains' phpstorm-stubs.

usage: gen_php.py STUBS_TAR OUTPUT

STUBS_TAR is the archive of the phpstorm-stubs repository at one commit:
a directory for each extension, holding PHP files that declare its classes
and functions without bodies, with the types the manual gives them in
PHPDoc comments and in attributes.
"""

import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import main, new_type, record, split

# The directories of the extensions that the PHP source tree itself holds
# (`Zend/`, `main/` and `ext/` of php-src).
BUNDLED = (
    "Core", "FFI", "PDO", "Phar", "Reflection", "SPL", "SimpleXML", "Zend OPcache",
    "bcmath", "bz2", "calendar", "com_dotnet", "ctype", "curl", "date", "dba", "dom",
    "enchant", "exif", "fileinfo", "filter", "ftp", "gd", "gettext", "gmp", "hash",
    "iconv", "intl", "json", "ldap", "libxml", "mbstring", "mysqli", "odbc",
    "openssl", "pcntl", "pcre", "pgsql", "posix", "random", "readline", "session",
    "shmop", "snmp", "soap", "sockets", "sodium", "sqlite3", "standard", "sysvmsg",
    "sysvsem", "sysvshm", "tidy", "tokenizer", "uri", "xml", "xmlreader",
    "xmlwriter", "xsl", "zip", "zlib",
)  # fmt: skip

TOKEN = re.compile(
    r"""(?P<doc>/\*\*.*?\*/)
    |(?P<skip>/\*.*?\*/|//[^\n]*|\#(?!\[)[^\n]*
        |<<<['"]?(?P<end>\w+)['"]?\n.*?\n\s*(?P=end)\b)
    |(?P<text>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
    |(?P<name>\$?[A-Za-z_\\][\w\\]*)|(?P<attribute>\#\[)|\.\.\.|\S""",
    re.DOTALL | re.VERBOSE,
)
MODIFIERS = {
    "abstract", "final", "readonly", "public", "protected", "private", "static", "var",
}  # fmt: skip
# What a type is written with besides the names of classes.
KEYWORDS = {
    "array", "bool", "callable", "false", "float", "int", "iterable", "mixed",
    "never", "null", "object", "resource", "static", "string", "true", "void",
}  # fmt: skip
BRACKETS = "<([{", ">)]}"
# `PhpStormStubsElementAvailable(from: '5.3', to: '7.4')`: no longer there.
ENDED = re.compile(r"PhpStormStubsElementAvailable\([^)]*\bto:")
# `LanguageLevelTypeAware(['8.0' => 'string', '8.4' => 'int'], default: '')`
TYPE_AWARE = re.compile(r"LanguageLevelTypeAware\(\s*\[(.*?)\]", re.DOTALL)
VERSION_TYPE = re.compile(r"'([\d.]+)'\s*=>\s*'([^']*)'")
TEMPLATE = re.compile(r"@template(?:-covariant|-contravariant)?\s+(\w+)")
GENERIC_BASE = re.compile(
    r"@(?:template-)?(?:extends|implements)\s+(\\?[\w\\]+<.*)", re.MULTILINE
)
RETURN = re.compile(r"@return\s+(\S.*)", re.MULTILINE)
# The types PHP has that are no class, with what `foreach` gives of them.
UNNAMED = {"array": ["iterable<TKey, TValue>"], "iterable": []}


def tokens(text):
    """Yield ``(kind, text)`` for each token of PHP source: `doc` for a
    PHPDoc comment, `attribute` for a whole `#[...]`, `text` for a string,
    `name` for a word or a variable, `` for any other; other comments and
    heredocs are left out."""
    at = text.find("<?php") + 5
    while found := TOKEN.search(text, at):
        at = found.end()
        kind = found.lastgroup
        if kind == "attribute":
            depth, start = 1, found.start()
            while depth:
                inner = TOKEN.search(text, at)
                at = inner.end()
                depth += (inner[0] in ("[", "#[")) - (inner[0] == "]")
            yield kind, text[start:at]
        elif kind != "skip":
            yield kind or "", found[0]


def type_text(text):
    """The type that `text`, the rest of a PHPDoc tag's line, starts with."""
    depth = 0
    for at, char in enumerate(text):
        depth += (char in BRACKETS[0]) - (char in BRACKETS[1])
        if char.isspace() and depth == 0 and not text[:at].rstrip().endswith(","):
            return text[:at]
    return text


class Reader:
    """Reads the declarations of the stub files into `types`, `functions`."""

    def __init__(self):
        self.types, self.functions = {}, {}
        # (table, name, type as written, what resolves its names)
        self.results = []
        self.bases = []  # (type, base as written, what resolves its names)

    def read(self, text):
        self.stream = list(tokens(text))
        self.at = 0
        self.namespace, self.imports = "", {}
        doc, attributes = "", []
        while self.at < len(self.stream):
            kind, token = self.take()
            if kind == "doc":
                doc = token
            elif kind == "attribute":
                attributes.append(token)
            elif token == "namespace":
                kind, name = self.take()
                self.namespace = name if kind == "name" else ""
            elif token == "use":
                self.use()
            elif token in ("class", "interface", "trait", "enum"):
                self.type(doc, attributes)
            elif token == "function":
                self.function(self.functions, doc, attributes, set(), None)
            if kind not in ("doc", "attribute") and token not in MODIFIERS:
                doc, attributes = "", []

    def take(self):
        self.at += 1
        return self.stream[self.at - 1]

    def until(self, closers):
        """The tokens up to the first of `closers` outside brackets, which
        is taken too."""
        taken, depth = [], 0
        while self.at < len(self.stream):
            token = self.take()[1]
            if depth == 0 and token in closers:
                break
            depth += (token in "([{") - (token in ")]}")
            taken.append(token)
        return taken

    def use(self):
        names = self.until(";")
        if names and names[0] not in ("function", "const"):
            path = names[0].lstrip("\\")
            self.imports[names[-1].rpartition("\\")[2]] = path

    def scope(self, templates, this):
        return self.namespace, dict(self.imports), templates, this

    def type(self, doc, attributes):
        name = self.take()[1]
        header = self.until("{")
        key = f"{self.namespace}\\{name}".lstrip("\\")
        # A class the newest PHP no longer has is read and left out.
        ended = any(map(ENDED.search, attributes))
        this = new_type(TEMPLATE.findall(doc))
        if not ended:
            this = self.types.setdefault(key, this)
        scope = self.scope(this["params"], key)
        written, listed = [], False
        for token in header:
            if token in ("extends", "implements"):
                listed = True
            elif listed and token != ",":
                written.append(token)
        generic = {base.partition("<")[0]: base for base in GENERIC_BASE.findall(doc)}
        for base in dict.fromkeys([*written, *generic]):
            self.bases.append((this, type_text(generic.get(base, base)), scope))
        doc, attributes, flags = "", [], set()
        while self.at < len(self.stream):
            kind, token = self.take()
            if token == "}":
                break
            if kind == "doc":
                doc = token
            elif kind == "attribute":
                attributes.append(token)
            elif token in MODIFIERS:
                flags.add(token)
            elif token == "function":
                self.function(this, doc, attributes, flags, scope)
            elif token in ("const", "case") or token.startswith("$"):
                self.until(";")
            if kind not in ("doc", "attribute") and token not in MODIFIERS:
                doc, attributes, flags = "", [], set()

    def function(self, owner, doc, attributes, flags, scope):
        kind, name = self.take()
        if name == "&":
            kind, name = self.take()
        if kind != "name" or self.take()[1] != "(":
            return  # a closure in a constant expression
        # A method named by a reserved word is declared under this prefix.
        name = name.removeprefix("PS_UNRESERVE_PREFIX_")
        self.until(")")
        tail = self.until(("{", ";"))
        if self.stream[self.at - 1][1] == "{":
            self.until("}")
        if (
            any(map(ENDED.search, attributes))
            or "@removed" in doc
            or "private" in flags
        ):
            return
        aware = [TYPE_AWARE.search(a) for a in attributes]
        aware = [VERSION_TYPE.findall(found[1]) for found in aware if found]
        native = "".join(tail[1:]) if tail[:1] == [":"] else ""
        if aware and aware[0]:
            native = max(aware[0], key=lambda v: tuple(map(int, v[0].split("."))))[1]
        told = RETURN.search(doc)
        told = type_text(told[1]) if told else ""
        templates = TEMPLATE.findall(doc)
        if scope is None:
            table = owner
            name = f"{self.namespace}\\{name}".lstrip("\\")
            scope = self.scope(templates, None)
        else:
            table = owner["statics" if "static" in flags else "methods"]
            scope = (*scope[:2], [*scope[2], *templates], scope[3])
        # The PHPDoc type where it says more: a type parameter or type
        # arguments; else the declared one.
        generic = "<" in told or told.endswith("[]") or told in scope[2]
        self.results.append(
            (table, name, told if generic or not native else native, scope)
        )

    def named(self, name, scope):
        """`name` as the file keys the class it stands for in `scope`."""
        namespace, imports, templates, this = scope
        if name in templates or name.lower() in KEYWORDS:
            return name if name in templates else name.lower()
        if name in ("self", "$this"):
            return this if name == "self" else "static"
        first, _, rest = name.partition("\\")
        if first in imports:
            return f"{imports[first]}\\{rest}".rstrip("\\")
        local = f"{namespace}\\{name}".lstrip("\\")
        return local if local in self.types else name.lstrip("\\")

    def spelled(self, written, scope):
        """The type `written` with each class named by its key; None for a
        function that returns nothing."""
        alternatives = []
        if written.startswith("?"):
            written = f"null|{written[1:]}"
        for alternative in split(written, "|", *BRACKETS):
            if alternative.endswith("[]"):
                element = self.spelled(alternative[:-2], scope)
                alternatives.append(f"array<int, {element}>")
            elif found := re.fullmatch(r"(\$?[\w\\]+)(?:<(.*)>)?", alternative):
                arguments = split(found[2] or "", ",", *BRACKETS)
                arguments = ", ".join(self.spelled(a, scope) or a for a in arguments)
                name = self.named(found[1], scope)
                alternatives.append(f"{name}<{arguments}>" if arguments else name)
            else:
                alternatives.append(alternative)
        returns = "|".join(dict.fromkeys(alternatives))
        return None if returns in ("", "void", "never") else returns

    def finish(self):
        for name, bases in UNNAMED.items():
            self.types[name] = new_type(("TKey", "TValue"))
            self.types[name]["bases"] = list(bases)
        for table, name, written, scope in self.results:
            record(table, name, self.spelled(written, scope))
        for this, written, scope in self.bases:
            base = self.spelled(written, scope)
            if base.partition("<")[0] in self.types and base not in this["bases"]:
                this["bases"].append(base)


def build(archive):
    reader = Reader()
    with tarfile.open(archive) as stubs:
        members = [m for m in stubs.getmembers() if m.name.endswith(".php")]
        root = members[0].name.partition("/")[0]
        for member in sorted(members, key=lambda m: m.name):
            if member.name.split("/")[1] in BUNDLED:
                reader.read(stubs.extractfile(member).read().decode("utf-8"))
    reader.finish()
    commit = root.removeprefix("phpstorm-stubs-")
    source = (
        f"JetBrains phpstorm-stubs {commit}: the extensions bundled with PHP,"
        " as the newest PHP has them"
    )
    return source, reader.types, reader.functions, {}


if __name__ == "__main__":
    main(build)
