"""PHP regex-based baseline parser.

Covers PHP namespaces, classes (with ``extends`` / ``implements``),
interfaces, traits, enums (PHP 8.1+), anonymous classes, methods,
constructors, properties, use-statements, and call sites. A type is the
context of the definitions up to its closing brace, a method or function
the source of the calls in its body.
"""

from __future__ import annotations

import re

from descry.generate import (
    BaseParser,
    _bracketed_parts,
    _is_call_chain,
    _use_leaves,
    blank_strings,
    is_generated_source,
    source_lines,
)

# namespace App\\Foo;  (one-line) or  namespace App\\Foo { ... }  (block),
# and  namespace { ... }  for the global one
_RE_NAMESPACE = re.compile(r"^\s*namespace(?:\s+([A-Za-z_\\][A-Za-z0-9_\\]*))?\s*[;{]")

# use App\\Foo;  /  use App\\Foo as Bar;  /  use function foo;  /  use const X;
# and, in a class, use SomeTrait;  -- the names may run on over lines
_RE_USE = re.compile(r"^\s*use\s+(?:(function|const)\s+)?([^;(]+)")

# class Foo extends Bar implements Baz, Qux  /  interface Foo  /  trait Foo
# enum Status: string (PHP 8.1+)
_RE_TYPE_DECL = re.compile(
    r"^\s*(?:(?:abstract|final|readonly)\s+)*"
    r"(class|interface|trait|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
# new class(...) extends Foo {
_RE_ANONYMOUS = re.compile(r"\bnew\s+(?:readonly\s+)?class\b")

# public function foo( / private static function bar( / abstract protected function baz(
# function &qux(
_METHOD_MODIFIERS = r"(?:public|protected|private|static|final|abstract|readonly)"
_RE_METHOD = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS})\s+)*"
    r"function\s+&?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
# function top_level_fn( at file scope
_RE_FUNCTION = re.compile(r"^\s*function\s+&?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# A type as a declaration writes it: ?Foo, Foo|null, \\App\\Foo, (A&B)|null
_TYPE = r"[?\\A-Za-z_(][\\\w|&()]*"

# private int $bar;  /  public readonly string $name;  /  var $baz = [
_RE_PROPERTY = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS}|var)\s+)+"
    rf"(?:(?P<type>{_TYPE})\s+)?"
    r"\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:[;=,]|$)"
)
# A constructor parameter that declares a property: private Foo $foo
_RE_PROMOTED = re.compile(
    r"(?:^|[(,])\s*(?:(?:public|protected|private|readonly)\s+)+"
    rf"(?:(?P<type>{_TYPE})\s+)?&?\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
# public const FOO = 1;  /  const BAR = [  /  final public const string BAZ = 'x';
_RE_CONSTANT = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS})\s+)*"
    rf"const\s+(?:{_TYPE}\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*="
)
# case Active = 'active';  in an enum
_RE_ENUM_CASE = re.compile(r"^\s*case\s+([A-Za-z_][A-Za-z0-9_]*)\s*[=;]")

# Call site: name(, $obj->method(, $this->prop->method(, Class::static(,
# \\Fully\\Qualified(, and the ->method( that goes on a chain
_RE_CALL = re.compile(
    r"(?<![\w$\\])"
    r"(\$?\\?[A-Za-z_][\w\\]*(?:(?:\?->|->|::)\$?[A-Za-z_]\w*)*)"
    r"\s*\("
)
_RE_ACCESS = re.compile(r"\?->|->|::")
# `new Type` without arguments
_RE_NEW = re.compile(
    r"(?<![\w$])new\s+((?!class\b)\\?[A-Za-z_][\w\\]*)(?![\w\\]|\s*[(:])"
)
# The variable or class and the properties read from it that a line ends
# with, for a `->method(` on the next line
_RE_TRAILING_PATH = re.compile(
    r"(?<![\w$\\>:])\$?\\?[A-Za-z_][\w\\]*(?:(?:\?->|->|::)\$?\w+)*\s*$"
)

_PHP_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "elseif",
        "else",
        "for",
        "foreach",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "return",
        "throw",
        "try",
        "catch",
        "finally",
        "match",
        "new",
        "instanceof",
        "isset",
        "unset",
        "empty",
        "list",
        "array",
        "echo",
        "print",
        "include",
        "include_once",
        "require",
        "require_once",
        "use",
        "yield",
        "fn",
        "function",
        "static",
        "self",
        "parent",
        "this",
        "true",
        "false",
        "null",
        "and",
        "or",
        "xor",
        "exit",
        "die",
    }
)

# A word before a variable that is no type: `return $x`, `private $x`
_NOT_A_TYPE = (
    r"(?!(?:return|echo|print|new|instanceof|throw|clone|yield|from|case|as|and|or"
    r"|xor|global|static|public|protected|private|var|readonly|final|abstract"
    r"|else|do|include|include_once|require|require_once|insteadof|use|const"
    r"|function|fn|break|continue)\b)"
)
_DECLARATIONS = (
    # `Type $name`: a parameter, a typed property or the variable of a
    # `catch`. The last type of a union is not read as the only one.
    re.compile(
        rf"(?<![|&\\\w$])(?<![|&] ){_NOT_A_TYPE}"
        r"(?P<type>\??\\?[A-Za-z_][\\\w]*)\s+(?:&\s*)?(?:\.\.\.\s*)?\$(?P<name>\w+)"
    ),
    # `$name = new Type`
    re.compile(
        r"(?<![\w$>:])\$(?P<name>\w+)\s*=\s*new\s+"
        r"(?P<type>(?!class\b)\\?[A-Za-z_][\\\w]*)"
    ),
)
# What gives the variables in its `names` a value no declaration types: the
# parameters of a function, the variables of a `foreach`, of a
# destructuring assignment and of a `catch` of several types
_REBINDINGS = (
    re.compile(r"(?<![\w$])(?:function|fn)\b[^()]*\((?P<names>[^()]*)\)"),
    re.compile(r"(?<![\w$])as\s+(?P<names>[^)]*)\)"),
    re.compile(r"^\s*(?:list\s*\(|\[)(?P<names>[^=]*)[\])]\s*=(?!=)"),
    re.compile(r"[|&]\s*[\\\w]+\s+(?P<names>\$\w+)"),
)
# A variable, a property or a constant read: `$other`, `$this->property`,
# `Type::CASE`
_PATH = r"(?:\$\w+|\\?[A-Za-z_][\w\\]*::\$?\w+)(?:\??->\w+)*"
_RE_PATH = re.compile(_PATH)
# `$name = $other;`
_RE_COPY = re.compile(rf"^\s*\$(?P<name>\w+)\s*=\s*(?P<path>{_PATH})\s*;\s*$")
# `foreach ($over as $key => $name)`, and `[$a, $b] = $over;`
_RE_FOREACH = re.compile(
    r"\bforeach\s*\((?P<over>.+)\s+as\s+(?:[^)]*=>\s*)?&?(?P<names>\$\w+)\s*\)"
)
_RE_DESTRUCTURING = re.compile(
    r"^\s*(?:list\s*\(|\[)(?P<names>[^=\[\]()]*)[\])]\s*=(?!=)\s*(?P<over>.+?);\s*$"
)
# `return $value;`
_RE_RETURN = re.compile(r"(?<![\w$>])return\b\s*(?P<value>[^;]*)(?P<end>;?)")
# `$name instanceof Type` or `$this->name instanceof Type`, not negated
_RE_INSTANCEOF = re.compile(
    r"(?<![!\w$>:])(?<!!\()(?<!! )\$(?P<name>\w+(?:->\w+)?)\s+instanceof\s+"
    r"(?P<type>\\?[A-Za-z_][\\\w]*)"
)

# PHPDoc tags: `@param Type $name`, `@var Type $name`, `@var Type`,
# `@return Type`, and on a class `@method Type name(`
# A type in a tag: `Foo`, `Foo[]|null`, `array<string, list<Foo>>`
_DOC_TYPE = r"[^\s<|]+(?:<(?:[^<>]|<[^<>]*>)*>)?(?:\[\])*"
_DOC_TYPE = rf"{_DOC_TYPE}(?:\|{_DOC_TYPE})*"
_RE_PARAM_TAG = re.compile(
    rf"@param\s+(?P<type>{_DOC_TYPE})\s+(?:\.\.\.)?\$(?P<name>\w+)"
)
_RE_VARIABLE_TAG = re.compile(rf"@var\s+(?P<type>{_DOC_TYPE})\s+\$(?P<name>\w+)")
_RE_VAR_TAG = re.compile(rf"@var\s+({_DOC_TYPE})")
_RE_RETURN_TAG = re.compile(rf"@return\s+({_DOC_TYPE})")
# A type written with what the code cannot say: its type arguments
_RE_TYPE_ARGUMENTS = re.compile(r"<|\[\]$")
_RE_METHOD_TAG = re.compile(
    r"@method\s+(?:static\s+)?(?:(?P<type>[^\s(]+)\s+)?(?P<name>\w+)\s*\("
)

# Type names that are no class: a value of one has no methods
_SCALAR_TYPES = frozenset(
    {"int", "float", "string", "bool", "array", "void", "never", "null", "false"}
)
# Type names that say nothing of the class of the value
_UNTYPED = frozenset({"mixed", "object", "callable", "parent"})

_STRINGS = ('"', "'")
_RE_HASH_COMMENT = re.compile(r"#(?!\[)")
_RE_ATTRIBUTE = re.compile(r"#\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]")
_RE_HEREDOC = re.compile(r"<<<[ \t]*(['\"]?)(\w+)\1")

# What the shared reader of call chains takes PHP's operators for: `.`
# reads a member and `:` is part of a path, and neither a concatenation nor
# a `?:` is a chain.
_RE_OPERATOR = re.compile(r"\?->|->|[.?\\]")
_CHAIN_OPERATORS = {"?->": "  .", "->": " .", ".": "+", "?": "|", "\\": ":"}


def _code_lines(lines: list[str]) -> list[str]:
    """``lines`` without their comments and attributes, and with the text
    of every string, heredoc and nowdoc turned to spaces or left out, so
    that none of it is read as code. Columns are kept."""
    codes = []
    inside = heredoc = None
    for line in lines:
        if heredoc:
            closed = re.match(rf"\s*{heredoc}\b", line)
            if not closed:
                codes.append("")
                continue
            line = " " * closed.end() + line[closed.end() :]
            heredoc = None
        code, ends_inside = blank_strings(line, inside, _STRINGS)
        comment = _RE_HASH_COMMENT.search(code)
        if comment:
            code, ends_inside = blank_strings(line[: comment.start()], inside, _STRINGS)
        inside = ends_inside
        opened = code.find("<<<")
        if opened >= 0 and (opener := _RE_HEREDOC.match(line, opened)):
            heredoc = opener[2]
            code = code[:opened]
        codes.append(_RE_ATTRIBUTE.sub(lambda found: " " * len(found.group()), code))
    return codes


def _signature(codes: list[str], index: int, column: int) -> tuple[str, str, int, bool]:
    """Read the function declared on ``codes[index]`` from ``column``, just
    inside its parameter list, through the lines the declaration runs on
    to. Returns the text of the parameter list, the return type as written,
    the index of the line the declaration ends on and whether it ends in a
    body."""
    text, depth, closed, read = codes[index][column:], 1, None, 0
    while True:
        for position in range(read, len(text)):
            char = text[position]
            if closed is None:
                depth += (char == "(") - (char == ")")
                if depth == 0:
                    closed = position
            elif char in "{;":
                returns = text[closed + 1 : position].strip().lstrip(":").strip()
                return text[:closed], returns, index, char == "{"
        if index + 1 == len(codes):
            return text, "", index, False
        index += 1
        read = len(text)
        text += " " + codes[index]


def _told(native: str, told: str | None) -> str | None:
    """The type a declaration is written with: the PHPDoc one, ``told``,
    where the code has none or where it adds type arguments."""
    return told if told and (not native or _RE_TYPE_ARGUMENTS.search(told)) else native


class PhpParser(BaseParser):
    """Regex-driven PHP parser."""

    binding_re = re.compile(r"(?<![\w$>:])\$(?P<name>\w+)\s*=(?![=>])")

    _namespace = ""  # of the file being parsed, with `::` for `\`
    # Local name -> full name of the classes and namespaces the file
    # imports, and of the functions it imports
    _imports: dict[str, str]
    _functions: dict[str, str]

    def _end_line(self):
        line, binding = self._line, self.binding_re.search(self._code)
        super()._end_line()
        if binding:
            self._unbind(binding["name"], line)

    def _unbind(self, name, lineno):
        """Leave the variable ``name`` without a known type from ``lineno``
        on, unless a declaration on that line gives it one."""
        entries = self.declared.setdefault(name, [])
        if not any(entry[0] == lineno for entry in entries):
            entries.append((lineno, "?", False))

    def _class_name(self, written):
        """Full name of the class the file being parsed writes as
        ``written``, by its namespace and its imports, with `::` for `\\`:
        `::App::Models::User`."""
        first, _, rest = written.replace("\\", "::").partition("::")
        if first == "namespace":
            first = self._namespace
        elif first:
            first = self._imports.get(first) or f"{self._namespace}::{first}".lstrip(
                ":"
            )
        return "::" + "::".join(filter(None, (first, rest)))

    def _type(self, written, owner):
        """What a declaration that writes its type as ``written`` says the
        value is: the full name of its class, `this` for an instance of the
        class the member is reached through, the name of a scalar type or
        `array` or `iterable`, each with its type arguments (`array<int,
        ::App::User>` for `User[]` and for `list<User>`), or None where it
        names no one type. ``owner`` is the full name of the enclosing
        class."""
        names = [name.strip() for name in re.split(r"\|(?![^<]*>)", written)]
        names = [name.lstrip("?") for name in names if name.lower() != "null"]
        if len(names) == 1 and names[0].endswith("[]"):
            return f"array<int, {self._type(names[0][:-2], owner) or 'mixed'}>"
        read = len(names) == 1 and re.fullmatch(
            r"(\$this|(?:\\|::)?[A-Za-z_][\\\w:]*)(?:<(.*)>)?", names[0]
        )
        if not read or read[1].lower() in _UNTYPED:
            return None
        name, lowered = read[1], read[1].lower()
        arguments = [
            self._type(argument, owner) or "mixed"
            for argument in _bracketed_parts(read[2] or "")
        ]
        if lowered == "list" or (
            lowered in ("array", "iterable") and arguments[1:] == []
        ):
            name = lowered = "array" if lowered == "list" else lowered
            arguments = ["int", *arguments] if arguments else []
        if lowered == "iterable" and not arguments:
            return None
        if lowered in _SCALAR_TYPES or lowered == "iterable":
            name = lowered
        elif name in ("static", "$this"):
            name = "this"
        elif name == "self":
            name = owner
        elif not name.startswith("::"):
            name = self._class_name(name)
        return f"{name}<{', '.join(arguments)}>" if name and arguments else name

    def _returned(self, body, type_id, owner):
        """What every `return` of the lines ``body`` gives, where each gives
        the same one thing that a declaration types: `$this`, a `new`
        object or a typed property of `$this`. None where one gives
        anything else, and for a body that holds a function of its own."""
        found = set()
        for code in body:
            if re.search(r"(?<![\w$>])(?:function|yield)\b", code):
                return None
            for returned in _RE_RETURN.finditer(code):
                value, held = returned["value"].strip(), None
                created = re.match(r"new\s+((?!class\b)\\?[A-Za-z_][\w\\]*)", value)
                read = re.fullmatch(r"\$this->(\w+)", value)
                if created:
                    held = self._type(created[1], owner)
                elif value == "$this" and returned["end"]:
                    held = "this"
                elif read and returned["end"]:
                    held = (self._node(f"{type_id}::{read[1]}") or {}).get("metadata")
                    held = held and held.get("value_type")
                found.add(held or None)
        return found.pop() if len(found) == 1 else None

    def _value(self, code, start, end, lineno, since, owner):
        """Origin path of the expression at columns ``start`` to ``end`` of
        ``code``, the line being read: of the call chain it is, or of the
        variable, property or constant it reads. None for any other."""
        text = code[start:end].strip()
        if text.endswith(")"):
            edge = self._closes.get(start + len(code[start:end].rstrip()) - 1)
            chain = edge is not None and _is_call_chain(self._code[start:end])
            return f"#{edge}" if chain else None
        if not _RE_PATH.fullmatch(text):
            return None
        held = self._receiver(_RE_ACCESS.split(text), lineno, since, owner)
        return None if held == "?" else held

    def _bases(self, header):
        """Full names of the types the header of a type extends and
        implements."""
        return [
            self._class_name(base.split()[0])
            for clause in re.split(r"\b(?:extends|implements)\b", header)[1:]
            for base in clause.split(",")
            if base.split()
        ]

    def _held(self, name, lineno, since, owner):
        """Origin path (see ``PythonParser._origin``) of what the variable
        ``name``, or the property written `name->property`, holds on line
        ``lineno``. `?` where no declaration from line ``since`` on, the
        start of the enclosing definition, types it."""
        value = self._declared(name, lineno, since) or "?"
        if value.endswith("()") and value[0] not in "=:":
            # A type as its declaration writes it.
            held = self._type(value[:-2], owner)
            value = "self" if held == "this" else f"{held}()" if held else "?"
        return value

    def _receiver(self, path, lineno, since, owner):
        """Origin path of the value a call is made on: ``path`` is `$this`,
        a variable or a class, and the properties read from it in turn."""
        root, *properties = path
        narrowed = "?"
        if properties:
            # An `instanceof` types a property as it does a variable.
            name = f"{root}->{properties[0]}".lstrip("$")
            narrowed = self._held(name, lineno, since, owner)
        if narrowed != "?":
            value, properties = narrowed, properties[1:]
        elif root in ("$this", "self", "static"):
            value = "self"
        elif root == "parent":
            value = "super()"
        elif root.startswith("$"):
            value = self._held(root[1:], lineno, since, owner)
        else:
            value = self._class_name(root)
        if value == "?":
            return value
        return ".".join([value, *(name.lstrip("$") for name in properties)])

    def _created(self, name, owner):
        """Origin path of the class that `new name` creates an instance of."""
        held = self._type(name, owner)
        return (owner if held == "this" else held) or "?"

    def _add_constant(self, parent_id, name, lineno, value_type=None):
        """Add the property, constant or enum case ``name`` of the type
        ``parent_id``. A method of the same name keeps `<type>::<name>`."""
        constant_id = f"{parent_id}::{name}"
        if constant_id in self.builder.node_registry:
            constant_id = f"{parent_id}::field_{name}"
        if constant_id not in self.builder.node_registry:
            self.builder.add_node(
                constant_id,
                "Constant",
                name=name,
                lineno=lineno,
                docstring="",
                value_type=value_type,
            )
            self.builder.add_edge(parent_id, constant_id, "DEFINES")

    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)

        lines = source_lines(content)
        skip_calls = is_generated_source(content)
        # Declarations and calls are read from `codes`; `lines` keeps the
        # doc comments.
        codes = _code_lines(lines)

        namespace = None
        for code in codes[:60]:
            m_ns = _RE_NAMESPACE.match(code)
            if m_ns and m_ns.group(1):
                namespace = m_ns.group(1)
                break
        if namespace:
            self.set_file_metadata(file_id, "php_namespace", namespace)
        self._namespace, self._imports, self._functions = "", {}, {}

        # Indented, so that no declaration is read as one at file level: a
        # function sees none of the file's variables.
        self.scan_declarations([f" {code}" for code in codes], _DECLARATIONS)

        # A type (class/interface/trait/enum) is the context for the
        # definitions up to its closing brace; a method or function is the
        # source of the calls in its body.
        self.enter_context(file_id, 0, len(lines))
        types: dict[str, str | None] = {}  # type ID -> full name, if it has one
        enums: set[str] = set()  # IDs of the enums
        # Type ID -> full names of its traits, then of what it extends and
        # implements: a trait's method is found before an inherited one.
        bases: dict[str, list[str]] = {}
        # (ID, name, line, return type) of each method a class declares by
        # a `@method` tag of its doc comment
        tags: list[tuple[str, str, int, str | None]] = []

        i = 0
        n = len(lines)
        while i < n:
            code = codes[i]
            lineno = i + 1
            context_id, context_start = self.context_at(lineno)
            in_type = context_id in types
            # The innermost enclosing type, and the member the calls of
            # the line are made by.
            type_id = context_id
            while type_id not in types and type_id != file_id:
                type_id = type_id.rpartition("::")[0]
            owner = types.get(type_id)
            source_id, since = context_id, context_start
            # Call scanning starts past a name declared on the line.
            call_scan_start = 0

            m_ns = _RE_NAMESPACE.match(code) if context_id == file_id else None
            if m_ns:
                # A namespace has its own imports.
                self._namespace = (m_ns.group(1) or "").replace("\\", "::")
                self._imports, self._functions = {}, {}

            m_use = _RE_USE.match(code) if in_type or context_id == file_id else None
            if m_use and in_type:
                traits = m_use.group(2).partition("{")[0].split(",")
                bases[context_id][:0] = [
                    self._class_name(trait.strip()) for trait in traits if trait.strip()
                ]
            elif m_use:
                statement = code[m_use.start(2) :]
                while ";" not in statement and i + 1 < n:
                    i += 1
                    statement += codes[i]
                tree = statement.partition(";")[0].replace("\\", "::")
                for alias, path in _use_leaves(f"{{{tree}}}"):
                    path = path.lstrip(":")
                    fqn = path.replace("::", "\\")
                    self.builder.add_edge(file_id, f"MODULE:{fqn}", "IMPORTS")
                    if m_use.group(1) == "function":
                        self._functions[alias] = f"::{path}"
                    elif not m_use.group(1):
                        self._imports[alias] = path
                i += 1
                continue

            # Type declarations: class / interface / trait / enum
            m_type = _RE_TYPE_DECL.match(code)
            if m_type:
                kind, name = m_type.groups()
                parent_id = context_id
                type_id = f"{parent_id}::{name}"

                # PHP commonly uses Allman style — `{` often lands on the
                # next line after `class Foo implements X, Y, Z`.
                brace_line_idx = self.opening_brace_line(codes, i, code)
                has_open_brace = brace_line_idx is not None

                end_lineno = (
                    self._find_block_end(codes, brace_line_idx)
                    if has_open_brace
                    else lineno
                )
                docstring = self.get_leading_docstring(lines, i)
                self.builder.add_node(
                    type_id,
                    "Class",
                    name=name,
                    kind=kind,
                    namespace=self._namespace.replace("::", "\\"),
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=(end_lineno - lineno + 1) * 10,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                header = " ".join(codes[i : (brace_line_idx or i) + 1])
                bases[type_id] = self._bases(header.partition("{")[0])
                types[type_id] = self._class_name(f"namespace\\{name}")
                if kind == "enum":
                    enums.add(type_id)
                above = i - 1
                while above >= 0 and lines[above].lstrip().startswith(("*", "/**")):
                    tag = _RE_METHOD_TAG.search(lines[above])
                    if tag:
                        returns = self._type(tag["type"] or "", types[type_id])
                        tags.append(
                            (
                                f"{type_id}::{tag['name']}",
                                tag["name"],
                                above + 1,
                                returns,
                            )
                        )
                    above -= 1
                if has_open_brace and end_lineno > brace_line_idx + 1:
                    self.enter_context(type_id, lineno, end_lineno)
                i = brace_line_idx + 1 if has_open_brace else i + 1
                continue

            m_anonymous = _RE_ANONYMOUS.search(code)
            brace_line_idx = m_anonymous and self.opening_brace_line(
                codes, i, code[m_anonymous.end() :]
            )
            if brace_line_idx is not None:
                column = m_anonymous.start() if brace_line_idx == i else 0
                end_lineno = self._find_block_end(codes, brace_line_idx, column)
                anonymous_id = self._nameless_owner(
                    context_id, f"class@{lineno}", lineno, end_lineno
                )
                header = [code[m_anonymous.end() :], *codes[i + 1 : brace_line_idx + 1]]
                bases[anonymous_id] = self._bases(" ".join(header).partition("{")[0])
                types[anonymous_id] = None
                bound = re.search(r"\$(\w+)\s*=\s*$", code[: m_anonymous.start()])
                if bound:
                    self.declared.setdefault(bound[1], []).append(
                        (lineno, f"=class@{lineno}()", False)
                    )
                # Its header is still code of the member it is created in.
                self.enter_context(anonymous_id, lineno, end_lineno)

            # Methods (inside a type) or functions
            m_method = (_RE_METHOD if in_type else _RE_FUNCTION).match(code)
            if m_method:
                call_scan_start = m_method.end()
                name = m_method.group(1)
                parent_id = context_id if in_type else file_id
                method_id = f"{parent_id}::{name}"
                _, returns, last, has_body = _signature(codes, i, m_method.end())
                end_lineno = self._find_block_end(codes, last) if has_body else last + 1
                docstring = self.get_leading_docstring(lines, i)
                # A property of the same name moves to `<type>::field_<name>`.
                declared = self._node(method_id)
                if declared and declared["type"] == "Constant":
                    self.rename_node(method_id, f"{parent_id}::field_{name}")
                tag = _RE_RETURN_TAG.search(docstring)
                returns = self._type(_told(returns, tag[1] if tag else ""), owner)
                if not returns and has_body:
                    body = [
                        codes[last].partition("{")[2],
                        *codes[last + 1 : end_lineno],
                    ]
                    returns = self._returned(body, context_id, owner)
                # A function is looked up by its namespace.
                space = {"namespace": self._namespace.replace("::", "\\")}
                self.builder.add_node(
                    method_id,
                    "Method" if in_type else "Function",
                    name=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=(end_lineno - lineno + 1) * 10,
                    docstring=docstring,
                    return_type=returns or "",
                    **({} if in_type else space),
                )
                self.builder.add_edge(parent_id, method_id, "DEFINES")
                if end_lineno > lineno:
                    self.enter_context(method_id, lineno, end_lineno)
                source_id, since = method_id, lineno
                told = {t["name"]: t["type"] for t in _RE_PARAM_TAG.finditer(docstring)}
                for parameter, written in told.items():
                    entries = self.declared.setdefault(parameter, [])
                    native = [e for e in entries if lineno <= e[0] <= last + 1]
                    if not native:
                        entries.append((lineno, f"{written}()", False))
                    elif _RE_TYPE_ARGUMENTS.search(written):
                        self._rebind(parameter, native[0][0], f"{written}()")
                if in_type and name == "__construct":
                    for index in range(i, last + 1):
                        for promoted in _RE_PROMOTED.finditer(codes[index]):
                            written = _told(
                                promoted["type"] or "", told.get(promoted["name"])
                            )
                            self._add_constant(
                                parent_id,
                                promoted["name"],
                                index + 1,
                                self._type(written, owner),
                            )
            elif in_type:
                # Properties / constants / enum cases
                m_prop = _RE_PROPERTY.match(code)
                if m_prop:
                    tag = _RE_VAR_TAG.search(self.get_leading_docstring(lines, i))
                    written = _told(m_prop["type"] or "", tag[1] if tag else "")
                    self._add_constant(
                        context_id, m_prop["name"], lineno, self._type(written, owner)
                    )
                m_const = _RE_CONSTANT.match(code)
                if m_const:
                    self._add_constant(context_id, m_const.group(1), lineno)
                m_case = context_id in enums and _RE_ENUM_CASE.match(code)
                if m_case:
                    self._add_constant(context_id, m_case.group(1), lineno, owner)

            # What the line says of its variables: `/** @var Type $name */`
            # types the statement below it; a rebinding leaves a variable
            # untyped; a copy has the type of what it copies; `instanceof`
            # types a variable inside the block the line opens, or from an
            # `assert` on.
            for tag in _RE_VARIABLE_TAG.finditer(lines[i]):
                self.declared.setdefault(tag["name"], []).append(
                    (lineno + 1, f"{tag['type']}()", False)
                )
            for rebinding in _REBINDINGS:
                for found in rebinding.finditer(code):
                    for name in re.findall(r"\$(\w+)", found["names"]):
                        self._unbind(name, lineno)
            copied = _RE_COPY.match(code)
            if copied:
                path = _RE_ACCESS.split(copied["path"])
                held = self._receiver(path, lineno, since, owner)
                if held != "?":
                    self.declared.setdefault(copied["name"], []).append(
                        (lineno, held, False)
                    )
            narrowed: dict[str, set[str]] = {}
            for found in _RE_INSTANCEOF.finditer(code):
                narrowed.setdefault(found["name"], set()).add(found["type"])
            asserted = code.lstrip().startswith("assert(")
            opens_block = code.rstrip().endswith("{") and not re.search(
                r"\|\||\bor\b|!\s*\(|\?", code
            )
            for name, held in narrowed.items():
                if len(held) == 1 and (asserted or opens_block):
                    entries = self.declared.setdefault(name, [])
                    before = self._declared(name, lineno, since) or "?"
                    entries.append((lineno, f"{held.pop()}()", False))
                    if not asserted:
                        end = self._find_block_end(codes, i, code.rindex("{"))
                        entries.append((end, before, False))

            # Calls (inside any context — methods, functions, etc.)
            if not skip_calls and context_id != file_id:
                chained = _RE_OPERATOR.sub(
                    lambda found: _CHAIN_OPERATORS[found.group()], code
                )
                self.begin_line(chained, lineno, raw=lines[i])
                for created in _RE_NEW.finditer(code, call_scan_start):
                    origin = self._created(created[1], owner)
                    column = created.start(1)
                    self.add_call(
                        source_id, created[1], lineno, since, column, -1, origin
                    )
                    # `(new Type)->name(` is a call on the instance.
                    self.continue_chain(created.end(1) - 1, len(self.builder.edges) - 1)
                for call_match in _RE_CALL.finditer(code, call_scan_start):
                    callee = call_match.group(1)
                    start = call_match.start(1)
                    *path, name = _RE_ACCESS.split(callee)
                    before = code[:start].rstrip()
                    function = None
                    if name.startswith("$"):
                        # A call of what a variable holds or names.
                        continue
                    if before.endswith(("->", "::")):
                        # A call on what the expression before it gives:
                        # another call, one in parentheses of their own, or
                        # the path the line above ends with.
                        receiver = before[:-2].rstrip().removesuffix("?").rstrip()
                        edge = (
                            self._closes.get(len(receiver) - 1)
                            if receiver
                            else self._tail
                        )
                        if edge is None and receiver.endswith(")"):
                            end = len(receiver) - 1
                            opened = [o for o, c in self._pairs.items() if c == end]
                            inner = opened and chained[opened[0] + 1 : end].rstrip()
                            if inner and _is_call_chain(inner):
                                edge = self._closes.get(opened[0] + len(inner))
                        above = not receiver and _RE_TRAILING_PATH.search(codes[i - 1])
                        if edge is not None:
                            properties = (step.lstrip("$") for step in path)
                            origin = ".".join([f"#{edge}", *properties, name])
                        elif above:
                            path = [*_RE_ACCESS.split(above.group().strip()), *path]
                            origin = self._receiver(path, lineno, since, owner)
                            origin = f"{origin}.{name}"
                        else:
                            origin = f"?.{name}"
                    elif path:
                        origin = f"{self._receiver(path, lineno, since, owner)}.{name}"
                    elif re.search(r"\bnew$", before):
                        if name == "class":
                            # The anonymous class the line declares.
                            callee = f"class@{lineno}"
                            origin = f"={callee}"
                        else:
                            origin = self._created(name, owner)
                    elif name in _PHP_CONTROL_KEYWORDS:
                        continue
                    else:
                        # A function, named in full: the one imported
                        # under the name or written with a namespace, else,
                        # without the first `\`, the one of the namespace
                        # or failing that the global one.
                        origin = None
                        exact = "\\" in name or name in self._functions
                        function = self._functions.get(name) or self._class_name(
                            name if exact else f"namespace\\{name}"
                        )
                        function = function.replace("::", "\\")
                        if not exact:
                            function = function[1:]
                    self.add_call(
                        source_id,
                        callee,
                        lineno,
                        since,
                        start,
                        call_match.end() - 1,
                        origin,
                    )
                    if function:
                        self.builder.edges[-1]["target"] = f"REF:{function}"
                # What a `foreach` or a destructuring assignment takes from
                # a value is an element of it, unless a `@var` says more.
                for found in (_RE_FOREACH.search(code), _RE_DESTRUCTURING.match(code)):
                    over = found and self._value(
                        code, *found.span("over"), lineno, since, owner
                    )
                    for name in re.findall(r"\$(\w+)", found["names"]) if over else ():
                        if self._declared(name, lineno, lineno) == "?":
                            self._rebind(name, lineno, f"{over}.*")

            i += 1

        for type_id, names in bases.items():
            for name in names:
                self.builder.add_edge(type_id, f"REF:{name}", "INHERITS")
        # A method a class declares only by a `@method` tag.
        for method_id, name, lineno, returns in tags:
            if method_id not in self.builder.node_registry:
                self.builder.add_node(
                    method_id,
                    "Method",
                    name=name,
                    lineno=lineno,
                    end_lineno=lineno,
                    token_count=10,
                    docstring="",
                    return_type=returns or "",
                )
                self.builder.add_edge(
                    method_id.rpartition("::")[0], method_id, "DEFINES"
                )
