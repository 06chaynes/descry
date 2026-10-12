"""Ruby regex-based baseline parser.

Covers the common shapes of Ruby symbol extraction: modules, classes
(with superclass and mixins), methods (``def name``, ``def self.name``,
one-line and endless definitions, ``define_method``, ``alias``), accessors
(``attr_reader``/``attr_writer``/``attr_accessor``, ``Struct.new`` members),
requires, constants, and call sites with or without parentheses.

Ruby closes blocks with ``end`` rather than ``}``, so this parser turns
comments and the text of literals to spaces and then pairs every ``class``,
``module``, ``def``, ``do``, ``begin``, ``if``, ``unless``, ``while``,
``until``, ``case`` and ``for`` with its ``end``. A definition belongs to
the innermost class or module that is open where it is written, a call to
the innermost definition. A name read without a receiver is a call unless
it is a local variable there, as it is to Ruby.
"""

from __future__ import annotations

import re

from descry.generate import BaseParser, _strip_parens, is_generated_source

_CONST = r"(?:::)?[A-Z]\w*(?:::[A-Z]\w*)*"
# A method name: an identifier, a setter, or an operator
_METHOD_NAME = (
    r"(?:[A-Za-z_]\w*(?:[?!]|=(?![=~>]))?|\[\]=?|<=>|===?|=~|![=~]?|<<|>>|<=?|>=?"
    r"|\*\*?|[-+]@?|[/%&|^~])"
)
# class Foo, class Foo < Bar, class Foo::Bar
_RE_CLASS = re.compile(rf"class\s+({_CONST})(?:\s*<\s*(?P<base>{_CONST}))?")
# class << self, class << object
_RE_SINGLETON_CLASS = re.compile(r"class\s*<<\s*(\w+)")
# module Foo
_RE_MODULE = re.compile(rf"module\s+({_CONST})")
# def foo, def self.foo, def object.foo
_RE_METHOD = re.compile(
    rf"def\s+(?:(?P<owner>[A-Za-z_]\w*)\s*\.\s*)?(?P<name>{_METHOD_NAME})"
)
# `= value` after the name or the parameters of an endless method
_RE_ENDLESS = re.compile(r"\s*(?:\((?:[^()]|\([^()]*\))*\))?\s*=(?![=~>])")
# require 'x', require "x", require_relative 'x'
_RE_REQUIRE = re.compile(
    r"^\s*(?:require|require_relative|load|autoload)\s+[\"']([^\"']+)[\"']"
)
# CONSTANT_NAME = value
_RE_CONSTANT = re.compile(r"\s*([A-Z]\w*)\s*(?:\|\|)?=(?![=~>])(?P<value>.*)")
# attr_reader :foo, :bar / attr_writer :x / attr_accessor :y
_RE_ATTR = re.compile(r"(?<![\w.])attr(?:_(reader|writer|accessor))?\b\s*\(?\s*(?=:)")
# alias new old / alias_method :new, :old
_RE_ALIAS = re.compile(rf"(?<![\w.])alias(?:_method)?\b\s*+\(?\s*+:?({_METHOD_NAME})")
# include Foo, Bar / extend Foo / prepend Foo
_RE_MIXIN = re.compile(
    rf"\s*+(?:include|extend|prepend)\b\s*+\(?\s*+({_CONST}(?:\s*,\s*{_CONST})*)\s*+\)?\s*$"
)
# `Class.new(Base)`, `Module.new`, `Struct.new(:a, :b)`, `Data.define(:a)`:
# an expression that makes a type, whose block is the body of that type
_RE_TYPE_MAKER = re.compile(
    r"(?<![\w:.])(?:::)?(?P<maker>Class|Module|Struct|Data)\.(?:new|define)\b"
    r"\s*(?P<arguments>\((?:[^()]|\([^()]*\))*\)|[^;{]*?(?=\s*(?:\bdo\b|\{|$)))"
)
# define_method(:name)
_RE_DEFINE_METHOD = re.compile(
    rf"(?<![\w.])define_(?:singleton_)?method\s*\(?\s*:({_METHOD_NAME})"
)
# `Const.new`, with any arguments, as a whole value
_RE_CONSTRUCTED = re.compile(
    rf"\s*(?P<type>{_CONST})\s*\.\s*new\b"
    r"(?:\((?:[^()]|\([^()]*\))*\)|[ \t]+(?=[\w:\"@$\[*&])(?:(?!\bdo\b)[^{};])*?)?"
    r"\s*(?:\b(?:if|unless)\b.*)?"
)
# A chain of names and calls, once the arguments of its calls are taken out
_RE_CHAIN = re.compile(
    rf"\s*(?:{_CONST}|@{{0,2}}[a-z_]\w*[?!]?)(?:\s*\.\s*[a-z_]\w*[?!]?)*\s*"
)
# `name = value` / `name ||= value` as a statement
_RE_ASSIGNMENT = re.compile(r"\s*(?P<name>[a-z_]\w*)\s*(?:\|\|)?=(?![=~>])")
# A local variable where it is assigned: `name =`, `name +=`
_RE_ASSIGNED = re.compile(
    r"(?<![\w.@$:])([a-z_]\w*)\s*(?:\|\||&&|\*\*|<<|>>|[-+*/%|&^])?=(?![=~>])"
)
# `name, @other, *rest = value` as a statement, its targets in group 1
_RE_MULTIPLE = re.compile(r"\s*((?:[*@\w.\[\]]+\s*,\s*)+[*@\w.\[\]]*)\s*=(?![=~>])")
# `rescue Error => name`, `for name, other in`
_RE_BOUND = re.compile(
    r"\brescue\b[^#]*?=>\s*([a-z_]\w*)|\bfor\s+([a-z_][\w\s,]*?)\s+in\b"
)
# An instance variable where it is assigned
_RE_IVAR_ASSIGNED = re.compile(
    r"(?<![\w@])(@\w+)\s*(?:(?P<plain>(?:\|\|)?=(?![=~>]))(?P<value>.*)"
    r"|(?:&&|\*\*|<<|>>|[-+*/%|&^])=(?![=~>]))"
)
# The parameters of a block, behind its `do` or `{`, and of a lambda
_RE_BLOCK_PARAMS = re.compile(r"\s*\|([^|]*)\|")
_RE_LAMBDA_PARAMS = re.compile(r"->\s*\(?([^(){}]*?)\)?\s*$")
# A parameter where a parameter list names it
_RE_PARAM = re.compile(r"(?:^|[(,;|])\s*[*&]*([a-z_]\w*)")
# `@param name [Type]` / `@param [Type] name` and `@return [Type]` in YARD
_RE_YARD_PARAM = re.compile(
    rf"@param\s+(?:(?P<name>\w+)\s+\[(?P<type>{_CONST})\]|\[(?P<type2>{_CONST})\]\s+(?P<name2>\w+))"
)
_RE_YARD_RETURN = re.compile(rf"@return\s+\[({_CONST})\]")
# sig { params(name: Type).returns(Type) } in Sorbet
_RE_SIG = re.compile(r"\s*sig\b")
_RE_SIG_PARAM = re.compile(rf"(\w+):\s*({_CONST})\s*[,)]")
_RE_SIG_RETURN = re.compile(rf"\breturns\(\s*({_CONST})\s*\)")
# A name where it may be called: not part of a longer one, nor a symbol,
# an instance or a global variable
_RE_NAME = re.compile(r"(?<![\w@$])(?<!(?<!:):)[a-z_]\w*(?:[?!](?![=~]))?")
# The name, variable or constant a `.` comes after, where that is all of
# the receiver
_RE_RECEIVER = re.compile(
    rf"(?<![\w@$.:)\]}}\"])(?:self(?:\s*\.\s*class)?|@@?\w+|\$\w+|[a-z_]\w*|{_CONST})$"
)
# A word that opens or closes a block or defines a method, and a brace
_RE_BLOCK_WORD = re.compile(
    r"(?<![\w.:@$])(?:class|module|def|do|begin|if|unless|while|until|case|for|end"
    r"|alias_method|alias|attr_accessor|attr_reader|attr_writer)(?![\w?!:])|[{}]"
)
# The method a block is passed to: the last name before the block, or the
# first of a statement that calls without parentheses
_RE_BLOCK_CALL = re.compile(
    r"(?<![\w@$:])(?P<method>[a-z_]\w*[?!]?)\s*(?:\((?:[^()]|\([^()]*\))*\))?\s*$"
)
_RE_COMMAND = re.compile(
    r"\s*(?:[\w@]+\s*=\s*)?(?:[\w@:]+\s*\.\s*)*(?P<method>[a-z_]\w*[?!]?)(?=\s)"
)
# Methods that run their block in the object they are called on
_EVALUATORS = frozenset(
    {
        "class_eval",
        "class_exec",
        "module_eval",
        "module_exec",
        "instance_eval",
        "instance_exec",
    }
)
# The last statement of a method written on one line, behind its parameters
_RE_INLINE_BODY = re.compile(
    r"\s*(?:\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\))?\s*(?:=(?![=~>])|[^;]*;)?"
    r"(?:[^;]*;)*?\s*(?!end\s*$)(?P<body>[^;]+?)\s*+(?:;\s*+)?(?:\bend)?\s*$"
)
# `return value` in a method body
_RE_RETURN = re.compile(r"(?<![\w.@$:])return\b(?P<value>[^;]*)")
# What follows the name of a call without parentheses where an argument
# does: a space, then the start of an operand rather than an operator
_RE_ARGUMENT_FOLLOWS = re.compile(
    r"[ \t]+(?!(?:if|unless|while|until|and|or|then|do|rescue|in|end)\b)"
    r"(?:[\w\"@$\[]|:(?=[\w\":])|->|[-*&!](?=[\w:\"@$\[(]))"
)
# Where the arguments of a call without parentheses end: a block or a
# statement modifier
_RE_ARGUMENTS_END = re.compile(
    r"\s(?:do|if|unless|while|until|rescue|and|or|then)\b|;|(?<![,(\[{\s])\s*\{"
    r"|(?<=[\w)\]\"])\s+\{"
)
# What an `if`, `unless`, `while` or `until` that modifies the statement
# before it, rather than opens a block of its own, comes after: the end of
# an expression
_RE_MODIFIED = re.compile(
    r"(?<!(?<![\w.])then)(?<!(?<![\w.])else)(?<!(?<![\w.])do)(?<!(?<![\w.])and)"
    r"(?<!(?<![\w.])or)(?<!(?<![\w.])not)(?<!(?<![\w.])begin)(?<!(?<![\w.])ensure)"
    r"(?<=[\w)\]}\"?!])\s*$"
)
# What a `{` that opens a block, not a hash, comes after
_RE_BLOCK_EXPECTED = re.compile(r"(?:[\w)?!]|->)\s*$")
# The opening of a %-literal and of a heredoc
_RE_PERCENT = re.compile(r"%(?:([qQwWiIrsx])[^\w\s]|[(\[{<|!/])")
_RE_HEREDOC = re.compile(r"<<([~-]?)(['\"`]?)([A-Za-z_]\w*)\2")
# What a `/` or `%` that opens a literal, not an operator, comes after
_RE_OPERAND_EXPECTED = re.compile(
    r"(?:^|[(\[{,=!|&?:;~<>+\-*%^]"
    r"|(?<![\w.])(?:if|elsif|unless|while|until|when|and|or|not|return|then|in))\s*$"
)
# A regexp from behind its opening `/` to its closing one, and its flags
_RE_REGEXP_END = re.compile(r"(?:[^/\\]|\\.)*/")
_RE_REGEXP_FLAGS = re.compile(r"[a-z]*")
_PAIRED = {"(": ")", "[": "]", "{": "}", "<": ">"}

# Words the call scan passes over: keywords, and the methods of Kernel and
# Module that load code, mix modules in, raise and print
_RUBY_KEYWORDS = frozenset(
    {
        "if",
        "unless",
        "while",
        "until",
        "for",
        "in",
        "case",
        "when",
        "else",
        "elsif",
        "begin",
        "rescue",
        "ensure",
        "return",
        "yield",
        "break",
        "next",
        "redo",
        "retry",
        "raise",
        "throw",
        "catch",
        "lambda",
        "proc",
        "loop",
        "do",
        "end",
        "then",
        "and",
        "or",
        "not",
        "defined?",
        "alias",
        "undef",
        "class",
        "module",
        "def",
        "self",
        "true",
        "false",
        "nil",
        "__FILE__",
        "__LINE__",
        "__ENCODING__",
        "require",
        "require_relative",
        "load",
        "autoload",
        "include",
        "extend",
        "prepend",
        "puts",
        "print",
        "p",
        "pp",
    }
)


def _opens_literal(before, after):
    """Whether a `/` or `%` between ``before`` and ``after`` on its line
    opens a literal: an operand is expected there, or it is the first
    argument of a call without parentheses, a space before it and none
    after."""
    if _RE_OPERAND_EXPECTED.search(before):
        return True
    called = before.rstrip()[-1:]
    spaced = before[-1:].isspace() and (called.isalnum() or called in ("_", "?", "!"))
    return spaced and after[:1] not in ("", " ", "=")


def _code_lines(lines):
    """``lines`` with their comments, the text of their string, symbol,
    regexp and %-literals and the bodies of their heredocs turned to spaces,
    so that none of it is read as code. Each literal is left as `"` at
    either end; code interpolated into one with `#{...}` stays."""
    out = []
    # (closer, opener that nests, interpolates, nesting depth, is a regexp)
    # of each literal open, innermost last; the closer of a `#{` is None.
    stack = []
    heredocs = []  # terminators of the heredocs whose bodies come next
    comment = False  # inside `=begin` ... `=end`
    for index, line in enumerate(lines):
        if heredocs:
            if line.strip() == heredocs[0]:
                heredocs.pop(0)
            out.append("")
            continue
        if not stack:
            if comment or line.startswith("=begin"):
                comment = not line.startswith("=end")
                out.append("")
                continue
            if line.rstrip() == "__END__":
                break
        code = []
        k = 0
        while k < len(line):
            char = line[k]
            frame = stack[-1] if stack else None
            if frame and frame[0]:
                closer, opener, interpolates, depth, regexp = frame
                if char == "\\":
                    code.append("  "[: len(line) - k])
                    k += 2
                elif interpolates and line.startswith("#{", k):
                    stack.append([None, None, False, 0, False])
                    code.append("  ")
                    k += 2
                elif char == closer and not depth:
                    stack.pop()
                    end = _RE_REGEXP_FLAGS.match(line, k + 1).end() if regexp else k + 1
                    code.append('"' + " " * (end - k - 1))
                    k = end
                else:
                    frame[3] += (char == opener) - (char == closer)
                    code.append(" ")
                    k += 1
                continue
            if char == "#" and not frame:
                break
            if char in "\"'`":
                stack.append([char, None, char != "'", 0, False])
                code.append('"')
                k += 1
            elif char == "/" and _opens_literal("".join(code), line[k + 1 :]):
                # Where an operand is expected a regexp may run on over
                # the next lines.
                closed = _RE_REGEXP_END.match(line, k + 1)
                if closed or _RE_OPERAND_EXPECTED.search("".join(code)):
                    stack.append(["/", None, True, 0, True])
                    code.append('"')
                else:
                    code.append(char)
                k += 1
            elif (
                char == "%"
                and (found := _RE_PERCENT.match(line, k))
                and _opens_literal("".join(code), line[k + 1 :])
            ):
                opener = found.group()[-1]
                stack.append(
                    [
                        _PAIRED.get(opener, opener),
                        opener if opener in _PAIRED else None,
                        (found[1] or "Q") not in "qwis",
                        0,
                        found[1] == "r",
                    ]
                )
                code.append('"' + " " * (len(found.group()) - 1))
                k = found.end()
            elif (
                char == "<"
                and (found := _RE_HEREDOC.match(line, k))
                and (found[1] or found[2] or _opens_literal("".join(code), found[3]))
                and any(later.strip() == found[3] for later in lines[index + 1 :])
            ):
                heredocs.append(found[3])
                code.append('""' + " " * (len(found.group()) - 2))
                k = found.end()
            elif char == "?" and (
                re.match(r"\?[\"'`#/%](?!\w)", line[k:])
                or re.match(r"\?\w(?!\w)", line[k:])
                and _RE_OPERAND_EXPECTED.search("".join(code))
            ):
                # A character literal: of what would open a literal or
                # comment, or of a letter where an operand is expected.
                code.append("  ")
                k += 2
            elif char == "$" and not re.match(r"\$\w", line[k:]):
                # A special global variable: `$'`, `$"`, `$/`.
                code.append("$ "[: len(line) - k])
                k += 2
            elif frame and char in "{}":
                if char == "}" and not frame[3]:
                    # The code interpolated is a statement of its own.
                    stack.pop()
                    char = ";"
                else:
                    frame[3] += 1 if char == "{" else -1
                code.append(char)
                k += 1
            else:
                code.append(char)
                k += 1
        out.append("".join(code))
    out.extend("" for _ in lines[len(out) :])
    return out


def _blocks(code_lines):
    """For each line of ``code_lines`` the blocks that open on it, in
    order, as ``[column, word, index of the line it closes on, how many
    blocks are open inside it, whether it defines a method]``: every
    `class`, `module` and `def`, and every `do` or `{` that opens a block
    passed to a call. A block defines a method where a `def`, an `alias`
    or an `attr_*` is written in it and in no definition inside it. With
    that, for each line, how many blocks of any kind are open where it
    starts."""
    opened = [[] for _ in code_lines]
    levels = []
    stack = []  # (the entry of a block listed in `opened` or None, is a brace)
    loop = False  # a `while`, `until` or `for` whose `do` is still to come
    for index, line in enumerate(code_lines):
        levels.append(len(stack))
        # The `do` is on the loop's line or starts the next.
        loop = loop and bool(re.match(r"\s*do\b", line))
        position = 0
        while found := _RE_BLOCK_WORD.search(line, position):
            word, column = found.group(), found.start()
            position = found.end()
            if word in ("}", "end"):
                # A brace left open is none of an `end`'s business.
                while word == "end" and stack and stack[-1][1]:
                    stack.pop()
                if stack and stack[-1][1] == (word == "}"):
                    entry = stack.pop()[0]
                    if entry:
                        entry[2] = index
                continue
            if word == "def" or word.startswith(("alias", "attr")):
                around = next((e for e, _ in reversed(stack) if e), None)
                if around and around[1] in ("do", "{"):
                    around[4] = True
                if word != "def":
                    continue
            modifies = word in ("if", "unless", "while", "until")
            if modifies and _RE_MODIFIED.search(line[:column]):
                continue
            if word == "do" and loop:
                loop = False
                continue
            entry = None
            if word == "{":
                if _RE_BLOCK_EXPECTED.search(line[:column]):
                    entry = [column, word, len(code_lines) - 1, len(stack) + 1, False]
            elif word in ("class", "module", "def", "do"):
                entry = [column, word, len(code_lines) - 1, len(stack) + 1, False]
                header = _RE_METHOD.match(line, column) if word == "def" else None
                if header:
                    # The name is no keyword, and an endless method no block.
                    position = header.end()
                    if _RE_ENDLESS.match(line, position):
                        entry[2] = index
                        opened[index].append(entry)
                        continue
            if entry:
                opened[index].append(entry)
            loop = word in ("while", "until", "for")
            stack.append((entry, word == "{"))
    return opened, levels


def _continued(code_lines, index, text):
    """``text``, which line ``index`` ends with, and the lines it runs on
    to while it has a parenthesis open or ends with a comma."""
    last = min(index + 30, len(code_lines) - 1)
    while index < last and (
        text.count("(") > text.count(")") or text.rstrip().endswith(",")
    ):
        index += 1
        text += " " + code_lines[index]
    return text


def _constructed(value):
    """Origin path of ``value`` where all of it is a `Const.new`, with any
    arguments in parentheses: an instance of Const, or for `Class.new(Base)`
    the class Base, whose methods the new class has, and `?` for any other
    new type. None for any other value."""
    built = _RE_CONSTRUCTED.fullmatch(value)
    made = built and _RE_TYPE_MAKER.match(value.lstrip())
    if made:
        base = re.fullmatch(rf"\(\s*({_CONST})\s*\)", made["arguments"])
        subclass = base and made["maker"] == "Class"
        return base[1].removeprefix("::") if subclass else "?"
    return built and f"{built['type'].removeprefix('::')}.new()"


def _parameters(text):
    """Names of the parameters in the text of a parameter list."""
    return [found[1] for found in _RE_PARAM.finditer(text)]


def _items(text):
    """The items of the list ``text``, split at its commas outside brackets."""
    items, depth, start = [], 0, 0
    for column, char in enumerate(f"{text},"):
        depth += (char in "([{") - (char in ")]}")
        if char == "," and depth == 0:
            items.append(text[start:column].strip())
            start = column + 1
    return [item for item in items if item]


def _arity(params):
    """What the parameter list ``params`` can take, as ``[required
    positional arguments, most of them or None for any number, keywords: 0
    for none, 1 where it takes some, 2 where it requires one]``."""
    required = optional = keywords = 0
    rest = False
    for param in _items(params):
        if param.startswith(("**", "...")):
            keywords = max(keywords, 1)
            rest = rest or param == "..."
        elif param.startswith("*"):
            rest = True
        elif label := re.match(r"\w+:\s*(\S?)", param):
            keywords = max(keywords, 1 if label[1] else 2)
        elif "=" in param:
            optional += 1
        elif not param.startswith("&"):
            required += 1
    return [required, None if rest else required + optional, keywords]


def _arguments(text):
    """What the argument list ``text`` passes, as ``[positional arguments,
    whether keyword ones follow]``. None where it cannot be counted: a
    splat, or an argument that is itself a call without parentheses."""
    count = keywords = 0
    for item in _items(text):
        flat, wider = item, None
        while flat != wider:
            wider, flat = flat, re.sub(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}", "", flat)
        if item.startswith("*") and not item.startswith("**"):
            return None
        if re.match(r"[a-z_]\w*[?!]?[ \t]+[\w:\"@$\[]", flat) and not re.match(
            r"\w+\s+(?:if|unless|and|or|rescue|while|until|then|do|\?|:)", flat
        ):
            return None
        if (
            item.startswith("**")
            or re.match(r"\w+:(?!:)|\"[^\"]*\":", flat)
            or "=>" in flat
        ):
            keywords = 1
        elif not item.startswith("&"):
            count += 1
    return [count, bool(keywords)]


def _command_arguments(rest):
    """Text of the arguments that a call without parentheses is given,
    where ``rest`` is what follows its name on the line: up to a block, a
    statement modifier or the end of a bracket around the call. Empty where
    no argument follows; None where the list runs on to the next line."""
    if rest.rstrip().endswith("\\"):
        return None
    if not _RE_ARGUMENT_FOLLOWS.match(rest):
        return ""
    depth = 0
    for column, char in enumerate(rest):
        depth += (char in "([{") - (char in ")]}")
        if depth < 0 or (depth == 0 and _RE_ARGUMENTS_END.match(rest, column)):
            rest = rest[:column]
            break
    return None if rest.rstrip().endswith(",") or depth > 0 else rest


class RubyParser(BaseParser):
    """Regex Ruby parser."""

    doc_line_prefix = "#"
    doc_strip_chars = "# "

    def _add(self, parent_id, name, node_type, lineno, end_lineno, **metadata):
        """Add the node of the definition ``name`` of ``parent_id``, unless
        that has one of the name already, and return its ID."""
        node_id = f"{parent_id}::{name}"
        span = {
            "lineno": lineno,
            "end_lineno": end_lineno,
            "token_count": (end_lineno - lineno + 1) * 10,
        }
        known = self._types.get(node_id)
        if node_id not in self.builder.node_registry:
            self.builder.add_node(
                node_id,
                node_type,
                name=name.rpartition("::")[2].rpartition(".")[2] or name,
                **span,
                **metadata,
            )
            self.builder.add_edge(parent_id, node_id, "DEFINES")
            if node_type == "Class":
                self._types[node_id] = self.builder.nodes[-1]["metadata"]
        elif known and known["end_lineno"] == known["lineno"] and lineno < end_lineno:
            # A type first named on one line, `module Empty; end`, spans the
            # first body that reopens it.
            known.update(span, docstring=metadata["docstring"])
        return node_id

    def _add_accessors(self, parent_id, kind, symbols, lineno):
        """Add the methods that an `attr_reader`, `attr_writer` or
        `attr_accessor`, as ``kind`` says, defines for ``symbols``."""
        for symbol in symbols:
            names = {"reader": [symbol], "writer": [f"{symbol}="]}.get(
                kind, [symbol, f"{symbol}="]
            )
            for name in names:
                taken = int(name.endswith("="))
                self._add(
                    parent_id,
                    name,
                    "Method",
                    lineno,
                    lineno,
                    docstring="",
                    is_accessor=True,
                    arity=[taken, taken, 0],
                )

    def _add_made(self, type_id, made, lineno):
        """Give ``type_id`` what the arguments of ``made``, a match of
        ``_RE_TYPE_MAKER``, make for it: the accessors of a `Struct.new` or
        `Data.define`, the base class of a `Class.new`."""
        if made["maker"] == "Class":
            for base in re.findall(_CONST, made["arguments"])[:1]:
                self.builder.add_edge(
                    type_id, f"REF:{base.removeprefix('::')}", "INHERITS"
                )
        elif made["maker"] != "Module":
            self._add_accessors(
                type_id,
                "accessor" if made["maker"] == "Struct" else "reader",
                re.findall(r":(\w+)", made["arguments"]),
                lineno,
            )

    def _kind(self, node_id):
        """Kind of the type of this file with ID ``node_id``: `class`,
        `module`, `struct` or `block`. None for the file."""
        return self._types.get(node_id, {}).get("kind")

    def _exit(self, text, raw, lineno, since, owner_id):
        """What a method returns where it returns the expression ``text``,
        which is ``raw`` before its literals were blanked: ``("type",
        name)`` for an instance of a type, `Self` for the receiver;
        ``("call", callee, origin)`` for the result of a call, as an edge
        has it; ``("held", type ID, name)`` for an instance variable;
        ``"nil"``; None where the expression does not tell."""
        text = re.sub(r"\s+(?:if|unless)\b.*", "", text)
        # A method that ends in an assignment returns the value assigned.
        assigned = re.match(r"\s*@?[a-z_]\w*\s*(?:\|\|)?=(?![=~>])", text)
        start = assigned.end() if assigned else 0
        start += len(text[start:]) - len(text[start:].lstrip())
        text, raw = text[start:].rstrip(), raw[start:]
        flat, wider = text, None
        while flat != wider:
            wider, flat = flat, re.sub(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}", "", flat)
        if text in ("", "nil"):
            return "nil"
        if text == "self":
            return ("type", "Self")
        if re.fullmatch(r'"[^"]*"', text):
            if raw.startswith(("/", "%r")):
                return ("type", "Regexp")
            if re.match(r"%[wWiI]", raw):
                return ("type", "Array")
            return ("type", "Symbol" if raw.startswith("%s") else "String")
        if re.fullmatch(r':\w+[?!=]?|:"[^"]*"', text):
            return ("type", "Symbol")
        if number := re.fullmatch(r"\d[\d_]*(\.\d+)?", text):
            return ("type", "Float" if number[1] else "Integer")
        if not flat and text[0] in "[{":
            return ("type", "Array" if text[0] == "[" else "Hash")
        made = _constructed(text)
        if made:
            return ("call", made[:-6], made[:-2]) if made.endswith(".new()") else None
        if re.fullmatch(r"@\w+", text):
            return ("held", owner_id, text)
        chain = _RE_CHAIN.match(flat)
        after = flat[chain.end() :] if chain else "?"
        command = (
            chain
            and chain.group()[-1:].isspace()
            and not re.search(r"\bdo\b|\{", after)
        )
        if not chain or after and not (command and re.match(r"[\w:\"@$\[]", after)):
            return None
        root, *links = re.split(r"\s*\.\s*", chain.group().strip())
        held = None if root[:1].isupper() else self._declared(root, lineno, since)
        if root[:1].isupper() or root.startswith("::"):
            origin = root.removeprefix("::")
        elif root.startswith(("@", "$")) or (held and not held.endswith("()")):
            return None
        elif held:
            if not links:
                typed = held.endswith(".new()")
                return ("call", held[:-6], held[:-2]) if typed else ("type", held[:-2])
            origin = held
        elif root == "self":
            origin = "self"
        elif root == "new" and self._kind(owner_id) == "class":
            links, origin = ["new", *links], "cls"
        elif root in _RUBY_KEYWORDS or root == "super":
            return None
        elif not links:
            return ("call", root, None)
        else:
            origin = f"self.{root}()"
        if not links:
            return None
        origin += "".join(f".{link}()" for link in links[:-1])
        return ("call", links[-1], f"{origin}.{links[-1]}")

    def _declare(self, name, lineno, since, origin, depth=None):
        """Bind the local variable ``name`` on ``lineno`` to a value of
        origin path ``origin``, `?` for one of no known type. A name new to
        the definition that starts on line ``since`` lasts to the end of
        the innermost block open, or of the innermost of the ``depth``
        outermost ones."""
        entries = self.declared.setdefault(name, [])
        scopes = self._scopes[:depth]
        if scopes and not self._declared(name, lineno, since):
            scopes[-1][1].setdefault(name, len(entries))
        entries.append((lineno, origin, False))

    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)

        lines = content.splitlines()
        skip_calls = is_generated_source(content)

        code_lines = _code_lines(lines)
        opened, levels = _blocks(code_lines)

        # Requires — one pass over the file
        for line, code in zip(lines, code_lines):
            m = code.strip() and _RE_REQUIRE.match(line)
            if m:
                self.builder.add_edge(file_id, f"MODULE:{m.group(1)}", "IMPORTS")

        self.enter_context(file_id, 0, len(lines))
        methods = {}  # ID of each method that is a context -> its name
        # ID of each of those whose node is new -> what its body returns:
        # its node, the depth of its statements, what each `return` gives
        # and what its last statement does (see `_exit`)
        bodies = {}
        self._types = {}  # ID of each type of the file -> its metadata
        # (last line, {local variable: where its bindings there start in
        # `declared`}, first line if a `class`, `module` or `def` opens it,
        # ID of the type whose body it is) of each open block that local
        # variables are scoped to
        self._scopes = []
        # (last line, first line, object) of each open `class << object`
        singleton_classes = []
        singletons = set()  # IDs of the singleton methods
        held = []  # (edge index, type ID, variable, method) per call on an ivar
        readers = []  # (node, type ID, variable) per reader of an ivar
        previous = ""  # the line of code before the one being read

        def owner(named=False):
            # A method belongs to the innermost class, module or block that
            # defines methods; a constant to the innermost of those with a
            # name, whatever block it is written in.
            return next(
                c[0]
                for c in reversed(self._contexts)
                if c[0] not in methods
                and not (named and c[0].rpartition("::")[2].startswith("<"))
            )

        for index, code in enumerate(code_lines):
            if not code.strip():
                continue
            lineno = index + 1
            continues = previous.endswith(
                (",", "(", "[", "{", "\\", ".", "|", "&", "+", "-", "*", "=")
            )
            self.context_at(lineno)
            while self._scopes and lineno > self._scopes[-1][0]:
                for name, start in self._scopes.pop()[1].items():
                    del self.declared[name][start:]
            while singleton_classes and lineno > singleton_classes[-1][0]:
                singleton_classes.pop()
            outer_id = owner(named=True)
            scan_start = 0
            # The scopes open before the line, and the column of each
            # block that opens one on it
            outer_scopes, opened_at = len(self._scopes), []
            # (column, context around it) of each context the line opens
            entered = []

            # 1. Classes, modules, methods and the blocks that are bodies
            for column, word, last, level, defines in opened[index]:
                end_lineno = last + 1
                owner_id = owner()
                before = code[:column]
                context_id = None
                if word in ("class", "module"):
                    header = (_RE_CLASS if word == "class" else _RE_MODULE).match(
                        code, column
                    )
                    opened_at.append(column)
                    if not header:
                        self._scopes.append((end_lineno, {}, lineno, owner_id))
                        of = _RE_SINGLETON_CLASS.match(code, column)
                        if of:
                            singleton_classes.append((end_lineno, lineno, of[1]))
                        continue
                    context_id = self._add(
                        owner(named=True),
                        header[1].removeprefix("::"),
                        "Class",
                        lineno,
                        end_lineno,
                        kind=word,
                        docstring=self.get_leading_docstring(lines, index),
                    )
                    self._scopes.append((end_lineno, {}, lineno, context_id))
                    scan_start = header.end(1)
                    if word == "class" and header["base"]:
                        base = header["base"].removeprefix("::")
                        self.builder.add_edge(context_id, f"REF:{base}", "INHERITS")
                        made = _RE_TYPE_MAKER.match(code, header.start("base"))
                        if made:
                            self._add_made(context_id, made, lineno)
                elif word == "def":
                    header = _RE_METHOD.match(code, column)
                    if not header:
                        continue
                    name = header["name"]
                    receiver = header["owner"]
                    if singleton_classes and not receiver:
                        # In `class << object` a method is that object's,
                        # unless a definition inside it encloses the method.
                        _, first, of = singleton_classes[-1]
                        receiver = of if first >= self._contexts[-1][1] else None
                    method_id = f"{owner_id}::{name}"
                    extra = {}
                    if receiver in ("self", owner_id.rpartition("::")[2]):
                        extra["is_singleton"] = True
                    if receiver and not extra:
                        # A method of one object is named with that object.
                        owner_id = self._contexts[-1][0]
                        name = f"{receiver}.{name}"
                    elif method_id in self.builder.node_registry and (
                        method_id in singletons
                    ) != bool(extra):
                        # A class may have a singleton method and an
                        # instance method of one name: the first is then
                        # `self.name`.
                        if extra:
                            name = f"self.{name}"
                        else:
                            self.rename_node(method_id, f"{owner_id}::self.{name}")
                            singletons.discard(method_id)
                    elif extra:
                        singletons.add(method_id)
                    doc = self.get_leading_docstring(lines, index)
                    typed = {
                        found["name"] or found["name2"]: found["type"] or found["type2"]
                        for found in _RE_YARD_PARAM.finditer(doc)
                    }
                    returns = _RE_YARD_RETURN.search(doc)
                    if index and _RE_SIG.match(code_lines[index - 1]):
                        typed.update(_RE_SIG_PARAM.findall(code_lines[index - 1]))
                        returns = _RE_SIG_RETURN.search(code_lines[index - 1])
                    if returns:
                        extra["return_type"] = returns[1]
                    scan_start = header.end()
                    params = _continued(code_lines, index, code[scan_start:])
                    listed = re.match(
                        r"\s*\(((?:[^()]|\((?:[^()]|\([^()]*\))*\))*)\)", params
                    )
                    if listed:
                        params = listed[1]
                    elif _RE_ENDLESS.match(code, scan_start):
                        params = ""
                    else:
                        params = params.partition(";")[0]
                    known = len(self.builder.nodes)
                    context_id = self._add(
                        owner_id,
                        name,
                        "Method",
                        lineno,
                        end_lineno,
                        docstring=doc,
                        arity=_arity(params),
                        **extra,
                    )
                    methods[context_id] = header["name"]
                    if len(self.builder.nodes) > known:
                        bodies[context_id] = {
                            "node": self.builder.nodes[-1]["metadata"],
                            "level": level,
                            "exits": [],
                            "last": "nil",
                        }
                    self._scopes.append((end_lineno, {}, lineno, None))
                    opened_at.append(column)
                    for param in _parameters(params):
                        origin = "?"
                        if param in typed:
                            origin = f"{typed[param].removeprefix('::')}()"
                        self._declare(param, lineno, lineno, origin)
                    inline = _RE_INLINE_BODY.match(code, scan_start)
                    if context_id in bodies and end_lineno == lineno and inline:
                        bodies[context_id]["last"] = self._exit(
                            inline["body"],
                            lines[index][inline.start("body") : inline.end("body")],
                            lineno,
                            lineno,
                            owner_id,
                        )
                else:
                    made = _RE_TYPE_MAKER.search(before)
                    defined = _RE_DEFINE_METHOD.search(before)
                    in_method = self._contexts[-1][0] in methods
                    if made and not before[made.end() :].strip():
                        constant = _RE_CONSTANT.match(before)
                        name = constant and constant[1]
                        owner_id = owner(named=True)
                        if in_method or not constant:
                            owner_id = self._contexts[-1][0]
                            name = f"<{made['maker']}:{lineno}>"
                        kind = made["maker"].lower()
                        context_id = self._add(
                            owner_id,
                            name,
                            "Class",
                            lineno,
                            end_lineno,
                            kind="class" if kind == "data" else kind,
                            docstring=self.get_leading_docstring(lines, index),
                        )
                        self._add_made(context_id, made, lineno)
                    elif (
                        defined
                        and not in_method
                        and not before[defined.end() :].strip(" )")
                    ):
                        context_id = self._add(
                            owner_id,
                            defined[1],
                            "Method",
                            lineno,
                            end_lineno,
                            docstring=self.get_leading_docstring(lines, index),
                        )
                        methods[context_id] = defined[1]
                    elif defines:
                        # A block that defines methods is a scope of its
                        # own, named after the method it is passed to,
                        # unless that runs it in `self`. Run in a class a
                        # constant names, it adds to that class.
                        call = _RE_BLOCK_CALL.search(before)
                        dotted = call and before[: call.start()].rstrip().endswith(".")
                        receiver = dotted and _RE_RECEIVER.search(
                            before[: call.start()].rstrip()[:-1].rstrip()
                        )
                        receiver = receiver.group() if receiver else dotted and "?"
                        call = call or _RE_COMMAND.match(before)
                        method = call["method"] if call else "block"
                        evaluated = method in _EVALUATORS
                        if not (evaluated and receiver in (None, False, "self")):
                            reopens = {}
                            if evaluated and re.fullmatch(_CONST, receiver):
                                reopens["reopens"] = receiver.removeprefix("::")
                            context_id = self._add(
                                self._contexts[-1][0],
                                f"<{method}:{lineno}>",
                                "Class",
                                lineno,
                                end_lineno,
                                kind="block",
                                docstring="",
                                **reopens,
                            )
                    # Only a `class`, `module` or `def` hides the local
                    # variables around it.
                    scope_of = None if context_id in methods else context_id
                    self._scopes.append((end_lineno, {}, 0, scope_of))
                    opened_at.append(column)
                    since = next((s[2] for s in reversed(self._scopes) if s[2]), 1)
                    params = _RE_BLOCK_PARAMS.match(code, column + len(word))
                    lambda_params = _RE_LAMBDA_PARAMS.search(before)
                    params = (params[1] if params else "") + ";"
                    params += lambda_params[1] if lambda_params else ""
                    for param in _parameters(params):
                        self._declare(param, lineno, since, "?")
                if context_id:
                    entered.append((column, self._contexts[-1][0]))
                    self.enter_context(context_id, lineno, end_lineno)

            context_id = self._contexts[-1][0]
            owner_id = owner()
            # The type `self` is an instance of: a block is run in the
            # `self` of what is around it.
            self_id = next(
                c[0]
                for c in reversed(self._contexts)
                if c[0] not in methods and self._kind(c[0]) != "block"
            )
            since = next((s[2] for s in reversed(self._scopes) if s[2]), 1)

            # 2. What a class or module body declares, outside its methods
            if context_id == owner_id:
                # attr_reader / writer / accessor — an accessor per symbol
                m_attr = _RE_ATTR.search(code)
                if m_attr:
                    symbols = _continued(code_lines, index, code[m_attr.end() :])
                    known = len(self.builder.nodes)
                    self._add_accessors(
                        owner_id,
                        m_attr[1] or "reader",
                        re.findall(r":(\w+[?!]?)", symbols),
                        lineno,
                    )
                    for node in self.builder.nodes[known:]:
                        name = node["metadata"]["name"]
                        if not name.endswith("="):
                            readers.append((node, self_id, f"@{name}"))
                m_alias = _RE_ALIAS.search(code)
                m_defined = m_alias or _RE_DEFINE_METHOD.search(code)
                if m_defined:
                    self._add(
                        owner_id, m_defined[1], "Method", lineno, lineno, docstring=""
                    )
                if m_alias:
                    scan_start = len(code)
                m_const = _RE_CONSTANT.match(code)
                if m_const:
                    value = _continued(code_lines, index, m_const["value"])
                    made = _RE_TYPE_MAKER.match(value.lstrip())
                    built = _RE_CONSTRUCTED.fullmatch(value)
                    if made:
                        kind = made["maker"].lower()
                        type_id = f"{outer_id}::{m_const[1]}"
                        if type_id not in self.builder.node_registry:
                            self._add(
                                outer_id,
                                m_const[1],
                                "Class",
                                lineno,
                                lineno,
                                kind="class" if kind == "data" else kind,
                                docstring=self.get_leading_docstring(lines, index),
                            )
                            self._add_made(type_id, made, lineno)
                    else:
                        held_by = {}
                        if built:
                            held_by["value_type"] = built["type"].removeprefix("::")
                        elif re.fullmatch(rf"\s*{_CONST}\s*", value):
                            # Another name for what a constant names.
                            held_by["value"] = value.strip().removeprefix("::")
                        self._add(
                            outer_id,
                            m_const[1],
                            "Constant",
                            lineno,
                            lineno,
                            docstring="",
                            **held_by,
                        )
                m_mixin = _RE_MIXIN.match(code)
                # A module is mixed into the class whose body the line is
                # in, not one that a block around the line stands for.
                if m_mixin and owner_id != file_id and self._scopes[-1][3] == owner_id:
                    for base in re.findall(_CONST, m_mixin[1]):
                        self.builder.add_edge(
                            owner_id, f"REF:{base.removeprefix('::')}", "INHERITS"
                        )

            # 3. What each instance variable of the type is assigned
            multiple = _RE_MULTIPLE.match(code)
            targets = multiple[1] if multiple else ""
            for variable in re.findall(r"(?<![\w@])@\w+", targets):
                self.fields.setdefault((self_id, variable), set()).add("?")
            for found in _RE_IVAR_ASSIGNED.finditer(code):
                held_as = None
                if found["plain"]:
                    value = _continued(code_lines, index, found["value"])
                    if value.strip() == "nil":
                        continue
                    held_as = _constructed(value)
                self.fields.setdefault((self_id, found[1]), set()).add(held_as or "?")

            # 4. Calls, and the local variables the line binds
            if skip_calls:
                continue
            # Each local variable the line assigns -> the column it is at
            assigned = {m[1]: m.start(1) for m in _RE_ASSIGNED.finditer(code)}
            for name in re.findall(r"(?<![\w@.])[a-z_]\w*(?![\w.(\[])", targets):
                assigned.setdefault(name, 0)
            for found in _RE_BOUND.finditer(code):
                for name in re.findall(r"\w+", found[1] or found[2]):
                    assigned.setdefault(name, found.start())
            # `a&.b` calls `b` as `a.b` does.
            code = code.replace("&.", " .")
            self.begin_line(code, lineno, raw=lines[index])
            # A line that goes on from a `.` starts with the method called.
            chained = previous.endswith(".") and not previous.endswith("..")
            previous = code.rstrip()
            for found in _RE_NAME.finditer(code, scan_start):
                name, start, end = found.group(), found.start(), found.end()
                before = code[:start].rstrip()
                paren = end if code[end : end + 1] == "(" else -1
                callee, origin = name, None
                if chained and not before:
                    origin = f"?.{name}"
                    if self._tail is not None:
                        origin = f"#{self._tail}.{name}"
                elif before.endswith((".", "::")) and not before.endswith(".."):
                    head = before[: -1 if before.endswith(".") else -2].rstrip()
                    receiver = _RE_RECEIVER.search(head)
                    root = receiver.group() if receiver else ""
                    setter = re.match(r"\s*=(?![=~>])", code[end:])
                    if setter and not name.endswith(("?", "!")):
                        callee = name = f"{name}="
                    if root == "self" and name == "class":
                        continue
                    if root.startswith("self"):
                        # `self.class` is the class of the method's own type.
                        callee = f"{''.join(root.split())}.{name}"
                        if root != "self" or (
                            name == "new" and self._kind(owner_id) == "class"
                        ):
                            origin = f"cls.{name}"
                    elif root[:1].isupper() or root.startswith("::"):
                        # `Const.new` is a call of Const.
                        path = root.removeprefix("::")
                        origin = f"{path}.{name}"
                        callee = path if name == "new" else origin
                    elif root.startswith(("@", "$")) or (
                        root not in _RUBY_KEYWORDS
                        and (root in assigned or self._declared(root, lineno, since))
                    ):
                        callee = f"{root}.{name}"
                    if callee != name:
                        start = receiver.start()
                else:
                    label = code[end : end + 1] == ":" and code[end : end + 2] != "::"
                    if name in _RUBY_KEYWORDS or label:
                        continue
                    # A name read is a local variable where one is in scope.
                    local = name in assigned or self._declared(name, lineno, since)
                    if paren < 0 and local:
                        continue
                    if name == "super":
                        if methods.get(context_id) is None:
                            continue
                        origin = f"super().{methods[context_id]}"
                    elif name == "new" and self._kind(owner_id) == "class":
                        origin = "cls.new"
                # What precedes a definition on its line is no part of it.
                source_id = next((c for at, c in entered if start < at), context_id)
                self.add_call(source_id, callee, lineno, since, start, paren, origin)
                edge = len(self.builder.edges) - 1
                if callee.endswith("="):
                    written = [1, False]
                elif paren < 0:
                    # `super` alone passes on what the method was given.
                    written = name != "super" and _command_arguments(code[end:])
                    written = written not in (None, False) and _arguments(written)
                else:
                    closes = self._pairs.get(paren)
                    written = closes and _arguments(code[paren + 1 : closes])
                if written:
                    self.builder.edges[edge]["metadata"]["args"] = written
                if paren < 0 and not callee.endswith("="):
                    # A call without parentheses ends with its name.
                    self.continue_chain(end - 1, edge)
                    self.continue_chain(end - 1 - name.endswith("?"), edge)
                if re.match(r"@\w", callee):
                    held.append((edge, self_id, *callee.split(".")))

            body = bodies.get(context_id)
            first, last = self._contexts[-1][1:3]
            if body and first < lineno:
                raw = lines[index]
                for found in _RE_RETURN.finditer(code):
                    span = slice(found.start("value"), found.end("value"))
                    body["exits"].append(
                        self._exit(code[span], raw[span], lineno, since, self_id)
                    )
                word = code.split()[0]
                if levels[index] != body["level"] or lineno == last or "ends" in body:
                    pass
                elif word in ("rescue", "else"):
                    # The body of a `rescue` is another way out.
                    body["last"] = None
                elif word == "ensure":
                    body["ends"] = lineno
                elif word in ("return", "raise", "fail"):
                    body["last"] = "nil"
                elif (
                    continues
                    or code.lstrip()[0] in ".)]}"
                    or opened[index]
                    or levels[index + 1] != body["level"]
                ):
                    body["last"] = None
                else:
                    body["last"] = self._exit(code, raw, lineno, since, self_id)

            bound = _RE_ASSIGNMENT.match(code)
            if bound:
                value = code[bound.end() :]
                origin = _constructed(_continued(code_lines, index, value))
                edge = self._closes.get(len(code.rstrip()) - 1)
                chain = _RE_CHAIN.fullmatch(_strip_parens(value))
                if not origin and edge is not None and chain:
                    origin = f"#{edge}"
                self._declare(bound["name"], lineno, since, origin or "?", outer_scopes)
                del assigned[bound["name"]]
            for name, column in assigned.items():
                # A variable is one of the block its assignment is in.
                depth = outer_scopes + sum(at < column for at in opened_at)
                self._declare(name, lineno, since, "?", depth)

        # An instance variable has a type where every value it is assigned
        # in its class is an instance of that one, and its reader returns it.
        for edge, owner_id, variable, method in held:
            types = self.fields.get((owner_id, variable), ())
            if len(types) == 1 and "?" not in types:
                metadata = self.builder.edges[edge]["metadata"]
                metadata["origin"] = f"{next(iter(types))}.{method}"
        for node, owner_id, variable in readers:
            types = self.fields.get((owner_id, variable), ())
            made = next(iter(types)) if len(types) == 1 else ""
            if made.endswith(".new()"):
                node["metadata"]["return_type"] = made.removesuffix(".new()")
        # A method returns what its last statement and its `return`s all
        # give, `nil` aside.
        for body in bodies.values():
            exits = set()
            for given in (*body["exits"], body["last"]):
                if given and given[0] == "held":
                    types = self.fields.get(given[1:], ())
                    made = next(iter(types)) if len(types) == 1 else ""
                    typed = made.endswith(".new()")
                    given = ("call", made[:-6], made[:-2]) if typed else None
                exits.add(given)
            exits.discard("nil")
            if len(exits) != 1 or None in exits or "return_type" in body["node"]:
                continue
            kind, *given = exits.pop()
            if kind == "type":
                body["node"]["return_type"] = given[0]
            else:
                body["node"]["returned"] = given
