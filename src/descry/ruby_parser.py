"""Ruby regex-based baseline parser.

Covers the common shapes of Ruby symbol extraction: modules, classes
(with optional superclass), methods (``def name`` and ``def self.name``),
accessors (``attr_reader``/``attr_writer``/``attr_accessor``), requires,
constants, and call sites that use explicit parentheses.

Ruby uses ``end`` to close blocks rather than ``}``, so this parser
tracks context via indentation rather than brace depth — a class/module/
def at indent ``I`` pops any sibling context at indent ``>= I``. This
matches how Rails and Sorbet-typed code are conventionally formatted.
Paren-less calls (``puts "x"``, ``attr_reader :foo``) are intentionally
out of scope for the regex extractor; scip-ruby fills them in when
available.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from descry.generate import BaseParser, is_non_project_call

logger = logging.getLogger(__name__)


# class Foo, class Foo < Bar
_RE_CLASS = re.compile(
    r"^(\s*)class\s+([A-Z][A-Za-z0-9_:]*)\s*(?:<\s*([A-Z][A-Za-z0-9_:]*))?"
)
# module Foo
_RE_MODULE = re.compile(r"^(\s*)module\s+([A-Z][A-Za-z0-9_:]*)")
# def foo, def self.foo, def Klass.foo
_RE_METHOD = re.compile(
    r"^(\s*)def\s+(?:self\.|[A-Z][A-Za-z0-9_]*\.)?([a-z_][A-Za-z0-9_]*[?!=]?)"
)
# require 'x', require "x", require_relative 'x'
_RE_REQUIRE = re.compile(
    r"^\s*(?:require|require_relative|load|autoload)\s+[\"']([^\"']+)[\"']"
)
# CONSTANT_NAME = value
_RE_CONSTANT = re.compile(r"^(\s*)([A-Z][A-Z0-9_]*)\s*=\s*[^=]")
# attr_reader :foo, :bar / attr_writer :x / attr_accessor :y
_RE_ATTR = re.compile(r"^(\s*)attr_(?:reader|writer|accessor)\s+([:a-zA-Z_,\s]+)")
# Call site: identifier( or receiver.identifier(
# Receiver can be a Capitalized constant (Class.method) or a lowercase
# variable/method (obj.method, @ivar.method, self.method). Method names
# may end in ``?`` or ``!`` (Ruby idiom). The ``%`` in the negative
# lookbehind prevents matching %w(...), %x(...), %i(...) literals.
_RE_CALL = re.compile(
    r"(?<![A-Za-z0-9_@.:%])"
    r"((?:[A-Za-z_@][A-Za-z0-9_:]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\.)?"
    r"[a-z_][A-Za-z0-9_]*[?!]?)"
    r"\s*\("
)
# Line-starts that don't count (already handled separately)
_RUBY_KEYWORDS = frozenset(
    {
        "if",
        "unless",
        "while",
        "until",
        "for",
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
        "defined",  # `defined?` strips the `?` before this check
        "alias",
        "undef",
        "super",
        "self",
        "true",
        "false",
        "nil",
        "new",  # Class.new — too generic, SCIP handles
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
        # Array/string literal openers — the call regex can match `%w(`,
        # `%x(` etc.; the negative lookbehind excludes most but be safe.
        "w",
        "x",
        "i",
        "r",
        "q",
        "Q",
    }
)


def _strip_line_comment(line: str) -> str:
    """Remove ``#...`` trailing comments, honoring string literals."""
    in_single = False
    in_double = False
    escape = False
    i = 0
    while i < len(line):
        c = line[i]
        if escape:
            escape = False
        elif c == "\\" and (in_single or in_double):
            escape = True
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == "'" and not in_double:
            in_single = not in_single
        elif not in_single and not in_double and c == "#":
            return line[:i]
        i += 1
    return line


class RubyParser(BaseParser):
    """Regex + indentation Ruby parser."""

    def parse(self, file_path, rel_path, content):
        file_id = f"FILE:{rel_path}"
        self.builder.add_node(
            file_id,
            "File",
            path=rel_path,
            name=Path(rel_path).name,
            token_count=len(content) // 4,
        )

        lines = content.splitlines()

        # Requires — one pass over the file
        for line in lines:
            stripped = _strip_line_comment(line)
            m = _RE_REQUIRE.match(stripped)
            if m:
                self.builder.add_edge(file_id, f"MODULE:{m.group(1)}", "IMPORTS")

        # Contexts are tracked by indent depth. A new class/module/def at
        # indent ``I`` pops any context whose indent is >= I (sibling or
        # outer that's closing).
        current_context: list[str] = [file_id]
        context_indent: list[int] = [-1]

        def _pop_to(indent: int) -> None:
            while len(current_context) > 1 and context_indent[-1] >= indent:
                current_context.pop()
                context_indent.pop()

        in_block_comment = False
        i = 0
        n = len(lines)
        while i < n:
            raw_line = lines[i]
            lineno = i + 1

            # Block comments (=begin ... =end at column 0) — suppress entirely.
            if in_block_comment:
                if raw_line.startswith("=end"):
                    in_block_comment = False
                i += 1
                continue
            if raw_line.startswith("=begin"):
                in_block_comment = True
                i += 1
                continue

            line = _strip_line_comment(raw_line)
            if not line.strip():
                i += 1
                continue

            indent = len(line) - len(line.lstrip())

            # 1. Class
            m_class = _RE_CLASS.match(line)
            if m_class:
                _pop_to(indent)
                _, name, superclass = m_class.groups()
                parent_id = current_context[-1]
                class_id = f"{parent_id}::{name}"
                docstring = self.get_leading_docstring(lines, i)
                self.builder.add_node(
                    class_id,
                    "Class",
                    name=name,
                    kind="class",
                    lineno=lineno,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, class_id, "DEFINES")
                if superclass:
                    self.builder.add_edge(class_id, f"REF:{superclass}", "INHERITS")
                # Single-line `class X; end` — don't push.
                if "end" not in line.split(";", 1)[1:] if ";" in line else True:
                    current_context.append(class_id)
                    context_indent.append(indent)
                i += 1
                continue

            # 2. Module
            m_module = _RE_MODULE.match(line)
            if m_module:
                _pop_to(indent)
                _, name = m_module.groups()
                parent_id = current_context[-1]
                mod_id = f"{parent_id}::{name}"
                docstring = self.get_leading_docstring(lines, i)
                self.builder.add_node(
                    mod_id,
                    "Class",
                    name=name,
                    kind="module",
                    lineno=lineno,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, mod_id, "DEFINES")
                current_context.append(mod_id)
                context_indent.append(indent)
                i += 1
                continue

            # 3. Method
            m_method = _RE_METHOD.match(line)
            if m_method:
                _pop_to(indent)
                _, name = m_method.groups()
                parent_id = current_context[-1]
                method_id = f"{parent_id}::{name}"
                docstring = self.get_leading_docstring(lines, i)
                self.builder.add_node(
                    method_id,
                    "Method",
                    name=name,
                    lineno=lineno,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, method_id, "DEFINES")
                # Methods don't nest in Ruby, but we push so calls inside
                # the body attribute to the method rather than the class.
                current_context.append(method_id)
                context_indent.append(indent)
                i += 1
                continue

            # 4. attr_reader / writer / accessor — emit an accessor per symbol
            m_attr = _RE_ATTR.match(line)
            if m_attr:
                parent_id = current_context[-1]
                for sym in re.findall(r":([a-z_][A-Za-z0-9_]*)", m_attr.group(2)):
                    attr_id = f"{parent_id}::{sym}"
                    existing = {n["id"] for n in self.builder.nodes}
                    if attr_id not in existing:
                        self.builder.add_node(
                            attr_id,
                            "Method",
                            name=sym,
                            lineno=lineno,
                            docstring="",
                            is_accessor=True,
                        )
                        self.builder.add_edge(parent_id, attr_id, "DEFINES")
                i += 1
                continue

            # 5. Top-level constants
            m_const = _RE_CONSTANT.match(line)
            if m_const:
                _, name = m_const.groups()
                parent_id = current_context[-1]
                const_id = f"{parent_id}::{name}"
                existing = {n["id"] for n in self.builder.nodes}
                if const_id not in existing:
                    self.builder.add_node(
                        const_id,
                        "Constant",
                        name=name,
                        lineno=lineno,
                        docstring="",
                    )
                    self.builder.add_edge(parent_id, const_id, "DEFINES")

            # 6. Calls inside a method/class/module context
            if current_context[-1] != file_id:
                parent_id = current_context[-1]
                for call_match in _RE_CALL.finditer(line):
                    callee = call_match.group(1)
                    last = callee.split(".")[-1]
                    # Check both the bare name and the ?/! stripped form.
                    if last in _RUBY_KEYWORDS or last.rstrip("?!") in _RUBY_KEYWORDS:
                        continue
                    if is_non_project_call(callee):
                        continue
                    self.builder.edges.append(
                        {
                            "source": parent_id,
                            "target": f"REF:{callee}",
                            "relation": "CALLS",
                            "metadata": {"lineno": lineno},
                        }
                    )

            i += 1
