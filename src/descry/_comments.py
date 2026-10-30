"""Comment stripping shared by the regex language parsers.

C# strips its own line comments: its verbatim strings take no backslash
escapes.
"""

from __future__ import annotations


def strip_line_comment(line: str, *, backticks: bool = False) -> str:
    """Return `line` without its trailing comment, which starts at the first
    ``//`` outside a string.

    Tracks simple string state so ``"http://"`` is preserved; a backquote
    opens a string only with `backticks`. Block comments (``/* ... */``) are
    left to `strip_block_comments`.
    """
    quotes = "\"'`" if backticks else "\"'"
    in_string = False
    quote_char = ""
    i = 0
    while i < len(line):
        ch = line[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == quote_char:
                in_string = False
        elif ch in quotes:
            in_string = True
            quote_char = ch
        elif line.startswith("//", i):
            return line[:i]
        i += 1
    return line


def strip_block_comments(line: str, in_block: bool) -> tuple[str, bool]:
    """Return `line` with ``/* ... */`` spans removed, and whether a block
    comment is still open at the end of it.

    `in_block` is that flag as returned for the previous line. Tracks simple
    string state so ``"/*"`` is preserved, and leaves a trailing ``//...``
    comment untouched.
    """
    out = []
    quote_char = ""
    i = 0
    while i < len(line):
        if in_block:
            end = line.find("*/", i)
            if end == -1:
                break
            in_block = False
            out.append(" ")
            i = end + 2
            continue
        ch = line[i]
        if quote_char:
            if ch == "\\":
                out.append(line[i : i + 2])
                i += 2
                continue
            if ch == quote_char:
                quote_char = ""
        elif ch in ('"', "'", "`"):
            quote_char = ch
        elif ch == "/" and line[i + 1 : i + 2] == "*" and line[i - 1 : i] != "\\":
            in_block = True
            i += 2
            continue
        elif ch == "/" and line[i + 1 : i + 2] == "/":
            out.append(line[i:])
            break
        out.append(ch)
        i += 1
    return "".join(out), in_block
