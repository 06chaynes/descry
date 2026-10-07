"""Comment stripping shared by the regex language parsers.

Only genuinely identical implementations belong here. Most parsers need
language-specific handling — PHP also honours ``#``, Ruby is ``#``-only, and
C/C++, C# and Dart each differ — so those keep their own versions rather than
being bent into a common one.
"""

from __future__ import annotations


def strip_double_slash_comment(line: str) -> str:
    """Return `line` with any trailing ``//...`` comment removed.

    Tracks simple string state so ``"http://"`` is preserved. Block comments
    (``/* ... */``) are not handled here — the parsers track those with an
    in-block flag across lines.
    """
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
        elif ch in ('"', "'", "`"):
            in_string = True
            quote_char = ch
        elif ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
            return line[:i]
        i += 1
    return line
