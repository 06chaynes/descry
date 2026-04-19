#!/usr/bin/env python3
"""
ast-grep based CALLS extraction for improved accuracy.

Uses tree-sitter via ast-grep to find function calls with higher precision
than regex-based detection. Handles:
- Method chaining: foo.bar().baz()
- Generic calls: Vec::<T>::new()
- Closures: |x| process(x)
- Macro invocations: vec![], println!()
"""

import json
import subprocess
from typing import Iterator

from descry._env import safe_env


def extract_calls_rust(file_path: str) -> Iterator[dict]:
    """Extract function calls from a Rust file using ast-grep.

    Yields dicts with:
        - callee: str - function/method name being called
        - lineno: int - line number
        - full_text: str - full call expression text

    Uses multiple patterns to capture:
    - Direct calls: function(args)
    - Method calls: receiver.method(args)
    - Qualified calls: Type::method(args)
    """
    # Patterns to match different call types
    patterns = [
        "$FUNC($$$ARGS)",  # Direct function calls
        "$RECEIVER.$METHOD($$$ARGS)",  # Method calls on expressions
        "$RECEIVER.$METHOD()",  # No-arg method calls
    ]

    seen = set()  # Deduplicate by (lineno, callee)

    for pattern in patterns:
        try:
            result = subprocess.run(
                # `--json` first so `--` terminates options cleanly and the
                # following `file_path` cannot be mis-parsed as a flag even
                # if a local source file literally begins with `-`.
                ["sg", "run", "-p", pattern, "-l", "rust", "--json", "--", file_path],
                capture_output=True,
                text=True,
                timeout=30,
                env=safe_env(),
            )

            if result.returncode != 0:
                continue

            matches = json.loads(result.stdout) if result.stdout else []

            for match in matches:
                meta = match.get("metaVariables", {}).get("single", {})

                # Try METHOD first (for method calls), then FUNC (for direct calls)
                func_meta = meta.get("METHOD") or meta.get("FUNC", {})
                func_name = func_meta.get("text", "") if func_meta else ""

                if not func_name:
                    continue

                # Skip malformed matches - ast-grep can capture entire method chains
                # as the "function" when matching builder patterns like Router::new().merge().layer()
                if "\n" in func_name or len(func_name) > 100:
                    continue

                # Skip control flow keywords that look like calls
                if func_name in ("if", "for", "while", "match", "loop", "return"):
                    continue

                # Skip common macros (they end with !)
                if func_name.endswith("!"):
                    continue

                # ast-grep uses 0-based line numbers, convert to 1-based
                lineno = match.get("range", {}).get("start", {}).get("line", 0) + 1

                # Deduplicate
                key = (lineno, func_name)
                if key in seen:
                    continue
                seen.add(key)

                # For method calls, include receiver context if it's a type
                receiver_meta = meta.get("RECEIVER", {})
                receiver_text = receiver_meta.get("text", "") if receiver_meta else ""

                # If receiver looks like a type (Self, starts with uppercase), qualify the call
                if receiver_text in ("self", "Self") or (
                    receiver_text
                    and receiver_text[0].isupper()
                    and "::" not in func_name
                ):
                    qualified_name = f"{receiver_text}.{func_name}"
                else:
                    qualified_name = func_name

                yield {
                    "callee": qualified_name,
                    "lineno": lineno,
                    "full_text": match.get("text", ""),
                }

        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            continue


def extract_calls_typescript(file_path: str) -> Iterator[dict]:
    """Extract function calls from a TypeScript file using ast-grep.

    Yields dicts with:
        - callee: str - function/method name being called
        - lineno: int - line number
        - full_text: str - full call expression text

    Uses multiple patterns to capture:
    - Direct calls: function(args)
    - Method calls: receiver.method(args)
    - Chained calls: obj.foo().bar()
    """
    # Patterns to match different call types
    patterns = [
        "$FUNC($$$ARGS)",  # Direct function calls
        "$RECEIVER.$METHOD($$$ARGS)",  # Method calls on expressions
        "$RECEIVER.$METHOD()",  # No-arg method calls
    ]

    seen = set()  # Deduplicate by (lineno, callee)

    for pattern in patterns:
        try:
            result = subprocess.run(
                # See note in Rust branch re: `--` separator for safety.
                [
                    "sg",
                    "run",
                    "-p",
                    pattern,
                    "-l",
                    "typescript",
                    "--json",
                    "--",
                    file_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env=safe_env(),
            )

            if result.returncode != 0:
                continue

            matches = json.loads(result.stdout) if result.stdout else []

            for match in matches:
                meta = match.get("metaVariables", {}).get("single", {})

                # Try METHOD first (for method calls), then FUNC (for direct calls)
                func_meta = meta.get("METHOD") or meta.get("FUNC", {})
                func_name = func_meta.get("text", "") if func_meta else ""

                if not func_name:
                    continue

                # Skip control flow and common JS patterns
                if func_name in (
                    "if",
                    "for",
                    "while",
                    "switch",
                    "catch",
                    "function",
                    "class",
                    "return",
                    "await",
                    "new",
                ):
                    continue

                # $FUNC can match a whole call expression when the outer
                # syntax is `foo(args)(args)` (curried / conditional-test
                # patterns like `test.runIf(isBuild)(...)` common in
                # Vitest/Jest). Reject anything that's clearly not a
                # plain identifier chain — no parens, braces, newlines,
                # template literals.
                if any(ch in func_name for ch in "(){}[]\n`\"'<>!=&|+*/%?:,;"):
                    continue

                # ast-grep uses 0-based line numbers, convert to 1-based
                lineno = match.get("range", {}).get("start", {}).get("line", 0) + 1

                # Deduplicate
                key = (lineno, func_name)
                if key in seen:
                    continue
                seen.add(key)

                yield {
                    "callee": func_name,
                    "lineno": lineno,
                    "full_text": match.get("text", ""),
                }

        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            continue


def is_ast_grep_available() -> bool:
    """Check if ast-grep (sg) is available on the system."""
    try:
        result = subprocess.run(
            ["sg", "--version"], capture_output=True, timeout=5, env=safe_env()
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Cache the availability check
_AST_GREP_AVAILABLE = None


def ast_grep_available() -> bool:
    """Cached check for ast-grep availability."""
    global _AST_GREP_AVAILABLE
    if _AST_GREP_AVAILABLE is None:
        _AST_GREP_AVAILABLE = is_ast_grep_available()
    return _AST_GREP_AVAILABLE


def extract_imports_typescript(file_path: str) -> dict:
    """Extract import declarations from a TypeScript file using ast-grep.

    Returns a dict with:
        - imports: dict mapping local names to (module_path, import_type)
        - namespaces: dict mapping namespace alias to module_path

    Import types:
        - "named": import { foo } from 'module'
        - "default": import foo from 'module'
        - "namespace": import * as foo from 'module'
        - "type": import type { Foo } from 'module'
    """
    result = {
        "imports": {},  # local_name -> (module_path, import_type)
        "namespaces": {},  # alias -> module_path
    }

    # Pattern 1: Named imports - import { foo, bar as baz } from 'module'
    patterns = [
        # Named imports
        ("import { $$$IMPORTS } from $SOURCE", "named"),
        # Default import
        ("import $DEFAULT from $SOURCE", "default"),
        # Namespace import
        ("import * as $ALIAS from $SOURCE", "namespace"),
        # Type-only imports
        ("import type { $$$TYPES } from $SOURCE", "type"),
        # Default with named: import Default, { named } from 'module'
        ("import $DEFAULT, { $$$NAMED } from $SOURCE", "default_named"),
    ]

    for pattern, import_type in patterns:
        try:
            proc_result = subprocess.run(
                # See note in Rust branch re: `--` separator for safety.
                [
                    "sg",
                    "run",
                    "-p",
                    pattern,
                    "-l",
                    "typescript",
                    "--json",
                    "--",
                    file_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env=safe_env(),
            )

            if proc_result.returncode != 0:
                continue

            matches = json.loads(proc_result.stdout) if proc_result.stdout else []

            for match in matches:
                meta = match.get("metaVariables", {})
                single = meta.get("single", {})
                multi = meta.get("multi", {})

                # Extract source module
                source_meta = single.get("SOURCE", {})
                source = source_meta.get("text", "").strip("'\"")
                if not source:
                    continue

                if import_type == "namespace":
                    # import * as alias from 'module'
                    alias_meta = single.get("ALIAS", {})
                    alias = alias_meta.get("text", "")
                    if alias:
                        result["namespaces"][alias] = source
                        result["imports"][alias] = (source, "namespace")

                elif import_type == "default":
                    # import Default from 'module'
                    default_meta = single.get("DEFAULT", {})
                    default_name = default_meta.get("text", "")
                    if default_name:
                        result["imports"][default_name] = (source, "default")

                elif import_type in ("named", "type"):
                    # import { foo, bar as baz } from 'module'
                    imports_meta = (
                        multi.get("IMPORTS", [])
                        if import_type == "named"
                        else multi.get("TYPES", [])
                    )
                    for imp in imports_meta:
                        text = imp.get("text", "").strip()
                        if not text:
                            continue
                        # Handle aliased imports: "foo as bar"
                        if " as " in text:
                            parts = text.split(" as ")
                            alias = parts[1].strip()
                            result["imports"][alias] = (source, import_type)
                        else:
                            result["imports"][text] = (source, import_type)

                elif import_type == "default_named":
                    # import Default, { named } from 'module'
                    default_meta = single.get("DEFAULT", {})
                    default_name = default_meta.get("text", "")
                    if default_name:
                        result["imports"][default_name] = (source, "default")

                    named_meta = multi.get("NAMED", [])
                    for imp in named_meta:
                        text = imp.get("text", "").strip()
                        if not text:
                            continue
                        if " as " in text:
                            parts = text.split(" as ")
                            alias = parts[1].strip()
                            result["imports"][alias] = (source, "named")
                        else:
                            result["imports"][text] = (source, "named")

        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            continue

    return result
