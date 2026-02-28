"""Generate codebase knowledge graph with optional SCIP-based type-aware resolution."""

import ast
import logging
import os
import json
import re
from pathlib import Path

# Configure logging for consistent error reporting
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Try to import ast-grep integration for improved CALLS detection
try:
    from descry.ast_grep import (
        ast_grep_available,
        extract_calls_rust,
        extract_calls_typescript,
        extract_imports_typescript,
    )

    USE_AST_GREP = ast_grep_available()
except ImportError:
    USE_AST_GREP = False
    extract_imports_typescript = None
    logger.debug("ast-grep not available, using regex fallback for CALLS detection")

# Try to import SCIP support for type-aware call resolution
try:
    from descry.scip.support import scip_available, get_scip_status
    from descry.scip.cache import ScipCacheManager
    from descry.scip.parser import ScipIndex

    SCIP_SUPPORT_LOADED = True
except ImportError:
    SCIP_SUPPORT_LOADED = False

    def scip_available():
        return False

    logger.debug("SCIP support not available")

# ============================================================================
# Non-project call filters
# ============================================================================
# Calls to these names/prefixes will never resolve to project nodes and would
# pollute caller/callee queries. They are split into two categories:
#
#   1. STDLIB  — language built-ins and standard library (Rust std, JS/TS
#                builtins, Python builtins)
#   2. LIBRARY — third-party crates, npm packages, and framework methods that
#                are dependencies of the project but not part of its source
#
# Both sets feed into `is_non_project_call()` which is the single entry point
# used by parsers and the resolver.
# ============================================================================

# --- 1. Standard library / language built-in names ---
_STDLIB_NAMES = frozenset(
    [
        # ── Rust primitives & wrapper types ──
        "Some",
        "None",
        "Ok",
        "Err",
        "Box",
        "Rc",
        "Arc",
        "Cell",
        "RefCell",
        "Vec",
        "String",
        "HashMap",
        "HashSet",
        "BTreeMap",
        "BTreeSet",
        "Option",
        "Result",
        "Default",
        "Clone",
        "Debug",
        "Display",
        "Duration",
        "Instant",
        "SystemTime",
        # ── Rust common methods & macros ──
        "new",
        "default",
        "clone",
        "into",
        "from",
        "as_ref",
        "as_mut",
        "unwrap",
        "expect",
        "unwrap_or",
        "unwrap_or_else",
        "unwrap_or_default",
        "unwrap_err",
        "ok",
        "err",
        "is_ok",
        "is_err",
        "is_some",
        "is_none",
        "is_some_and",
        "map",
        "map_err",
        "and_then",
        "or_else",
        "filter",
        "flatten",
        "collect",
        "iter",
        "into_iter",
        "iter_mut",
        "push",
        "pop",
        "insert",
        "remove",
        "get",
        "set",
        "len",
        "is_empty",
        "contains",
        "extend",
        "clear",
        "to_string",
        "to_owned",
        "as_str",
        "as_bytes",
        "format",
        "println",
        "print",
        "eprintln",
        "eprint",
        "dbg",
        "vec",
        "panic",
        "assert",
        "assert_eq",
        "assert_ne",
        "debug_assert",
        # ── Rust iterator / option / result methods ──
        "eq",
        "ne",
        "cmp",
        "partial_cmp",
        "first",
        "last",
        "next",
        "take",
        "skip",
        "zip",
        "enumerate",
        "peekable",
        "chain",
        "rev",
        "cloned",
        "copied",
        "ok_or",
        "ok_or_else",
        "map_or",
        "map_or_else",
        "as_deref",
        "as_deref_mut",
        "transpose",
        "into_inner",
        "inner",
        "inner_mut",
        "get_mut",
        "get_ref",
        "borrow",
        "borrow_mut",
        "deref",
        "deref_mut",
        "filter_map",
        "for_each",
        "flat_map",
        "fold",
        "find_map",
        "position",
        "any",
        "all",
        "min",
        "max",
        "nth",
        "sorted",
        "copy",
        "drop",
        # ── Rust std additional methods ──
        "push_str",
        "try_into",
        "try_from",
        "into_bytes",
        "floor",
        "ceil",
        "clamp",
        "retain",
        "write_all",
        "sync_all",
        "with_capacity",
        "to_str",
        "to_string_lossy",
        "fetch_add",
        "as_millis",
        "as_secs",
        "changed",
        "sort",
        "sort_by",
        "sort_by_key",
        "send",
        "recv",
        "try_recv",
        "try_send",
        "lock",
        "try_lock",
        "display",
        "flush",
        "read_to_string",
        "write_fmt",
        "as_path",
        "to_path_buf",
        "extension",
        "file_name",
        "parent",
        "exists",
        "is_file",
        "is_dir",
        "metadata",
        "canonicalize",
        "with_extension",
        "into_string",
        "entry",
        "or_insert",
        "or_insert_with",
        "or_default",
        "and_modify",
        "key",
        "value",
        "values_mut",
        "keys",
        "drain",
        "truncate",
        "resize",
        "capacity",
        "reserve",
        "shrink_to_fit",
        "swap",
        "swap_remove",
        "binary_search",
        "contains_key",
        "remove_entry",
        "split_at",
        "windows",
        "chunks",
        "as_slice",
        "as_mut_slice",
        "downcast_ref",
        "downcast_mut",
        "type_id",
        "extend_from_slice",
        "copy_from_slice",
        "saturating_sub",
        "is_ascii_hexdigit",
        "is_ascii",
        "as_ptr",
        "as_u16",
        "sleep",
        "abort",
        "stderr",
        "stdout",
        "args",
        # ── Rust async ──
        "await",
        "async",
        "spawn",
        "block_on",
        # ── Rust string methods ──
        "starts_with",
        "ends_with",
        "replace",
        "replace_all",
        "trim",
        "trim_start",
        "trim_end",
        "trim_end_matches",
        "trim_start_matches",
        "to_lowercase",
        "to_uppercase",
        "chars",
        "bytes",
        "lines",
        "match",
        "matches",
        "find",
        "rfind",
        "strip_prefix",
        "strip_suffix",
        "repeat",
        # ── Rust I/O & channels ──
        "read",
        "write",
        "close",
        "open",
        "fill_bytes",
        # ── JavaScript / TypeScript built-ins ──
        "console",
        "log",
        "error",
        "warn",
        "info",
        "debug",
        "setTimeout",
        "setInterval",
        "clearTimeout",
        "clearInterval",
        "Promise",
        "resolve",
        "reject",
        "then",
        "catch",
        "finally",
        "Array",
        "Object",
        "Map",
        "Set",
        "JSON",
        "Math",
        "Date",
        "RegExp",
        "parseInt",
        "parseFloat",
        "isNaN",
        "isFinite",
        "toString",
        "valueOf",
        "hasOwnProperty",
        "push",
        "pop",
        "shift",
        "unshift",
        "slice",
        "splice",
        "concat",
        "filter",
        "map",
        "reduce",
        "forEach",
        "find",
        "findIndex",
        "some",
        "every",
        "join",
        "split",
        "trim",
        "toLowerCase",
        "toUpperCase",
        "stringify",
        "parse",
        "includes",
        "indexOf",
        "lastIndexOf",
        "localeCompare",
        "normalize",
        "search",
        "at",
        "startsWith",
        "endsWith",
        "replaceAll",
        "trimStart",
        "trimEnd",
        "padStart",
        "padEnd",
        "substring",
        "substr",
        "charAt",
        "charCodeAt",
        "codePointAt",
        "pad_start",
        "pad_end",
        "strip",
        "lstrip",
        "rstrip",
        # ── JS DOM / Browser ──
        "querySelector",
        "querySelectorAll",
        "addEventListener",
        "removeEventListener",
        "scrollTo",
        "scrollIntoView",
        "focus",
        "blur",
        "preventDefault",
        "stopPropagation",
        "dispatchEvent",
        "getItem",
        "setItem",
        "removeItem",
        "toISOString",
        # ── Python built-ins ──
        "print",
        "len",
        "range",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "open",
        "append",
        "extend",
        "pop",
        "get",
        "keys",
        "values",
        "items",
        "isinstance",
        "issubclass",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "super",
        "type",
        "id",
        "hash",
        "repr",
        "format",
        "lower",
        "sorted",
    ]
)

# --- 2. Third-party library / framework names ---
# Methods and identifiers from external dependencies that will never resolve
# to project nodes.
_LIBRARY_NAMES = frozenset(
    [
        # ── Diesel ORM DSL ──
        "optional",
        "nullable",
        "eq_any",
        "ne_all",
        "load",
        "execute",
        "get_result",
        "get_results",
        "order",
        "order_by",
        "inner_join",
        "left_join",
        "group_by",
        "having",
        "select",
        "count",
        "sum",
        "limit",
        "offset",
        "is_not_null",
        "desc",
        "asc",
        "returning",
        "into_boxed",
        "or",
        "like",
        "not_like",
        "between",
        "is_null",
        "gt",
        "lt",
        "ge",
        "le",
        "on",
        "on_conflict",
        # ── Chrono ──
        "naive_utc",
        "timestamp",
        "timestamp_millis",
        "now",
        "elapsed",
        "duration_since",
        "to_rfc3339",
        "and_utc",
        "num_milliseconds",
        # ── anyhow / error handling ──
        "with_context",
        "context",
        # ── colored (terminal formatting) ──
        "red",
        "green",
        "yellow",
        "blue",
        "cyan",
        "magenta",
        "white",
        "bold",
        "dim",
        "underline",
        "italic",
        # ── tracing ──
        "instrument",
        "in_scope",
        "enter",
        "span",
        # ── indicatif (progress bars) ──
        "set_message",
        "finish_with_message",
        "inc",
        # ── X.509 / TLS / crypto libraries ──
        "parse_x509_certificate",
        "serialize_pem",
        "pem_parse",
        "pem",
        "validity",
        "add_attribute",
        "add_header",
        "finalize",
        # ── Redis ──
        "smembers",
        # ── Playwright / vitest / test frameworks ──
        "test",
        "describe",
        "it",
        "expect",
        "assert_status",
        "assert_status_ok",
        "toBeVisible",
        "toHaveText",
        "toBe",
        "toEqual",
        "toContain",
        "toHaveLength",
        "toBeNull",
        "toBeUndefined",
        "toBeTruthy",
        "toBeFalsy",
        "toBeInTheDocument",
        "toHaveURL",
        "toHaveClass",
        "toHaveAttribute",
        "toHaveCount",
        "toHaveValue",
        "toBeChecked",
        "toBeDisabled",
        "toBeEnabled",
        "toBeHidden",
        "toMatchObject",
        "toThrow",
        "toHaveBeenCalled",
        "click",
        "fill",
        "locator",
        "getByRole",
        "getByText",
        "getByTestId",
        "waitFor",
        "waitForTimeout",
        "waitForSelector",
        "waitForLoadState",
        "waitForURL",
        "selectOption",
        "goto",
        "beforeEach",
        "afterEach",
        "before",
        "after",
        "black_box",
        # ── Svelte / SvelteKit ──
        "json",
        "derived",
        "style",
        # ── clap (CLI argument parsing) ──
        "arg",
        # ── misc library methods ──
        "as_array",
        "headers",
        "contents",
        "extensions",
        "attributes",
        # ── TS/Python external module names (used as receiver patterns) ──
        "authenticatedPage",
        "container",
        "page",
        "pptx",
        "mcp",
        "re",
        "sys",
        "os",
        "pathlib",
        "subprocess",
        "shutil",
        "glob",
        "yaml",
        "toml",
    ]
)

# Combined filter used by is_non_project_call()
STDLIB_FILTER = _STDLIB_NAMES | _LIBRARY_NAMES

# --- Prefix-based filters ---
# Qualified name prefixes that indicate non-project calls.

_STDLIB_PREFIXES = (
    # ── Rust standard library ──
    "std::",
    "core::",
    "alloc::",
    "fs::",
    "io::",
    "env::",
    "net::",
    "path::",
    # ── Rust std types (qualified calls) ──
    "Duration::",
    "Instant::",
    "SystemTime::",
    "Vec::",
    "String::",
    "Arc::",
    "Rc::",
    "Box::",
    "Cell::",
    "RefCell::",
    "HashMap::",
    "HashSet::",
    "BTreeMap::",
    "BTreeSet::",
    "Option::",
    "Result::",
    "Path::",
    "PathBuf::",
    "Ok::<",
    "Err::<",
    "Cow::",
    "Mutex::",
    "RwLock::",
    "OsStr::",
    "OsString::",
    "NonZeroU",
    "PhantomData::",
    "Command::",
    "Stdio::",
    "Ordering::",
    "AtomicBool::",
    "AtomicUsize::",
    "Sender::",
    "Receiver::",
    "Channel::",
    "TcpListener::",
    "TcpStream::",
    "SocketAddr::",
    "UdpSocket::",
    "IpAddr::",
    "OpenOptions::",
    # NOTE: "Self::" was removed here — Self::method() calls are project calls.
    # Self::new/default/from/into are caught by STDLIB_FILTER via the last_part check.
    # The qualified-call shortcut at line ~877 has a "Self" exclusion to ensure
    # Self:: calls fall through to the last_part filter instead of early-exiting.
    # ── JavaScript / TypeScript built-in objects ──
    "console.",
    "Math.",
    "JSON.",
    "Object.",
    "Array.",
    "Promise.",
    "window.",
    "document.",
    "navigator.",
    "sessionStorage.",
    "localStorage.",
    # ── Python stdlib modules ──
    "os.",
    "sys.",
    "re.",
    "pathlib.",
    "subprocess.",
    "shutil.",
)

_LIBRARY_PREFIXES = (
    # ── Rust async runtimes ──
    "tokio::",
    "async_std::",
    # ── Serialization ──
    "serde::",
    "serde_json::",
    # ── Logging / tracing ──
    "tracing::",
    "log::",
    # ── Web frameworks ──
    "axum::",
    "axum_extra::",
    "tower::",
    "tower_http::",
    "hyper::",
    "http::",
    "reqwest::",
    # ── gRPC ──
    "tonic::",
    "prost::",
    # ── Database ──
    "diesel::",
    "schema::",
    "dsl::",
    # ── Error handling ──
    "anyhow::",
    "thiserror::",
    # ── Crypto / TLS ──
    "ring::",
    "rcgen::",
    "openssl::",
    "rustls::",
    "pem::",
    "x509_parser::",
    # ── Messaging ──
    "lapin::",
    # ── Encoding ──
    "base64::",
    "hex::",
    # ── Misc crates ──
    "uuid::",
    "chrono::",
    "rand::",
    "regex::",
    "url::",
    "futures::",
    "clap::",
    "config::",
    "figment::",
    "tempfile::",
    "walkdir::",
    "notify::",
    "crossbeam::",
    "parking_lot::",
    "dashmap::",
    "once_cell::",
    "bytes::",
    "pin_project::",
    "git2::",
    "testcontainers::",
    "dirs::",
    # ── External type prefixes ──
    "Status::",
    "StatusCode::",
    "HeaderValue::",
    "HeaderName::",
    "Method::",
    "TempDir::",
    "Regex::",
    "Uuid::",
    "DateTime::",
    "NaiveDate::",
    "NaiveDateTime::",
    "Utc::",
    "Router::",
    "KeyPair::",
    "Version::",
    "Endpoint::",
    "ProgressStyle::",
    "X509Certificate::",
    "Certificate::",
    "GenericImage::",
    "EnvFilter::",
    "Issuer::",
    "ClientTlsConfig::",
    "FieldValue::",
    "ProtoDeploymentSourceType::",
    "Identity::",
    "SetResponseHeaderLayer::",
    "EncodingKey::",
    "Confirm::",
    "Input::",
    "Password::",
    "query.load::",
    # ── Test receiver patterns ──
    "authenticatedPage.",
    "page.",
    "response.",
    "cert.",
    "conn.",
    "match.",
    "stripped.",
    "mcp.",
    "pptx.",
    "logger.",
    "spinner.",
    "hasher.",
    "pb.",
    "np.",
    "node_id.",
    "line.",
    "table.",
    "group.",
    "c.",
    "prs.",
    "subparsers.",
    "pytest.",
)

# Combined tuple used by is_non_project_call()
STDLIB_PREFIXES = _STDLIB_PREFIXES + _LIBRARY_PREFIXES


def build_line_to_context_map(nodes: list, file_id: str) -> dict:
    """Build a mapping from line numbers to containing function/method context.

    Uses span-size ordering to ensure innermost (most specific) context wins.
    For nested functions, the inner function should be the context for its lines,
    not the outer function.

    Args:
        nodes: List of graph nodes
        file_id: File ID prefix (e.g., "FILE:path/to/file.rs")

    Returns:
        Dict mapping line number -> node ID of containing function/method
    """
    # Collect all function/method spans in this file
    spans = []
    for node in nodes:
        if node["id"].startswith(file_id + "::"):
            if node["type"] in ("Function", "Method"):
                start = node["metadata"].get("lineno", 0)
                end = node["metadata"].get("end_lineno", start + 100)
                span_size = end - start
                spans.append((span_size, start, end, node["id"]))

    # Sort by span size DESCENDING (largest first)
    # This way, smaller (inner) spans override larger (outer) spans
    spans.sort(key=lambda x: -x[0])

    # Build the mapping - last write wins, so inner functions override
    line_to_context = {}
    for _, start, end, node_id in spans:
        for ln in range(start, end + 1):
            line_to_context[ln] = node_id

    return line_to_context


def is_non_project_call(callee: str) -> bool:
    """Check if a callee is a non-project call (stdlib or third-party library) that should be filtered."""
    # Direct match
    if callee in STDLIB_FILTER:
        return True

    # Extract last component for method calls like "foo.bar.unwrap"
    last_part = callee.split(".")[-1].split("::")[-1]

    # Diesel DSL column access pattern: table_name::column.method (e.g., targets::id.eq)
    # These are ORM DSL calls that never resolve to project functions
    if "::" in callee and "." in callee:
        parts = callee.split("::")
        if len(parts) == 2 and "." in parts[1]:
            col_part = parts[1].split(".")[0]
            # If both table and column are lowercase identifiers, it's Diesel DSL
            if parts[0].islower() and col_part.islower():
                return True

    # For qualified calls like Type::new, only filter if the type is also stdlib
    # This allows AppState::new but filters Vec::new, HashMap::new
    if "::" in callee:
        type_part = callee.split("::")[-2] if "::" in callee else ""
        # If qualified with a custom type (not stdlib and not Self), don't filter.
        # Self:: must fall through to the last_part check so that Self::new,
        # Self::default, Self::from etc. are correctly filtered by STDLIB_FILTER,
        # while Self::project_method passes through as a project call.
        if type_part and type_part != "Self" and type_part not in STDLIB_FILTER and not any(
            callee.startswith(p) for p in STDLIB_PREFIXES
        ):
            return False

    if last_part in STDLIB_FILTER:
        return True

    # Prefix match for qualified names
    for prefix in STDLIB_PREFIXES:
        if callee.startswith(prefix):
            return True

    return False


# --- Schema Definition ---
# Nodes: { "id": str, "type": str, "metadata": dict }
# Edges: { "source": str, "target": str, "relation": str }


class SymbolTable:
    def __init__(self):
        # Maps local_alias -> fully_qualified_name
        self.imports = {}

    def add_import(self, name, module=None, alias=None):
        key = alias if alias else name
        if module:
            value = f"{module}.{name}"
        else:
            value = name
        self.imports[key] = value

    def resolve(self, name):
        parts = name.split(".")
        root = parts[0]
        if root in self.imports:
            resolved_root = self.imports[root]
            if len(parts) > 1:
                return f"{resolved_root}.{'.'.join(parts[1:])}"
            return resolved_root
        return name


class TypeScriptSymbolTable:
    """Symbol table for TypeScript files with import tracking.

    Tracks imports to:
    - Skip type-only imports (they don't generate runtime calls)
    - Identify namespace imports for better SCIP matching

    Resolution is primarily handled by SCIP for accuracy.
    This table provides supplementary information.
    """

    def __init__(self, file_path: str, project_root: str):
        """Initialize the symbol table for a TypeScript file.

        Args:
            file_path: Absolute or relative path to the TypeScript file
            project_root: Root directory of the project
        """
        self.file_path = file_path
        self.file_dir = os.path.dirname(file_path)
        self.project_root = project_root

        # local_name -> (module_path, import_type)
        # import_type: "named", "default", "namespace", "type"
        self.imports: dict = {}

        # namespace_alias -> module_path (for import * as alias from 'module')
        self.namespaces: dict = {}

    def load_imports(self, import_data: dict):
        """Load import data from ast-grep extraction.

        Args:
            import_data: Dict with 'imports' and 'namespaces' keys
        """
        self.imports = import_data.get("imports", {})
        self.namespaces = import_data.get("namespaces", {})

    def is_type_import(self, name: str) -> bool:
        """Check if a name was imported with 'import type'.

        Type-only imports don't generate runtime calls.
        """
        root = name.split('.')[0]
        if root in self.imports:
            _, import_type = self.imports[root]
            return import_type == "type"
        return False

    def is_namespace_call(self, name: str) -> bool:
        """Check if name is a call on a namespace import (e.g., api.get).

        Namespace calls like 'schedulesApi.list' are qualified names that
        SCIP can resolve when given the full qualified form.
        """
        if '.' not in name:
            return False
        root = name.split('.')[0]
        return root in self.namespaces

    def get_import_source(self, name: str) -> str | None:
        """Get the import source for a name if it was imported.

        Args:
            name: Local name to look up

        Returns:
            Module path if imported, None otherwise
        """
        root = name.split('.')[0]
        if root in self.imports:
            module, _ = self.imports[root]
            return module
        if root in self.namespaces:
            return self.namespaces[root]
        return None


class BaseParser:
    def __init__(self, builder):
        self.builder = builder

    def parse(self, file_path, rel_path, content):
        raise NotImplementedError

    def get_leading_docstring(self, lines, start_idx):
        doc_lines = []
        j = start_idx - 1
        while j >= 0:
            line = lines[j].strip()
            # Handle common comment styles: ///, //, /*, *, /**
            if (
                line.startswith("///")
                or line.startswith("//!")
                or line.startswith("*")
                or line.startswith("/**")
                or line.startswith("//")
            ):
                # Skip end of block comments if they don't contain content
                if line == "*/":
                    j -= 1
                    continue
                cleaned = line.lstrip("/!* ").strip()
                doc_lines.insert(0, cleaned)
                j -= 1
            else:
                break
        return "\n".join(doc_lines)

    def _find_block_end(self, lines, start_idx):
        """Find the closing brace line for a block starting at start_idx.

        Counts braces to find the matching closing brace.
        Returns the 1-indexed line number of the closing brace.
        """
        brace_depth = 0
        found_open = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == "{":
                    brace_depth += 1
                    found_open = True
                elif char == "}":
                    brace_depth -= 1
                    if found_open and brace_depth == 0:
                        return i + 1  # 1-indexed line number

        # If no closing brace found, estimate based on file end
        return min(start_idx + 100, len(lines))


class PythonParser(BaseParser):
    def parse(self, file_path, rel_path, content):
        file_id = f"FILE:{rel_path}"
        self.builder.current_file_id = file_id
        self.builder.current_scope = SymbolTable()

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {rel_path}:{e.lineno}: {e.msg}")
            return
        except Exception as e:
            logger.warning(f"Parse error in {rel_path}: {e}")
            return

        self.builder.add_node(
            file_id,
            "File",
            path=rel_path,
            name=Path(rel_path).name,
            token_count=len(content) // 4,
        )
        self.visit_node(tree, file_id)

    def _get_type_annotation(self, annotation):
        """Extract type annotation as a string.

        Handles:
        - Simple types: int, str, MyClass
        - Qualified types: module.Type
        - Generic types: List[int], Dict[str, int]
        - Union types (3.10+): int | str
        - Tuple types: tuple[int, str]
        - Optional: Optional[int]
        - Literal: Literal["foo"]
        """
        if annotation is None:
            return "Any"
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            base = self._get_attribute_name(annotation.value)
            return f"{base}.{annotation.attr}" if base else annotation.attr
        elif isinstance(annotation, ast.Subscript):
            base = self._get_type_annotation(annotation.value)
            slice_val = annotation.slice
            # Handle tuple slices (Dict[K, V], Callable[[...], R])
            if isinstance(slice_val, ast.Tuple):
                elts = ", ".join(self._get_type_annotation(e) for e in slice_val.elts)
                return f"{base}[{elts}]"
            else:
                return f"{base}[{self._get_type_annotation(slice_val)}]"
        elif isinstance(annotation, ast.Constant):
            # Literal types like Literal["foo"]
            return (
                repr(annotation.value)
                if isinstance(annotation.value, str)
                else str(annotation.value)
            )
        elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            # Union types (3.10+): int | str
            left = self._get_type_annotation(annotation.left)
            right = self._get_type_annotation(annotation.right)
            return f"{left} | {right}"
        elif isinstance(annotation, ast.Tuple):
            # Bare tuple in annotations
            elts = ", ".join(self._get_type_annotation(e) for e in annotation.elts)
            return f"({elts})"
        elif isinstance(annotation, ast.List):
            # List in Callable[[arg1, arg2], return]
            elts = ", ".join(self._get_type_annotation(e) for e in annotation.elts)
            return f"[{elts}]"
        return "Complex"

    def _get_args_info(self, args_node):
        args_list = []
        for arg in args_node.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_type_annotation(arg.annotation)}"
            args_list.append(arg_str)
        return args_list

    def _get_param_types(self, args_node):
        """Extract parameter types as a list for metadata storage."""
        param_types = []
        for arg in args_node.args:
            if arg.annotation:
                param_types.append(self._get_type_annotation(arg.annotation))
            else:
                param_types.append("Any")
        return param_types

    def _estimate_tokens(self, node):
        if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
            return (node.end_lineno - node.lineno + 1) * 10
        return 0

    def visit_node(self, node, parent_id):
        if isinstance(node, ast.ClassDef):
            class_id = f"{parent_id}::{node.name}"
            self.builder.add_node(
                class_id,
                "Class",
                name=node.name,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno),
                token_count=self._estimate_tokens(node),
                docstring=ast.get_docstring(node) or "",
            )
            self.builder.add_edge(parent_id, class_id, "DEFINES")

            # Track decorator calls (e.g., @dataclass, @pytest.mark.parametrize(...))
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    self._record_call(decorator, class_id, decorator.lineno)
                elif isinstance(decorator, ast.Attribute):
                    # Handle @module.decorator without call parens
                    name = self._get_attribute_name(decorator)
                    if name and not is_non_project_call(name):
                        self.builder.edges.append(
                            {
                                "source": class_id,
                                "target": f"REF:{name}",
                                "relation": "CALLS",
                                "metadata": {"lineno": decorator.lineno},
                            }
                        )

            for base in node.bases:
                base_name = (
                    self._get_attribute_name(base)
                    if isinstance(base, ast.Attribute)
                    else (base.id if isinstance(base, ast.Name) else None)
                )
                if base_name:
                    resolved_base = self.builder.current_scope.resolve(base_name)
                    self.builder.add_edge(class_id, f"REF:{resolved_base}", "INHERITS")

            for child in node.body:
                self.visit_node(child, class_id)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_id = f"{parent_id}::{node.name}"
            is_method = "::" in parent_id.split("FILE:")[1]
            node_type = "Method" if is_method else "Function"

            args = self._get_args_info(node.args)
            param_types = self._get_param_types(node.args)
            return_type = (
                self._get_type_annotation(node.returns) if node.returns else "None"
            )
            signature = f"def {node.name}({', '.join(args)}) -> {return_type}"

            self.builder.add_node(
                func_id,
                node_type,
                name=node.name,
                signature=signature,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno),
                token_count=self._estimate_tokens(node),
                docstring=ast.get_docstring(node) or "",
                return_type=return_type,
                param_types=param_types,
            )
            self.builder.add_edge(parent_id, func_id, "DEFINES")

            # Track decorator calls (e.g., @lru_cache(), @pytest.fixture)
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    self._record_call(decorator, func_id, decorator.lineno)
                elif isinstance(decorator, ast.Attribute):
                    name = self._get_attribute_name(decorator)
                    if name and not is_non_project_call(name):
                        self.builder.edges.append(
                            {
                                "source": func_id,
                                "target": f"REF:{name}",
                                "relation": "CALLS",
                                "metadata": {"lineno": decorator.lineno},
                            }
                        )
                elif isinstance(decorator, ast.Name):
                    if not is_non_project_call(decorator.id):
                        self.builder.edges.append(
                            {
                                "source": func_id,
                                "target": f"REF:{decorator.id}",
                                "relation": "CALLS",
                                "metadata": {"lineno": decorator.lineno},
                            }
                        )

            self.visit_body_for_calls(node.body, func_id)

            for child in node.body:
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    self.visit_node(child, func_id)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                self.builder.current_scope.add_import(alias.name, alias=alias.asname)
                self.builder.add_edge(
                    self.builder.current_file_id, f"MODULE:{alias.name}", "IMPORTS"
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            for alias in node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                self.builder.current_scope.add_import(
                    alias.name, module=module, alias=alias.asname
                )
                self.builder.add_edge(
                    self.builder.current_file_id, f"MODULE:{full_name}", "IMPORTS"
                )

        elif isinstance(node, ast.Module):
            for child in node.body:
                self.visit_node(child, parent_id)

    def visit_body_for_calls(self, body, parent_func_id):
        """Find all Call nodes in the function body using ast.walk().

        This catches calls in all contexts: comprehensions, generators,
        with statements, if/while conditions, for iterables, decorators,
        nested calls, await expressions, etc.
        """
        for node in body:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    lineno = getattr(child, "lineno", 0)
                    self._record_call(child, parent_func_id, lineno)

    def _record_call(self, call_node, parent_func_id, lineno):
        func_name = self._get_func_name(call_node.func)
        if func_name:
            resolved_name = self.builder.current_scope.resolve(func_name)
            if not is_non_project_call(resolved_name):
                self.builder.edges.append(
                    {
                        "source": parent_func_id,
                        "target": f"REF:{resolved_name}",
                        "relation": "CALLS",
                        "metadata": {"lineno": lineno},
                    }
                )

    def _get_func_name(self, func_node):
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return self._get_attribute_name(func_node)
        return None

    def _get_attribute_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_attribute_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return None


class RustParser(BaseParser):
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
        current_context = [file_id]  # Stack of IDs
        context_indent = [0]  # Stack of indentation levels

        # Regex patterns
        re_fn_start = re.compile(
            r"^\s*(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([a-zA-Z0-9_]+)"
        )
        re_struct = re.compile(
            r"^\s*(?:pub\s+)?(?:struct|trait)\s+([a-zA-Z0-9_]+)"
        )
        re_enum = re.compile(r"^\s*(?:pub\s+)?enum\s+([a-zA-Z0-9_]+)")
        # Enum variant patterns: Name, Name(Type), Name { field: Type }
        re_enum_variant = re.compile(r"^\s*([A-Z][a-zA-Z0-9_]*)\s*(?:[({\,]|$)")
        re_impl = re.compile(
            r"^\s*impl(?:<.*?>)?\s+(?:([a-zA-Z0-9_:<>,\s]+?)\s+for\s+)?([a-zA-Z0-9_:]+)"
        )
        re_mod = re.compile(r"^\s*(?:pub\s+)?mod\s+([a-zA-Z0-9_]+)\s*\{?")
        re_use = re.compile(r"^\s*(?:pub\s+)?use\s+([^;]+);")
        re_const = re.compile(
            r"^\s*(?:pub(?:\(.*?\))?\s+)?(?:const|static)\s+([A-Z0-9_]+)\s*:\s*.*?\s*=\s*(.*?);"
        )
        re_call = re.compile(r"([a-zA-Z0-9_:]+)\(")

        current_impl_target = None
        current_trait_impl = None  # Track trait name when in "impl Trait for Struct" block
        current_enum_id = None  # Track if we're inside an enum for variant parsing

        i = 0
        while i < len(lines):
            line = lines[i]
            lineno = i + 1
            indent = len(line) - len(line.lstrip())

            # Pop context
            if "}" in line:
                while len(context_indent) > 1 and indent <= context_indent[-1]:
                    context_indent.pop()
                    popped_id = current_context.pop()
                    # Reset enum tracking when leaving enum scope
                    if current_enum_id and popped_id == current_enum_id:
                        current_enum_id = None
                    if len(current_context) == 1:
                        current_impl_target = None
                        current_trait_impl = None

            parent_id = current_context[-1]

            # 0. Constants
            if match := re_const.search(line):
                name, val = match.groups()
                docstring = self.get_leading_docstring(lines, i)
                cid = f"{parent_id}::{name}"
                self.builder.add_node(
                    cid,
                    "Constant",
                    name=name,
                    value=val,
                    lineno=lineno,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")

            # 1. Functions
            if match := re_fn_start.search(line):
                name = match.group(1)
                j, full_sig = self._consume_until_body(lines, i)

                ret = "()"
                param_types = []
                if "->" in full_sig:
                    try:
                        parts = full_sig.split("->")
                        ret_part = parts[-1].split("{")[0].strip().strip(";")
                        ret = ret_part
                    except (IndexError, ValueError):
                        pass

                # Extract parameter types using lightweight heuristics
                param_types = self._extract_rust_param_types(full_sig)

                sig = f"fn {name}(...) -> {ret}"
                docstring = self.get_leading_docstring(lines, i)

                if current_impl_target:
                    func_id = f"{file_id}::{current_impl_target}::{name}"
                    node_type = "Method"
                else:
                    func_id = f"{parent_id}::{name}"
                    node_type = "Function"

                # Find actual function end line by counting braces
                end_lineno = (
                    self._find_block_end(lines, j) if "{" in lines[j] else lineno
                )

                # Estimate tokens: ~10 tokens per line of code
                token_count = (end_lineno - lineno + 1) * 10

                # Build metadata dict, only include trait_impl if set
                node_kwargs = dict(
                    name=name,
                    signature=sig,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
                    return_type=ret,
                    param_types=param_types,
                )
                if current_trait_impl and node_type == "Method":
                    node_kwargs["trait_impl"] = current_trait_impl

                self.builder.add_node(func_id, node_type, **node_kwargs)
                self.builder.add_edge(parent_id, func_id, "DEFINES")

                if "{" in lines[j]:
                    current_context.append(func_id)
                    context_indent.append(indent)
                i = j

            # 2. Structs/Traits
            elif match := re_struct.search(line):
                name = match.group(1)
                docstring = self.get_leading_docstring(lines, i)
                type_id = f"{parent_id}::{name}"
                # Find end of struct/trait block
                end_lineno = self._find_block_end(lines, i) if "{" in line else lineno
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    type_id, "Class", name=name, lineno=lineno, end_lineno=end_lineno,
                    token_count=token_count, docstring=docstring
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                if "{" in line:
                    current_context.append(type_id)
                    context_indent.append(indent)

            # 2b. Enums (separate to track for variant parsing)
            elif match := re_enum.search(line):
                name = match.group(1)
                docstring = self.get_leading_docstring(lines, i)
                type_id = f"{parent_id}::{name}"
                # Find end of enum block
                end_lineno = self._find_block_end(lines, i) if "{" in line else lineno
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    type_id, "Class", name=name, lineno=lineno, end_lineno=end_lineno,
                    token_count=token_count, docstring=docstring
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                if "{" in line:
                    current_context.append(type_id)
                    context_indent.append(indent)
                    current_enum_id = type_id  # Track enum for variant parsing

            # 2c. Enum variants (when inside an enum)
            elif current_enum_id and (match := re_enum_variant.search(line)):
                variant_name = match.group(1)
                # Skip common Rust keywords/patterns that match variant regex
                if variant_name not in ("Self", "Some", "None", "Ok", "Err", "Box", "Vec", "Option", "Result"):
                    variant_id = f"{current_enum_id}::{variant_name}"
                    self.builder.add_node(
                        variant_id,
                        "Function",  # Treat as callable for resolution
                        name=variant_name,
                        signature=f"{current_enum_id.split('::')[-1]}::{variant_name}",
                        lineno=lineno,
                        docstring="",
                    )
                    self.builder.add_edge(current_enum_id, variant_id, "DEFINES")

            # 3. Impl
            elif match := re_impl.search(line):
                trait, target = match.groups()
                real_target = (target if target else trait).split("<")[0]
                current_impl_target = real_target.split("::")[-1]

                if trait:
                    trait_name = trait.split("<")[0].split("::")[-1]
                    current_trait_impl = trait_name  # Track trait for method metadata
                    struct_id = f"{file_id}::{current_impl_target}"
                    self.builder.add_edge(struct_id, f"REF:{trait_name}", "INHERITS")
                else:
                    current_trait_impl = None  # Inherent impl, no trait

                if "{" in line:
                    struct_id = f"{file_id}::{current_impl_target}"
                    current_context.append(struct_id)
                    context_indent.append(indent)

            # 4. Modules
            elif match := re_mod.search(line):
                name = match.group(1)
                mod_id = f"{parent_id}::{name}"
                if "{" in line:
                    current_context.append(mod_id)
                    context_indent.append(indent)

            # 5. Imports
            elif match := re_use.search(line):
                path = match.group(1).replace("::", ".")
                self.builder.add_edge(file_id, f"MODULE:{path}", "IMPORTS")

            # 6. Calls (regex fallback - only used if ast-grep unavailable)
            if not USE_AST_GREP and parent_id != file_id:
                for call_match in re_call.finditer(line):
                    callee = call_match.group(1)
                    if not is_non_project_call(callee):
                        self.builder.edges.append(
                            {
                                "source": parent_id,
                                "target": f"REF:{callee}",
                                "relation": "CALLS",
                                "metadata": {"lineno": lineno},
                            }
                        )
            i += 1

        # Use ast-grep for more accurate CALLS detection
        if USE_AST_GREP:
            self._extract_calls_with_ast_grep(str(file_path), file_id, lines)

    def _extract_calls_with_ast_grep(self, file_path: str, file_id: str, lines: list):
        """Extract calls using ast-grep for higher accuracy."""
        # Build a line-to-context map from our parsed structure
        # Uses span-size ordering to ensure innermost context wins for nested functions
        line_to_context = build_line_to_context_map(self.builder.nodes, file_id)

        for call in extract_calls_rust(file_path):
            lineno = call["lineno"]
            callee = call["callee"]

            # Filter stdlib calls
            if is_non_project_call(callee):
                continue

            # Find the containing function for this line
            context_id = line_to_context.get(lineno)
            if not context_id:
                # Try nearby lines
                for offset in range(1, 10):
                    context_id = line_to_context.get(
                        lineno - offset
                    ) or line_to_context.get(lineno + offset)
                    if context_id:
                        break

            # Fall back to file_id if no function context found (file-level calls)
            source_id = context_id if context_id else file_id
            self.builder.edges.append(
                {
                    "source": source_id,
                    "target": f"REF:{callee}",
                    "relation": "CALLS",
                    "metadata": {"lineno": lineno},
                }
            )

            # Track struct instantiation patterns
            # Expanded pattern list covers common Rust constructors
            constructor_patterns = (
                "new", "default", "builder",
                "from", "try_from", "from_str", "from_bytes", "from_slice",
                "open", "create", "connect",
                "with_capacity", "with_config", "with_options",
                "init", "initialize",
            )
            if "::" in callee:
                parts = callee.split("::")
                if len(parts) >= 2:
                    method_name = parts[-1]
                    struct_name = parts[-2]
                    # Match constructor patterns (exact or prefix for with_*)
                    is_constructor = (
                        method_name in constructor_patterns or
                        method_name.startswith("with_") or
                        method_name.startswith("from_")
                    )
                    # Skip if it's a module path like std::collections::HashMap::new
                    if is_constructor and struct_name[0].isupper():
                        self.builder.edges.append(
                            {
                                "source": source_id,
                                "target": f"REF:{struct_name}",
                                "relation": "INSTANTIATES",
                                "metadata": {"lineno": lineno, "via": callee},
                            }
                        )

        # Also extract struct literal instantiations
        self._extract_struct_literals(file_path, file_id, lines, line_to_context)

        # Extract calls from macro bodies (tokio::select!, etc.)
        self._extract_macro_body_calls(file_id, lines, line_to_context)

    def _extract_struct_literals(
        self, file_path: str, file_id: str, lines: list, line_to_context: dict
    ):
        """Extract struct literal instantiations like `MyStruct { field: value }`.

        Uses regex to find patterns that look like struct literals.
        """
        re_struct_literal = re.compile(r"\b([A-Z][a-zA-Z0-9_]*)\s*\{")

        for lineno, line in enumerate(lines, 1):
            # Skip struct/enum/trait definitions
            if re.match(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+", line):
                continue
            # Skip impl blocks
            if re.match(r"^\s*impl", line):
                continue

            for match in re_struct_literal.finditer(line):
                struct_name = match.group(1)
                # Skip common non-struct patterns
                if struct_name in ("Some", "None", "Ok", "Err", "Self"):
                    continue

                context_id = line_to_context.get(lineno)
                if not context_id:
                    for offset in range(1, 10):
                        context_id = line_to_context.get(
                            lineno - offset
                        ) or line_to_context.get(lineno + offset)
                        if context_id:
                            break

                source_id = context_id if context_id else file_id
                self.builder.edges.append(
                    {
                        "source": source_id,
                        "target": f"REF:{struct_name}",
                        "relation": "INSTANTIATES",
                        "metadata": {"lineno": lineno, "via": "literal"},
                    }
                )

    def _extract_macro_body_calls(
        self, file_id: str, lines: list, line_to_context: dict
    ):
        """Extract function/method calls from inside macro bodies.

        Macro bodies like tokio::select! aren't parsed by ast-grep because
        they're not valid syntax until macro expansion. This uses regex
        to find common call patterns inside macro invocations.

        Patterns detected:
        - self.method() and self.method().await
        - receiver.method() where receiver is a variable
        - function() calls
        """
        # Patterns for calls inside macro bodies
        # Method call: word.method() or word.method().await
        re_method_call = re.compile(
            r"\b([a-z_][a-z0-9_]*)\s*\.\s*([a-z_][a-z0-9_]*)\s*\(\s*\)"
        )
        # Method call with args: word.method(args)
        re_method_call_args = re.compile(
            r"\b([a-z_][a-z0-9_]*)\s*\.\s*([a-z_][a-z0-9_]*)\s*\([^)]*\)"
        )
        # Chained await: .method().await
        re_await_method = re.compile(
            r"\.([a-z_][a-z0-9_]*)\s*\(\s*\)\s*\.await"
        )

        # Track which lines are inside macro bodies
        in_macro = False
        macro_depth = 0

        for lineno, line in enumerate(lines, 1):
            # Detect macro start (common async macros)
            if re.search(r"\b(tokio::select|tokio::spawn|async_std::task)", line):
                in_macro = True
                macro_depth = line.count("{") - line.count("}")

            if in_macro:
                macro_depth += line.count("{") - line.count("}")
                if macro_depth <= 0:
                    in_macro = False
                    continue

                # Find method calls in this line
                for pattern in [re_method_call, re_method_call_args, re_await_method]:
                    for match in pattern.finditer(line):
                        if pattern == re_await_method:
                            method_name = match.group(1)
                            receiver = None
                        else:
                            receiver = match.group(1)
                            method_name = match.group(2)

                        # Skip stdlib methods
                        if is_non_project_call(method_name):
                            continue

                        # Skip if receiver is a common variable pattern
                        if receiver in ("e", "err", "error", "_"):
                            continue

                        context_id = line_to_context.get(lineno)
                        if not context_id:
                            for offset in range(1, 20):
                                context_id = line_to_context.get(
                                    lineno - offset
                                ) or line_to_context.get(lineno + offset)
                                if context_id:
                                    break

                        source_id = context_id if context_id else file_id

                        # Create qualified name if receiver is self
                        if receiver == "self":
                            callee = f"self.{method_name}"
                        else:
                            callee = method_name

                        self.builder.edges.append(
                            {
                                "source": source_id,
                                "target": f"REF:{callee}",
                                "relation": "CALLS",
                                "metadata": {"lineno": lineno, "via": "macro"},
                            }
                        )

    def _consume_until_body(self, lines, start_idx):
        sig_lines = [lines[start_idx]]
        j = start_idx
        open_p = lines[j].count("(")
        close_p = lines[j].count(")")
        while (
            open_p > close_p or ("{" not in lines[j] and ";" not in lines[j])
        ) and j < len(lines) - 1:
            j += 1
            sig_lines.append(lines[j])
            open_p += lines[j].count("(")
            close_p += lines[j].count(")")
        return j, " ".join([ln.strip() for ln in sig_lines])

    def _extract_rust_param_types(self, full_sig: str) -> list:
        """Extract parameter types from Rust function signature.

        Uses lightweight heuristics to extract types without full parsing.
        Handles common patterns like:
        - &self, &mut self
        - name: Type
        - name: &Type
        - name: impl Trait
        - name: Box<T>
        """
        param_types = []
        try:
            # Extract content between first ( and matching )
            paren_start = full_sig.find("(")
            if paren_start == -1:
                return []

            # Find matching close paren (handle nested)
            depth = 0
            paren_end = -1
            for i, c in enumerate(full_sig[paren_start:], paren_start):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        paren_end = i
                        break

            if paren_end == -1:
                return []

            params_str = full_sig[paren_start + 1 : paren_end].strip()
            if not params_str:
                return []

            # Split on comma, but not inside angle brackets or parens
            params = []
            current = []
            depth = 0
            for c in params_str:
                if c in "<(":
                    depth += 1
                elif c in ">)":
                    depth -= 1
                elif c == "," and depth == 0:
                    params.append("".join(current).strip())
                    current = []
                    continue
                current.append(c)
            if current:
                params.append("".join(current).strip())

            for param in params:
                param = param.strip()
                if not param:
                    continue

                # Handle &self, &mut self, self
                if param in ("self", "&self", "&mut self"):
                    param_types.append(param)
                    continue

                # Handle name: Type pattern
                if ":" in param:
                    parts = param.split(":", 1)
                    if len(parts) == 2:
                        type_part = parts[1].strip()
                        # Simplify complex types for readability
                        if len(type_part) > 50:
                            type_part = type_part[:47] + "..."
                        param_types.append(type_part)
                else:
                    # No type annotation
                    param_types.append("_")

        except (IndexError, ValueError):
            pass

        return param_types


class ProtoParser(BaseParser):
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
        current_context = [file_id]
        context_indent = [0]

        re_service = re.compile(r"^\s*service\s+([a-zA-Z0-9_]+)")
        re_rpc = re.compile(
            r"^\s*rpc\s+([a-zA-Z0-9_]+)\s*\((.*?)\)\s*returns\s*\((.*?)\)"
        )
        re_message = re.compile(r"^\s*message\s+([a-zA-Z0-9_]+)")

        for i, line in enumerate(lines):
            lineno = i + 1
            indent = len(line) - len(line.lstrip())

            if "}" in line:
                while len(context_indent) > 1 and indent <= context_indent[-1]:
                    context_indent.pop()
                    current_context.pop()

            parent_id = current_context[-1]

            if match := re_service.search(line):
                name = match.group(1)
                docstring = self.get_leading_docstring(lines, i)
                sid = f"{parent_id}::{name}"
                self.builder.add_node(
                    sid, "Class", name=name, lineno=lineno, docstring=docstring
                )
                self.builder.add_edge(parent_id, sid, "DEFINES")
                if "{" in line:
                    current_context.append(sid)
                    context_indent.append(indent)

            elif match := re_rpc.search(line):
                name, req, res = match.groups()
                docstring = self.get_leading_docstring(lines, i)
                rid = f"{parent_id}::{name}"
                sig = f"rpc {name}({req}) returns ({res})"
                self.builder.add_node(
                    rid,
                    "Function",
                    name=name,
                    signature=sig,
                    lineno=lineno,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, rid, "DEFINES")
                if req:
                    self.builder.add_edge(rid, f"REF:{req}", "CALLS")
                if res:
                    self.builder.add_edge(rid, f"REF:{res}", "CALLS")

            elif match := re_message.search(line):
                name = match.group(1)
                docstring = self.get_leading_docstring(lines, i)
                mid = f"{parent_id}::{name}"
                self.builder.add_node(
                    mid, "Class", name=name, lineno=lineno, docstring=docstring
                )
                self.builder.add_edge(parent_id, mid, "DEFINES")
                if "{" in line:
                    current_context.append(mid)
                    context_indent.append(indent)


class TSParser(BaseParser):
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
        current_context = [file_id]

        # Initialize TypeScript symbol table for import tracking
        self.symbol_table = TypeScriptSymbolTable(
            str(file_path), str(self.builder.root_dir)
        )

        # Extract imports using ast-grep if available
        if USE_AST_GREP and extract_imports_typescript is not None:
            try:
                import_data = extract_imports_typescript(str(file_path))
                self.symbol_table.load_imports(import_data)
            except Exception:
                pass  # Fall back to unqualified calls

        re_class = re.compile(
            r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([a-zA-Z0-9_]+)"
        )
        re_interface = re.compile(r"^\s*(?:export\s+)?interface\s+([a-zA-Z0-9_]+)")
        re_func = re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_]+)"
        )
        re_method = re.compile(
            r"^\s*(?:private|public|protected)?\s*(?:static\s+)?(?:async\s+)?([a-zA-Z0-9_]+)\s*(?:<[^>]*>)?\s*[(]"
        )
        # Getter/setter methods: get name() or set name(value)
        re_getter_setter = re.compile(
            r"^\s*(?:private|public|protected)?\s*(?:static\s+)?(get|set)\s+([a-zA-Z0-9_]+)\s*[(]"
        )
        re_const_arrow = re.compile(
            r"^\s*(?:export\s+)?const\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?(?:[^)]*|[^=]*)\s*=>"
        )
        re_enum = re.compile(r"^\s*(?:export\s+)?enum\s+([a-zA-Z0-9_]+)")
        re_const = re.compile(
            r"^\s*(?:export\s+)?const\s+([a-zA-Z0-9_]+)\s*(?::\s*.*?)?\s*=\s*(.*?);"
        )
        re_import = re.compile(r'import\s+.*?from\s+([\'"])(.*?)\1')
        re_call_candidate = re.compile(
            r"([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)\s*(?=[<(])"
        )
        # Configuration patterns: interceptors, middleware, event handlers
        # These are method calls that set up important runtime behavior
        re_interceptor = re.compile(
            r"^\s*([a-zA-Z0-9_]+)\.interceptors\.(request|response)\.use\s*\("
        )
        re_middleware = re.compile(
            r"^\s*(?:app|router|server)\.use\s*\(\s*(['\"]?)([^'\"(),]+)\1"
        )
        re_event_handler = re.compile(
            r"^\s*([a-zA-Z0-9_]+)\.on\s*\(\s*['\"]([a-zA-Z0-9_:]+)['\"]"
        )

        brace_balance = 0
        i = 0
        while i < len(lines):
            line = lines[i]
            lineno = i + 1

            brace_balance += line.count("{") - line.count("}")
            if brace_balance < len(current_context) - 1:
                if len(current_context) > 1:
                    current_context.pop()

            parent_id = current_context[-1]

            def consume_sig(idx):
                nonlocal brace_balance
                sig_lines = [lines[idx]]
                j = idx
                while (
                    "{" not in lines[j] and ";" not in lines[j] and j < len(lines) - 1
                ):
                    j += 1
                    sig_lines.append(lines[j])
                    brace_balance += lines[j].count("{") - lines[j].count("}")
                return j, " ".join([ln.strip() for ln in sig_lines])

            if match := re_const.search(line):
                name, val = match.groups()
                doc = self.get_leading_docstring(lines, i)
                cid = f"{parent_id}::{name}"
                self.builder.add_node(
                    cid, "Constant", name=name, value=val, lineno=lineno, docstring=doc
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")
            elif match := re_enum.search(line):
                name = match.group(1)
                doc = self.get_leading_docstring(lines, i)
                cid = f"{parent_id}::{name}"
                self.builder.add_node(
                    cid, "Class", name=name, lineno=lineno, docstring=doc
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")
                if "{" in line:
                    current_context.append(cid)
            elif match := re_class.search(line):
                name = match.group(1)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig = consume_sig(i)
                cid = f"{parent_id}::{name}"
                # Find end of class block
                end_lineno = self._find_block_end(lines, j) if "{" in lines[j] else lineno
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    cid, "Class", name=name, lineno=lineno, end_lineno=end_lineno,
                    token_count=token_count, docstring=docstring
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")
                if "extends" in full_sig:
                    try:
                        base = full_sig.split("extends")[1].split()[0].split("{")[0]
                        self.builder.add_edge(cid, f"REF:{base}", "INHERITS")
                    except (IndexError, ValueError):
                        pass
                if "{" in lines[j]:
                    current_context.append(cid)
                i = j
            elif match := re_interface.search(line):
                name = match.group(1)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig = consume_sig(i)
                cid = f"{parent_id}::{name}"
                # Find end of interface block
                end_lineno = self._find_block_end(lines, j) if "{" in lines[j] else lineno
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    cid, "Class", name=name, lineno=lineno, end_lineno=end_lineno,
                    token_count=token_count, docstring=docstring
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")
                if "extends" in full_sig:
                    try:
                        bases = full_sig.split("extends")[1].split("{")[0].split(",")
                        for b in bases:
                            self.builder.add_edge(cid, f"REF:{b.strip()}", "INHERITS")
                    except (IndexError, ValueError):
                        pass
                if "{" in lines[j]:
                    current_context.append(cid)
                i = j
            elif match := re_func.search(line):
                name = match.group(1)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig = consume_sig(i)
                fid = f"{parent_id}::{name}"
                return_type, param_types = self._extract_ts_types(full_sig)
                end_lineno = (
                    self._find_block_end(lines, j) if "{" in lines[j] else lineno
                )
                signature = self._build_ts_signature(name, param_types, return_type)
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    fid,
                    "Function",
                    name=name,
                    signature=signature,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
                    return_type=return_type,
                    param_types=param_types,
                )
                self.builder.add_edge(parent_id, fid, "DEFINES")
                if "{" in lines[j]:
                    current_context.append(fid)
                i = j
            elif match := re_const_arrow.search(line):
                name = match.group(1)
                docstring = self.get_leading_docstring(lines, i)
                fid = f"{parent_id}::{name}"
                end_lineno = self._find_block_end(lines, i) if "{" in line else lineno
                # Arrow functions don't have easily extractable signatures
                signature = f"const {name} = (...) => ..."
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    fid,
                    "Function",
                    name=name,
                    signature=signature,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, fid, "DEFINES")
                if "{" in line:
                    current_context.append(fid)
            elif (
                "::" in parent_id
                and parent_id != file_id
                and (match := re_method.search(line))
            ):
                name = match.group(1)
                if name not in ["if", "for", "while", "switch", "catch"]:
                    docstring = self.get_leading_docstring(lines, i)
                    j, full_sig = consume_sig(i)
                    mid = f"{parent_id}::{name}"
                    return_type, param_types = self._extract_ts_types(full_sig)
                    end_lineno = (
                        self._find_block_end(lines, j) if "{" in lines[j] else lineno
                    )
                    # Build signature like: function name(p1: T1, p2: T2): ReturnType
                    signature = self._build_ts_signature(name, param_types, return_type)
                    token_count = (end_lineno - lineno + 1) * 10
                    self.builder.add_node(
                        mid,
                        "Method",
                        name=name,
                        signature=signature,
                        lineno=lineno,
                        end_lineno=end_lineno,
                        token_count=token_count,
                        docstring=docstring,
                        return_type=return_type,
                        param_types=param_types,
                    )
                    self.builder.add_edge(parent_id, mid, "DEFINES")
                    if "{" in lines[j]:
                        current_context.append(mid)
                    i = j
            # Handle getter/setter methods: get name() or set name(value)
            # Only match when parent is a Class (not inside a function/method)
            elif (
                "::" in parent_id
                and parent_id != file_id
                and self._is_class_context(parent_id)
                and (match := re_getter_setter.search(line))
            ):
                accessor_type = match.group(1)  # "get" or "set"
                name = match.group(2)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig = consume_sig(i)
                # Include accessor type in ID to distinguish get/set pairs
                mid = f"{parent_id}::{accessor_type}_{name}"
                return_type, param_types = self._extract_ts_types(full_sig)
                end_lineno = (
                    self._find_block_end(lines, j) if "{" in lines[j] else lineno
                )
                # Build signature for getter/setter
                if accessor_type == "get":
                    signature = f"get {name}(): {return_type}"
                else:
                    param_str = ", ".join(param_types) if param_types else "value"
                    signature = f"set {name}({param_str})"
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    mid,
                    "Method",
                    name=name,
                    signature=signature,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
                    return_type=return_type,
                    param_types=param_types,
                    accessor=accessor_type,
                )
                self.builder.add_edge(parent_id, mid, "DEFINES")
                # Don't push getters/setters to context - they're leaf methods
                # that shouldn't contain nested definitions
                i = j
            elif match := re_import.search(line):
                self.builder.add_edge(file_id, f"MODULE:{match.group(2)}", "IMPORTS")

            # Configuration patterns - capture as Configuration nodes
            # These are important setup calls that often need to be found during investigation
            if match := re_interceptor.search(line):
                client_name = match.group(1)
                interceptor_type = match.group(2)
                config_name = f"{client_name}_{interceptor_type}_interceptor"
                config_id = f"{file_id}::{config_name}"
                # Find the callback's end by tracking braces
                end_lineno = self._find_block_end(lines, i) if "{" in line or "=>" in line else lineno
                # Look for next few lines if arrow function continues
                if end_lineno == lineno:
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if "{" in lines[j] or "=>" in lines[j]:
                            end_lineno = self._find_block_end(lines, j)
                            break
                token_count = max((end_lineno - lineno + 1) * 10, 50)
                docstring = self.get_leading_docstring(lines, i)
                self.builder.add_node(
                    config_id,
                    "Configuration",
                    name=config_name,
                    signature=f"{client_name}.interceptors.{interceptor_type}.use(...)",
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring or f"Configures {interceptor_type} interceptor for {client_name}",
                    config_type="interceptor",
                    target=client_name,
                )
                self.builder.add_edge(file_id, config_id, "DEFINES")
            elif match := re_middleware.search(line):
                middleware_name = match.group(2).strip()
                if middleware_name and not middleware_name.startswith("("):
                    config_name = f"middleware_{middleware_name.replace('/', '_').replace('.', '_')}"
                    config_id = f"{file_id}::{config_name}"
                    end_lineno = self._find_block_end(lines, i) if "{" in line else lineno
                    token_count = max((end_lineno - lineno + 1) * 10, 30)
                    self.builder.add_node(
                        config_id,
                        "Configuration",
                        name=config_name,
                        signature=f"app.use({middleware_name})",
                        lineno=lineno,
                        end_lineno=end_lineno,
                        token_count=token_count,
                        docstring=f"Registers middleware: {middleware_name}",
                        config_type="middleware",
                    )
                    self.builder.add_edge(file_id, config_id, "DEFINES")
            elif match := re_event_handler.search(line):
                emitter_name = match.group(1)
                event_name = match.group(2)
                config_name = f"{emitter_name}_on_{event_name.replace(':', '_')}"
                config_id = f"{file_id}::{config_name}"
                end_lineno = self._find_block_end(lines, i) if "{" in line or "=>" in line else lineno
                token_count = max((end_lineno - lineno + 1) * 10, 30)
                docstring = self.get_leading_docstring(lines, i)
                self.builder.add_node(
                    config_id,
                    "Configuration",
                    name=config_name,
                    signature=f"{emitter_name}.on('{event_name}', ...)",
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring or f"Handles '{event_name}' event on {emitter_name}",
                    config_type="event_handler",
                    target=emitter_name,
                    event=event_name,
                )
                self.builder.add_edge(file_id, config_id, "DEFINES")

            # Calls (regex fallback - only used if ast-grep unavailable)
            if not USE_AST_GREP and parent_id != file_id:
                for match in re_call_candidate.finditer(line):
                    callee = match.group(1)
                    if is_non_project_call(callee):
                        continue
                    rem = line[match.end() :].lstrip()
                    is_call = False
                    if rem.startswith("("):
                        is_call = True
                    elif rem.startswith("<"):
                        bal = 0
                        for idx, c in enumerate(rem):
                            if c == "<":
                                bal += 1
                            elif c == ">":
                                bal -= 1
                            if bal == 0:
                                if rem[idx + 1 :].lstrip().startswith("("):
                                    is_call = True
                                break
                    if is_call:
                        self.builder.edges.append(
                            {
                                "source": parent_id,
                                "target": f"REF:{callee}",
                                "relation": "CALLS",
                                "metadata": {"lineno": lineno},
                            }
                        )
            i += 1

        # Use ast-grep for more accurate CALLS detection
        if USE_AST_GREP:
            self._extract_calls_with_ast_grep(str(file_path), file_id, lines)

    def _extract_ts_types(self, full_sig: str) -> tuple:
        """Extract return type and parameter types from TypeScript function signature.

        Returns: (return_type: str, param_types: list[str])

        Handles common patterns:
        - function foo(x: string): number
        - async function bar(): Promise<void>
        - (a: Type, b: Type) => ReturnType
        """
        return_type = "void"
        param_types = []

        try:
            # Extract return type (after closing paren, before opening brace)
            # Pattern: ): ReturnType { or ): ReturnType =>
            ret_match = re.search(r"\)\s*:\s*([^{=]+?)(?:\s*[{=]|$)", full_sig)
            if ret_match:
                return_type = ret_match.group(1).strip()
                if len(return_type) > 50:
                    return_type = return_type[:47] + "..."

            # Extract parameters
            paren_match = re.search(r"\(([^)]*)\)", full_sig)
            if paren_match:
                params_str = paren_match.group(1).strip()
                if params_str:
                    # Split on comma, respecting nested generics
                    params = []
                    current = []
                    depth = 0
                    for c in params_str:
                        if c in "<([{":
                            depth += 1
                        elif c in ">)]}":
                            depth -= 1
                        elif c == "," and depth == 0:
                            params.append("".join(current).strip())
                            current = []
                            continue
                        current.append(c)
                    if current:
                        params.append("".join(current).strip())

                    for param in params:
                        param = param.strip()
                        if not param:
                            continue
                        # Handle name: Type pattern
                        if ":" in param:
                            parts = param.split(":", 1)
                            if len(parts) == 2:
                                type_part = parts[1].strip()
                                # Remove default values
                                if "=" in type_part:
                                    type_part = type_part.split("=")[0].strip()
                                if len(type_part) > 40:
                                    type_part = type_part[:37] + "..."
                                param_types.append(type_part)
                        else:
                            param_types.append("any")

        except (IndexError, ValueError):
            pass

        return return_type, param_types

    def _build_ts_signature(self, name: str, param_types: list, return_type: str) -> str:
        """Build a TypeScript-style signature string.

        Returns: "function name(p1: T1, p2: T2): ReturnType"
        """
        if param_types:
            # Create param list with generic names if we only have types
            params = ", ".join(f"p{i}: {t}" for i, t in enumerate(param_types))
        else:
            params = ""
        return f"function {name}({params}): {return_type}"

    def _is_class_context(self, parent_id: str) -> bool:
        """Check if parent_id refers to a Class node (not Function/Method).

        Used to distinguish class getters/setters from object literal ones.
        """
        for node in self.builder.nodes:
            if node["id"] == parent_id:
                return node["type"] == "Class"
        return False

    def _extract_calls_with_ast_grep(self, file_path: str, file_id: str, lines: list):
        """Extract calls using ast-grep for higher accuracy.

        Uses the TypeScriptSymbolTable to filter type-only imports.
        SCIP handles actual symbol resolution for both Rust and TypeScript.
        """
        # Build a line-to-context map from our parsed structure
        # Uses span-size ordering to ensure innermost context wins for nested functions
        line_to_context = build_line_to_context_map(self.builder.nodes, file_id)

        for call in extract_calls_typescript(file_path):
            lineno = call["lineno"]
            callee = call["callee"]

            # Filter stdlib calls
            if is_non_project_call(callee):
                continue

            # Skip type-only imports (they don't generate runtime calls)
            if hasattr(self, 'symbol_table') and self.symbol_table.is_type_import(callee):
                continue

            context_id = line_to_context.get(lineno)
            if not context_id:
                for offset in range(1, 10):
                    context_id = line_to_context.get(
                        lineno - offset
                    ) or line_to_context.get(lineno + offset)
                    if context_id:
                        break

            # Fall back to file_id if no function context found (file-level calls)
            source_id = context_id if context_id else file_id
            self.builder.edges.append(
                {
                    "source": source_id,
                    "target": f"REF:{callee}",
                    "relation": "CALLS",
                    "metadata": {"lineno": lineno},
                }
            )


class CodeGraphBuilder:
    def __init__(self, root_dir, excluded_dirs=None):
        self.root_dir = Path(root_dir).resolve()
        self.excluded_dirs = excluded_dirs or {
            "target", "node_modules", "dist", "build", "docs",
            "coverage", "demo-output", ".beads",
        }
        self.nodes = []
        self.edges = []
        self.node_registry = set()
        self.current_file_id = None
        self.current_scope = SymbolTable()

    def add_node(self, node_id, node_type, **metadata):
        if node_id not in self.node_registry:
            self.nodes.append({"id": node_id, "type": node_type, "metadata": metadata})
            self.node_registry.add(node_id)

    def add_edge(self, source, target, relation, metadata=None):
        edge = {"source": source, "target": target, "relation": relation}
        if metadata:
            edge["metadata"] = metadata
        self.edges.append(edge)

    def get_rel_path(self, path):
        try:
            return str(path.relative_to(self.root_dir))
        except ValueError:
            return str(path)

    def process_directory(self):
        for root, dirs, files in os.walk(self.root_dir):
            dirs.sort()
            files.sort()
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in self.excluded_dirs
            ]
            for file in files:
                file_path = Path(root) / file
                rel_path = self.get_rel_path(file_path)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except OSError as e:
                    logger.warning(f"Cannot read {rel_path}: {e.strerror}")
                    continue
                except Exception as e:
                    logger.warning(f"Error reading {rel_path}: {e}")
                    continue
                if file.endswith(".py"):
                    PythonParser(self).parse(file_path, rel_path, content)
                elif file.endswith(".rs"):
                    RustParser(self).parse(file_path, rel_path, content)
                elif file.endswith(".proto"):
                    ProtoParser(self).parse(file_path, rel_path, content)
                elif (
                    file.endswith(".ts")
                    or file.endswith(".tsx")
                    or file.endswith(".js")
                ):
                    TSParser(self).parse(file_path, rel_path, content)
                elif file.endswith(".svelte"):
                    if "<script" in content:
                        parts = content.split("<script")
                        if len(parts) > 1:
                            idx = parts[1].find(">")
                            if idx != -1:
                                TSParser(self).parse(
                                    file_path,
                                    rel_path,
                                    parts[1][idx + 1 :].split("</script>")[0],
                                )

    def _get_language(self, node_id: str) -> str:
        """Extract language from node ID based on file extension."""
        # Node ID format: FILE:path/to/file.ext::Symbol
        if not node_id.startswith("FILE:"):
            return "unknown"
        file_part = node_id.split("::")[0].replace("FILE:", "")
        if file_part.endswith(".rs"):
            return "rust"
        elif file_part.endswith((".ts", ".tsx", ".js", ".jsx", ".svelte")):
            return "typescript"
        elif file_part.endswith(".py"):
            return "python"
        elif file_part.endswith(".proto"):
            return "proto"
        return "unknown"

    def resolve_references(self, scip_index=None):
        """Resolve REF: targets to actual node IDs with smart matching.

        Resolution priorities:
        0. SCIP lookup (highest accuracy, type-aware)
        1. Same language (required for regex fallback)
        2. Same struct/class for self.method patterns
        3. Qualified name match (Type::method)
        4. Same file
        5. Any same-language match

        Args:
            scip_index: Optional ScipIndex for type-aware resolution
        """
        # Track resolution statistics
        stats = {"scip": 0, "regex": 0, "self_type": 0, "unresolved": 0}

        # Build lookup tables for regex fallback
        # name -> [(node_id, language), ...]
        node_lookup = {}
        # qualified_name -> [(node_id, language), ...]  (e.g., "Struct::method")
        qualified_lookup = {}

        for node in self.nodes:
            nid = node["id"]
            parts = nid.split("::")
            if len(parts) >= 2:
                name_part = parts[-1]
                lang = self._get_language(nid)

                # Simple name lookup
                if name_part not in node_lookup:
                    node_lookup[name_part] = []
                node_lookup[name_part].append((nid, lang))

                # Qualified name lookup (Struct::method)
                if len(parts) >= 3:
                    # parts: ['FILE:path/file.rs', 'Struct', 'method']
                    qualified = f"{parts[-2]}::{parts[-1]}"
                    if qualified not in qualified_lookup:
                        qualified_lookup[qualified] = []
                    qualified_lookup[qualified].append((nid, lang))

        # Resolve edges with smart matching
        edges_to_remove = set()
        for idx, edge in enumerate(self.edges):
            target = edge["target"]
            if target.startswith("REF:"):
                ref_name = target.replace("REF:", "")
                source_id = edge["source"]
                source_lang = self._get_language(source_id)

                resolved = None
                resolution_source = None

                # Pre-filter: skip non-project refs (stdlib/library) that slipped through parsing
                # These will never resolve and shouldn't count as unresolved
                if is_non_project_call(ref_name):
                    edges_to_remove.add(idx)
                    continue

                # Strategy 0: SCIP lookup (highest accuracy, supports Rust and TypeScript)
                if scip_index and source_lang in ("rust", "typescript"):
                    # Get source file and line for SCIP lookup
                    source_file = source_id.split("::")[0].replace("FILE:", "")
                    lineno = edge.get("metadata", {}).get("lineno", 0)
                    scip_resolved = scip_index.resolve(ref_name, source_file, lineno)
                    if scip_resolved:
                        # Validate cross-crate resolutions to prevent false positives
                        # Extract crate names from paths (first directory component)
                        source_crate = source_file.split("/")[0] if "/" in source_file else ""
                        resolved_file = scip_resolved.split("::")[0].replace("FILE:", "")
                        target_crate = resolved_file.split("/")[0] if "/" in resolved_file else ""

                        # If cross-crate, verify ref_name looks related to the target
                        # This prevents spurious matches like EnvFilter -> AuthInterceptor
                        if source_crate and target_crate and source_crate != target_crate:
                            # Extract the target symbol name
                            target_parts = scip_resolved.split("::")
                            target_name = target_parts[-1] if target_parts else ""

                            # Only accept if ref_name contains the target name or vice versa
                            if target_name and target_name.lower() in ref_name.lower():
                                resolved = scip_resolved
                                resolution_source = "scip"
                            # else: Skip this cross-crate match as likely false positive
                        else:
                            # Same crate or can't determine - accept the resolution
                            resolved = scip_resolved
                            resolution_source = "scip"

                # Normalize crate:: prefix (Rust-specific)
                # crate::path::Symbol -> path::Symbol for matching
                if ref_name.startswith("crate::"):
                    ref_name = ref_name[7:]  # Remove "crate::"

                # Strip trailing method chains for resolution
                # Handles: Type::new(&arg).unwrap -> Type::new
                #          chrono::Utc::now().naive_utc -> chrono::Utc::now
                #          TempDir::new().unwrap -> TempDir::new
                base_ref = ref_name
                if "::" in ref_name:
                    # Match multi-segment qualified names (A::B or A::B::C etc.)
                    # stopping before ( or . that follows the last segment
                    chain_match = re.match(
                        r'^((?:[A-Za-z_][A-Za-z0-9_]*::)+[A-Za-z_][A-Za-z0-9_]*)', ref_name
                    )
                    if chain_match:
                        base_ref = chain_match.group(1)
                        # For lookup, also try just the last two segments
                        # chrono::Utc::now -> Utc::now (which matches qualified_lookup)
                        segments = base_ref.split("::")
                        if len(segments) > 2:
                            base_ref_short = f"{segments[-2]}::{segments[-1]}"
                            if base_ref_short in qualified_lookup:
                                base_ref = base_ref_short

                # Strategy 1: Try qualified name match (Type::method or Type::Variant)
                # Use base_ref which has trailing method chains stripped
                if not resolved and "::" in base_ref and base_ref in qualified_lookup:
                    matches = [m for m in qualified_lookup[base_ref] if m[1] == source_lang]
                    if matches:
                        resolved = matches[0][0]
                        resolution_source = "regex"

                # Strategy 1.5: Resolve Self::method to StructName::method
                # Self:: refers to the concrete struct of the enclosing impl block.
                # We infer the struct name from the source node ID (FILE:path::Struct::method).
                if not resolved and base_ref.startswith("Self::"):
                    self_method = base_ref.replace("Self::", "", 1)
                    source_parts = source_id.split("::")
                    if len(source_parts) >= 3:
                        source_struct = source_parts[-2]
                        qualified_self = f"{source_struct}::{self_method}"
                        if qualified_self in qualified_lookup:
                            matches = [m for m in qualified_lookup[qualified_self] if m[1] == source_lang]
                            if matches:
                                resolved = matches[0][0]
                                resolution_source = "self_type"
                        # Also try via node_lookup with struct preference
                        if not resolved and self_method in node_lookup:
                            matches = [m for m in node_lookup[self_method] if m[1] == source_lang]
                            struct_matches = [
                                m for m in matches
                                if f"::{source_struct}::{self_method}" in m[0]
                            ]
                            if struct_matches:
                                resolved = struct_matches[0][0]
                                resolution_source = "self_type"

                # Strategy 2: For self.method, prefer same-struct methods
                if not resolved and ref_name.startswith("self."):
                    method_name = ref_name.replace("self.", "")
                    if method_name in node_lookup:
                        matches = [m for m in node_lookup[method_name] if m[1] == source_lang]
                        if matches:
                            # Extract struct name from source (FILE:path::Struct::method)
                            source_parts = source_id.split("::")
                            if len(source_parts) >= 3:
                                source_struct = source_parts[-2]
                                # Prefer methods in the same struct
                                struct_matches = [
                                    m for m in matches
                                    if f"::{source_struct}::{method_name}" in m[0]
                                ]
                                if struct_matches:
                                    resolved = struct_matches[0][0]
                                else:
                                    resolved = matches[0][0]
                                resolution_source = "regex"

                # Strategy 3: Extract base name and try standard resolution
                if not resolved:
                    # Get base name (last part after . or ::)
                    base_name = ref_name.split(".")[-1].split("::")[-1]
                    if base_name in node_lookup:
                        matches = [m for m in node_lookup[base_name] if m[1] == source_lang]
                        if matches:
                            # Prefer same-file matches
                            source_file = source_id.split("::")[0]
                            same_file = [m for m in matches if m[0].startswith(source_file)]
                            if same_file:
                                resolved = same_file[0][0]
                            else:
                                resolved = matches[0][0]
                            resolution_source = "regex"

                if resolved:
                    edge["target"] = resolved
                    edge["metadata"] = edge.get("metadata", {})
                    edge["metadata"]["resolution_source"] = resolution_source
                    if edge["relation"] == "CALLS":
                        edge["relation"] = "CALLS_RESOLVED"
                        stats[resolution_source] += 1
                    elif edge["relation"] == "INSTANTIATES":
                        edge["relation"] = "INSTANTIATES_RESOLVED"
                elif edge["relation"] == "CALLS":
                    stats["unresolved"] += 1

        # Remove filtered non-project edges (stdlib + third-party library calls)
        if edges_to_remove:
            self.edges = [e for i, e in enumerate(self.edges) if i not in edges_to_remove]
            logger.info(f"Filtered {len(edges_to_remove)} non-project edges during resolution")

        # Log resolution statistics
        total = stats["scip"] + stats["regex"] + stats["self_type"] + stats["unresolved"]
        if total > 0:
            scip_pct = 100 * stats["scip"] / total if total else 0
            regex_pct = 100 * stats["regex"] / total if total else 0
            self_type_pct = 100 * stats["self_type"] / total if total else 0
            resolved_total = stats["scip"] + stats["regex"] + stats["self_type"]
            resolved_pct = 100 * resolved_total / total if total else 0
            logger.info(
                f"CALLS Resolution: {resolved_pct:.0f}% "
                f"(SCIP: {stats['scip']}/{scip_pct:.0f}%, "
                f"Regex: {stats['regex']}/{regex_pct:.0f}%, "
                f"Self: {stats['self_type']}/{self_type_pct:.0f}%, "
                f"Unresolved: {stats['unresolved']})"
            )

    def export(self, output_path, scip_index=None):
        """Export the graph to JSON.

        Args:
            output_path: Path to write the graph JSON
            scip_index: Optional ScipIndex for type-aware resolution
        """
        self.resolve_references(scip_index=scip_index)
        in_degree = {}
        for edge in self.edges:
            t = edge["target"]
            if t:
                in_degree[t] = in_degree.get(t, 0) + 1
        for node in self.nodes:
            node["metadata"]["in_degree"] = in_degree.get(node["id"], 0)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"nodes": self.nodes, "edges": self.edges}, f, indent=2)
        print(
            f"Graph exported to {output_path}. Stats: {len(self.nodes)} nodes, {len(self.edges)} edges"
        )


def main():
    """CLI entry point for descry-generate."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate codebase knowledge graph")
    parser.add_argument(
        "path", nargs="?", default=".", help="Root path to index (default: .)"
    )
    parser.add_argument(
        "--no-scip",
        action="store_true",
        help="Disable SCIP resolution (use regex only)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    target = args.path

    # Load config from .descry.toml if present
    from descry.handlers import DescryConfig
    config = DescryConfig(project_root=Path(target).resolve())
    toml_data = DescryConfig._load_toml(config.project_root)
    config._apply_toml(toml_data)
    config_excluded_dirs = config.excluded_dirs if toml_data else None

    # Handle SCIP opt-out
    if args.no_scip:
        os.environ["DESCRY_NO_SCIP"] = "1"

    # Create cache directory if it doesn't exist
    cache_dir = Path(".descry_cache")
    cache_dir.mkdir(exist_ok=True)

    # Build the graph
    builder = CodeGraphBuilder(target, excluded_dirs=config_excluded_dirs)
    builder.process_directory()

    # Generate SCIP indices if available
    scip_index = None
    if SCIP_SUPPORT_LOADED and scip_available():
        try:
            scip_status = get_scip_status()
            indexers = scip_status.get("indexers", {})
            enabled_indexers = [
                name for name, info in indexers.items() if info.get("available")
            ]
            logger.info(f"SCIP: Enabled ({', '.join(enabled_indexers) or 'none'})")

            cache_manager = ScipCacheManager(
                Path(target).resolve(),
                scip_extra_args=config.scip_extra_args,
                scip_skip_crates=config.scip_skip_crates,
                scip_toolchain=config.scip_rust_toolchain,
            )
            # Generate SCIP for all supported languages (Rust and TypeScript)
            scip_files = list(cache_manager.update_all().values())

            if scip_files:
                scip_index = ScipIndex(scip_files)
                stats = scip_index.get_stats()
                logger.info(
                    f"SCIP: Loaded {stats['definitions']} definitions, "
                    f"{stats['unique_names']} unique names from {len(scip_files)} files"
                )
        except Exception as e:
            logger.warning(f"SCIP: Failed to load ({e}), using regex only")
    elif SCIP_SUPPORT_LOADED:
        status = get_scip_status()
        if status.get("disabled_by_env"):
            logger.info("SCIP: Disabled (DESCRY_NO_SCIP=1)")
        else:
            logger.info("SCIP: Unavailable (no indexers found: install rust-analyzer and/or scip-typescript)")

    # Export with optional SCIP resolution
    graph_path = cache_dir / "codebase_graph.json"
    builder.export(str(graph_path), scip_index=scip_index)

    # Generate embeddings for semantic search (if dependencies available)
    no_embeddings = os.environ.get("DESCRY_NO_EMBEDDINGS", "")
    if no_embeddings.lower() not in ("1", "true", "yes"):
        try:
            from descry.embeddings import embeddings_available, SemanticSearcher
            if embeddings_available():
                logger.info("Generating embeddings for semantic search...")
                searcher = SemanticSearcher(str(graph_path), force_rebuild=True)
                logger.info(f"Embeddings generated: {len(searcher.nodes)} nodes indexed")
            else:
                logger.debug("Embeddings: sentence-transformers not available, skipping")
        except ImportError:
            logger.debug("Embeddings: module not available, skipping")
        except Exception as e:
            logger.warning(f"Embeddings: Failed to generate ({e})")


if __name__ == "__main__":
    main()
