"""Generate codebase knowledge graph with optional SCIP-based type-aware resolution."""

import ast
import json
import logging
import os
import posixpath
import re
from collections import deque
from pathlib import Path

from descry._comments import strip_block_comments, strip_line_comment

# Configure logging for consistent error reporting
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Refuse to index individual source files above this size. Matches the
# /api/source cap in web/server.py so the indexer and viewer share a budget.
_MAX_SOURCE_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB

_JSTS_EXTENSIONS = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
_RUBY_EXTENSIONS = (".rb", ".rake", ".gemspec")
_DOTNET_EXTENSIONS = (".cs", ".vb")
_CLANG_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx")
# Manifests read for the package names call resolution needs. They do not
# become graph nodes.
_MANIFESTS = ("go.mod", "Cargo.toml", "package.json")
_GO_MODULE_RE = re.compile(r"^module\s+(\S+)", re.MULTILINE)
_CARGO_PACKAGE_RE = re.compile(
    r'^\[package\][^\[]*?^name\s*=\s*"([^"]+)"', re.MULTILINE | re.DOTALL
)
# A TypeScript or JavaScript project file, read for its import path aliases
_TSCONFIG_RE = re.compile(r"[jt]sconfig(?:\.[\w.-]+)?\.json")
# What a tsconfig may hold that JSON may not, next to the strings to keep:
# comments, then commas before a closing bracket.
_JSON_STRING = r'"(?:[^"\\]|\\.)*"'
_JSONC_COMMENT_RE = re.compile(_JSON_STRING + r"|//[^\n]*|/\*.*?\*/", re.DOTALL)
_JSONC_COMMA_RE = re.compile(_JSON_STRING + r"|,(?=\s*[}\]])")


def _loads_jsonc(text: str):
    """Parse JSON that may have comments and trailing commas."""
    for extra in (_JSONC_COMMENT_RE, _JSONC_COMMA_RE):
        text = extra.sub(lambda m: m.group() if m.group()[0] == '"' else "", text)
    return json.loads(text)


# Extensions the indexer has a parser for. Besides these it opens only the
# manifests in _MANIFESTS and the project files _TSCONFIG_RE names.
_INDEXED_EXTENSIONS = (
    ".py",
    ".rs",
    ".proto",
    ".java",
    ".go",
    *_RUBY_EXTENSIONS,
    ".php",
    *_DOTNET_EXTENSIONS,
    *_CLANG_EXTENSIONS,
    ".dart",
    ".svelte",
    *_JSTS_EXTENSIONS,
)

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
    from descry.scip.cache import ScipCacheManager
    from descry.scip.parser import ScipIndex
    from descry.scip.support import get_scip_status, scip_available

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
        # ── clap (ubiquitous Rust CLI parsing crate) ──
        "value_name",
        "help_heading",
        "help_template",
        "long_help",
        "short_flag",
        "long_flag",
        "visible_alias",
        "visible_aliases",
        "next_line_help",
        "hide_possible_values",
        "hide_default_value",
        "hide_long_help",
        "value_parser",
        "value_delimiter",
        "num_args",
        "action",
        "require_equals",
        "allow_hyphen_values",
        "multiple_values",
        "conflicts_with",
        "conflicts_with_all",
        "requires",
        "requires_all",
        "requires_if",
        "required_if_eq",
        "required_if_eq_any",
        "required_if_eq_all",
        "required_unless_present",
        "required_unless_present_any",
        "required_unless_present_all",
        "overrides_with",
        "overrides_with_all",
        "default_value",
        "default_value_if",
        "default_value_ifs",
        "default_missing_value",
        "default_missing_values",
        "env",
        "hide",
        "hide_short_help",
        "global",
        "subcommand_required",
        "arg_required_else_help",
        "subcommand_precedence_over_arg",
        "allow_external_subcommands",
        "args_conflicts_with_subcommands",
        "color",
        "styles",
        "bin_name",
        "propagate_version",
        "disable_help_subcommand",
        "disable_help_flag",
        "disable_version_flag",
        "max_term_width",
        "term_width",
        "mut_arg",
        "mut_group",
        "next_help_heading",
        "before_help",
        "after_help",
        "before_long_help",
        "after_long_help",
        "version",
        "long_version",
        "about",
        "long_about",
        "author",
        "display_order",
        "get_one",
        "get_many",
        "get_flag",
        "get_count",
        "get_raw",
        "get_occurrences",
        "contains_id",
        "value_source",
        "index_of",
        "indices_of",
        "subcommand_name",
        "remove_one",
        "remove_many",
        "present",
        # ── snapbox (clap-ecosystem test crate) ──
        "stdout_eq",
        "stderr_eq",
        "subset_matches",
        "subset_matches_path",
        "assert_ui",
        "cmd",
        "assert_data_eq",
        "str",
        "file",
        "is_json",
        "is_yaml",
        "is_toml",
        "unordered",
        # ── Rust numeric arithmetic / slice / pointer methods ──
        "wrapping_add",
        "wrapping_sub",
        "wrapping_mul",
        "wrapping_div",
        "wrapping_rem",
        "wrapping_neg",
        "wrapping_shl",
        "wrapping_shr",
        "wrapping_pow",
        "saturating_add",
        "saturating_sub",
        "saturating_mul",
        "saturating_div",
        "saturating_neg",
        "saturating_pow",
        "checked_add",
        "checked_sub",
        "checked_mul",
        "checked_div",
        "checked_rem",
        "checked_neg",
        "checked_shl",
        "checked_shr",
        "checked_pow",
        "overflowing_add",
        "overflowing_sub",
        "overflowing_mul",
        "leading_zeros",
        "trailing_zeros",
        "count_ones",
        "count_zeros",
        "rotate_left",
        "rotate_right",
        "pow",
        "abs",
        "signum",
        "to_le_bytes",
        "to_be_bytes",
        "to_ne_bytes",
        "from_le_bytes",
        "from_be_bytes",
        "from_ne_bytes",
        "compare_exchange",
        "compare_exchange_weak",
        "fetch_sub",
        "fetch_and",
        "fetch_or",
        "fetch_xor",
        "fetch_nand",
        "fetch_max",
        "fetch_min",
        "fetch_update",
        "load",
        "store",
        "to_vec",
        "into_vec",
        "as_ptr",
        "as_mut_ptr",
        "as_non_null_ptr",
        "cast",
        "add",
        "offset",
        "offset_from",
        "wrapping_offset",
        "read_volatile",
        "write_volatile",
        "read_unaligned",
        "write_unaligned",
        "copy_to",
        "copy_from",
        "copy_nonoverlapping",
        "is_null",
        "is_aligned",
        "raw_os_error",
        "last_os_error",
        "from_raw",
        "into_raw",
        "into_raw_parts",
        "from_raw_parts",
        "from_raw_parts_mut",
        "is_ascii_digit",
        "is_ascii_alphabetic",
        "is_ascii_alphanumeric",
        "is_ascii_whitespace",
        "is_ascii_uppercase",
        "is_ascii_lowercase",
        "is_ascii_hexdigit",
        "is_ascii_punctuation",
        "is_ascii_graphic",
        "is_ascii_control",
        "is_digit",
        "is_alphabetic",
        "is_alphanumeric",
        "is_whitespace",
        "is_uppercase",
        "is_lowercase",
        "is_control",
        "make_ascii_uppercase",
        "make_ascii_lowercase",
        "to_ascii_uppercase",
        "to_ascii_lowercase",
        "split_whitespace",
        "split_ascii_whitespace",
        "lines",
        "bytes",
        "chars",
        "char_indices",
        "from_raw_fd",
        "as_raw_fd",
        "into_raw_fd",
        "borrow_raw",
        "copy_from_slice",
        "is_ascii",
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
        "shift",
        "unshift",
        "slice",
        "splice",
        "concat",
        "reduce",
        "forEach",
        "findIndex",
        "some",
        "every",
        "join",
        "split",
        "toLowerCase",
        "toUpperCase",
        "stringify",
        "parse",
        "toFixed",
        "toPrecision",
        "toExponential",
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
        # ── TypeScript tslib emit helpers (generated by tsc) ──
        "__awaiter",
        "__generator",
        "__await",
        "__asyncGenerator",
        "__asyncValues",
        "__asyncDelegator",
        "__addDisposableResource",
        "__disposeResources",
        "__rest",
        "__spreadArray",
        "__spreadArrays",
        "__assign",
        "__extends",
        "__decorate",
        "__param",
        "__metadata",
        "__exportStar",
        "__importDefault",
        "__importStar",
        "__createBinding",
        "__values",
        "__read",
        "__makeTemplateObject",
        "__classPrivateFieldGet",
        "__classPrivateFieldSet",
        "__classPrivateFieldIn",
        "__setFunctionName",
        "__runInitializers",
        "__esDecorate",
        "__propKey",
        # ── JS runtime / CommonJS ──
        "require",
        "module",
        "exports",
        "globalThis",
        "window",
        "document",
        "navigator",
        "location",
        "history",
        "screen",
        "alert",
        "confirm",
        "prompt",
        "fetch",
        "Response",
        "Request",
        "Headers",
        "URL",
        "URLSearchParams",
        "FormData",
        "Blob",
        "File",
        "FileReader",
        "AbortController",
        "AbortSignal",
        "WeakMap",
        "WeakSet",
        "WeakRef",
        "Symbol",
        "Proxy",
        "Reflect",
        "BigInt",
        "Number",
        "Boolean",
        "Error",
        "TypeError",
        "RangeError",
        "SyntaxError",
        "ReferenceError",
        "EvalError",
        "URIError",
        "AggregateError",
        "encodeURIComponent",
        "decodeURIComponent",
        "encodeURI",
        "decodeURI",
        "structuredClone",
        "queueMicrotask",
        "setImmediate",
        "clearImmediate",
        "performance",
        "crypto",
        "atob",
        "btoa",
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
        "textContent",
        "innerText",
        "innerHTML",
        "outerHTML",
        "getAttribute",
        "setAttribute",
        "removeAttribute",
        "hasAttribute",
        "getAttributeNames",
        "getBoundingClientRect",
        "getClientRects",
        "appendChild",
        "removeChild",
        "insertBefore",
        "replaceChild",
        "cloneNode",
        "closest",
        "classList",
        "className",
        "nodeName",
        "nodeType",
        "nodeValue",
        "parentNode",
        "parentElement",
        "childNodes",
        "children",
        "firstChild",
        "lastChild",
        "nextSibling",
        "previousSibling",
        "createElement",
        "createTextNode",
        "createDocumentFragment",
        "postMessage",
        # ── Array/Object static methods ──
        "isArray",
        "of",
        "entries",
        "values",
        "assign",
        "defineProperty",
        "defineProperties",
        "getPrototypeOf",
        "setPrototypeOf",
        "freeze",
        "isFrozen",
        "create",
        "getOwnPropertyNames",
        "getOwnPropertyDescriptor",
        "getOwnPropertyDescriptors",
        "getOwnPropertySymbols",
        "preventExtensions",
        "isExtensible",
        "seal",
        "isSealed",
        "fromEntries",
        "groupBy",
        # ── Function.prototype ──
        "bind",
        "apply",
        # ── Playwright (language-general JS/TS testing) ──
        "page.$",
        "page.$$",
        "page.textContent",
        "page.innerText",
        "page.innerHTML",
        "page.getAttribute",
        "page.waitForEvent",
        "page.waitForSelector",
        "page.waitForLoadState",
        "page.waitForURL",
        "page.waitForTimeout",
        "page.waitForResponse",
        "page.waitForRequest",
        "page.waitForNavigation",
        "page.waitForFunction",
        "page.evaluate",
        "page.evaluateHandle",
        "page.click",
        "page.dblclick",
        "page.fill",
        "page.type",
        "page.press",
        "page.keyboard",
        "page.mouse",
        "page.goto",
        "page.reload",
        "page.goBack",
        "page.goForward",
        "page.close",
        "page.screenshot",
        "page.setViewportSize",
        "page.setContent",
        "page.content",
        "page.title",
        "page.url",
        "page.locator",
        "page.getByRole",
        "page.getByText",
        "page.getByLabel",
        "page.getByPlaceholder",
        "page.getByTitle",
        "page.getByTestId",
        "page.getByAltText",
        "page.hover",
        "page.focus",
        "page.selectOption",
        "page.check",
        "page.uncheck",
        "page.setInputFiles",
        "page.dragAndDrop",
        "page.isVisible",
        "page.isEnabled",
        "page.isHidden",
        "page.isDisabled",
        "page.isChecked",
        "page.isEditable",
        "waitForEvent",
        "waitForSelector",
        "waitForLoadState",
        "waitForURL",
        "waitForTimeout",
        "waitForResponse",
        "waitForRequest",
        "waitForNavigation",
        "waitForFunction",
        # ── Vitest / Jest conditional + snapshot ──
        "runIf",
        "skipIf",
        "concurrent",
        "sequential",
        "expectSnapshot",
        "toMatchSnapshot",
        "toMatchFileSnapshot",
        "toThrowErrorMatchingSnapshot",
        "toThrowErrorMatchingInlineSnapshot",
        "assertions",
        # ── Jest / Vitest todo / skip / only / each modifiers ──
        "todo",
        "it.todo",
        "it.skip",
        "it.only",
        "it.each",
        "it.concurrent",
        "it.failing",
        "test.todo",
        "test.only",
        "test.each",
        "test.concurrent",
        "test.failing",
        "describe.todo",
        "describe.skip",
        "describe.only",
        "describe.each",
        "describe.concurrent",
        "describe.sequential",
        "describe.failing",
        "toHaveBeenCalledWith",
        "toHaveBeenCalledTimes",
        "toHaveBeenCalled",
        "toHaveBeenLastCalledWith",
        "toHaveBeenNthCalledWith",
        "toHaveReturned",
        "toHaveReturnedTimes",
        "toHaveReturnedWith",
        "toHaveLastReturnedWith",
        "toHaveNthReturnedWith",
        "toHaveProperty",
        "toBeCalledWith",
        "toBeCalledTimes",
        "toBeCalled",
        # ── React core hooks (language-general React pattern) ──
        "useState",
        "useEffect",
        "useLayoutEffect",
        "useInsertionEffect",
        "useCallback",
        "useMemo",
        "useRef",
        "useImperativeHandle",
        "useContext",
        "useReducer",
        "useDebugValue",
        "useId",
        "useDeferredValue",
        "useTransition",
        "useSyncExternalStore",
        "useEffectEvent",
        "useActionState",
        "useFormStatus",
        "useFormState",
        "useOptimistic",
        "createContext",
        "createRef",
        "createPortal",
        "forwardRef",
        "memo",
        "lazy",
        "Suspense",
        "Fragment",
        "StrictMode",
        "Profiler",
        "startTransition",
        # ── TanStack React Query (widely used React data-fetching lib) ──
        "useQuery",
        "useMutation",
        "useInfiniteQuery",
        "useQueryClient",
        "useQueries",
        "useSuspenseQuery",
        "useSuspenseQueries",
        "useIsFetching",
        "useIsMutating",
        "useMutationState",
        "QueryClient",
        "QueryClientProvider",
        "queryOptions",
        "mutationOptions",
        "invalidateQueries",
        "setQueryData",
        "getQueryData",
        "getQueriesData",
        "setQueriesData",
        "removeQueries",
        "resetQueries",
        "refetchQueries",
        "prefetchQuery",
        "prefetchInfiniteQuery",
        "ensureQueryData",
        "fetchQuery",
        "fetchInfiniteQuery",
        "cancelQueries",
        "isFetching",
        "isMutating",
        "mutate",
        "mutateAsync",
        # ── React Native / Expo APIs (broadly used in RN apps) ──
        "Alert.alert",
        "Alert.prompt",
        "Linking.openURL",
        "Linking.canOpenURL",
        "Linking.addEventListener",
        "Linking.getInitialURL",
        "StyleSheet.create",
        "StyleSheet.flatten",
        "StyleSheet.compose",
        "StyleSheet.hairlineWidth",
        "StyleSheet.absoluteFill",
        "StyleSheet.absoluteFillObject",
        "Platform.OS",
        "Platform.Version",
        "Platform.select",
        "Platform.isPad",
        "Platform.isTV",
        "Dimensions.get",
        "Dimensions.addEventListener",
        "Keyboard.dismiss",
        "Keyboard.addListener",
        "Keyboard.removeListener",
        "Notifications.setNotificationChannelAsync",
        "Notifications.getExpoPushTokenAsync",
        "Notifications.requestPermissionsAsync",
        "Notifications.getPermissionsAsync",
        "Notifications.scheduleNotificationAsync",
        "Notifications.cancelScheduledNotificationAsync",
        "Notifications.cancelAllScheduledNotificationsAsync",
        "Notifications.presentNotificationAsync",
        "Notifications.dismissNotificationAsync",
        "Notifications.dismissAllNotificationsAsync",
        "Notifications.setBadgeCountAsync",
        "Notifications.getBadgeCountAsync",
        "Notifications.setNotificationHandler",
        "Notifications.addNotificationReceivedListener",
        "Notifications.addNotificationResponseReceivedListener",
        "Notifications.removeNotificationSubscription",
        "setNotificationChannelAsync",
        "getExpoPushTokenAsync",
        "requestPermissionsAsync",
        "getPermissionsAsync",
        "scheduleNotificationAsync",
        "setNotificationHandler",
        "addNotificationReceivedListener",
        "addNotificationResponseReceivedListener",
        "removeNotificationSubscription",
        # ── Node assert (strict + deep variants) ──
        "strictEqual",
        "notStrictEqual",
        "deepStrictEqual",
        "notDeepStrictEqual",
        "deepEqual",
        "notDeepEqual",
        "ifError",
        "rejects",
        "doesNotReject",
        "throws",
        "doesNotThrow",
        # ── Lingui i18n (broadly used react-i18n) ──
        "useLingui",
        "loadAndActivate",
        "activate",
        "Trans",
        "plural",
        "select",
        "selectOrdinal",
        "setLocale",
        # ── nanoid (ubiquitous npm ID generator) ──
        "nanoid",
        "customAlphabet",
        "urlAlphabet",
        # ── uuid (ubiquitous npm ID generator) ──
        # Use the qualified form; `v4`, `validate`, `stringify` alone
        # are far too generic to filter globally.
        "uuid.v1",
        "uuid.v3",
        "uuid.v4",
        "uuid.v5",
        "uuid.v6",
        "uuid.v7",
        "uuid.validate",
        "uuid.stringify",
        "uuid.parse",
        "uuid.version",
        "uuid.NIL",
        "uuid.MAX",
        # ── date-fns (ubiquitous npm date library) — only include
        # names that are date-fns-distinctive; avoid generic names like
        # format/parse/isValid/getDate that collide across ecosystems. ──
        "addDays",
        "subDays",
        "addHours",
        "subHours",
        "addMinutes",
        "subMinutes",
        "addSeconds",
        "subSeconds",
        "addMilliseconds",
        "subMilliseconds",
        "addWeeks",
        "subWeeks",
        "addMonths",
        "subMonths",
        "addYears",
        "subYears",
        "addQuarters",
        "subQuarters",
        "differenceInDays",
        "differenceInHours",
        "differenceInMinutes",
        "differenceInSeconds",
        "differenceInMilliseconds",
        "differenceInWeeks",
        "differenceInMonths",
        "differenceInYears",
        "differenceInQuarters",
        "formatDistance",
        "formatDistanceToNow",
        "formatDistanceStrict",
        "formatRelative",
        "parseISO",
        "parseJSON",
        "startOfDay",
        "endOfDay",
        "startOfWeek",
        "endOfWeek",
        "startOfMonth",
        "endOfMonth",
        "startOfYear",
        "endOfYear",
        "startOfHour",
        "endOfHour",
        "startOfMinute",
        "endOfMinute",
        "startOfSecond",
        "endOfSecond",
        "isBefore",
        "isAfter",
        "isSameDay",
        "isSameWeek",
        "isSameMonth",
        "isSameYear",
        "isSameHour",
        "isSameMinute",
        "isSameSecond",
        "isToday",
        "isYesterday",
        "isTomorrow",
        "isFuture",
        "isPast",
        # ── Python built-ins ──
        "range",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "tuple",
        "append",
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
        "lower",
    ]
)

# --- 2. Third-party library / framework names ---
# Methods and identifiers from external dependencies that will never resolve
# to project nodes.
_LIBRARY_NAMES = frozenset(
    [
        # ── Java stdlib: common classes called bare (unqualified) ──
        "AtomicInteger",
        "AtomicLong",
        "AtomicBoolean",
        "AtomicReference",
        "Collections",
        "Arrays",
        "Objects",
        "Optional",
        "List",
        "ArrayList",
        "LinkedList",
        "TreeMap",
        "TreeSet",
        "LinkedHashMap",
        "LinkedHashSet",
        "ConcurrentHashMap",
        "CopyOnWriteArrayList",
        "Thread",
        "ThreadLocal",
        "ThreadFactory",
        "Runnable",
        "Callable",
        "Executor",
        "Executors",
        "ExecutorService",
        "ScheduledExecutorService",
        "CompletableFuture",
        "Future",
        "CountDownLatch",
        "CyclicBarrier",
        "Semaphore",
        "Phaser",
        "ReentrantLock",
        "ReentrantReadWriteLock",
        "ReadWriteLock",
        "Lock",
        "Condition",
        "Stream",
        "IntStream",
        "LongStream",
        "DoubleStream",
        "Collectors",
        "Files",
        "Paths",
        "Path",
        "InputStream",
        "OutputStream",
        "Reader",
        "Writer",
        "BufferedReader",
        "BufferedWriter",
        "ByteArrayInputStream",
        "ByteArrayOutputStream",
        "DataInputStream",
        "DataOutputStream",
        "PrintStream",
        "PrintWriter",
        "FileInputStream",
        "FileOutputStream",
        "FileWriter",
        "InputStreamReader",
        "OutputStreamWriter",
        "ByteBuffer",
        "CharBuffer",
        "Charset",
        "StandardCharsets",
        "StringBuilder",
        "StringBuffer",
        "StringJoiner",
        "Integer",
        "Long",
        "Double",
        "Float",
        "Byte",
        "Short",
        "Character",
        "BigInteger",
        "BigDecimal",
        "System",
        "Runtime",
        "Class",
        "Exception",
        "RuntimeException",
        "IllegalArgumentException",
        "IllegalStateException",
        "NullPointerException",
        "UnsupportedOperationException",
        "ClassCastException",
        "IOException",
        "FileNotFoundException",
        "InterruptedException",
        "NumberFormatException",
        "IndexOutOfBoundsException",
        "ArrayIndexOutOfBoundsException",
        "NoSuchElementException",
        "ConcurrentModificationException",
        "Throwable",
        "LocalDate",
        "LocalDateTime",
        "LocalTime",
        "ZonedDateTime",
        "OffsetDateTime",
        "Period",
        "ChronoUnit",
        "TimeUnit",
        "Locale",
        "UUID",
        "Random",
        "SecureRandom",
        "URI",
        "URLEncoder",
        "URLDecoder",
        "Pattern",
        "Matcher",
        "Base64",
        "Comparator",
        "Iterable",
        "Iterator",
        "ListIterator",
        "Entry",
        "Enum",
        "Annotation",
        "Properties",
        "Scanner",
        # ── Java stdlib static-import method names (bare form) ──
        "asList",
        "emptyList",
        "emptyMap",
        "emptySet",
        "singleton",
        "singletonList",
        "singletonMap",
        "unmodifiableList",
        "unmodifiableMap",
        "unmodifiableSet",
        "requireNonNull",
        "requireNonNullElse",
        "nonNull",
        "isNull",
        "hashCode",
        "equals",
        "ofMillis",
        "ofSeconds",
        "ofMinutes",
        "ofHours",
        "ofDays",
        "ofNanos",
        # ── Common Java getters / builtins ──
        "getClass",
        "getMessage",
        "getCause",
        "getStackTrace",
        "printStackTrace",
        "toMap",
        "toList",
        "toSet",
        "getName",
        "getSimpleName",
        "setName",
        "isDaemon",
        "setDaemon",
        # ── JSON annotations (never resolve to project symbols) ──
        "JsonProperty",
        "JsonIgnore",
        "JsonIgnoreProperties",
        "JsonCreator",
        "JsonFormat",
        "JsonInclude",
        "JsonSubTypes",
        "JsonTypeInfo",
        "JsonUnwrapped",
        "JsonValue",
        # ── Jackson classes ──
        "ObjectMapper",
        "JsonNode",
        "TypeReference",
        "JsonParser",
        "JsonGenerator",
        # ── Standard annotation markers ──
        "SuppressWarnings",
        "Deprecated",
        "Override",
        "FunctionalInterface",
        "SafeVarargs",
        "Retention",
        "Target",
        "Inherited",
        "Documented",
        # ── JMH benchmark annotations ──
        "Setup",
        "TearDown",
        "Benchmark",
        "Param",
        "State",
        "BenchmarkMode",
        "OutputTimeUnit",
        "Warmup",
        "Measurement",
        "Fork",
        "Threads",
        # ── C standard library functions ──
        "printf",
        "fprintf",
        "sprintf",
        "snprintf",
        "vprintf",
        "vfprintf",
        "vsprintf",
        "vsnprintf",
        "scanf",
        "fscanf",
        "sscanf",
        "getchar",
        "putchar",
        "puts",
        "gets",
        "fgets",
        "fgetc",
        "fputc",
        "fopen",
        "fclose",
        "fread",
        "fwrite",
        "fseek",
        "ftell",
        "rewind",
        "feof",
        "ferror",
        "fflush",
        "freopen",
        "setvbuf",
        "rename",
        "tmpfile",
        "tmpnam",
        "malloc",
        "calloc",
        "realloc",
        "free",
        "aligned_alloc",
        "memcpy",
        "memmove",
        "memset",
        "memcmp",
        "memchr",
        "strcpy",
        "strncpy",
        "strcat",
        "strncat",
        "strcmp",
        "strncmp",
        "strlen",
        "strnlen",
        "strchr",
        "strrchr",
        "strstr",
        "strtok",
        "strtok_r",
        "strdup",
        "strndup",
        "strerror",
        "strtol",
        "strtoll",
        "strtoul",
        "strtoull",
        "strtod",
        "strtof",
        "atoi",
        "atol",
        "atoll",
        "atof",
        "exit",
        "_exit",
        "atexit",
        "quick_exit",
        "at_quick_exit",
        "getenv",
        "putenv",
        "setenv",
        "unsetenv",
        "system",
        "errno",
        "perror",
        "isdigit",
        "isalpha",
        "isalnum",
        "isspace",
        "isupper",
        "islower",
        "isprint",
        "ispunct",
        "isxdigit",
        "iscntrl",
        "tolower",
        "toupper",
        "labs",
        "llabs",
        "div",
        "ldiv",
        "lldiv",
        "rand",
        "srand",
        "time",
        "clock",
        "difftime",
        "mktime",
        "asctime",
        "ctime",
        "gmtime",
        "localtime",
        "strftime",
        "qsort",
        "bsearch",
        "signal",
        "raise",
        "kill",
        "sigaction",
        "sigprocmask",
        "usleep",
        "nanosleep",
        "lseek",
        "pipe",
        "dup",
        "dup2",
        "fork",
        "exec",
        "execl",
        "execle",
        "execlp",
        "execv",
        "execve",
        "execvp",
        "wait",
        "waitpid",
        "getpid",
        "getppid",
        "getuid",
        "geteuid",
        "getgid",
        "getegid",
        "socket",
        "listen",
        "accept",
        "connect",
        "sendto",
        "recvfrom",
        "setsockopt",
        "getsockopt",
        "gethostbyname",
        "poll",
        "epoll_create",
        "epoll_ctl",
        "epoll_wait",
        "kqueue",
        "kevent",
        "mmap",
        "munmap",
        "mprotect",
        "pthread_create",
        "pthread_join",
        "pthread_mutex_init",
        "pthread_mutex_destroy",
        "pthread_mutex_lock",
        "pthread_mutex_unlock",
        "pthread_cond_init",
        "pthread_cond_destroy",
        "pthread_cond_wait",
        "pthread_cond_signal",
        "pthread_cond_broadcast",
        "pthread_rwlock_init",
        "pthread_rwlock_rdlock",
        "pthread_rwlock_wrlock",
        "pthread_rwlock_unlock",
        # C++ STL bare identifiers (often used via `using namespace std;`)
        "vector",
        "string",
        "string_view",
        "unordered_map",
        "unordered_set",
        "deque",
        "stack",
        "queue",
        "priority_queue",
        "array",
        "pair",
        "optional",
        "variant",
        "unique_ptr",
        "shared_ptr",
        "weak_ptr",
        "make_unique",
        "make_shared",
        "make_pair",
        "make_tuple",
        "move",
        "forward",
        "copy_n",
        "copy_if",
        "fill",
        "fill_n",
        "transform",
        "find_if",
        "find_if_not",
        "all_of",
        "any_of",
        "none_of",
        "count",
        "count_if",
        "equal",
        "mismatch",
        "stable_sort",
        "partial_sort",
        "is_sorted",
        "nth_element",
        "reverse",
        "rotate",
        "shuffle",
        "min_element",
        "max_element",
        "minmax_element",
        "accumulate",
        "iota",
        "adjacent_difference",
        "partial_sum",
        "inner_product",
        "ostream",
        "istream",
        "iostream",
        "stringstream",
        "ostringstream",
        "istringstream",
        "fstream",
        "ofstream",
        "ifstream",
        "cout",
        "cin",
        "cerr",
        "clog",
        "endl",
        "ends",
        "hex",
        "dec",
        "oct",
        "setw",
        "setprecision",
        "setfill",
        "thread",
        "mutex",
        "lock_guard",
        "unique_lock",
        "scoped_lock",
        "shared_lock",
        "condition_variable",
        "future",
        "promise",
        "chrono",
        "duration",
        "time_point",
        "system_clock",
        "steady_clock",
        "high_resolution_clock",
        "exception",
        "runtime_error",
        "logic_error",
        "invalid_argument",
        "out_of_range",
        "bad_alloc",
        "bad_cast",
        "nullopt",
        "nullptr_t",
        # ── .NET BCL classes accessed bare (via `using` of their namespace) ──
        "Console",
        "Convert",
        "Trace",
        "Task",
        "ValueTask",
        "Timer",
        "CancellationToken",
        "CancellationTokenSource",
        "Environment",
        "Dictionary",
        "SortedDictionary",
        "SortedSet",
        "Queue",
        "Stack",
        "CultureInfo",
        "CultureNotFoundException",
        "ReferenceEquals",
        "RequiresUnreferencedCode",
        "RequiresDynamicCode",
        "UnconditionalSuppressMessage",
        "DynamicallyAccessedMembers",
        "ConcurrentBag",
        "ConcurrentQueue",
        "ConcurrentDictionary",
        "ConcurrentStack",
        "IEnumerable",
        "IList",
        "IDictionary",
        "ICollection",
        "KeyValuePair",
        "Tuple",
        "ValueTuple",
        "StringComparison",
        "StringComparer",
        "StringSplitOptions",
        "Regex",
        "Match",
        "Capture",
        "Guid",
        "DateTime",
        "DateTimeOffset",
        "TimeSpan",
        "TimeOnly",
        "DateOnly",
        "TimeZoneInfo",
        "Stopwatch",
        "Directory",
        "FileInfo",
        "DirectoryInfo",
        "MemoryStream",
        "FileStream",
        "StreamReader",
        "StreamWriter",
        "BinaryReader",
        "BinaryWriter",
        "TextReader",
        "TextWriter",
        "Encoding",
        "UTF8Encoding",
        "JsonSerializer",
        "JsonConvert",
        "JsonDocument",
        "JsonElement",
        "JsonObject",
        "JsonArray",
        "Enumerable",
        "Queryable",
        "MathF",
        "Activator",
        "Type",
        "Assembly",
        "Delegate",
        "Action",
        "Func",
        "Predicate",
        "Expression",
        "Lazy",
        "Nullable",
        "ArgumentException",
        "ArgumentNullException",
        "ArgumentOutOfRangeException",
        "InvalidOperationException",
        "NotSupportedException",
        "NotImplementedException",
        "NullReferenceException",
        "IndexOutOfRangeException",
        "KeyNotFoundException",
        "FormatException",
        "OverflowException",
        "ObjectDisposedException",
        "DirectoryNotFoundException",
        "UnauthorizedAccessException",
        "OperationCanceledException",
        "TaskCanceledException",
        "AggregateException",
        "HttpClient",
        "HttpRequestMessage",
        "HttpResponseMessage",
        "HttpMethod",
        "HttpStatusCode",
        "Uri",
        "UriBuilder",
        "ILogger",
        "ILoggerFactory",
        "IServiceProvider",
        "IServiceCollection",
        "IConfiguration",
        "IOptions",
        # ── .NET more stdlib classes ──
        "StringWriter",
        "StringReader",
        "EqualityComparer",
        "Comparer",
        "BindingFlags",
        "FieldInfo",
        "PropertyInfo",
        "MethodInfo",
        "ConstructorInfo",
        "EventInfo",
        "ParameterInfo",
        "CustomAttributeData",
        "Attribute",
        "NotNullWhen",
        "MaybeNullWhen",
        "MemberNotNull",
        "MemberNotNullWhen",
        "NotNull",
        "MaybeNull",
        "DoesNotReturn",
        "DoesNotReturnIf",
        "DebuggerDisplay",
        "DebuggerHidden",
        "DebuggerStepThrough",
        "DebuggerNonUserCode",
        "Flags",
        "Serializable",
        "NonSerialized",
        "Obsolete",
        "ThreadStatic",
        "Conditional",
        "Category",
        "Description",
        "DisplayName",
        "DataContract",
        "DataMember",
        "JsonPropertyName",
        "JsonConstructor",
        "JsonConverter",
        # ── xUnit / NUnit / MSTest test framework ──
        "Assert",
        "Fact",
        "Theory",
        "InlineData",
        "MemberData",
        "ClassData",
        "Trait",
        "Skip",
        "TestMethod",
        "TestClass",
        "TestInitialize",
        "TestCleanup",
        "Test",
        "TestCase",
        "TestFixture",
        "SetUp",
        "OneTimeSetUp",
        "OneTimeTearDown",
        # ── Moq / NSubstitute ──
        "Mock",
        "It",
        "Returns",
        "ReturnsAsync",
        "Throws",
        "ThrowsAsync",
        "Callback",
        "Verify",
        "VerifyAll",
        "VerifyNoOtherCalls",
        "Substitute",
        "Received",
        "DidNotReceive",
        "Arg",
        # ── FluentAssertions ──
        "Should",
        "BeEquivalentTo",
        "BeOfType",
        "NotBeNull",
        "HaveCount",
        "ContainSingle",
        # ── PHP built-in functions (most common) ──
        "array_map",
        "array_filter",
        "array_reduce",
        "array_merge",
        "array_merge_recursive",
        "array_combine",
        "array_keys",
        "array_values",
        "array_flip",
        "array_reverse",
        "array_slice",
        "array_splice",
        "array_search",
        "array_unique",
        "array_diff",
        "array_intersect",
        "array_walk",
        "array_fill",
        "array_pad",
        "array_chunk",
        "array_column",
        "array_key_exists",
        "in_array",
        "sizeof",
        "is_array",
        "is_string",
        "is_int",
        "is_integer",
        "is_numeric",
        "is_float",
        "is_double",
        "is_bool",
        "is_object",
        "is_callable",
        "is_scalar",
        "is_countable",
        "is_iterable",
        "isset",
        "unset",
        "empty",
        # PHP stdlib globals / sys functions that commonly appear as
        # calls without a receiver and never resolve to project code.
        "stdClass",
        "get_debug_type",
        "get_class",
        "get_parent_class",
        "get_object_vars",
        "get_class_methods",
        "get_class_vars",
        "class_exists",
        "interface_exists",
        "trait_exists",
        "method_exists",
        "property_exists",
        "function_exists",
        "defined",
        "constant",
        "define",
        "sys_get_temp_dir",
        "sys_getloadavg",
        "stream_get_contents",
        "stream_get_line",
        "stream_get_meta_data",
        "stream_set_blocking",
        "stream_set_timeout",
        "stream_context_create",
        "stream_filter_append",
        "stream_filter_remove",
        "fputs",
        "flock",
        "file_exists",
        "file_get_contents",
        "file_put_contents",
        "is_link",
        "is_readable",
        "is_writable",
        "is_executable",
        "realpath",
        "dirname",
        "basename",
        "pathinfo",
        "tempnam",
        "chmod",
        "chown",
        "chgrp",
        "mkdir",
        "rmdir",
        "unlink",
        "filesize",
        "filetype",
        "fileatime",
        "filemtime",
        "filectime",
        "filegroup",
        "fileowner",
        "fileperms",
        "fileinode",
        "error_reporting",
        "ini_get",
        "ini_set",
        "set_error_handler",
        "restore_error_handler",
        "set_exception_handler",
        "restore_exception_handler",
        "trigger_error",
        "error_get_last",
        "microtime",
        "gmmktime",
        "date",
        "gmdate",
        "strtotime",
        "setlocale",
        "mb_strlen",
        "strpos",
        "mb_strpos",
        "strrpos",
        "mb_strrpos",
        "stripos",
        "str_replace",
        "str_ireplace",
        "str_contains",
        "str_starts_with",
        "str_ends_with",
        "str_repeat",
        "str_pad",
        "str_split",
        "str_word_count",
        "explode",
        "implode",
        "ltrim",
        "rtrim",
        "strtolower",
        "strtoupper",
        "ucfirst",
        "ucwords",
        "lcfirst",
        "mb_strtolower",
        "mb_strtoupper",
        "mb_substr",
        "substr_count",
        "substr_replace",
        "number_format",
        "ord",
        "chr",
        "htmlspecialchars",
        "htmlentities",
        "nl2br",
        "urlencode",
        "urldecode",
        "rawurlencode",
        "rawurldecode",
        "json_encode",
        "json_decode",
        "json_validate",
        "serialize",
        "unserialize",
        "base64_encode",
        "base64_decode",
        "md5",
        "sha1",
        "hash_hmac",
        "password_hash",
        "password_verify",
        "preg_match",
        "preg_match_all",
        "preg_replace",
        "preg_replace_callback",
        "preg_split",
        "preg_quote",
        "intval",
        "floatval",
        "strval",
        "boolval",
        "settype",
        "gettype",
        "enum_exists",
        "call_user_func",
        "call_user_func_array",
        "func_get_args",
        "func_num_args",
        "func_get_arg",
        "round",
        "sqrt",
        "rsort",
        "asort",
        "arsort",
        "ksort",
        "krsort",
        "usort",
        "uasort",
        "uksort",
        "natsort",
        "natcasesort",
        "array_push",
        "array_pop",
        "array_shift",
        "array_unshift",
        "current",
        "prev",
        "reset",
        "end",
        "each",
        "fgetcsv",
        "fputcsv",
        "is_writeable",
        "glob",
        "scandir",
        "date_create",
        "date_format",
        "checkdate",
        "getcwd",
        "chdir",
        "die",
        "var_dump",
        "print_r",
        "var_export",
        "debug_backtrace",
        "debug_print_backtrace",
        "phpversion",
        "php_sapi_name",
        "spl_autoload_register",
        "spl_object_hash",
        "spl_object_id",
        "iterator_to_array",
        "iterator_count",
        # ── More PHP builtins / Reflection ──
        "compact",
        "extract",
        "ReflectionClass",
        "ReflectionMethod",
        "ReflectionProperty",
        "ReflectionParameter",
        "ReflectionFunction",
        "ReflectionType",
        "ReflectionNamedType",
        "ReflectionUnionType",
        "ReflectionIntersectionType",
        "ReflectionEnum",
        "ReflectionAttribute",
        "Closure",
        "WeakReference",
        "ArrayObject",
        "ArrayIterator",
        "Generator",
        "IteratorAggregate",
        "Traversable",
        "Countable",
        "ArrayAccess",
        "Stringable",
        "ValueError",
        "LogicException",
        "InvalidArgumentException",
        "OutOfBoundsException",
        "OutOfRangeException",
        "UnexpectedValueException",
        "DomainException",
        "BadFunctionCallException",
        "BadMethodCallException",
        "DateTimeImmutable",
        "DateInterval",
        "DatePeriod",
        "DateTimeZone",
        "SplStack",
        "SplQueue",
        "SplDoublyLinkedList",
        "SplObjectStorage",
        "SplPriorityQueue",
        "SplHeap",
        "SplFixedArray",
        "SplFileInfo",
        "SplFileObject",
        # ── PHP binary / encoding / output stdlib ──
        "pack",
        "unpack",
        "bin2hex",
        "hex2bin",
        "curl_init",
        "curl_setopt",
        "curl_setopt_array",
        "curl_exec",
        "curl_close",
        "curl_getinfo",
        "curl_error",
        "curl_errno",
        "curl_multi_init",
        "curl_multi_add_handle",
        "curl_multi_exec",
        "curl_multi_remove_handle",
        "curl_multi_close",
        "curl_version",
        "ob_start",
        "ob_end_flush",
        "ob_end_clean",
        "ob_get_contents",
        "ob_get_clean",
        "ob_get_length",
        "ob_get_level",
        "ob_flush",
        "ob_clean",
        "ob_implicit_flush",
        "ob_list_handlers",
        "output_add_rewrite_var",
        "output_reset_rewrite_vars",
        "strspn",
        "strcspn",
        "strtr",
        "strrev",
        "wordwrap",
        "chunk_split",
        "quoted_printable_encode",
        "quoted_printable_decode",
        "addslashes",
        "stripslashes",
        "addcslashes",
        "stripcslashes",
        "quotemeta",
        "nl_langinfo",
        "money_format",
        "RangeException",
        "LengthException",
        "UnderflowException",
        # ── PHP DateTime / Carbon common methods ──
        "version_compare",
        "getTimestamp",
        "getOffset",
        "setTimestamp",
        "setTimezone",
        "getTimezone",
        "getLastErrors",
        "createFromFormat",
        "createFromImmutable",
        "createFromMutable",
        "createFromInterface",
        "diff",
        "modify",
        "setDate",
        "setTime",
        "setISODate",
        "setTimeUTC",
        "toDateTimeString",
        "toDateString",
        "toTimeString",
        "toAtomString",
        "toW3cString",
        "toRfc3339String",
        "toRfc2822String",
        "toDayDateTimeString",
        "toFormattedDateString",
        "toFormattedDayDateString",
        "toRfc822String",
        "toRfc850String",
        "toRfc1036String",
        "toRfc1123String",
        "isBirthday",
        "isSameAs",
        "isMonday",
        "isTuesday",
        "isWednesday",
        "isThursday",
        "isFriday",
        "isSaturday",
        "isSunday",
        "isWeekend",
        "isWeekday",
        "isStartOfDay",
        "isEndOfDay",
        "isStartOfWeek",
        "isEndOfWeek",
        "isLastDayOfMonth",
        "isFirstDayOfMonth",
        "isLeapYear",
        "isLongYear",
        "isDst",
        "isUtc",
        "isLocal",
        "isCurrentYear",
        "isCurrentMonth",
        "isCurrentDay",
        "isCurrentHour",
        "isCurrentMinute",
        "isCurrentSecond",
        "isNextYear",
        "isLastYear",
        "isNextMonth",
        "isLastMonth",
        "isNextWeek",
        "isLastWeek",
        "addDay",
        "subDay",
        "addHour",
        "subHour",
        "addMinute",
        "subMinute",
        "addSecond",
        "subSecond",
        "addMonth",
        "subMonth",
        "addYear",
        "subYear",
        "diffInDays",
        "diffInHours",
        "diffInMinutes",
        "diffInSeconds",
        "diffInMilliseconds",
        "diffInMicroseconds",
        "diffInWeeks",
        "diffInMonths",
        "diffInYears",
        "diffForHumans",
        "diffInDaysFiltered",
        "diffInHoursFiltered",
        "ageInDays",
        "ageInHours",
        "ageInMinutes",
        "ageInSeconds",
        "ageInYears",
        "average",
        "nowWithSameTz",
        "floatDiffInSeconds",
        "floatDiffInMinutes",
        "floatDiffInHours",
        "floatDiffInDays",
        # ── Python unittest.TestCase assertions (called bare on `self`). ──
        # The qualified form `self.assertEqual(...)` resolves by last_part
        # to these names; without them, every Django / stdlib-style test
        # file surfaces hundreds of unresolved REFs.
        "assertEqual",
        "assertNotEqual",
        "assertIs",
        "assertIsNot",
        "assertIsNone",
        "assertIsNotNone",
        "assertIn",
        "assertNotIn",
        "assertIsInstance",
        "assertNotIsInstance",
        "assertRaises",
        "assertRaisesRegex",
        "assertRaisesRegexp",
        "assertWarns",
        "assertWarnsRegex",
        "assertLogs",
        "assertNoLogs",
        "assertAlmostEqual",
        "assertNotAlmostEqual",
        "assertGreater",
        "assertGreaterEqual",
        "assertLess",
        "assertLessEqual",
        "assertRegex",
        "assertNotRegex",
        "assertCountEqual",
        "assertMultiLineEqual",
        "assertSequenceEqual",
        "assertListEqual",
        "assertTupleEqual",
        "assertSetEqual",
        "assertDictEqual",
        "assertDictContainsSubset",
        "subTest",
        "skipTest",
        "addCleanup",
        "doCleanups",
        "addClassCleanup",
        "doClassCleanups",
        "failureException",
        "longMessage",
        "maxDiff",
        "_callTestMethod",
        "_callSetUp",
        "_callTearDown",
        "shortDescription",
        "setUpClass",
        "tearDownClass",
        "addTypeEqualityFunc",
        # ── Go primitive type conversions: `int64(x)`, `float64(y)`,
        # `byte(z)` etc. syntactically look like calls to the regex but
        # are never project references. Filtering by the bare type name
        # catches them.
        "int8",
        "int16",
        "int32",
        "int64",
        "uint",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "uintptr",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "byte",
        "rune",
        # ── Protobuf runtime API (generated Go code). The receiver-called
        # forms `ms.StoreMessageInfo`, `ms.LoadMessageInfo`, etc. appear
        # in every .pb.go file that protoc-gen-go emits; the receiver
        # variable is always `ms` in the generated code. Filtering by
        # last-name catches them regardless of receiver variable name.
        "StoreMessageInfo",
        "LoadMessageInfo",
        "UnmarshalVT",
        "MarshalVT",
        "SizeVT",
        "CloneVT",
        "EqualVT",
        "StaticMethod",
        # ── Go testify library (`require.NoError`, `assert.NotNil` etc.) ──
        # The qualified `require.*` / `assert.*` forms are covered by
        # prefix filter below; keep the bare names here too so matches
        # like `NoError(...)` in testify-style DSL code drop out.
        "NoError",
        "NotNil",
        "Empty",
        "NotEmpty",
        "NotZero",
        "Zero",
        "ErrorIs",
        "ErrorAs",
        "ErrorContains",
        "ErrorMatches",
        "Never",
        "Eventually",
        "WithinDuration",
        "Subset",
        "ElementsMatch",
        "Implements",
        "Exactly",
        "Same",
        "NotSame",
        "PanicsWithValue",
        "PanicsWithError",
        "Panics",
        "NotPanics",
        # Go testing.T methods (`t.Helper`, `t.Parallel`, `t.Cleanup`, etc.).
        "Parallel",
        "TempDir",
        "Helper",
        "Cleanup",
        "Setenv",
        "Chdir",
        "Failed",
        "FailNow",
        "Skipped",
        "SkipNow",
        "Errorf",
        "Fatalf",
        "Logf",
        "Skipf",
        # Go testing.B / testing.F benchmark helpers. Skipping `Add`
        # and `Fuzz` (too generic — over-filters project code).
        "Loop",
        "ReportAllocs",
        "ResetTimer",
        "StartTimer",
        "StopTimer",
        "ReportMetric",
        # ── Rust clap (de-facto CLI parser, used by cargo, coreutils,
        # ripgrep, rustup — every serious Rust CLI). ArgMatches receiver
        # methods don't resolve via scip-clang because clap is an
        # external crate.
        "try_get_one",
        "try_get_many",
        "ids",
        # Rust rand crate (universal random number generator).
        "random_range",
        "gen_range",
        "gen_bool",
        "gen_ratio",
        "random_bool",
        "random_ratio",
        "random_iter",
        "sample",
        "sample_iter",
        "partial_shuffle",
        "choose",
        "choose_multiple",
        "choose_weighted",
        "try_fill",
        "from_entropy",
        "from_os_rng",
        "from_rng",
        "seed_from_u64",
        # Rust criterion/bencher benchmarking
        "bench",
        "bench_function",
        "bench_with_input",
        "benchmark_group",
        "criterion_group",
        "criterion_main",
        "measurement",
        "with_input",
        "with_inputs",
        "iter_custom",
        "iter_batched",
        "iter_batched_ref",
        "setup_wrappers_and_push",
        "black_box",
        "bench_values",
        "throughput",
        "sample_size",
        "measurement_time",
        "warm_up_time",
        "confidence_level",
        "significance_level",
        "noise_threshold",
        "plot_config",
        # Rust tempfile crate (widely used for tests).
        "tempdir",
        "tempfile",
        "persist",
        "persist_noclobber",
        # ── Rust std::fmt Formatter/DebugStruct API (receiver-called) ──
        "debug_struct",
        "debug_tuple",
        "debug_list",
        "debug_map",
        "debug_set",
        "write_str",
        "pad",
        "finish",
        "field",
        # ── Rust future/async surface ──
        "poll_fn",
        "from_waker",
        "noop_waker",
        # ── mockall ──
        "in_sequence",
        "expect_call",
        "returning",
        "times",
        # ── Criterion benchmark framework surface ──
        "measure",
        "configure_from_args",
        "iter_with_setup",
        "iter_with_large_drop",
        # ── PHPUnit assertions and mocking ──
        "assertTrue",
        "assertFalse",
        "assertNull",
        "assertNotNull",
        "assertEmpty",
        "assertNotEmpty",
        "assertCount",
        "assertNotCount",
        "assertContains",
        "assertNotContains",
        "assertContainsEquals",
        "assertStringContainsString",
        "assertFileContains",
        "assertFileDoesNotContain",
        "assertFileEqualsIgnoringCase",
        "assertFileNotEquals",
        "assertFileNotEqualsCanonicalizing",
        "assertFileNotEqualsIgnoringCase",
        "assertEqualsCanonicalizing",
        "assertEqualsIgnoringCase",
        "assertEqualsWithDelta",
        "assertNotEqualsCanonicalizing",
        "assertNotEqualsIgnoringCase",
        "assertNotEqualsWithDelta",
        "assertMatchesSnapshot",
        "assertDatabaseHas",
        "assertDatabaseMissing",
        "assertDatabaseCount",
        "assertSoftDeleted",
        "assertModelExists",
        "assertModelMissing",
        "assertStringNotContainsString",
        "assertStringStartsWith",
        "assertStringEndsWith",
        "assertStringMatchesFormat",
        "assertMatchesRegularExpression",
        "assertGreaterThan",
        "assertGreaterThanOrEqual",
        "assertLessThan",
        "assertLessThanOrEqual",
        "assertInstanceOf",
        "assertClassHasAttribute",
        "assertClassHasStaticAttribute",
        "assertObjectHasAttribute",
        "assertArrayHasKey",
        "assertArrayNotHasKey",
        "assertArraySubset",
        "assertJson",
        "assertJsonStringEqualsJsonString",
        "assertJsonStringEqualsJsonFile",
        "assertFileExists",
        "assertFileDoesNotExist",
        "assertDirectoryExists",
        "assertFileEquals",
        "assertSame",
        "assertNotSame",
        "assertIsArray",
        "assertIsNotArray",
        "assertIsBool",
        "assertIsNotBool",
        "assertIsCallable",
        "assertIsNotCallable",
        "assertIsFloat",
        "assertIsNotFloat",
        "assertIsInt",
        "assertIsNotInt",
        "assertIsNumeric",
        "assertIsNotNumeric",
        "assertIsObject",
        "assertIsNotObject",
        "assertIsResource",
        "assertIsNotResource",
        "assertIsScalar",
        "assertIsNotScalar",
        "assertIsString",
        "assertIsNotString",
        "assertIsIterable",
        "assertIsNotIterable",
        "assertIsClosedResource",
        "assertIsNotClosedResource",
        "assertIsReadable",
        "assertIsNotReadable",
        "assertIsWritable",
        "assertIsNotWritable",
        "assertContainsOnlyInstancesOf",
        "assertNotContainsEquals",
        "assertClassNotHasAttribute",
        "assertClassNotHasStaticAttribute",
        "assertObjectNotHasAttribute",
        "assertObjectHasProperty",
        "assertObjectNotHasProperty",
        "assertClassHasProperty",
        "assertClassNotHasProperty",
        "assertDoesNotMatchRegularExpression",
        "assertFinite",
        "assertInfinite",
        "assertNan",
        "assertStringNotMatchesFormat",
        "assertStringMatchesFormatFile",
        "assertStringNotMatchesFormatFile",
        "assertZero",
        "assertNotZero",
        "assertStringEqualsFile",
        "assertStringEqualsFileCanonicalizing",
        "assertStringEqualsFileIgnoringCase",
        "assertStringNotEqualsFile",
        "assertStringNotEqualsFileCanonicalizing",
        "assertStringNotEqualsFileIgnoringCase",
        "addToAssertionCount",
        "getCount",
        "resetCount",
        "PHPUnit::assertTrue",
        "PHPUnit::assertFalse",
        "PHPUnit::assertEquals",
        "PHPUnit::assertSame",
        "PHPUnit::assertNull",
        "PHPUnit::assertNotNull",
        "PHPUnit::assertInstanceOf",
        "PHPUnit::assertIsArray",
        "PHPUnit::assertIsString",
        "PHPUnit::assertIsInt",
        "PHPUnit::assertIsBool",
        "PHPUnit::assertIsObject",
        "expectException",
        "expectExceptionMessage",
        "expectExceptionCode",
        "expectExceptionMessageMatches",
        "expectExceptionObject",
        "expectNotToPerformAssertions",
        "markTestSkipped",
        "markTestIncomplete",
        "setUp",
        "tearDown",
        "setUpBeforeClass",
        "tearDownAfterClass",
        "dataProvider",
        "onNotSuccessfulTest",
        "getMock",
        "getMockBuilder",
        "createMock",
        "createStub",
        "createPartialMock",
        "createConfiguredMock",
        "getMockForAbstractClass",
        "getMockFromWsdl",
        "onlyMethods",
        "addMethods",
        "setMethods",
        "setConstructorArgs",
        "disableOriginalConstructor",
        "enableOriginalConstructor",
        "willReturn",
        "willReturnMap",
        "willReturnCallback",
        "willReturnArgument",
        "willReturnSelf",
        "willReturnOnConsecutiveCalls",
        "willThrowException",
        # ── Mockery ──
        "andReturn",
        "andThrow",
        "andThrowExceptions",
        "andReturnValues",
        "andReturnUsing",
        "andReturnSelf",
        "andReturnNull",
        "andReturnTrue",
        "andReturnFalse",
        "andYield",
        "allows",
        "expects",
        "shouldReceive",
        "shouldNotReceive",
        "shouldHaveReceived",
        "shouldNotHaveReceived",
        "zeroOrMoreTimes",
        "atLeastOnce",
        "atMostOnce",
        "byDefault",
        "passthru",
        "never",
        "once",
        "twice",
        "thrice",
        "makePartial",
        "withArgs",
        "with",
        "mock",
        "spy",
        "instanceMock",
        "namedMock",
        # ── Laravel framework helpers ──
        # Only distinctive / non-ambiguous names kept here. Generic
        # helpers like `collect`, `tap`, `now`, `today`, `value`,
        # `last`, `head`, `with`, `retry`, `dump`, `logger`, `report`,
        # `str`, `blank`, `filled`, `dd`, `faker` were removed — they
        # collide with legitimate project function names across
        # every other language / framework. Facade class-names like
        # `Auth`, `Request`, `Response`, `Cache`, `DB` were also
        # removed because they filter a bare `Request()` call in any
        # non-Laravel codebase (FastAPI, Django, etc.).
        "data_get",
        "data_set",
        "Illuminate\\",
        "Eloquent::",
        # ── Ruby stdlib / Kernel / Object reflection ──
        # These are never defined in user code so they must be filtered
        # even when called bare (`send(...)`, `sprintf(...)`).
        "public_send",
        "__send__",
        "method",
        "methods",
        "public_methods",
        "private_methods",
        "protected_methods",
        "singleton_methods",
        "method_defined?",
        "respond_to?",
        "respond_to_missing?",
        "instance_variable_get",
        "instance_variable_set",
        "instance_variables",
        "instance_variable_defined?",
        "class_variable_get",
        "class_variable_set",
        "class_variables",
        "instance_eval",
        "class_eval",
        "module_eval",
        "instance_exec",
        "class_exec",
        "module_exec",
        "define_method",
        "define_singleton_method",
        "method_missing",
        "remove_method",
        "undef_method",
        "alias_method",
        "const_get",
        "const_set",
        "const_defined?",
        "constants",
        "ancestors",
        "included_modules",
        "instance_method",
        "public_method",
        "singleton_class",
        "object_id",
        "eql?",
        "equal?",
        "frozen?",
        "tap",
        "yield_self",
        "inspect",
        "to_proc",
        "proc",
        "lambda",
        "block_given?",
        "binding",
        "eval",
        "getc",
        "readline",
        "readlines",
        "exit!",
        "at_exit",
        "caller",
        "caller_locations",
        "__method__",
        "__callee__",
        "__dir__",
        "gem",
        "trap",
        "Hash",
        "Rational",
        "Complex",
        # ── Ruby String / Enumerable / IO core methods (called bare on
        #    `self` or on any value — never project symbols). ──
        "gsub",
        "gsub!",
        "sub",
        "sub!",
        "scan",
        "match?",
        "index",
        "rindex",
        "partition",
        "rpartition",
        "chomp",
        "chomp!",
        "chop",
        "chop!",
        "strip!",
        "lstrip!",
        "rstrip!",
        "squeeze",
        "squeeze!",
        "tr",
        "tr!",
        "tr_s",
        "tr_s!",
        "delete",
        "delete!",
        "crypt",
        "upcase",
        "upcase!",
        "downcase",
        "downcase!",
        "capitalize",
        "capitalize!",
        "swapcase",
        "swapcase!",
        "center",
        "ljust",
        "rjust",
        "reverse!",
        "prepend",
        "to_sym",
        "to_f",
        "to_r",
        "to_c",
        "to_h",
        "to_a",
        "to_s",
        "to_i",
        "intern",
        "length",
        "size",
        "bytesize",
        "codepoints",
        "each_char",
        "each_byte",
        "each_line",
        "each_codepoint",
        "encoding",
        "encode",
        "encode!",
        "force_encoding",
        "valid_encoding?",
        "ascii_only?",
        "end_with?",
        "start_with?",
        "include?",
        "empty?",
        "nil?",
        "is_a?",
        "kind_of?",
        "instance_of?",
        "key?",
        "has_key?",
        "has_value?",
        "value?",
        "all?",
        "any?",
        "none?",
        "one?",
        "each_with_index",
        "each_with_object",
        "each_pair",
        "each_entry",
        "each_slice",
        "each_cons",
        "map!",
        "collect!",
        "collect_concat",
        "select!",
        "filter!",
        "reject!",
        "inject",
        "detect",
        "find_all",
        "find_index",
        "group_by",
        "chunk",
        "chunk_while",
        "slice_when",
        "slice_before",
        "slice_after",
        "take_while",
        "drop_while",
        "min_by",
        "max_by",
        "minmax",
        "minmax_by",
        "sort!",
        "sort_by!",
        "tally",
        "uniq",
        "uniq!",
        "compact!",
        "flatten!",
        "rotate!",
        "shuffle!",
        "cycle",
        "each_index",
        "assoc",
        "rassoc",
        "dig",
        "merge",
        "merge!",
        "update",
        "invert",
        "values_at",
        "pairs",
        "transform_keys",
        "transform_keys!",
        "transform_values",
        "transform_values!",
        "except",
        "compact_by",
        "readpartial",
        "sysread",
        "sysseek",
        "syswrite",
        "wait_readable",
        "wait_writable",
        "on",
        "on_head",
        "on_tail",
        "order!",
        "parse!",
        "permute!",
        "environment",
        "read_nonblock",
        "write_nonblock",
        "closed?",
        "eof?",
        "tell",
        "pos",
        "seek",
        "stat",
        "fileno",
        "fcntl",
        "ioctl",
        # ── Ruby stdlib classes used as constructors/methods ──
        "Addrinfo",
        "OpenSSL",
        "Socket",
        "TCPSocket",
        "TCPServer",
        "UDPSocket",
        "UNIXSocket",
        "UNIXServer",
        # ── JRuby Java-side API method bare names (called on
        # ThreadContext, IRubyObject, etc.) ──
        "getRuntime",
        "callMethod",
        "newArray",
        "newInstance",
        "newString",
        "newSymbol",
        "newFixnum",
        "newFloat",
        "newHash",
        "newProc",
        "newArgumentError",
        "newTypeError",
        "newRuntimeError",
        "convertToString",
        "convertToArray",
        "convertToHash",
        "convertToInteger",
        "convertToFloat",
        "asJavaString",
        # ── Ruby Minitest / Test::Unit assertions ──
        "assert_equal",
        "assert_not_equal",
        "assert_not_operator",
        "assert_not_predicate",
        "assert_not_respond_to",
        "assert_match",
        "assert_no_match",
        "assert_nil",
        "assert_not_nil",
        "assert_raise",
        "assert_raises",
        "assert_nothing_raised",
        "assert_includes",
        "assert_not_includes",
        "assert_empty",
        "assert_not_empty",
        "assert_in_delta",
        "assert_in_epsilon",
        "assert_instance_of",
        "assert_kind_of",
        "assert_respond_to",
        "assert_operator",
        "assert_predicate",
        "assert_same",
        "assert_not_same",
        "assert_throws",
        "assert_send",
        "assert_output",
        "assert_silent",
        "assert_dom_equal",
        "assert_dom_not_equal",
        "assert_response",
        "assert_redirected_to",
        "assert_template",
        "assert_select",
        "assert_difference",
        "assert_no_difference",
        "assert_changes",
        "assert_no_changes",
        "assert_enqueued_with",
        "assert_enqueued_jobs",
        "assert_performed_with",
        "refute",
        "refute_equal",
        "refute_nil",
        "refute_match",
        "refute_empty",
        "refute_includes",
        "refute_respond_to",
        "refute_instance_of",
        "refute_kind_of",
        # ── RSpec ──
        "describe",
        "context",
        "it",
        "specify",
        "before",
        "after",
        "around",
        "let",
        "let!",
        "subject",
        "be",
        "be_truthy",
        "be_falsey",
        "be_nil",
        "eql",
        "include",
        "have_attributes",
        "raise_error",
        "change",
        "satisfy",
        "receive",
        "allow",
        "double",
        "instance_double",
        "class_double",
        "stub",
        # ── Ruby common builtins accessed bare ──
        "Dir",
        "IO",
        "StringIO",
        "Tempfile",
        "Pathname",
        "Time",
        "Net",
        "YAML",
        "Marshal",
        "Digest",
        "Logger",
        "Mutex",
        "Fiber",
        "Enumerator",
        "Range",
        "Regexp",
        "Proc",
        "Method",
        "Struct",
        "OpenStruct",
        "StandardError",
        "RuntimeError",
        "ArgumentError",
        "NameError",
        "NoMethodError",
        "IOError",
        "Errno",
        "Kernel",
        "BasicObject",
        "Module",
        "Comparable",
        # ── JUnit / testing ──
        "assertEquals",
        "assertNotEquals",
        "assertThrows",
        "assertDoesNotThrow",
        "assertAll",
        "assertArrayEquals",
        "assertIterableEquals",
        "assertLinesMatch",
        "fail",
        "assumeTrue",
        "assumeFalse",
        "assumingThat",
        "BeforeEach",
        "AfterEach",
        "BeforeAll",
        "AfterAll",
        "Disabled",
        "Nested",
        "Tag",
        "Timeout",
        "ParameterizedTest",
        "ValueSource",
        "CsvSource",
        "MethodSource",
        "EnumSource",
        # ── Hamcrest matchers (commonly used via assertThat(x, is(...))) ──
        "is",
        "not",
        "equalTo",
        "containsString",
        "hasSize",
        "hasItem",
        "hasItems",
        "hasEntry",
        "hasKey",
        "hasValue",
        "containsInAnyOrder",
        "allOf",
        "anyOf",
        "instanceOf",
        "nullValue",
        "notNullValue",
        "greaterThan",
        "lessThan",
        "greaterThanOrEqualTo",
        "lessThanOrEqualTo",
        # ── Mockito ──
        "when",
        "verify",
        "atLeast",
        "atMost",
        "exactly",
        "only",
        "anyInt",
        "anyLong",
        "anyString",
        "anyBoolean",
        "anyByte",
        "anyShort",
        "anyFloat",
        "anyDouble",
        "anyChar",
        "anyObject",
        "anyList",
        "anyMap",
        "anyCollection",
        "anyIterable",
        "anySet",
        "argThat",
        "thenReturn",
        "thenThrow",
        "thenAnswer",
        "thenCallRealMethod",
        "doReturn",
        "doThrow",
        "doNothing",
        "doAnswer",
        "doCallRealMethod",
        "inOrder",
        "verifyNoInteractions",
        "verifyNoMoreInteractions",
        "clearInvocations",
        "ArgumentCaptor",
        "forClass",
        "capture",
        "getValue",
        "getAllValues",
        "mockStatic",
        "mockConstruction",
        "withSettings",
        "invocation",
        "getArgument",
        "getArguments",
        "callRealMethod",
        "getMethod",
        # ── AssertJ ──
        "assertThat",
        "assertThatCode",
        "assertThatThrownBy",
        "assertThatExceptionOfType",
        "assertThatIllegalArgumentException",
        "assertThatIllegalStateException",
        "assertThatNullPointerException",
        "assertThatIOException",
        "assertThatNoException",
        "assertThatObject",
        "isInstanceOf",
        "isNotInstanceOf",
        "hasMessage",
        "hasMessageContaining",
        "hasMessageStartingWith",
        "hasMessageEndingWith",
        "hasMessageMatching",
        "hasCauseInstanceOf",
        "hasNoCause",
        "hasRootCauseInstanceOf",
        "hasRootCauseMessage",
        "withMessage",
        "withMessageContaining",
        "withCauseInstanceOf",
        "withNoCause",
        "isThrownBy",
        "isThrownByCallable",
        "satisfies",
        "doesNotSatisfy",
        "extracting",
        "flatExtracting",
        "returns",
        "doesNotReturn",
        # ── BDDMockito / Mockito BDD style ──
        "given",
        "willThrow",
        "willDoNothing",
        "willAnswer",
        "willCallRealMethod",
        "shouldHaveNoMoreInteractions",
        "shouldHaveNoInteractions",
        # ── Project Reactor ──
        "StepVerifier",
        "expectNext",
        "expectError",
        "expectComplete",
        "verifyComplete",
        "verifyErrorMatches",
        "flatMap",
        "concatMap",
        "flatMapMany",
        "zipWith",
        "switchIfEmpty",
        "defaultIfEmpty",
        "onErrorResume",
        "onErrorReturn",
        "onErrorMap",
        "doOnNext",
        "doOnError",
        "doOnComplete",
        "doOnSubscribe",
        "doOnRequest",
        "doFinally",
        "subscribeOn",
        "publishOn",
        "block",
        "blockFirst",
        "blockLast",
        "collectList",
        "collectMap",
        "collectSortedList",
        "fromIterable",
        "fromStream",
        "fromSupplier",
        "fromCallable",
        "fromFuture",
        "just",
        # ── .NET (C#) string static methods ──
        "string.IsNullOrEmpty",
        "string.IsNullOrWhiteSpace",
        "string.Format",
        "string.Concat",
        "string.Join",
        "string.Compare",
        "string.CompareOrdinal",
        "string.Equals",
        "string.Copy",
        "string.Intern",
        "string.IsInterned",
        "IsNullOrEmpty",
        "IsNullOrWhiteSpace",
        # ── FakeItEasy (.NET mocking, widely used) ──
        "A.CallTo",
        "A.Fake",
        "A.Dummy",
        "A.Ignored",
        "A._",
        "MustHaveHappened",
        "MustNotHaveHappened",
        "MustHaveHappenedOnceExactly",
        "MustHaveHappenedOnceOrMore",
        "ReturnsLazily",
        "ReturnsNextFromSequence",
        "Invokes",
        # ── C variadic args (stdarg.h) ──
        "va_start",
        "va_end",
        "va_arg",
        "va_copy",
        "va_list",
        # ── C atomics (stdatomic.h) / jemalloc atomic_*_zu aliases ──
        "atomic_load",
        "atomic_store",
        "atomic_compare_exchange",
        "atomic_compare_exchange_strong",
        "atomic_compare_exchange_weak",
        "atomic_exchange",
        "atomic_fetch_add",
        "atomic_fetch_sub",
        "atomic_fetch_and",
        "atomic_fetch_or",
        "atomic_fetch_xor",
        "atomic_init",
        "atomic_thread_fence",
        "atomic_signal_fence",
        "atomic_flag_test_and_set",
        "atomic_flag_clear",
        # ── SLF4J / logging common patterns ──
        "getLogger",
        "trace",
        # ── Diesel ORM DSL ──
        "nullable",
        "eq_any",
        "ne_all",
        "execute",
        "get_result",
        "get_results",
        "order",
        "order_by",
        "inner_join",
        "left_join",
        "having",
        "sum",
        "limit",
        "is_not_null",
        "desc",
        "asc",
        "into_boxed",
        "or",
        "like",
        "not_like",
        "between",
        "gt",
        "lt",
        "ge",
        "le",
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
        "toMatchInlineSnapshot",
        "toMatch",
        "toStrictEqual",
        "toContainEqual",
        "toInclude",
        "toThrowError",
        "toBeCloseTo",
        "toBeGreaterThan",
        "toBeGreaterThanOrEqual",
        "toBeLessThan",
        "toBeLessThanOrEqual",
        "toBeInstanceOf",
        "toBeTypeOf",
        "toBeDefined",
        "toBeNaN",
        "toSatisfy",
        # Vitest/Jest lifecycle helpers called on test/describe.
        # Deliberately avoided generic names (only, skip, todo, each,
        # throw, concurrent, sequential) — they collide too easily with
        # project method names. `runIf` / `skipIf` are Vitest-specific
        # enough to safely filter.
        # ── Node.js stdlib fs methods (destructured imports are
        # common: `const { readFileSync } = require('fs')`), so the
        # `fs.` prefix alone doesn't catch every call site. These are
        # distinctive enough to be safe as bare filters.
        "readFileSync",
        "writeFileSync",
        "appendFileSync",
        "existsSync",
        "statSync",
        "lstatSync",
        "readdirSync",
        "mkdirSync",
        "rmdirSync",
        "rmSync",
        "unlinkSync",
        "renameSync",
        "copyFileSync",
        "realpathSync",
        "symlinkSync",
        "readlinkSync",
        "openSync",
        "closeSync",
        "readSync",
        "writeSync",
        "accessSync",
        "chmodSync",
        "chownSync",
        "truncateSync",
        "utimesSync",
        # Async fs promises.
        "promises.readFile",
        "promises.writeFile",
        # Jest/Vitest matcher factories reached via `expect.*`/bare.
        "stringMatching",
        "stringContaining",
        "arrayContaining",
        "objectContaining",
        "closeTo",
        "anything",
        # Node.js Buffer / path / os distinctive methods.
        "Buffer.from",
        "Buffer.alloc",
        "Buffer.allocUnsafe",
        "Buffer.concat",
        "Buffer.isBuffer",
        "pathToFileURL",
        "fileURLToPath",
        # Widely-used npm utility packages (distinctive function names
        # that appear as bare or receiver-called after destructuring).
        "stripAnsi",
        "stripIndent",
        "stripIndents",
        "dedent",
        "outdent",
        "chalk",
        # JS built-ins often called bare after destructuring.
        # Zod schema validator (de-facto TS schema library).
        "z.string",
        "z.number",
        "z.boolean",
        "z.object",
        "z.array",
        "z.union",
        "z.literal",
        "z.optional",
        "z.nullable",
        "z.enum",
        "z.record",
        "z.tuple",
        "z.date",
        "z.infer",
        "toThrow",
        "click",
        "locator",
        "getByRole",
        "getByText",
        "getByTestId",
        "waitFor",
        "selectOption",
        "goto",
        "beforeEach",
        "afterEach",
        # ── Svelte / SvelteKit ──
        "json",
        "derived",
        "style",
        # ── clap (CLI argument parsing) ──
        "arg",
        # ── Python stdlib module names (bare — used as receivers
        # for module-level calls, e.g. `re.compile`, `os.path.join`).
        # Also covered by `re.` / `os.` prefix filters above, but the
        # bare form catches cases where the module is reassigned to a
        # local before use. ──
        "re",
        "sys",
        "os",
        "pathlib",
        "subprocess",
        "shutil",
        "yaml",
        "toml",
        # ── Python stdlib module names (bare) ──
        "datetime",
        "asyncio",
        "collections",
        "hashlib",
        "hmac",
        "logging",
        "argparse",
        "threading",
        "multiprocessing",
        "functools",
        "itertools",
        "operator",
        "typing",
        "dataclasses",
        "enum",
        "abc",
        "contextlib",
        "shlex",
        "urllib",
        "http",
        "base64",
        "secrets",
        "random",
        "math",
        "statistics",
        "struct",
        "platform",
        "tomllib",
        "tomli",
        "io",
        "weakref",
        "textwrap",
        "traceback",
        "warnings",
        "unittest",
        "email",
        "csv",
        "gzip",
        "zipfile",
        "tarfile",
        "sqlite3",
        "importlib",
        # ── Python builtin exceptions (never project symbols) ──
        "BaseException",
        "KeyError",
        "IndexError",
        "AttributeError",
        "NotImplementedError",
        "StopIteration",
        "StopAsyncIteration",
        "FileNotFoundError",
        "FileExistsError",
        "IsADirectoryError",
        "NotADirectoryError",
        "PermissionError",
        "TimeoutError",
        "OSError",
        "OverflowError",
        "ZeroDivisionError",
        "ArithmeticError",
        "AssertionError",
        "LookupError",
        "ImportError",
        "ModuleNotFoundError",
        "UnboundLocalError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "UnicodeError",
        "SystemExit",
        "KeyboardInterrupt",
        "GeneratorExit",
        "RecursionError",
        "BrokenPipeError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
        "BlockingIOError",
        "InterruptedError",
        "ChildProcessError",
        "ProcessLookupError",
        # ── Python builtin types / constructors ──
        "frozenset",
        "bytearray",
        "complex",
        "reversed",
        "object",
        "property",
        "staticmethod",
        "classmethod",
        # ── Python builtin functions ──
        "ascii",
        "bin",
        "divmod",
        "vars",
        "dir",
        "callable",
        "input",
        "globals",
        "locals",
        "compile",
        "breakpoint",
        "help",
        # ── Python stdlib / builtin method names (no reliable project
        # symbol collision) — these are method calls on stdlib-typed
        # receivers that never resolve to project definitions.
        "write_text",
        "read_text",
        "write_bytes",
        "read_bytes",
        "as_posix",
        "as_uri",
        "touch",
        "hexdigest",
        "digest",
        "startswith",
        "endswith",
        "decode",
        "upper",
        "title",
        "casefold",
        "rsplit",
        "splitlines",
        "removeprefix",
        "removesuffix",
        "discard",
        "setdefault",
        "popitem",
        "zfill",
        "expandtabs",
        "translate",
        "maketrans",
        "format_map",
        "is_symlink",
        "is_absolute",
        "lstat",
        "rglob",
        "iterdir",
        "relative_to",
        "with_suffix",
        "with_stem",
        "with_name",
        "joinpath",
        "samefile",
        "cwd",
        "home",
        "writelines",
        # Common method-chain intermediates.
        "reshape",
        "astype",
        "cpu",
        "gpu",
        "detach",
        # More stdlib method names
        "isnumeric",
        "isidentifier",
        "isprintable",
        "istitle",
        "isascii",
        "issubset",
        "issuperset",
        "symmetric_difference",
        "intersection",
        "union",
        "difference",
        "setLevel",
        "getLevelName",
        "basicConfig",
        "cache_clear",
        "cache_info",
        "popleft",
        "popright",
        "appendleft",
        "group",
        "groups",
        "groupdict",
        "start",
        "submit",
        "cancel",
        "done",
        "result",
        "as_completed",
        "gather",
        "wait_for",
        "create_task",
        "ensure_future",
        "run_until_complete",
        "run",
        "new_event_loop",
        "asynccontextmanager",
        "contextmanager",
        "wraps",
        "partial",
        "lru_cache",
        "cache",
        "total_ordering",
        "singledispatch",
        "TYPE_CHECKING",
        "Any",
        "Union",
        "Dict",
        "AsyncIterator",
        "AsyncIterable",
        "Awaitable",
        "Coroutine",
        "AsyncGenerator",
        # ── Dart core / Flutter / common packages ──
        "debugPrint",
        "identical",
        "identityHashCode",
        "runtimeType",
        "noSuchMethod",
        "runApp",
        "runZoned",
        "runZonedGuarded",
        "setState",
        "initState",
        "dispose",
        "build",
        "didChangeDependencies",
        "didUpdateWidget",
        "createState",
        "debugFillProperties",
        "showDialog",
        "showModalBottomSheet",
        "showMenu",
        "Navigator",
        "MaterialApp",
        "Scaffold",
        "AppBar",
        "Text",
        "Container",
        "Row",
        "Column",
        "Padding",
        "Center",
        "Expanded",
        "SizedBox",
        "ElevatedButton",
        "TextButton",
        "IconButton",
        "FloatingActionButton",
        "ListView",
        "GridView",
        "SingleChildScrollView",
        "Icon",
        "Image",
        "ListTile",
        "Theme",
        "MediaQuery",
        "ValueNotifier",
        "ChangeNotifier",
        "StreamController",
        "StreamBuilder",
        "FutureBuilder",
        "Completer",
        "SplayTreeMap",
        "SplayTreeSet",
        "ListQueue",
        "DoubleLinkedQueue",
        "jsonDecode",
        "jsonEncode",
        "utf8",
        "latin1",
        "base64Encode",
        "base64Decode",
        "exp",
        "sin",
        "cos",
        "tan",
        "Platform",
        "Process",
        "HttpServer",
        "WebSocket",
        "ServerSocket",
        "setUpAll",
        "tearDownAll",
        "expectAsync",
        "expectLater",
        "testWidgets",
        "verifyNever",
        "verifyInOrder",
        # Dart built-in types used as calls (Function-type invocations,
        # numeric / string / collection constructors).
        "Function",
        "Null",
        # package:test stream-style matchers (emits*, neverEmits).
        "emits",
        "emitsAnyOf",
        "emitsInOrder",
        "emitsThrough",
        "emitsDone",
        "emitsError",
        "neverEmits",
        "mayEmit",
        "mayEmitMultiple",
        # package:matcher matcher constructors — functions that return
        # Matcher instances, used pervasively in Dart test suites.
        "isNot",
        "isEmpty",
        "isNotEmpty",
        "isNotNaN",
        "isTrue",
        "isFalse",
        "isNotNull",
        "isZero",
        "isNonZero",
        "isNegative",
        "isNonNegative",
        "isPositive",
        "isNonPositive",
        "equalsIgnoringCase",
        "equalsIgnoringWhitespace",
        "containsPair",
        "containsAll",
        "containsAllInOrder",
        "everyElement",
        "anyElement",
        "orderedEquals",
        "unorderedEquals",
        "pairwiseCompare",
        "predicate",
        "throwsA",
        "throwsArgumentError",
        "throwsFormatException",
        "throwsRangeError",
        "throwsStateError",
        "throwsException",
        "throwsUnsupportedError",
        "throwsUnimplementedError",
        "throwsAssertionError",
        "throwsNoSuchMethodError",
        "throwsConcurrentModificationError",
        "throwsTypeError",
        "returnsNormally",
        "completes",
        "completion",
        "doesNotComplete",
        # package:test annotations and lifecycle helpers.
        "TestOn",
        "OnPlatform",
        "Retry",
        "Tags",
        "addTearDown",
        "addSetUp",
        # ── package:test_descriptor (Dart test fixture builder; "d" is
        #    the conventional alias used by its own readme and the
        #    pub_semver / dart test ecosystem). ──
        "d.file",
        "d.dir",
        "d.nothing",
        "d.async",
        "d.pattern",
        "d.validate",
        "td.file",
        "td.dir",
        # ── package:test_process (shouldExit, shouldWrite, stdout stream). ──
        "shouldExit",
        "shouldWrite",
        "shouldNotWrite",
        "stdout.expect",
        "stderr.expect",
        # ── dart:typed_data / dart:convert ──
        "Uint8List.fromList",
        "Uint8List.view",
        "Uint16List.fromList",
        "Uint32List.fromList",
        "Uint64List.fromList",
        "Int8List.fromList",
        "Int16List.fromList",
        "Int32List.fromList",
        "Int64List.fromList",
        "Float32List.fromList",
        "Float64List.fromList",
        "ByteData.view",
        "ByteData.sublistView",
        "JsonEncoder.withIndent",
        "JsonEncoder",
        "JsonDecoder",
        "Utf8Encoder",
        "Utf8Decoder",
        "AsciiEncoder",
        "AsciiDecoder",
        "LineSplitter",
        "Base64Encoder",
        "Base64Decoder",
        # ── shelf (Dart HTTP framework) ──
        "shelf.Response.ok",
        "shelf.Response.notFound",
        "shelf.Response.badRequest",
        "shelf.Response.forbidden",
        "shelf.Response.internalServerError",
        "shelf.Response.seeOther",
        "shelf.Response.movedPermanently",
        "shelf.Response.found",
        "shelf.Response.unauthorized",
        "shelf.Handler",
        "shelf.Middleware",
        "shelf.Pipeline",
        "shelf.Request",
        "shelf.Response",
        "shelf.serveRequests",
        # ── pub_semver + meta + Dart SDK error constructors ──
        "VersionConstraint.compatibleWith",
        "VersionConstraint.parse",
        "VersionConstraint.any",
        "VersionConstraint.empty",
        "ArgumentError.checkNotNull",
        "ArgumentError.notNull",
        "ArgumentError.value",
        # ── package:args (ubiquitous in Dart CLI tools) ──
        "addFlag",
        "addOption",
        "addMultiOption",
        "addCommand",
        "addSubcommand",
        "wasParsed",
        "flag",
        "option",
        "multiOption",
        "rest",
        "usage",
        "usageException",
        "argResults",
        "addSeparator",
        # ── package:yaml (ubiquitous in Dart config) ──
        "loadYaml",
        "loadYamlDocument",
        "loadYamlStream",
        "loadYamlNode",
        # ── Dart SDK error types commonly thrown ──
        "StateError",
        "UnsupportedError",
        "UnimplementedError",
        "ConcurrentModificationError",
        "NoSuchMethodError",
        "OutOfMemoryError",
        "StackOverflowError",
        # ── package:pub_semver (Version) ──
        "Version",
        "VersionConstraint",
        "VersionRange",
        # ── java.util / java.time / java.nio common ──
        "ofEpochMilli",
        "ofEpochSecond",
        "toEpochMilli",
        "toMillis",
        "toNanos",
        "toSeconds",
        "setProperty",
        "getProperty",
        "stringPropertyNames",
        "getAbsolutePath",
        "getCanonicalPath",
        "getAbsoluteFile",
        "getCanonicalFile",
        "toPath",
        "toFile",
        "toURI",
        "toURL",
        "isDirectory",
        "isFile",
        "canRead",
        "canWrite",
        "canExecute",
        "mkdirs",
        "listFiles",
        "lastModified",
        "createUnresolved",
        "putInt",
        "getInt",
        "putLong",
        "getLong",
        "putDouble",
        "getDouble",
        "putFloat",
        "getFloat",
        "putShort",
        "getShort",
        "putChar",
        "getChar",
        "putByte",
        "getByte",
        "putBytes",
        "allocate",
        "allocateDirect",
        "wrap",
        "flip",
        "hasRemaining",
        "remaining",
        # ── .NET Reflection / Activity / Span ──
        "GetType",
        "GetMethod",
        "GetMethods",
        "GetProperty",
        "GetProperties",
        "GetField",
        "GetFields",
        "GetConstructor",
        "GetConstructors",
        "GetEvent",
        "GetEvents",
        "GetParameter",
        "GetParameters",
        "GetReturnType",
        "GetCustomAttributes",
        "GetGenericArguments",
        "GetInterfaces",
        "GetNestedType",
        "GetNestedTypes",
        "MakeGenericType",
        "MakeGenericMethod",
        "IsAssignableFrom",
        "IsGenericType",
        "IsGenericTypeDefinition",
        "IsInterface",
        "IsAbstract",
        "IsSealed",
        "IsEnum",
        "IsValueType",
        "IsSubclassOf",
        "Invoke",
        "InvokeMember",
        "CreateInstance",
        "CreateDelegate",
        "TryFormat",
        "TryParse",
        "TryParseExact",
        "TryWriteBytes",
        "TryGetValue",
        "TryAdd",
        "TryRemove",
        "TryUpdate",
        "TryDequeue",
        "TryEnqueue",
        "TryPop",
        "TryPeek",
        "Slice",
        "CopyTo",
        "ToArray",
        "ToList",
        "ToDictionary",
        "ToHashSet",
        "ToLookup",
        "AddRange",
        "RemoveRange",
        "InsertRange",
        "GetRange",
        "BinarySearch",
        "CreateRandom",
        "Append",
        "AppendLine",
        "AppendFormat",
        "AppendJoin",
        "Remove",
        "Replace",
        "Reverse",
        "Clear",
        "CopyFrom",
        "Concat",
        "ExpandoObject",
        "Claim",
        "OverloadResolutionPriority",
        "CLSCompliant",
        # ── Moq (.NET mocking, widely used alongside FakeItEasy) ──
        "It.IsAny",
        "It.Is",
        "It.IsNotNull",
        "It.IsIn",
        "It.IsInRange",
        "It.IsNotIn",
        "It.IsRegex",
        "Mock.Of",
        "Mock.Get",
        "SetupGet",
        "SetupSet",
        "SetupSequence",
        "SetupProperty",
        "SetupAllProperties",
        "VerifyGet",
        "VerifySet",
        "CallBase",
        "Raises",
        "RaisesAsync",
        # ── ASP.NET Core DI surface ──
        "AddSingleton",
        "AddTransient",
        "AddScoped",
        "AddHostedService",
        "AddHttpClient",
        "AddMvc",
        "AddControllers",
        "AddControllersWithViews",
        "AddRazorPages",
        "AddAuthentication",
        "AddAuthorization",
        "AddIdentity",
        "AddDbContext",
        "AddDbContextPool",
        "AddLogging",
        "AddOptions",
        "AddRouting",
        "AddCors",
        "AddResponseCompression",
        "AddResponseCaching",
        "AddMemoryCache",
        "AddDistributedMemoryCache",
        "AddSignalR",
        "AddSwaggerGen",
        "AddEndpointsApiExplorer",
        "BuildServiceProvider",
        "GetService",
        "GetServices",
        "GetRequiredService",
        "CreateScope",
        "CreateAsyncScope",
        "ServiceCollection",
        "UseMiddleware",
        "UseRouting",
        "UseEndpoints",
        "UseStaticFiles",
        "UseAuthentication",
        "UseAuthorization",
        "UseCors",
        "UseHttpsRedirection",
        "UseResponseCompression",
        "UseResponseCaching",
        "UseRequestLocalization",
        "UseExceptionHandler",
        "UseHsts",
        "UseDeveloperExceptionPage",
        "MapControllers",
        "MapControllerRoute",
        "MapGet",
        "MapPost",
        "MapPut",
        "MapPatch",
        "MapDelete",
        "MapHub",
        "MapHealthChecks",
        "MapRazorPages",
        "MapFallbackToFile",
        # ── .NET common attributes / attributes/types frequently reified ──
        "LoggerMessage",
        "MethodImpl",
        "TaskCompletionSource",
        # ── Selenium WebDriver (widely used .NET + Java + Python) ──
        "FindElement",
        "FindElements",
        "By.CssSelector",
        "By.Id",
        "By.Name",
        "By.ClassName",
        "By.TagName",
        "By.LinkText",
        "By.PartialLinkText",
        "By.XPath",
        "SendKeys",
        "Click",
        "Submit",
        "GetAttribute",
        "GetCssValue",
        "Navigate",
        "GoToUrl",
        "WebDriverWait",
        "ExpectedConditions",
        "Actions",
        # ── Hamcrest (Java matchers, widely used beyond JUnit) ──
        "sameInstance",
        "hasToString",
        "aMapWithSize",
        "aMapContaining",
        "anEmptyMap",
        "hasItemInArray",
        "hasProperty",
        "containsInRelativeOrder",
        "arrayContainingInAnyOrder",
        "arrayWithSize",
        "emptyArray",
        "emptyIterable",
        "emptyCollectionOf",
        "emptyIterableOf",
        "iterableWithSize",
        "isIn",
        "isOneOf",
        "isA",
        "typeCompatibleWith",
        "comparesEqualTo",
        "either",
        "both",
        "equalToObject",
        "equalToCompressingWhiteSpace",
        "equalToIgnoringCase",
        "endsWithIgnoringCase",
        "startsWithIgnoringCase",
        "containsStringIgnoringCase",
        "stringContainsInOrder",
        "matchesPattern",
        "matchesRegex",
        # ── ANTLR-generated parser plumbing (any ANTLR-consuming project) ──
        "getRuleContext",
        "getParent",
        "getChild",
        "getChildCount",
        "getChildren",
        "getStart",
        "getStop",
        "getText",
        "getTokens",
        "getToken",
        "getType",
        "getLine",
        "getCharPositionInLine",
        "getTokenIndex",
        "getSymbol",
        "getSourceInterval",
        "getRuleIndex",
        "getAltNumber",
        "enterRule",
        "exitRule",
        "enterOuterAlt",
        "visitChildren",
        "visit",
        "visitTerminal",
        "visitErrorNode",
        "MatchWildcard",
        "notifyErrorListeners",
        "recoverInline",
        "consume",
        "ExceptionType",
        "semanticPredicate",
        # ── Apache Lucene (broadly used in Java search projects) ──
        "BytesRef",
        "BytesRefBuilder",
        "NumericDocValuesField",
        "SortedNumericDocValuesField",
        "SortedDocValuesField",
        "SortedSetDocValuesField",
        "BinaryDocValuesField",
        "StringField",
        "TextField",
        "IntPoint",
        "LongPoint",
        "FloatPoint",
        "DoublePoint",
        "StoredField",
        "FieldType",
        "IndexWriter",
        "IndexReader",
        "DirectoryReader",
        "RandomIndexWriter",
        "IndexWriterConfig",
        "IndexSearcher",
        "Term",
        "TermQuery",
        "BooleanQuery",
        "PhraseQuery",
        "PrefixQuery",
        "WildcardQuery",
        "FuzzyQuery",
        "RangeQuery",
        "MatchAllDocsQuery",
        "RamUsageEstimator",
        # ── UncheckedIOException + other java.lang/java.util exceptions ──
        "UncheckedIOException",
        "NoSuchMethodException",
        "NoSuchFieldException",
        "EOFException",
        "SocketException",
        "SocketTimeoutException",
        "UnknownHostException",
        "MalformedURLException",
        # ── java.sql + jakarta/javax.servlet + javax.xml + java.util.Locale ──
        "SQLException",
        "SQLWarning",
        "SQLDataException",
        "SQLIntegrityConstraintViolationException",
        "SQLRecoverableException",
        "SQLTransientException",
        "SQLNonTransientException",
        "SQLTimeoutException",
        "SQLClientInfoException",
        "SQLFeatureNotSupportedException",
        "SQLSyntaxErrorException",
        "SQLTransactionRollbackException",
        "BatchUpdateException",
        "DataTruncation",
        "ResultSet",
        "PreparedStatement",
        "CallableStatement",
        "Statement",
        "Connection",
        "DataSource",
        "DriverManager",
        "ServletException",
        "ServletRequest",
        "ServletResponse",
        "HttpServletRequest",
        "HttpServletResponse",
        "HttpSession",
        "ServletContext",
        "ServletConfig",
        "FilterChain",
        "FilterConfig",
        "RequestDispatcher",
        "Cookie",
        "UnavailableException",
        "InputSource",
        "SAXException",
        "SAXParseException",
        "SAXNotRecognizedException",
        "SAXNotSupportedException",
        "DocumentBuilderFactory",
        "DocumentBuilder",
        "TransformerFactory",
        "Transformer",
        "XPathFactory",
        "XPath",
        "XPathExpression",
        "ParserConfigurationException",
        "Locale.getDefault",
        "Locale.setDefault",
        "Locale.US",
        "Locale.ROOT",
        "Locale.UK",
        "Locale.FRANCE",
        "Locale.GERMANY",
        "Locale.ITALY",
        "Locale.JAPAN",
        "Locale.KOREA",
        "Locale.CHINA",
        "Locale.TAIWAN",
        "Locale.ENGLISH",
        "Locale.FRENCH",
        "Locale.GERMAN",
        "Locale.ITALIAN",
        "Locale.JAPANESE",
        "Locale.KOREAN",
        "Locale.CHINESE",
        "Locale.SIMPLIFIED_CHINESE",
        "Locale.TRADITIONAL_CHINESE",
        "Locale.CANADA",
        "Locale.CANADA_FRENCH",
        "Locale.PRC",
        "getAvailableLocales",
        "forLanguageTag",
        "toLanguageTag",
        "getLanguage",
        "getCountry",
        "getScript",
        "getVariant",
        "getDisplayName",
        "getDisplayLanguage",
        "getDisplayCountry",
        "getDisplayScript",
        "getDisplayVariant",
        "getISO3Language",
        "getISO3Country",
        # java.lang.reflect
        "getTargetException",
        "setStackTrace",
        "fillInStackTrace",
        "getSuppressed",
        "addSuppressed",
        "toMethodArgs",
        "getParameterCount",
        "getParameterTypes",
        "getParameterAnnotations",
        "getGenericParameterTypes",
        "getExceptionTypes",
        "getGenericExceptionTypes",
        "getGenericReturnType",
        "getGenericSuperclass",
        "getSuperclass",
        "getInterfaces",
        "getGenericInterfaces",
        "getDeclaredMethods",
        "getDeclaredFields",
        "getDeclaredConstructors",
        "getDeclaredMethod",
        "getDeclaredField",
        "getDeclaredConstructor",
        "getDeclaredAnnotations",
        "getDeclaredAnnotation",
        "getAnnotations",
        "getAnnotation",
        "isAnnotationPresent",
        "setAccessible",
        "trySetAccessible",
        "canAccess",
        # ── Go encoding/binary + sync + bytes.Buffer stdlib surface ──
        "binary.BigEndian.Uint16",
        "binary.BigEndian.Uint32",
        "binary.BigEndian.Uint64",
        "binary.BigEndian.PutUint16",
        "binary.BigEndian.PutUint32",
        "binary.BigEndian.PutUint64",
        "binary.LittleEndian.Uint16",
        "binary.LittleEndian.Uint32",
        "binary.LittleEndian.Uint64",
        "binary.LittleEndian.PutUint16",
        "binary.LittleEndian.PutUint32",
        "binary.LittleEndian.PutUint64",
        "binary.Read",
        "binary.Write",
        "binary.Size",
        "binary.Varint",
        "binary.Uvarint",
        "binary.PutVarint",
        "binary.PutUvarint",
        "BigEndian.Uint16",
        "BigEndian.Uint32",
        "BigEndian.Uint64",
        "BigEndian.PutUint16",
        "BigEndian.PutUint32",
        "BigEndian.PutUint64",
        "LittleEndian.Uint16",
        "LittleEndian.Uint32",
        "LittleEndian.Uint64",
        "LittleEndian.PutUint16",
        "LittleEndian.PutUint32",
        "LittleEndian.PutUint64",
        # sync package methods (called on Mutex/RWMutex/WaitGroup values)
        "Unlock",
        "RLock",
        "RUnlock",
        "TryLock",
        "TryRLock",
        "RLocker",
        "Add",
        "Done",
        "Wait",
        "Go",
        "SetLimit",
        "Signal",
        "Broadcast",
        "Do",
        # bytes.Buffer + strings.Builder common methods
        "WriteByte",
        "WriteRune",
        "WriteString",
        "WriteTo",
        "ReadFrom",
        "ReadByte",
        "ReadRune",
        "ReadString",
        "ReadBytes",
        "Next",
        "Cap",
        "Grow",
        "Truncate",
        "UnreadByte",
        "UnreadRune",
        # ── golang.org/x/sync/errgroup (widely-used Go concurrency lib) ──
        "errgroup.WithContext",
        "errgroup.Group",
        "errgroup.New",
        "errgroup.SetLimit",
        "errgroup.TryGo",
        # ── go.uber.org/goleak (widely-used Go goroutine leak detector) ──
        "goleak.VerifyNone",
        "goleak.VerifyTestMain",
        "goleak.IgnoreCurrent",
        "goleak.IgnoreTopFunction",
        "goleak.IgnoreAnyFunction",
        "goleak.Cleanup",
        "goleak.Errf",
        "VerifyNone",
        "VerifyTestMain",
        "IgnoreCurrent",
        "IgnoreTopFunction",
        "IgnoreAnyFunction",
        # ── AWS SDK v2 Go helpers ──
        "aws.Int32",
        "aws.Int64",
        "aws.Int",
        "aws.Uint32",
        "aws.Uint64",
        "aws.Float32",
        "aws.Float64",
        "aws.String",
        "aws.StringMap",
        "aws.StringSlice",
        "aws.Bool",
        "aws.Time",
        "aws.Duration",
        "aws.ToString",
        "aws.ToInt32",
        "aws.ToInt64",
        "aws.ToBool",
        "aws.ToTime",
        "aws.ToFloat32",
        "aws.ToFloat64",
        "aws.NewConfig",
        "aws.Config",
        # ── ULID (widely-used ID library for Go) ──
        "ulid.MustNew",
        "ulid.New",
        "ulid.Make",
        "ulid.MakeULID",
        "ulid.Parse",
        "ulid.Timestamp",
        "ulid.Monotonic",
        "ulid.MonotonicULID",
        # ── .NET P/Invoke attributes (Ansible ships C# modules for
        # Windows; any cross-lang shim uses these) ──
        "DllImport",
        "StructLayout",
        "MarshalAs",
        "FieldOffset",
        "InAttribute",
        "OutAttribute",
        "UnmanagedFunctionPointer",
        "ComVisible",
        "ComImport",
        "InterfaceType",
        "PreserveSig",
        "MarshalAsAttribute",
        # ── POSIX threads (pthread_* — language-general C API) ──
        "pthread_detach",
        "pthread_exit",
        "pthread_cancel",
        "pthread_self",
        "pthread_equal",
        "pthread_once",
        "pthread_key_create",
        "pthread_key_delete",
        "pthread_setspecific",
        "pthread_getspecific",
        "pthread_attr_init",
        "pthread_attr_destroy",
        "pthread_attr_setdetachstate",
        "pthread_attr_getdetachstate",
        "pthread_attr_setstacksize",
        "pthread_attr_getstacksize",
        "pthread_attr_setstackaddr",
        "pthread_attr_getstackaddr",
        "pthread_attr_setguardsize",
        "pthread_attr_getguardsize",
        "pthread_attr_setschedpolicy",
        "pthread_attr_getschedpolicy",
        "pthread_attr_setschedparam",
        "pthread_attr_getschedparam",
        "pthread_attr_setscope",
        "pthread_attr_getscope",
        "pthread_attr_setinheritsched",
        "pthread_attr_getinheritsched",
        "pthread_mutex_trylock",
        "pthread_mutex_timedlock",
        "pthread_mutexattr_init",
        "pthread_mutexattr_destroy",
        "pthread_mutexattr_settype",
        "pthread_mutexattr_gettype",
        "pthread_mutexattr_setprotocol",
        "pthread_mutexattr_getprotocol",
        "pthread_mutexattr_setpshared",
        "pthread_mutexattr_getpshared",
        "pthread_cond_timedwait",
        "pthread_condattr_init",
        "pthread_condattr_destroy",
        "pthread_condattr_setpshared",
        "pthread_condattr_getpshared",
        "pthread_rwlock_destroy",
        "pthread_rwlock_tryrdlock",
        "pthread_rwlock_trywrlock",
        "pthread_rwlock_timedrdlock",
        "pthread_rwlock_timedwrlock",
        "pthread_rwlockattr_init",
        "pthread_rwlockattr_destroy",
        "pthread_rwlockattr_setpshared",
        "pthread_rwlockattr_getpshared",
        "pthread_spin_init",
        "pthread_spin_destroy",
        "pthread_spin_lock",
        "pthread_spin_trylock",
        "pthread_spin_unlock",
        "pthread_barrier_init",
        "pthread_barrier_destroy",
        "pthread_barrier_wait",
        "pthread_barrierattr_init",
        "pthread_barrierattr_destroy",
        "pthread_barrierattr_setpshared",
        "pthread_barrierattr_getpshared",
        "pthread_setcancelstate",
        "pthread_setcanceltype",
        "pthread_testcancel",
        "pthread_cleanup_push",
        "pthread_cleanup_pop",
        "pthread_setschedparam",
        "pthread_getschedparam",
        "pthread_setschedprio",
        "pthread_sigmask",
        "pthread_kill",
        "pthread_yield",
        "pthread_setname_np",
        "pthread_getname_np",
        "pthread_getcpuclockid",
        "pthread_atfork",
        # ── BSD-specific strings (widely available in modern libc) ──
        "strlcpy",
        "strlcat",
        "strnstr",
        "strsep",
        "strmode",
        "strtonum",
        "fgetln",
        "fgetwln",
        "fpurge",
        "arc4random",
        "arc4random_buf",
        "arc4random_uniform",
        "reallocarray",
        "explicit_bzero",
        # ── BSD socket byte-order + network helpers (POSIX C API) ──
        "ntohs",
        "ntohl",
        "htons",
        "htonl",
        "ntohll",
        "htonll",
        "inet_ntoa",
        "inet_aton",
        "inet_ntop",
        "inet_pton",
        "inet_addr",
        "inet_network",
        "inet_lnaof",
        "inet_netof",
        "inet_makeaddr",
        "getaddrinfo",
        "freeaddrinfo",
        "getnameinfo",
        "gai_strerror",
        "gethostbyaddr",
        "gethostname",
        "getservbyname",
        "getservbyport",
        "sendmsg",
        "recvmsg",
        "shutdown",
        "getsockname",
        "getpeername",
        # ── HTTP response helpers common across Laravel/Symfony/ASP.NET
        #    controllers + any HTTP framework. ──
        "getStatusCode",
        "setStatusCode",
        "getContent",
        "setContent",
        "getBody",
        "setBody",
        "getHeaders",
        "setHeaders",
        "getHeader",
        "setHeader",
        "hasHeader",
        "removeHeader",
        "withHeader",
        "withStatus",
        "withBody",
        "getReasonPhrase",
        "getProtocolVersion",
        "withProtocolVersion",
        "withMethod",
        "getUri",
        "withUri",
        "getRequestTarget",
        "withRequestTarget",
        "getQueryParams",
        "withQueryParams",
        "getParsedBody",
        "withParsedBody",
        "getAttributes",
        "withAttribute",
        "withoutAttribute",
        "getUploadedFiles",
        "withUploadedFiles",
        "getServerParams",
        "getCookieParams",
        "withCookieParams",
        "getClientOriginalName",
        "getClientOriginalExtension",
        "getClientMimeType",
        "getPathname",
        "getRealPath",
        "getFilename",
        "getExtension",
        "getMimeType",
        "getSize",
        "getError",
        "getErrorMessage",
        "isValid",
        # ── voluptuous (Python validation — widely used) ──
        "voluptuous.Schema",
        "voluptuous.Required",
        "voluptuous.Optional",
        "voluptuous.All",
        "voluptuous.Any",
        "voluptuous.Invalid",
        "voluptuous.MultipleInvalid",
        "voluptuous.Length",
        "voluptuous.Range",
        "voluptuous.Match",
        "voluptuous.Url",
        "voluptuous.Email",
        "voluptuous.Datetime",
        "voluptuous.In",
        "voluptuous.NotIn",
        "voluptuous.Coerce",
        "voluptuous.Boolean",
        "voluptuous.Number",
        # ── Python abc / unittest.mock / stdlib extras ──
        "abstractmethod",
        "abstractproperty",
        "abstractstaticmethod",
        "abstractclassmethod",
        "MagicMock",
        "AsyncMock",
        "NonCallableMagicMock",
        "NonCallableMock",
        "PropertyMock",
        "create_autospec",
        "patch.object",
        "patch.dict",
        "patch.multiple",
        "patch.stopall",
    ]
)

# Combined filter used by is_non_project_call()
STDLIB_FILTER = _STDLIB_NAMES | _LIBRARY_NAMES

# JavaScript / TypeScript control-flow keywords. The TS regex call fallback
# matches any `ident(` pattern, which scoops up `if (`, `while (`, etc. —
# these are language keywords, never calls. Keep this as a small, fixed set
# of reserved words so the filter is language-general.
_JSTS_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "for",
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
        "typeof",
        "instanceof",
        "in",
        "of",
        "new",
        "delete",
        "void",
        "yield",
        "await",
        "async",
        "function",
        "class",
        "extends",
        "implements",
        "super",
        "this",
        "true",
        "false",
        "null",
        "undefined",
        "NaN",
        "Infinity",
        "var",
        "let",
        "const",
        "export",
        "import",
        "as",
        "from",
        "with",
        "constructor",
    }
)

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
    # ── JavaScript / TypeScript built-in objects and Node stdlib ──
    "console.",
    "Math.",
    "JSON.",
    "Object.",
    "Array.",
    "Promise.",
    "Reflect.",
    "Symbol.",
    "Proxy.",
    "Map.",
    "Set.",
    "WeakMap.",
    "WeakSet.",
    "Number.",
    "String.",
    "Date.",
    "Error.",
    "TypeError.",
    "RangeError.",
    "window.",
    "document.",
    "navigator.",
    "sessionStorage.",
    "localStorage.",
    # Node.js stdlib modules (both Node ESM and CommonJS paths).
    "fs.",
    "fs/promises.",
    "node:fs.",
    "node:fs/promises.",
    "node:path.",
    "os.",
    "node:os.",
    "crypto.",
    "node:crypto.",
    "stream.",
    "node:stream.",
    "util.",
    "node:util.",
    "buffer.",
    "node:buffer.",
    "child_process.",
    "node:child_process.",
    "http.",
    "node:http.",
    "https.",
    "node:https.",
    "url.",
    "node:url.",
    "querystring.",
    "readline.",
    "events.",
    "node:events.",
    "zlib.",
    "node:zlib.",
    # Jest / Vitest matcher factories reached via `expect.*`.
    "expect.",
    "jest.",
    "vi.",
    "vitest.",
    # ── Python stdlib modules ──
    "os.",
    "sys.",
    "re.",
    "pathlib.",
    "subprocess.",
    "shutil.",
    "decimal.",
    "ctypes.",
    "datetime.",
    "json.",
    "asyncio.",
    "collections.",
    "functools.",
    "itertools.",
    "contextlib.",
    "typing.",
    "base64.",
    "hashlib.",
    "hmac.",
    "tempfile.",
    "logging.",
    "threading.",
    "multiprocessing.",
    "socket.",
    "select.",
    "struct.",
    "copy.",
    "operator.",
    "enum.",
    "dataclasses.",
    "inspect.",
    "traceback.",
    "warnings.",
    "weakref.",
    "ast.",
    "codecs.",
    "io.",
    "csv.",
    "gzip.",
    "zipfile.",
    "tarfile.",
    "sqlite3.",
    "uuid.",
    "http.",
    "urllib.",
    "email.",
    "random.",
    "math.",
    "statistics.",
    "calendar.",
    "time.",
    "platform.",
    "signal.",
    "queue.",
    "heapq.",
    "bisect.",
    "array.",
    "importlib.",
    "unittest.",
    "argparse.",
    # ── Python common third-party libs often imported by module name ──
    "selenium.",
    "asgiref.",
    "pytest.",
    "numpy.",
    "pandas.",
    "scipy.",
    "torch.",
    "sklearn.",
    "matplotlib.",
    "tensorflow.",
    # Note: `cursor.` and `conn.` are Python DB-API 2.0 convention but
    # intentionally NOT added as global prefixes — they're common
    # variable names in many languages (DB cursors, text cursors, UI
    # cursors) and filtering them globally over-matches project code.
)

_LIBRARY_PREFIXES = (
    # ── Go standard library ──
    "fmt.",
    "strings.",
    "strconv.",
    "os.",
    "io.",
    "bytes.",
    "bufio.",
    "encoding/json.",
    "encoding/base64.",
    "encoding/hex.",
    "encoding/xml.",
    "net/http.",
    "net/url.",
    "net.",
    "context.",
    "sync.",
    "sync/atomic.",
    "time.",
    "errors.",
    "sort.",
    "path/filepath.",
    "path.",
    "filepath.",
    "rand.",
    "reflect.",
    "unicode.",
    "regexp.",
    "log.",
    "log/slog.",
    "math.",
    "math/rand.",
    "runtime.",
    "syscall.",
    "os/exec.",
    "container/list.",
    "container/heap.",
    "hash.",
    "hash/crc32.",
    "compress/gzip.",
    "compress/zlib.",
    "archive/tar.",
    "archive/zip.",
    "crypto.",
    "crypto/aes.",
    "crypto/sha256.",
    "crypto/rand.",
    "crypto/tls.",
    # ── Go testing / testify package-level names (short forms).
    # These are the package-qualified call prefixes as they appear in
    # Go source: `require.NoError`, `assert.Equal`, etc. We deliberately
    # DON'T add `t.` as a prefix because `t` is a legal project variable
    # name in other languages; rely on the bare-method STDLIB_FILTER
    # additions (Parallel, Helper, TempDir, etc.) to catch `t.*` instead.
    "testing.",
    "require.",
    "assert.",
    "mock.",
    "suite.",
    "httptest.",
    "httpmock.",
    "gomock.",
    "slog.",
    "maps.",
    "slices.",
    "cmp.",
    "unsafe.",
    "atomic.",
    "iter.",
    "weak.",
    # Protobuf + gRPC generated Go code — emitted by `protoc-gen-go`
    # into every .pb.go file regardless of project. General, not
    # codebase-specific.
    "protoimpl.",
    "protoreflect.",
    "proto.",
    "protojson.",
    "prototext.",
    "protowire.",
    "anypb.",
    "durationpb.",
    "timestamppb.",
    "wrapperspb.",
    "structpb.",
    "emptypb.",
    "fieldmaskpb.",
    # ── Go common third-party ──
    "github.com/stretchr/testify",
    "google.golang.org/protobuf",
    "google.golang.org/grpc",
    "github.com/spf13/cobra",
    "github.com/spf13/viper",
    "github.com/sirupsen/logrus",
    "go.uber.org/zap",
    "github.com/pkg/errors",
    "github.com/go-chi/chi",
    "github.com/gorilla/mux",
    "github.com/gin-gonic/gin",
    "github.com/labstack/echo",
    # ── klog (canonical Kubernetes/K8s-ecosystem logger) ──
    "klog.",
    # ── Ginkgo BDD (widely-used Go testing framework) ──
    "ginkgo.",
    "Gomega.",
    # ── testify (widely-used Go testing helpers) ──
    "assert.",
    "require.",
    "suite.",
    "mock.",
    # ── prometheus/client_golang (widely-used Go metrics library) ──
    "prometheus.MustRegister",
    "prometheus.Register",
    "prometheus.Unregister",
    "prometheus.NewCounter",
    "prometheus.NewCounterVec",
    "prometheus.NewGauge",
    "prometheus.NewGaugeVec",
    "prometheus.NewHistogram",
    "prometheus.NewHistogramVec",
    "prometheus.NewSummary",
    "prometheus.NewSummaryVec",
    "prometheus.NewRegistry",
    "prometheus.NewDesc",
    "prometheus.Labels",
    "prometheus.Handler",
    "prometheus.Collector",
    "prometheus.DefaultRegisterer",
    "prometheus.DefaultGatherer",
    "promauto.With",
    "promauto.NewCounter",
    "promauto.NewCounterVec",
    "promauto.NewGauge",
    "promauto.NewGaugeVec",
    "promauto.NewHistogram",
    "promauto.NewHistogramVec",
    "promhttp.Handler",
    "promhttp.HandlerFor",
    "promhttp.InstrumentHandlerCounter",
    "promhttp.InstrumentHandlerDuration",
    "promhttp.InstrumentRoundTripperCounter",
    "promhttp.InstrumentRoundTripperDuration",
    "metav1.Now",
    "metav1.ObjectMeta",
    "metav1.TypeMeta",
    "metav1.ListOptions",
    "metav1.CreateOptions",
    "metav1.UpdateOptions",
    "metav1.DeleteOptions",
    "metav1.GetOptions",
    "metav1.PatchOptions",
    "apierrors.IsNotFound",
    "apierrors.IsAlreadyExists",
    "apierrors.IsConflict",
    "apierrors.IsGone",
    "apierrors.IsForbidden",
    "apierrors.IsUnauthorized",
    "apierrors.IsBadRequest",
    "apierrors.IsInternalError",
    "apierrors.IsInvalid",
    "apierrors.IsServiceUnavailable",
    "apierrors.IsTimeout",
    "apierrors.NewNotFound",
    "apierrors.NewAlreadyExists",
    "apierrors.NewConflict",
    "apierrors.NewInternalError",
    "apierrors.NewInvalid",
    "apierrors.NewBadRequest",
    # ── Ruby stdlib & ecosystem prefixes ──
    "FileUtils.",
    "Process.",
    "Signal.",
    "I18n.",
    "ActiveSupport::",
    "ActionDispatch::",
    "ActionController::",
    "ActionView::",
    "ActiveRecord::",
    "ActiveModel::",
    "ActiveJob::",
    "ActionMailer::",
    "ActionCable::",
    "ActiveStorage::",
    "Minitest::",
    "RSpec::",
    "Rails::",
    # ── Ruby stdlib classes accessed via their simple name ──
    "File.",
    "Dir.",
    "IO.",
    "StringIO.",
    "Tempfile.",
    "Pathname.",
    "URI.",
    "DateTime.",
    "Date.",
    "Time.",
    "JSON.",
    "YAML.",
    "Marshal.",
    "Base64.",
    "Digest.",
    "Thread.",
    "Mutex.",
    "Fiber.",
    "Enumerator.",
    "Range.",
    "Regexp.",
    "Struct.",
    "OpenStruct.",
    "Kernel.",
    "Object.",
    "Module.",
    "Class.",
    "Comparable.",
    "Enumerable.",
    "Net::",
    "Net::HTTP",
    "OpenSSL::",
    "Rails.",
    "ActiveRecord::",
    "ActiveSupport::",
    "ActiveModel::",
    "ActionController::",
    "ActionView::",
    "ActionDispatch::",
    "Sidekiq.",
    "Sidekiq::",
    "SecureRandom.",
    "SecureRandom::",
    "Open3.",
    "Open3::",
    "Addrinfo.",
    "Addrinfo::",
    "Socket.",
    "Socket::",
    "TCPSocket.",
    "TCPSocket::",
    "TCPServer.",
    "TCPServer::",
    "UDPSocket.",
    "UDPSocket::",
    "UNIXSocket.",
    "UNIXSocket::",
    "UNIXServer.",
    "UNIXServer::",
    "IPAddr.",
    "IPAddr::",
    "Resolv.",
    "Resolv::",
    "Zlib.",
    "Zlib::",
    "CGI.",
    "CGI::",
    "ERB.",
    "ERB::",
    "Liquid::",
    # ── C++ std:: qualified prefix ──
    "std::",
    "__builtin_",
    "__sync_",
    "__atomic_",
    # ── Common C++ third-party ──
    "boost::",
    "absl::",
    "folly::",
    "glog::",
    "gflags::",
    "gtest::",
    "testing::",
    "benchmark::",
    "fmt::",
    "nlohmann::",
    "rapidjson::",
    "tbb::",
    "Eigen::",
    "cv::",
    # ── .NET BCL classes via qualified prefix ──
    "Guid.",
    "DateTime.",
    "DateTimeOffset.",
    "TimeSpan.",
    "TimeOnly.",
    "DateOnly.",
    "CultureInfo.",
    "Encoding.",
    "Path.",
    "File.",
    "Directory.",
    "Environment.",
    "Type.",
    "Math.",
    "Console.",
    "Convert.",
    "Task.",
    "ValueTask.",
    "Thread.",
    "Activator.",
    "Regex.",
    "Enum.",
    "Array.",
    "Buffer.",
    "Span.",
    "Memory.",
    "ReadOnlySpan.",
    "ReadOnlyMemory.",
    "Encoding.UTF8.",
    "Interlocked.",
    "Volatile.",
    "Monitor.",
    "Assert.",
    "Assume.",
    "Record.",
    "Theory.",
    "ClassicAssert.",
    "CollectionAssert.",
    "StringAssert.",
    "System.",
    "Microsoft.",
    "Microsoft.AspNetCore.",
    "Microsoft.Extensions.",
    "Microsoft.EntityFrameworkCore.",
    "Xunit.",
    "NUnit.",
    "MSTest.",
    "Moq.",
    "NSubstitute.",
    "FluentAssertions.",
    "Newtonsoft.",
    "AutoMapper.",
    "MediatR.",
    "Serilog.",
    "NLog.",
    "FluentValidation.",
    "Dapper.",
    "Autofac.",
    "Polly.",
    # ── PHP package-qualified prefixes ──
    "Illuminate\\",
    "Symfony\\",
    "Doctrine\\",
    "PHPUnit\\",
    "Psr\\",
    "Mockery\\",
    "Monolog\\",
    "GuzzleHttp\\",
    "Carbon\\",
    "Ramsey\\",
    "Laravel\\",
    # ── Java / Kotlin / Scala standard library (package-qualified) ──
    "java.lang.",
    "java.util.",
    "java.io.",
    "java.nio.",
    "java.net.",
    "java.time.",
    "java.math.",
    "java.text.",
    "java.sql.",
    "java.security.",
    "java.util.concurrent.",
    "java.util.stream.",
    "java.util.function.",
    "java.util.regex.",
    "java.util.logging.",
    "javax.",
    "jdk.",
    "sun.",
    "com.sun.",
    "kotlin.",
    "kotlinx.",
    "scala.",
    # ── Java stdlib classes accessed via their simple name ──
    # (e.g., `Collections.singletonList(...)`, `Arrays.asList(...)`)
    "Collections.",
    "Arrays.",
    "Objects.",
    "Optional.",
    "Files.",
    "Paths.",
    "Math.",
    "Executors.",
    "TimeUnit.",
    "ChronoUnit.",
    "Duration.",
    "Instant.",
    "LocalDate.",
    "LocalDateTime.",
    "LocalTime.",
    "ZonedDateTime.",
    "UUID.",
    "Pattern.",
    "Base64.",
    "StandardCharsets.",
    "Comparator.",
    "Stream.",
    "IntStream.",
    "LongStream.",
    "DoubleStream.",
    "Collectors.",
    "CompletableFuture.",
    "Integer.",
    "Long.",
    "Double.",
    "Float.",
    "Boolean.",
    "Byte.",
    "Short.",
    "Character.",
    "String.",
    "Thread.",
    "System.",
    "Class.",
    "Runtime.",
    # ── Java common third-party ──
    "org.springframework.",
    "org.apache.",
    "org.junit.",
    "org.hibernate.",
    "org.slf4j.",
    "org.mockito.",
    "org.assertj.",
    "com.google.common.",
    "com.google.gson.",
    "com.google.inject.",
    "com.fasterxml.jackson.",
    "lombok.",
    "io.netty.",
    "io.grpc.",
    "io.reactivex.",
    "reactor.",
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
    # ── Test/logging module-qualified calls. Local-variable receiver names
    # (parser, args, config, result, request, …) do not belong here: a
    # prefix applies to every language, where they are ordinary identifiers.
    "pytest.",
    "logger.",
    # ── Python stdlib modules (dotted call form: json.dumps, asyncio.Lock) ──
    "json.",
    "time.",
    "datetime.",
    "asyncio.",
    "collections.",
    "hashlib.",
    "hmac.",
    "logging.",
    "argparse.",
    "subprocess.",
    "threading.",
    "multiprocessing.",
    "functools.",
    "itertools.",
    "operator.",
    "inspect.",
    "typing.",
    "dataclasses.",
    "enum.",
    "abc.",
    "contextlib.",
    "tempfile.",
    "shlex.",
    "signal.",
    "socket.",
    "select.",
    "urllib.",
    "http.",
    "base64.",
    "secrets.",
    "random.",
    "math.",
    "statistics.",
    "struct.",
    "copy.",
    "stat.",
    "platform.",
    "errno.",
    "tomllib.",
    "tomli.",
    "io.",
    "weakref.",
    "textwrap.",
    "traceback.",
    "warnings.",
    "unittest.",
    "email.",
    "csv.",
    "gzip.",
    "zipfile.",
    "tarfile.",
    "sqlite3.",
    "importlib.",
    "pkgutil.",
    # ── Python third-party libraries used by descry's deps ──
    "starlette.",
    "uvicorn.",
    "fastapi.",
    "pydantic.",
    "httpx.",
    "requests.",
    "aiohttp.",
    "numpy.",
    "scipy.",
    "pandas.",
    "torch.",
    "sentence_transformers.",
    "transformers.",
    "tokenizers.",
    "huggingface_hub.",
    "sklearn.",
    "tqdm.",
    "rich.",
    "click.",
    "typer.",
    "jinja2.",
    "yaml.",
    "toml.",
    "tomli.",
    "mcp.",
    "tree_sitter.",
    "tree_sitter_typescript.",
    "tree_sitter_javascript.",
    "concurrent.",
    "concurrent.futures.",
    "ast.",
    "fcntl.",
    "_fcntl.",
    "monkeypatch.",
    "psutil.",
    "scip_pb2.",
    # ── Dart core / Flutter / common packages (prefix form) ──
    "dart.",
    "dart:",
    "package:",
    "Flutter.",
    "Material.",
    "Cupertino.",
    "Widgets.",
    "Navigator.",
    "Scaffold.",
    "Theme.",
    "MediaQuery.",
    "Platform.",
    "Process.",
    "File.",
    "Directory.",
    "Uri.",
    "DateTime.",
    "Duration.",
    "Future.",
    "Stream.",
    "List.",
    "Map.",
    "Set.",
    "String.",
    "int.",
    "double.",
    "bool.",
    "num.",
    "Object.",
    "Iterable.",
    "Iterator.",
    "RegExp.",
    "math.",
    "async.",
    "convert.",
    "io.",
    "isolate.",
    "typed_data.",
    "collection.",
    "developer.",
    "mirrors.",
    "riverpod.",
    "Provider.",
    "Riverpod.",
    "bloc.",
    "Bloc.",
    "Cubit.",
    "get.",
    "Get.",
    "GetX.",
    # ── OpenSSL C API. Ubiquitous in C/C++ projects that ship TLS/
    # crypto (puma's miniSSL, any HTTPS client, many database
    # drivers). Prefix catches every function in the public surface.
    "BIO_",
    "ERR_",
    "EVP_",
    "SSL_",
    "SSL_CTX_",
    "X509_",
    "RSA_",
    "DSA_",
    "EC_",
    "AES_",
    "DES_",
    "MD5_",
    "SHA1_",
    "SHA256_",
    "PKCS12_",
    "PKCS7_",
    "PEM_",
    "ASN1_",
    "OPENSSL_",
    "CRYPTO_",
    "OBJ_",
    "BN_",
    "HMAC_",
    "EC_KEY_",
    "EC_GROUP_",
    "EC_POINT_",
    "ECDH_",
    "ECDSA_",
    "ENGINE_",
    "CONF_",
    "i2d_",
    "d2i_",
    "DH_",
    "DSA_SIG_",
    "ECDSA_SIG_",
    "X509V3_",
    "X509_NAME_",
    "X509_STORE_",
    "X509_REQ_",
    "X509_CRL_",
    "CMS_",
    "OCSP_",
    "TLS_",
    "DTLS_",
    "ERR_clear_error",
    "ERR_get_error",
    "ERR_peek_error",
    "ERR_load_",
    "ERR_free_strings",
    "OPENSSL_add_all_algorithms_noconf",
    "OPENSSL_cleanup",
    # ── TCL C API (sqlite, other C projects with TCL bindings) ──
    "Tcl_",
    # ── LLVM C API (postgres JIT, many compiler/JIT projects) ──
    "LLVM",
    # ── CPython C API (any Python C extension) ──
    "Py_",
    "PyObject_",
    "PyDict_",
    "PyList_",
    "PyTuple_",
    "PyLong_",
    "PyUnicode_",
    "PyFloat_",
    "PyBool_",
    "PyNumber_",
    "PySequence_",
    "PyMapping_",
    "PyBytes_",
    "PyByteArray_",
    "PyIter_",
    "PyCallable_",
    "PyImport_",
    "PyModule_",
    "PyErr_",
    "PyExc_",
    "PyType_",
    "PyCapsule_",
    "PyCFunction_",
    "PyMethod_",
    "PyEval_",
    "PyRun_",
    "PyArg_",
    "PyThread_",
    "PyThreadState_",
    "PyGILState_",
    "PyMem_",
    "PyOS_",
    "PyStructSequence_",
    "PyBuffer_",
    "PyWeakref_",
    "PyGen_",
    "PyCode_",
    "PyFrame_",
    "PySet_",
    "PyFrozenSet_",
    "PyComplex_",
    "PyDateTime_",
    "PyTime_",
    # ── Win32 C API (cross-platform C/C++ that targets Windows) ──
    "CreateFile",
    "ReadFile",
    "WriteFile",
    "CloseHandle",
    "CreateProcess",
    "TerminateProcess",
    "WaitForSingleObject",
    "WaitForMultipleObjects",
    "CreateMutex",
    "CreateSemaphore",
    "CreateEvent",
    "ReleaseMutex",
    "ReleaseSemaphore",
    "SetEvent",
    "ResetEvent",
    "CreateThread",
    "ExitThread",
    "GetCurrentThread",
    "GetCurrentThreadId",
    "GetCurrentProcess",
    "GetCurrentProcessId",
    "GetLastError",
    "SetLastError",
    "FormatMessage",
    "LocalAlloc",
    "LocalFree",
    "GlobalAlloc",
    "GlobalFree",
    "HeapAlloc",
    "HeapFree",
    "HeapCreate",
    "HeapDestroy",
    "VirtualAlloc",
    "VirtualFree",
    "VirtualProtect",
    "LoadLibrary",
    "LoadLibraryEx",
    "FreeLibrary",
    "GetProcAddress",
    "GetModuleHandle",
    "GetModuleFileName",
    "RegOpenKeyEx",
    "RegCloseKey",
    "RegQueryValueEx",
    "RegSetValueEx",
    "RegCreateKeyEx",
    "RegDeleteKey",
    "RegDeleteValue",
    "MessageBox",
    "SendMessage",
    "PostMessage",
    "DispatchMessage",
    "GetMessage",
    "TranslateMessage",
    "CopyMemory",
    "ZeroMemory",
    "FillMemory",
    "MoveMemory",
    # ── jemalloc internals. jemalloc is a widely-vendored C allocator
    # shipped with many C/C++ projects; its public/private symbols
    # leak into call graphs because they're compiled into the same
    # translation units. They're never project symbols.
    "mallctl",
    "mallctlbymib",
    "mallctlnametomib",
    "malloc_mutex_",
    "malloc_stats_",
    "malloc_printf",
    "malloc_vcprintf",
    "malloc_write",
    "xallocx",
    "mallocx",
    "rallocx",
    "sallocx",
    "dallocx",
    "sdallocx",
    "nallocx",
    "tsd_tsdn",
    "tsd_tcache_",
    "tsd_arena_",
    "tsd_",
    "tsdn_",
    "arena_",
    "arenas_",
    "emitter_",
    "tcache_",
    "extent_",
    "edata_",
    "emap_",
    "chunk_",
    "bin_",
    "large_",
    "sz_",
    "pa_",
    "pai_",
    "pac_",
    "hpa_",
    "hpdata_",
    "prof_",
    "atomic_load_zu",
    "atomic_store_zu",
    "atomic_fetch_add_zu",
    "atomic_fetch_sub_zu",
    "atomic_load_u",
    "atomic_store_u",
    "atomic_fetch_add_u",
    "atomic_fetch_sub_u",
    "atomic_load_p",
    "atomic_store_p",
    "atomic_compare_exchange_weak_zu",
    "atomic_compare_exchange_strong_zu",
    "witness_",
    "background_thread_",
    "ehooks_",
    "base_",
    "mutex_prof_",
    "nstime_",
    "fxp_",
    "ckh_",
    "ql_",
    "qr_",
    "ph_",
    "rtree_",
    # ── gRPC-generated Go boilerplate: every .pb.go service surfaces a
    # mustEmbedUnimplemented<Server> forward-compat stub that users never
    # call explicitly; filtering by prefix catches the full family.
    "mustEmbedUnimplemented",
    # ── Ruby MRI C API (rb_* functions, RSTRING_PTR, NIL_P, etc.) —
    # the canonical interface for Ruby C extensions, used by any gem
    # that ships a native extension. Called from .c files; never
    # project symbols.
    "rb_",
    "RARRAY_",
    "RSTRING_",
    "RHASH_",
    "RTEST",
    "NIL_P",
    "StringValue",
    "StringValuePtr",
    "StringValueCStr",
    "TypedData_",
    "Data_Get_Struct",
    "Data_Make_Struct",
    "RUBY_TYPED_",
    "FIX2",
    "INT2FIX",
    "INT2NUM",
    "NUM2INT",
    "NUM2LONG",
    "NUM2ULONG",
    "ID2SYM",
    "SYM2ID",
    "CLASS_OF",
    "SPECIAL_CONST_P",
    "RB_TYPE_P",
    "BUILTIN_TYPE",
    # ── JRuby Java-side API. These `Ruby*` classes only exist in
    # JRuby extensions (ByteList is JRuby-specific too). Prefix-based
    # so `RubyString.newString(...)`, `ByteList.plain(...)` all match.
    "RubyString.",
    "RubyArray.",
    "RubyHash.",
    "RubyObject.",
    "RubyBasicObject.",
    "RubyInteger.",
    "RubyFixnum.",
    "RubyBignum.",
    "RubyFloat.",
    "RubyRange.",
    "RubyRegexp.",
    "RubySymbol.",
    "RubyProc.",
    "RubyMethod.",
    "RubyClass.",
    "RubyModule.",
    "RubyException.",
    "RubyIO.",
    "RubyFile.",
    "RubyDir.",
    "RubyTime.",
    "RubyThread.",
    "RubyEncoding.",
    "RubyNumeric.",
    "RubyNil.",
    "RubyBoolean.",
    "RubyStruct.",
    "RubyMatchData.",
    "RubyKernel.",
    "RubyComparable.",
    "RubyEnumerable.",
    "RubyBinding.",
    "RubyRuntime.",
    "RubyContinuation.",
    "RubyConverter.",
    "ByteList.",
    "IRubyObject.",
    "ThreadContext.",
    "DynamicMethod.",
    "StaticScope.",
    "JavaProxy.",
    "JavaObject.",
    "JavaClass.",
    "JavaMethod.",
    "JavaField.",
    "JavaConstructor.",
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
        if node["id"].startswith(file_id + "::") and node["type"] in (
            "Function",
            "Method",
        ):
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


_GENERATED_MARKERS = (
    # Go convention (https://pkg.go.dev/cmd/go#hdr-Generate_Go_files_by_processing_source)
    "// Code generated ",
    "DO NOT EDIT",
    # C/C++ autogenerated
    "/* Automatically generated",
    "/* This file was generated",
    # Python (oneof from common codegens)
    "# This file was automatically generated",
    "# This file is autogenerated",
    "# Generated by ",
    "# AUTO-GENERATED",
    "# Automatically generated",
    # TS / JS (various tools). Note: `/* eslint-disable */` alone is
    # intentionally NOT used as a marker — it appears in hand-written
    # files that suppress lint rules and would false-positive.
    "// This file was automatically generated",
    "// This file is autogenerated",
    "// Generated by ",
    "// @generated",
    # Java
    "@Generated",
    # PHP
    "/** @codegen",
)


def is_generated_source(content: str, first_n_lines: int = 30) -> bool:
    """True if the file looks like autogenerated code.

    Checked against the first ``first_n_lines`` lines — generators
    almost always stamp a marker comment in the header (Go's
    well-known "// Code generated ... DO NOT EDIT", protoc-gen-go,
    C autoconf output, @generated from Buck/Facebook, etc.). False
    positives here only cost us the CALLS edges from that file; node
    extraction still runs so cross-file references can still resolve.
    """
    if not content:
        return False
    head = "\n".join(content.splitlines()[:first_n_lines])
    return any(marker in head for marker in _GENERATED_MARKERS)


def _strip_generics(callee: str) -> str:
    """Remove Rust/TS generic arguments (`::<T>`, `<T, U>`) from a callee, so
    `matches.get_one::<String>` reads as `matches.get_one`."""
    bare = callee
    if "<" in bare:
        # Innermost pairs first, repeated, so nested generics come out too.
        prev = None
        while prev != bare:
            prev = bare
            bare = re.sub(r"<[^<>]*>", "", bare)
        # `Vec::<T>::new` leaves `Vec::::new`; a trailing generic leaves `::`.
        bare = re.sub(r":{3,}", "::", bare).rstrip(":")
    return bare


def is_non_project_call(callee: str) -> bool:
    """Check if a callee is a non-project call (stdlib or third-party library) that should be filtered."""
    # Direct match
    if callee in STDLIB_FILTER:
        return True

    bare = _strip_generics(callee)

    # Extract last component for method calls. Split on `.` (most langs),
    # `::` (Rust/C++/Ruby), and `->` (PHP / C struct pointer). We want
    # the trailing bare method name regardless of how the language
    # qualifies it, so `headers->set` matches `set` in the filter.
    last_part = bare.split(".")[-1].split("::")[-1].split("->")[-1]

    # Diesel DSL column access pattern: table_name::column.method (e.g., targets::id.eq)
    # These are ORM DSL calls that never resolve to project functions
    if "::" in callee and "." in callee:
        parts = callee.split("::")
        if len(parts) == 2 and "." in parts[1]:
            col_part = parts[1].split(".")[0]
            # Both lowercase, and the first plural like a table name — which
            # keeps crate::helpers.load / self::state.reset out of this bucket.
            if parts[0].islower() and parts[0].endswith("s") and col_part.islower():
                return True

    # For qualified calls like Type::new, only filter if the type is also stdlib
    # This allows AppState::new but filters Vec::new, HashMap::new
    if "::" in bare:
        type_part = bare.split("::")[-2] if "::" in bare else ""
        # If qualified with a custom type (not stdlib and not the
        # language's "same class" keyword — `Self` in Rust, `self`,
        # `static`, `parent` in PHP), don't filter. These keywords must
        # fall through to the last_part check so that `self::assertSame`
        # (PHP), `Self::new` (Rust), etc. get filtered when their last
        # part is in STDLIB_FILTER, while `self::project_method` passes
        # through as a project call.
        self_like = {"Self", "self", "static", "parent", "this"}
        if (
            type_part
            and type_part not in self_like
            and type_part not in STDLIB_FILTER
            and not any(callee.startswith(p) for p in STDLIB_PREFIXES)
        ):
            return False

    if last_part in STDLIB_FILTER:
        return True

    # Prefix match for qualified names
    return any(callee.startswith(prefix) for prefix in STDLIB_PREFIXES)


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
        root = name.split(".")[0]
        if root in self.imports:
            _, import_type = self.imports[root]
            return import_type == "type"
        return False

    def is_namespace_call(self, name: str) -> bool:
        """Check if name is a call on a namespace import (e.g., api.get).

        Namespace calls like 'schedulesApi.list' are qualified names that
        SCIP can resolve when given the full qualified form.
        """
        if "." not in name:
            return False
        root = name.split(".")[0]
        return root in self.namespaces

    def get_import_source(self, name: str) -> str | None:
        """Get the import source for a name if it was imported.

        Args:
            name: Local name to look up

        Returns:
            Module path if imported, None otherwise
        """
        root = name.split(".")[0]
        if root in self.imports:
            module, _ = self.imports[root]
            return module
        if root in self.namespaces:
            return self.namespaces[root]
        return None


_LICENSE_HEADER_RE = re.compile(r"copyright|licen[cs]e|spdx-", re.IGNORECASE)
# What a count of braces must skip: "..." and `...` strings, '{' / '}' char
# literals and // comments. A // right after `:`, a quote or `}` is left
# alone: that is a URL, a quoted "//" or template text, not a comment. A
# backslash-escaped character outside a string (a regex literal's `\}`, an
# escaped quote) is consumed so it neither counts nor opens a string.
_NON_CODE_RE = re.compile(
    r"""\\.|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`|'[{}]'|(?<![:'"`}])//.*"""
)


_LITERAL_RE = re.compile(
    r"""//|/\*|(?<![A-Za-z0-9_])[bc]?r(#*)"|\"{3}|"|`|'(?:[^'\\]|\\.)'"""
)
_STRING_END_RE = re.compile(r'(?:[^"\\]|\\.)*"')


def blank_strings(line, inside, spanning):
    """Return ``line`` without its comments and with its string and
    character literals turned to spaces, so that text in one is not read as
    code, and what it ends inside of, if anything.

    ``inside`` is what the line starts inside of: `/*`, the opener of a
    literal named in ``spanning``, or None. ``spanning`` lists the literals
    that can continue on the next line: a backquote for a raw string, `"` for
    an ordinary one, `r` for `r#"..."#` and three quotes for a text block.
    """
    code = []
    position = 0
    spans = True
    while True:
        start = position
        if inside is None:
            match = _LITERAL_RE.search(line, position)
            if not match or match.group() == "//":
                code.append(line[position : match.start() if match else None])
                return "".join(code), None
            opener = match.group()
            start = match.start()
            code.append(line[position:start])
            position = match.end()
            if opener.startswith("'"):
                code.append(" " * len(opener))
                continue
            if match.group(1) is not None and "r" in spanning:
                inside = "r" + match.group(1)
            elif opener in ("/*", "`") or opener in spanning:
                inside = opener
            else:
                # A prefix or two more quotes are no part of the delimiter.
                quote = start + opener.index('"')
                code.append(line[start:quote])
                start, position, inside = quote, quote + 1, '"'
            spans = inside == "/*" or inside[0] in spanning or inside in spanning
        if inside == '"':
            end = _STRING_END_RE.match(line, position)
            end = end.end() if end else -1
        else:
            closer = {"/*": "*/"}.get(inside, inside.replace("r", '"', 1))
            end = line.find(closer, position)
            end = end + len(closer) if end >= 0 else -1
        if end < 0:
            code.append(" " * (len(line) - start))
            return "".join(code), inside if spans else None
        code.append(" " * (end - start))
        position, inside = end, None


def _scip_columns(prefix: str):
    """Column a SCIP index gives the character after ``prefix`` on its line:
    one number, or ``(UTF-8, UTF-16, UTF-32)`` code-unit offsets where those
    differ, in the order of SCIP's position encodings."""
    if prefix.isascii():
        return len(prefix)
    return (
        len(prefix.encode()),
        len(prefix.encode("utf-16-le")) // 2,
        len(prefix),
    )


class BaseParser:
    # Prefix of a doc-comment line, and the characters stripped from one.
    doc_line_prefix = "//"
    doc_strip_chars = "/!* "

    # Matches the `name =` of a statement whose value types ``name``, in a
    # ``name`` group; an ``unwrap`` group matches where the value is awaited.
    binding_re = None

    def __init__(self, builder):
        self.builder = builder
        self.declared = {}
        self._line = 0
        self._code = ""
        self._pairs = {}  # column of a `(` on the line -> column of its `)`
        self._closes = {}  # column of a `)` -> edge index of the call it ends
        self._opens = {}  # column of a `(` left open on the line -> edge index
        self._open_calls = []  # edge index of each call open across lines
        self._tail = None  # edge index of the call the previous line ends with
        self._scope = None  # definition the line being read belongs to
        self._raw = ""  # the line being read, as written
        self._locals = set()  # (definition ID, name of a parameter or variable)
        self._first_node = 0  # where the nodes and edges of the file being
        self._first_edge = 0  # parsed start in the builder's lists
        self._pending = None  # (name, unwrap) of a binding still being read
        # (node ID, first line, last line, parameter names) of each context
        self._contexts = []

    def parse(self, file_path, rel_path, content):
        raise NotImplementedError

    def enter_context(self, node_id, first_line, last_line, params=frozenset()):
        """Make ``node_id``, which spans ``first_line`` to ``last_line``, the
        innermost context. ``params`` are the names of its parameters."""
        self._contexts.append((node_id, first_line, last_line, params))

    def context_at(self, lineno):
        """``(node ID, first line)`` of the innermost context entered that
        reaches line ``lineno``. Lines are asked for in ascending order."""
        while lineno > self._contexts[-1][2]:
            self._contexts.pop()
        return self._contexts[-1][:2]

    @staticmethod
    def leave_contexts(contexts, levels, level):
        """Pop the innermost context, never the first, for as long as its
        entry in ``levels``, the brace depth or indentation it was entered
        at, is ``level`` or more."""
        while len(contexts) > 1 and level <= levels[-1]:
            contexts.pop()
            levels.pop()

    def scan_declarations(self, lines, patterns):
        """Record every declaration in ``lines`` that one of ``patterns``
        reads as ``name`` having type ``type``, or as type parameter
        ``param`` being bounded by ``type``."""
        # name -> [(lineno, origin of its value, at file level)], and
        # <param> -> [(lineno, bound, at file level)]
        self.declared = {}
        for lineno, line in enumerate(lines, 1):
            for pattern in patterns:
                for match in pattern.finditer(line):
                    groups = match.groupdict()
                    # A qualified type is one step of an origin path.
                    declared_type = groups["type"].replace(".", "::")
                    # Unindented and not in a parameter list.
                    file_level = (
                        not line[:1].isspace() and "(" not in line[: match.start()]
                    )
                    if groups.get("generic"):
                        declared_type += "<>"
                    if groups.get("name"):
                        key, value = groups["name"], f"{declared_type}()"
                    else:
                        key, value = f"<{groups['param']}>", declared_type
                    self.declared.setdefault(key, []).append(
                        (lineno, value, file_level)
                    )

    def _declared(self, name, lineno, since, anywhere=False):
        """What ``name`` is declared as at ``lineno``: by its last
        declaration from line ``since``, the start of the enclosing
        definition. Failing that what its file-level declarations agree on.
        ``anywhere`` looks at every declaration instead, for a field or a
        type parameter."""
        declarations = self.declared.get(name, ())
        if not anywhere:
            local = [d for d in declarations if since <= d[0] <= lineno]
            if local:
                return max(local, key=lambda d: d[0])[1]
        values = {v for _, v, top in declarations if anywhere or top or not since}
        return values.pop() if len(values) == 1 else None

    def call_origin(self, callee, lineno, since):
        """Origin path (see ``PythonParser._origin``) of a `receiver.method`
        callee whose receiver has a declared type or was bound to a call's
        result; None for any other. ``since`` is the first line of the
        enclosing definition, or 0 when the parser does not track one."""
        receiver, _, method = callee.rpartition(".")
        field = receiver.removeprefix("this.").removeprefix("self.")
        value = self._declared(field, lineno, since, field != receiver)
        if value is None:
            return None
        if not value.startswith("#"):
            # A type parameter stands for its bound.
            param = f"<{value[:-2]}>"
            bound = self._declared(param, lineno, since)
            bound = bound or self._declared(param, lineno, since, True)
            value = f"{bound}()" if bound else value
        return f"{value}.{method}"

    def begin_line(self, code, lineno, scope=None, raw=None):
        """Start reading the calls of ``code``, source line ``lineno``. Call
        this for every line whose calls are recorded, in order. ``scope`` is
        the definition a variable the line binds is local to, and ``raw``
        the line as written where ``code`` has its literals blanked."""
        self._end_line()
        self._line, self._code = lineno, code
        self._scope, self._raw = scope, raw or code
        self._pairs, self._closes = {}, {}
        if code[:1].strip():
            # An unindented line starts or ends a top-level item.
            self._open_calls.clear()
        stack = []
        for column, char in enumerate(code):
            if char == "(":
                stack.append(column)
            elif char == ")" and stack:
                self._pairs[stack.pop()] = column
            elif char == ")" and self._open_calls:
                self._closes[column] = self._open_calls.pop()
        self._opens = dict.fromkeys(stack)

    def _end_line(self):
        """Carry what the finished line leaves open over to the next: its
        unclosed calls, the call it ends with, and what it binds."""
        self._open_calls.extend(self._opens.values())
        statement = self._code.rstrip().rstrip(";").rstrip()
        value = statement.rstrip("?")
        chain = value.rstrip(".").rstrip()
        self._tail = self._closes.get(len(chain) - 1) if chain else None
        binding = self.binding_re.search(self._code) if self.binding_re else None
        if binding:
            name, unwrap = binding["name"], bool(binding.groupdict().get("unwrap"))
            expression = self._code[binding.end() :]
            if self._scope:
                self.add_local(self._scope, name)
        elif self._pending and self._code.lstrip()[:1] in (".", ")"):
            name, unwrap = self._pending
            expression = self._code
        else:
            name = None
        self._pending = (name, unwrap) if name else None
        self._code = ""
        if name is None or self._tail is None or not value.endswith(")"):
            return
        # Bind only where the whole value is one call chain.
        flat, previous = expression, None
        while flat != previous:
            previous, flat = flat, re.sub(r"\([^()]*\)", "", flat)
        if re.fullmatch(
            r"[\s)]*(?:(?:await|new)\s+|&(?:mut\s+)?)*[\w$.:?\s]*;?\s*", flat
        ):
            unwrap = unwrap or value != statement
            origin = f"#{self._tail}.?" if unwrap else f"#{self._tail}"
            self.declared.setdefault(name, []).append((self._line, origin, False))

    def call_closed_at(self, column):
        """The call whose `)` is at ``column`` of the line being read, as
        ``(edge index, column of its "(")``, else None."""
        edge = self._closes.get(column)
        opened = [start for start, end in self._pairs.items() if end == column]
        return (edge, opened[0]) if edge is not None and opened else None

    def continue_chain(self, column, edge):
        """Read a call that follows the character at ``column`` of the line
        being read as made on what the call at edge index ``edge`` returned."""
        self._closes[column] = edge

    def add_local(self, scope, name):
        """Record that ``name`` is a parameter or variable of ``scope``: a
        bare call of it there calls a value, not a definition of that name
        from outside ``scope``."""
        self._locals.add((scope, name))

    def add_call(self, source_id, callee, lineno, since, start, paren, origin=None):
        """Record a call of ``callee`` from ``source_id`` on ``lineno``, the
        line given to ``begin_line``. ``start`` and ``paren`` are the columns
        in it of the callee and of the `(` after it. A call on the result of
        another call (`a().b()`, or `.b()` on the next line) gets an origin
        path that names that call by its edge index, unless ``origin`` gives
        one, and the edge carries the column of the called name for a SCIP
        lookup."""
        if origin is None:
            before = self._code[:start].rstrip()
            if before.endswith(".") and not before.endswith(".."):
                receiver = before[:-1].rstrip()
                edge = (
                    self._closes.get(len(receiver.rstrip("?")) - 1)
                    if receiver
                    else self._tail
                )
                unwrap = ".?" if receiver.endswith("?") else ""
                origin = (
                    f"#{edge}{unwrap}.{callee}" if edge is not None else f"?.{callee}"
                )
        if origin is None and (source_id, callee) in self._locals:
            origin = f"={callee}"
        origin = origin or self.call_origin(callee, lineno, since)
        metadata = {"lineno": lineno}
        if origin:
            metadata["origin"] = origin
        column = start + len(callee) - len(_split_callee(callee)[1])
        metadata["col"] = _scip_columns(self._raw[:column])
        index = len(self.builder.edges)
        self.builder.edges.append(
            {
                "source": source_id,
                "target": f"REF:{callee}",
                "relation": "CALLS",
                "metadata": metadata,
            }
        )
        if paren in self._pairs:
            self._closes[self._pairs[paren]] = index
        elif paren in self._opens:
            self._opens[paren] = index

    def add_config_node(
        self,
        file_id,
        config_id,
        name,
        *,
        signature,
        docstring,
        config_type,
        lineno,
        end_lineno,
        token_count,
        **extra,
    ):
        """Register a Configuration node and the DEFINES edge from its file."""
        self.builder.add_node(
            config_id,
            "Configuration",
            name=name,
            signature=signature,
            lineno=lineno,
            end_lineno=end_lineno,
            token_count=token_count,
            docstring=docstring,
            config_type=config_type,
            **extra,
        )
        self.builder.add_edge(file_id, config_id, "DEFINES")

    def add_file_node(self, rel_path, content):
        """Register the File node for a source file and return its id."""
        file_id = f"FILE:{rel_path}"
        self._first_node = len(self.builder.nodes)
        self._first_edge = len(self.builder.edges)
        self.builder.add_node(
            file_id,
            "File",
            path=rel_path,
            name=Path(rel_path).name,
            token_count=len(content) // 4,
        )
        return file_id

    def set_file_metadata(self, file_id, key, value):
        """Set ``key`` in the metadata of the File node ``file_id``."""
        for node in reversed(self.builder.nodes):
            if node["id"] == file_id:
                node["metadata"][key] = value
                break

    def rename_node(self, old_id, new_id):
        """Give the node ``old_id`` of the file being parsed, and all that
        is defined inside it, the ID ``new_id``."""

        def moved(node_id):
            if node_id == old_id or node_id.startswith(f"{old_id}::"):
                return new_id + node_id[len(old_id) :]
            return node_id

        for node in self.builder.nodes[self._first_node :]:
            if node["id"] != moved(node["id"]):
                self.builder.node_registry.discard(node["id"])
                node["id"] = moved(node["id"])
                self.builder.node_registry.add(node["id"])
        for edge in self.builder.edges[self._first_edge :]:
            edge["source"] = moved(edge["source"])
            edge["target"] = moved(edge["target"])

    def get_leading_docstring(self, lines, start_idx):
        doc_lines = []
        # Walking upward: True between a block comment's `*/` and its `/*`.
        # Outside one, a line starting with `*` is code (`*ptr = value;`).
        in_block = False
        j = start_idx - 1
        while j >= 0:
            line = lines[j].strip()
            if in_block:
                if line.startswith("/*"):
                    in_block = False
                elif not line.startswith("*"):
                    break
            elif line.endswith("*/") and line.startswith(("*", "/*")):
                in_block = not line.startswith("/*")
                # Skip end of block comments if they don't contain content
                if line == "*/":
                    j -= 1
                    continue
            elif not line.startswith(self.doc_line_prefix):
                break
            cleaned = line.lstrip(self.doc_strip_chars).strip()
            doc_lines.insert(0, cleaned)
            j -= 1
        docstring = "\n".join(doc_lines)
        # A licence notice running from the top of the file documents the
        # file, not the first declaration under it.
        if j < 0 and _LICENSE_HEADER_RE.search(docstring):
            return ""
        return docstring

    @staticmethod
    def opening_brace_line(lines, index, code):
        """Index of the line with the `{` that opens the block declared on
        ``lines[index]``. ``code`` is that line without its comments; where
        it has no `{` the next nine lines are searched. None if none of
        them has one either."""
        if "{" in code:
            return index
        for ahead in range(index + 1, min(index + 10, len(lines))):
            if "{" in lines[ahead]:
                return ahead
        return None

    def _find_block_end(self, lines, start_idx):
        """Find the closing brace line for a block starting at start_idx.

        Counts braces to find the matching closing brace, skipping
        comments and string/char literals.
        Returns the 1-indexed line number of the closing brace.
        """
        brace_depth = 0
        found_open = False
        in_comment = False

        for i in range(start_idx, len(lines)):
            line, in_comment = strip_block_comments(lines[i], in_comment)
            line = _NON_CODE_RE.sub("", line)
            for char in line:
                if char == "{":
                    brace_depth += 1
                    found_open = True
                elif char == "}":
                    brace_depth -= 1
                    # On the first line a brace pair may belong to the
                    # signature (`f(opts: { a: T }) {`); only its end counts.
                    if found_open and brace_depth == 0 and i > start_idx:
                        return i + 1  # 1-indexed line number
            if found_open and brace_depth == 0:
                return i + 1

        # If no closing brace found, estimate based on file end
        return min(start_idx + 100, len(lines))


_PY_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# Compound statements whose blocks can hold definitions and imports.
_PY_COMPOUND_STMTS = (
    ast.If,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Match,
)


def _py_block_statements(node):
    """Yield the statements in every block of a compound statement."""
    for block in ("body", "orelse", "finalbody"):
        yield from getattr(node, block, ())
    for handler in getattr(node, "handlers", ()):
        yield from handler.body
    for case in getattr(node, "cases", ()):
        yield from case.body


_PY_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
# Nodes whose `name` they bind: `except E as name`, `case ... as name`, `*name`.
_PY_NAME_BINDERS = (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)


def _py_import_bindings(node):
    """Yield ``(local name, dotted target)`` for each name an import statement
    binds. A relative target keeps its leading dots."""
    for alias in node.names:
        if isinstance(node, ast.Import):
            root = alias.name.split(".")[0]
            yield (alias.asname, alias.name) if alias.asname else (root, root)
        else:
            module = "." * node.level + (node.module or "")
            sep = "." if node.module else ""
            yield alias.asname or alias.name, f"{module}{sep}{alias.name}"


def _py_nested_definitions(statements):
    """Yield the definitions among ``statements``, looking through compound
    statements but not into other definitions."""
    for stmt in statements:
        if isinstance(stmt, _PY_DEFINITIONS):
            yield stmt
        elif isinstance(stmt, _PY_COMPOUND_STMTS):
            yield from _py_nested_definitions(_py_block_statements(stmt))


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
        except (ValueError, RecursionError) as e:
            logger.warning(f"Parse error in {rel_path}: {e}")
            return

        self.add_file_node(rel_path, content)
        self.source_lines = content.split("\n")
        self.class_ids = set()
        self.class_attrs = [{}]
        self.import_targets = {}
        self.bindings = []
        self._push_bindings(tree.body)
        self.visit_node(tree, file_id)
        self.builder.imports[file_id] = self.import_targets

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
            self.class_ids.add(class_id)

            # Track decorator calls (e.g., @dataclass, @pytest.mark.parametrize(...))
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    self._record_call(decorator, class_id, decorator.lineno)
                elif isinstance(decorator, ast.Attribute):
                    # Handle @module.decorator without call parens
                    self._record_callee(decorator, class_id, decorator.lineno)

            for base in node.bases:
                base_name = (
                    self._get_attribute_name(base)
                    if isinstance(base, ast.Attribute)
                    else (base.id if isinstance(base, ast.Name) else None)
                )
                if base_name:
                    resolved_base = self.builder.current_scope.resolve(base_name)
                    self.builder.add_edge(class_id, f"REF:{resolved_base}", "INHERITS")

            self.class_attrs.append(self._class_attrs(node))
            for child in node.body:
                self.visit_node(child, class_id)
            self.class_attrs.pop()

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_id = f"{parent_id}::{node.name}"
            # D.1: a function is a method iff its enclosing parent node is a Class.
            # Previous logic used string matching on parent_id which incorrectly
            # classified nested functions inside other functions as Methods.
            is_method = parent_id in self.class_ids
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
                else:
                    self._record_callee(decorator, func_id, decorator.lineno)

            self._push_bindings(node.body, node, is_method)
            fixture = self._fixture_name(node)
            if fixture:
                self.builder.fixtures[parent_id, fixture] = (
                    func_id,
                    self._returned_origin(node),
                )
            self.visit_body_for_calls(node.body, func_id)

            for child in _py_nested_definitions(node.body):
                self.visit_node(child, func_id)
            self.bindings.pop()

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

        elif isinstance(node, _PY_COMPOUND_STMTS):
            for child in _py_block_statements(node):
                self.visit_node(child, parent_id)

    def visit_body_for_calls(self, body, parent_func_id):
        """Find all Call nodes in the function body.

        This catches calls in all contexts: comprehensions, generators,
        with statements, if/while conditions, for iterables, decorators,
        nested calls, await expressions, etc. Nested function and class
        definitions are not entered: visit_node records their calls.
        """
        todo = deque(body)
        while todo:
            node = todo.popleft()
            if isinstance(node, _PY_DEFINITIONS):
                continue
            todo.extend(ast.iter_child_nodes(node))
            if isinstance(node, ast.Call):
                self._record_call(node, parent_func_id, node.lineno)

    def _record_call(self, call_node, parent_func_id, lineno):
        self._record_callee(call_node.func, parent_func_id, lineno)

    def _record_callee(self, func_node, parent_func_id, lineno):
        func_name = self._get_func_name(func_node)
        if func_name:
            resolved_name = self.builder.current_scope.resolve(func_name)
            metadata = {"lineno": lineno, "origin": self._origin(func_node, lineno)}
            if func_node.end_lineno == lineno:
                # The called name ends where the callee expression does. It
                # is read from the line: the AST holds its NFKC form, which
                # can differ in length.
                line = self.source_lines[lineno - 1].encode()
                written = line[: func_node.end_col_offset].decode()
                name = re.search(r"\w*$", written)
                metadata["col"] = _scip_columns(written[: name.start()])
            self.builder.edges.append(
                {
                    "source": parent_func_id,
                    "target": f"REF:{resolved_name}",
                    "relation": "CALLS",
                    "metadata": metadata,
                }
            )

    def _origin(self, node, lineno):
        """Write the expression ``node`` as an origin path for CallResolver:
        dotted names with ``()`` after each call, imported names expanded
        behind ``@``, and local names replaced by what they were bound to
        before ``lineno``. ``?`` stands for anything that has no such path."""
        if isinstance(node, ast.Name):
            for depth, frame in enumerate(reversed(self.bindings)):
                if node.id in frame:
                    bound = [o for at, o in frame[node.id] if depth or at < lineno]
                    return bound[-1] if bound else "?"
            imported = self.import_targets.get(node.id)
            return f"@{imported}" if imported else node.id
        if isinstance(node, ast.Attribute):
            base = self._origin(node.value, lineno)
            if base == "self" and node.attr in self.class_attrs[-1]:
                return self.class_attrs[-1][node.attr]
            return f"{base}.{node.attr}"
        if isinstance(node, ast.Call):
            return f"{self._origin(node.func, lineno)}()"
        if isinstance(node, ast.Await):
            return self._origin(node.value, lineno)
        return "?"

    def _fixture_name(self, func):
        """Name of the pytest fixture ``func`` is decorated as, if any."""
        for decorator in func.decorator_list:
            called = isinstance(decorator, ast.Call)
            target = decorator.func if called else decorator
            if self._origin(target, func.lineno) != "@pytest.fixture":
                continue
            for keyword in decorator.keywords if called else ():
                given = keyword.value
                if keyword.arg == "name" and isinstance(given, ast.Constant):
                    return str(given.value)
            return func.name
        return None

    def _returned_origin(self, func):
        """Origin of the one value ``func`` returns or yields, None when it
        has several or the value has no origin path."""
        origins = set()
        todo = deque(func.body)
        while todo:
            node = todo.popleft()
            if isinstance(node, (ast.Return, ast.Yield)) and node.value:
                origins.add(self._origin(node.value, node.lineno))
            if not isinstance(node, (*_PY_DEFINITIONS, ast.Lambda)):
                todo.extend(ast.iter_child_nodes(node))
        origin = origins.pop() if len(origins) == 1 else "?"
        return None if "?" in origin else origin

    def _instance_origin(self, annotation, lineno):
        """Origin of a value annotated with ``annotation``."""
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                annotation = ast.parse(annotation.value, mode="eval").body
            except SyntaxError:
                return "?"
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            sides = [annotation.left, annotation.right]
            typed = [
                s
                for s in sides
                if not (isinstance(s, ast.Constant) and s.value is None)
            ]
            if len(typed) != 1:
                return "?"
            annotation = typed[0]
        if isinstance(annotation, (ast.Name, ast.Attribute)):
            return f"{self._origin(annotation, lineno)}()"
        return "?"

    def _push_bindings(self, body, func=None, is_method=False):
        """Push a frame mapping each name bound in ``body`` (and by the
        parameters of ``func``) to its ``(lineno, origin)`` bindings in
        source order. Module-level imports go to ``import_targets``."""
        frame = {}
        self.bindings.append(frame)

        def bind(target, lineno, origin):
            if isinstance(target, ast.Name):
                frame.setdefault(target.id, []).append((lineno, origin))
            elif isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    bind(element, lineno, "?")
            elif isinstance(target, ast.Starred):
                bind(target.value, lineno, "?")

        if func is not None:
            args = func.args
            params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            params += [a for a in (args.vararg, args.kwarg) if a]
            decorators = {d.id for d in func.decorator_list if isinstance(d, ast.Name)}
            positional = len(args.posonlyargs) + len(args.args)
            # pytest passes a test or fixture the fixtures its parameters
            # name, unless a `parametrize` mark supplies the argument.
            injected = func.name.startswith("test") or self._fixture_name(func)
            parametrized = {
                name.strip()
                for decorator in func.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "parametrize"
                and decorator.args
                for constant in ast.walk(decorator.args[0])
                if isinstance(constant, ast.Constant)
                and isinstance(constant.value, str)
                for name in constant.value.split(",")
            }
            for index, arg in enumerate(params):
                if (
                    index == 0
                    and positional
                    and is_method
                    and "staticmethod" not in decorators
                ):
                    origin = "cls" if "classmethod" in decorators else "self"
                elif arg.annotation:
                    origin = self._instance_origin(arg.annotation, func.lineno)
                elif injected and arg.arg not in parametrized:
                    origin = f"~{arg.arg}"
                else:
                    origin = "?"
                frame[arg.arg] = [(func.lineno, origin)]

        found = []
        todo = deque(body)
        while todo:
            node = todo.popleft()
            found.append(node)
            if not isinstance(node, _PY_DEFINITIONS):
                todo.extend(ast.iter_child_nodes(node))
        for node in sorted(found, key=lambda n: getattr(n, "lineno", 0)):
            lineno = getattr(node, "lineno", 0)
            if isinstance(node, _PY_DEFINITIONS):
                frame.pop(node.name, None)
            elif isinstance(node, ast.Assign):
                origin = self._origin(node.value, lineno)
                for target in node.targets:
                    bind(target, lineno, origin)
            elif isinstance(node, ast.AnnAssign):
                origin = self._instance_origin(node.annotation, lineno)
                bind(node.target, lineno, origin)
            elif isinstance(node, ast.NamedExpr):
                bind(node.target, lineno, self._origin(node.value, lineno))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for name, target in _py_import_bindings(node):
                    if func is None:
                        self.import_targets[name] = target
                    else:
                        frame.setdefault(name, []).append((lineno, f"@{target}"))
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                bind(node.target, lineno - 1, "?")
            elif isinstance(node, _PY_COMPREHENSIONS):
                for generator in node.generators:
                    bind(generator.target, lineno - 1, "?")
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    bind(item.optional_vars, lineno - 1, "?")
            elif isinstance(node, ast.Lambda):
                for arg in ast.walk(node.args):
                    if isinstance(arg, ast.arg):
                        frame.setdefault(arg.arg, []).append((lineno - 1, "?"))
            elif isinstance(node, _PY_NAME_BINDERS) and node.name:
                frame.setdefault(node.name, []).append((lineno - 1, "?"))

    def _class_attrs(self, class_node):
        """Map each attribute of ``class_node`` to the origin of the value
        its body annotates it with or a method assigns to it."""
        attrs = {}
        for stmt in class_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                origin = self._instance_origin(stmt.annotation, stmt.lineno)
                attrs.setdefault(stmt.target.id, origin)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._push_bindings(stmt.body, stmt, is_method=True)
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Assign):
                        targets, value = node.targets, node.value
                    elif isinstance(node, ast.AnnAssign):
                        targets, value = [node.target], node.annotation
                    else:
                        continue
                    for target in targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and self._origin(target.value, node.lineno) == "self"
                        ):
                            origin = (
                                self._origin(value, node.lineno)
                                if isinstance(node, ast.Assign)
                                else self._instance_origin(value, node.lineno)
                            )
                            attrs.setdefault(target.attr, origin)
                self.bindings.pop()
        return {name: origin for name, origin in attrs.items() if "?" not in origin}

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


_RUST_DECLARATIONS = (
    # `name: Type`, `name: &'a mut Type`, `name: &dyn Trait`, `name: Type<'a>`;
    # not `name: None`, a field of a struct literal. A type with type
    # arguments is marked generic: unless the project defines it (`Arc<T>`),
    # its methods are mostly those of what it wraps.
    re.compile(
        r"\b(?P<name>[a-z_]\w*)\s*:\s*(?:&\s*)?(?:'\w+\s+)?(?:mut\s+)?"
        r"(?:dyn\s+|impl\s+)?(?P<type>(?!None\b)[A-Z]\w*)\b(?![:(!])"
        r"(?:<'\w+>|(?P<generic><))?"
    ),
    # `let name = Type { ... }`
    re.compile(r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*=\s*(?P<type>[A-Z]\w*)\s*\{"),
    # A type parameter's bound, in `<T: Trait>` or a `where` clause
    re.compile(
        r"(?:[<,]|\bwhere\s|^)\s*(?P<param>[A-Z]\w*)\s*:\s*"
        r"(?P<type>[A-Z]\w*)\b(?![<:(!])"
    ),
)


# `type Name = ...` in Rust or TypeScript, behind any visibility or `export`
_TYPE_ALIAS_RE = re.compile(r"^[ \t]*(?:[\w()]+\s+)*type\s+(\w+)[^=;{]*=", re.MULTILINE)


def _use_leaves(tree: str, prefix: str = ""):
    """Yield ``(local name, full path)`` for each name a `use` tree binds:
    `a::{b, c::d as e, f::*}` binds b to `a::b` and e to `a::c::d`. The
    glob is yielded under its own path, `a::f::*`."""
    head, brace, group = tree.strip().partition("{")
    if not brace:
        path, _, alias = (prefix + head).partition(" as ")
        path = path.strip().removesuffix("::self")
        if path.endswith("::*"):
            yield path, path
        elif head.strip():
            yield alias.strip() or path.rpartition("::")[2], path
        return
    depth = 0
    start = 0
    group = group.rpartition("}")[0] + ","
    for column, char in enumerate(group):
        depth += (char == "{") - (char == "}")
        if char == "," and depth == 0:
            yield from _use_leaves(group[start:column], prefix + head.strip())
            start = column + 1


# An optional visibility: pub, pub(crate), pub(super), pub(in path)
_RUST_VIS = r"(?:pub(?:\([^)]*\))?\s+)?"
# A `use` declaration's tree, which may span lines
_RUST_USE_RE = re.compile(rf"^\s*{_RUST_VIS}use\s+([^;]+);", re.MULTILINE)


# A parameter name in a `fn` signature
_RUST_PARAMETER_RE = re.compile(r"[(,]\s*(?:mut\s+)?([a-z_]\w*)\s*:")
# A `fn` with its body on the line of the `impl` block it is in
_RUST_INLINE_FN_RE = re.compile(
    r"\bfn\s+(?P<name>\w+)[^{;]*?(?:->(?P<returns>[^{;]+))?\{(?P<body>[^{}]*)\}"
)


class RustParser(BaseParser):
    binding_re = re.compile(r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*=(?!=)")

    def _generic_path_origin(self, callee):
        """Origin of a path that starts at a type parameter: `T::name` is a
        function of the bound of `T`, `T::Assoc::name` one of a type that
        only an implementation of the bound names."""
        root, *path = callee.split("::")
        if not path or f"<{root}>" not in self.declared:
            return None
        bounds = {bound for _, bound, _ in self.declared[f"<{root}>"]}
        if len(path) == 1 and len(bounds) == 1:
            return f"{bounds.pop()}.{path[0]}"
        return f"?.{path[-1]}"

    def parse(self, file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)

        lines = content.splitlines()
        current_context = [file_id]  # Stack of IDs
        context_indent = [0]  # Stack of indentation levels

        # Regex patterns
        re_fn_start = re.compile(
            rf"^\s*{_RUST_VIS}(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
            r"(?:extern\s+\"[^\"]*\"\s+)?fn\s+([a-zA-Z0-9_]+)"
        )
        re_struct = re.compile(rf"^\s*{_RUST_VIS}(?:struct|trait)\s+([a-zA-Z0-9_]+)")
        re_enum = re.compile(rf"^\s*{_RUST_VIS}enum\s+([a-zA-Z0-9_]+)")
        # Enum variant patterns: Name, Name(Type), Name { field: Type }
        re_enum_variant = re.compile(r"^\s*([A-Z][a-zA-Z0-9_]*)\s*(?:[({\,]|$)")
        re_impl = re.compile(
            r"^\s*impl(?:<.*?>)?\s+(?:([a-zA-Z0-9_:<>,\s]+?)\s+for\s+)?([a-zA-Z0-9_:]+)"
        )
        re_mod = re.compile(rf"^\s*{_RUST_VIS}mod\s+([a-zA-Z0-9_]+)\s*\{{?")
        re_const = re.compile(
            r"^\s*(?:pub(?:\(.*?\))?\s+)?(?:const|static)\s+([A-Z0-9_]+)\s*:\s*.*?\s*=\s*(.*?);"
        )
        re_call = re.compile(r"([a-zA-Z0-9_:.]+)\(")
        self.scan_declarations(lines, _RUST_DECLARATIONS)
        self.builder.type_aliases.update(_TYPE_ALIAS_RE.findall(content))
        for tree in _RUST_USE_RE.findall(content):
            self.builder.imports.setdefault(file_id, {}).update(_use_leaves(tree))

        current_impl_target = None
        current_trait_impl = (
            None  # Track trait name when in "impl Trait for Struct" block
        )
        current_enum_id = None  # Track if we're inside an enum for variant parsing
        impl_targets = {}  # id of each impl block's type -> (name, lineno)
        fn_start = 0  # first line of the function being read
        fn_ids = set()  # functions whose body is a context; a `fn` in one is nested

        def opens_block(text):
            """True if ``text`` leaves a `{` open; `struct P { x: i32 }` does not."""
            return text.count("{") > text.count("}")

        # Each line without comments and literal text, and whether it lies
        # partly inside a comment or literal that spans lines.
        codes = []
        inside = None
        for raw in lines:
            code, ends_inside = blank_strings(raw, inside, ('"', "r"))
            codes.append((code, bool(inside or ends_inside)))
            inside = ends_inside

        i = 0
        while i < len(lines):
            code, spanned = codes[i]
            # The text of a string or comment declares nothing.
            line = code if spanned else lines[i]
            lineno = i + 1
            indent = len(line) - len(line.lstrip())
            # The regex call fallback starts past a name declared on the line.
            call_scan_start = 0

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
                call_scan_start = match.end()
                fn_start = lineno
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

                if current_impl_target and parent_id not in fn_ids:
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
                node_kwargs = {
                    "name": name,
                    "signature": sig,
                    "lineno": lineno,
                    "end_lineno": end_lineno,
                    "token_count": token_count,
                    "docstring": docstring,
                    "return_type": ret,
                    "param_types": param_types,
                }
                if current_trait_impl and node_type == "Method":
                    node_kwargs["trait_impl"] = current_trait_impl

                self.builder.add_node(func_id, node_type, **node_kwargs)
                self.builder.add_edge(parent_id, func_id, "DEFINES")
                for parameter in _RUST_PARAMETER_RE.findall(full_sig):
                    self.add_local(func_id, parameter)

                if opens_block(lines[j]):
                    current_context.append(func_id)
                    context_indent.append(indent)
                    fn_ids.add(func_id)
                i = j

            # 2. Structs/Traits
            elif match := re_struct.search(line):
                name = match.group(1)
                call_scan_start = match.end()
                docstring = self.get_leading_docstring(lines, i)
                type_id = f"{parent_id}::{name}"
                # Find end of struct/trait block
                end_lineno = self._find_block_end(lines, i) if "{" in line else lineno
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    type_id,
                    "Class",
                    name=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                if opens_block(line):
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
                    type_id,
                    "Class",
                    name=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                if opens_block(line):
                    current_context.append(type_id)
                    context_indent.append(indent)
                    current_enum_id = type_id  # Track enum for variant parsing

            # 2c. Enum variants (when inside an enum)
            elif current_enum_id and (match := re_enum_variant.search(line)):
                variant_name = match.group(1)
                call_scan_start = match.end(1)
                # Skip common Rust keywords/patterns that match variant regex
                if variant_name not in (
                    "Self",
                    "Some",
                    "None",
                    "Ok",
                    "Err",
                    "Box",
                    "Vec",
                    "Option",
                    "Result",
                ):
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
                struct_id = f"{file_id}::{current_impl_target}"
                impl_targets.setdefault(struct_id, (current_impl_target, lineno))

                if trait:
                    trait_name = trait.split("<")[0].split("::")[-1]
                    current_trait_impl = trait_name  # Track trait for method metadata
                    self.builder.add_edge(struct_id, f"REF:{trait_name}", "INHERITS")
                else:
                    current_trait_impl = None  # Inherent impl, no trait

                if opens_block(line):
                    current_context.append(struct_id)
                    context_indent.append(indent)
                elif "{" in line:  # `impl Marker for Foo {}`
                    # Methods of a block written on one line.
                    body = line.index("{")
                    self.begin_line(code, lineno, raw=lines[i])
                    for method in _RUST_INLINE_FN_RE.finditer(code, body):
                        method_id = f"{struct_id}::{method['name']}"
                        self.builder.add_node(
                            method_id,
                            "Method",
                            name=method["name"],
                            signature=f"fn {method['name']}(...)",
                            lineno=lineno,
                            end_lineno=lineno,
                            token_count=10,
                            docstring="",
                            return_type=(method["returns"] or "()").strip(),
                        )
                        self.builder.add_edge(struct_id, method_id, "DEFINES")
                        for call_match in re_call.finditer(method["body"]):
                            callee = call_match.group(1).lstrip(".")
                            if callee:
                                end = method.start("body") + call_match.end()
                                self.add_call(
                                    method_id,
                                    callee,
                                    lineno,
                                    lineno,
                                    end - 1 - len(callee),
                                    end - 1,
                                )
                    call_scan_start = len(line)
                    current_impl_target = None
                    current_trait_impl = None

            # 4. Modules
            elif match := re_mod.search(line):
                name = match.group(1)
                mod_id = f"{parent_id}::{name}"
                if opens_block(line):
                    # The module is the source of its items' DEFINES edges.
                    self.builder.add_node(
                        mod_id,
                        "Class",
                        name=name,
                        kind="module",
                        lineno=lineno,
                        docstring=self.get_leading_docstring(lines, i),
                    )
                    self.builder.add_edge(parent_id, mod_id, "DEFINES")
                    current_context.append(mod_id)
                    context_indent.append(indent)

            # 5. Imports
            elif match := _RUST_USE_RE.search(line):
                path = match.group(1).replace("::", ".")
                self.builder.add_edge(file_id, f"MODULE:{path}", "IMPORTS")

            # 6. Calls (regex fallback - only used if ast-grep unavailable)
            if not self.builder.use_ast_grep and parent_id != file_id:
                code = codes[lineno - 1][0]
                self.begin_line(code, lineno, parent_id, lines[lineno - 1])
                for call_match in re_call.finditer(code, call_scan_start):
                    # In `0..items.len()` the callee follows the range's `..`.
                    callee = call_match.group(1).rpartition("..")[2].lstrip(".")
                    # `pub(crate)` is a visibility, not a call.
                    if callee and callee != "pub":
                        self.add_call(
                            parent_id,
                            callee,
                            lineno,
                            fn_start,
                            call_match.end(1) - len(callee),
                            call_match.end() - 1,
                            self._generic_path_origin(callee),
                        )
            i += 1

        # An impl block's edges are sourced at its type. A type declared in
        # another file has no node here, so give it one.
        for struct_id, (name, impl_lineno) in impl_targets.items():
            if struct_id not in self.builder.node_registry:
                self.builder.add_node(
                    struct_id,
                    "Class",
                    name=name,
                    kind="impl",
                    lineno=impl_lineno,
                    docstring="",
                )
                self.builder.add_edge(file_id, struct_id, "DEFINES")

        # Use ast-grep for more accurate CALLS detection
        if self.builder.use_ast_grep:
            self._extract_calls_with_ast_grep(str(file_path), file_id, lines)

    def _extract_calls_with_ast_grep(self, file_path: str, file_id: str, lines: list):
        """Extract calls using ast-grep for higher accuracy."""
        # Build a line-to-context map from our parsed structure
        # Uses span-size ordering to ensure innermost context wins for nested functions
        line_to_context = build_line_to_context_map(self.builder.nodes, file_id)

        for call in extract_calls_rust(file_path):
            lineno = call["lineno"]
            callee = call["callee"]

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
                "new",
                "default",
                "builder",
                "from",
                "try_from",
                "from_str",
                "from_bytes",
                "from_slice",
                "open",
                "create",
                "connect",
                "with_capacity",
                "with_config",
                "with_options",
                "init",
                "initialize",
            )
            if "::" in callee and not is_non_project_call(callee):
                parts = callee.split("::")
                if len(parts) >= 2:
                    method_name = parts[-1]
                    struct_name = parts[-2]
                    # Match constructor patterns (exact or prefix for with_*)
                    is_constructor = (
                        method_name in constructor_patterns
                        or method_name.startswith(("with_", "from_"))
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
        self._extract_struct_literals(file_id, lines, line_to_context)

        # Extract calls from macro bodies (tokio::select!, etc.)
        self._extract_macro_body_calls(file_id, lines, line_to_context)

    def _extract_struct_literals(
        self, file_id: str, lines: list, line_to_context: dict
    ):
        """Extract struct literal instantiations like `MyStruct { field: value }`.

        Uses regex to find patterns that look like struct literals.
        """
        re_struct_literal = re.compile(r"\b([A-Z][a-zA-Z0-9_]*)\s*\{")

        for lineno, line in enumerate(lines, 1):
            # Skip struct/enum/trait definitions
            if re.match(rf"^\s*{_RUST_VIS}(?:struct|enum|trait)\s+", line):
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
        re_await_method = re.compile(r"\.([a-z_][a-z0-9_]*)\s*\(\s*\)\s*\.await")

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
    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)

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
                self.leave_contexts(current_context, context_indent, indent)

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


_TS_DECLARATIONS = (
    # `name: Type` on a parameter (alone on its line or not), variable or field
    re.compile(
        r"(?:^|[(,]|\b(?:const|let|var|private|public|protected|readonly)\s)\s*"
        r"(?P<name>[A-Za-z_$][\w$]*)\??\s*:\s*(?P<type>[A-Z][\w$.]*)\b(?![<(\[])"
    ),
    # `name = new Type(`
    re.compile(r"\b(?P<name>[A-Za-z_$][\w$]*)\s*=\s*new\s+(?P<type>[A-Za-z_$][\w$.]*)"),
    # A type parameter's bound: `<T extends Type`
    re.compile(
        r"[<,]\s*(?P<param>[A-Z]\w*)\s+extends\s+(?P<type>[A-Z][\w$.]*)\b(?![<(])"
    ),
)


_TS_VARIABLE_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)")


def _arrow_function(statement: str) -> tuple[str, str] | None:
    """Read `const name = (params) => ...`, with or without a type
    annotation on ``name``, or `const name = function (params) ...`. Returns
    the name and the function from its parameters on, else None."""
    match = _TS_VARIABLE_RE.match(statement)
    if not match:
        return None
    # The initialiser follows the first `=` outside brackets that is not
    # the `=>` of a function type.
    depth = 0
    for column in range(match.end(), len(statement)):
        char = statement[column]
        if char in "([{<":
            depth += 1
        elif char in ")]}>" and statement[column - 1] != "=":
            depth -= 1
        elif char == "=" and depth == 0 and statement[column + 1 : column + 2] != ">":
            break
    else:
        return None
    value = re.sub(r"^\s*(?:async\b\s*)?", "", statement[column + 1 :])
    if re.match(r"function\b|[A-Za-z_$][\w$]*\s*=>", value):
        return match.group(1), value
    if not value.startswith("("):
        return None
    depth = 0
    for column, char in enumerate(value):
        depth += (char == "(") - (char == ")")
        if depth == 0:
            # `=>` follows the parameters, after any return type.
            rest = value[column + 1 :].lstrip()
            is_arrow = rest.startswith("=>") or (rest.startswith(":") and "=>" in rest)
            return (match.group(1), value) if is_arrow else None
    return None


def _ts_params(signature: str) -> set[str]:
    """Names of the parameters in the first parameter list of ``signature``."""
    names = set()
    depth = 0
    at_start = False
    tokens = re.findall(r"[A-Za-z_$][\w$]*|\.\.\.|=>|\S", signature)
    for token in tokens[tokens.index("(") if "(" in tokens else len(tokens) :]:
        if token in ("(", "<", "[", "{"):
            depth += 1
            at_start = depth == 1
        elif token in (")", ">", "]", "}"):
            depth -= 1
            if depth == 0:
                break
        elif depth == 1 and token == ",":
            at_start = True
        elif at_start and token not in _TS_PARAM_MODIFIERS:
            if token[0].isalpha() or token[0] in "_$":
                names.add(token)
            at_start = False
    return names


_TS_METHOD_MODIFIERS = ("", "async", "static", "get", "set", "*", "async *")
_TS_PARAM_MODIFIERS = ("...", "public", "private", "protected", "readonly")
# `export * from "<specifier>"`, `export { <clause> } from "<specifier>"` and,
# with no specifier, `export { <clause> }`
_TS_REEXPORT_RE = re.compile(
    r"""^\s*export\s+(?:type\s+)?(?:(\*)\s+from\s+|\{([^}]*)\}(?:\s+from\s+)?)"""
    r"""(?:['"]([^'"]+)['"])?""",
    re.MULTILINE,
)
# `import <clause> from "<specifier>"`, the clause possibly over several lines
_TS_IMPORT_RE = re.compile(
    r"""^\s*import\s+(?:type\s+)?([^;'"]*?)\s+from\s+['"]([^'"]+)['"]""", re.MULTILINE
)


# A '...' literal closed on its line; a lone apostrophe, as in JSX text, is not one.
_JS_SINGLE_QUOTED_RE = re.compile(r"'(?:[^'\\]|\\.)*'")


def _blank_js_strings(line: str) -> str:
    """Return ``line`` with the text of "...", '...' and `...` literals turned
    to spaces. The `${...}` expressions of a template literal are kept."""
    out = []
    stack = []  # open literals: a quote character, or a `${`'s brace depth
    i = 0
    while i < len(line):
        ch = line[i]
        top = stack[-1] if stack else None
        if isinstance(top, str):
            if ch == "\\":
                out.append("  ")
                i += 2
                continue
            if top == "`" and line.startswith("${", i):
                stack.append(0)
                out.append("${")
                i += 2
                continue
            if ch == top:
                stack.pop()
            out.append(ch if ch == top else " ")
        else:
            if ch in ('"', "`") or (ch == "'" and _JS_SINGLE_QUOTED_RE.match(line, i)):
                stack.append(ch)
            elif ch == "{" and top is not None:
                stack[-1] += 1
            elif ch == "}" and top is not None:
                if top == 0:
                    stack.pop()
                else:
                    stack[-1] -= 1
            out.append(ch)
        i += 1
    return "".join(out)[: len(line)]


def _opens_body(signature: str) -> bool:
    """True when ``signature`` reads `name(params) {` or `name(params): T {`:
    its first brace outside parentheses directly follows a parameter list."""
    depth = 0
    after_params = False
    for char in signature:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            after_params = depth == 0
        elif depth == 0 and char == "{":
            return after_params
        elif depth == 0 and char in "=;,.":
            after_params = False
    return False


class TSParser(BaseParser):
    binding_re = re.compile(
        r"\b(?:const|let|var)\s+(?P<name>[\w$]+)\s*=(?![=>])\s*(?P<unwrap>await\b)?"
    )

    def parse(self, file_path, rel_path, content, line_offset=0):
        """Parse ``content``: the text of ``file_path``, or one ``<script>``
        block of it that starts ``line_offset`` lines in. Emitted line
        numbers are relative to ``content``."""
        file_id = self.add_file_node(rel_path, content)

        lines = content.splitlines()
        self.line_offset = line_offset
        self.line_count = len(lines)
        self.enter_context(file_id, 0, len(lines))

        # Initialize TypeScript symbol table for import tracking
        self.symbol_table = TypeScriptSymbolTable(
            str(file_path), str(self.builder.root_dir)
        )

        # Extract imports using ast-grep if available
        if self.builder.use_ast_grep and extract_imports_typescript is not None:
            try:
                import_data = extract_imports_typescript(str(file_path))
                self.symbol_table.load_imports(import_data)
            except Exception as e:  # noqa: BLE001 — boundary over the external ast-grep binary; falls back to unqualified calls
                # Fall back to unqualified calls — debug-level so it's
                # traceable without spamming INFO on every parser failure.
                logger.debug(
                    f"ast-grep TS import extraction failed for {file_path}: {e}"
                )

        re_class = re.compile(
            r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([a-zA-Z0-9_$]+)"
        )
        re_interface = re.compile(r"^\s*(?:export\s+)?interface\s+([a-zA-Z0-9_$]+)")
        re_func = re.compile(
            r"^\s*(?:export\s+)?(?:declare\s+)?(?:default\s+)?(?:async\s+)?"
            r"function\s+([a-zA-Z0-9_$]+)"
        )
        re_method = re.compile(
            r"^\s*(?:private|public|protected)?\s*(?:static\s+)?(?:async\s+)?([a-zA-Z0-9_$]+)\s*(?:<[^>]*>)?\s*[(]"
        )
        # Getter/setter methods: get name() or set name(value)
        re_getter_setter = re.compile(
            r"^\s*(?:private|public|protected)?\s*(?:static\s+)?(get|set)\s+([a-zA-Z0-9_$]+)\s*[(]"
        )
        re_enum = re.compile(r"^\s*(?:export\s+)?enum\s+([a-zA-Z0-9_$]+)")
        re_const = re.compile(
            r"^\s*(?:export\s+)?const\s+([a-zA-Z0-9_$]+)\s*(?::\s*.*?)?\s*=\s*(.*?);"
        )
        re_import = re.compile(r'import\s+.*?from\s+([\'"])(.*?)\1')
        re_call_candidate = re.compile(
            r"([a-zA-Z0-9_$]+(?:\.[a-zA-Z0-9_$]+)*)\s*(?=[<(])"
        )
        self.scan_declarations(lines, _TS_DECLARATIONS)
        self.builder.type_aliases.update(_TYPE_ALIAS_RE.findall(content))
        for star, clause, specifier in _TS_REEXPORT_RE.findall(content):
            names = [("*", "")] if star else []
            for part in clause.split(","):
                # `name`, `type name`, `name as alias`
                original, _, alias = part.partition(" as ")
                if original.split():
                    names.append(
                        ((alias or original).split()[-1], original.split()[-1])
                    )
            self.builder.reexports.setdefault(file_id, []).extend(
                (exported, specifier, original) for exported, original in names
            )
        for clause, specifier in _TS_IMPORT_RE.findall(content):
            for part in re.split(r"[{},]", clause):
                # `name`, `type name`, `name as alias`, `* as alias`
                local = part.rpartition(" as ")[2].split()[-1:]
                if local:
                    self.builder.imports.setdefault(file_id, {})[local[0]] = specifier
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

        class_ids = set()  # class/interface/enum nodes declared in this file
        interface_ids = set()  # interfaces that have the ID `<scope>::<name>`
        value_ids = set()  # functions and variables, which always have it

        def enter(node_id, lineno, end_lineno, signature=""):
            """Make a definition that spans further lines the context."""
            if end_lineno > lineno:
                self.enter_context(node_id, lineno, end_lineno, _ts_params(signature))

        in_comment = False
        i = 0
        while i < len(lines):
            line, in_comment = strip_block_comments(lines[i], in_comment)
            line = strip_line_comment(line, backticks=True)
            lineno = i + 1
            # Column of a function/method name declared on this line, which
            # the regex call fallback must not read as a call.
            declared_at = None

            parent_id = self.context_at(lineno)[0]

            def claim(node_id):
                """Take ``node_id`` for a function or variable; an interface
                declared under it moves to `<scope>::interface_<name>`."""
                value_ids.add(node_id)
                if node_id in interface_ids:
                    interface_ids.discard(node_id)
                    scope, _, own = node_id.rpartition("::")
                    self.rename_node(node_id, f"{scope}::interface_{own}")

            def read_arrow(idx):
                """Read `const name = (params) => ...` from line ``idx`` to
                the line that closes its parameter list. Returns that line's
                index and what ``_arrow_function`` makes of the statement."""
                text = ""
                for j in range(idx, min(idx + 40, len(lines))):
                    text = f"{text} {lines[j].strip()}".lstrip()
                    arrow = _arrow_function(text)
                    code = _NON_CODE_RE.sub("", text).rstrip()
                    # The value may start on the line after the `=`.
                    open_ended = code.endswith("=") or code.count("(") > code.count(")")
                    if arrow or not open_ended:
                        return j, arrow
                return idx, None

            def consume_sig(idx):
                sig_lines = [lines[idx]]
                j = idx
                while (
                    "{" not in lines[j] and ";" not in lines[j] and j < len(lines) - 1
                ):
                    j += 1
                    sig_lines.append(lines[j])
                return j, " ".join([ln.strip() for ln in sig_lines])

            if match := re_const.search(line):
                name, val = match.groups()
                doc = self.get_leading_docstring(lines, i)
                cid = f"{parent_id}::{name}"
                claim(cid)
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
                class_ids.add(cid)
                if "{" in line:
                    enter(cid, lineno, self._find_block_end(lines, i))
            elif match := re_class.search(line):
                name = match.group(1)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig = consume_sig(i)
                cid = f"{parent_id}::{name}"
                # Find end of class block
                end_lineno = (
                    self._find_block_end(lines, j) if "{" in lines[j] else lineno
                )
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    cid,
                    "Class",
                    name=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")
                class_ids.add(cid)
                # Generic arguments go first: `class A<T extends B> extends C<T>`
                # inherits from C.
                heritage = _strip_generics(full_sig)
                if "extends" in heritage:
                    try:
                        base = heritage.split("extends")[1].split()[0].split("{")[0]
                        self.builder.add_edge(cid, f"REF:{base}", "INHERITS")
                    except (IndexError, ValueError):
                        pass
                enter(cid, lineno, end_lineno)
                i = j
            elif match := re_interface.search(line):
                name = match.group(1)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig = consume_sig(i)
                cid = f"{parent_id}::{name}"
                if cid in value_ids:
                    # The function or variable of that name has this ID.
                    cid = f"{parent_id}::interface_{name}"
                else:
                    interface_ids.add(cid)
                # Find end of interface block
                end_lineno = (
                    self._find_block_end(lines, j) if "{" in lines[j] else lineno
                )
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    cid,
                    "Class",
                    name=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")
                class_ids.add(cid)
                heritage = _strip_generics(full_sig)
                if "extends" in heritage:
                    try:
                        bases = heritage.split("extends")[1].split("{")[0].split(",")
                        for b in bases:
                            self.builder.add_edge(cid, f"REF:{b.strip()}", "INHERITS")
                    except (IndexError, ValueError):
                        pass
                enter(cid, lineno, end_lineno)
                i = j
            elif match := re_func.search(line):
                name = match.group(1)
                declared_at = match.start(1)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig = consume_sig(i)
                fid = f"{parent_id}::{name}"
                return_type, param_types = self._extract_ts_types(full_sig)
                end_lineno = (
                    self._find_block_end(lines, j) if "{" in lines[j] else lineno
                )
                signature = self._build_ts_signature(name, param_types, return_type)
                token_count = (end_lineno - lineno + 1) * 10
                claim(fid)
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
                enter(fid, lineno, end_lineno, full_sig)
                i = j
            elif _TS_VARIABLE_RE.match(line) and (
                arrow := (statement := read_arrow(i))[1]
            ):
                name, parameters = arrow
                j = statement[0]
                docstring = self.get_leading_docstring(lines, i)
                fid = f"{parent_id}::{name}"
                # A block is left open where the statement stops; an
                # expression body ends there.
                last = _NON_CODE_RE.sub("", lines[j])
                end_lineno = (
                    self._find_block_end(lines, j)
                    if last.count("{") > last.count("}")
                    else j + 1
                )
                # Arrow functions don't have easily extractable signatures
                signature = f"const {name} = (...) => ..."
                token_count = (end_lineno - lineno + 1) * 10
                claim(fid)
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
                enter(fid, lineno, end_lineno, parameters)
                i = j
            elif (
                "::" in parent_id
                and parent_id != file_id
                and (match := re_method.search(line))
            ):
                name = match.group(1)
                j, full_sig = consume_sig(i)
                # `name(` declares a method directly in a class body, or where
                # a block follows its parameter list; otherwise it is a call.
                if name not in ["if", "for", "while", "switch", "catch"] and (
                    parent_id in class_ids or _opens_body(full_sig)
                ):
                    declared_at = match.start(1)
                    docstring = self.get_leading_docstring(lines, i)
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
                    enter(mid, lineno, end_lineno, full_sig)
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
                declared_at = match.start(2)
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
                end_lineno = (
                    self._find_block_end(lines, i)
                    if "{" in line or "=>" in line
                    else lineno
                )
                # Look for next few lines if arrow function continues
                if end_lineno == lineno:
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if "{" in lines[j] or "=>" in lines[j]:
                            end_lineno = self._find_block_end(lines, j)
                            break
                token_count = max((end_lineno - lineno + 1) * 10, 50)
                docstring = self.get_leading_docstring(lines, i)
                self.add_config_node(
                    file_id,
                    config_id,
                    config_name,
                    signature=f"{client_name}.interceptors.{interceptor_type}.use(...)",
                    docstring=docstring
                    or f"Configures {interceptor_type} interceptor for {client_name}",
                    config_type="interceptor",
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    target=client_name,
                )
            elif match := re_middleware.search(line):
                middleware_name = match.group(2).strip()
                if middleware_name and not middleware_name.startswith("("):
                    config_name = f"middleware_{middleware_name.replace('/', '_').replace('.', '_')}"
                    config_id = f"{file_id}::{config_name}"
                    end_lineno = (
                        self._find_block_end(lines, i) if "{" in line else lineno
                    )
                    token_count = max((end_lineno - lineno + 1) * 10, 30)
                    self.add_config_node(
                        file_id,
                        config_id,
                        config_name,
                        signature=f"app.use({middleware_name})",
                        docstring=f"Registers middleware: {middleware_name}",
                        config_type="middleware",
                        lineno=lineno,
                        end_lineno=end_lineno,
                        token_count=token_count,
                    )
            elif match := re_event_handler.search(line):
                emitter_name = match.group(1)
                event_name = match.group(2)
                config_name = f"{emitter_name}_on_{event_name.replace(':', '_')}"
                config_id = f"{file_id}::{config_name}"
                end_lineno = (
                    self._find_block_end(lines, i)
                    if "{" in line or "=>" in line
                    else lineno
                )
                token_count = max((end_lineno - lineno + 1) * 10, 30)
                docstring = self.get_leading_docstring(lines, i)
                self.add_config_node(
                    file_id,
                    config_id,
                    config_name,
                    signature=f"{emitter_name}.on('{event_name}', ...)",
                    docstring=docstring
                    or f"Handles '{event_name}' event on {emitter_name}",
                    config_type="event_handler",
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    target=emitter_name,
                    event=event_name,
                )

            # Calls (regex fallback - only used if ast-grep unavailable)
            # Module-level calls are sourced at the file, as with ast-grep.
            if not self.builder.use_ast_grep:
                code = _blank_js_strings(line)
                # A module's top level takes in the bodies of callbacks, whose
                # variables are not the module's.
                scope = None if parent_id == file_id else parent_id
                self.begin_line(code, lineno, scope, line)
                for match in re_call_candidate.finditer(code):
                    if match.start(1) == declared_at:
                        continue
                    callee = match.group(1)
                    # `name(params) {` at the start of a line defines a
                    # method of an object literal.
                    if (
                        code[: match.start(1)].strip() in _TS_METHOD_MODIFIERS
                        and "." not in callee
                        and _opens_body(code[match.start(1) :])
                    ):
                        continue
                    # JS/TS control-flow keywords look like calls (`if(x)`,
                    # `while (y)`). Reject them by simple name — language
                    # general, not codebase specific.
                    simple_name = callee.split(".")[-1]
                    if simple_name in _JSTS_CONTROL_KEYWORDS:
                        continue
                    paren = match.end()
                    if code[paren] == "<":
                        # Generic arguments: `name<T, U>(`
                        depth = 0
                        for offset, char in enumerate(code[paren:]):
                            depth += (char == "<") - (char == ">")
                            if depth == 0:
                                rest = code[paren + offset + 1 :]
                                paren += offset + 1 + len(rest) - len(rest.lstrip())
                                break
                    if code[paren : paren + 1] == "(":
                        # A call of a value passed in as a parameter, unless
                        # the name is a member of an expression.
                        is_parameter = any(
                            callee in params for *_, params in self._contexts
                        ) and not code[: match.start(1)].rstrip().endswith(".")
                        self.add_call(
                            parent_id,
                            callee,
                            lineno,
                            self.context_at(lineno)[1],
                            match.start(1),
                            paren,
                            "?" if is_parameter else None,
                        )
            i += 1

        # Use ast-grep for more accurate CALLS detection
        if self.builder.use_ast_grep:
            self._extract_calls_with_ast_grep(str(file_path), file_id)

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

    def _build_ts_signature(
        self, name: str, param_types: list, return_type: str
    ) -> str:
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

    def _extract_calls_with_ast_grep(self, file_path: str, file_id: str):
        """Extract calls using ast-grep for higher accuracy.

        Uses the TypeScriptSymbolTable to filter type-only imports.
        SCIP handles actual symbol resolution for both Rust and TypeScript.
        """
        # Build a line-to-context map from our parsed structure
        # Uses span-size ordering to ensure innermost context wins for nested functions
        line_to_context = build_line_to_context_map(self.builder.nodes, file_id)

        for call in extract_calls_typescript(file_path):
            # ast-grep numbers the lines of the file on disk; keep only the
            # calls inside the parsed text, renumbered relative to it.
            lineno = call["lineno"] - self.line_offset
            if not 1 <= lineno <= self.line_count:
                continue
            callee = call["callee"]

            # Skip type-only imports (they don't generate runtime calls)
            if hasattr(self, "symbol_table") and self.symbol_table.is_type_import(
                callee
            ):
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


# Languages whose methods are reachable only through an explicit receiver:
# a bare `name(...)` there never calls a method of the enclosing class.
_EXPLICIT_RECEIVER_LANGS = frozenset({"python", "rust", "go", "typescript"})
# Languages whose parser reports every member call with its receiver, so a
# bare callee is a free call.
_FREE_BARE_CALL_LANGS = frozenset({"python", "go"})
# Languages where `receiver.name(...)` can reach a top-level definition
# through a namespace object.
_NAMESPACE_RECEIVER_LANGS = frozenset({"typescript"})
# Languages where a bare call reaches another file's definition only
# through an import of its name.
_IMPORT_SCOPED_LANGS = frozenset({"rust", "typescript"})
# Methods that hand on what their receiver wraps: `Result<T, E>::unwrap`.
_UNWRAPPING_METHODS = {"rust": ("unwrap", "expect")}
# Languages that spread one type's members over several declarations.
_SPLIT_TYPE_LANGS = frozenset({"rust", "go", "clang", "dotnet", "ruby", "typescript"})
_SELF_RECEIVERS = frozenset({"self", "this", "Self", "cls", "static", "parent"})
_EXTERNAL = "external"
_UNRESOLVED = "unresolved"
# What a call returns when nothing is known of it, not even that it comes
# from outside the project.
_OPAQUE = "opaque"


def _split_callee(callee: str) -> tuple[list[str], str]:
    """Split a callee into its qualifier segments and its name."""
    bare = _strip_generics(callee)
    if "(" in bare:
        # ast-grep hands over whole chains: `Type::new(&arg).unwrap` is read
        # as `Type::new`, `make(x).run` as `run`.
        head = re.match(r"(?:[A-Za-z_]\w*::)+[A-Za-z_]\w*", bare)
        bare = head.group(0) if head else bare[bare.rfind(")") + 1 :]
    parts = [part for part in re.split(r"::|->|\.|\\", bare) if part]
    return parts[:-1], parts[-1] if parts else ""


def _star_match(pattern: str, text: str) -> str | None:
    """What the `*` of ``pattern`` stands for where ``pattern`` matches
    ``text``: the empty string for a pattern without one, None for no match."""
    prefix, star, suffix = pattern.partition("*")
    if star and text.startswith(prefix) and text.endswith(suffix):
        return text[len(prefix) : len(text) - len(suffix)]
    return "" if pattern == text else None


class CallResolver:
    """Decides which project definition, if any, a call site reaches.

    ``resolve`` returns ``(node_id, source)`` for a call pinned to one
    definition, ``(None, "external")`` for a call that cannot reach a project
    definition (no project symbol carries its name, its receiver comes from
    outside the project, or it calls a value held in a variable), and
    ``(None, "unresolved")`` for a call that may reach one of several
    project definitions.
    """

    def __init__(self, builder):
        self.builder = builder
        self.lang = builder._get_language
        self.nodes = {node["id"]: node for node in builder.nodes}
        self.by_name = {}  # (language, name) -> node IDs
        self.members = {}  # parent ID -> {name: node ID}
        # ID of a value -> ID of the interface declared under its name
        self.companions = {}
        self.renamed_types = {}  # (language, declared name) -> those interfaces
        for node_id, node in self.nodes.items():
            parent, sep, name = node_id.rpartition("::")
            if sep:
                lang = self.lang(node_id)
                self.by_name.setdefault((lang, name), []).append(node_id)
                self.members.setdefault(parent, {})[name] = node_id
                declared = self._name(node_id)
                if declared != name and node["type"] == "Class":
                    # A type is looked up by the name it was declared under.
                    self.renamed_types.setdefault((lang, declared), []).append(node_id)
                    self.companions[f"{parent}::{declared}"] = node_id
        self.bases = {}  # class ID -> base type names
        for edge in builder.edges:
            if edge["relation"] == "INHERITS" and edge["target"].startswith("REF:"):
                base = _split_callee(edge["target"][4:])[1]
                self.bases.setdefault(edge["source"], []).append(base)
        self.containers = {}  # (language, module or package name) -> file IDs
        self.go_packages = {}  # directory -> Go file IDs
        self.go_modules = sorted(  # (go.mod directory, module path), longest first
            builder.go_modules.items(), key=lambda item: -len(item[1])
        )
        self.java_types = {}  # dotted class name -> class ID
        self.py_modules = {}  # dotted module name -> file ID
        self.py_module_of = {}  # file ID -> dotted module name
        files = {nid for nid, node in self.nodes.items() if node["type"] == "File"}
        py_files = []
        for file_id in sorted(files):
            parts = file_id[5:].rsplit(".", 1)[0].split("/")
            lang = self.lang(file_id)
            if lang == "python":
                top = len(parts) - 1
                while top and f"FILE:{'/'.join(parts[:top])}/__init__.py" in files:
                    top -= 1
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                if parts[top:]:
                    self.py_module_of[file_id] = ".".join(parts[top:])
                    py_files.append((top, parts, file_id))
            elif lang == "rust":
                if parts[-1] in ("lib", "main") and parts[-2:-1] == ["src"]:
                    parts = parts[:-2]
                elif parts[-1] == "mod":
                    parts = parts[:-1]
                if parts:
                    key = (lang, parts[-1].replace("-", "_"))
                    self.containers.setdefault(key, []).append(file_id)
            elif lang == "go":
                self.go_packages.setdefault("/".join(parts[:-1]), []).append(file_id)
                if len(parts) > 1:
                    self.containers.setdefault((lang, parts[-2]), []).append(file_id)
            elif lang == "java" and f"{file_id}::{parts[-1]}" in self.nodes:
                # A public class is importable by any dotted suffix of its path.
                for start in range(len(parts)):
                    self.java_types.setdefault(
                        ".".join(parts[start:]), f"{file_id}::{parts[-1]}"
                    )
        # A module is importable by its path below its topmost package and,
        # through namespace packages, by any longer path no other module owns.
        for top, parts, file_id in py_files:
            self.py_modules.setdefault(".".join(parts[top:]), file_id)
        for top, parts, file_id in py_files:
            for start in range(top):
                self.py_modules.setdefault(".".join(parts[start:]), file_id)
        self.py_roots = {module.split(".")[0] for module in self.py_modules}
        self.shared_declaration = {}  # candidate node IDs -> node ID or None
        self.returned = {}  # edge index of a call -> the value it returns
        self._result = None  # value of the call being resolved, where known

    def _name(self, node_id):
        """The name ``node_id`` was declared under, which its ID may prefix."""
        declared = self.nodes[node_id]["metadata"].get("name")
        return declared or node_id.rpartition("::")[2]

    def resolve(self, callee, source_id, origin=None):
        self._result = None
        verdict = self._verdict(callee, source_id, origin)
        if verdict[1] == _EXTERNAL and self._result is None:
            # What a call returns is from outside too when its receiver or
            # its path is.
            self._result = _EXTERNAL
        return verdict

    def _verdict(self, callee, source_id, origin):
        lang = self.lang(source_id)
        qualifier, name = _split_callee(callee)
        candidates = self.by_name.get((lang, name), [])
        if origin is not None:
            verdict = self._resolve_origin(origin, source_id)
            if verdict is None:
                verdict = self._resolve_member(candidates, lang, callee)
            if verdict[0] or candidates:
                return verdict
            # No project symbol carries the name, so the call is external;
            # what it returns is unknown unless its receiver is external too.
            if verdict[1] != _EXTERNAL:
                self._result = _OPAQUE
            return None, _EXTERNAL
        imported = self._imported(qualifier[0] if qualifier else name, source_id)
        if imported == _EXTERNAL:
            return None, _EXTERNAL
        if not candidates:
            # No project symbol has the name; that alone does not say where
            # the receiver, and so the result, comes from.
            self._result = _OPAQUE
            return None, _EXTERNAL
        if imported is not None and len(qualifier) <= 1:
            # An import names the definition itself or the scope it is in.
            hits = {i for i in imported if not qualifier and i.endswith(f"::{name}")}
            hits |= {self._find_member(scope, name)[0] for scope in imported} - {None}
            hit = self._nearest(sorted(hits), source_id) if hits else None
            return (hit, "regex") if hit else (None, _UNRESOLVED)
        if not qualifier:
            return self._resolve_bare(name, candidates, source_id, lang, callee)
        receiver = qualifier[-1]
        if receiver in _SELF_RECEIVERS:
            owners = [self._enclosing_class(source_id)] if len(qualifier) == 1 else []
            source = "self_type"
        else:
            owners = self._types(lang, receiver)
            if lang != "rust" or callee.endswith(f"::{name}"):
                # A Rust module is named in a path, never before a `.`.
                owners += self.containers.get((lang, receiver), [])
            source = "regex"
            root = qualifier[0]
            # A path of two or more segments, not a chain of fields, that
            # starts outside the project.
            if (
                lang == "rust"
                and len(qualifier) > 1
                and callee.startswith(f"{root}::")
                and root != "Self"
                and self._use_crate(root, source_id.split("::")[0]) == ()
            ):
                # `io::Error` is not a project type; the project can
                # still give it methods through a trait impl.
                hits = {
                    self.members[o][name]
                    for o in owners
                    if self.nodes[o]["metadata"].get("kind") == "impl"
                    and name in self.members.get(o, ())
                }
                hit = self._nearest(sorted(hits), source_id) if hits else None
                return (hit, source) if hit else (None, _EXTERNAL)
        if any(owners):
            hits, complete = set(), True
            for owner in owners:
                hit, owner_complete = self._find_member(owner, name)
                complete = complete and owner_complete
                if hit:
                    hits.add(hit)
            if hits:
                hit = self._nearest(sorted(hits), source_id)
                return (hit, source) if hit else (None, _UNRESOLVED)
            # A member a project type has from outside may return that type.
            self._result = _OPAQUE
            return None, _UNRESOLVED if complete else _EXTERNAL
        if lang == "rust" and "::" in callee:
            return None, _EXTERNAL
        return self._resolve_member(candidates, lang, callee)

    def _imported(self, name, scope_id):
        """What an import in the file of ``scope_id`` binds ``name`` to: the
        IDs of the project definitions or files it names (none when they
        cannot be located), ``"external"`` for a target outside the project,
        None when no import binds it."""
        file_id = scope_id.split("::")[0]
        target = self.builder.imports.get(file_id, {}).get(name)
        lang = self.lang(file_id)
        if target is None or lang not in ("go", "java", "typescript", "rust"):
            return None
        if lang == "typescript":
            return self._js_module(target, file_id)
        if lang == "rust":
            *path, leaf = target.split("::")
            modules = path[1:]
            crate = self._use_crate(target, file_id)
            if not crate:
                return None if crate is None else _EXTERNAL
            found = [
                c
                for c in self.by_name.get((lang, leaf), ())
                if (self._unit(c, lang),) == crate
            ]
            # Prefer the definition under the module the path names last.
            inside = [
                c
                for c in found
                if modules and re.search(rf"[/:]{modules[-1]}(\.rs|/mod\.rs|::)", c)
            ]
            return inside or found or None
        if lang == "go":
            for folder, module in self.go_modules:
                if target == module or target.startswith(f"{module}/"):
                    parts = (folder, target[len(module) + 1 :])
                    return self.go_packages.get("/".join(p for p in parts if p), [])
            return _EXTERNAL if self.go_modules else None
        if target in self.java_types:
            return [self.java_types[target]]
        owner, _, member = target.rpartition(".")
        if owner in self.java_types:
            hit = self._find_member(self.java_types[owner], member)[0]
            return [hit] if hit else []
        return _EXTERNAL

    def result(self, target):
        """Value returned by the call just resolved, to ``target`` if it was."""
        if target:
            return self._called(("def", target))
        return None if self._result == _OPAQUE else self._result

    def _use_crate(self, path, file_id):
        """Where a Rust path written in ``file_id`` leads: ``(crate
        directory,)``, ``()`` for a crate outside the project, None when
        the path starts at a module or type in scope."""
        root = path.split("::")[0]
        if root in ("crate", "self", "super"):
            return (self._unit(file_id, "rust"),)
        if root in self.builder.rust_crates:
            return (self.builder.rust_crates[root],)
        if ("rust", root) in self.containers or self._types("rust", root):
            return None
        return ()

    def _js_module(self, specifier, file_id):
        """Resolve a module specifier written in ``file_id``: ``[file ID]``
        for a project module, ``[]`` for one that cannot be located,
        ``"external"`` for a declared dependency, None for anything else."""
        folder = posixpath.dirname(file_id[5:])
        if specifier.startswith("."):
            found = self._js_file(posixpath.join(folder, specifier))
            return [found] if found else []
        aliased = self._js_alias(specifier, folder)
        if aliased is not None:
            return aliased
        for name, (folder, exports) in self.builder.js_packages.items():
            if specifier != name and not specifier.startswith(f"{name}/"):
                continue
            subpath = f".{specifier[len(name) :]}"
            if exports is None:
                # Without `exports` a subpath is a path inside the package.
                found = self._js_file(posixpath.join(folder, subpath))
                return [found] if found else []
            if not isinstance(exports, dict) or not all(k[:1] == "." for k in exports):
                exports = {".": exports}
            for key, target in exports.items():
                matched = _star_match(key, subpath)
                if matched is None:
                    continue
                # A conditional target lists one path per condition.
                pending = [target]
                while pending:
                    target = pending.pop(0)
                    if isinstance(target, dict):
                        pending.extend(target.values())
                    elif isinstance(target, str):
                        path = target.replace("*", matched)
                        if found := self._js_file(posixpath.join(folder, path)):
                            return [found]
            return []
        package = "/".join(specifier.split("/")[: 2 if specifier[0] == "@" else 1])
        outside = package in self.builder.js_dependencies
        return _EXTERNAL if outside or specifier.startswith("node:") else None

    def _js_alias(self, specifier, folder):
        """Resolve ``specifier`` through the `paths` and `baseUrl` of the
        tsconfig nearest above ``folder``: ``[file ID]``, ``[]`` when a
        pattern matches but no target is a project module, None when the
        configuration says nothing about it."""
        configs = self.builder.ts_configs
        while True:
            names = (
                posixpath.join(folder, n) for n in ("tsconfig.json", "jsconfig.json")
            )
            config = next((name for name in names if name in configs), None)
            if config or not folder:
                break
            folder = posixpath.dirname(folder)
        if config is None:
            return None
        base, paths = self._ts_options(config, set())
        # The pattern with the longest prefix wins.
        for pattern in sorted(paths or (), key=lambda p: -len(p.partition("*")[0])):
            matched = _star_match(pattern, specifier)
            if matched is None:
                continue
            targets = (t.replace("*", matched) for t in paths[pattern])
            found = (self._js_file(target) for target in targets)
            return [target for target in found if target][:1]
        found = base is not None and self._js_file(posixpath.join(base, specifier))
        return [found] if found else None

    def _ts_options(self, config, seen):
        """The effective ``(baseUrl directory, {pattern: [target paths]})``
        of the tsconfig at ``config``, either of them None when neither it
        nor a config it extends sets one."""
        extends, base, paths = self.builder.ts_configs[config]
        seen.add(config)
        folder = posixpath.dirname(config)
        if base is not None:
            base = posixpath.normpath(posixpath.join(folder, base))
        if paths is not None:
            # Targets are relative to `baseUrl`, else to the config's folder.
            paths = {
                pattern: [posixpath.join(base or folder, t) for t in targets]
                for pattern, targets in paths.items()
            }
        for parent in extends:
            parent = posixpath.normpath(posixpath.join(folder, parent))
            if not parent.endswith(".json"):
                parent += ".json"
            if parent in self.builder.ts_configs and parent not in seen:
                inherited = self._ts_options(parent, seen)
                base = inherited[0] if base is None else base
                paths = inherited[1] if paths is None else paths
        return base, paths

    def _js_file(self, path):
        """File ID of the JS/TS module at ``path``, written with any or no
        extension, or naming a directory with an index module."""
        stem = re.sub(r"\.[cm]?[jt]sx?$", "", posixpath.normpath(path))
        for tail in ("", "/index"):
            for extension in _JSTS_EXTENSIONS:
                if f"FILE:{stem}{tail}{extension}" in self.nodes:
                    return f"FILE:{stem}{tail}{extension}"
        return None

    def _exported(self, file_id, name, seen):
        """Definition a file makes available as ``name``: its own, or one
        it re-exports from another module."""
        hit = self.members.get(file_id, {}).get(name)
        if hit or file_id in seen:
            return hit
        seen.add(file_id)
        for exported, specifier, original in self.builder.reexports.get(file_id, ()):
            if exported in ("*", name):
                # Without a specifier the name is exported from the file itself.
                modules = (
                    self._js_module(specifier, file_id) if specifier else [file_id]
                )
                for module in modules or ():
                    hit = self._exported(module, original or name, seen)
                    if hit:
                        return hit
        return None

    def _resolve_bare(self, name, candidates, source_id, lang, callee):
        explicit = lang in _EXPLICIT_RECEIVER_LANGS
        hit = self._lexical(name, source_id, skip_classes=explicit)
        if hit:
            return hit, "regex"
        owner = None if explicit else self._enclosing_class(source_id)
        if owner:
            hit, _ = self._find_member(owner, name)
            if hit:
                return hit, "regex"
        free = [c for c in candidates if self._is_top_level(c)]
        file_id = source_id.split("::")[0]
        imports = self.builder.imports.get(file_id, {})
        if lang in _IMPORT_SCOPED_LANGS and name not in imports:
            # Only a glob import can bring the name in, from the crate it
            # names; one that starts at something in scope stays in this one.
            own = (self._unit(file_id, lang),)
            crates = set()
            for path in imports:
                if path.endswith("::*"):
                    found = self._use_crate(path, file_id)
                    crates.update(own if found is None else found)
            free = [c for c in free if self._unit(c, lang) in crates]
        if lang == "go":
            # A bare Go call stays inside its package, which is its directory.
            package = self._unit(file_id, lang)
            free = [c for c in free if self._unit(c, lang) == package]
        if free:
            hit = self._nearest(free, source_id)
            return (hit, "regex") if hit else (None, _UNRESOLVED)
        if lang in _FREE_BARE_CALL_LANGS:
            self._result = _OPAQUE
            return None, _EXTERNAL
        if explicit:
            # Without a receiver a method is out of reach.
            candidates = [c for c in candidates if self.nodes[c]["type"] != "Method"]
        return self._resolve_member(candidates, lang, callee)

    def _resolve_member(self, candidates, lang, callee):
        """Resolve a call through a receiver whose type is not known."""
        namespace = lang in _NAMESPACE_RECEIVER_LANGS
        pool = [
            c
            for c in candidates
            if self._is_class(c.rpartition("::")[0])
            or (namespace and self._is_top_level(c))
        ]
        if not pool or is_non_project_call(callee):
            # The receiver is not known, so neither is what is returned.
            self._result = _OPAQUE
            return None, _EXTERNAL
        pool = tuple(pool)
        if pool not in self.shared_declaration:
            self.shared_declaration[pool] = self._shared_declaration(pool)
        hit = self.shared_declaration[pool]
        return (hit, "regex") if hit else (None, _UNRESOLVED)

    def _shared_declaration(self, pool):
        """The one definition every other one in ``pool`` overrides, if any."""
        if len(pool) == 1:
            return pool[0]
        owners = {c: c.rpartition("::")[0] for c in pool}
        ancestors = {owner: self._ancestors(owner) for owner in owners.values()}
        for candidate, owner in owners.items():
            base = self._name(owner)
            if all(c == candidate or base in ancestors[o] for c, o in owners.items()):
                return candidate
        return None

    def _resolve_origin(self, origin, source_id):
        """Follow an origin path (see ``PythonParser._origin``) to a verdict.
        Returns None when the receiver of the last attribute is of unknown
        type."""
        value, steps, source = self._root_value(origin, source_id, set())
        if value is None and origin.startswith("@"):
            return None, _UNRESOLVED
        if value is None and origin.startswith("="):
            # A value held in a variable is no definition, and what calling
            # it returns is not known.
            self._result = _OPAQUE
            return None, _EXTERNAL
        unwraps = _UNWRAPPING_METHODS.get(self.lang(source_id), ())
        for index, step in enumerate(steps):
            if value is None or value == _EXTERNAL:
                break
            last = index == len(steps) - 1
            if step == "()":
                value = self._called(value)
            elif value[0] == "wrapped":
                unwrapped = ("self", value[1])
                if last and step in unwraps:
                    self._result = unwrapped
                    return None, _EXTERNAL
                value = unwrapped if step == "?" else None
            elif step != "?":
                value = self._attr(value, step, called=last)
                if value is None and last:
                    return None, _UNRESOLVED
        if isinstance(value, tuple) and value[0] in ("def", "cls"):
            return value[1], source
        if value == _EXTERNAL:
            return None, _EXTERNAL
        # A call of a value held in a variable, or returned by another call.
        if origin.endswith(")") or "." not in origin:
            self._result = _OPAQUE
            return None, _UNRESOLVED
        return None

    def _root_value(self, origin, source_id, seen):
        """Evaluate the first step of an origin path written in
        ``source_id``: an imported path behind `@`, `self`, `cls` or
        `super`, `#N` for what the call at edge index N returned, `~name`
        for what pytest passes as the parameter `name`, `=name` for a
        parameter or variable of ``source_id`` (a definition only where it
        is one of ``source_id``'s own), `?` for a value of no known origin,
        or a name read in scope. Returns its value, the steps after it, and
        the resolution source a call found through it has. ``seen`` holds
        the fixtures being evaluated."""
        file_id = source_id.split("::")[0]
        path = origin.lstrip("@")
        steps = re.findall(r"\(\)|[^.()]+", path)
        if origin.startswith("@"):
            dots = len(path) - len(path.lstrip("."))
            return (*self._import_value(steps, dots, file_id), "regex")
        source = "regex"
        root, steps = steps[0], steps[1:]
        owner = self._enclosing_class(source_id)
        if root in ("self", "cls", "super") and owner:
            source = "self_type"
            value = (root, owner)
            if root == "super":
                steps = steps[1:]
        elif root in ("self", "cls", "super", "?"):
            value = None
        elif root.startswith("#"):
            value = self.returned.get(int(root[1:]))
        elif root.startswith("~"):
            value = self._fixture_value(root[1:], source_id, seen)
        elif root.startswith("="):
            hit = self.members.get(source_id, {}).get(root[1:])
            value = ("def", hit) if hit else None
        else:
            value = self._name_value(root, source_id, set())
        return value, steps, source

    def _fixture_value(self, name, scope_id, seen):
        """Value pytest passes for the parameter ``name`` of the test or
        fixture ``scope_id``: what the nearest fixture of that name gives,
        looking in the enclosing classes, the module, and then each
        `conftest.py` from the module's directory upwards."""
        file_id = scope_id.split("::")[0]
        scopes = [s for s in self._scopes(scope_id) if self._is_class(s)]
        scopes.append(file_id)
        folder = posixpath.dirname(file_id[len("FILE:") :])
        while True:
            scopes.append(f"FILE:{posixpath.join(folder, 'conftest.py')}")
            if not folder:
                break
            folder = posixpath.dirname(folder)
        for scope in scopes:
            function_id, origin = self.builder.fixtures.get((scope, name), (None, None))
            # A fixture that takes a parameter of its own name gets the one
            # it overrides.
            if function_id is None or function_id == scope_id or function_id in seen:
                continue
            seen.add(function_id)
            value = self._called(("def", function_id))
            if value is None and origin:
                value, steps, _ = self._root_value(origin, function_id, seen)
                for step in steps:
                    if not isinstance(value, tuple):
                        break
                    value = (
                        self._called(value) if step == "()" else self._attr(value, step)
                    )
            return value
        return None

    def _name_value(self, name, scope_id, seen):
        """Value of a name read inside ``scope_id``."""
        hit = self._lexical(name, scope_id, skip_classes=True)
        if hit:
            return ("def", hit)
        lang = self.lang(scope_id)
        if lang != "python":
            # `Type<>` is a type written with type arguments.
            name, generic, _ = name.partition("<>")
            found = self._type_named(name, scope_id)
            name = name.rpartition("::")[2]
            if generic and found == _EXTERNAL:
                # A wrapper from outside around what may be a project type.
                return None
            if found is None and not self._types(lang, name):
                # An alias may stand for a project type, as may what a
                # generic type wraps.
                aliased = generic or name in self.builder.type_aliases
                return None if aliased else _EXTERNAL
            return ("def", found) if found not in (None, _EXTERNAL) else found
        file_id = scope_id.split("::")[0]
        imports = self.builder.imports.get(file_id, {})
        if name in imports and (file_id, name) not in seen:
            seen.add((file_id, name))
            target = imports[name]
            dots = len(target) - len(target.lstrip("."))
            value, steps = self._import_value(
                target.strip(".").split("."), dots, file_id
            )
            for step in steps:
                if value is None or value == _EXTERNAL:
                    break
                value = self._attr(value, step, seen)
            return value
        # Neither defined nor imported: a builtin, unless a star import hides it.
        return None if "*" in imports else _EXTERNAL

    def _import_value(self, steps, dots, file_id):
        """Split an imported dotted path into the project module it starts
        with and the steps left to take inside it."""
        names = steps[: steps.index("()")] if "()" in steps else steps
        package = []
        if dots:
            package = self.py_module_of.get(file_id, "").split(".")
            if not file_id.endswith("/__init__.py"):
                package = package[:-1]
            package = package[: len(package) - (dots - 1)]
        for size in range(len(names), -1, -1):
            module = ".".join(package + names[:size])
            if module in self.py_modules:
                return ("module", module), steps[size:]
        if dots or names[0] in self.py_roots:
            return None, []
        return _EXTERNAL, []

    def _attr(self, value, name, seen=None, called=False):
        """Value of attribute ``name`` of ``value``; ``called`` says the
        attribute is the callee itself rather than part of its receiver."""
        kind, ref = value
        if kind == "module":
            if f"{ref}.{name}" in self.py_modules:
                return ("module", f"{ref}.{name}")
            file_id = self.py_modules[ref]
            hit = self.members.get(file_id, {}).get(name)
            if hit:
                return ("def", hit)
            # A name the module itself imports is re-exported by it.
            if name in self.builder.imports.get(file_id, {}):
                return self._name_value(name, file_id, set() if seen is None else seen)
            return None
        if kind == "def" and self.nodes[ref]["type"] != "Class":
            return None
        hit, complete = self._find_member(ref, name, inherited_only=kind == "super")
        if hit:
            return ("def", hit)
        # A method missing from a type with a base outside the project is
        # inherited from that base, and may return the type; a missing data
        # attribute need not be inherited.
        if called and not complete:
            self._result = _OPAQUE
            return _EXTERNAL
        return None

    def _type_named(self, name, scope_id):
        """The project type a possibly qualified type name written in
        ``scope_id`` refers to, ``"external"`` for one an import places
        outside the project, None when it cannot be told."""
        lang = self.lang(scope_id)
        *qualifier, name = [part for part in re.split(r"::|\.", name) if part]
        imported = self._imported(qualifier[0] if qualifier else name, scope_id)
        if imported == _EXTERNAL:
            return _EXTERNAL
        if qualifier and imported:
            found = (self._find_member(scope, name)[0] for scope in imported)
            found = (self.companions.get(hit, hit) for hit in found)
            types = [hit for hit in found if hit and self._is_class(hit)]
        else:
            types = [i for i in imported or () if self._is_class(i)]
        types = types or self._types(lang, name)
        hit = self._nearest(types, scope_id) if types else None
        if hit is None and len({self._unit(t, lang) for t in types}) == 1:
            # Declarations of one split type, in one crate or package.
            hit = types[0]
        return hit

    def _called(self, value):
        """Value returned by calling ``value``."""
        kind, ref = value
        if kind == "cls" or (kind == "def" and self.nodes[ref]["type"] == "Class"):
            return ("self", ref)
        if kind != "def":
            return None
        returns = self.nodes[ref]["metadata"].get("return_type") or ""
        if self.lang(ref) != "python":
            # A value declared under the name of an interface constructs it,
            # unless its return type says otherwise.
            companion = self.companions.get(ref)
            found = self._returned_type(returns, ref)
            return found or (companion and ("self", companion))
        match = re.fullmatch(r"([A-Za-z_][\w.]*)(?: \| None)?", returns.strip("'\""))
        # "None" and "Any" are also what an unannotated function records.
        if not match or match[1] in ("None", "Any"):
            return None
        root, *attrs = match[1].split(".")
        # The annotation is read where the function is defined.
        result = self._name_value(root, ref.rpartition("::")[0], set())
        for attr in attrs:
            if result is None or result == _EXTERNAL:
                break
            result = self._attr(result, attr)
        if isinstance(result, tuple):
            return self._called(result) if self._is_class(result[1]) else None
        return result

    def _returned_type(self, returns, function_id):
        """Value of the declared return type ``returns`` of ``function_id``:
        an instance of a project type, a wrapper around one (`Result<T, E>`,
        `Promise<T>`), ``"external"``, or None."""
        match = re.match(
            r"[&*\s]*(?:'\w+\s+)?(?:mut\s+)?([\w$.:]+)\s*(?:<\s*([\w$.:]+))?",
            returns,
        )
        if not match:
            return None
        owner = self._enclosing_class(function_id)
        for kind, name in zip(("self", "wrapped"), match.groups()):
            if name in ("Self", "this"):
                found = owner
            else:
                found = name and self._type_named(name, function_id)
            if found == _EXTERNAL:
                return _EXTERNAL if kind == "self" else None
            if found:
                return (kind, found)
        return None

    def _scopes(self, node_id):
        """Yield ``node_id`` and each scope enclosing it, out to its file."""
        while node_id:
            yield node_id
            node_id = node_id.rpartition("::")[0]

    def _is_class(self, node_id):
        node = self.nodes.get(node_id)
        return bool(
            node
            and node["type"] == "Class"
            and node["metadata"].get("kind") != "module"
        )

    def _is_top_level(self, node_id):
        parent = self.nodes.get(node_id.rpartition("::")[0])
        return parent is None or parent["type"] == "File"

    def _enclosing_class(self, node_id):
        return next((s for s in self._scopes(node_id) if self._is_class(s)), None)

    def _lexical(self, name, scope_id, skip_classes):
        """Definition of ``name`` in the innermost scope enclosing ``scope_id``."""
        for scope in self._scopes(scope_id):
            if skip_classes and self._is_class(scope):
                continue
            hit = self.members.get(scope, {}).get(name)
            if hit:
                return hit
        return None

    def _types(self, lang, name):
        return [
            node_id
            for node_id in self.by_name.get((lang, name), ())
            if self.nodes[node_id]["type"] == "Class"
        ] + self.renamed_types.get((lang, name), [])

    def _find_member(self, owner_id, name, inherited_only=False, seen=None):
        """Look ``name`` up in a type or module and then in the type's bases.

        Returns ``(node_id, complete)``; ``complete`` is False when a base is
        not a project type, so a miss may be a member defined outside.
        """
        if not self._is_class(owner_id):
            return self._exported(owner_id, name, set()), True
        lang = self.lang(owner_id)
        type_name = self._name(owner_id)
        seen = set() if seen is None else seen
        if type_name in seen:
            return None, True
        seen.add(type_name)
        declarations = [owner_id]
        if lang in _SPLIT_TYPE_LANGS:
            unit = self._unit(owner_id, lang)
            declarations += [
                t
                for t in self._types(lang, type_name)
                if t != owner_id and self._unit(t, lang) == unit
            ]
        if not inherited_only:
            for declaration in declarations:
                hit = self.members.get(declaration, {}).get(name)
                if hit:
                    return hit, True
        # A type the project only writes impl blocks for is declared outside.
        kinds = {self.nodes[d]["metadata"].get("kind") for d in declarations}
        complete = kinds != {"impl"}
        for declaration in declarations:
            for base in self.bases.get(declaration, ()):
                # The base is the type its name stands for where the
                # subtype is declared; every type of that name when that
                # cannot be told.
                named = self._type_named(base, declaration)
                if named == _EXTERNAL:
                    base_types = []
                else:
                    base_types = [named] if named else self._types(lang, base)
                if not base_types:
                    complete = False
                for base_type in base_types:
                    hit, base_complete = self._find_member(base_type, name, seen=seen)
                    if hit:
                        return hit, True
                    complete = complete and base_complete
        return None, complete

    def _unit(self, node_id, lang):
        """What bounds one split type: the file in TypeScript, the package
        directory in Go, the crate in Rust."""
        path = node_id.split("::")[0][5:]
        if lang == "typescript":
            return path
        if lang == "go":
            return path.rpartition("/")[0]
        if lang == "rust":
            folders = [f for f in self.builder.rust_crates.values() if f]
            inside = [f for f in folders if path.startswith(f"{f}/")]
            return max(inside, key=len, default="")
        return ""

    def _ancestors(self, owner_id, seen=None):
        """Names of every type ``owner_id`` inherits from."""
        seen = set() if seen is None else seen
        lang = self.lang(owner_id)
        for base in self.bases.get(owner_id, ()):
            if base not in seen:
                seen.add(base)
                for base_type in self._types(lang, base):
                    self._ancestors(base_type, seen)
        return seen

    def _nearest(self, candidates, source_id):
        """The candidate whose file shares the longest path prefix with the
        caller's, or None when two are equally near."""
        if len(candidates) == 1:
            return candidates[0]
        source = source_id.split("::")[0].split("/")

        def shared(node_id):
            path = node_id.split("::")[0].split("/")
            return next(
                (i for i, (a, b) in enumerate(zip(source, path)) if a != b),
                min(len(source), len(path)),
            )

        ranked = sorted(candidates, key=shared, reverse=True)
        return ranked[0] if shared(ranked[0]) > shared(ranked[1]) else None


class CodeGraphBuilder:
    # ast-grep is invoked as a subprocess **per file** for TS/JS parsing
    # (imports + calls). On macOS each `fork+exec` costs ~10ms, so a
    # 20k-file monorepo (next.js, TypeScript compiler) spends 10+
    # minutes in fork overhead alone before any real work happens. On
    # corpora above this threshold we fall back to the regex parser
    # only — accuracy is slightly lower for TS imports but the index
    # actually completes. Overridable via ``DESCRY_AST_GREP_MAX_FILES``.
    _AST_GREP_MAX_FILES_DEFAULT = 5000

    def __init__(self, root_dir, excluded_dirs=None, rel_base=None):
        self.root_dir = Path(root_dir).resolve()
        # What node-ID paths are relative to: the project root, which is
        # above root_dir when only a subtree is indexed.
        self.rel_base = Path(rel_base).resolve() if rel_base else self.root_dir
        self.excluded_dirs = excluded_dirs or {
            "target",
            "node_modules",
            "dist",
            "build",
            "docs",
            "coverage",
            "demo-output",
            ".beads",
        }
        self.nodes = []
        self.edges = []
        self.node_registry = set()
        self.current_file_id = None
        self.current_scope = SymbolTable()
        # For CallResolver: file ID -> {local name: import target},
        # go.mod directory -> module path, Cargo package name -> directory,
        # and the npm packages the project defines and depends on.
        self.imports = {}
        self.go_modules = {}
        self.rust_crates = {}
        self.js_packages = {}  # name -> (directory, `exports` of package.json)
        self.js_dependencies = set()
        # tsconfig path -> (configs it extends, `baseUrl`, `paths`), as written
        self.ts_configs = {}
        # JS/TS file ID -> [(exported name or "*", module specifier, name there)]
        self.reexports = {}
        # (defining file or class ID, fixture name) -> (function ID, origin
        # path of the value the fixture returns or yields, if it has one)
        self.fixtures = {}
        # Names declared as type aliases, which are not graph nodes.
        self.type_aliases = set()
        # Per-run ast-grep enablement. Checked by parsers instead of
        # the module-level USE_AST_GREP so project size can gate it.
        self.use_ast_grep = USE_AST_GREP

    def _decide_ast_grep(self, ts_js_file_count: int) -> None:
        """Disable per-file ast-grep on very large corpora.

        Exposed as a method (not done in __init__) so callers can
        decide after they know the file count. Respects
        ``DESCRY_AST_GREP_MAX_FILES`` env override.
        """
        if not self.use_ast_grep:
            return
        try:
            threshold = int(
                os.environ.get(
                    "DESCRY_AST_GREP_MAX_FILES",
                    self._AST_GREP_MAX_FILES_DEFAULT,
                )
            )
        except ValueError:
            threshold = self._AST_GREP_MAX_FILES_DEFAULT
        if threshold > 0 and ts_js_file_count > threshold:
            logger.info(
                f"ast-grep disabled: {ts_js_file_count} TS/JS files exceeds "
                f"threshold of {threshold} (per-file subprocess overhead "
                f"would dominate). Override with DESCRY_AST_GREP_MAX_FILES."
            )
            self.use_ast_grep = False

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
        # Always use forward-slash separators in node IDs so the graph is
        # cross-platform (node IDs generated on Windows remain readable and
        # queryable on Linux/macOS, and vice versa).
        try:
            return path.relative_to(self.rel_base).as_posix()
        except ValueError:
            return Path(path).as_posix()

    def process_directory(self):
        # Resolve root once; reject entering any child directory whose resolved
        # path escapes the project root (defends against malicious symlinks).
        root_real = Path(self.root_dir).resolve()

        # First pass: count TS/JS files to decide ast-grep opt-out for
        # large corpora. Cheap walk — stats filenames only.
        if self.use_ast_grep:
            ts_js_count = 0
            for _r, _d, _f in os.walk(self.root_dir, followlinks=False):
                _d[:] = [
                    d
                    for d in _d
                    if not d.startswith(".") and d not in self.excluded_dirs
                ]
                for fn in _f:
                    if fn.endswith(_JSTS_EXTENSIONS):
                        ts_js_count += 1
            self._decide_ast_grep(ts_js_count)

        for root, dirs, files in os.walk(self.root_dir, followlinks=False):
            dirs.sort()
            files.sort()
            # Filter by name first (excluded dirs, dotfiles).
            dirs[:] = [
                d for d in dirs if not d.startswith(".") and d not in self.excluded_dirs
            ]
            # Then filter by realpath containment — drop directories that are
            # symlinks pointing outside the project root.
            safe_dirs = []
            for d in dirs:
                try:
                    child_real = (Path(root) / d).resolve(strict=False)
                    if child_real == root_real or root_real in child_real.parents:
                        safe_dirs.append(d)
                    else:
                        logger.warning(
                            "Skipping %s — escapes project root via symlink",
                            child_real,
                        )
                except OSError:
                    continue
            dirs[:] = safe_dirs
            for file in files:
                if not (
                    file.endswith(_INDEXED_EXTENSIONS)
                    or file in _MANIFESTS
                    or _TSCONFIG_RE.fullmatch(file)
                ):
                    continue
                file_path = Path(root) / file
                # Same containment check for files (catches symlink files).
                try:
                    file_real = file_path.resolve(strict=False)
                    if not (file_real == root_real or root_real in file_real.parents):
                        logger.warning(
                            "Skipping %s — escapes project root via symlink",
                            file_real,
                        )
                        continue
                except OSError:
                    continue
                rel_path = self.get_rel_path(file_path)
                # Skip pathologically large files before opening. A hostile
                # repo with a multi-GB `.py` would otherwise OOM the
                # generator by pulling the whole buffer into memory.
                try:
                    if file_path.stat().st_size > _MAX_SOURCE_FILE_BYTES:
                        logger.warning(
                            "Skipping %s — %.1f MiB exceeds source cap",
                            rel_path,
                            file_path.stat().st_size / (1024 * 1024),
                        )
                        continue
                except OSError:
                    continue
                # O_NOFOLLOW on the final open so a post-check symlink swap
                # can't redirect us outside the project root. Mirrors the
                # api_source hardening in web/server.py.
                try:
                    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                    fd = os.open(str(file_path), flags)
                    with os.fdopen(fd, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except OSError as e:
                    logger.warning(f"Cannot read {rel_path}: {e.strerror}")
                    continue
                except ValueError as e:
                    logger.warning(f"Error reading {rel_path}: {e}")
                    continue
                # A file one parser chokes on must not abort the whole index.
                try:
                    if file.endswith(".py"):
                        PythonParser(self).parse(file_path, rel_path, content)
                    elif file.endswith(".rs"):
                        RustParser(self).parse(file_path, rel_path, content)
                    elif file.endswith(".proto"):
                        ProtoParser(self).parse(file_path, rel_path, content)
                    elif file.endswith(".java"):
                        from descry.java_parser import JavaParser

                        JavaParser(self).parse(file_path, rel_path, content)
                    elif file.endswith(".go"):
                        from descry.go_parser import GoParser

                        GoParser(self).parse(file_path, rel_path, content)
                    elif file == "go.mod":
                        if match := _GO_MODULE_RE.search(content):
                            folder = rel_path.rpartition("/")[0]
                            self.go_modules[folder] = match.group(1)
                    elif file == "Cargo.toml":
                        if match := _CARGO_PACKAGE_RE.search(content):
                            crate = match.group(1).replace("-", "_")
                            self.rust_crates[crate] = rel_path.rpartition("/")[0]
                    elif file == "package.json":
                        manifest = json.loads(content)
                        folder = rel_path.rpartition("/")[0]
                        exports = manifest.get("exports")
                        self.js_packages[manifest.get("name")] = (folder, exports)
                        for key, value in manifest.items():
                            if key.endswith("ependencies") and isinstance(value, dict):
                                self.js_dependencies.update(value)
                    elif _TSCONFIG_RE.fullmatch(file):
                        config = _loads_jsonc(content)
                        options = config.get("compilerOptions") or {}
                        extends = config.get("extends")
                        if isinstance(extends, str):
                            extends = [extends]
                        base, paths = options.get("baseUrl"), options.get("paths")
                        self.ts_configs[rel_path] = (
                            [e for e in extends or () if isinstance(e, str)],
                            base if isinstance(base, str) else None,
                            {
                                str(pattern): [t for t in targets if isinstance(t, str)]
                                for pattern, targets in paths.items()
                                if isinstance(targets, list)
                            }
                            if isinstance(paths, dict)
                            else None,
                        )
                    elif file.endswith(_RUBY_EXTENSIONS):
                        from descry.ruby_parser import RubyParser

                        RubyParser(self).parse(file_path, rel_path, content)
                    elif file.endswith(".php"):
                        from descry.php_parser import PhpParser

                        PhpParser(self).parse(file_path, rel_path, content)
                    elif file.endswith(_DOTNET_EXTENSIONS):
                        from descry.dotnet_parser import DotnetParser

                        DotnetParser(self).parse(file_path, rel_path, content)
                    elif file.endswith(_CLANG_EXTENSIONS):
                        from descry.clang_parser import ClangParser

                        ClangParser(self).parse(file_path, rel_path, content)
                    elif file.endswith(".dart"):
                        from descry.dart_parser import DartParser

                        DartParser(self).parse(file_path, rel_path, content)
                    elif file.endswith(_JSTS_EXTENSIONS):
                        TSParser(self).parse(file_path, rel_path, content)
                    elif file.endswith(".svelte"):
                        # D.2: iterate all <script> blocks and shift each block's
                        # parsed linenos by its offset within the .svelte file.
                        pos = 0
                        while True:
                            start_tag = content.find("<script", pos)
                            if start_tag == -1:
                                break
                            gt = content.find(">", start_tag)
                            if gt == -1:
                                break
                            close = content.find("</script>", gt)
                            if close == -1:
                                break
                            body = content[gt + 1 : close]
                            # Lines before the script body (the character after '>').
                            line_offset = content.count("\n", 0, gt + 1)
                            before = len(self.nodes)
                            edges_before = len(self.edges)
                            TSParser(self).parse(
                                file_path, rel_path, body, line_offset=line_offset
                            )
                            # Shift newly-added node and edge linenos by the offset.
                            for node in self.nodes[before:]:
                                meta = node.get("metadata", {})
                                for k in ("lineno", "end_lineno"):
                                    v = meta.get(k)
                                    if isinstance(v, int):
                                        meta[k] = v + line_offset
                            for edge in self.edges[edges_before:]:
                                if "lineno" in edge.get("metadata", {}):
                                    edge["metadata"]["lineno"] += line_offset
                            pos = close + len("</script>")
                except Exception as e:  # noqa: BLE001 — boundary over every parser and the external ast-grep binary
                    logger.warning(f"Failed to parse {rel_path}: {e!r}")

    def _get_language(self, node_id: str) -> str:
        """Extract language from node ID based on file extension."""
        # Node ID format: FILE:path/to/file.ext::Symbol
        if not node_id.startswith("FILE:"):
            return "unknown"
        file_part = node_id.split("::")[0].replace("FILE:", "")
        if file_part.endswith(".rs"):
            return "rust"
        elif file_part.endswith((*_JSTS_EXTENSIONS, ".svelte")):
            return "typescript"
        elif file_part.endswith(".py"):
            return "python"
        elif file_part.endswith(".proto"):
            return "proto"
        elif file_part.endswith((".java", ".kt", ".scala")):
            return "java"
        elif file_part.endswith(".go"):
            return "go"
        elif file_part.endswith(_RUBY_EXTENSIONS):
            return "ruby"
        elif file_part.endswith(".php"):
            return "php"
        elif file_part.endswith(_DOTNET_EXTENSIONS):
            return "dotnet"
        elif file_part.endswith(_CLANG_EXTENSIONS):
            return "clang"
        elif file_part.endswith(".dart"):
            return "dart"
        return "unknown"

    def resolve_references(self, scip_index=None):
        """Resolve REF: targets to actual node IDs.

        A CALLS reference goes to SCIP first (type-aware) and then to
        CallResolver, which pins it to a definition, leaves it unresolved,
        or drops it as external. Other references are matched by name within
        the source's language: qualified name (Type::method), then same
        file, then any match.

        Args:
            scip_index: Optional ScipIndex for type-aware resolution
        """
        # Track resolution statistics
        stats = {"scip": 0, "regex": 0, "self_type": 0, "unresolved": 0, "external": 0}
        call_resolver = CallResolver(self)
        # (file, line, name) of each definition -> its node ID
        nodes_at = {
            (
                node["id"].split("::")[0][5:],
                node["metadata"].get("lineno"),
                node["id"].rpartition("::")[2],
            ): node["id"]
            for node in self.nodes
        }

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
                is_call = edge["relation"] == "CALLS"
                origin = edge.get("metadata", {}).pop("origin", None)
                column = edge.get("metadata", {}).pop("col", None)

                # A type reference to a stdlib/library name never resolves.
                if not is_call and is_non_project_call(ref_name):
                    edges_to_remove.add(idx)
                    continue

                # Strategy 0: SCIP lookup (highest accuracy). Gated on the
                # source language having a SCIP adapter registered — rust,
                # typescript, python, java (via scip-java), go (via scip-go).
                if scip_index and source_lang in (
                    "rust",
                    "typescript",
                    "python",
                    "java",
                    "go",
                    "ruby",
                    "php",
                    "dotnet",
                    "clang",
                    "dart",
                ):
                    # Get source file and line for SCIP lookup
                    source_file = source_id.split("::")[0].replace("FILE:", "")
                    lineno = edge.get("metadata", {}).get("lineno", 0)
                    if is_call:
                        # A call is taken from SCIP only where SCIP saw it; a
                        # name match is left to CallResolver.
                        found = scip_index.definition_at(
                            ref_name, source_file, lineno, column
                        )
                        scip_resolved = found and (
                            nodes_at.get((*found[1:], _split_callee(ref_name)[1]))
                            or found[0]
                        )
                    else:
                        scip_resolved = scip_index.resolve(
                            ref_name, source_file, lineno
                        )
                    # SCIP spells a symbol's path its own way; an ID that is
                    # no node of this graph would leave the edge dangling.
                    if scip_resolved in self.node_registry:
                        # Validate cross-crate resolutions to prevent false positives
                        # Extract crate names from paths (first directory component)
                        source_crate = (
                            source_file.split("/")[0] if "/" in source_file else ""
                        )
                        resolved_file = scip_resolved.split("::")[0].replace(
                            "FILE:", ""
                        )
                        target_crate = (
                            resolved_file.split("/")[0] if "/" in resolved_file else ""
                        )

                        # If cross-crate, verify ref_name looks related to the target
                        # This prevents spurious matches like EnvFilter -> AuthInterceptor
                        if (
                            source_crate
                            and target_crate
                            and source_crate != target_crate
                        ):
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

                if is_call:
                    if not resolved:
                        resolved, resolution_source = call_resolver.resolve(
                            ref_name, source_id, origin
                        )
                    call_resolver.returned[idx] = call_resolver.result(resolved)
                    if resolution_source == _EXTERNAL:
                        edges_to_remove.add(idx)
                        stats["external"] += 1
                        continue

                # Normalize crate:: prefix (Rust-specific)
                # crate::path::Symbol -> path::Symbol for matching
                ref_name = ref_name.removeprefix("crate::")

                # Strip trailing method chains for resolution
                # Handles: Type::new(&arg).unwrap -> Type::new
                #          chrono::Utc::now().naive_utc -> chrono::Utc::now
                #          TempDir::new().unwrap -> TempDir::new
                base_ref = ref_name
                if "::" in ref_name:
                    # Match multi-segment qualified names (A::B or A::B::C etc.)
                    # stopping before ( or . that follows the last segment
                    chain_match = re.match(
                        r"^((?:[A-Za-z_][A-Za-z0-9_]*::)+[A-Za-z_][A-Za-z0-9_]*)",
                        ref_name,
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
                if (
                    not resolved
                    and not is_call
                    and "::" in base_ref
                    and base_ref in qualified_lookup
                ):
                    matches = [
                        m for m in qualified_lookup[base_ref] if m[1] == source_lang
                    ]
                    if matches:
                        resolved = matches[0][0]
                        resolution_source = "regex"

                # Strategy 3: Extract base name and try standard resolution
                if not resolved and not is_call:
                    # Get base name (last part after . or ::)
                    base_name = ref_name.split(".")[-1].split("::")[-1]
                    if base_name in node_lookup:
                        matches = [
                            m for m in node_lookup[base_name] if m[1] == source_lang
                        ]
                        if matches:
                            # Prefer same-file matches
                            source_file = source_id.split("::")[0]
                            same_file = [
                                m for m in matches if m[0].startswith(source_file)
                            ]
                            if same_file:
                                resolved = same_file[0][0]
                            else:
                                resolved = matches[0][0]
                            resolution_source = "regex"

                if resolved:
                    edge["target"] = resolved
                    edge["metadata"] = edge.get("metadata", {})
                    edge["metadata"]["resolution_source"] = resolution_source
                    if is_call:
                        edge["relation"] = "CALLS_RESOLVED"
                        stats[resolution_source] += 1
                    elif edge["relation"] == "INSTANTIATES":
                        edge["relation"] = "INSTANTIATES_RESOLVED"
                elif is_call:
                    stats["unresolved"] += 1

        # Remove filtered non-project edges (stdlib + third-party library calls)
        if edges_to_remove:
            self.edges = [
                e for i, e in enumerate(self.edges) if i not in edges_to_remove
            ]
            logger.info(
                f"Filtered {len(edges_to_remove)} non-project edges during resolution"
            )

        # Log resolution statistics
        total = (
            stats["scip"] + stats["regex"] + stats["self_type"] + stats["unresolved"]
        )
        if total > 0:
            scip_pct = 100 * stats["scip"] / total if total else 0
            regex_pct = 100 * stats["regex"] / total if total else 0
            self_type_pct = 100 * stats["self_type"] / total if total else 0
            resolved_total = stats["scip"] + stats["regex"] + stats["self_type"]
            resolved_pct = 100 * resolved_total / total if total else 0
            logger.info(
                f"CALLS Resolution: {resolved_pct:.0f}% of {total} project calls "
                f"(SCIP: {stats['scip']}/{scip_pct:.0f}%, "
                f"Regex: {stats['regex']}/{regex_pct:.0f}%, "
                f"Self: {stats['self_type']}/{self_type_pct:.0f}%, "
                f"Unresolved: {stats['unresolved']}); "
                f"{stats['external']} external calls not counted"
            )

    def export(self, output_path, scip_index=None):
        """Export the graph to JSON.

        Args:
            output_path: Path to write the graph JSON
            scip_index: Optional ScipIndex for type-aware resolution
        """
        from descry._graph import CURRENT_SCHEMA

        self.resolve_references(scip_index=scip_index)
        in_degree = {}
        for edge in self.edges:
            t = edge["target"]
            if t:
                in_degree[t] = in_degree.get(t, 0) + 1
        for node in self.nodes:
            node["metadata"]["in_degree"] = in_degree.get(node["id"], 0)
        # Atomic write: stream to a sibling tmp file then os.replace. A
        # concurrent reader (MCP prewarm, web `/api/health`, another CLI
        # invocation) that stats the destination mid-write would otherwise
        # observe a partial JSON file and raise `json.JSONDecodeError`.
        # os.replace is atomic on POSIX and on Windows >=3.3.
        output_path = (
            Path(output_path) if not isinstance(output_path, Path) else output_path
        )
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schema_version": CURRENT_SCHEMA,
                        "nodes": self.nodes,
                        "edges": self.edges,
                    },
                    f,
                    indent=2,
                )
            os.replace(tmp_path, output_path)
        except Exception:
            # Leave the previous good graph in place if anything fails mid-write.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        logger.info(
            "Graph exported to %s. Stats: %d nodes, %d edges",
            output_path,
            len(self.nodes),
            len(self.edges),
        )


def main():
    """CLI entry point for descry-generate."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate codebase knowledge graph")
    parser.add_argument(
        "path", nargs="?", default=".", help="Root path to index (default: .)"
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root, when PATH is a directory inside it: config, cache "
        "and node-ID paths come from the root (default: PATH)",
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

    # The layers of DescryConfig.from_env(), with the root taken from the
    # arguments instead of auto-detected from the working directory.
    from descry.handlers import DescryConfig

    if args.no_scip:
        os.environ["DESCRY_NO_SCIP"] = "1"

    project_root = Path(args.project_root or target).resolve()
    config = DescryConfig(project_root=project_root)
    config._apply_toml(DescryConfig._load_toml(config.project_root))
    config._apply_env()

    # B.3: always pass config.excluded_dirs so non-TOML projects still get
    # the full default exclusion set.
    config_excluded_dirs = config.excluded_dirs

    # B.3: respect config.cache_dir instead of hardcoded ".descry_cache"
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build the graph
    builder = CodeGraphBuilder(
        target, excluded_dirs=config_excluded_dirs, rel_base=project_root
    )
    builder.process_directory()

    # Generate SCIP indices if available and enabled
    scip_index = None
    if config.enable_scip and SCIP_SUPPORT_LOADED and scip_available():
        try:
            scip_status = get_scip_status()
            indexers = scip_status.get("indexers", {})
            enabled_indexers = [
                name for name, info in indexers.items() if info.get("available")
            ]
            logger.info(f"SCIP: Enabled ({', '.join(enabled_indexers) or 'none'})")

            cache_manager = ScipCacheManager(
                project_root,
                scip_extra_args=config.scip_extra_args,
                scip_skip_crates=config.scip_skip_crates,
                scip_toolchain=config.scip_rust_toolchain,
                scip_timeout_minutes=config.scip_timeout_minutes,
                cache_dir=config.cache_dir,
            )
            # Generate SCIP for all supported languages (Rust and TypeScript)
            scip_paths = cache_manager.update_all()
            scip_files = list(scip_paths.values())

            if scip_files:
                scip_index = ScipIndex(
                    scip_files, cache_manager.project_dirs(scip_paths), project_root
                )
                stats = scip_index.get_stats()
                logger.info(
                    f"SCIP: Loaded {stats['definitions']} definitions, "
                    f"{stats['unique_names']} unique names from {len(scip_files)} files"
                )
        except Exception as e:  # noqa: BLE001 — boundary over external SCIP indexers; indexing continues regex-only
            logger.warning(f"SCIP: Failed to load ({e}), using regex only")
    elif SCIP_SUPPORT_LOADED:
        status = get_scip_status()
        if status.get("disabled_by_env"):
            logger.info("SCIP: Disabled (DESCRY_NO_SCIP=1)")
        else:
            logger.info(
                "SCIP: Unavailable (no indexers found — install any of: "
                "rust-analyzer, scip-typescript, scip-python, scip-java, "
                "scip-go, scip-ruby, scip-php, scip-dotnet, scip-clang, "
                "scip-dart)"
            )

    # Export with optional SCIP resolution
    graph_path = cache_dir / "codebase_graph.json"
    builder.export(str(graph_path), scip_index=scip_index)

    # Generate embeddings for semantic search (if dependencies available and enabled)
    if config.enable_embeddings:
        try:
            from descry.embeddings import SemanticSearcher, embeddings_available

            if embeddings_available():
                logger.info("Generating embeddings for semantic search...")
                # B.3: respect config.embedding_model
                searcher = SemanticSearcher(
                    str(graph_path),
                    cache_dir=str(config.cache_dir),
                    force_rebuild=True,
                    model_name=config.embedding_model,
                    source_root=str(config.project_root),
                )
                logger.info(
                    f"Embeddings generated: {len(searcher.nodes)} nodes indexed"
                )
            else:
                logger.debug(
                    "Embeddings: sentence-transformers not available, skipping"
                )
        except ImportError:
            logger.debug("Embeddings: module not available, skipping")
        except Exception as e:  # noqa: BLE001 — boundary over sentence-transformers/torch; indexing continues without embeddings
            logger.warning(f"Embeddings: Failed to generate ({e})")


if __name__ == "__main__":
    main()
