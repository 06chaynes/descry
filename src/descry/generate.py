"""Generate codebase knowledge graph with optional SCIP-based type-aware resolution."""

import ast
import contextlib
import json
import logging
import os
import posixpath
import re
import signal
import stat
import threading
from bisect import bisect_left, bisect_right
from collections import deque
from functools import cache
from operator import itemgetter
from pathlib import Path

from descry import _stdlib
from descry._comments import strip_block_comments
from descry._files import create_sibling

# Configure logging for consistent error reporting
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# Named rather than __name__: run with `-m` this module is `__main__`, which
# the level DESCRY_LOG_LEVEL sets on the "descry" logger would not reach.
logger = logging.getLogger("descry.generate")

_JSTS_EXTENSIONS = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
_RUBY_EXTENSIONS = (".rb", ".rake", ".gemspec")
_DOTNET_EXTENSIONS = (".cs", ".vb")
_CLANG_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx")
_C_HEADER_EXTENSIONS = (".h", ".hh", ".hpp", ".hxx")
# Manifests read for the package names call resolution needs. They do not
# become graph nodes.
_MANIFESTS = ("go.mod", "Cargo.toml", "package.json", "pubspec.yaml")
_GO_MODULE_RE = re.compile(r"^module\s+(\S+)", re.MULTILINE)
_PUBSPEC_NAME_RE = re.compile(r"^name:\s*['\"]?([\w$]+)", re.MULTILINE)
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


_GENERATED_MARKERS = (
    # Go convention (https://pkg.go.dev/cmd/go#hdr-Generate_Go_files_by_processing_source)
    "// Code generated ",
    "DO NOT EDIT",
    # C/C++ autogenerated
    "/* Automatically generated",
    "/* This file was generated",
    "/* A Bison parser, made by",
    "/* A lexical scanner generated by flex",
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


def source_lines(content):
    """The lines of ``content``, read as text, numbered as an editor and
    ``readlines`` number them: split at `\\n` alone, where
    ``str.splitlines`` also splits at `\\v`, `\\f` and other characters
    that end no line of source."""
    lines = content.split("\n")
    return lines[:-1] if lines[-1] == "" else lines


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
    head = "\n".join(source_lines(content)[:first_n_lines])
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


_LICENSE_HEADER_RE = re.compile(r"copyright|licen[cs]e|spdx-", re.IGNORECASE)
# The wording of a licence notice itself, which no description of a symbol
# shares, for a comment that does not open the file.
_LICENSE_NOTICE_RE = re.compile(
    r"copyright\s*(?:\(c\)|©|\d{4})|spdx-license-identifier|licensed (?:under|to) ",
    re.IGNORECASE,
)
# What a count of braces must skip: "..." and `...` strings, '{' / '}' char
# literals and // comments. A // right after `:`, a quote or `}` is left
# alone: that is a URL, a quoted "//" or template text, not a comment. A
# backslash-escaped character outside a string (a regex literal's `\}`, an
# escaped quote) is consumed so it neither counts nor opens a string.
_NON_CODE_RE = re.compile(
    r"""\\.|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`|'[{}]'|(?<![:'"`}])//.*"""
)


_LITERAL_RE = re.compile(
    r"""//|/\*|(?<![A-Za-z0-9_])[bc]?r(#*)"(?!"")|\"{3}|"|`|'(?:[^'\\]|\\.)'|'"""
)
_STRING_END_RE = {
    quote: re.compile(rf"(?:[^{quote}\\]|\\.)*{quote}") for quote in ('"', "'")
}
_WORD_BEFORE_RE = re.compile(r"\w")


def blank_strings(line, inside, spanning, holes=None):
    """Return ``line`` without its comments and with its string and
    character literals turned to spaces, so that text in one is not read as
    code, and what it ends inside of, if anything.

    ``inside`` is what the line starts inside of: `/*`, the opener of a
    literal named in ``spanning``, or None. ``spanning`` lists the literals
    that can continue on the next line: a backquote for a raw string, `"` for
    an ordinary one, `r` for `r#"..."#` and three quotes for a text block. A
    `'` in it also makes a single-quoted string of any length a literal;
    without one only a single character between quotes is. ``holes`` maps
    the opener of a literal whose text can hold code to the pattern that
    opens that code, Dart's `${`: the code is kept up to the `}` that
    closes it, which is blanked with the pattern, and a raw literal, one
    right after `r`, holds none. A line that ends in that code ends inside
    ``(opener, braces open in the code)``.
    """
    code, _, inside = _blank_from(line, 0, inside, spanning, holes or {})
    return code, inside


def _blank_from(line, position, inside, spanning, holes, depth=None):
    """``blank_strings`` of ``line`` from ``position``: ``(code, len(line),
    what the line ends inside of)``. With ``depth``, the braces open in a
    literal's code it starts in, it stops at the `}` that closes that code,
    ``(code, the position after it, None)``, or ends the line with
    ``(code, len(line), the braces open)``."""
    code = []
    if isinstance(inside, tuple):
        held, position, braces = _blank_from(
            line, position, None, spanning, holes, inside[1]
        )
        code.append(held.ljust(position))
        if braces is not None:
            return "".join(code), len(line), (inside[0], braces)
        inside = inside[0]
    spans = True
    raw = False
    while True:
        start = position
        if inside is None:
            match = _LITERAL_RE.search(line, position)
            stop = match.start() if match else len(line)
            for at in range(position, stop) if depth is not None else ():
                depth += (line[at] == "{") - (line[at] == "}")
                if depth < 0:
                    code.append(line[position:at])
                    return "".join(code), at + 1, None
            if not match or match.group() == "//":
                code.append(line[position:stop])
                return "".join(code), len(line), depth
            opener = match.group()
            start = match.start()
            code.append(line[position:start])
            position = match.end()
            if opener == "'" and "'" not in spanning:
                # A quote that opens no literal, as before a Rust lifetime.
                code.append(opener)
                continue
            if len(opener) > 1 and opener.startswith("'"):
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
            raw = line[start - 1 : start] == "r" and not _WORD_BEFORE_RE.match(
                line[max(start - 2, 0) : start - 1]
            )
        if inside in _STRING_END_RE:
            end = _STRING_END_RE[inside].match(line, position)
            end = end.end() if end else -1
        else:
            closer = {"/*": "*/"}.get(inside, inside.replace("r", '"', 1))
            end = line.find(closer, position)
            end = end + len(closer) if end >= 0 else -1
        opens = None if raw else holes.get(inside)
        limit = end if end >= 0 else len(line)
        found = next(
            (
                found
                for found in (opens.finditer(line, position, limit) if opens else ())
                if (found.start() - len(line[: found.start()].rstrip("\\"))) % 2 == 0
            ),
            None,
        )
        if found:
            code.append(" " * (found.end() - start))
            held, position, braces = _blank_from(
                line, found.end(), None, spanning, holes, 0
            )
            code.append(held.ljust(position - found.end()))
            if braces is not None:
                # The code goes on to the next line; in a literal that is in
                # code itself it is not followed there.
                left = depth if depth is not None else (inside, braces)
                return "".join(code), len(line), left
            continue
        if end < 0:
            code.append(" " * (len(line) - start))
            left = depth if depth is not None else inside if spans else None
            return "".join(code), len(line), left
        code.append(" " * (end - start))
        position, inside = end, None


# The line that opens the body of a type: `struct Name`, `class Name`,
# `interface Name`, Go's `type Name struct` or TypeScript's `type Name = {`
_TYPE_HEADER_RE = re.compile(
    r"\btype\s+(\w+)\s+struct\b|\b(?:struct|class|interface)\s+([A-Za-z_$][\w$]*)"
    r"|\btype\s+([A-Za-z_$][\w$]*)\s*(?:<[^=]*>)?\s*=\s*\{"
)
# What may stand before the name of a field where it is declared
_FIELD_MODIFIERS_RE = re.compile(
    r"\s*(?:(?:pub(?:\([^)]*\))?|public|private|protected|readonly|static|declare)\s+)*"
)
# The name or dotted path a line of code ends with.
_TRAILING_PATH_RE = re.compile(r"(?<![\w$.)\]])[A-Za-z_$][\w$]*(?:\.[\w$]+)*$")


def _strip_parens(text: str) -> str:
    """``text`` without its parentheses and what they hold."""
    previous = None
    while text != previous:
        previous, text = text, re.sub(r"\([^()]*\)", "", text)
    return text


def _casts(expression: str) -> bool:
    """Whether ``expression`` opens with a type in parentheses that what
    follows is cast to: `(T) value`, where `(value).member` reads a member."""
    grouped = expression.lstrip()
    if not grouped.startswith("("):
        return False
    depth = 0
    for at, char in enumerate(grouped):
        depth += (char == "(") - (char == ")")
        if not depth:
            break
    return grouped[at + 1 :].lstrip()[:1] not in ("", ".", "?", ";", ")")


def _is_call_chain(expression: str) -> bool:
    """Whether ``expression``, once the arguments of its calls are taken
    out, is one chain of names: `a.b(x).c()`, `new T(x)`, `await f()`, or
    the `).c()` such a chain goes on with on a later line. A cast is none:
    its value has the type it is cast to."""
    return not _casts(expression) and bool(
        re.fullmatch(
            r"[\s)]*(?:(?:await|new)\s+|&(?:mut\s+)?)*[\w$.:?\s]*;?\s*",
            _strip_parens(_without_turbofish(expression)),
        )
    )


def _without_turbofish(text: str) -> str:
    """``text`` without Rust's `::<...>` type arguments."""
    while (at := text.find("::<")) >= 0:
        depth, end = 0, at + 2
        for end in range(at + 2, len(text)):
            depth += (text[end] == "<") - (text[end] == ">")
            if not depth:
                break
        text = text[:at] + text[end + 1 :]
    return text


def _code_units(text: str, before=(0, 0, 0)):
    """The UTF-8, UTF-16 and UTF-32 code units of ``text``, added to
    ``before``."""
    if text.isascii():
        return tuple(count + len(text) for count in before)
    return (
        before[0] + len(text.encode()),
        before[1] + len(text.encode("utf-16-le")) // 2,
        before[2] + len(text),
    )


def _scip_columns(units):
    """Column a SCIP index gives the character after text of ``units``
    code units on its line: one number, or ``(UTF-8, UTF-16, UTF-32)``
    code-unit offsets where those differ, in the order of SCIP's position
    encodings."""
    return units[2] if units[0] == units[2] else units


class BaseParser:
    # Prefix of a doc-comment line, and the characters stripped from one.
    doc_line_prefix = "//"
    doc_strip_chars = "/!* "

    # Matches the `name =` of a statement whose value types ``name``, in a
    # ``name`` group; an ``unwrap`` group matches where the value is awaited.
    binding_re = None
    # Yields ``(name, steps)`` for each name a pattern of the language binds:
    # the steps of an origin path that lead from the value matched to it.
    bindings = None
    self_name = "self"  # what a method calls the instance it is called on
    # What a count of braces skips in the lines it is given
    _non_code = _NON_CODE_RE
    # Whether a value that goes on below its `name =`, on lines that start
    # with `.`, is followed through the calls those lines leave open.
    chained_values = False
    # The language the parser reads, where its generic declarations need
    # language-specific reading.
    language = ""

    def __init__(self, builder):
        self.builder = builder
        self.declared = {}
        self.fields = {}  # (declaring type, field name) -> its declared types
        self._line = 0
        self._code = ""
        self._pairs = {}  # column of a `(` on the line -> column of its `)`
        self._closes = {}  # column of a `)` -> edge index of the call it ends
        self._opens = {}  # column of a `(` left open on the line -> edge index
        # Where each `(` and `)` of the line starts to count, the column after
        # a `(` and that of a `)`, and the innermost `(` open from there; None
        # first, for the start of the line
        self._parens, self._innermost = [], [None]
        # The same for each bracket, and the innermost bracket open from
        # there: its column and character, the column None for one an
        # earlier line left open and the character "" for none
        self._brackets, self._inside = [], [(None, "")]
        self._carried = ""  # the brackets open across lines, outermost first
        # What each of them holds, by ``_held_label``, and what the innermost
        # of them that is open at each bracket holds
        self._carried_labels, self._inside_labels = [], [None]
        self._open_calls = []  # edge index of each call open across lines
        self._tail = None  # edge index of the call the previous line ends with
        self._tail_unwraps = 0  # `?` operators that follow that call
        self._tail_name = None  # name path the previous line ends with instead
        self._tail_dot = False  # whether the previous line ends with a member's `.`
        self._scope = None  # definition the line being read belongs to
        self._raw = ""  # the line being read, as written
        self._units = 0, (0, 0, 0)  # a column of it and the code units before
        self._locals = set()  # (definition ID, name of a parameter or variable)
        # The last line a declaration on a line is in scope on, where the
        # reader tells: ``(line, name) -> line``, and that of each binding a
        # pattern makes
        self._scope_end = None
        self._binding_ends = {}
        self._first_node = 0  # where the nodes and edges of the file being
        self._first_edge = 0  # parsed start in the builder's lists
        # (name, unwrap, depth) of a binding still being read. depth is how
        # many calls were open before the one its value starts with, while
        # that call is open; None otherwise.
        self._pending = None
        # the line declaring the type of the name that pending binds, if any
        self._pending_typed = None
        self._pending_closed = None  # column of the `)` that closes that call
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

    def enclosing_scope(self, node_id):
        """The innermost context around ``node_id``: where a class or an
        object it defines is evaluated."""
        for context in reversed(self._contexts):
            if context[0] != node_id:
                return context[0]
        return node_id

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
        self.fields = {}
        depth = 0  # parentheses open at the start of the line
        owner = None  # type whose body the line is in
        for lineno, line in enumerate(lines, 1):
            if line[:1].strip():
                depth = 0
                owner = None
            if header := _TYPE_HEADER_RE.search(line):
                owner = header[1] or header[2] or header[3]
            for pattern in patterns:
                for match in pattern.finditer(line):
                    groups = match.groupdict()
                    # A qualified type is one step of an origin path.
                    declared_type = re.sub(r"\s*::\s*|\.", "::", groups["type"])
                    # Unindented and not in a parameter list.
                    file_level = (
                        not line[:1].isspace() and "(" not in line[: match.start()]
                    )
                    if groups.get("generic"):
                        # C# keeps what the type's arguments are: an
                        # array's element carries its `IEnumerable<T>`.
                        declared_type += (
                            groups["generic"]
                            if self.language in ("dotnet", "dart")
                            else "<>"
                        )
                    if groups.get("name"):
                        held = _raw_pointer(declared_type, groups.get("pointer"))
                        key, value = groups["name"], f"{held}()"
                        if self.language == "clang" and line[match.end() :].startswith(
                            "["
                        ):
                            # A C array: `T()[]`, of elements of the type.
                            value += "[]"
                        if self.language == "clang" and declared_type == "auto":
                            # `for (auto& name : expr)` — `name` holds what
                            # iterating `expr` gives: what a call it is
                            # returns, or `*expr`, what the variable or the
                            # field `expr` holds where the call is read.
                            iterated = re.match(
                                r"\s*([\w.:>-]+(?:\(\))?)", line[match.end() :]
                            )
                            if iterated:
                                name = iterated[1]
                                value = (
                                    f"={name}.?" if name.endswith("()") else f"*{name}"
                                )
                    else:
                        key, value = f"<{groups['param']}>", declared_type
                    # What is declared `typeof x` holds `x` itself.
                    queried = _TS_TYPEOF_RE.fullmatch(groups["type"])
                    self.declared.setdefault(key, []).append(
                        (lineno, queried[1] if queried else value, file_level)
                    )
                    # A field is declared at the start of a line, behind
                    # modifiers at most, and not among parameters.
                    if (
                        groups.get("name")
                        and owner
                        and not depth
                        and _FIELD_MODIFIERS_RE.fullmatch(line[: match.start("name")])
                    ):
                        self.fields.setdefault((owner, key), set()).add(value)
            depth = max(0, depth + line.count("(") - line.count(")"))

    def _declared(self, name, lineno, since, anywhere=False):
        """What ``name`` is declared as at ``lineno``: by its last
        declaration from line ``since``, the start of the enclosing
        definition. Failing that what its file-level declarations agree on.
        ``anywhere`` looks at every declaration instead, for a field or a
        type parameter."""
        declarations = self.declared.get(name, ())
        if not anywhere:
            # A declaration counts in the block it is scoped to.
            ends = self._scope_end
            local = [
                d
                for d in declarations
                if since <= d[0] <= lineno and (not ends or lineno <= ends(d[0], name))
            ]
            if local:
                return max(local, key=lambda d: d[0])[1]
        # What a call returned is bound to a variable, never to a field. A
        # parameter or variable of the definition hides a file-level name.
        hidden = self._is_local(name)
        values = {
            v
            for _, v, top in declarations
            if (anywhere and v and not v.startswith("#"))
            or (top and not hidden)
            or not since
        }
        return values.pop() if len(values) == 1 else None

    def call_origin(self, callee, lineno, since, owner=None):
        """Origin path (see ``PythonParser._origin``) of a `receiver.method`
        callee whose receiver has a declared type or was bound to a call's
        result; None for any other. ``since`` is the first line of the
        enclosing definition, or 0 when the parser does not track one, and
        ``owner`` the name of the type `self` or `this` is an instance of."""
        receiver, _, method = callee.rpartition(".")
        field = receiver.removeprefix("this.").removeprefix("self.")
        member = field != receiver
        # A member read on the receiver, `b.held.meow`, follows it.
        field, _, path = field.partition(".")
        if member and (owner, field) in self.fields:
            types = self.fields[owner, field]
            value = next(iter(types)) if len(types) == 1 else None
        else:
            value = self._declared(field, lineno, since, member)
        if value is None:
            return None
        if not value.startswith("#"):
            # A type parameter stands for its bound.
            param = f"<{value[:-2]}>"
            bound = self._declared(param, lineno, since)
            bound = bound or self._declared(param, lineno, since, True)
            value = f"{bound}()" if bound else value
        return f"{value}.{path}.{method}" if path else f"{value}.{method}"

    def begin_line(self, code, lineno, scope=None, raw=None):
        """Start reading the calls of ``code``, source line ``lineno``. Call
        this for every line whose calls are recorded, in order. ``scope`` is
        the definition a variable the line binds is local to, and ``raw``
        the line as written where ``code`` has its literals blanked."""
        self._end_line()
        self._line, self._code = lineno, code
        self._scope, self._raw = scope, raw or code
        self._units = 0, (0, 0, 0)
        self._pairs, self._closes = {}, {}
        self._parens, self._innermost = [], [None]
        if code[:1].strip() and code[0] not in ")}]":
            # An unindented line starts a top-level item, unless it opens
            # with what closes the one before.
            self._open_calls.clear()
            self._carried, self._carried_labels = "", []
        self._brackets, self._inside = [], [(None, self._carried[-1:])]
        self._inside_labels = [
            self._carried_labels[-1] if self._carried_labels else None
        ]
        stack, brackets = [], []
        depth = self._pending[2] if self._pending else None
        for column, char in enumerate(code):
            if char == "(":
                stack.append(column)
            elif char == ")" and stack:
                self._pairs[stack.pop()] = column
            elif char == ")" and self._open_calls:
                self._closes[column] = self._open_calls.pop()
                if len(self._open_calls) == depth:
                    self._pending_closed = column
            if char in "()":
                self._parens.append(column + (char == "("))
                self._innermost.append(stack[-1] if stack else None)
            if char in "([{":
                brackets.append(column)
            elif char not in ")]}":
                continue
            elif brackets:
                brackets.pop()
            else:
                self._carried, self._carried_labels = (
                    self._carried[:-1],
                    self._carried_labels[:-1],
                )
            self._brackets.append(column + (char in "([{"))
            self._inside.append(
                (brackets[-1], code[brackets[-1]])
                if brackets
                else (None, self._carried[-1:])
            )
            held = not brackets and self._carried_labels
            self._inside_labels.append(held[-1] if held else None)
        self._carried += "".join(code[column] for column in brackets)
        self._left_open = brackets
        self._opens = dict.fromkeys(stack)

    def _end_line(self):
        """Carry what the finished line leaves open over to the next: its
        unclosed calls, the call it ends with, and what it binds. A line
        whose code is blank ends nothing — comments and preprocessor
        lines leave the open statement alone."""
        if not self._code.strip():
            return
        self._carried_labels += [
            self._held_label(self._code, c) for c in self._left_open
        ]
        open_before = len(self._open_calls)
        self._open_calls.extend(self._opens.values())
        statement = self._code.rstrip().rstrip(";").rstrip()
        value = statement.rstrip("?")
        # A `.` a line ends with is a member's, where it is no range's,
        # spread's or number's: the next line goes on with that member.
        dot = bool(re.search(r"(?<!\.)\.\Z", value)) and not _NUMBER_DOT_END_RE.search(
            value
        )
        chain = value[:-1].rstrip() if dot else value
        called = chain.rstrip("?")
        self._tail = self._closes.get(len(called) - 1) if called else None
        self._tail_unwraps = len(statement) - len(value) + len(chain) - len(called)
        ends_with = None if self._tail is not None else _TRAILING_PATH_RE.search(chain)
        continued, self._tail_dot = self._tail_dot, dot
        if ends_with and continued and not chain[: ends_with.start()].strip():
            # The path goes on from the one the line before ends with.
            self._tail_name = self._tail_name and f"{self._tail_name}.{ends_with[0]}"
        else:
            self._tail_name = ends_with and ends_with.group()
        binding = self.binding_re.search(self._code) if self.binding_re else None
        if binding and not self._binds_name(binding["name"]):
            # A name the file gets from elsewhere is no declaration of it.
            binding = None
        closed, self._pending_closed = self._pending_closed, None
        depth = None
        # The line that declares the name's type, where the line has one
        declares = None
        if binding:
            name, unwrap = binding["name"], bool(binding.groupdict().get("unwrap"))
            expression = self._code[binding.end() :]
            if self._scope:
                self.add_local(self._scope, name)
            declares = self._typed_line(name)
            # The value goes on over the next lines inside the call it
            # starts with.
            opened = [column for column in self._opens if column >= binding.end()]
            if opened and _is_call_chain(self._code[binding.end() : opened[0]]):
                depth = open_before + list(self._opens).index(opened[0])
        elif self._pending and closed is not None:
            name, unwrap, _ = self._pending
            declares = self._pending_typed
            expression = self._code[closed:]
        elif (
            self._pending
            and self._pending[2] is not None
            and len(self._open_calls) > self._pending[2]
        ):
            self._code = ""
            return
        elif self._pending and (continued or self._code.lstrip()[:1] in (".", ")")):
            name, unwrap, _ = self._pending
            declares = self._pending_typed
            expression = self._code
            opened = list(self._opens)
            if (
                self.chained_values
                and opened
                and _is_call_chain(expression[: opened[0]])
            ):
                depth = open_before
        else:
            name = None
        self._pending = (name, unwrap, depth) if name else None
        self._pending_typed = declares if name else None
        self._code = ""
        if name is None or self._tail is None or not value.endswith(")"):
            return
        # Bind only where the whole value is one call chain.
        if _is_call_chain(expression):
            unwraps = max(unwrap, len(statement) - len(value))
            origin = f"#{self._tail}" + ".?" * unwraps
            entries = self.declared.setdefault(name, [])
            # A type the file declares the name with is what it is read as.
            typed = any(value and top for _, value, top in entries)
            # Where the line that declares the name's type is the line that
            # binds it, the call's result is read through that type: one
            # declaration, holding both.
            declared = declares and next(
                (
                    entry
                    for entry in entries
                    if entry[0] == declares
                    and entry[1]
                    and not entry[1].startswith("#")
                ),
                None,
            )
            read_as = declared and self._typed_origin(origin, declared[1])
            if read_as:
                top = declared[2] or self._binds_file(name)
                entries[entries.index(declared)] = (declared[0], read_as, top)
            else:
                entries.append(
                    (self._line, origin, self._binds_file(name) and not typed)
                )

    def _typed_line(self, name):
        """The line being read, where it declares ``name`` with a type, else
        None."""
        for lineno, value, _ in self.declared.get(name, ()):
            if lineno == self._line and value and not value.startswith("#"):
                return lineno
        return None

    def paren_around(self, column):
        """The column of the innermost `(` of the line being read that is
        open at ``column``, else None."""
        return self._innermost[bisect_right(self._parens, column)]

    def bracket_around(self, column):
        """The innermost bracket open at ``column`` of the line being read:
        its column, None for one an earlier line left open, and its
        character, "" for none."""
        return self._inside[bisect_right(self._brackets, column)]

    def _element_origin(self, end, callee, lineno, since, source_id):
        """Origin path of a call of ``callee`` on what the code before
        column ``end`` of the line being read, an element access, gives;
        None for no reader of it."""

    def held_around(self, column):
        """What the bracket an earlier line left open that is innermost at
        ``column`` of the line being read holds, by ``_held_label``."""
        return self._inside_labels[bisect_right(self._brackets, column)]

    def _held_label(self, code, column):
        """What the bracket at ``column`` of ``code``, which the line leaves
        open, holds as a call's argument; None for no reader of it."""

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

    def _binds_file(self, name):
        """Whether what the line being read binds ``name`` to is in scope in
        every definition of the file."""
        return False

    def _nameless_owner(self, parent_id, name, lineno, last, kind="class"):
        """Make the definition ``name`` that source cannot write, held by
        ``parent_id`` and spanning lines ``lineno`` to ``last``: the members
        of a block for no type source can name are its own. Another of the
        name in ``parent_id``, one opened on the same line, is `name#2` and
        so on. Returns its ID."""
        owner_id, count = f"{parent_id}::{name}", 1
        while owner_id in self.builder.node_registry:
            count += 1
            owner_id = f"{parent_id}::{name}#{count}"
        self.builder.add_node(
            owner_id,
            "Class",
            name=name,
            kind=kind,
            lineno=lineno,
            end_lineno=last,
            token_count=(last - lineno + 1) * 10,
            docstring="",
        )
        self.builder.add_edge(parent_id, owner_id, "DEFINES")
        return owner_id

    def _binds_name(self, name):
        """Whether a line declaring ``name`` gives it a value of this file's
        own."""
        return True

    def _typed_origin(self, origin, declared):
        """Origin path of a name that a line both declares with the type
        ``declared`` and binds to the call at ``origin``; None where the
        language reads such a name by its type alone."""
        return

    def _is_local(self, name):
        """Whether ``name`` is a parameter or a variable of the definition
        the line being read belongs to."""
        return (self._scope, name) in self._locals

    def add_local(self, scope, name):
        """Record that ``name`` is a parameter or variable of ``scope``: a
        bare call of it there calls a value, not a definition of that name
        from outside ``scope``."""
        self._locals.add((scope, name))

    def _operand(self, text):
        """``(operand, unwraps)`` of the expression ``text``: what it is
        without the operators around it that leave its origin as it is, and
        how many times its value is unwrapped."""
        raise NotImplementedError

    def _expression_origin(self, code, start, end, lineno, since, owner):
        """Origin path of the value of the expression at columns ``start``
        to ``end`` of ``code``, the line being read: of the call chain it
        is, or of the name or the field it reads. None for any other."""
        plain, unwraps = self._operand(code[start:end].strip())
        unwraps = ".?" * unwraps
        if plain.endswith(")"):
            edge = self._closes.get(len(code[:end].rstrip().rstrip("?!")) - 1)
            if edge is None or not _is_call_chain(plain):
                return None
            return f"#{edge}{unwraps}"
        names = plain.split(".")
        if not all(re.fullmatch(r"[\w$]+", name) for name in names):
            return None
        # `self.field` has the type its struct declares, where this file
        # has that; any further name is a field of what is before it.
        size = 2 if names[0] == self.self_name else 1
        head = self.call_origin(".".join([*names[:size], "x"]), lineno, since, owner)
        if head is None and names[0] != self.self_name:
            return None
        head, size = (head[:-2], size) if head else ("self", 1)
        return ".".join([head, *names[size:]]) + unwraps

    def _rebind(self, name, lineno, origin, until=None, after=None):
        """Give ``name``, which a pattern binds on line ``lineno`` and which
        was read as of no declared type, the origin path ``origin``, up to
        line ``until`` where the pattern's block ends there, from where it
        is ``after`` again."""
        entries = self.declared.setdefault(name, [])
        # What a block binds, as a `for` does, is the block's.
        top = not until and self._binds_file(name)
        bound = [entry for entry in entries if entry[0] == lineno and not entry[2]]
        if bound:
            # The first of a line is what the name is read as.
            entries[entries.index(bound[0])] = (lineno, origin, top)
        else:
            entries.append((lineno, origin, top))
        if until and self._scope_end:
            # The binding is in scope up to the line before.
            self._binding_ends[lineno, name] = until - 1
        elif until:
            entries.append((until, after, False))

    def _bind(self, bindings, index, lines, block):
        """Give each name of ``bindings`` that a pattern on line ``index``
        binds its origin path there, up to the end of the block the line
        opens where ``block`` says it has one, else to the end of the line."""
        lineno = index + 1
        until = block and bindings and self._find_block_end(lines, index)
        for name, origin in bindings.items():
            self._rebind(name, lineno, origin, until or lineno + 1)

    def _bind_closures(self, code, index, lines, before):
        """Type the parameters of the closures that open before column
        ``before`` of line ``index`` and are arguments of a call, from what
        the call passes to them. ``self._closures`` has, for each closure
        of the line still to be read, the column it starts at, its
        parameters as written and the column its body starts at."""
        while self._closures and self._closures[0][0] < before:
            bar, params, end = self._closures.pop(0)
            opened, bracket = self.bracket_around(bar)
            edge, position, closed = None, "?", None
            if bracket != "(":
                continue
            if opened is not None:
                closed = self._pairs.get(opened)
                edge = self._closes.get(closed, self._opens.get(opened))
                position = (
                    len(
                        _bracketed_parts(
                            code[opened + 1 : bar] + "x",
                            expression=True,
                            turbofish=self.language == "rust",
                        )
                    )
                    - 1
                )
            elif self._open_calls:
                edge = self._open_calls[-1]
            if edge is None:
                continue
            if closed is not None:
                self._bodies.append((edge, position, end, closed))
            # In a block the names last to its end, else to the end of the
            # line or of the call.
            bindings = {
                name: _bound_origin(f"#{edge}|{position}|{number}", steps)
                for number, param in enumerate(_bracketed_parts(params + ")"))
                for name, steps in self.bindings(param)
            }
            block = closed is None and code.rstrip().endswith("{")
            self._bind(bindings, index, lines, block)

    def _name_results(self, code, lineno, since, owner):
        """Add to the origin path of each call that was passed a closure on
        the line what the closure returns, `{argument=origin}`, where its
        body is one call or one name whose origin is a call or a closure's
        parameter."""
        for edge, position, start, end in self._bodies:
            metadata = self.builder.edges[edge]["metadata"]
            body = self._expression_origin(code, start, end, lineno, since, owner)
            if body and re.fullmatch(r"#[\d|?]+", body) and "origin" in metadata:
                passed = f"{position}={body}"
                joined = metadata["origin"].endswith("}")
                metadata["origin"] = (
                    f"{metadata['origin'][:-1]},{passed}}}"
                    if joined
                    else f"{metadata['origin']}{{{passed}}}"
                )

    def add_call(self, source_id, callee, lineno, since, start, paren, origin=None):
        """Record a call of ``callee`` from ``source_id`` on ``lineno``, the
        line given to ``begin_line``. ``start`` and ``paren`` are the columns
        in it of the callee and of the `(` after it. A call on the result of
        another call (`a().b()`, or `.b()` on the next line) gets an origin
        path that names that call by its edge index, unless ``origin`` gives
        one, and the edge carries the column of the called name for a SCIP
        lookup."""
        name_start = start + len(callee) - len(_split_callee(callee)[1])
        if origin is None:
            # The code before the callee ends at ``end``, and the code before
            # a `.` there at ``receiver``, or without its `?` at ``called``.
            end = _run_start(self._code, start)
            dot = self._code[end - 1 : end] == "." if end else self._tail_dot
            receiver = _run_start(self._code, end - 1) if end else 0
            if dot and not receiver and self._tail is None and self._tail_name:
                # The receiver is the name the line before ends with.
                callee = f"{self._tail_name}.{callee}"
            elif dot and self._code[max(end - 2, 0) : end] != "..":
                called = receiver
                while called and self._code[called - 1] == "?":
                    called -= 1
                edge = self._closes.get(called - 1) if receiver else self._tail
                unwrap = ".?" * (receiver - called if receiver else self._tail_unwraps)
                origin = (
                    f"#{edge}{unwrap}.{callee}"
                    if edge is not None
                    else self._element_origin(called, callee, lineno, since, source_id)
                    or f"?.{callee}"
                )
        if origin is None and (source_id, callee) in self._locals:
            origin = f"={callee}"
        owner = source_id.split("::")[-2] if source_id.count("::") > 1 else None
        origin = origin or self.call_origin(callee, lineno, since, owner)
        metadata = {"lineno": lineno}
        if origin:
            metadata["origin"] = origin
        at, units = self._units if name_start >= self._units[0] else (0, (0, 0, 0))
        self._units = name_start, _code_units(self._raw[at:name_start], units)
        metadata["col"] = _scip_columns(self._units[1])
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

    @staticmethod
    def blank_lines(lines, spanning):
        """Each of ``lines`` as `blank_strings` blanks it, in turn."""
        inside, blanked = None, []
        for line in lines:
            code, inside = blank_strings(line, inside, spanning)
            blanked.append(code)
        return blanked

    def split_source(
        self,
        lines,
        code_lines,
        definition_head,
        type_head=None,
        ends=";",
        statement=None,
    ):
        """The lines to read ``lines`` as, split as `_split_lines` splits
        them; `restore_source_lines` gives what is read their source lines."""
        lines, self._rows, self._shifts = _split_lines(
            lines, code_lines, definition_head, type_head, ends, statement
        )
        self._first_rows = {
            row: at for at, row in reversed(list(enumerate(self._rows)))
        }
        return lines

    def read_line(self, lineno):
        """The first line read of source line ``lineno``."""
        return self._first_rows[lineno - 1] + 1 if self._rows else lineno

    def source_line(self, lineno):
        """The source line of line ``lineno`` read."""
        rows = self._rows
        if not rows or lineno < 1:
            return lineno
        return rows[min(lineno, len(rows)) - 1] + 1 + max(lineno - len(rows), 0)

    def restore_source_lines(self):
        """Give the nodes and the edges of the file being parsed the source
        lines and columns of the lines `split_source` read."""
        if not self._rows:
            return
        for item in self.builder.nodes[self._first_node :]:
            meta = item["metadata"]
            read = [meta.get(key) for key in ("lineno", "end_lineno")]
            for key in ("lineno", "end_lineno"):
                if isinstance(meta.get(key), int):
                    meta[key] = self.source_line(meta[key])
            # A definition's size is ten tokens to a line, of the source's:
            # a line split adds none.
            if all(isinstance(at, int) and at > 0 for at in read) and read[1] > read[0]:
                added = read[1] - read[0] - (meta["end_lineno"] - meta["lineno"])
                if added and meta.get("token_count", 0) > 10 * added:
                    meta["token_count"] -= 10 * added
        for item in self.builder.edges[self._first_edge :]:
            meta = item.get("metadata", {})
            if isinstance(meta.get("lineno"), int):
                lineno = meta["lineno"]
                shift = 1 <= lineno <= len(self._shifts) and self._shifts[lineno - 1]
                if "col" in meta and shift and any(shift):
                    col = meta["col"]
                    units = (col,) * 3 if isinstance(col, int) else col
                    meta["col"] = _scip_columns(tuple(map(sum, zip(units, shift))))
                meta["lineno"] = self.source_line(lineno)

    def add_file_node(self, rel_path, content):
        """Register the File node for a source file and return its id."""
        file_id = f"FILE:{rel_path}"
        self._first_node = len(self.builder.nodes)
        self._first_edge = len(self.builder.edges)
        self._rows = self._shifts = None
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

    def _node(self, node_id):
        """The node of this file with ID ``node_id``, if there is one."""
        if node_id in self.builder.node_registry:
            for node in reversed(self.builder.nodes):
                if node["id"] == node_id:
                    return node
        return None

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
        # file, not the first declaration under it; one further down documents
        # nothing.
        notice = _LICENSE_HEADER_RE if j < 0 else _LICENSE_NOTICE_RE
        if notice.search(docstring):
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

    def _find_block_end(self, lines, start_idx, column=0):
        """Find the closing brace line for a block starting at start_idx.

        Counts braces to find the matching closing brace, skipping
        comments and string/char literals, and on the first line what is
        before ``column`` of it once its block comments are out.
        Returns the 1-indexed line number of the closing brace.
        """
        brace_depth = 0
        found_open = False
        in_comment = False

        for i in range(start_idx, len(lines)):
            line, in_comment = strip_block_comments(lines[i], in_comment)
            line = line[column:] if i == start_idx else line
            if self._non_code:
                line = self._non_code.sub("", line)
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
            # A block opened and closed on its first line ends there, also
            # where the line goes on to close what encloses it (`f() {} };`).
            if found_open and brace_depth <= 0:
                return i + 1

        # If no closing brace found, estimate based on file end
        return min(start_idx + 100, len(lines))


_PY_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _py_start(node):
    """The line and column ``node`` starts at, its decorators included."""
    first = (getattr(node, "decorator_list", None) or [node])[0]
    return first.lineno, first.col_offset


def _py_end(node):
    """The line and column ``node`` ends at."""
    return node.end_lineno, node.end_col_offset


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
# The nodes ``PythonParser._push_bindings`` reads a binding, a narrowing or
# a returned value from
_PY_BINDING_NODES = (
    *_PY_DEFINITIONS,
    *_PY_COMPREHENSIONS,
    *_PY_NAME_BINDERS,
    ast.Assign,
    ast.AnnAssign,
    ast.NamedExpr,
    ast.Import,
    ast.ImportFrom,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
    ast.Lambda,
    ast.If,
    ast.IfExp,
    ast.While,
    ast.Assert,
    ast.Match,
    ast.Return,
    ast.Yield,
    ast.YieldFrom,
)
# The builtin type of what each literal and display builds
_PY_LITERALS = {
    ast.JoinedStr: "str",
    ast.List: "list",
    ast.ListComp: "list",
    ast.Dict: "dict",
    ast.DictComp: "dict",
    ast.Set: "set",
    ast.SetComp: "set",
    ast.Tuple: "tuple",
}
# The method a binary operator calls on its left operand
_PY_OPERATORS = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.MatMult: "__matmul__",
    ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.LShift: "__lshift__",
    ast.RShift: "__rshift__",
    ast.BitOr: "__or__",
    ast.BitXor: "__xor__",
    ast.BitAnd: "__and__",
}
# Origins of the decorators that make a method an attribute
_PY_PROPERTIES = ("property", "@functools.cached_property")
# The decorators that make a generator function return a context manager,
# which hands the `with` block what the generator yields
_PY_CONTEXT_MANAGERS = ("contextlib.contextmanager", "contextlib.asynccontextmanager")
# What an instance of a tuple, and of `type[C]`, starts with as a value
_PY_TUPLES = (("outside", "builtins.tuple"), ("outside", "typing.Tuple"))
_PY_CLASS_OBJECTS = (("outside", "builtins.type"), ("outside", "typing.Type"))


def _py_import_bindings(node):
    """Yield ``(local name, dotted target)`` for each name an import statement
    binds. A relative target keeps its leading dots, and a star import is
    yielded under its own target, `pkg.mod.*`."""
    for alias in node.names:
        if isinstance(node, ast.Import):
            root = alias.name.split(".")[0]
            yield (alias.asname, alias.name) if alias.asname else (root, root)
        else:
            module = "." * node.level + (node.module or "")
            sep = "." if node.module else ""
            target = f"{module}{sep}{alias.name}"
            name = target if alias.name == "*" else alias.asname or alias.name
            yield name, target


def _py_is_none(node):
    """Whether ``node`` is `None`, which says nothing of what the name it
    is assigned to holds otherwise."""
    return isinstance(node, ast.Constant) and node.value is None


def _py_targets(target, origin):
    """Yield ``(name, origin)`` for each name an assignment of the value at
    ``origin`` to ``target`` binds: `@N` leads to item N of what is
    unpacked."""
    if isinstance(target, ast.Name):
        yield target.id, origin
    elif isinstance(target, (ast.Tuple, ast.List)):
        starred = False
        for position, element in enumerate(target.elts):
            starred = starred or isinstance(element, ast.Starred)
            yield from _py_targets(element, "?" if starred else f"{origin}.@{position}")
    elif isinstance(target, ast.Starred):
        yield from _py_targets(target.value, "?")


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
        self.import_targets = {}
        self.bindings = []
        # id() of a name read in a comprehension, a lambda or a narrowed
        # block -> origin of what it is bound to there
        self.scoped = {}
        self._push_bindings(tree.body)
        self.builder.fields[file_id] = {
            name: sorted({origin for _, origin in bound})
            for name, bound in self.bindings[0].items()
        }
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
                if isinstance(base, ast.Subscript):
                    base = base.value
                base_name = (
                    self._get_attribute_name(base)
                    if isinstance(base, ast.Attribute)
                    else (base.id if isinstance(base, ast.Name) else None)
                )
                if base_name:
                    resolved_base = self.builder.current_scope.resolve(base_name)
                    self.builder.add_edge(
                        class_id,
                        f"REF:{resolved_base}",
                        "INHERITS",
                        {"origin": self._origin(base, node.lineno)},
                    )

            self.builder.fields[class_id] = self._class_attrs(node)
            for child in node.body:
                self.visit_node(child, class_id)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_id = f"{parent_id}::{node.name}"
            # A function is a method iff its enclosing parent node is a Class.
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
            if node.decorator_list and not is_method:
                # The name is bound to what the outermost decorator returns.
                made = self._origin(node.decorator_list[0], node.lineno)
                self.builder.fields.setdefault(parent_id, {})[node.name] = [f"{made}()"]

            self._push_bindings(node.body, node, is_method)
            origins, yields = self._returned_origins()
            fixture = self._fixture_name(node)
            if fixture:
                origin = origins[0] if len(origins) == 1 else None
                self.builder.fixtures[parent_id, fixture] = (func_id, origin, yields)
            # What calling the function gives, `()`, is what it returns
            # where it is no generator and neither an annotation nor a
            # decorator says otherwise. Under a context-manager decorator
            # the key is the decorator, for what the function yields.
            made = {self._origin(d, node.lineno) for d in node.decorator_list}
            manager = next((m for m in _PY_CONTEXT_MANAGERS if f"@{m}" in made), None)
            plain = not (
                node.returns or yields or made - {"staticmethod", "classmethod"}
            )
            step = "__anext__" if isinstance(node, ast.AsyncFunctionDef) else "__next__"
            if manager and node.returns:
                origins = [f"{self._instance_origin(node.returns)}.{step}()"]
            if origins and (plain or manager):
                held = self.builder.fields.setdefault(func_id, {})
                held[manager or "()"] = origins
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
        definitions are not entered: visit_node records their calls. The
        default arguments of a nested function are evaluated where it is
        defined, so their calls belong here.
        """
        todo = deque(body)
        while todo:
            node = todo.popleft()
            if isinstance(node, _PY_DEFINITIONS):
                if not isinstance(node, ast.ClassDef):
                    defaults = (*node.args.defaults, *node.args.kw_defaults)
                    todo.extend(default for default in defaults if default)
                continue
            todo.extend(ast.iter_child_nodes(node))
            if isinstance(node, ast.Call):
                self._record_call(node, parent_func_id, node.lineno)

    def _record_call(self, call_node, parent_func_id, lineno):
        index = len(self.builder.edges)
        self._record_callee(call_node.func, parent_func_id, lineno)
        if index == len(self.builder.edges):
            return
        # `#N|argument|P` is parameter P of the callable passed as that
        # argument to the call at edge index N.
        passed = [(str(at), arg) for at, arg in enumerate(call_node.args)]
        passed += [(keyword.arg, keyword.value) for keyword in call_node.keywords]
        for argument, value in passed:
            if isinstance(value, ast.Lambda):
                for position, arg in enumerate(value.args.args):
                    origin = f"#{index}|{argument}|{position}"
                    self._bind_scoped(value.body, ast.Name(arg.arg), origin)

    def _record_callee(self, func_node, parent_func_id, lineno):
        func_name = self._get_func_name(func_node)
        if func_name:
            resolved_name = self.builder.current_scope.resolve(func_name)
            origin = self._origin(func_node, lineno)
            if isinstance(func_node, ast.Name):
                # A variable that is called holds a value, `=name`, which
                # returns `Type` where it is annotated as a callable that
                # does: `=name>Type`.
                annotated = re.fullmatch(r":(?:\w+::)*Callable\[(.+)\]", origin)
                if origin == "?":
                    origin = f"={func_name}"
                elif annotated:
                    origin = f"={func_name}>{_bracketed_parts(annotated[1])[-1]}"
            metadata = {"lineno": lineno, "origin": origin}
            if func_node.end_lineno == lineno:
                # The called name ends where the callee expression does. It
                # is read from the line: the AST holds its NFKC form, which
                # can differ in length.
                line = self.source_lines[lineno - 1].encode()
                written = line[: func_node.end_col_offset].decode()
                name = re.search(r"\w*$", written)
                metadata["col"] = _scip_columns(_code_units(written[: name.start()]))
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
        before ``lineno``. A literal is a call of its builtin type, a
        subscript and a binary operator a call of the method they stand
        for. ``?`` stands for anything that has no such path."""
        if isinstance(node, ast.Name):
            if id(node) in self.scoped:
                return self.scoped[id(node)]
            read = getattr(node, "lineno", lineno), getattr(node, "col_offset", 0)
            for depth, frame in enumerate(reversed(self.bindings)):
                if node.id in frame:
                    bound = [b for b in frame[node.id] if depth or b[0] <= read]
                    return max(bound, key=itemgetter(0))[1] if bound else "?"
            imported = self.import_targets.get(node.id)
            return f"@{imported}" if imported else node.id
        if isinstance(node, ast.Attribute):
            return f"{self._origin(node.value, lineno)}.{node.attr}"
        if isinstance(node, ast.Call):
            return f"{self._origin(node.func, lineno)}()"
        if isinstance(node, ast.Await):
            return self._origin(node.value, lineno)
        literal = _PY_LITERALS.get(type(node))
        if isinstance(node, ast.Constant) and node.value not in (None, ...):
            literal = type(node.value).__name__
        if literal:
            return f"@builtins.{literal}()"
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            return f"{self._origin(node.value, lineno)}.__getitem__()"
        if isinstance(node, ast.BinOp) and type(node.op) in _PY_OPERATORS:
            return f"{self._origin(node.left, lineno)}.{_PY_OPERATORS[type(node.op)]}()"
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

    def _returned_origins(self):
        """``(origins, yields)`` for the function whose bindings were pushed
        last: the origins of the values other than `None` that it returns or
        yields, none where one of them has no origin path, and whether it
        yields."""
        origins, yields = set(), False
        for node in self.exits:
            if isinstance(node, ast.YieldFrom):
                iterated = self._origin(node.value, node.lineno)
                origins.add(f"{iterated}.__iter__().__next__()")
            elif node.value and not _py_is_none(node.value):
                origins.add(self._origin(node.value, node.lineno))
            yields = yields or not isinstance(node, ast.Return)
        known = not any("?" in origin for origin in origins)
        return sorted(origins) if known else [], yields

    def _instance_origin(self, annotation):
        """Origin of a value annotated with ``annotation``: `:type`, the
        type as written with its dots as `::`."""
        written = self._get_type_annotation(annotation)
        if "(" in written or "Complex" in written:
            return "?"
        return ":" + written.replace(".", "::")

    def _bind_scoped(self, owner, target, origin):
        """Bind the names an assignment of the value at ``origin`` to
        ``target`` binds, where ``owner`` reads them."""
        bound = dict(_py_targets(target, origin))
        for read in ast.walk(owner):
            if isinstance(read, ast.Name) and read.id in bound:
                self.scoped[id(read)] = bound[read.id]

    def _push_bindings(self, body, func=None, is_method=False):
        """Push a frame mapping each name bound in ``body`` (and by the
        parameters of ``func``) to its ``((line, column), origin)``
        bindings, at where each takes effect: the end of what binds it, or
        the start of the body a parameter, a loop, a `with` or an `except`
        binds it for. Module-level imports go to ``import_targets``."""
        frame = {}
        self.bindings.append(frame)

        def bind(target, at, origin):
            for name, bound in _py_targets(target, origin):
                frame.setdefault(name, []).append((at, bound))

        def iterated(value, loop, lineno):
            """Origin of what ``loop`` binds on each pass over ``value``."""
            step = "__iter__().__next__()"
            if isinstance(loop, ast.AsyncFor) or getattr(loop, "is_async", 0):
                step = "__aiter__().__anext__()"
            return f"{self._origin(value, lineno)}.{step}"

        def checked(test):
            """Yield ``(name, origin of an instance of the class)`` for
            each `isinstance(name, Class)` that holds where ``test`` does."""
            conjunction = isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
            for check in test.values if conjunction else [test]:
                if (
                    isinstance(check, ast.Call)
                    and isinstance(check.func, ast.Name)
                    and check.func.id == "isinstance"
                    and len(check.args) == 2
                    and isinstance(check.args[0], ast.Name)
                ):
                    yield check.args[0], self._instance_origin(check.args[1])

        def narrow(blocks, name, origin):
            """Type ``name`` where ``blocks`` read it, unless they rebind
            it."""
            reads = [
                read
                for block in blocks
                for read in ast.walk(block)
                if isinstance(read, ast.Name) and read.id == name.id
            ]
            if "?" not in origin and all(isinstance(r.ctx, ast.Load) for r in reads):
                self.scoped.update((id(read), origin) for read in reads)

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
                elif arg is args.vararg:
                    origin = "@builtins.tuple()"
                elif arg is args.kwarg:
                    origin = "@builtins.dict()"
                elif arg.annotation:
                    origin = self._instance_origin(arg.annotation)
                elif injected and arg.arg not in parametrized:
                    origin = f"~{arg.arg}"
                else:
                    origin = "?"
                frame[arg.arg] = [(_py_start(func.body[0]), origin)]

        found = []
        self.exits = []  # the `return` and `yield` nodes of ``body``
        todo = deque(body)
        while todo:
            node = todo.popleft()
            if isinstance(node, _PY_BINDING_NODES):
                found.append(node)
            if not isinstance(node, _PY_DEFINITIONS):
                todo.extend(ast.iter_child_nodes(node))
        for node in sorted(found, key=lambda n: getattr(n, "lineno", 0)):
            lineno = getattr(node, "lineno", 0)
            if isinstance(node, _PY_DEFINITIONS):
                frame.pop(node.name, None)
            elif isinstance(node, ast.Assign) and not _py_is_none(node.value):
                origin = self._origin(node.value, lineno)
                for target in node.targets:
                    bind(target, _py_end(node), origin)
            elif isinstance(node, ast.AnnAssign):
                origin = self._instance_origin(node.annotation)
                bind(node.target, _py_end(node), origin)
            elif isinstance(node, ast.NamedExpr):
                bind(node.target, _py_end(node), self._origin(node.value, lineno))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for name, target in _py_import_bindings(node):
                    if func is None:
                        self.import_targets[name] = target
                    else:
                        frame.setdefault(name, []).append((_py_end(node), f"@{target}"))
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                origin = iterated(node.iter, node, lineno)
                bind(node.target, _py_start(node.body[0]), origin)
            elif isinstance(node, _PY_COMPREHENSIONS):
                for generator in node.generators:
                    origin = iterated(generator.iter, generator, lineno)
                    self._bind_scoped(node, generator.target, origin)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                enter = "__enter__" if isinstance(node, ast.With) else "__aenter__"
                for item in node.items:
                    origin = f"{self._origin(item.context_expr, lineno)}.{enter}()"
                    bind(item.optional_vars, _py_start(node.body[0]), origin)
            elif isinstance(node, ast.Lambda):
                for arg in ast.walk(node.args):
                    if isinstance(arg, ast.arg):
                        self._bind_scoped(node.body, ast.Name(arg.arg), "?")
            elif isinstance(node, (ast.If, ast.IfExp, ast.While)):
                body = node.body if isinstance(node.body, list) else [node.body]
                for name, origin in checked(node.test):
                    narrow([node.test, *body], name, origin)
            elif isinstance(node, ast.Assert):
                for name, origin in checked(node.test):
                    if name.id in frame and "?" not in origin:
                        bind(name, _py_end(node), origin)
            elif isinstance(node, ast.Match) and isinstance(node.subject, ast.Name):
                for case in node.cases:
                    pattern = getattr(case.pattern, "pattern", None) or case.pattern
                    if isinstance(pattern, ast.MatchClass):
                        origin = self._instance_origin(pattern.cls)
                        narrow(case.body, node.subject, origin)
            elif isinstance(node, _PY_NAME_BINDERS) and node.name:
                origin = "?"
                if isinstance(getattr(node, "pattern", None), ast.MatchClass):
                    origin = self._instance_origin(node.pattern.cls)
                body = getattr(node, "body", None)
                at = _py_start(body[0]) if body else _py_end(node)
                frame.setdefault(node.name, []).append((at, origin))
            elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
                self.exits.append(node)

    def _class_attrs(self, class_node):
        """Map each attribute of ``class_node`` to the origin of the value
        it is annotated with or of what a property of that name is
        annotated to return, else to the origins of the values other than
        `None` that its body and its methods assign to it."""
        attrs = {}
        assigned = {}
        for stmt in class_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                origin = self._instance_origin(stmt.annotation)
                attrs.setdefault(stmt.target.id, origin)
            elif isinstance(stmt, ast.Assign) and not _py_is_none(stmt.value):
                origin = self._origin(stmt.value, stmt.lineno)
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        assigned.setdefault(target.id, set()).add(origin)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                made = [self._origin(d, stmt.lineno) for d in stmt.decorator_list]
                read = any(maker in _PY_PROPERTIES for maker in made)
                if read and stmt.returns:
                    origin = self._instance_origin(stmt.returns)
                    attrs.setdefault(stmt.name, origin)
                self._push_bindings(stmt.body, stmt, is_method=True)
                returned, yields = self._returned_origins()
                if read and not stmt.returns and not yields:
                    assigned.setdefault(stmt.name, set()).update(returned)
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Assign):
                        targets, value = node.targets, node.value
                    elif isinstance(node, ast.AnnAssign):
                        targets, value = [node.target], node.annotation
                    else:
                        continue
                    for target in targets:
                        if (
                            not isinstance(target, ast.Attribute)
                            or self._origin(target.value, node.lineno) != "self"
                        ):
                            continue
                        if isinstance(node, ast.AnnAssign):
                            origin = self._instance_origin(value)
                            attrs.setdefault(target.attr, origin)
                        elif not _py_is_none(value):
                            origin = self._origin(value, node.lineno)
                            assigned.setdefault(target.attr, set()).add(origin)
                self.bindings.pop()
        held = {name: sorted(origins) for name, origins in assigned.items()}
        held.update((name, [origin]) for name, origin in attrs.items())
        return {
            name: origins
            for name, origins in held.items()
            if not any("?" in origin for origin in origins)
        }

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


# Where a pattern binds names: closure parameters, a `for` pattern, a `let`
# with a pattern rather than one name, and a match arm.
_RUST_BINDER_RES = (
    re.compile(r"\|([^|]*)\|"),
    re.compile(r"\bfor\s++(.+?)\s+in\b"),
    re.compile(r"\b(?:if|while)\s++let\s++(.+?)=(?!=)"),
    re.compile(r"\blet\s+(?:mut\s+)?([({\[A-Z].*?)=(?!=)"),
    re.compile(r"^\s*(.+?)=>"),
)


# `let name = vec![...];`, `let name = "...";` and `const NAME: Type = ...`,
# which is read in every function below where it is declared
_RUST_VEC_LET_RE = re.compile(
    r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*=\s*vec!\[[^\[\]]*\];"
)
_RUST_STRING_LET_RE = re.compile(
    r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*=\s*(?P<bytes>b)?r?#*\""
)
_RUST_CONST_RE = re.compile(
    r"\b(?:const|static)\s+(?:mut\s+)?(?P<name>[A-Z][A-Z0-9_]*)\s*:\s*"
    r"(?P<type>[^=;().]+?)\s*="
)
_RUST_INDEX_LET_RE = re.compile(
    r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*=\s*(?P<of>\w+)\s*\["
)
# The traits a `#[derive(...)]` implements for the item below it
_RUST_DERIVE_RE = re.compile(r"#\[derive\(([^)]*)\)\]")
# Type arguments written after a path's `::`, nested to three levels
_RUST_TURBOFISH = r"::<(?:[^<>()]|<(?:[^<>()]|<[^<>()]*>)*>)*>"
# The value in the head of a block, up to the `{` that opens it: the first
# outside parentheses, as no struct literal stands there bare
_RUST_HEAD_VALUE = r"[^(){}]*"
for _ in range(4):
    _RUST_HEAD_VALUE = rf"(?:[^(){{}}]|\({_RUST_HEAD_VALUE}\))*"
_RUST_HEAD_VALUE = rf"(?P<value>{_RUST_HEAD_VALUE}?)\s*\{{"
# Where a pattern is matched against a value: `for` and `if let` or `while
# let` with the block they open, `let` with a pattern, and `match`
_RUST_FOR_RE = re.compile(r"\bfor\s++(?P<pattern>.+?)\s+in\s+" + _RUST_HEAD_VALUE)
_RUST_IF_LET_RE = re.compile(
    r"\b(?:if|while)\s++let\s++(?P<pattern>.+?)\s*=(?!=)\s*" + _RUST_HEAD_VALUE
)
_RUST_LET_RE = re.compile(
    r"\blet\s+(?P<pattern>[^=]*[(\[{][^=]*?)\s*=(?!=)\s*(?P<value>.+?)"
    r"(?:;|\s+else\s*\{)\s*$"
)
_RUST_MATCH_RE = re.compile(r"\bmatch\s++" + _RUST_HEAD_VALUE)
_RUST_ARM_RE = re.compile(r"^\s*+(?P<pattern>.+?)(?:\s+if\s.+?)?\s*=>")
# The parameters of a closure passed to a call
_RUST_CLOSURE_RE = re.compile(r"(?:(?<=[(,])|^)\s*(?:move\s+)?\|(?P<params>[^|]*)\|")
# How a type is borrowed or abstracted over, before its name
_RUST_BORROW_RE = re.compile(r"^(?:[&\s]|'\w+\s|(?:mut|dyn|impl)\s)+")


@cache
def _rust_declarations():
    """The patterns ``scan_declarations`` reads a Rust file with. The
    primitive types are those the standard library's data names."""
    scope = _stdlib.load("rust")["scope"]
    primitives = "|".join(sorted(name for name, key in scope.items() if name == key))
    name = r"\b(?P<name>[a-z_]\w*)\s*:\s*(?:&\s*)?(?:'\w+\s+)?(?:mut\s+)?"
    # Type arguments, nested three deep, that name no tuple or function type
    plain = r"[^<>().;={}]*"
    arguments = rf"<{plain}(?:<{plain}(?:<{plain}>{plain})*>{plain})*>"
    return (
        # `name: Type`, `name: &'a mut Type`, `name: &dyn Trait`, `name: Type<'a>`;
        # not `name: None`, a field of a struct literal. A type is read with
        # its type arguments, `Vec<Type>`, or marked generic where they are
        # more than those: unless the project defines it, its methods are
        # mostly those of what it wraps.
        re.compile(
            rf"{name}(?:dyn\s+|impl\s+)?(?P<type>(?!None\b)(?:[a-z_]\w*::)*[A-Z]\w*"
            rf"\b(?![:(!])(?:(?!<'\w+>){arguments})?)(?:<'\w+>|(?P<generic><))?"
        ),
        # `name: &[Type]`, `name: [u8; 4]`
        re.compile(
            rf"{name}(?P<type>\[\s*(?:&\s*)?(?:'\w+\s+)?(?:mut\s+)?"
            rf"(?:(?:[a-z_]\w*::)*[A-Z]|(?:{primitives})\b)[^\[\](),.={{}}]*\])"
        ),
        # `name: &str`, `name: usize`
        re.compile(rf"{name}(?P<type>{primitives})\b(?!\s*[:(!<])"),
        # `let name = Type { ... }`
        re.compile(r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*=\s*(?P<type>[A-Z]\w*)\s*\{"),
        # A type parameter's bound, in `<T: Trait>` or a `where` clause
        re.compile(
            r"(?:[<,]|\bwhere\s|^)\s*(?P<param>[A-Z]\w*)\s*:\s*"
            r"(?P<type>[A-Z]\w*)\b(?![<:(!])"
        ),
    )


# A TypeScript alias written on one line: its name, its type parameters
# and the type it stands for
_TS_ALIAS_RE = re.compile(
    r"^[ \t]*(?:export\s+)?(?:declare\s+)?type\s+([\w$]+)\s*(?:<(.*?)>)?\s*=\s*"
    r"([^;\n]+?);?[ \t]*$",
    re.MULTILINE,
)
# `type Name = ...` in Rust or TypeScript, behind any visibility or `export`
_TYPE_ALIAS_RE = re.compile(
    r"^[ \t]*(?:[\w()]+\s+)*type\s+([\w$]+)[^=;{]*=", re.MULTILINE
)

# What can follow the `>` that ends type arguments in an expression: what
# starts no operand, other than `<`, `>`, `+` and `-`
_TYPE_ARGUMENTS_END_RE = re.compile(
    r"""\s*+(?:[(`]|(?:as|in|instanceof|satisfies)\b|(?![\w$'"[{!~#@<>+-]|\.\d))"""
)


def _bracketed_parts(text, separator=",", expression=False, turbofish=False):
    """The ``separator``-separated parts of ``text``, which follows an
    opening bracket, up to the bracket that closes it. A `<` is a bracket
    unless the bracket it is in closes first; in an ``expression`` it is
    one only where a `>` closes it that type arguments can end with, and
    with ``turbofish``, Rust's, only after `::`."""
    angles, opened, end = set(), [], len(text)
    for column, char in enumerate(text):
        if char == "<" and turbofish and text[column - 2 : column] != "::":
            continue
        if char in "<([{":
            opened.append(column)
        elif char == ">" and text[column - 1] not in "-=":
            if opened and text[opened[-1]] == "<":
                angle = opened.pop()
                if (
                    not expression
                    or (opened and text[opened[-1]] == "<")
                    or _TYPE_ARGUMENTS_END_RE.match(text, column + 1)
                ):
                    angles.update((angle, column))
            elif not opened and not expression:
                end = column
                break
        elif char in ")]}":
            while opened and text[opened[-1]] == "<":
                opened.pop()
            if not opened:
                end = column
                break
            opened.pop()
    if not expression:
        angles.update(column for column in opened if text[column] == "<")
    parts, depth, start = [], 0, 0
    for column, char in enumerate(text[:end]):
        if char in "([{" or char == "<" and column in angles:
            depth += 1
        elif char in ")]}" or char == ">" and column in angles:
            depth -= 1
        elif char == separator and not depth:
            parts.append(text[start:column])
            start = column + 1
    parts.append(text[start:end])
    return [part.strip() for part in parts if part.strip()]


def _bound_origin(origin, steps):
    """The origin of a name a pattern binds: the steps from the value's
    ``origin`` to its part, `?` for a part of no type the value gives, and
    None where the value's is not known."""
    if steps is None:
        return "?"
    return origin and ".".join([origin, *steps])


def _rust_held(text):
    """``{position or field name: type}`` of what the `(A, B)` or the `{ a:
    A }` after the name of a struct or of a variant says it holds."""
    text = re.sub(r"^\s*<[^(){]*>", "", text).lstrip()
    if text[:1] not in ("(", "{"):
        return {}
    parts = [re.sub(rf"^{_RUST_VIS}", "", p) for p in _bracketed_parts(text[1:])]
    if text[0] == "(":
        return dict(enumerate(parts))
    fields = (part.partition(":") for part in parts)
    return {name.strip(): held.strip() for name, _, held in fields if held}


def _rust_bindings(pattern, steps=()):
    """Yield ``(name, steps)`` for each name the Rust pattern binds: the
    steps lead from the value matched to the part bound, `Some@0` for what
    `Some(..)` holds first, `@1` for the second of a tuple, `Point@x` for
    a field and `*` for an element of a slice."""
    # The first of `A | B`: every alternative binds the same names.
    first = _bracketed_parts(pattern.replace("|", ",") + ")")[:1]
    pattern = re.sub(r"^(?:&\s*|(?:mut|ref|box)\s+)+", "", "".join(first))
    name, at, rest = pattern.partition("@")
    if at:
        yield from _rust_bindings(name, steps)
        pattern = rest.strip()
    opened = re.match(r"((?:\w+::)*[A-Z]\w*)?\s*([(\[{])", pattern)
    if re.fullmatch(r"[a-z]\w*|_\w+", pattern):
        yield pattern, steps
    elif opened:
        head = (opened[1] or "").rpartition("::")[2]
        for position, part in enumerate(_bracketed_parts(pattern[opened.end() :])):
            if part.startswith(".."):
                break
            field, _, inner = part.partition(":")
            if opened[2] == "[":
                yield from _rust_bindings(part, (*steps, "*"))
            elif opened[2] == "(":
                yield from _rust_bindings(part, (*steps, f"{head}@{position}"))
            else:
                field = field.split()[-1]
                yield from _rust_bindings(inner or part, (*steps, f"{head}@{field}"))


@cache
def _rust_type_parts(text):
    """``(name, type arguments, associated types)`` of a Rust type as
    written, whatever it is borrowed as. A slice, an array and a tuple have
    the names the standard library's data gives them: `&'a [T]` is `[T]`."""
    text = _RUST_BORROW_RE.sub("", text)
    if text.startswith("["):
        element, _, length = ", ".join(_bracketed_parts(text[1:])).partition(";")
        if length:
            return "[T; N]", (element.strip(), length.strip()), {}
        return "[T]", (element,), {}
    if text.startswith("("):
        held = tuple(_bracketed_parts(text[1:]))
        return ("(T,)" if held else "()"), held, {}
    name = re.match(r"[\w:]*", text)[0]
    parts = []
    if text.startswith("<", len(name)):
        parts = _bracketed_parts(text[len(name) + 1 :])
    associated = dict(re.split(r"\s*=\s*", p, maxsplit=1) for p in parts if "=" in p)
    return (
        name,
        tuple(p for p in parts if "=" not in p and not p.startswith("'")),
        associated,
    )


@cache
def _php_type_parts(text):
    """``(name, type arguments, named arguments)`` of a PHP type as the
    parser and the standard library's data write it: a class by its full
    name, `::App::User` or `App\\User`, `static` and `this` as `Self`, with
    its type arguments. A type that may also be `null` or `false` is the
    type; any other union has no name. An `iterable` names its second
    argument, what a `foreach` over it gives, `TValue`."""
    alternatives, depth, start = [], 0, 0
    for at, char in enumerate(f"{text}|"):
        depth += (char in "<({") - (char in ">)}")
        if char == "|" and not depth:
            alternatives.append(text[start:at].strip())
            start = at + 1
    alternatives = [a for a in alternatives if a not in ("null", "false")]
    if len(alternatives) != 1:
        return "", (), {}
    name, opened, rest = alternatives[0].partition("<")
    arguments = tuple(_bracketed_parts(rest)) if opened else ()
    name = name.removeprefix("::").replace("::", "\\")
    if name in ("static", "this"):
        name = "Self"
    named = {"TValue": arguments[1]} if name == "iterable" and arguments[1:] else {}
    return name, arguments, named


@cache
def _go_type_parts(text):
    """``(name, type arguments, {})`` of a Go type as written, whatever it
    points at: `*bytes.Buffer` is `bytes.Buffer`. A type Go writes without
    a name has the one the standard library's data gives it, and as its
    arguments what it holds: `[]T` and `[n]T` are `[]` of `int`, the index,
    and `T`; `map[K]V` is `map` of `K` and `V`; `chan T` is `chan` of `T`;
    the results `(A, B)` are `()` of `A` and `B`. A function has no name."""
    # An origin path writes the `.` of a qualified type as `::`.
    text = text.lstrip("*").replace("::", ".")
    if text.startswith("("):
        return "()", tuple(_bracketed_parts(text[1:])), {}
    if text.startswith("["):
        return "[]", ("int", text[text.index("]") + 1 :]), {}
    if text.startswith("map["):
        key = _bracketed_parts(text[4:])[0]
        return "map", (key, text[text.index(key, 4) + len(key) + 1 :]), {}
    if channel := re.match(r"(?:<-\s*)?chan\b(?:\s*<-)?\s*", text):
        return "chan", (text[channel.end() :],), {}
    name = re.match(r"[\w$./-]*", text)[0]
    arguments = ()
    if text.startswith("[", len(name)):
        arguments = tuple(_bracketed_parts(text[len(name) + 1 :]))
    return name, arguments, {}


# What a value of each of Java's primitive types is boxed as
_JAVA_BOXES = {
    "boolean": "Boolean",
    "byte": "Byte",
    "char": "Character",
    "short": "Short",
    "int": "Integer",
    "long": "Long",
    "float": "Float",
    "double": "Double",
}
# The class a Java type has as a base without naming it, by the kind of its
# declaration; `java.lang.Object` for any other
_JAVA_ROOTS = {
    "enum": "java.lang.Enum",
    "record": "java.lang.Record",
    "@interface": "java.lang.annotation.Annotation",
}


@cache
def _java_type_parts(text):
    """``(name, type arguments, {})`` of a Java type as written. A wildcard
    is the bound it extends and a primitive its box; an array, which has
    the methods of `Object`, is that class holding its element type."""
    # An origin path writes the `.` of a qualified type as `::`.
    text = re.sub(r"^\?\s*extends\s+", "", text.replace("::", ".").strip())
    if text.endswith(("]", "...")):
        element = text.rpartition("[" if text.endswith("]") else "...")[0]
        return "java.lang.Object", (element,), {}
    name = re.match(r"[\w$.]*", text)[0]
    arguments = ()
    if text.startswith("<", len(name)):
        arguments = tuple(_bracketed_parts(text[len(name) + 1 :]))
    return _JAVA_BOXES.get(name, name), arguments, {}


def _dotnet_type_parts(text):
    """``(name, type arguments, {})`` of a C# type as written. The name
    carries the arity its data keys it by: `List` written with one type
    argument is ``List`1``. An array is `System.Array` holding its
    element type, a nullable value type the type it wraps."""
    text = text.replace("::", ".").strip()
    if text.endswith("[]"):
        return "System.Array", (text[:-2].strip(),), {}
    if text.endswith("?") and "[" not in text:
        return _dotnet_type_parts(text[:-1])
    name = re.match(r"[\w$.`]*", text)[0]
    arguments = ()
    if text.startswith("<", len(name)) and text.endswith(">"):
        inner = text[len(name) + 1 : -1]
        # `Type<>` written bare takes one argument; a name the data keys
        # by its backticked arity carries it already. `System.Array<e>`
        # is the array of `e`, which the data keys without arity.
        arguments = tuple(_bracketed_parts(inner)) or (("",) if not inner else ())
        if "`" not in name and name != "System.Array":
            name = f"{name}`{len(arguments)}"
    return name, arguments, {}


def _clang_type_parts(text):
    """``(name, type arguments, {})`` of a C or C++ type as written, or
    the standard library's data writes it: `std::vector<int>` is
    `std::vector` with one argument. The name drops the specifiers and
    the pointer/reference markers a declaration wraps it in."""
    text = re.sub(
        r"^(?:const|volatile|static|inline|constexpr|virtual|struct|class|enum)\s+",
        "",
        text.strip(),
    )
    text = re.sub(r"\s*[*&]+\s*$", "", text).strip()
    name = re.match(r"[\w:$.`]+", text)
    name = name[0] if name else ""
    arguments = ()
    if text.startswith("<", len(name)) and text.endswith(">"):
        inner = text[len(name) + 1 : -1]
        arguments = tuple(_bracketed_parts(inner)) or (("",) if not inner else ())
    return name, arguments, {}


def _ruby_type_parts(text):
    """``(name, type arguments, {})`` of a Ruby type as written, or the
    standard library's data writes it: `Array[String]` is `Array` with one
    argument."""
    text = text.strip()
    match = re.match(r"[A-Z_a-z][\w:$.]*", text)
    if not match:
        return "", (), {}
    name = match[0]
    arguments = ()
    if text.startswith("[", len(name)) and text.endswith("]"):
        arguments = tuple(_bracketed_parts(text[len(name) + 1 : -1]))
    return name, arguments, {}


def _dart_type_parts(text):
    """``(name, type arguments, {})`` of a Dart type as written:
    ``List<E>`` is ``List`` with one argument; a nullable is the type it
    wraps."""
    text = text.strip().rstrip("?")
    match = re.match(r"[A-Za-z_][\w$.]*", text)
    if not match:
        return "", (), {}
    name = match[0]
    arguments = ()
    if text.startswith("<", len(name)) and text.endswith(">"):
        arguments = tuple(_bracketed_parts(text[len(name) + 1 :]))
    return name, arguments, {}


@cache
def _java_packages():
    """The first name of every package the Java standard library's data
    has a type of."""
    return frozenset(key.partition(".")[0] for key in _stdlib.load("java")["types"])


# What `typing` writes around a type to say how a variable is used, or that
# a value is awaited: read as the type it holds
_PY_QUALIFIERS = frozenset(
    {
        "Annotated",
        "Awaitable",
        "ClassVar",
        "Coroutine",
        "Final",
        "NotRequired",
        "ReadOnly",
        "Required",
    }
)


@cache
def _py_type_parts(text):
    """``(name, type arguments, {})`` of a Python type as an annotation or
    the standard library's data writes it, also between quotes and with the
    `::` an origin path has for its dots. `Optional[T]` and a union of `T`
    with `None` or `Any` are `T`; any other union has no name and the types
    it is of as its arguments. A qualifier (`Final[T]`, `Annotated[T, ...]`)
    is the type it holds and an awaitable what awaiting it gives. A tuple
    has, before the types of its items, the one type they all are, if any:
    `tuple[T, ...]` has that alone."""
    text = text.strip("'\" ").replace("::", ".")
    members, depth, start = [], 0, 0
    for column, char in enumerate(f"{text}|"):
        depth += (char == "[") - (char == "]")
        if char == "|" and not depth:
            members.append(text[start:column].strip())
            start = column + 1
    members = [m for m in members if m.rpartition(".")[2] not in ("None", "Any")]
    if len(members) != 1:
        return "", tuple(members), {}
    name, bracket, rest = members[0].partition("[")
    arguments = tuple(_bracketed_parts(rest)) if bracket else ()
    short = name.rpartition(".")[2]
    if short in ("Optional", "Union"):
        return _py_type_parts(" | ".join(arguments))
    if short in _PY_QUALIFIERS and arguments:
        return _py_type_parts(arguments[-1 if short == "Coroutine" else 0])
    if short == "Self":
        return short, (), {}
    if short in ("tuple", "Tuple") and arguments:
        if arguments[1:] in (("...",), ("Ellipsis",)):
            arguments = arguments[:1]
        else:
            shared = arguments[0] if len(set(arguments)) == 1 else ""
            arguments = (shared, *arguments)
    return name, arguments, {}


@cache
def _py_std_names():
    """The dotted names Python's standard-library data has: its types, its
    functions and variables, and the modules and classes they are in."""
    data = _stdlib.load("python")
    return frozenset(
        key.rsplit(".", cut)[0]
        for key in (*data["types"], *(name.lstrip(".") for name in data["functions"]))
        for cut in range(key.count(".") + 1)
    )


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
# A type parameter bound by a closure trait: `F: Fn(A) -> R`
# A closure trait a parameter is declared with: `impl Fn(A, B)`, `&dyn FnMut(A)`
_RUST_FN_TYPE_RE = re.compile(r"\b(?:impl|dyn)\s+(?:Fn|FnMut|FnOnce)\s*\(([^()]*)\)")
_RUST_FN_BOUND_RE = re.compile(r"\b(\w+)\s*:\s*((?:Fn|FnMut|FnOnce)\s*\([^()]*\))")
_RUST_PARAMETER_RE = re.compile(r"[(,]\s*(?:mut\s+)?([a-z_]\w*)\s*:")
# A `fn` with its body on the line of the `impl` block it is in
_RUST_INLINE_FN_RE = re.compile(
    r"\bfn\s+(?P<name>\w+)[^{;]*?(?:->(?P<returns>[^{;]+))?\{(?P<body>[^{}]*)\}"
)


# Where an `impl` block's header starts
_RUST_IMPL_RE = re.compile(r"^\s*(?:unsafe\s+)?impl(?=[<\s]|$)")
# A bracketed group that holds no other
_RUST_GROUP_RE = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|<[^<>]*>")
# What an `impl` header is cut at, and the brackets that nest
_RUST_HEADER_RE = re.compile(r"->|[<(\[]|[>)\]]|\bfor\b|\bwhere\b|[{;]")


def _rust_named(text):
    """The path a type or trait is written with, behind `&`, `dyn` and the
    like; None where it is not written with a path, as a macro call is not."""
    text = _RUST_BORROW_RE.sub("", text)
    named = re.match(r"(?:\w*::)*(?!fn\b)\w+", text)
    called = named and text[named.end() :].lstrip().startswith("!")
    return named[0] if named and not called else None


def _last_segment(path):
    """The last segment of ``path``, None for None."""
    return path and path.rpartition("::")[2]


def _rust_impl(text):
    """What ``text``, an `impl` header read from after `impl`, says: the
    trait, the owner of the block's items, and the bounds of a blanket `impl`.

    The trait is given by its path as written, the owner by the last
    segment of its type's: `<'de> de::Deserialize<'de> for Wrap<T> {` gives
    `("de::Deserialize", "Wrap", None)`. An inherent `impl` has no trait. A
    type source cannot name, as a tuple or a parameter of the block, gives
    an owner no source can name either: `<Trait for (A, B)>`. Where the type
    is a parameter of the block, or a reference to one, the block is for
    every type that meets the parameter's bounds, which are given by name;
    otherwise the bounds are None."""
    # The text is cut where no bracket is open: after the block's own
    # parameters, at `for`, and at `where` or the `{` that ends the header.
    depth, cuts = 0, {}
    for found in _RUST_HEADER_RE.finditer(text):
        token = found[0]
        if token in "<([":
            depth += 1
        elif token in ">)]":
            depth -= 1
            if not depth:
                cuts.setdefault("<", found.end())
        elif token != "->" and not depth:
            if token in "{;":
                cuts["{"] = found.start()
                break
            cuts.setdefault(token, found.start())
    opened = cuts.get("<", 0) if text.lstrip().startswith("<") else 0
    end = cuts.get("{", len(text))
    constrained = cuts.get("where", end)
    named = text[opened:constrained]
    trait = None
    if "for" in cuts and opened <= cuts["for"] < constrained:
        trait = _rust_named(text[opened : cuts["for"]])
        named = text[cuts["for"] + 3 : constrained]
    # Each parameter and each `where` predicate, without what it nests.
    declared = f"{text[:opened].strip()[1:-1]},{text[constrained:end][5:]}"
    while (reduced := _RUST_GROUP_RE.sub("", declared)) != declared:
        declared = reduced
    parameters = dict(
        re.findall(r"(?:^|,)\s*(?:const\s+)?(\w+)\s*(?::([^,]*))?", declared)
    )
    owner = _last_segment(_rust_named(named))
    if owner not in parameters:
        bounds = None
    else:
        bounds = []
        for part in declared.split(","):
            name, _, bound = part.partition(":")
            if name.strip() == owner:
                # `for<'a> Trait<'a>` is a bound by `Trait`.
                traits = re.sub(r"\bfor\b", "", bound).split("+")
                bounds += filter(None, map(_last_segment, map(_rust_named, traits)))
    if not owner or bounds is not None:
        if not trait:
            return None, None, None
        written = re.sub(r"(?:\w*::)+", "", " ".join(named.split()))
        owner = f"<{_last_segment(trait)} for {written}>"
    return trait, owner, bounds


# What a Rust item that holds others opens its body after, and an item a
# line may go on to after the end of another
_RUST_TYPE_HEAD_RE = re.compile(
    rf"\s*(?:{_RUST_VIS}\s*)?(?:unsafe\s+)?(?:impl|trait|mod)\b"
)
_RUST_DEFINITION_HEAD_RE = re.compile(
    rf"(?:{_RUST_VIS}\s*)?(?:(?:const|async|unsafe|extern(?:\s+\"[^\"]*\")?)\s+)*"
    r"(?:fn|impl|struct|enum|union|trait|mod|type|static|const|macro_rules!)\b"
)


class RustParser(BaseParser):
    language = "rust"
    binding_re = re.compile(r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*=(?!=)")
    bindings = staticmethod(_rust_bindings)

    def _operand(self, text):
        text = re.sub(r"^(?:[&*]\s*(?:mut\s+)?)+", "", text)
        plain = text.rstrip("?")
        return plain, len(text) - len(plain)

    def _bind_arm(self, code, index, lines):
        """Type the names that an arm of a `match` binds on line ``index``,
        from the value matched, up to the end of the arm."""
        indent = len(code) - len(code.lstrip())
        while self._matches and code.strip() and indent <= self._matches[-1][0]:
            self._matches.pop()
        found = self._matches and "=>" in code and _RUST_ARM_RE.search(code)
        if found and self._matches[-1][1]:
            bindings = {
                name: ".".join([self._matches[-1][1], *steps])
                for name, steps in self.bindings(found["pattern"])
            }
            self._bind(bindings, index, lines, code.rstrip().endswith("{"))

    def _bind_patterns(self, code, index, lines, since, owner):
        """Type the names that a `for`, an `if let`, a `while let` or a
        `let` with a pattern binds on line ``index``, from the value the
        pattern is matched against, and note the value of a `match`."""
        lineno = index + 1
        found = None
        if "for " in code:
            found = _RUST_FOR_RE.search(code)
        if not found and "let " in code:
            found = _RUST_IF_LET_RE.search(code) or _RUST_LET_RE.search(code)
        if found and not re.match(r"(?:mut\s+)?\w+\s*:", found["pattern"]):
            span = found.span("value")
            origin = self._expression_origin(code, *span, lineno, since, owner)
            if origin and found.re is _RUST_FOR_RE:
                origin += ".*"
            bindings = {
                name: ".".join([origin, *steps])
                for name, steps in self.bindings(found["pattern"])
                if origin
            }
            if found.re is _RUST_LET_RE:
                for name, bound in bindings.items():
                    self._rebind(name, lineno, bound)
            else:
                self._bind(bindings, index, lines, True)
        if "match " in code and (found := _RUST_MATCH_RE.search(code)):
            span = found.span("value")
            origin = self._expression_origin(code, *span, lineno, since, owner)
            indent = len(code) - len(code.lstrip())
            self._matches.append((indent, origin))

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

        lines = source_lines(content)
        lines = self.split_source(
            lines,
            self.blank_lines(lines, ('"', "r")),
            _RUST_DEFINITION_HEAD_RE,
            _RUST_TYPE_HEAD_RE,
        )
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
        re_mod = re.compile(rf"^\s*{_RUST_VIS}mod\s+([a-zA-Z0-9_]+)\s*\{{?")
        re_const = re.compile(
            r"^\s*(?:pub(?:\(.*?\))?\s+)?(?:const|static)\s+([A-Z0-9_]+)\s*:\s*.*?\s*=\s*(.*?);"
        )
        re_call = re.compile(rf"((?:{_RUST_TURBOFISH}|[a-zA-Z0-9_:.<>])+)\(")
        self.scan_declarations(lines, _rust_declarations())
        # A name a pattern binds has, from there on, no declared type.
        for lineno, line in enumerate(lines, 1):
            for binder in _RUST_BINDER_RES:
                for pattern in binder.findall(line):
                    for name in re.findall(r"\b[a-z_]\w*\b", pattern):
                        self.declared.setdefault(name, []).append((lineno, None, False))
        aliases = _TYPE_ALIAS_RE.findall(content)
        self.builder.type_aliases.update(aliases)
        self.builder.rust_aliases[file_id] = aliases
        for tree in _RUST_USE_RE.findall(content):
            self.builder.imports.setdefault(file_id, {}).update(_use_leaves(tree))

        current_impl_target = impl_bounds = None
        current_trait_impl = (
            None  # Track trait name when in "impl Trait for Struct" block
        )
        current_enum_id = None  # Track if we're inside an enum for variant parsing
        # id of each impl block's owner -> (name, lineno, bounds of a blanket one)
        impl_targets = {}
        fn_start = 0  # first line of the function being read
        fn_ids = set()  # functions whose body is a context; a `fn` in one is nested
        self._matches = []  # (indentation, origin of the value) of each open `match`

        def opens_block(text):
            """True if ``text`` leaves a `{` open; `struct P { x: i32 }` does not."""
            return text.count("{") > text.count("}")

        # Each line without comments and literal text, and whether it lies
        # partly inside a comment or literal that spans lines.
        codes = []
        inside = None
        for lineno, raw in enumerate(lines, 1):
            code, ends_inside = blank_strings(raw, inside, ('"', "r"))
            codes.append((code, bool(inside or ends_inside)))
            # What a `let` binds to a literal, and a constant, have a type.
            value, item = None, False
            if match := _RUST_VEC_LET_RE.search(code):
                value = "Vec<>()"
            elif match := _RUST_STRING_LET_RE.search(raw):
                # The literal is all of the value where `;` ends it.
                if not inside and code[match.end() - 1 :].strip() == ";":
                    value = "[u8]()" if match["bytes"] else "str()"
            elif match := _RUST_CONST_RE.search(code):
                value, item = f"{_RUST_BORROW_RE.sub('', match['type'])}()", True
            elif match := _RUST_INDEX_LET_RE.search(code):
                # `let x = xs[i]` gives `x` the element of `xs`'s type.
                held = [
                    declared
                    for at, declared, _ in self.declared.get(match["of"], ())
                    if at <= lineno and declared
                ]
                if held and held[-1].startswith("#"):
                    # `xs` holds what a call returned: its element.
                    value = f"{held[-1]}.*"
                elif held:
                    element = re.match(
                        r"(?:&\s*(?:'\w+\s*)?(?:mut\s+)?)?"
                        r"(?:\[\s*([^;\]]+)\]|"
                        r"[\w:]+(?:<([\w:&,' ]+)>)?)",
                        held[-1],
                    )
                    if element and (element[1] or element[2]):
                        # `map[k]` gives the value, `xs[i]` the element.
                        taken = _bracketed_parts(element[1] or element[2])
                        value = f"{taken[-1].strip()}()"
            if value:
                self.declared.setdefault(match["name"], []).append(
                    (lineno, value, item)
                )
            inside = ends_inside

        def add_derived(type_id, index):
            """Record the standard traits that a `#[derive(...)]` above line
            ``index`` implements for ``type_id``."""
            first = index
            while first and codes[first - 1][0].rstrip()[-1:] not in (";", "{", "}"):
                first -= 1
            attributes = " ".join(code for code, _ in codes[first:index])
            scope = _stdlib.load("rust")["scope"]
            self.builder.derived[type_id] = [
                scope[trait]
                for derives in _RUST_DERIVE_RE.findall(attributes)
                for trait in re.findall(r"(\w+)\s*(?:,|$)", derives)
                if trait in scope
            ]

        def impl_at(index, line):
            """What ``_rust_impl`` reads in the header of the `impl` block
            that starts with ``line``, line ``index``, and the bounds the
            block gives its type parameters, its own or by `where`; four
            None where none does. The header runs to its `{`, on that line
            or a later one."""
            if not _RUST_IMPL_RE.match(line):
                return None, None, None, None
            header = []
            for code, _ in codes[index:]:
                header.append(code)
                if "{" in code or ";" in code:
                    break
            text = _RUST_IMPL_RE.sub("", " ".join(header), count=1)
            bounds = _rust_type_parameters(text)
            where = text.partition(" where ")[2]
            for name, bound in re.findall(r"\b(\w+)\s*:\s*([\w:]+)", where):
                bounds[name] = bounds.get(name) or bound
            return (*_rust_impl(text), {k: v for k, v in bounds.items() if v})

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
                    if popped_id == f"{file_id}::{current_impl_target}":
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
                # A signature holds no call: the line's calls start in its body.
                body = line.find("{", match.end())
                call_scan_start = body + 1 if body >= 0 else len(line)
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
                if impl_bounds and node_type == "Method":
                    # What its `impl` block bounds the type's parameters by
                    node_kwargs["impl_bounds"] = impl_bounds

                self.builder.add_node(func_id, node_type, **node_kwargs)
                self.builder.add_edge(parent_id, func_id, "DEFINES")
                for parameter in _RUST_PARAMETER_RE.findall(full_sig):
                    self.add_local(func_id, parameter)

                if opens_block(lines[j]):
                    current_context.append(func_id)
                    context_indent.append(indent)
                    fn_ids.add(func_id)
                elif j == i and "{" in line:
                    # A body on the signature's line: its calls are the function's.
                    parent_id = func_id
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
                    **_generic_metadata(_rust_type_parameters(line[match.end() :])),
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                add_derived(type_id, i)
                if held := _rust_held(line[match.end() :]):
                    self.builder.fields[type_id] = held
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
                    **_generic_metadata(_rust_type_parameters(line[match.end() :])),
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                add_derived(type_id, i)
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
                    if held := _rust_held(line[match.end(1) :]):
                        self.builder.fields[variant_id] = held

            # 3. Impl
            elif (named := impl_at(i, line))[1]:
                # The trait is tracked for method metadata.
                trait, current_impl_target, bounds, impl_bounds = named
                current_trait_impl = _last_segment(trait)
                struct_id = f"{file_id}::{current_impl_target}"
                impl_targets.setdefault(
                    struct_id, (current_impl_target, lineno, bounds)
                )

                if trait:
                    # A path says where the trait is from.
                    self.builder.add_edge(
                        struct_id,
                        f"REF:{current_trait_impl}",
                        "INHERITS",
                        {"origin": trait} if "::" in trait else None,
                    )

                if opens_block(line) or "{" not in line:
                    # The body opens here or, after a `where` clause, below.
                    current_context.append(struct_id)
                    context_indent.append(indent)
                else:  # `impl Marker for Foo {}`
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

            # 6. Calls
            if parent_id != file_id:
                code = codes[lineno - 1][0]
                self.begin_line(code, lineno, parent_id, lines[lineno - 1])
                self._bind_arm(code, i, lines)
                # The closures on the line whose parameters are still to be read
                self._closures = []
                if "|" in code:
                    self._closures = [
                        (found.start("params") - 1, found["params"], found.end())
                        for found in _RUST_CLOSURE_RE.finditer(code)
                    ]
                self._bodies = []  # (call, argument, columns of the body) of each
                owner = parent_id.split("::")[-2] if parent_id.count("::") > 1 else None
                # The block a `for` or an `if let` opens, where its body goes
                # on on the line: the names the head binds hold there.
                head = ("for " in code and _RUST_FOR_RE.search(code)) or (
                    "let " in code and _RUST_IF_LET_RE.search(code)
                )
                body = head.end() if head else len(code)
                for call_match in re_call.finditer(code, call_scan_start):
                    # In `0..items.len()` the callee follows the range's `..`.
                    callee = call_match.group(1).rpartition("..")[2].lstrip(".")
                    start = call_match.end(1) - len(callee)
                    # `Boxed::<Dog>::new` is a function of the type written
                    # with its arguments.
                    typed = re.fullmatch(r"(\w+)::<(.+)>::(\w+)", callee)
                    typed = typed and f"{typed[1]}<{typed[2]}>.{typed[3]}"
                    # `parse::<T>` calls `parse`; the type arguments are the
                    # call site's, not part of the callee's name.
                    callee = _strip_generics(callee)
                    if start >= body:
                        self._bind_patterns(code, i, lines, fn_start, owner)
                        body = len(code) + 1
                    # `pub(crate)` is a visibility, not a call.
                    if callee and callee != "pub":
                        self._bind_closures(code, i, lines, call_match.start(1))
                        self.add_call(
                            parent_id,
                            callee,
                            lineno,
                            fn_start,
                            start,
                            call_match.end() - 1,
                            self._generic_path_origin(callee) or typed,
                        )
                self._bind_closures(code, i, lines, len(code))
                self._name_results(code, lineno, fn_start, owner)
                if body <= len(code):
                    self._bind_patterns(code, i, lines, fn_start, owner)
            i += 1

        # A field's type, for each struct of that name the file declares.
        for (owner, field), held in self.fields.items():
            for node in self.builder.nodes[self._first_node :]:
                if node["id"].endswith(f"::{owner}") and len(held) == 1:
                    fields = self.builder.fields.setdefault(node["id"], {})
                    fields.setdefault(field, next(iter(held))[:-2])

        # An impl block's edges are sourced at its type. A type declared in
        # another file has no node here, so give it one.
        for struct_id, (name, impl_lineno, bounds) in impl_targets.items():
            if struct_id not in self.builder.node_registry:
                self.builder.add_node(
                    struct_id,
                    "Class",
                    name=name,
                    kind="impl",
                    lineno=impl_lineno,
                    docstring="",
                    # The traits a type must have for a blanket impl to be its.
                    **({} if bounds is None else {"bounds": bounds}),
                )
                self.builder.add_edge(file_id, struct_id, "DEFINES")

        self.restore_source_lines()

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
            # The parameter list is the first `(` after the type parameters,
            # whose `Fn(...)` bounds type the parameters they name.
            paren_start = full_sig.find("(")
            angle = re.match(
                r"\s*(?:pub(?:\([^)]*\))?\s+)?[\w\s]*?fn\s+\w+\s*<", full_sig
            )
            bounds, generics = {}, ""
            if angle:
                depth = 0
                for at in range(angle.end() - 1, len(full_sig)):
                    depth += (full_sig[at] == "<") - (
                        full_sig[at] == ">" and full_sig[at - 1] != "-"
                    )
                    if not depth:
                        paren_start = full_sig.find("(", at)
                        generics = full_sig[angle.end() : at]
                        break
                bounds = dict(_RUST_FN_BOUND_RE.findall(generics))
            bounds.update(_RUST_FN_BOUND_RE.findall(full_sig.partition(" where ")[2]))
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

            named = {"Self", *re.findall(r"(?:^|,)\s*(?:const\s+)?(\w+)", generics)}
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
                        if type_part in bounds:
                            type_part = f"impl {bounds[type_part]}"
                        # A closure parameter of a type the function's type
                        # parameters or `Self` name is of no known type.
                        type_part = _RUST_FN_TYPE_RE.sub(
                            lambda found: found[0].replace(
                                found[1],
                                ", ".join(
                                    "_"
                                    if named & set(re.findall(r"\w+", part))
                                    else part
                                    for part in _bracketed_parts(found[1] + ")")
                                ),
                            ),
                            type_part,
                        )
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

        lines = source_lines(content)
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


# Type arguments, nested three deep, that name no function type
_TS_PLAIN = r"[^<>();={}]*"
_TS_ARGUMENTS = (
    rf"<{_TS_PLAIN}(?:<{_TS_PLAIN}(?:<{_TS_PLAIN}>{_TS_PLAIN})*>{_TS_PLAIN})*>"
)
_TS_DECLARATIONS = (
    # `name: Type` on a parameter (alone on its line or not), variable or
    # field: a primitive, or a named type with its type arguments, either as
    # an array and whether or not it may be undefined or null instead
    re.compile(
        r"(?:^|[(,]|\b(?:const|let|var|private|public|protected|readonly)\s)\s*"
        r"(?:\.\.\.)?(?P<name>[A-Za-z_$][\w$]*)\??\s*:\s*(?P<type>(?:(?:[a-z_$][\w$]*\.)*"
        rf"[A-Z$][\w$.]*(?:{_TS_ARGUMENTS})?|string|number|boolean|bigint|symbol"
        r"|typeof\s+[\w$]+(?:\.[\w$]+)*)"
        r"(?:\[\])*)(?:\s*\|\s*(?:undefined|null))*(?=\s*(?:[,)=;]|$))"
    ),
    # `name = new Type(`
    re.compile(
        r"\b(?P<name>[A-Za-z_$][\w$]*)\s*=\s*new\s+"
        rf"(?P<type>[A-Za-z_$][\w$.]*(?:{_TS_ARGUMENTS})?)"
    ),
    # `name = value as Type`, where the statement ends with the type and
    # no operator before it binds less tightly than `as`
    re.compile(
        r"\b(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=[^;?|&=]*\bas\s+"
        rf"(?P<type>(?:[a-z_$][\w$]*\.)*[A-Z$][\w$.]*(?:{_TS_ARGUMENTS})?(?:\[\])*)"
        r"\s*;?\s*$"
    ),
    # A type parameter's bound: `<T extends Type`
    re.compile(
        r"[<,]\s*(?P<param>[A-Z]\w*)\s+extends\s+(?P<type>[A-Z][\w$.]*)\b(?![<(])"
    ),
)


# A number written with a `.` at its end (`1.`), whose `.` is no member's:
# its decimal point, after decimal digits that no `.`, prefix, exponent or
# suffix comes with (`1.5.`, `0x1.`, `3e5.` and `1n.` are read through a
# member), in JavaScript and Rust alike
_NUMBER_DOT = r"(?<![\w$.])\d[\d_]*\."
_NUMBER_DOT_END_RE = re.compile(_NUMBER_DOT + r"\Z")
# Where JavaScript declares names: a variable or what a pattern takes apart,
# a function, a type, and what a `catch` is handed. A member's name, written
# after a `.` or a `#`, is read as one and binds nothing.
_JS_DECLARED_RE = re.compile(
    _NUMBER_DOT + r"|[.#]\s*[\w$]+"
    r"|\b(?:const|let|var|function\*?|class|interface|type|enum|namespace)\s+"
    r"([\w$]+|\{[^=;]*?\}|\[[^=;]*?\])|\bcatch\s*\(\s*([\w$]+)"
)
# A parameter list, as a function's is followed by its body or its `=>`,
# and the one parameter written without a list
_JS_PARAMETERS_RE = re.compile(
    r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*(?::[^=;{()]*)?(?:=>|\{)"
    r"|(?<![\w$])([\w$]+)\s*=>"
)
# The statements whose parenthesised head is followed by a block
_JS_BLOCK_RE = re.compile(r"\b(?:if|for|while|switch|catch|with)\s*$")


def _run_start(code, at, marks=""):
    """Where the run of whitespace and of ``marks`` that ends at column
    ``at`` of ``code`` begins."""
    at = min(at, len(code))
    while at and (code[at - 1].isspace() or code[at - 1] in marks):
        at -= 1
    return at


def _js_member_name(lines, row, column):
    """Whether the word at column ``column`` of line ``row`` of ``lines`` is
    a member's name: written after a `.` or a `#`, on its line or one
    above, where the `.` ends no number."""
    line = lines[row]
    end = _run_start(line, column)
    while not end and row > 0:
        row -= 1
        line = lines[row]
        end = _run_start(line, len(line))
    digits = end - 1
    while digits > 0 and (line[digits - 1].isdecimal() or line[digits - 1] == "_"):
        digits -= 1
    return line[end - 1 : end] in (".", "#") and not _NUMBER_DOT_END_RE.match(
        line, digits, end
    )


def _js_bound(content):
    """The names the JavaScript source ``content`` declares or takes as
    parameters, in any of its scopes."""
    held = ["".join(found) for found in _JS_DECLARED_RE.findall(content)]
    for found in _JS_PARAMETERS_RE.finditer(content):
        if not _JS_BLOCK_RE.search(content, max(0, found.start() - 8), found.start()):
            # A parameter's name or pattern stands before its type and default.
            parts = _bracketed_parts((found[1] or "") + ")")
            held += [
                name
                for p in parts
                for name in _bracketed_parts(re.sub("[:=]", ",", p) + ")")[:1]
            ]
            held.append(found[2] or "")
    return set(re.findall(r"[A-Za-z_$][\w$]*", " ".join(held)))


# A `|` between the alternatives of a union, outside any bracket
_TS_UNION_RE = re.compile(r"\s*\|\s*(?![^<>\[\](){}]*[>\])}])")


# A string literal type, as written
_TS_STRING_RE = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|`[^`]*`")


def _ts_unquoted(text):
    """``text`` with the text of each string literal in it turned to `_`,
    its quotes and columns kept."""
    return _TS_STRING_RE.sub(
        lambda found: found[0][0] + "_" * (len(found[0]) - 2) + found[0][-1], text
    )


@cache
def _ts_type_parts(text):
    """``(name, type arguments, {})`` of a TypeScript type as written,
    whether or not it may be undefined or null instead. An array is an
    `Array`, `this` is `Self`, what a type predicate stands for is a
    `boolean` and a string literal type is a `string`. A union of two types,
    a tuple, an object and a function type have no name."""
    # An origin path writes the `.` of a qualified type as `::`. A string
    # literal's text holds no mark.
    written = text.replace("::", ".").strip()
    marks = list(_TS_UNION_RE.finditer(_ts_unquoted(written)))
    starts = [0] + [mark.end() for mark in marks]
    ends = [mark.start() for mark in marks] + [len(written)]
    options = {written[start:end] for start, end in zip(starts, ends)}
    options -= {"undefined", "null"}
    text = options.pop() if len(options) == 1 else ""
    if text.endswith("[]"):
        return "Array", (text[:-2].removeprefix("readonly "),), {}
    if re.search(r"\bis\s", text):
        return "boolean", (), {}
    if _TS_STRING_RE.fullmatch(text):
        return "string", (), {}
    name = re.match(r"[\w$.]*", text)[0]
    arguments = ()
    if text.startswith("<", len(name)) and text.endswith(">"):
        arguments = tuple(_bracketed_parts(text[len(name) + 1 :]))
    elif len(name) < len(text):
        return "", (), {}
    return "Self" if name == "this" else name, arguments, {}


@cache
def _ts_compound(text):
    """``("?", branches)`` of a TypeScript type that is a conditional type,
    ``("&", operands)`` of an intersection and ``("|", alternatives)`` of
    a union — None for any other. A union's `null` and `undefined`
    alternatives are dropped: a value that may be one reads as the rest."""
    depth, marks = 0, []
    for column, char in enumerate(_ts_unquoted(text)):
        if char in "<([{":
            depth += 1
        elif char in ")]}" or (char == ">" and text[column - 1] != "="):
            depth -= 1
        elif (
            not depth
            and char in "?:&|"
            and "::" not in text[max(column - 1, 0) : column + 2]
        ):
            # A `::` is the `.` of a name an origin path writes, which no
            # TypeScript type has.
            marks.append((column, char))
    asked = [column for column, char in marks if char == "?"]
    if asked:
        # Its `:` is the one that no `?` after its own is still open at.
        pending = 0
        for column, char in marks:
            pending += (column >= asked[0]) * ((char == "?") - (char == ":"))
            if char == ":" and column > asked[0] and not pending:
                branches = text[asked[0] + 1 : column], text[column + 1 :]
                nested = [_ts_compound(branch.strip()) for branch in branches]
                return "?", [
                    part
                    for branch, found in zip(branches, nested)
                    for part in (found[1] if found and found[0] == "?" else [branch])
                ]
        return None
    if {char for _, char in marks} == {"&"}:
        cuts = [-1, *(column for column, _ in marks), len(text)]
        return "&", [text[cuts[n] + 1 : cuts[n + 1]] for n in range(len(marks) + 1)]
    if {char for _, char in marks} in ({"|"}, {"|", "&"}):
        cuts = [-1, *(column for column, char in marks if char == "|"), len(text)]
        alternatives = [text[cuts[n] + 1 : cuts[n + 1]] for n in range(len(cuts) - 1)]
        alternatives = [
            one.strip()
            for one in alternatives
            if one.strip() not in ("undefined", "null", "void")
        ]
        if len(alternatives) > 1:
            return "|", alternatives
    return None


# The name of a class member, `#` before a private one
_TS_MEMBER_NAME = r"#?[A-Za-z0-9_$]+"
# A member's name as the source writes it: a plain one, or a string literal
# with its quotes, a name no plain method has. A call's name is always a
# plain one: a quoted string is a value.
_TS_WRITTEN_MEMBER_NAME = r"#?(?:[A-Za-z0-9_$]+|\"[^\"\n]*\"|'[^'\n]*')"
_TS_VARIABLE_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)")
# The name of a property in an object literal: `name: value`, quoted or not
_TS_PROPERTY_RE = re.compile(rf"^\s*({_TS_WRITTEN_MEMBER_NAME})(?=\s*:)")


def _ts_unquote(name: str) -> str:
    """A member's name as the source names it: a string-literal key's
    content, its quotes syntax only — `"~validate"` and `~validate` are
    one name, as `{"m": f}` and `{m: f}` are one property."""
    if len(name) > 1 and name[0] == name[-1] and name[0] in "\"'":
        return name[1:-1]
    return name


def _arrow_function(statement: str) -> tuple[str, str] | None:
    """Read `const name = (params) => ...`, with or without a type
    annotation on ``name``, `const name = function (params) ...`, or either
    function as the value of a property: `name: (params) => ...`. Returns
    the name and the function from its parameters on, else None."""
    match = _TS_VARIABLE_RE.match(statement) or _TS_PROPERTY_RE.match(statement)
    if not match:
        return None
    # The initialiser follows the first `=` outside brackets that is not
    # the `=>` of a function type, the value of a property its `:`.
    assign = ":" if match.re is _TS_PROPERTY_RE else "="
    depth = 0
    for column in range(match.end(), len(statement)):
        char = statement[column]
        if char in "([{<":
            depth += 1
        elif char in ")]}>" and statement[column - 1] != "=":
            depth -= 1
        elif (
            char == assign and depth == 0 and statement[column + 1 : column + 2] != ">"
        ):
            break
    else:
        return None
    value = _function_value(statement[column + 1 :])
    return (_ts_unquote(match.group(1)), value) if value else None


_TS_OBJECT_PROPERTY_RE = re.compile(r"\s*(?:readonly\s+)?([\w$]+)\??\s*:\s*\{")
_TS_FUNCTION_PROPERTY_RE = re.compile(
    r"\s*(?:readonly\s+)?([\w$]+)\??\s*:\s*(<[^()=]*>)?\s*\("
)


def _function_value(text: str) -> str | None:
    """The function ``text``, the value given a name, is written as, from
    its parameters on, where it is an arrow function or a `function`
    expression; else None."""
    # `((params) => ...) as Type` holds the function in parentheses.
    value = re.sub(r"^\s*(?:\((?=\())?(?:async\b\s*)?", "", text)
    if value.startswith("<"):
        # Type parameters before the parameters: `<T,>(x: T) => ...`; the
        # `>` of a `=>` in one of their defaults closes nothing.
        depth = 0
        for column, char in enumerate(value):
            if char == "<":
                depth += 1
            elif char == ">" and value[column - 1] != "=":
                depth -= 1
                if depth == 0:
                    value = value[column + 1 :].lstrip()
                    break
    if re.match(r"function\b|[A-Za-z_$][\w$]*\s*=>", value):
        return value
    if not value.startswith("("):
        return None
    depth = 0
    for column, char in enumerate(value):
        depth += (char == "(") - (char == ")")
        if depth == 0:
            # `=>` follows the parameters, after any return type.
            rest = value[column + 1 :].lstrip()
            is_arrow = rest.startswith("=>") or (rest.startswith(":") and "=>" in rest)
            return value if is_arrow else None
    return None


def _raw_pointer(written: str, pointer: str | None) -> str:
    """The C++ type ``written`` with a `*` where ``pointer``, the `*` and
    `&` of its declarator, makes a raw pointer to a template's instance,
    whose `->` reaches the instance's own members, not what it wraps."""
    return f"{written}*" if "<" in written and "*" in (pointer or "") else written


def _bounded_parameters(text: str) -> dict[str, str]:
    """``{name: bound}`` of the type parameters ``text`` opens with,
    `<const T extends Bound = Default, U>`, the bound "" for one of none."""
    text = text.lstrip()
    found = {}
    for part in _bracketed_parts(text[1:]) if text.startswith("<") else ():
        name, _, bound = re.split(r"\s*=(?!>)", part)[0].partition(" extends ")
        if name.split():
            found[name.split()[-1]] = bound.strip()
    return found


def _rust_type_parameters(text: str) -> dict[str, str]:
    """``{name: bound}`` of the type parameters ``text`` opens with,
    `<'a, T: 'a + Bound + Other = Default, const N: usize>`: no lifetime,
    and of a bound its first trait."""
    text = text.lstrip()
    found = {}
    for part in _bracketed_parts(text[1:]) if text.startswith("<") else ():
        head, _, bound = part.split("=")[0].partition(":")
        words = head.split()
        if words and not words[-1].startswith("'"):
            traits = [b.strip() for b in bound.split("+")]
            traits = [b for b in traits if b and b[0] not in "'?"]
            found[words[-1]] = traits[0] if traits and words[0] != "const" else ""
    return found


def _generic_metadata(params: dict[str, str]) -> dict:
    """The metadata of a type that declares the type parameters ``params``:
    their names in order and the bounds of those that have one."""
    if not params:
        return {}
    bounds = {name: bound for name, bound in params.items() if bound}
    return {"generics": tuple(params), **({"type_bounds": bounds} if bounds else {})}


def _ts_type_member(lines: list[str], index: int) -> tuple[str, str] | None:
    """``(name, type)`` of a property that line ``index`` of ``lines``, code
    lines, declares in an interface or an object type: a function type as
    ``_ts_function_property`` writes it, or an object type whole, over the
    lines it goes on to. None for any other line."""
    if written := _ts_function_property(lines[index]):
        return written
    found = _TS_OBJECT_PROPERTY_RE.match(lines[index])
    if not found:
        return None
    text = "\n".join(lines[index : index + 40])
    close = _block_close(text, found.end() - 1)
    return close and (found[1], " ".join(text[found.end() - 1 : close + 1].split()))


def _ts_function_property(code: str) -> tuple[str, str] | None:
    """``(name, signature)`` of a property ``code`` declares with a
    function type, `name?: <T>(a: A) => R;`, the signature written as a
    method's, `name<T>(a: A): R`; None for any other line."""
    found = _TS_FUNCTION_PROPERTY_RE.match(code)
    if not found:
        return None
    depth = 0
    for column in range(found.end() - 1, len(code)):
        depth += (code[column] == "(") - (code[column] == ")")
        if not depth:
            returned = re.match(r"\s*=>\s*(.+?)\s*[;,]?\s*$", code[column + 1 :])
            params = code[found.end() - 1 : column + 1]
            if returned:
                return found[1], f"{found[1]}{found[2] or ''}{params}: {returned[1]}"
            return None
    return None


def _class_property(
    code: str, start: int = 0, text: str | None = None
) -> tuple[str, int, int] | None:
    """A property declared at column ``start`` of ``code`` with a value:
    its name, the column its name ends at, and the column its value starts
    at, past an annotation of any shape (`t: (a: A) => B = ...`). Where
    ``code`` has its literals blanked, ``text`` is the line with them, which
    a quoted name is read from."""
    named = _TS_CLASS_PROPERTY_RE.match(code, start)
    if not named:
        return None
    depth = 0
    for column in range(named.end(), len(code)):
        char = code[column]
        if char in "([{<":
            depth += 1
        elif char in ")]}>" and code[column - 1] != "=":
            depth -= 1
        elif char == ";" and depth == 0:
            return None
        elif (
            char == "="
            and depth == 0
            and code[column + 1 : column + 2] not in ("=", ">")
            and code[column - 1] not in "!<>="
        ):
            value = (
                column + 1 + len(code[column + 1 :]) - len(code[column + 1 :].lstrip())
            )
            written = (text or code)[named.start(1) : named.end(1)]
            return _ts_unquote(written), named.end(1), value
        if depth < 0:
            return None
    return None


def _member_written(
    line: str, in_class: bool, method_re, value: str | None = None
) -> tuple[str, int, int, bool] | None:
    """The member ``line`` opens: its name, the columns its name starts and
    ends at, and whether a value is assigned to it. `name(` and `name = (`
    as ``method_re`` reads them, and in a class a property holding a
    function whatever its annotation, on its own line or over the lines of
    its value, ``value`` being that value's text."""
    if match := method_re.search(line):
        return (
            _ts_unquote(match.group(1)),
            match.start(1),
            match.end(1),
            bool(match.group(2)),
        )
    if (
        in_class
        and (prop := _class_property(line))
        and _function_value(value if value is not None else line[prop[2] :])
    ):
        return prop[0], prop[1] - len(prop[0]), prop[1], True
    return None


# What a line ends with, or the line after it starts with, where the
# expression goes on from the one to the other
_JS_CONTINUES_EXPRESSION = tuple(".?:+-*/%&|^=<>")


def _statement_end(lines: list[str], index: int, column: int = 0) -> int:
    """The index of the last line of the statement that starts at column
    ``column`` of line ``index``: the first line at which its brackets are
    closed and that
    neither ends with an operator or an arrow's `=>`, nor is followed by a
    line going on with it; a `,` there ends the property or declarator.
    ``lines`` are code lines with the text of their literals blanked."""
    depth = 0
    for row in range(index, len(lines)):
        code = lines[row][column if row == index else 0 :].strip()
        for char in code:
            depth += (char in "([{") - (char in ")]}")
            if depth < 0:
                return row  # the line closes what holds the statement
        if depth > 0 or not code:
            continue
        following = lines[row + 1].lstrip() if row + 1 < len(lines) else ""
        if not (
            code.endswith(_JS_CONTINUES_EXPRESSION + ("=>",))
            or following.startswith(_JS_CONTINUES_EXPRESSION)
        ):
            return row
    return len(lines) - 1


def _own_line_span(code: str, name_end: int) -> tuple[int, int] | None:
    """The columns of ``code`` holding the parameters and the first line of
    the body of a definition whose name ends at ``name_end``: code that
    runs when the definition is called, its parameter defaults with it
    (ECMA-262 2026 10.2.11 FunctionDeclarationInstantiation), so a call on
    its own line is its, not its enclosing scope's.
    """
    paren = code.find("(", name_end)
    arrow = code.find("=>", name_end)
    if arrow >= 0 and (paren < 0 or arrow < paren):
        # One parameter, unparenthesised (`x => g(x)`): the body is the
        # expression after the arrow, to the `,` or `;` that ends it.
        depth = 0
        for body in range(arrow + 2, len(code)):
            depth += (code[body] in "([{") - (code[body] in ")]}")
            if depth == 0 and code[body] in ",;":
                return (arrow, body)
        return (arrow, len(code))
    if paren < 0:
        return None
    depth = 0
    close = -1
    for col in range(paren, len(code)):
        depth += (code[col] == "(") - (code[col] == ")")
        if depth == 0:
            close = col
            break
    if close < 0:
        # The parameters span lines; the rest of this line is theirs.
        return (paren, len(code))
    # The body follows, past any return-type annotation.
    for offset, char in enumerate(code[close + 1 :]):
        if char == "{":
            # A block: where its braces balance on this line, else the
            # first line of a block that spans lines.
            body = _block_close(code, close + 1 + offset)
            return (paren, len(code) if body is None else body + 1)
        if char == "=" and code[close + 2 + offset : close + 3 + offset] == ">":
            # An expression: to the `,` or `;` that ends it, at bracket depth.
            depth = 0
            for body in range(close + 3 + offset, len(code)):
                char = code[body]
                depth += (char in "([{") - (char in ")]}")
                if depth == 0 and char in ",;":
                    return (paren, body)
            return (paren, len(code))
    return None


# The name, annotation and `=` of a declarator after a `,`
_TS_DECLARATOR_RE = re.compile(r"\s*([A-Za-z_$][\w$]*)\s*(?::[^=,;]*)?=(?![=>])\s*")


def _value_end(code: str, start: int) -> int:
    """The column at which the declarator's value starting at column
    ``start`` of ``code`` ends: outside brackets, the `;` after it or the `,`
    before another declarator, else the end of the line."""
    depth = 0
    for column in range(start, len(code)):
        depth += (code[column] in "([{") - (code[column] in ")]}")
        if depth < 0 or (
            depth == 0
            and (
                code[column] == ";"
                or (
                    code[column] == ","
                    and (
                        not code[column + 1 :].strip()
                        or _TS_DECLARATOR_RE.match(code, column + 1)
                    )
                )
            )
        ):
            return column
    return len(code)


def _passed_objects(
    code: str, brackets: list[bool], before: str, calls: set[int]
) -> tuple[list[int], str]:
    """The columns at which the object literals open that ``code`` passes
    to a call and leaves open: each `{` that starts an argument, after the
    `(` of a call or the `,` after another argument. ``brackets`` are the
    brackets open where the line starts, innermost last, True for the `(`
    of a call, and are left as the line leaves them; ``calls`` are the
    columns of the `(` of the calls on the line, and ``before`` is the
    last character of code above it. Returns the columns and the last
    character of code read."""
    if code[:1].strip() and code[0] not in ")}]":
        # An unindented line starts a top-level item, unless it opens
        # with what closes the one before.
        brackets.clear()
    opened = {}  # how deep each such object is, while open -> its column
    for column, char in enumerate(code):
        if char in "([{":
            if char == "{" and brackets[-1:] == [True] and before in ("(", ","):
                opened[len(brackets)] = column
            brackets.append(char == "(" and column in calls)
        elif char in ")]}" and brackets:
            brackets.pop()
            opened.pop(len(brackets), None)
        if not char.isspace():
            before = char
    return sorted(opened.values()), before


def _declarators(
    lines: list[str], index: int, after: int
) -> list[tuple[int, str, int, int]]:
    """Each declarator after the first of the `const`, `let` or `var`
    statement whose first binding, a name or a destructuring pattern, ends
    at column ``after`` of line ``index``: `b` in `const a = f, b = () =>
    g();` or on a line the statement goes on to, as the index of its line,
    its name, the column its name ends at and the column its value starts
    at. ``lines`` are the file's lines with the text of their literals
    blanked."""
    declarators = []
    depth = 0
    in_value = False  # past the first declarator's `=`
    row, column = index, after
    while row < len(lines):
        code = lines[row]
        while column < len(code):
            char = code[column]
            if not in_value:
                # The annotation's `<>` hold commas of their own.
                depth += (char in "([{<") - (char in ")]}>" and code[column - 1] != "=")
                if (
                    char == "="
                    and depth == 0
                    and code[column + 1 : column + 2] not in "=>"
                ):
                    in_value = True
            else:
                depth += (char in "([{") - (char in ")]}")
                if depth < 0 or (depth == 0 and char == ";"):
                    return declarators
                if depth == 0 and char == ",":
                    rest, at, offset = code[column + 1 :], row, column + 1
                    if not rest.strip() and row + 1 < len(lines):
                        rest, at, offset = lines[row + 1], row + 1, 0
                    named = _TS_DECLARATOR_RE.match(rest)
                    if named:
                        declarators.append(
                            (at, named[1], offset + named.end(1), offset + named.end())
                        )
            column += 1
        # The statement goes on where a bracket is open or the line ends
        # with the `,` before another declarator.
        if not (depth > 0 or code.rstrip().endswith(",")):
            return declarators
        row, column = row + 1, 0
    return declarators


def _class_body_open(code: str, start: int) -> int | None:
    """The column of the `{` that opens the body of the class whose name
    ends at column ``start`` of ``code``, where it is on that line: the
    first outside the brackets and type arguments of its heritage."""
    depth = 0
    for column in range(start, len(code)):
        char = code[column]
        if char == "{" and depth == 0:
            return column
        depth += (char in "([<") - (char in ")]>")
    return None


def _member_filed(found: re.Match, text: str) -> str:
    """The name the member ``found``, a match of ``_TS_LINE_MEMBER_RE``, is
    filed under: the one ``text``, the line with its literals, writes
    there, after `get` or `set` for an accessor."""
    name = _ts_unquote(text[found.start(2) : found.end(2)])
    return " ".join(filter(None, (found.group(1), name)))


def _members_on_line(
    code: str, text: str, start: int
) -> tuple[list[tuple[str, int, str]], list[tuple[int, int]], int | None]:
    """The methods and accessors written on the line of a class, in the body
    whose `{` is at column ``start`` of ``code``, and the properties holding
    an arrow function, as on a later line: each as the name it is filed
    under (`get size` for an accessor), the column its name ends at, and its
    text. Also the columns of the value of each other field, and the column
    at which a member or field starts that the line leaves open, None where
    there is none. ``code`` is a line with the text of its literals
    blanked, ``text`` that line with it."""
    members = []
    fields = []
    column = start + 1
    while column < len(code):
        found = _TS_LINE_MEMBER_RE.match(code, column)
        end = None
        depth = 0
        for at in range(found.end() - 1 if found else column, len(code)):
            depth += (code[at] in "([{") - (code[at] in ")]}")
            if depth < 0:
                if found:
                    return members, fields, None
                end = at  # a field runs to the class's own `}`
                break
            if found and depth == 0 and code[at] == ")":
                # The parameters closed; the member ends with its body,
                # unless a `;` ends a body-less overload first.
                semi = brace = -1
                inner = 0
                for col in range(at + 1, len(code)):
                    ch = code[col]
                    if inner == 0 and ch == ";":
                        semi = col
                        break
                    if inner == 0 and ch == "{":
                        brace = col
                        break
                    inner += (ch in "([{") - (ch in ")]}")
                    if inner < 0:
                        break
                if semi >= 0:
                    end = semi + 1
                    break
                close = _block_close(code, brace) if brace >= 0 else None
                if close is None:
                    break
                end = close + 1
                break
            if not found and depth == 0 and code[at] in ";}":
                end = at + 1
                break
        if end is None:
            return members, fields, column if code[column:].strip() else None
        if end <= column:
            return members, fields, None
        if found:
            members.append(
                (_member_filed(found, text), found.end(2), code[column:end].strip())
            )
        elif prop := _class_property(code[:end], column, text):
            if _function_value(code[prop[2] : end]):
                members.append((prop[0], prop[1], code[prop[2] : end].strip()))
            else:
                fields.append((prop[2], end))
        column = end
    return members, fields, None


def _member_close_column(code: str) -> int | None:
    """The column of the `}` that closes the member whose next member is
    written on the same line, where ``code`` has it: the line's first `}`
    followed by a member declaration. None otherwise."""
    stripped = code.lstrip()
    if not stripped.startswith("}"):
        return None
    close = len(code) - len(stripped)
    rest = code[close + 1 :].lstrip()
    if not rest:
        return None
    at = close + 1 + len(code[close + 1 :]) - len(rest)
    # A keyword — `else`, `catch`, `for (`, `default:` — continues or
    # starts a statement; it names no member.
    if found := _TS_LINE_MEMBER_RE.match(code, at):
        name = found.group(2)
    elif prop := _class_property(code, at):
        name = prop[0]
    else:
        return None
    return None if name in _JSTS_CONTROL_KEYWORDS else close


def _closes_first(code: str) -> bool:
    """Whether the `}` of ``code`` that close blocks opened on earlier lines
    lead it: there is one, and before the last of them the line has nothing
    but closing brackets and separators."""
    depth = lowest = 0
    last = None
    for column, char in enumerate(code):
        depth += (char == "{") - (char == "}")
        if depth < lowest:
            lowest, last = depth, column
    return last is not None and not code[:last].strip("})];, \t")


def _braces_open(
    lines: list[str], first: int, start: int, index: int, column: int
) -> int:
    """How many of the `{` written from column ``start`` of line ``first``
    on are open past column ``column`` of the later line ``index``.
    ``lines`` are code lines with the text of their literals blanked."""
    written = "".join(
        (lines[first][start:], *lines[first + 1 : index], lines[index][: column + 1])
    )
    return written.count("{") - written.count("}")


def _block_close(code: str, opened: int) -> int | None:
    """The column of the `}` that closes the `{` at column ``opened`` of
    ``code``, where it is on that line."""
    depth = 0
    for column in range(opened, len(code)):
        depth += (code[column] == "{") - (code[column] == "}")
        if depth == 0:
            return column
    return None


def _definition_time_region(code: str) -> tuple[int, int] | None:
    """The columns of the decorators and computed keys that lead ``code``,
    each evaluated where the class or object is defined, not in the member
    they lead: a computed key by ECMA-262 2026 15.7.14
    ClassDefinitionEvaluation and 13.2.5.6 PropertyDefinitionEvaluation, a
    decorator, which ECMA-262 does not define, by TypeScript's decorator
    evaluation.
    """
    start = len(code) - len(code.lstrip())
    end = start
    while code[end : end + 1] in ("@", "["):
        opened = code[end]
        if opened == "@":
            named = re.match(r"@\s*[\w$.]+", code[end:])
            if named is None:
                break
            end += named.end()
            if code[end : end + 1] == "(":
                depth = 0
                for col in range(end, len(code)):
                    depth += (code[col] == "(") - (code[col] == ")")
                    if depth == 0:
                        end = col + 1
                        break
                else:
                    break
        else:
            depth = 0
            for col in range(end, len(code)):
                depth += (code[col] == "[") - (code[col] == "]")
                if depth == 0:
                    end = col + 1
                    break
            else:
                break
        end += len(code[end:]) - len(code[end:].lstrip())
    return (start, end) if end > start else None


def _ts_params(signature: str) -> set[str]:
    """The names that the parameters of ``signature`` bind: those in its
    first parameter list, or the one written alone before its `=>`, and
    those of an arrow function it returns (`a => (b) => ...`)."""
    names = set()
    depth = 0
    at_start = False
    found = list(re.finditer(r"[A-Za-z_$][\w$]*|\.\.\.|=>|\S", signature))
    tokens = [match[0] for match in found]
    arrow = tokens.index("=>") if "=>" in tokens else 0
    returned = ""
    if arrow and "(" not in tokens[:arrow]:
        names.add(tokens[arrow - 1])
        returned = signature[found[arrow].end() :]
    # The parameter list is the first `(` outside the type parameters.
    first, angles = len(tokens), 0
    for number, token in enumerate([] if returned else tokens):
        angles += (token == "<") - (token == ">")
        if token == "(" and not angles:
            first = number
            break
    for match, token in zip(found[first:], tokens[first:]):
        if token in ("(", "<", "[", "{"):
            if at_start and token in "[{":
                names.update(
                    name for name, _ in _js_bindings(signature[match.start() :])
                )
            depth += 1
            at_start = depth == 1
        elif token in (")", ">", "]", "}"):
            depth -= 1
            if depth == 0:
                body = re.match(r"\s*(?::[^=(){]+)?=>", signature[match.end() :])
                returned = body.string[body.end() :] if body else ""
                break
        elif depth == 1 and token == ",":
            at_start = True
        elif at_start and token not in _TS_PARAM_MODIFIERS:
            if token[0].isalpha() or token[0] in "_$":
                names.add(token)
            at_start = False
    if _TS_ARROW_HEAD_RE.match(returned):
        names |= _ts_params(returned)
    return names


# An arrow function from its parameters to its `=>`
_TS_ARROW_HEAD_RE = re.compile(
    r"\s*(?:async\s+)?(?:\([^()]*\)\s*(?::[^=(){]+)?|[\w$]+\s*)=>"
)
# What a line may hold before the name of an object literal's method
_TS_METHOD_MODIFIERS_RE = re.compile(r"\s*(?:async \*|async|static|get|set|\*)?\s*")
# Modifiers a class member is declared behind, in any order
_TS_MEMBER_MODIFIERS = r"(?:(?:private|public|protected|static|abstract|override|declare|readonly|async)\s+)*"
# A method or accessor a class declares: `name(`, `get name(`, `async *name<T>(`
# `const`, `let` or `var` and the first binding of its statement: a name, or
# a destructuring pattern without nested brackets
_TS_DECLARATION_HEAD_RE = re.compile(
    r"^\s*(?:export\s+)?(const|let|var)\s+"
    r"(?:[A-Za-z_$][\w$]*|\{[^{}]*\}|\[[^\[\]]*\])"
)
# A property a class declares, up to its annotation or value: `name`,
# `name?`, `name!`, plain- or quoted-keyed
_TS_CLASS_PROPERTY_RE = re.compile(
    rf"\s*{_TS_MEMBER_MODIFIERS}({_TS_WRITTEN_MEMBER_NAME})\s*[?!]?\s*(?=[:=])"
)
_TS_LINE_MEMBER_RE = re.compile(
    rf"\s*{_TS_MEMBER_MODIFIERS}(?:(get|set)\s+)?(?:\*\s*)?"
    rf"({_TS_WRITTEN_MEMBER_NAME})\s*(?:<[^>]*>)?\s*\("
)
_TS_TYPEOF_RE = re.compile(r"typeof\s+([\w$]+(?:(?:\.|::)[\w$]+)*)")
# `const name = require("m")`, `const { a, b: c } = await import("m")`, a
# `/* */` comment before the module too
_TS_REQUIRE_RE = re.compile(
    r"""\b(?:const|let|var|import)\s+([\w$]+|\{[^}]*\})\s*=\s*(?:await\s+)?"""
    r"""(require|import)\(\s*(?:/\*(?:[^*]|\*(?!/))*\*/\s*)*['"]([^'"]+)['"]\s*\)"""
)
_TS_PARAM_MODIFIERS = ("...", "public", "private", "protected", "readonly")
# `export * from "<specifier>"`, `export { <clause> } from "<specifier>"` and,
# with no specifier, `export { <clause> }`
_TS_REEXPORT_RE = re.compile(
    r"""^\s*export\s+(?:type\s+)?"""
    r"""(?:(\*)(?:\s+as\s+([\w$]+))?\s+from\s+|\{([^}]*)\}(?:\s+from\s+)?)"""
    r"""(?:['"]([^'"]+)['"])?""",
    re.MULTILINE,
)
# What a module exports as its default, `export default <name>` or a
# declaration of a name after it, and whole, `export = <name>` or
# `module.exports = <name>`
_TS_DEFAULT_EXPORT_RE = re.compile(
    r"^[ \t]*(?:export[ \t]+(default)[ \t]+(?:"
    r"(?:(?:async[ \t]+)?function\b[ \t]*\*?|(?:abstract[ \t]+)?class\b|interface\b)"
    r"[ \t]*(?!extends\b|implements\b)([\w$]+)|([\w$]+)(?=[ \t]*;?[ \t]*$))"
    r"|(?:export|module\.exports)[ \t]*=[ \t]*([\w$]+)(?=[ \t]*;?[ \t]*$))",
    re.MULTILINE,
)
# `import <clause> from "<specifier>"`, the clause possibly over several lines
_TS_IMPORT_RE = re.compile(
    r"""^\s*import\s++(?:type\s++)?([^;'"]*?)\s+from\s+['"]([^'"]+)['"]""", re.MULTILINE
)


# A '...' literal closed on its line; a lone apostrophe, as in JSX text, is not one.
_JS_SINGLE_QUOTED_RE = re.compile(r"'(?:[^'\\]|\\.)*'")
# A JSX element's opening tag up to its attributes: its name with the type
# arguments TypeScript gives it, or no name for a fragment's `<>`
_JSX_TAG_RE = re.compile(
    r"<(?:[A-Za-z_$][\w$.:-]*(?:\s*<[^<>]*(?:<[^<>]*>[^<>]*)*>)?(?=[\s/>{]|$)|(?=>))"
)
# Type parameters where a value can start, which a `.tsx` file writes as
# `<T,>`, `<T extends U>` or `<T = U>`, and as `<T>` before a parameter list
_TS_TYPE_PARAMETERS_RE = re.compile(
    r"<\s*(?:const\s+)?[\w$]+\s*(?:,|=|extends\b|>\s*\()"
)
# The files JSX can be written in
_JSX_EXTENSIONS = (".tsx", ".js", ".jsx", ".mjs", ".cjs")
# A /.../ literal with its flags, before what can follow a value. One starts
# where a value can: after what no value ends with, on its line or the one
# before.
_JS_REGEX_RE = re.compile(
    r"/(?:[^/\\\[*]|\\.|\[(?:[^\]\\]|\\.)*\])(?:[^/\\\[]|\\.|\[(?:[^\]\\]|\\.)*\])*/[a-z]*(?=\s*(?:[.,;:)\]}|&?/]|$))"
)
# A keyword that expects an operand where it ends the text before a `/`, so
# the `/` opens a regular expression (ECMAScript reads `/` for a literal
# where an operand is expected and for a division where a value just
# ended). `if`/`for`/`while`/`switch` take `(`, not a literal; `async`,
# `get` and `static` are names.
_JS_OPERAND_WORDS = {
    "return",
    "typeof",
    "instanceof",
    "in",
    "of",
    "new",
    "delete",
    "void",
    "case",
    "throw",
    "yield",
    "await",
    "do",
    "else",
    "default",
    "break",
    "continue",
}
# A `break` or `continue` with the label it names on its line, which end a
# statement
_JS_LABEL_RE = re.compile(r"(?<![\w$.])(?:break|continue)[^\S\r\n\u2028\u2029]+[\w$]+$")
# The keyword whose parenthesised head a statement follows
_JS_STATEMENT_HEAD_RE = re.compile(r"(?<![\w$])(?:if|for(?:\s+await)?|while|with)\s*$")


def _js_regex_position(code, before=""):
    """Whether a ``/`` after ``code``, on a line that follows the code
    ``before``, can open a regular expression: true where no value precedes
    it, so at a line's start or after an operator, a bracket that opens, a
    spread's `...`, the `)` of an `if`, `for`, `while` or `with` head, a
    keyword that expects an operand, or the label of a `break` or
    `continue`. False after what ends a value: a word, a member's name, any
    other `)`, `]`, `}`, `.`, a postfix `++`/`--`, and a quote a blanked
    literal leaves."""
    lines, row = [before, code], 1
    end = _run_start(code, len(code))
    if not end:
        row, end = 0, _run_start(before, len(before))
    text = lines[row]
    if not end:
        return True
    if text.endswith(("++", "--"), 0, end):
        return False
    if text.endswith("...", 0, end):
        return True
    start = end
    while start and (text[start - 1].isalnum() or text[start - 1] in "_$"):
        start -= 1
    if start < end:
        if _JS_LABEL_RE.search(text, max(_run_start(text, start) - 8, 0), end):
            return True
        word = text[start:end]
        return word in _JS_OPERAND_WORDS and not _js_member_name(lines, row, start)
    if text[end - 1] == ")":
        depth, rows = 0, range(row, -1, -1)
        for row in rows:
            text = lines[row]
            for at in range(end - 1, -1, -1):
                depth += {")": 1, "(": -1}.get(text[at], 0)
                if not depth:
                    end = _run_start(text, at)
                    if not end and row:
                        row, text = 0, before
                        end = _run_start(text, len(text))
                    # The keyword ends at `end`: `for await` is the longest.
                    start = max(_run_start(text, max(end - 5, 0)) - 3, 0)
                    head = _JS_STATEMENT_HEAD_RE.search(text, start, end)
                    return bool(head) and not _js_member_name(lines, row, head.start())
            end = len(before)
        return False
    return text[end - 1] not in "]}.`'\""


# The end of a blanked /.../, "...", '...' or `...` literal a member is read of
_JS_LITERAL_RECEIVER_RE = re.compile(r"""([/"'`])\s*\.\s*$""")
# The type arguments a call is written with: `name<T, U>(`
_JS_TYPE_ARGUMENTS_RE = re.compile(
    r"(?<=[\w$])<(?:[^<>()]|<(?:[^<>()]|<[^<>()]*>)*>)*>(?=\s*\()"
)
# The `as Type` a declaration casts a value to, which leaves the value as
# it is
_JS_AS_TYPE_RE = re.compile(r"\bas\s+(?:[\w$]+\.)*[\w$]+(?:<[^<>]*>)?(?:\[\])*")
# `const name: Type = `, up to where the value starts
_TS_TYPED_VALUE_RE = re.compile(
    r"\b(?:const|let|var)\s+[\w$]+\s*:\s*(?P<type>[^=;]+?)\s*=(?![=>])\s*"
)
# A `return` that is a statement of its own, up to its value, and the
# `new` that the value may start with
_JS_RETURN_RE = re.compile(
    r"^\s*(?:(?:\}\s*)?else\s+|if\s*\(.*\)\s*|case\b[^:]*:\s*|default\s*:\s*)*"
    r"return\b\s*(?P<new>new\s+)?"
)
# A bracketed group of an object literal's text that holds no other
_JS_NESTED_RE = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}")


def _nested_part(match):
    """A bracketed group of an object literal, as one marker of the length
    it is written: the text around it keeps the positions it has."""
    return "\0" + " " * (len(match[0]) - 1)


# A function without a name of its own whose body is a block that the line
# leaves open
_JS_BLOCK_FUNCTION_RE = re.compile(
    rf"(?:=>|\bfunction\b[^{{}}]*\)|\b[gs]et\s+{_TS_MEMBER_NAME}\s*\([^()]*\))"
    r"\s*(?::[^{}=]+)?\{\s*$"
)
# The `new` a constructed type is named after
_JS_NEW_RE = re.compile(r"\bnew\s+$")
_JS_FUNCTION_HEAD_RE = re.compile(r"\bfunction\s*\*?\s*$")
# What a function expression follows where it is a value
_JS_VALUE_START_RE = re.compile(r"(?:[(,=?:!|&]|\b(?:return|default|void))\s*$")
# A parameter that is passed a function, with that function's parameters
_TS_CALLABLE_RE = re.compile(r"[\w$]+\??\s*:\s*\(?\((?P<params>[^()]*)\)\s*=>")
# A variable whose value is one blanked literal, or an array
_JS_LITERAL_RE = re.compile(
    r"""\b(?:const|let|var)\s+(?P<name>[\w$]+)\s*=\s*"""
    r"""(?P<literal>/\s*|"\s*"|'\s*'|`[^`]*`|\[.*\]|\[$)\s*(?:as\s+const\s*)?;?\s*$"""
)
# The type of a literal, by the character it opens with: any other is a string
_JS_LITERAL_TYPES = {"/": "RegExp", "[": "Array"}
# How a line `_JS_FOR_RE` matches ends, looked for once before it is tried
_JS_HEAD_END = r"\)\s*\{?\s*$"
_JS_HEAD_END_RE = re.compile(_JS_HEAD_END)
# `for (const pattern of value)` and `for (const name in value)`
_JS_FOR_RE = re.compile(
    r"\bfor\s*(?:await\s*)?\(\s*(?:const|let|var)\s+(?P<pattern>.+?)\s+"
    r"(?P<kind>of|in)\s+(?P<value>.+?)" + _JS_HEAD_END
)
# The line a declaration's pattern goes on over lines from, and the last
# line of one, which closes it and gives the value
_JS_PATTERN_OPENS_RE = re.compile(r"\b(?:const|let|var)\s+(?P<open>[\[{])[^\]}=]*$")
_JS_PATTERN_END_RE = re.compile(
    r"\s*[^=]*?(?P<close>[\]}])\s*=(?![=>])\s*(?P<value>.+?);?\s*$"
)
# A declaration that takes a value apart: `const { a, b } = value`
_JS_PATTERN_RE = re.compile(
    r"\b(?:const|let|var)\s+(?P<pattern>[\[{].*[\]}])\s*=(?!=)\s*(?P<value>.+?);?\s*$"
)
# A function passed to a call: its parameters, in a list or alone before the
# `=>`, and where its body starts
_JS_CLOSURE_RE = re.compile(
    r"(?:(?<=[(,=?!|&])|(?<=[=?!|&]\s)|(?<=return\s)|(?<=default\s)|(?<=void\s)"
    r"|(?<=:)(?=\s*(?:async\s+)?function\b)|^)\s*(?:async\s+)?"
    r"(?P<closure>(?:function\s*\*?\s*(?P<named>[\w$]+\s*)?)?"
    r"\((?P<params>[^()]*)\)(?:\s*:[^=(){]+)?\s*(?:=>|\{)|(?P<param>[\w$]+)\s*=>)"
)
# A function written as a property's value: `key: (params) =>`, `key: p =>`
# or `key: function (params) {`
_JS_PROPERTY_CLOSURE_RE = re.compile(
    r"(?:^|[{,])\s*[\w$]+\s*:\s*(?:async\s+)?(?P<closure>(?:function\s*\*?\s*[\w$]*\s*)?"
    r"\((?P<params>[^()]*)\)(?:\s*:[^=(){]+)?\s*(?:=>|\{)|(?P<param>[\w$]+)\s*=>)"
)
# A `for` head up to the declaration it starts with, or a `catch` head
# up to its binding
_JS_FOR_HEAD_RE = re.compile(r"(?<![\w$])(?:for\s*(?:await\s*)?|(catch)\s*)\(\s*$")
# What starts a statement of a keyword's: the keyword, of one that has a
# parenthesised head or of one that has none, or a label; and the clause
# that may go on with one: an `if`'s `else`, a `try`'s `catch` or
# `finally`, a `catch`'s `finally`, a `do`'s `while`
_JS_COMPOUND_RE = re.compile(
    r"(?:(if|for(?:\s+await)?|while|with|switch|catch)|(else|do|try|finally))\b"
    r"|[\w$]+\s*:"
)
_JS_CLAUSE_RE = {
    "if": re.compile(r"else\b"),
    "try": re.compile(r"(?:catch|finally)\b"),
    "catch": re.compile(r"finally\b"),
    "do": re.compile(r"while\b"),
}
# Where a variable's binding starts, a name or a pattern, and the keyword
# that declares it: `const`, `let`, `var`, or a `catch` clause's
_JS_BINDING_HEAD_RE = re.compile(
    r"(?<![\w$])(?:(const|let|var)\s+|catch\s*\(\s*)(?=[\w$\[{])"
)
# The key of a property in an object pattern and the `:` after it, where
# the pattern binds another name than the key: `key: name`, `[computed]:`
_JS_PROPERTY_KEY_RE = re.compile(
    r"""(\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]|"[^"]*"|'[^']*'|[\w$]*)"""
    r"\s*(:?)"
)


def _js_bindings(pattern, steps=()):
    """Yield ``(name, steps)`` for each name the JavaScript pattern binds
    without declaring its type: the steps lead from the value taken apart
    to the part bound, the property's name for `{ name }` and `*` for an
    element of `[first, second]`, and are None for what `...rest` gathers,
    which is of no type the value gives; `&` for a rest parameter, which
    gathers the parameters from its own on."""
    pattern = pattern.strip()
    if pattern[:1] in ("[", "{"):
        for part in _bracketed_parts(pattern[1:]):
            if part.startswith("..."):
                yield from ((name, None) for name, _ in _js_bindings(part[3:]))
            elif pattern[0] == "[":
                yield from _js_bindings(part, (*steps, "*"))
            else:
                key = _JS_PROPERTY_KEY_RE.match(part)
                bound = part[key.end() :] if key[2] else part
                yield from _js_bindings(bound, (*steps, key[1]))
    elif pattern.startswith("..."):
        if re.fullmatch(r"[A-Za-z_$][\w$]*", name := pattern[3:].strip()):
            yield name, (*steps, "&")
    elif re.fullmatch(r"[A-Za-z_$][\w$]*", name := pattern.partition("=")[0].strip()):
        yield name, steps


def _blank_js_strings(
    line: str, stack: list | None = None, keep=False, jsx=False, before=""
) -> str:
    """Return ``line`` with the text of "...", '...', `...` and /.../
    literals and of ``/* */`` comments turned to spaces, and everything a
    ``//`` comment covers cut. The `${...}` expressions of a template
    literal are kept. ``stack`` holds the literals and comments open where
    the line starts and is left holding those open where it ends; a literal
    carried over from an earlier line is blanked together with the quote
    that closes it. ``keep`` leaves the text of the literals that open on
    the line where it is, for a reader of what a literal holds: an import's
    module, a constant's value. With ``jsx``, the text of a JSX element is
    turned to spaces too, and its tags and `{...}` expressions are kept;
    ``before`` is the code the line follows, which no line break ends
    before a `<` or a `/`."""
    out = ""
    # open literals and comments: a quote character, a `${`'s brace depth,
    # `/*` for a comment still open, `<` or `</` for a JSX tag and `>` for
    # an element's text
    stack = [] if stack is None else stack
    carried = len(stack)
    ended = None  # how much of `out` there was where a /.../ literal ended
    # What ``keep`` shows at a column of `out`, which holds the blanked line
    # either way, so that what is code is read alike with ``keep`` and without
    kept_at = {}
    i = 0
    while i < len(line):
        ch = line[i]
        top = stack[-1] if stack else None
        if top == "/*":
            end = line.find("*/", i)
            if end == -1:
                out += " " * (len(line) - i)
                break
            stack.pop()
            # A comment carried over has ended.
            carried = min(carried, len(stack))
            out += " " * (end + 2 - i)
            i = end + 2
            continue
        elif top == ">":
            # An element's text, up to a `{...}` or a tag in it
            if ch == "{":
                stack.append(0)
            elif line.startswith("</", i):
                stack.append("</")
            elif tag := _JSX_TAG_RE.match(line, i):
                stack.append("<")
                out += tag.group()
                i = tag.end()
                continue
            out += ch if ch in "{<" else " "
        elif top in ("<", "</"):
            # A tag: its name, its attributes, their "..." and {...} values
            if ch in "\"'":
                stack.append(ch)
            elif ch == "{":
                stack.append(0)
            elif ch == "/" and line[i + 1 : i + 2] == "/":
                break
            elif ch == "/" and line[i + 1 : i + 2] == "*":
                stack.append("/*")
                out += "  "
                i += 2
                continue
            elif ch == ">" and top == "<":
                stack[-1] = ">"
            elif ch == ">" or line.startswith("/>", i):
                # A closing tag, or a tag that closes itself, ends its element.
                del stack[-2 if top == "</" else -1 :]
                carried = min(carried, len(stack))
                if ch == "/":
                    out += "/"
                    i += 1
                ch = ">"
            out += ch
        elif isinstance(top, str):
            # The text of a literal that opened on this line.
            kept = keep and len(stack) > carried
            if ch == "\\":
                if kept:
                    kept_at[len(out)] = line[i : i + 2]
                out += "  "
                i += 2
                continue
            if top == "`" and line.startswith("${", i):
                stack.append(0)
                out += "${"
                i += 2
                continue
            if ch == top:
                stack.pop()
            if len(stack) < carried:
                # A literal carried over has ended.
                carried = len(stack)
                ch = " "
            if kept:
                kept_at[len(out)] = ch
            out += ch if ch == top else " "
        else:
            if ch in ('"', "`") or (ch == "'" and _JS_SINGLE_QUOTED_RE.match(line, i)):
                stack.append(ch)
            elif (
                jsx
                and ch == "<"
                and (tag := _JSX_TAG_RE.match(line, i))
                and not _TS_TYPE_PARAMETERS_RE.match(line, i)
                # An element starts where a value can.
                and _js_regex_position(out, before)
            ):
                stack.append("<")
                out += tag.group()
                i = tag.end()
                continue
            elif (
                ch == "/"
                and (regex := _JS_REGEX_RE.match(line, i))
                # Straight after a /.../ literal a `/` divides.
                and (ended is None or _run_start(out, len(out)) > ended)
                and _js_regex_position(out, before)
            ):
                if keep:
                    kept_at[len(out)] = regex.group()
                out += f"/{' ' * (len(regex.group()) - 1)}"
                ended = len(out)
                i = regex.end()
                continue
            elif ch == "/" and line[i + 1 : i + 2] == "/":
                break
            elif ch == "/" and line[i + 1 : i + 2] == "*":
                stack.append("/*")
                out += "  "
                i += 2
                continue
            elif ch == "{" and top is not None:
                stack[-1] += 1
            elif ch == "}" and top is not None:
                if top == 0:
                    stack.pop()
                else:
                    stack[-1] -= 1
            elif ch == "\ufeff":
                ch = " "
            out += ch
        i += 1
    # Only a template literal, a block comment, a JSX element and the value of
    # one of its tag's attributes go on over the next line.
    while stack[-1:] in (['"'], ["'"]) and stack[-2:-1] != ["<"]:
        stack.pop()
    shown, at = [], 0
    for column, kept in kept_at.items():
        shown += out[at:column], kept
        at = column + len(kept)
    return ("".join(shown) + out[at:])[: len(line)]


def _blank_js_lines(lines, jsx=False, code=None):
    """``_blank_js_strings`` of each of a file's ``lines``, read in turn.
    Given ``code``, what this returns for them without it, the text of the
    literals that open on a line is kept, and the code a line follows is
    read from ``code``."""
    stack, blanked, before = [], [], ""
    for row, line in enumerate(lines):
        blanked.append(_blank_js_strings(line, stack, code is not None, jsx, before))
        read = (code or blanked)[row]
        if read.strip():
            # A `}` that ends a line is read as a block's, after which a
            # statement starts.
            before = "" if read.rstrip().endswith("}") else read
    return blanked


# What a declaration ends with where a type is still to come, so that a `{`
# there starts an object type and a line break does not end the declaration
_TS_TYPE_EXPECTED_RE = re.compile(r"(?:[:|&?]|\b(?:extends|is|keyof|readonly))\s*$")
# What a line starts with where it carries on the type or opens the body of
# the declaration before it
_TS_TYPE_CONTINUES = ("|", "&", "?", ":", "{")
# A bare `=` that ends a line: a value is assigned that starts on the next
_JS_ASSIGNMENT_END_RE = re.compile(r"(?<![=<>!])=(?!=)\s*$")


def _opens_body(signature: str, start: int = 0) -> bool:
    """True when ``signature`` reads `name(params) {` or `name(params): T {`
    from column ``start``: its first brace outside parentheses, before the
    `;` that ends the statement, directly follows a parameter list."""
    depth = 0
    after_params = False
    for column in range(start, len(signature)):
        char = signature[column]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            after_params = depth == 0
        elif depth == 0 and char == "{":
            return after_params
        elif depth == 0 and char == ";":
            return False
        elif depth == 0 and char in "=,.":
            after_params = False
    return False


# What a class's `extends` gives as its base, up to the body or `implements`
_TS_EXTENDS_RE = re.compile(r"\bextends\s+(.+?)\s*(?:\bimplements\b[^{]*)?\{?\s*$")
# What a class's or an interface's body, or that of an object a
# declaration, an arrow it declares, `export default`, a `return` or a
# property gives, opens after
_TS_BODY_HEAD_RE = re.compile(
    r"\s*(?:(?P<type>(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?"
    r"(?:class|interface)\b[^{]*|type\s+[\w$]+\s*(?:<[^{]*>)?\s*="
    r"|(?:extends|implements)\b[^{]*)"
    r"|(?:export\s+default|return"
    r"|(?:export\s+)?(?:const|let|var)\s+[\w$]+\s*(?::[^=]*)?=\s*"
    r"(?:\[|(?:async\s+)?(?:\([^()]*\)|[\w$]+)\s*(?::[^=]*?)?=>\s*\()?"
    rf"|(?P<nested>{_TS_WRITTEN_MEMBER_NAME})\s*:)\s*\Z)"
)
# A line that opens an object it ends with: `export default {`, a
# property's `key: {` or an element's `{`
_TS_OBJECT_OPENS_RE = re.compile(
    rf"\s*(?:(?P<default>export\s+default\s*)|(?P<key>{_TS_WRITTEN_MEMBER_NAME})\s*:\s*)?"
    r"\{\s*$"
)
# What starts a statement after a block's `}` or a statement's `;`: not a
# word that goes on with the statement before it, nor a template's `${`
_TS_STATEMENT_START_RE = re.compile(
    r"(?!(?:else|catch|finally|while|as|satisfies|instanceof|in|of|extends|implements"
    r"|from|is)"
    r"(?![\w$]))(?!\$\{)[A-Za-z_$]"
)
# A keyword whose block a `{` after it opens, as a head's `)`, an arrow's
# `>` and the start of a statement are
_BLOCK_KEYWORD_RE = re.compile(r"(?<![\w$])(?:else|try|finally|do)\Z")
# A namespace written with its name, not a module a string names
_TS_NAMESPACE_RE = re.compile(
    r"\s*(?:export\s+)?(?:declare\s+)?(?:namespace|module)\s+([\w$]+(?:\.[\w$]+)*)\s*\{"
)
# What may follow the `}` or `;` that ends a statement before the next
_TS_SEPARATORS_RE = re.compile(r"[ \t,;]*")
_TS_SPACE_RE = re.compile(r"[ \t]*")
# What a line may go on to after the `}` or `;` that ends a definition:
# another declaration, or a member of a class or an object with its body
_TS_DEFINITION_HEAD_RE = re.compile(
    r"(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?"
    r"(?:function|class|interface|enum|const|let|var)\b"
    rf"|{_TS_MEMBER_MODIFIERS}(?:\*\s*)?(?:[gs]et\s+)?(?P<member>{_TS_MEMBER_NAME})\s*[(<]"
)


def _split_lines(
    lines, code_lines, definition_head, type_head=None, ends=";", statement=None
):
    """``lines`` with a line split where it goes on to a definition after
    the `}` or `;` that ends the one before it, and where a type's body
    that opens on it holds members: after the body's `{`, after each member
    and before its `}`. ``definition_head`` matches where a definition
    starts, its group `member`, where it has one, a member that needs a
    body; ``type_head`` the start of a statement that a type body's `{`
    ends, its group `type`, where it has one, a type's, else an object's,
    and its group `nested` one that opens an object only in an object. Each
    part after the first is indented as its line is and once more for each
    such body it is in; a member ends with its `}` or a character of
    ``ends``. Where ``statement`` matches what follows a block's `}` or a
    statement's `;`, the line is split there too.
    Returns the lines, for each the index in ``lines`` of the one it is
    part of, and the code units (see `_code_units`) its columns are right
    of its own there. ``code_lines`` are ``lines`` blanked."""
    split, rows, shifts = [], [], []
    for row, (line, code) in enumerate(zip(lines, code_lines)):
        indent = line[: len(line) - len(line.lstrip())]
        cuts = []  # the column each part starts at, and the bodies it is in
        depth, bodies, begun, values = 0, [], 0, []
        # Whether each bracket opened on the line and still open is a
        # block's `{`: a statement starts only in the innermost one's
        opened_here = []
        for column, char in enumerate(code):
            head = char == "{" and type_head and type_head.match(code, begun, column)
            # A head that opens a body only inside the body of an object.
            nested = head and head.groupdict().get("nested")
            if head and (not nested or bodies and bodies[-1][2]):
                bodies.append((depth + 1, column, not head.groupdict().get("type")))
                cuts.append((column + 1, len(bodies)))
            # A `{` that opens a value, a template's `${` or one after `=`
            # as a JSX attribute's, closes no block.
            before = char == "{" and _run_start(code, column)
            if char == "{" and (
                code[column - 1 : column] == "$"
                or code[before - 1 : before] == "="
                and code[before - 2 : before - 1] not in "=!<>"
            ):
                values.append(depth + 1)
            closes_value = char == "}" and values and depth == values[-1]
            if closes_value:
                values.pop()
            if char in "([{":
                depth += 1
                lead = _run_start(code, column)
                opened_here.append(
                    char == "{"
                    and (
                        not lead
                        or code[lead - 1] in ")>{};"
                        or bool(_BLOCK_KEYWORD_RE.search(code, max(lead - 7, 0), lead))
                    )
                )
            elif char in ")]}":
                if opened_here:
                    opened_here.pop()
                if char == "}" and bodies and depth == bodies[-1][0]:
                    # A body that holds nothing is left as it is.
                    opened = bodies.pop()[1]
                    if code[opened + 1 : column].strip():
                        cuts.append((column, len(bodies)))
                    else:
                        cuts.remove((opened + 1, len(bodies) + 1))
                depth -= 1
            if closes_value:
                continue
            if char in "{};":
                begun = column + 1
            if bodies and depth == bodies[-1][0] and (char == "}" or char in ends):
                cuts.append((column + 1, len(bodies)))
            elif char in "};" and not bodies and opened_here[-1:] != [False]:
                at = _TS_SEPARATORS_RE.match(code, column + 1).end()
                head = definition_head.match(code, at)
                member = head and head.groupdict().get("member")
                defines = head and not (
                    member
                    and (member in _JSTS_CONTROL_KEYWORDS or not _opens_body(code, at))
                )
                after = _TS_SPACE_RE.match(code, column + 1).end()
                if defines:
                    cuts.append((at, 0))
                elif statement and statement.match(code, after):
                    cuts.append((after, 0))
        if not cuts:
            split.append(line)
            rows.append(row)
            shifts.append((0, 0, 0))
            continue
        start, level, units = 0, 0, (0, 0, 0)
        for at, inside in [*cuts, (len(line), 0)]:
            if at <= start:
                continue
            if not start or line[start:at].strip():
                prefix = indent + "    " * level if start else ""
                split.append(prefix + line[start:at])
                rows.append(row)
                shifts.append(tuple(unit - len(prefix) for unit in units))
            units = _code_units(line[start:at], units)
            start, level = at, inside
    return split, rows, shifts


class TSParser(BaseParser):
    binding_re = re.compile(
        r"\b(?:const|let|var)\s+(?P<name>[\w$]+)\s*(?::[^;=]*?)?=(?![=>])\s*"
        r"(?P<unwrap>await\b)?"
    )
    bindings = staticmethod(_js_bindings)
    self_name = "this"
    chained_values = True
    # The lines braces are counted in have their literals blanked already.
    _non_code = None

    def _typed_origin(self, origin, declared):
        """A name declared with a type and bound to a call is read as what
        the call returns: the type reads the members that gives none."""
        return f"{origin}>{declared[:-2]}"

    def _binds_name(self, name):
        """A name the file imports is what the module gives it, whatever a
        declaration of that name in a narrower scope says, and one a block
        the line closes declares is bound on no later line."""
        return name not in self._imported and name not in self._closed_names.get(
            self._line, ()
        )

    def _read_as(self, origin):
        """The origin path a folded declaration reads a name as: what the
        call a name was bound to returns, read in the type it is declared
        with. Any other path is read as it is."""
        call, _, declared = (origin or "").partition(">")
        if not call.startswith("#") or not declared:
            return origin
        name, dot, member = declared.partition(".")
        return f"{name}(){dot}{member}"

    def _bind_closures(self, code, index, lines, before):
        """Type the parameters of an arrow function that opens before
        ``before`` as the value of an annotated declaration, an
        assignment or a `return`, from the function type it is given;
        arguments of calls go on to the shared path."""
        while self._closures and self._closures[0][0] < before:
            bar, params, _end = self._closures[0]
            if self.bracket_around(bar)[1] == "(":
                break
            head = code[:bar]
            origin = None
            if re.search(r"\breturn\s*$", head):
                # `return (x) => ...`: the parameters the enclosing
                # function's declared return type takes.
                function_id = self.context_at(index + 1)[0]
                written = (self._signatures.get(function_id) or [{}])[-1].get("returns")
                if written:
                    escaped = (
                        written.replace("(", "<").replace(")", ">").replace(".", "::")
                    )
                    origin = f">{escaped}"
            else:
                declared = re.search(
                    r"\b(?:const|let|var)\s+([\w$]+)(?:\s*:\s*([^;]*?))?\s*=\s*$",
                    head,
                )
                assigned = re.search(r"(?<![\w$.])([\w$]+)\s*=(?![=>])\s*$", head)
                if declared and declared[2]:
                    # The annotation is the closure's own type: read its
                    # parameters where they are written, not through the
                    # variable — a definition the value becomes does not
                    # hold it as a member. A declaration without a type
                    # gives the closure none.
                    escaped = (
                        declared[2]
                        .strip()
                        .replace("(", "<")
                        .replace(")", ">")
                        .replace(".", "::")
                    )
                    origin = f">{escaped}"
                elif assigned and not declared:
                    origin = f"={assigned[1]}"
            if origin is None:
                break
            self._closures.pop(0)
            bindings = {
                name: _bound_origin(f"{origin}.^{number}", steps)
                for number, param in enumerate(_bracketed_parts(params + ")"))
                for name, steps in self.bindings(param)
            }
            self._bind(bindings, index, lines, code.rstrip().endswith("{"))
        while self._held_closures and self._held_closures[0][0] < before:
            # A function given as a property of an object literal that a
            # call is passed takes what the call passes to that property.
            bar, params = self._held_closures.pop(0)
            if held := self._argument_label(code, bar):
                bindings = {
                    name: _bound_origin(f"{held}|{number}", steps)
                    for number, param in enumerate(_bracketed_parts(params + ")"))
                    for name, steps in self.bindings(param)
                }
                self._bind(bindings, index, lines, code.rstrip().endswith("{"))
        super()._bind_closures(code, index, lines, before)

    def _held_label(self, code, column):
        return self._argument_label(code, column)

    def _element_origin(self, end, callee, lineno, since, source_id):
        """`items.*.callee` for a call on an element of an array,
        `items[i].callee()`: what iterating `items` gives."""
        code = self._code
        if code[end - 1 : end] != "]":
            return None
        depth = 0
        for at in range(end - 1, -1, -1):
            depth += (code[at] == "]") - (code[at] == "[")
            if not depth:
                break
        if depth:
            return None
        if code[at - 1 : at] == ")":
            edge = self._closes.get(at - 1)
            held = None if edge is None else f"#{edge}"
        else:
            path = re.search(r"(?<![\w$.])(?:this\.)?[\w$]+(?:\.[\w$]+)*$", code[:at])
            owner = source_id.split("::")[-2] if source_id.count("::") > 1 else None
            held = path and self.call_origin(f"{path[0]}.x", lineno, since, owner)
            held = held and held[:-2]
        return held and f"{held}.*.{callee}"

    def _argument_label(self, code, column):
        """``#edge|argument`` for what starts at ``column`` of the line
        being read where it is an argument of a call on the line, and
        ``...:key`` where it is the value of the property `key` of an object
        literal so labelled, opened on the line or an earlier one; else
        None."""
        opened, bracket = self.bracket_around(column)
        if bracket == "(" and opened is not None:
            edge = self._closes.get(self._pairs.get(opened), self._opens.get(opened))
            parts = _bracketed_parts(code[opened + 1 : column] + "x", expression=True)
            if edge is None or parts[-1].strip() != "x":
                return None
            return f"#{edge}|{len(parts) - 1}"
        key = bracket == "{" and re.search(
            r"(?:^|[{,])\s*([\w$]+)\s*:\s*(?:async\s+)?$",
            code[opened or 0 : column],
        )
        if not key:
            return None
        held = (
            self.held_around(column)
            if opened is None
            else self._argument_label(code, opened)
        )
        return held and f"{held}:{key[1]}"

    def _this_in_object(self, code, column):
        """``#edge|argument|this`` origin for a `this` at ``column`` that is
        inside an object literal given as an argument of a call on the
        line, else None."""
        opened = self.paren_around(column)
        if opened is None:
            return None
        # The object literal holds the `this`: a `{` inside the call's
        # arguments that is still open at the `this`.
        if code[opened:column].count("{") <= code[opened:column].count("}"):
            return None
        edge = self._closes.get(self._pairs.get(opened), self._opens.get(opened))
        if edge is None:
            return None
        brace = code.index("{", opened)
        position = (
            len(_bracketed_parts(code[opened + 1 : brace] + "x", expression=True)) - 1
        )
        return f"#{edge}|{position}|this"

    def _operand(self, text):
        operand = re.sub(r"^await\s+", "", text)
        return operand.rstrip("!"), len(operand) < len(text)

    def _bind_patterns(self, code, index, lines, since, owner):
        """Type the names that a `for` or a declaration with a pattern
        binds on line ``index``, from the value iterated or taken apart. A
        `for...in` gives the names of properties."""
        found = (
            _JS_HEAD_END_RE.search(code) and _JS_FOR_RE.search(code)
        ) or _JS_PATTERN_RE.search(code)
        pattern = found and found["pattern"]
        if not found and (ended := _JS_PATTERN_END_RE.match(code)):
            # A pattern over lines, from the line its declaration opens it on.
            opened = None
            for row in range(index - 1, max(index - 40, -1), -1):
                if opened := _JS_PATTERN_OPENS_RE.search(lines[row]):
                    break
            if not opened:
                return
            parts = [lines[row][opened.start("open") :], *lines[row + 1 : index]]
            pattern = " ".join([*parts, code[: ended.end("close")]])
            found = ended
        if not found:
            return
        lineno = index + 1
        kind = found.groupdict().get("kind")
        span = found.span("value")
        origin = self._expression_origin(code, *span, lineno, since, owner)
        if origin is None and not kind:
            origin = self._opened_value(code, span[0], index, lines)
        if kind == "of" and code[span[0]] == "[" and code[span[1] - 1] == "]":
            key = f"[{len(self.builder.literals)}]"
            self.builder.literals[key] = self._elements(
                code, span[0] + 1, lineno, since, owner
            )
            origin = f"Array<{key}>()"
        if kind == "of":
            origin = origin and f"{origin}.*"
        elif kind:
            origin = "string()"
        bindings = {
            name: _bound_origin(origin, steps) for name, steps in self.bindings(pattern)
        }
        if kind:
            self._bind(bindings, index, lines, code.rstrip().endswith("{"))
        else:
            for name, bound in bindings.items():
                self._rebind(name, lineno, bound)

    def _binds_file(self, name):
        """A line outside every definition that starts at the margin, as
        the body of a callback does not, binds for the module, unless the
        name is an import's. The margin is where the least indented line of
        the text being parsed starts: a `<script>` block may be indented as
        a whole."""
        margin = len(self._raw) - len(self._raw.lstrip()) == self._margin
        return self._scope is None and margin and name not in self._imported

    def call_origin(self, callee, lineno, since, owner=None):
        """As ``BaseParser.call_origin``, and for a bare callee that a
        pattern bound to a member of a value, `const { run } = make()`, the
        origin path of that member, and for one declared `typeof x`, `x`."""
        bound = "." not in callee and callee not in self._imported
        held = bound and self._declared(callee, lineno, since)
        if held and re.search(r"(?:^|\.)[\w$]+$", held):
            return held
        return super().call_origin(callee, lineno, since, owner)

    def _is_local(self, name):
        """As the base's; the parameters of the definition are read from its
        signature, whether or not they are typed."""
        written = (self._signatures.get(self._scope) or [{}])[-1].get("params", ())
        names = (
            re.findall(r"[\w$]+", re.split(r"[:=]", param)[0]) for param in written
        )
        return super()._is_local(name) or any(name in found for found in names)

    def _object_held(self, lines, index, column, since, owner):
        """What each property of the object literal that opens at ``column``
        of line ``index`` holds, by name: the origin path of the name or
        member it is given as, of the call written for it, or of the call's
        result that a local it names was bound to; its own name where it is
        a method or a function written there, None where it is anything
        else. A property under a name that is no plain one, a spread among
        them, is left out. With it, the number of the line that closes the
        object. ``lines`` are the file's code lines."""
        written, depth = [], 0
        for line in (lines[index][column:], *lines[index + 1 :]):
            written.append(_blank_js_strings(line))
            depth += written[-1].count("{") - written[-1].count("}")
            if depth <= 0:
                break
        text = "\n".join(written)
        depth = 0
        for end, char in enumerate(text):
            depth += (char == "{") - (char == "}")
            if not depth:
                break
        body = text[1:end]
        # The text around a nested part keeps the column it is written at.
        while (reduced := _JS_NESTED_RE.sub(_nested_part, body)) != body:
            body = reduced
        held = {}
        at = 1  # where the part being read starts in `text`
        for piece in body.split(","):
            start = at + len(piece) - len(piece.lstrip())
            at += len(piece) + 1
            part = piece.strip()
            if not part:
                continue
            head, colon, tail = part.partition(":")
            name, value = head.strip(), tail.strip()
            # A function written in the object is a member of it, filed
            # under the object's own ID, where it is written below the line
            # the object opens on. That line is read as what opens it, so a
            # function on it gets no node.
            member = start > len(written[0])
            method = re.fullmatch(
                r"(?:async\s+|(get|set)\s+)?\*?\s*([\w$]+)\s*\0\s*\0", name
            )
            if method and not colon:
                # An accessor is filed as `get name` or `set name`.
                filed = " ".join(filter(None, method.groups()))
                held[filed] = f"={filed}" if member else filed
            elif not re.fullmatch(r"[\w$]+", name):
                continue
            elif not colon or re.fullmatch(r"[\w$]+(?:\.[\w$]+)*", value):
                held[name] = self._name_held(value or name, index + 1, since)
            else:
                made = "=>" in value or re.match(r"(?:async\s+)?function\b", value)
                value_at = start + len(head) + 1 + len(tail) - len(tail.lstrip())
                if made:
                    held[name] = f"={name}" if member else name
                else:
                    self._call_held(
                        held, name, text, index, column, value_at, at - 1, since, owner
                    )
        return held, index + 1 + text.count("\n", 0, end)

    def _returned_object_at(self, returned_objects, parent_id, lineno):
        """The object that the definition ``parent_id`` returns at the line
        being read, where the line is one of its lines: the innermost of
        those it returns that reaches it, as its ID and the line the names
        written in it are read from — the definition that returns it."""
        for parent, first, last, owner_id, scope in reversed(returned_objects):
            if parent == parent_id and first <= lineno <= last:
                return owner_id, scope
        return None, None

    def _name_held(self, value, lineno, since):
        """What a property of the object being read holds where the value it
        is given is a name: the origin path of that name where a local of
        the function was bound to a call's result, else the name itself."""
        bound = re.fullmatch(r"[\w$]+", value) and self._declared(value, lineno, since)
        return bound if bound and bound.startswith("#") else value

    def _call_held(self, held, name, text, index, column, start, end, since, owner):
        """Set ``held[name]`` to the origin path of the call that the
        property at ``start`` to ``end`` of the text of the object that
        opens at ``column`` of line ``index`` is given: now for one on that
        line, the line being read, else once its own line is read."""
        row = text.count("\n", 0, start)
        held[name] = None
        if row != text.count("\n", 0, end):
            return
        if row:
            line_start = text.rfind("\n", 0, start) + 1
            later = (held, name, start - line_start, end - line_start)
            self._held_later.setdefault(index + row, []).append(later)
            return
        held[name] = self._expression_origin(
            self._code, column + start, column + end, index + 1, since, owner
        )

    def _opened_value(self, code, start, index, lines):
        """Origin path of a value that is one call which ``code``, line
        ``index``, leaves open from column ``start`` and a later line
        closes, ending the statement. None for any other value."""
        opened = sorted(column for column in self._opens if column >= start)
        if not opened or not _is_call_chain(code[start : opened[0]]):
            return None
        below = lines[index + 1 :]
        depth = 0
        for line in (code[opened[0] :], *below):
            for column, char in enumerate(line):
                depth += (char == "(") - (char == ")")
                if not depth:
                    # A `(` no call was read at, as `import(`'s, holds no
                    # call's result.
                    edge = self._opens[opened[0]]
                    ended = not line[column + 1 :].strip(" ;")
                    return f"#{edge}" if ended and edge is not None else None
        return None

    def _ends_value(self, code, edge):
        """Whether ``code``, the line being read, closes the call at edge
        index ``edge`` and has nothing after it but the end of the
        statement; None where it does not close it."""
        closed = [column for column, call in self._closes.items() if call == edge]
        return not code[max(closed) + 1 :].strip(" ;") if closed else None

    def _elements(self, code, start, lineno, since, owner):
        """The elements of an array literal or the arguments of a call that
        ``code``, the line being read, has from column ``start`` up to the
        bracket that closes it or to its end: the origin path of each, for
        an object literal those of its properties by name, None where it
        has none."""
        elements = []
        closed = self._pairs.get(start - 1)
        for part in _bracketed_parts(code[start:closed], expression=True):
            start = code.index(part, start)
            end = start + len(part)
            if part[0] + part[-1] != "{}":
                held = self._expression_origin(code, start, end, lineno, since, owner)
                elements.append(held)
                start = end
                continue
            held = {}
            for item in _bracketed_parts(part[1:], expression=True):
                start = code.index(item, start)
                name, colon, _ = item.partition(":")
                span = start + len(name) * bool(colon) + bool(colon), start + len(item)
                held[name.strip()] = self._expression_origin(
                    code, *span, lineno, since, owner
                )
                start = span[1]
            named = all(re.fullmatch(r"[\w$]+", name) for name in held)
            elements.append(held if named else None)
        return elements

    def parse(self, file_path, rel_path, content, line_offset=0):
        """Parse ``content``: the text of ``file_path``, or one ``<script>``
        block of it that starts ``line_offset`` lines in. Emitted line
        numbers are relative to ``content``."""
        file_id = self.add_file_node(rel_path, content)

        jsx = str(file_path).endswith(_JSX_EXTENSIONS)
        lines = source_lines(content)
        self.line_offset = line_offset
        self.line_count = len(lines)
        # The lines read, a definition after another's end on a line on one
        # of its own, and the source line each is part of
        lines = self.split_source(
            lines,
            _blank_js_lines(lines, jsx=jsx),
            _TS_DEFINITION_HEAD_RE,
            _TS_BODY_HEAD_RE,
            ";,",
            _TS_STATEMENT_START_RE,
        )

        written = (line for line in lines if line.strip())
        self._margin = min((len(l) - len(l.lstrip()) for l in written), default=0)
        self.enter_context(file_id, 0, len(lines))

        # A class at a line's margin, or a class expression: `const C =
        # class`, as `let` or `var`, `member = class`, and inside brackets
        # `new (class` or `call(a, class`, where `class` is no property's
        # name. Searched on the line with its literals blanked: an
        # expression is written anywhere on a line, and a string may hold
        # one.
        re_class = re.compile(
            r"(?P<expression>"
            r"(?:const|let|var)\s+(?P<declarator>[\w$]+)\s*(?::[^=,;]*)?=\s*"
            r"|^\s*[\w$][\w$.]*\s*=\s*"
            r"|(?P<enclosed>[(,]\s*|[\w$]\s*=\s*|\breturn\s+|=>\s*|[:?]\s*)"
            r"(?=class\s*(?:[\w$]+\s*)?(?:[{<]|(?:extends|implements)\b)))"
            r"class\b"
            r"|^\s*(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:abstract\s+)?class\b(?!\s*=)"
            r"(?:\s+(?!extends\b|implements\b)(?P<plain>[a-zA-Z0-9_$]+))?"
        )
        # `interface Name` and an object type, `type Name<T> = {`
        re_interface = re.compile(
            r"^\s*(?:export\s+)?(?:declare\s+)?"
            r"(?:interface\s+([\w$]+)|type\s+([\w$]+)\s*(?:<[^=]*>)?\s*=\s*\{)"
        )
        re_func = re.compile(
            r"^\s*(?:export\s+)?(?:declare\s+)?(?:default\s+)?(?:async\s+)?"
            r"function\s+([a-zA-Z0-9_$]+)"
        )
        # `name(`, or `name = (` where a class property holds an arrow function,
        # either with type parameters before the `(`
        re_method = re.compile(
            rf"^\s*{_TS_MEMBER_MODIFIERS}"
            rf"(?:\*\s*)?({_TS_WRITTEN_MEMBER_NAME})\s*(=\s*(?:async\s+)?)?(?:[(]|<(?![\s=]))"
        )
        # Getter/setter methods: get name() or set name(value)
        re_getter_setter = re.compile(
            rf"^\s*{_TS_MEMBER_MODIFIERS}(get|set)\s+({_TS_WRITTEN_MEMBER_NAME})\s*[(]"
        )
        re_enum = re.compile(r"^\s*(?:export\s+)?enum\s+([a-zA-Z0-9_$]+)")
        # `const name: Type = value;`, the type kept where it is one named
        # type, or `const name` of a statement that goes on over the next lines
        re_const = re.compile(
            r"^\s*(?:export\s+)?(?:declare\s+)?const\s+([a-zA-Z0-9_$]+)"
            r"(?:\s*:\s*(\([^()]*\)\s*=>\s*[\w$.]+|[\w$.]+(?:<[^|&]*>)?)"
            r"(?=\s*(?:=(?!>)|;?\s*$)))?"
            r"(?:\s*(?::\s*.*?)?\s*=(?![=>])\s*(.*?);)?"
        )
        re_import = re.compile(r'import\s+.*?from\s+([\'"])(.*?)\1')
        re_call_candidate = re.compile(
            rf"({_TS_MEMBER_NAME}(?:\??\.{_TS_MEMBER_NAME})*)\s*(?:\?\.)?(?=[<(])"
        )
        # In an object literal: `...name`, `key: name` or `name` alone
        re_held = re.compile(r"^\s*(\.\.\.)?(?:([\w$]+)\s*:\s*)?([\w$]+)\s*,?\s*$")
        self.scan_declarations(lines, _TS_DECLARATIONS)
        self.builder.type_aliases.update(_TYPE_ALIAS_RE.findall(content))
        for name, generics, text in _TS_ALIAS_RE.findall(content):
            params = tuple(part.split()[0] for part in _bracketed_parts(f"{generics}>"))
            self.builder.ts_aliases[file_id, name] = params, text.strip()
        for star, namespace, clause, specifier in _TS_REEXPORT_RE.findall(content):
            # `export * as ns` exports the module itself, under a name.
            names = [(namespace or "*", "*" if namespace else "")] if star else []
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
        for default, *names in _TS_DEFAULT_EXPORT_RE.findall(content):
            exported = "default" if default else "="
            entry = (exported, "", "".join(names))
            self.builder.reexports.setdefault(file_id, []).append(entry)
        names = self.builder.import_names.setdefault(file_id, {})
        for clause, specifier in _TS_IMPORT_RE.findall(content):
            for part in re.finditer(r"[^{},]+", clause):
                # `name`, `type name`, `name as alias`, `* as alias`; before
                # a `{`, the module's default
                original, renamed, alias = part[0].partition(" as ")
                local = alias.split()[-1:] if renamed else original.split()[-1:]
                if local:
                    self.builder.imports.setdefault(file_id, {})[local[0]] = specifier
                    if original.split() == ["*"]:
                        names[local[0]] = "="
                    elif "{" not in clause[: part.start()]:
                        names[local[0]] = "default"
                    elif renamed and original.split():
                        names[local[0]] = original.split()[-1]
        # The lines a `require` or an `import()` binds names on
        required = set()
        for found in _TS_REQUIRE_RE.finditer(content):
            clause, call, specifier = found.groups()
            required.add(self.read_line(content.count("\n", 0, found.start()) + 1) - 1)
            if "{" not in clause:
                names[clause] = "="
            for part in re.split(r"[{},]", clause):
                # `name`, or `name: local` in a destructuring pattern
                exported, renamed, local = part.rpartition(":")
                local = local.strip()
                if local:
                    self.builder.imports.setdefault(file_id, {})[local] = specifier
                    if call == "require":
                        self.builder.required_names.setdefault(file_id, set()).add(
                            local
                        )
                    if renamed and exported.strip():
                        names[local] = exported.strip()
        imported = self._imported = self.builder.imports.get(file_id, ())
        # Which characters are code is read once for the file. Each of
        # ``code_lines`` is a line without its comments and with the text of
        # its literals blanked, for the names it binds and the braces
        # counted on it.
        code_lines = _blank_js_lines(lines, jsx=jsx)
        bound = _js_bound("\n".join(code_lines))
        self.builder.js_bound[file_id] = bound.union(imported)
        # An imported name is what its module exports, whatever a field or
        # a parameter of that name is declared as.
        for name in imported:
            self.declared.pop(name, None)
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

        untyped = ("", 0)  # the variable the line before declared, and that line
        # The call that starts the value of a variable declared with a type,
        # and that type, until the call is closed
        initialises = None
        array = None  # the array literal being read, and its open brackets
        # The constant whose value is an array over lines, and its last line
        elements_of = None
        # The call a `return` gives that is still open, with where it is
        # recorded, and the last line of the nameless functions each
        # definition being read has open: a `return` past it is that
        # function's, not the definition's.
        returning = None
        nameless = {}
        # Each object a definition returns: the definition, the lines the
        # object spans and the object's own ID. A function written on one of
        # those lines is a member of the object, not of the definition.
        returned_objects = []
        # Line index -> (what an object holds, property, columns of its
        # value) for each property given on that line of an object read before
        self._held_later = {}
        self._held_closures = []
        # Each callback given an owner, innermost last: its last line and how
        # many calls are open around it. A call left open inside it, not one
        # around it, makes a closure on a later line one of its arguments.
        callbacks = []
        callback_ids = set()  # the owners of those callbacks
        # The contexts that end on the line before the one their block closes
        # on: those of callbacks and of nameless functions
        closed_after = set()
        expression_ids = set()  # functions whose body is an expression
        # The definition, and its first and last line and column, of each
        # context: a column of None where it holds the whole line
        extents = []
        # context -> each name its body declares -> the lines it is in scope
        declared_names = {}
        self._closed_names = {}  # line -> the names declared in a block it closes
        # The brackets open where the line being read starts, innermost
        # last, True for the `(` of a call, and the last character of code
        # before the line: an object that follows such a `(`, or a `,`
        # inside it, is an argument of the call.
        brackets = []
        before = ""
        self._signatures = {}  # node ID -> the signatures it is declared with
        class_ids = set()  # class/interface/enum nodes declared in this file
        namespaces = {}  # namespace ID -> the metadata of its node
        class_at = {}  # class ID -> the column its `class` ends at on its line
        # constants of the file or a callback that hold an object literal
        object_ids = set()
        interface_ids = set()  # interfaces that have the ID `<scope>::<name>`
        type_bodies = set()  # interfaces and object types, whatever their ID
        value_ids = set()  # functions and variables, which always have it

        def enter(node_id, lineno, end_lineno, signature="", span=None):
            """Make a definition that spans further lines the context, and
            record its ``span``: by default from the line after ``lineno``
            to the `}` that closes its body on line ``end_lineno``, or that
            whole line where its body is an expression."""
            if end_lineno > lineno:
                self.enter_context(node_id, lineno, end_lineno, _ts_params(signature))
                # An arrow function's signature starts with its parameters.
                arrow = re.match(r"\s*(?:\(|[\w$]+\s*=>)", signature)
                if arrow and signature.rpartition("=>")[2].strip() != "{":
                    expression_ids.add(node_id)
                closed = None
                if node_id not in expression_ids:
                    closed = body_end(lineno, end_lineno)
                extents.append(
                    (node_id, *(span or ((lineno - 1, None), (end_lineno - 1, closed))))
                )

        def body_end(lineno, end_lineno):
            """The column of the `}` on line ``end_lineno`` that closes the
            body of a definition written on line ``lineno``: the one whose
            `{` opens first there or after. None where there is none."""
            row = end_lineno - 1
            columns = braces_on[row][0] if row < len(braces_on) else ()
            closing = [
                (open_at[(row, column)], column)
                for column in columns
                if (row, column) in open_at and open_at[(row, column)][0] >= lineno - 1
            ]
            return min(closing)[1] if closing else None

        def owner_at(row, column):
            """The innermost definition whose span holds column ``column``
            of line ``row``, where a span starts or ends at a known column
            of it; else None, and the line is read whole. ``extents`` holds
            those that end on the line or after, outermost first."""
            if not any(
                start[0] == row
                and start[1] is not None
                or end[0] == row
                and end[1] is not None
                for _, start, end in extents
            ):
                return None
            for node_id, (first, opened), (last, closed) in reversed(extents):
                after = first < row or (
                    first == row and opened is not None and opened < column
                )
                before = last > row or closed is None or column < closed
                if after and before:
                    return node_id
            return file_id

        def value_of(name, lineno, since, owner):
            """The origin of a call of ``name`` on line ``lineno`` that calls a
            value: `?` for a parameter, what one declared `typeof x` holds,
            and `=name` for a variable a function declares, where no
            definition there defines the name and the variable's origin is
            not known."""
            for context in reaching(lineno):
                if name in context[3]:
                    held = self._declared(name, lineno, since)
                    return held if held and not held.endswith("()") else "?"
                if f"{context[0]}::{name}" in self.builder.node_registry:
                    return None
                lines_in = declared_names.get(context[:3], {}).get(name, ())
                if any(first <= lineno <= last for first, last in lines_in):
                    held = self.call_origin(name, lineno, since, owner)
                    return None if held else f"={name}"
            return None

        def reaching(at):
            """The contexts that reach line ``at``, innermost first."""
            return [
                context
                for context in reversed(self._contexts)
                if at <= context[2] + (context[:3] in closed_after)
            ]

        def code_at(row, column):
            """The line and column of the first code at or after column
            ``column`` of line ``row``."""
            while row < len(code_lines) and not code_lines[row][column:].strip():
                row, column = row + 1, 0
            if row < len(code_lines):
                text = code_lines[row]
                column = len(text) - len(text[column:].lstrip())
            return row, column

        def past(row, column, last=None):
            """Just past the brackets that open at column ``column`` of line
            ``row`` or, for the statement there that ends by line ``last``,
            past its `;`, or at the bracket that closes what holds it."""
            depth = 0
            for at_row in range(row, len(code_lines) if last is None else last + 1):
                for at in range(column, len(code_lines[at_row])):
                    char = code_lines[at_row][at]
                    depth += (char in "([{") - (char in ")]}")
                    if depth < 0:
                        return at_row, at
                    if depth == 0 and (last is None or char == ";"):
                        return at_row, at + 1
                column = 0
            return (len(code_lines) if last is None else last + 1), 0

        def past_statement(row, column):
            """The line and column just past the statement that starts at or
            after column ``column`` of line ``row``: a block, a statement a
            keyword starts with its head, its body and the clauses that go on
            with it, a labelled statement, or one statement."""
            opened = []  # the keywords of the statements whose body is read
            while True:
                row, column = code_at(row, column)
                if row == len(code_lines):
                    return row, 0
                compound = _JS_COMPOUND_RE.match(code_lines[row], column)
                if code_lines[row].startswith("{", column):
                    row, column = past(row, column)
                elif compound and not (compound[1] or compound[2]):
                    column = compound.end()
                    continue
                elif compound:
                    opened.append(compound[1] or compound[2])
                    row, column = code_at(row, compound.end())
                    if (
                        compound[1]
                        and row < len(code_lines)
                        and code_lines[row].startswith("(", column)
                    ):
                        row, column = past(row, column)
                    continue
                else:
                    last = _statement_end(code_lines, row, column)
                    row, column = past(row, column, last)
                # The statement read ends those whose body it is, but for the
                # clause that goes on with one.
                while opened:
                    keyword = opened.pop()
                    ahead, at = code_at(row, column)
                    clause = (
                        ahead < len(code_lines)
                        and keyword in _JS_CLAUSE_RE
                        and _JS_CLAUSE_RE[keyword].match(code_lines[ahead], at)
                    )
                    if clause and keyword == "do":
                        row, column = past(*code_at(ahead, clause.end()))
                        ahead, at = code_at(row, column)
                        if ahead < len(code_lines) and code_lines[ahead].startswith(
                            ";", at
                        ):
                            row, column = ahead, at + 1
                    elif clause:
                        row, column = ahead, at
                        break
                else:
                    return row, column

        def in_scope(row, column, kind, head=None):
            """The last line a variable declared at column ``column`` of line
            ``row`` is in scope on: that of the block open there, for a `var`
            that of the innermost function's block, and for the binding a
            `for` or `catch` head declares, whose `(` is at ``head``, that of
            the statement after the head."""
            if head and kind != "var":
                row, column = past_statement(*past(*head))
                return row + (column > 0)
            columns, innermost = braces_on[row]
            opened = innermost[bisect_left(columns, column)]
            while opened:
                if kind != "var" or function_body(opened):
                    return closes_on.get(opened, len(lines))
                opened = outer_brace[opened]
            return len(lines)

        @cache
        def function_body(opened):
            """Whether the `{` at ``opened`` opens a function's body."""
            row, column = opened
            return bool(_JS_BLOCK_FUNCTION_RE.search(code_lines[row], 0, column + 1))

        # Each of ``text_lines`` keeps the text of the literals that open on
        # it, and is the line the loop below reads.
        text_lines = _blank_js_lines(lines, jsx=jsx, code=code_lines)
        # The line each `{` is closed on and the `{` open where it opens,
        # and for each line the columns of its braces with the innermost
        # `{` open before the first and after each
        braces_on, closes_on, outer_brace, braces = [], {}, {}, []
        # (line, column) of each `}` -> that of the `{` it closes, and back
        open_at, close_at = {}, {}
        for row, text in enumerate(code_lines):
            columns, innermost = [], [braces[-1] if braces else None]
            for at, char in enumerate(text):
                if char == "{":
                    outer_brace[(row, at)] = innermost[-1]
                    braces.append((row, at))
                elif char == "}" and braces:
                    open_at[(row, at)] = braces[-1]
                    close_at[braces[-1]] = row, at
                    closes_on[braces.pop()] = row + 1
                else:
                    continue
                columns.append(at)
                innermost.append(braces[-1] if braces else None)
            braces_on.append((columns, innermost))

        @cache
        def declared_until(lineno, name):
            """The last line a `const`, `let` or `var` of ``name`` declared
            on line ``lineno`` is in scope on; the file's last for any other
            declaration."""
            found = re.search(
                rf"\b(const|let|var)\b[^;=]*?(?<![\w$]){re.escape(name)}(?![\w$])",
                code_lines[lineno - 1],
            )
            return (
                in_scope(lineno - 1, found.start(), found[1]) if found else len(lines)
            )

        self._binding_ends = {}
        self._scope_end = lambda lineno, name: (
            self._binding_ends.get((lineno, name)) or declared_until(lineno, name)
        )

        i = 0
        signature_until = 0  # the last line of a definition's signature
        later_declarators = {}  # line index -> declarators a statement gives it
        while i < len(lines):
            line = text_lines[i]
            lineno = i + 1
            # Column of a function/method name declared on this line, which
            # the regex call fallback must not read as a call.
            declared_at = None
            valued = None  # the signature of the arrow function the line is
            arrow_id = None  # the node of the arrow function the line declares
            own_defs = []  # (node ID, end of its name) declared on the line
            holder = None  # the constant the line declares, and where its value starts
            field_spans = []  # (start, end, class ID) of a field's value on it
            nodes_before = len(self.builder.nodes)

            extents[:] = [extent for extent in extents if extent[2][0] >= i]
            parent_id = self.context_at(lineno)[0]
            # The line that the `}` closing a definition leads is read
            # outside it, as the line that closes a callback is. An
            # expression body ends with its statement, not at a `}`.
            if lineno > signature_until and _closes_first(code_lines[i]):
                unclosed = callback_ids | expression_ids
                while len(self._contexts) > 1 and (
                    self._contexts[-1][2] < lineno
                    or self._contexts[-1][2] == lineno
                    and self._contexts[-1][0] not in unclosed
                ):
                    self._contexts.pop()
                parent_id = self._contexts[-1][0]
            # The object the definition returns, where the line being read is
            # one of its lines: a function written there is a member of that
            # object, which no source can name, and reads the names the
            # definition that returns it does.
            object_id, object_scope = self._returned_object_at(
                returned_objects, parent_id, lineno
            )

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
                    text = f"{text} {text_lines[j].strip()}".lstrip()
                    arrow = _arrow_function(text)
                    code = _NON_CODE_RE.sub("", text).rstrip()
                    # The value may start on the line after the `=`.
                    open_ended = code.endswith("=") or code.count("(") > code.count(")")
                    if arrow or not open_ended:
                        return j, arrow
                return idx, None

            def consume_sig(idx, start=0):
                """Read the declaration that starts at column ``start`` of
                line ``idx`` up to the `{` that opens its body, or to the end
                of one that has none. Returns the index of that line, the
                declaration's text and the line its body ends on: its first
                line when it has none, and where it ends with an arrow's
                `=>`, the last line of the expression after it."""
                j, sig, end = read_sig(idx, start)
                if sig.rstrip().endswith("=>"):
                    end = _statement_end(code_lines, idx) + 1
                return j, sig, end

            def read_sig(idx, start):
                """What ``consume_sig`` returns, but for an arrow's
                expression body."""
                sig = ""
                stack = []  # brackets left open
                closed = False  # a parameter list has been read
                for j in range(idx, min(idx + 40, len(lines))):
                    skipped = start if j == idx else 0
                    text = " " * skipped + text_lines[j][skipped:]
                    code = " " * skipped + code_lines[j][skipped:]
                    for column, char in enumerate(code):
                        if (
                            char == "{"
                            and not stack
                            and not _TS_TYPE_EXPECTED_RE.search(sig + text[:column])
                        ):
                            end = self._find_block_end(code_lines, j, column)
                            return j, sig + text[: column + 1].lstrip(), end
                        # `a < b` compares; `Type<` opens type arguments.
                        if char in "([{" or (
                            char == "<"
                            and code[column + 1 : column + 2] not in (" ", "=")
                        ):
                            stack.append(char)
                        elif char == ">":
                            if stack[-1:] == ["<"] and code[column - 1 : column] != "=":
                                stack.pop()
                        elif char in ")]}":
                            while stack[-1:] == ["<"]:
                                stack.pop()
                            if not stack:
                                return j, (sig + text[:column]).strip(), j + 1
                            stack.pop()
                            closed = closed or (char == ")" and not stack)
                        elif char == ";" and not stack:
                            return j, sig + text[: column + 1].lstrip(), j + 1
                    sig += f"{text.strip()} "
                    following = code_lines[j + 1].lstrip() if j + 1 < len(lines) else ""
                    if (
                        closed
                        and not stack
                        and not _TS_TYPE_EXPECTED_RE.search(sig)
                        and not following.startswith(_TS_TYPE_CONTINUES)
                        and not _JS_ASSIGNMENT_END_RE.search(sig)
                    ):
                        return j, sig.strip(), j + 1
                return idx, code_lines[idx].strip(), idx + 1

            def members_on(class_id, idx, code, start, lineno):
                """Read the members written on ``code``, line ``idx``, from
                column ``start``, filed under ``class_id``, and make each a
                node. Returns the member the line leaves open, the spans of
                the values of the fields on it, the last line of a field
                the line leaves open, 0 where there is none, and each member
                as its ID and the column its name ends at."""
                raw = text_lines[idx]
                written, fields, open_at = _members_on_line(code, raw, start)
                spans = [(*field, class_id) for field in fields]
                written = [(*member, lineno) for member in written]
                sig_until = 0
                if open_at is not None:
                    member_end = _statement_end(code_lines, idx, open_at) + 1
                    found = _TS_LINE_MEMBER_RE.match(code, open_at)
                    prop = _class_property(code, open_at, raw)
                    if found:
                        filed = _member_filed(found, raw)
                        written.append(
                            (filed, found.end(2), code[open_at:], member_end)
                        )
                    elif prop and _function_value(raw[prop[2] :]):
                        written.append((prop[0], prop[1], raw[prop[2] :], member_end))
                    else:
                        sig_until = member_end
                        if prop:
                            spans.append((prop[2], len(code), class_id))
                open_member = None
                made = []
                for member, name_end, text, member_end in written:
                    mid = f"{class_id}::{member}"
                    return_type, param_types, typed = self._extract_ts_types(text)
                    self.builder.add_node(
                        mid,
                        "Method",
                        name=member,
                        signature=self._build_ts_signature(
                            member, param_types, return_type
                        ),
                        lineno=lineno,
                        end_lineno=member_end,
                        token_count=(member_end - lineno + 1) * 10,
                        docstring="",
                        return_type=return_type,
                        param_types=param_types,
                        signatures=self._overloads(mid, typed),
                    )
                    self.builder.add_edge(class_id, mid, "DEFINES")
                    made.append((mid, name_end))
                    if member_end > lineno:
                        open_member = (mid, lineno, member_end, text)
                return open_member, spans, sig_until, made

            def field_property(idx):
                """The class property written on line ``idx`` and the text of
                its value over the lines the statement spans, with the index
                of its last line; None where the line writes no property."""
                prop = _class_property(code_lines[idx])
                if prop is None:
                    return None
                end = _statement_end(code_lines, idx)
                value = " ".join(
                    [text_lines[idx][prop[2] :]]
                    + [t.strip() for t in text_lines[idx + 1 : min(end, idx + 40) + 1]]
                )
                return prop, value, end

            # A line of the signature of a definition above, whose calls are
            # its: it declares nothing of its own, opens no nameless function
            # and holds none of the definition's `return`s.
            continuing = lineno <= signature_until
            # A `for` or `catch` head the line above leaves open holds the
            # line's binding, a `catch`'s as written, and the line declares
            # nothing else.
            above = i - 1
            while above >= 0 and not code_lines[above].strip():
                above -= 1
            held_open = above >= 0 and _JS_FOR_HEAD_RE.search(code_lines[above])
            if held_open and _js_member_name(code_lines, above, held_open.start()):
                held_open = None
            if held_open:
                held_open = (above, code_lines[above].rindex("("), held_open[1])
            # The class the line declares. An expression whose body closes
            # on the line is a const's value, read where the const is; one
            # that goes on over lines is the class's line, not the const's,
            # unless it is written inside brackets, where the line declares
            # what it does without it.
            cx = None if continuing else re_class.search(code_lines[i])
            if continuing or held_open or (cx and cx.group("enclosed") is None):
                pass
            elif (match := re_const.search(line)) and (
                match.group(3) is not None or not read_arrow(i)[1]
            ):
                name, value_type, val = match.groups()
                if parent_id == file_id and line[:1].isspace():
                    # In a callback, where another may declare the name too.
                    value_type = None
                doc = self.get_leading_docstring(lines, i)
                cid = f"{parent_id}::{name}"
                claim(cid)
                holder = cid, match.end(1)
                if value_type and cid in namespaces:
                    # A namespace and a value of one name are one: the
                    # value's type gives the members the namespace has not.
                    namespaces[cid]["value_type"] = value_type
                if val and (held := _arrow_function(line)):
                    valued = self._arrow_typed(held[1], line)
                    arrow_id = cid
                    own_defs.append((cid, match.end(1)))
                self.builder.add_node(
                    cid,
                    "Constant",
                    name=name,
                    value=val
                    and line[
                        match.start(3) : _value_end(code_lines[i], match.start(3))
                    ],
                    value_type=value_type,
                    lineno=lineno,
                    docstring=doc,
                    signatures=valued and self._overloads(cid, valued),
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")
                if parent_id not in class_ids and re.search(r"=\s*\{\s*$", line):
                    # The functions it holds are its members.
                    object_ids.add(cid)
                    enter(cid, lineno, self._find_block_end(code_lines, i))
                elif parent_id not in class_ids and re.search(r"=\s*\[\s*$", line):
                    elements_of = cid, _statement_end(code_lines, i) + 1
                elif parent_id not in class_ids and re.search(r"=\s*\[\s*\{\s*$", line):
                    # Its first element, an object, owns the functions it holds.
                    last = self._find_block_end(code_lines, i)
                    element = self._nameless_owner(
                        cid,
                        f"object@{self.source_line(lineno)}",
                        lineno,
                        last,
                        "object",
                    )
                    object_ids.add(element)
                    enter(element, lineno, last)
            elif parent_id not in class_ids and (
                opened := _TS_OBJECT_OPENS_RE.match(code_lines[i])
            ):
                # An object `export default` gives, a property's or an
                # element's: its functions are its members. A property's is
                # named as source reaches it, through its key; the others no
                # name source writes.
                last = self._find_block_end(code_lines, i)
                key = opened["key"] and parent_id in object_ids and opened["key"]
                if key:
                    oid = f"{parent_id}::{_ts_unquote(key)}"
                    self.builder.add_node(
                        oid, "Constant", name=_ts_unquote(key), lineno=lineno
                    )
                    self.builder.add_edge(parent_id, oid, "DEFINES")
                elif opened["key"] is None and (opened["default"] or array):
                    held = elements_of and lineno <= elements_of[1] and elements_of[0]
                    oid = self._nameless_owner(
                        held or parent_id,
                        f"object@{self.source_line(lineno)}",
                        lineno,
                        last,
                        "object",
                    )
                    if opened["default"]:
                        entry = ("default", "", oid.rpartition("::")[2])
                        self.builder.reexports.setdefault(file_id, []).append(entry)
                else:
                    oid = None
                if oid:
                    object_ids.add(oid)
                    enter(oid, lineno, last)
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
                    enter(cid, lineno, self._find_block_end(code_lines, i))
            elif match := _TS_NAMESPACE_RE.match(code_lines[i]):
                # A namespace owns what it declares, which source names
                # through it; `A.B` is `B` in `A`.
                last = self._find_block_end(code_lines, i)
                owner = parent_id
                for part in match[1].split("."):
                    nested = f"{owner}::{part}"
                    if nested not in self.builder.node_registry:
                        self.builder.add_node(
                            nested,
                            "Class",
                            name=part,
                            kind="namespace",
                            lineno=lineno,
                            end_lineno=last,
                            token_count=(last - lineno + 1) * 10,
                            docstring=self.get_leading_docstring(lines, i),
                        )
                        self.builder.add_edge(owner, nested, "DEFINES")
                        namespaces[nested] = self.builder.nodes[-1]["metadata"]
                    owner = nested
                enter(owner, lineno, last)
            elif match := re_interface.search(line):
                name = match.group(1) or match.group(2)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig, end_lineno = consume_sig(i)
                cid = f"{parent_id}::{name}"
                if cid in value_ids:
                    # The function or variable of that name has this ID.
                    cid = f"{parent_id}::interface_{name}"
                else:
                    interface_ids.add(cid)
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    cid,
                    "Class",
                    name=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
                    **_generic_metadata(
                        _bounded_parameters(line[match.end(1 if match[1] else 2) :])
                    ),
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")
                class_ids.add(cid)
                type_bodies.add(cid)
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
                j, full_sig, end_lineno = consume_sig(i)
                fid = f"{parent_id}::{name}"
                return_type, param_types, typed = self._extract_ts_types(full_sig)
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
                    signatures=self._overloads(fid, typed),
                )
                self.builder.add_edge(parent_id, fid, "DEFINES")
                own_defs.append((fid, match.end(1)))
                enter(fid, lineno, end_lineno, full_sig)
                signature_until = j + 1  # its last line, as a line number
            elif (
                _TS_VARIABLE_RE.match(line) or parent_id in object_ids or object_id
            ) and (arrow := (statement := read_arrow(i))[1]):
                name, parameters = arrow
                j = statement[0]
                docstring = self.get_leading_docstring(lines, i)
                fid = f"{object_id or parent_id}::{name}"
                # A block is left open where the statement stops; an
                # expression body ends with the statement.
                last = code_lines[j]
                end_lineno = (
                    self._find_block_end(code_lines, j)
                    if last.count("{") > last.count("}")
                    else _statement_end(code_lines, i) + 1
                )
                # Arrow functions don't have easily extractable signatures
                signature = f"const {name} = (...) => ..."
                token_count = (end_lineno - lineno + 1) * 10
                claim(fid)
                typed = self._arrow_typed(parameters, line)
                if end_lineno == lineno or not _JS_BLOCK_FUNCTION_RE.search(parameters):
                    # A body that is an expression gives what it reads, on
                    # this line or one below; a block's `return`s are read
                    # where they are.
                    valued = typed
                self.builder.add_node(
                    fid,
                    "Function",
                    name=name,
                    signature=signature,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
                    signatures=self._overloads(fid, typed),
                )
                self.builder.add_edge(object_id or parent_id, fid, "DEFINES")
                named = _TS_VARIABLE_RE.match(line) or _TS_PROPERTY_RE.match(line)
                if named and _ts_unquote(named.group(1)) == name:
                    own_defs.append((fid, named.end(1)))
                enter(
                    fid,
                    object_scope if object_scope is not None else lineno,
                    end_lineno,
                    parameters,
                )
                arrow_id = fid
                signature_until = j + 1  # its last line, as a line number
            elif parent_id in object_ids and (match := re_held.match(line)):
                # The object holds by its key what the name is, and spread
                # what that holds.
                spread, key, name = match.groups()
                held = ("*" if spread else key or name, "", name)
                self.builder.reexports.setdefault(parent_id, []).append(held)
            elif (
                parent_id in type_bodies
                and (written := _ts_type_member(code_lines, i))
                and _braces_open(
                    code_lines,
                    next(c[1] for c in reversed(self._contexts) if c[0] == parent_id)
                    - 1,
                    0,
                    i,
                    -1,
                )
                == 1
            ):
                # A property of a function type is a method the type
                # declares; one of an object type a field of that type.
                name, full_sig = written
                if full_sig.startswith("{"):
                    self.builder.fields.setdefault(parent_id, {})[name] = full_sig
                else:
                    mid = f"{parent_id}::{name}"
                    return_type, param_types, typed = self._extract_ts_types(full_sig)
                    self.builder.add_node(
                        mid,
                        "Method",
                        name=name,
                        signature=self._build_ts_signature(
                            name, param_types, return_type
                        ),
                        lineno=lineno,
                        end_lineno=lineno,
                        token_count=10,
                        docstring=self.get_leading_docstring(lines, i),
                        return_type=return_type,
                        param_types=param_types,
                        signatures=self._overloads(mid, typed),
                    )
                    self.builder.add_edge(parent_id, mid, "DEFINES")
            elif (
                parent_id in class_ids
                and not re_method.search(line)
                and (fv := field_property(i)) is not None
                and not _function_value(fv[1])
            ):
                # A field's value: the lines it goes on over declare nothing.
                signature_until = fv[2] + 1
            elif (
                "::" in parent_id
                and parent_id != file_id
                and (
                    enclosing := (
                        parent_id
                        if parent_id in class_ids
                        else self.enclosing_scope(parent_id)
                    )
                )
                in class_ids
                and (close_col := _member_close_column(code_lines[i])) is not None
                # The `}` closes a member where it leaves the class's body
                # alone open, a block of a member's body where it does not.
                and _braces_open(
                    code_lines,
                    next(c[1] for c in reversed(self._contexts) if c[0] == enclosing)
                    - 1,
                    class_at.get(enclosing, 0),
                    i,
                    close_col,
                )
                == 1
            ):
                # A member written on the line that closes the member before
                # it is the class's, read as on its own line.
                open_member, field_spans, sig_until, made = members_on(
                    enclosing, i, code_lines[i], close_col, lineno
                )
                own_defs.extend(made)
                if sig_until:
                    signature_until = sig_until
                if open_member:
                    enter(*open_member)
            # A callback's body, as the file's, holds no members of its own: a
            # method written there is one of an object literal. Inside an
            # object that owns what it holds, a member is one of its own.
            elif (
                object_id
                or (
                    "::" in parent_id
                    and parent_id != file_id
                    and parent_id not in callback_ids
                )
            ) and (
                member := _member_written(
                    line,
                    parent_id in class_ids,
                    re_method,
                    (fv := field_property(i)) and fv[1],
                )
            ):
                name, name_start, name_end, assigned = member
                j, full_sig, end_lineno = consume_sig(i)
                # `name(` declares a method directly in a class body, or where
                # a block follows its parameter list; otherwise it is a call.
                # Outside a class a keyword names no member: the block after
                # `catch (` is that of a `try` statement, and `function (`
                # starts a function expression.
                arrow = parent_id in class_ids and re.search(
                    r"\)\s*(?::[^=;]+)?=>", full_sig
                )
                if name not in ["if", "for", "while", "switch"] and (
                    arrow
                    if assigned
                    else parent_id in class_ids
                    or (name not in _JSTS_CONTROL_KEYWORDS and _opens_body(full_sig))
                ):
                    declared_at = name_start
                    docstring = self.get_leading_docstring(lines, i)
                    mid = f"{object_id or parent_id}::{name}"
                    return_type, param_types, typed = self._extract_ts_types(full_sig)
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
                        signatures=self._overloads(mid, typed),
                    )
                    self.builder.add_edge(object_id or parent_id, mid, "DEFINES")
                    own_defs.append((mid, name_end))
                    value = assigned and _class_property(full_sig)
                    enter(
                        mid,
                        object_scope if object_scope is not None else lineno,
                        end_lineno,
                        full_sig[value[2] :] if value else full_sig,
                    )
                    signature_until = j + 1  # its last line, as a line number
                elif assigned and parent_id in class_ids:
                    # A field whose parenthesised value is no function: its
                    # lines declare nothing, its calls the class's.
                    signature_until = _statement_end(code_lines, i) + 1
            # Handle getter/setter methods: get name() or set name(value)
            # Only match when the parent is a Class, or an object that owns
            # what it holds — one returned, passed to a call, or a constant's
            elif (
                object_id
                or parent_id in object_ids
                or (
                    "::" in parent_id
                    and parent_id != file_id
                    and parent_id not in callback_ids
                    and self._is_class_context(parent_id)
                )
            ) and (match := re_getter_setter.search(line)):
                accessor_type = match.group(1)  # "get" or "set"
                name = _ts_unquote(match.group(2))
                declared_at = match.start(2)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig, end_lineno = consume_sig(i)
                # An accessor is filed as source writes it, `get name`, which
                # tells a pair apart and is no name a method can have.
                mid = f"{object_id or parent_id}::{accessor_type} {name}"
                return_type, param_types, typed = self._extract_ts_types(full_sig)
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
                    signatures=self._overloads(mid, typed),
                )
                self.builder.add_edge(object_id or parent_id, mid, "DEFINES")
                own_defs.append((mid, match.end(2)))
                enter(
                    mid,
                    object_scope if object_scope is not None else lineno,
                    end_lineno,
                    full_sig,
                )
                signature_until = j + 1  # its last line, as a line number
            elif match := re_import.search(line):
                self.builder.add_edge(file_id, f"MODULE:{match.group(2)}", "IMPORTS")

            if cx:
                # The declarator names the class it is given. An
                # expression's own name is visible only inside it: one given
                # to anything else has no name that source can write.
                name = (
                    cx.group("declarator")
                    or cx.group("plain")
                    or f"class@{self.source_line(lineno)}"
                )
                docstring = self.get_leading_docstring(lines, i)
                # The statement may start before the class and go on past it:
                # its heritage and its body are read from its keyword.
                j, full_sig, end_lineno = consume_sig(i, cx.end())
                body = _class_body_open(code_lines[i], cx.end())
                # What its lines are all in holds the class: a definition
                # the line declares, not one it closes.
                holders = reaching(end_lineno)
                held_by = holders[0][0]
                for node_id, name_end in own_defs:
                    span = _own_line_span(code_lines[i], name_end)
                    if span and span[0] <= cx.start() < span[1]:
                        held_by = node_id
                cid = f"{held_by}::{name}"
                token_count = (end_lineno - lineno + 1) * 10
                self.builder.add_node(
                    cid,
                    "Class",
                    name=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
                    **_generic_metadata(_bounded_parameters(full_sig)),
                )
                self.builder.add_edge(held_by, cid, "DEFINES")
                class_ids.add(cid)
                class_at[cid] = cx.end()
                # The methods and accessors written on the class's own line,
                # and the member or field the line leaves open, which ends
                # with its statement.
                # A field's value is evaluated with the class's instance or,
                # static, with the class: the class's.
                open_member = None
                # Its lines declare its members, whatever statement of the
                # line goes on over them.
                signature_until = min(signature_until, lineno)
                if body is not None:
                    open_member, field_spans, sig_until, made = members_on(
                        cid, i, code_lines[i], body, lineno
                    )
                    own_defs.extend(made)
                    if sig_until:
                        signature_until = sig_until
                # Generic arguments go first: `class A<T extends B> extends C<T>`
                # inherits from C.
                # The base is an expression: a name a type has, or a value
                # no name stands for, a call's or a cast's, as written.
                heritage = _strip_generics(full_sig)
                if extended := _TS_EXTENDS_RE.search(heritage):
                    base = " ".join(extended[1].split())
                    # A parameter of a definition around the class holds a
                    # value: its name is no type's.
                    passed = any(base.split(".")[0] in params for *_, params in holders)
                    self.builder.add_edge(
                        cid,
                        f"REF:{base}",
                        "INHERITS",
                        {"origin": f"={base}"} if passed else None,
                    )
                enter(cid, lineno, end_lineno)
                if open_member:
                    enter(*open_member)
                # A body that opens on this line has had what follows it
                # read; one on a later line is read from that line.
                if body is None:
                    i = j

            # Configuration patterns - capture as Configuration nodes
            # These are important setup calls that often need to be found during investigation
            if match := re_interceptor.search(line):
                client_name = match.group(1)
                interceptor_type = match.group(2)
                config_name = f"{client_name}_{interceptor_type}_interceptor"
                config_id = f"{file_id}::{config_name}"
                # Find the callback's end by tracking braces
                end_lineno = (
                    self._find_block_end(code_lines, i)
                    if "{" in line or "=>" in line
                    else lineno
                )
                # Look for next few lines if arrow function continues
                if end_lineno == lineno:
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if "{" in lines[j] or "=>" in lines[j]:
                            end_lineno = self._find_block_end(code_lines, j)
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
                        self._find_block_end(code_lines, i) if "{" in line else lineno
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
                    self._find_block_end(code_lines, i)
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

            # Calls. Module-level calls are sourced at the file.
            code = code_lines[lineno - 1]
            # The type arguments of a call are no part of its chain.
            code = _JS_TYPE_ARGUMENTS_RE.sub(lambda m: " " * len(m[0]), code)
            if _TS_VARIABLE_RE.match(code):
                # `const name = value as Type` casts nothing: the value
                # the line gives the name is the one written.
                code = _JS_AS_TYPE_RE.sub(lambda m: " " * len(m[0]), code)
            # A module's top level takes in the bodies of callbacks, whose
            # variables are not the module's.
            scope = None if parent_id == file_id else parent_id
            self.begin_line(code, lineno, scope, line)
            # A variable has what the line that declared it gave it:
            # nothing where that gave it no type, whatever an earlier
            # declaration of its name did.
            if name := untyped[0]:
                entries = self.declared.get(name, [])
                kept = [e for e in entries if e[0] != untyped[1] or e[1]]
                if not any(e[0] == untyped[1] for e in kept):
                    kept.append((untyped[1], None, False))
                self.declared[name] = kept
            untyped = ("", 0)
            if literal := _JS_LITERAL_RE.search(code):
                builtin = _JS_LITERAL_TYPES.get(literal["literal"][0], "string")
                if builtin == "Array":
                    # Its elements are read once its calls are.
                    array = f"[{len(self.builder.literals)}]", 0
                    self.builder.literals[array[0]] = []
                    builtin = f"Array<{array[0]}>"
                self.declared.setdefault(literal["name"], []).append(
                    (lineno, f"{builtin}()", not line[:1].isspace())
                )
            elif binding := self.binding_re.search(code):
                untyped = (binding["name"], lineno)
            since = self.context_at(lineno)[1]
            # The definition the line is in, before it enters those the
            # line opens
            scope = reaching(lineno)[0]
            owner = parent_id.split("::")[-2] if parent_id.count("::") > 1 else None
            self._closures = [
                (
                    found.start("closure"),
                    found["params"] or found["param"] or "",
                    found.end(),
                )
                for found in _JS_CLOSURE_RE.finditer(code)
                # A function with a name is an expression where it is a
                # value, else a declaration.
                if not found["named"]
                or _JS_VALUE_START_RE.search(
                    code, max(_run_start(code, found.start()) - 7, 0), found.start()
                )
            ]
            self._held_closures = [
                (found.start("closure"), found["params"] or found["param"] or "")
                for found in _JS_PROPERTY_CLOSURE_RE.finditer(code)
            ]
            self._bodies = []
            # A nameless function passed as an argument, whose block body
            # opens here and goes on over later lines, is a scope of its
            # own: the locals it declares are its, not the enclosing
            # definition's, and each such function its own. One that
            # opens after another on the line is written inside it.
            within = parent_id
            # The definitions the line declares, by their own columns
            own_spans = [
                (node_id, span)
                for node_id, name_end in own_defs
                if (span := _own_line_span(code, name_end))
            ]
            for bar, _params, end in self._closures:
                # The innermost of those that holds it holds it.
                holding = [
                    node_id
                    for node_id, (start, stop) in own_spans
                    if start <= bar < stop
                ]
                if holding and not within.startswith(f"{holding[-1]}::"):
                    within = holding[-1]
                text = code[bar:end].rstrip()
                if text.endswith("{"):
                    opened = end - 1
                elif text.endswith("=>") and code[end:].lstrip().startswith("{"):
                    opened = end + len(code[end:]) - len(code[end:].lstrip())
                else:
                    continue  # an expression body, or one over later lines
                if _block_close(code, opened) is not None:
                    continue  # the block closes on this line
                while callbacks and callbacks[-1][0] < lineno:
                    callbacks.pop()
                outer = callbacks[-1][1] if callbacks else 0
                block_end = self._find_block_end(code_lines, i, opened)
                if block_end <= lineno:
                    continue
                if self.paren_around(bar) is None and len(self._open_calls) <= outer:
                    # Not an argument: the lines of its body are those
                    # of the definition around it, or of the constant
                    # whose value holds it, and read its parameters too.
                    context = (*reaching(block_end)[0][:2], block_end - 1)
                    if holder and ";" not in code[holder[1] : bar]:
                        context = (holder[0], lineno, block_end - 1)
                    self.enter_context(*context, _ts_params(code[bar : opened + 1]))
                else:
                    callback_id = within = self._nameless_owner(
                        within,
                        f"callback@{self.source_line(lineno)}",
                        lineno,
                        block_end,
                        "callback",
                    )
                    callbacks.append(
                        (block_end - 1, len(self._open_calls) + len(self._opens))
                    )
                    callback_ids.add(callback_id)
                    context = (callback_id, since, block_end - 1)
                    span = (
                        (i, opened),
                        close_at.get((i, opened), (block_end - 1, None)),
                    )
                    enter(*context, code[bar : opened + 1], span)
                closed_after.add(context)
                # A function expression's own name is a variable of its
                # body.
                if named := re.match(r"function\b\s*\*?\s*([\w$]+)", code[bar:]):
                    declared_names.setdefault(context, {}).setdefault(
                        named[1], []
                    ).append((lineno, block_end))
            # The declarators of a statement after its first, on this line
            # or one the statement goes on to, are definitions of their
            # own, as its first is and where it is: a `const` a constant,
            # a `let` or `var` one holding a function that function. One
            # holding a function takes its calls.
            if not continuing and (
                head := _TS_DECLARATION_HEAD_RE.match(code_lines[i])
            ):
                for at, *declarator in _declarators(code_lines, i, head.end()):
                    later_declarators.setdefault(at, []).append(
                        (parent_id, head[1], *declarator)
                    )
            # The variables the line declares, each with the last line
            # it is in scope on: a `var` to the end of the function it
            # is in, any other to the end of its block, the binding of a
            # `for` or a `catch` to the end of the body. The name of a
            # function or class expression is in scope only inside it.
            bindings = []
            for found in _JS_BINDING_HEAD_RE.finditer(code):
                if _js_member_name(code_lines, i, found.start()):
                    continue
                head = None
                if not found[1] or _JS_FOR_HEAD_RE.search(
                    code,
                    _run_start(code, found.start(), "(forawaitcatch"),
                    found.start(),
                ):
                    head = (i, code.rindex("(", 0, found.end()))
                elif held_open and not _run_start(code, found.start()):
                    head = held_open[:2]
                bindings.append((found[1] or "catch", found.end(), head))
            written = code.lstrip()
            if held_open and held_open[2] and re.match(r"[\w$\[{]", written):
                bindings.append(("catch", len(code) - len(written), held_open[:2]))
            names = []
            for kind, at, head in bindings:
                last = in_scope(i, at, kind, head)
                written = code[at:]
                bound = (
                    [name for name, _ in _js_bindings(written)]
                    if written[0] in "[{"
                    else [re.match(r"[\w$]+", written)[0]]
                )
                names.extend((name, last) for name in bound)
            declared_here = later_declarators.pop(i, ())
            for scope_id, kind, later, name_end, value_at in declared_here:
                # A later declarator is typed by its value as the first is.
                value_end = _value_end(code_lines[i], value_at)
                typed_as = self._expression_origin(
                    code, value_at, value_end, lineno, since, owner
                )
                if typed_as:
                    self.declared.setdefault(later, []).append(
                        (lineno, typed_as, self._binds_file(later))
                    )
                held = _function_value(line[value_at:])
                constant = kind == "const"
                if not (constant or held):
                    names.append((later, in_scope(i, name_end, kind)))
                    continue
                lid = f"{scope_id}::{later}"
                claim(lid)
                held_typed = held and self._arrow_typed(held, line)
                value = line[value_at : _value_end(code_lines[i], value_at)]
                self.builder.add_node(
                    lid,
                    "Constant" if constant else "Function",
                    name=later,
                    value=value.strip(),
                    value_type=None,
                    lineno=lineno,
                    docstring="",
                    signatures=held_typed and self._overloads(lid, held_typed),
                )
                self.builder.add_edge(scope_id, lid, "DEFINES")
                if held:
                    own_defs.append((lid, name_end))
            if closed := {name for name, last in names if last == lineno}:
                self._closed_names[lineno] = closed
            # A name a function's body declares is the function's own:
            # a call of it calls the value the name holds. A line that
            # defines a function may declare the names in that, and one
            # a `require` binds is the module's.
            if scope[0] not in (file_id, *class_ids) and not (
                continuing or own_defs or i in required
            ):
                for name, last in names:
                    declared_names.setdefault(scope[:3], {}).setdefault(
                        name, []
                    ).append((lineno, last))
            typed = _TS_TYPED_VALUE_RE.search(code)
            calls = []  # (edge index, column of its `(`) of each on the line
            # A definition on this line takes the calls its parameters
            # and the first line of its body enclose, which run when it
            # is called, where the node is its own: one that an earlier
            # definition of the name already holds is not. A decorator
            # or computed key leading the line runs where the class or
            # object is defined, not in the member it leads.
            added = {node["id"] for node in self.builder.nodes[nodes_before:]}
            spans = field_spans + [
                (*span, node_id)
                for node_id, name_end in own_defs
                if node_id in added and (span := _own_line_span(code, name_end))
            ]
            if (parent_id in class_ids or parent_id in object_ids) and (
                region := _definition_time_region(code)
            ):
                spans.append((*region, self.enclosing_scope(parent_id)))
            for match in re_call_candidate.finditer(code):
                if match.start(1) == declared_at:
                    continue
                # An optional call (`a?.b?.()`) is a call; only its
                # receiver may be nullish.
                callee = match.group(1).replace("?.", ".")
                # `function name(` names the function it defines.
                if _JS_FUNCTION_HEAD_RE.search(
                    code,
                    max(_run_start(code, match.start(1), "*") - 8, 0),
                    match.start(1),
                ):
                    continue
                # `name(params) {` at the start of a line defines a
                # method of an object literal.
                if (
                    _TS_METHOD_MODIFIERS_RE.fullmatch(code, 0, match.start(1))
                    and "." not in callee
                    and _opens_body(code[match.start(1) :])
                ):
                    continue
                # JS/TS control-flow keywords look like calls (`if(x)`,
                # `while (y)`). Reject them by name — language general,
                # not codebase specific. After a `.` one names a member.
                member = _js_member_name(code_lines, i, match.start(1))
                if callee in _JSTS_CONTROL_KEYWORDS and not member:
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
                    # A call of a value a parameter or a variable holds,
                    # unless the name is a member of an expression.
                    origin = None
                    if not member:
                        origin = value_of(callee, lineno, since, owner)
                    # A literal is a value of its built-in type.
                    quote = _JS_LITERAL_RECEIVER_RE.search(
                        code,
                        max(_run_start(code, match.start(1), ".") - 1, 0),
                        match.start(1),
                    )
                    if quote:
                        builtin = _JS_LITERAL_TYPES.get(quote[1], "string")
                        origin = f"{builtin}().{callee}"
                    self._bind_closures(code, i, lines, match.start(1))
                    # `a.b.name()` is called on the property `b` of `a`.
                    receiver, _, called = callee.rpartition(".")
                    start = match.start(1)
                    if "." in receiver and not member:
                        span = (start, start + len(receiver))
                        held = self._expression_origin(
                            code, *span, lineno, since, owner
                        )
                        origin = origin or (held and f"{held}.{called}")
                    if callee.startswith("this."):
                        # `this` in an object literal that is the
                        # argument of a call is the `ThisType` of the
                        # parameter that holds it.
                        held = self._this_in_object(code, match.start(1))
                        if held:
                            origin = f"{held}.{called}"
                    source = owner_at(i, match.start(1)) or parent_id
                    for span_start, span_end, node_id in spans:
                        if span_start <= match.start(1) < span_end:
                            source = node_id
                    self.add_call(
                        source,
                        callee,
                        lineno,
                        since,
                        match.start(1),
                        paren,
                        origin,
                    )
                    # `new Type<arguments>(` constructs the type it names.
                    made = self.builder.edges[-1]["metadata"]
                    written = line[start + len(callee) : paren].strip()
                    if "origin" not in made and _JS_NEW_RE.search(
                        code, max(_run_start(code, start) - 3, 0), start
                    ):
                        made["origin"] = f"new {callee}{written}"
                    elif written:
                        made["site"] = {"types": written}
                    calls.append((len(self.builder.edges) - 1, paren, start))
                    if typed and start == typed.end():
                        initialises = (calls[-1][0], typed["type"])
            if initialises:
                # The type is that of the call's result where the value
                # ends with the call.
                ended = self._ends_value(code, initialises[0])
                if ended:
                    site = self.builder.edges[initialises[0]]["metadata"]
                    site.setdefault("site", {})["declared"] = initialises[1]
                if ended is not None:
                    initialises = None
            if returning and self._ends_value(code, returning[0]) is not None:
                if not self._ends_value(code, returning[0]):
                    returning[1][returning[2]] = None
                returning = None
            # What a function declared without a return type returns:
            # the value of each `return` of its body, or of an arrow
            # function the expression it is.
            returned = (self._signatures.get(parent_id) or [{}])[-1].get("returned")
            returns = (
                not continuing
                and lineno > nameless.get(parent_id, 0)
                and len(re.findall(r"\breturn\b", code))
            )
            found = None
            # Whether the value the line gives is an expression: an
            # arrow's `{` opens a block unless its body is parenthesised,
            # which an object expression must be.
            expression = returns == 1
            if valued and code.count("=>") == 1:
                returned = valued["returned"]
                found = re.search(r"=>\s*(?P<new>new\s+)?", code)
                expression = bool(found) and code[found.end() :].lstrip().startswith(
                    "("
                )
            elif returns == 1:
                found = _JS_RETURN_RE.match(code)
            if returned is not None and (found or returns):
                value = code[found.end() :].strip(" ;") if found else ""
                first = [
                    index for index, _, start in calls if found and start == found.end()
                ]
                record = None
                if value == "this" and not found["new"]:
                    record = "", "self"
                elif first:
                    made = self.builder.edges[first[0]]
                    origin = made["metadata"].get("origin")
                    ended = self._ends_value(code, first[0])
                    # One call, whose receiver is no other call's result.
                    read = self._read_as(origin) if origin else origin
                    if "#" not in (read or "") and ended is not False:
                        record = made["target"][4:], read
                    if record and ended is None:
                        returning = first[0], returned, len(returned)
                elif re.fullmatch(r"[\w$]+", value):
                    held = self._read_as(
                        self.call_origin(f"{value}.x", lineno, since, owner)
                    )
                    if held and "#" not in held:
                        record = "", held[:-2]
                elif value.lstrip("(").startswith("{") and expression:
                    # An object: its properties are read where the
                    # function is written, and the functions among them
                    # are members of the object itself.
                    opened = code.index("{", found.end())
                    key = f"[{len(self.builder.literals)}]"
                    # An arrow's own node holds the object its body is,
                    # which the arrow may declare on this line.
                    holder = arrow_id or parent_id
                    held, object_end = self._object_held(
                        code_lines, i, opened, since, owner
                    )
                    self.builder.literals[key] = [held]
                    self.builder.literal_scopes[key] = holder
                    record = "", key
                    # An object that holds a function written in it owns
                    # that function; one that holds none is no owner.
                    if any(str(value).startswith("=") for value in held.values()):
                        object_id = self._nameless_owner(
                            holder,
                            f"object@{self.source_line(lineno)}",
                            lineno,
                            object_end,
                            "object",
                        )
                        self.builder.literal_scopes[key] = object_id
                        returned_objects.append(
                            (holder, lineno, object_end, object_id, since)
                        )
                    # A `return` on a line of the object is one of a
                    # function written in it.
                    nameless[holder] = max(nameless.get(holder, 0), object_end)
                returned.append(record)
            # An object literal a call is passed, that its line leaves
            # open over later lines and that holds a function, owns the
            # functions written in it, a name no source can write. Its
            # properties are read as the ones of the object a function
            # returns are. One that opens after another on the line is
            # written inside it.
            passed, before = _passed_objects(
                code, brackets, before, {paren for _, paren, _ in calls}
            )
            for opened in passed:
                held, object_end = self._object_held(
                    code_lines, i, opened, since, owner
                )
                if not any(str(value).startswith("=") for value in held.values()):
                    continue
                opened_here = returned_objects and returned_objects[-1][1] == lineno
                # Its lines are those of the definition the next line is
                # in: one the line opens, or the one around one it closes.
                scope_id = self.context_at(lineno + 1)[0]
                returned_objects.append(
                    (
                        scope_id,
                        lineno,
                        object_end,
                        self._nameless_owner(
                            returned_objects[-1][3] if opened_here else scope_id,
                            f"object@{self.source_line(lineno)}",
                            lineno,
                            object_end,
                            "object",
                        ),
                        since,
                    )
                )
            if (
                not continuing
                and lineno > nameless.get(parent_id, 0)
                and _JS_BLOCK_FUNCTION_RE.search(code)
                and self.context_at(lineno + 1)[0] == parent_id
            ):
                nameless[parent_id] = self._find_block_end(code_lines, i)
            for held, name, start, end in self._held_later.pop(i, ()):
                # A property an object read on an earlier line is given
                # on this one holds what this line's calls give it.
                held[name] = self._expression_origin(
                    code, start, end, lineno, since, owner
                )
            self._bind_closures(code, i, code_lines, len(code))
            self._name_results(code, lineno, since, owner)
            self._bind_patterns(code, i, code_lines, since, owner)
            for index, paren, _ in calls:
                # What a call closed on its line is passed, unless it
                # spreads a value over several arguments.
                closed = self._pairs.get(paren)
                if closed and "..." not in code[paren:closed]:
                    site = self.builder.edges[index]["metadata"]
                    site.setdefault("site", {})["arguments"] = self._elements(
                        code, paren + 1, lineno, since, owner
                    )
            if array:
                # One element to a line, or all on the line of the `[`;
                # one that goes on over lines leaves them unknown.
                start = literal.end("name") if literal and not array[1] else 0
                opened = code.index("[", start) + 1 if start else 0
                held = self.builder.literals[array[0]]
                if held is not None:
                    held += self._elements(code, opened, lineno, since, owner)
                depth = array[1] + sum(
                    (char in "([{") - (char in ")]}")
                    for char in code[max(opened - 1, 0) :]
                )
                if depth > 1:
                    self.builder.literals[array[0]] = None
                array = (array[0], depth) if depth > 0 else None
            i += 1

        # What the resolver reads by the ID a node ends up with: its
        # signatures, and each field's type for each class or interface of
        # that name in the file.
        for node in self.builder.nodes[self._first_node :]:
            if typed := node["metadata"].pop("signatures", None):
                self.builder.signatures[node["id"]] = typed
            for (owner, field), held in self.fields.items():
                if node["metadata"].get("name") == owner and len(held) == 1:
                    fields = self.builder.fields.setdefault(node["id"], {})
                    fields.setdefault(field, next(iter(held))[:-2])

        self.restore_source_lines()

    def _extract_ts_types(self, full_sig: str) -> tuple:
        """Extract return type and parameter types from TypeScript function signature.

        Returns: (return_type: str, param_types: list[str], typed: dict),
        the last the signature as the resolver reads it: the bound of each
        type parameter, the parameters and the return type as written, and
        the types of the parameters of each function the signature takes,
        by the position of the argument it is.

        Handles common patterns:
        - function foo(x: string): number
        - async function bar(): Promise<void>
        - (a: Type, b: Type) => ReturnType
        """
        return_type = "void"
        param_types = []
        typed = {"generics": {}, "params": [], "returns": "", "callbacks": {}}
        typed["body"] = full_sig.rstrip().endswith("{")
        # What each `return` of the body gives, as the call loop reads them;
        # None where what is returned is wrapped: in a promise or a generator
        typed["returned"] = [] if typed["body"] else None
        callbacks = typed["callbacks"]

        try:
            # The parameter list is the first `(...)` outside type
            # parameters. The return type follows it, up to the `{` or `;`
            # the signature ends with or the `=>` of an arrow function,
            # which unlike that of a function type follows no `)`.
            depth = 0
            opened = angle = None
            for closing, char in enumerate(full_sig):
                if char in "(<[{":
                    if char == "(" and not depth:
                        opened = closing
                    elif char == "<" and not depth:
                        angle = closing
                    depth += 1
                elif char in ")]}" or (char == ">" and full_sig[closing - 1] != "="):
                    depth -= 1
                    if not depth and opened is not None:
                        break
            ret_match = opened is not None and re.match(
                r"\s*:\s*(.+?)(?:(?<![)\s])\s*=>.*|\s*[{;])?$", full_sig[closing + 1 :]
            )
            if ret_match:
                return_type = typed["returns"] = ret_match.group(1).strip()
                if len(return_type) > 50:
                    return_type = return_type[:47] + "..."

            if opened is not None and re.search(r"\basync\b|\*", full_sig[:opened]):
                typed["returned"] = None

            if angle is not None and full_sig[:opened].rstrip().endswith(">"):
                typed["generics"].update(_bounded_parameters(full_sig[angle:opened]))

            if opened is not None:
                params_str = full_sig[opened + 1 : closing].strip()
                typed["params"] = _bracketed_parts(params_str + ")")
                if re.match(r"this\s*:", params_str):
                    # What the function is called on is no argument.
                    del typed["params"][0]
                for position, param in enumerate(typed["params"]):
                    if found := _TS_CALLABLE_RE.match(param):
                        callbacks[str(position)] = [
                            part.partition(":")[2].strip()
                            for part in _bracketed_parts(found["params"] + ")")
                            if not part.startswith("this")
                        ]
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

        return return_type, param_types, typed

    def _arrow_typed(self, parameters, line):
        """The signature, as the resolver reads it, of the arrow function
        that ``line`` declares, ``parameters`` its text from its parameter
        list on."""
        typed = self._extract_ts_types(parameters)[2]
        if not parameters.startswith("("):
            typed = dict(typed, params=[], callbacks={})
        typed["returned"] = None if re.search(r"\basync\b", line) else []
        return typed

    def _overloads(self, node_id, typed):
        """The signatures that ``node_id`` is declared with up to this one,
        ``typed``: those of its overloads before it, and its own."""
        if node_id not in self.builder.node_registry:
            self._signatures[node_id] = []
        declared = self._signatures.setdefault(node_id, [])
        declared.append(typed)
        return declared

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


# Languages whose methods are reachable only through an explicit receiver:
# a bare `name(...)` there never calls a method of the enclosing class.
_EXPLICIT_RECEIVER_LANGS = frozenset({"python", "rust", "go", "typescript"})
# Languages whose parser reports every member call with its receiver, so a
# bare callee is a free call.
_FREE_BARE_CALL_LANGS = frozenset({"python", "go"})
# Languages where `receiver.name(...)` can reach a top-level definition
# through a namespace object. Only an import shows which; without one the
# definition is a second that may be meant, and is not pinned by name.
_NAMESPACE_RECEIVER_LANGS = frozenset({"typescript"})
# Languages where a file that is no module, one that neither imports nor
# re-exports, shares its top-level definitions with every other file.
_SCRIPT_GLOBAL_LANGS = frozenset({"typescript"})
# Languages where a bare call reaches another file's definition only
# through an import of its name.
_IMPORT_SCOPED_LANGS = frozenset({"rust", "typescript"})
# Languages where a bare call is a call on `self`: it reaches a member of the
# enclosing type or of what that inherits, or a top-level definition, never a
# member of an unrelated type.
_IMPLICIT_SELF_LANGS = frozenset({"ruby"})
# Methods that hand on what their receiver wraps: `Result<T, E>::unwrap`.
_UNWRAPPING_METHODS = {"rust": ("unwrap", "expect")}
# Languages where a call on a type from outside reaches that type's own
# methods only, never those of a type argument.
_OWN_METHOD_LANGS = frozenset({"java", "php"})
# Languages that spread one type's members over several declarations.
_SPLIT_TYPE_LANGS = frozenset({"rust", "go", "clang", "dotnet", "ruby", "typescript"})
# Languages whose libraries nothing lists. On a receiver of unknown type a
# name may be a library's, so one project definition of it is not the only
# one that fits; and what a library's call returns may be a project value.
_UNLISTED_LIBRARY_LANGS = frozenset({"php"})
# What a method calls the object or the class it is used on, by language.
_SELF_RECEIVERS = {
    "python": frozenset({"self", "cls"}),
    "rust": frozenset({"self", "Self"}),
    "ruby": frozenset({"self"}),
    "typescript": frozenset({"this"}),
    "java": frozenset({"this"}),
    "dotnet": frozenset({"this"}),
    "clang": frozenset({"this"}),
    "dart": frozenset({"this"}),
}
_EXTERNAL = "external"
# The name of an npm package, scoped or not
_NPM_PACKAGE_RE = re.compile(r"(?:@[a-z0-9~-][a-z0-9._~-]*/)?[a-z0-9~-][a-z0-9._~-]*")


def _ancestors(folder):
    """``folder`` and each directory above it, nearest first."""
    while True:
        yield folder
        parent = posixpath.dirname(folder)
        if parent == folder:
            return
        folder = parent


_UNRESOLVED = "unresolved"
# What `_find_member` gives for whether its lookup is complete where the
# several types a base may be have the member: a miss that is not one
_SEVERAL = "several"
# What a call returns when nothing is known of it, not even that it comes
# from outside the project.
_OPAQUE = "opaque"

_GOOS = frozenset(
    [
        "aix",
        "android",
        "darwin",
        "dragonfly",
        "freebsd",
        "hurd",
        "illumos",
        "ios",
        "js",
        "linux",
        "netbsd",
        "openbsd",
        "plan9",
        "solaris",
        "wasip1",
        "windows",
        "zos",
    ]
)
_GOARCH = frozenset(
    [
        "386",
        "amd64",
        "arm",
        "arm64",
        "loong64",
        "mips",
        "mips64",
        "mipsle",
        "ppc",
        "ppc64",
        "ppc64le",
        "riscv64",
        "s390x",
        "sparc64",
        "wasm",
    ]
)
_RE_GO_BUILD = re.compile(r"^//go:build\s+(\S.*)")


def go_constraints(name, head):
    """Build constraints of the Go file ``name`` (its filename) whose first
    lines are ``head``: ``(requires, excludes)`` of GOOS/GOARCH atoms.
    ``_GOOS``/``_GOARCH`` suffixes require; a `//go:build` line's
    conjunction requires each term and excludes each negated term (an
    alternation says nothing a build must hold)."""
    requires, excludes = set(), set()
    base = name.rpartition("/")[2].removesuffix(".go")
    parts = base.split("_")[1:]
    requires.update(part for part in parts if part in _GOOS or part in _GOARCH)
    for line in head:
        directive = _RE_GO_BUILD.match(line)
        if not directive:
            continue
        terms = directive[1].split("&&")
        if any("||" in term for term in terms):
            continue
        for term in terms:
            term = term.strip().strip("()")
            if term.startswith("!"):
                excludes.add(term[1:].strip())
            else:
                requires.add(term)
    return frozenset(requires), frozenset(excludes)


def _go_builds_meet(a, b):
    """The platform constraints two Go files hold are compatible unless one
    requires an atom the other rules out, or they require different values
    of the same axis (GOOS or GOARCH)."""
    req_a, excl_a = a
    req_b, excl_b = b
    if req_a & excl_b or excl_a & req_b:
        return False
    for axis in (_GOOS, _GOARCH):
        if req_a & axis and req_b & axis and req_a & axis != req_b & axis:
            return False
    return True


def _flip_sign(sign):
    return ("-" if sign.startswith("+") else "+") + sign[1:]


def _split_callee(callee: str) -> tuple[list[str], str]:
    """Split a callee into its qualifier segments and its name."""
    bare = _strip_generics(callee)
    if "(" in bare:
        # A callee written as a whole chain: `Type::new(&arg).unwrap` is
        # read as `Type::new`, `make(x).run` as `run`.
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


# The named type argument a base of a type is written with for what
# iterating the type gives
_ITERATED = {"rust": "Item", "php": "TValue", "dotnet": "T", "dart": "E"}
# The object whose members a language's globals are, as a callee starts with it
_GLOBAL_OBJECT = {"typescript": "globalThis."}
# The type every type of a language has as a base that the data does not
# list among its bases
_ROOT = {"java": "java.lang.Object", "python": "builtins.object"}


class StandardLibrary:
    """One language's standard library as its generated data has it (see
    ``descry._stdlib``): which members a type has, through its bases, and
    what they return.

    An instance of a standard type is the value ``("outside", key,
    arguments)``: its key in the data's `types` and the values of its
    `params` in order, None for one that is not known. Every other value
    (an instance of a project type, None for an unknown one) is the
    caller's, taken as an argument and handed back untouched.

    A language supplies two functions, and in ``_ITERATED`` the named type
    argument its data writes what iterating a type gives under:

    - ``parts(text)`` reads a type as the language writes it, in the data
      and in source, and returns ``(name, arguments, associated)``: the
      name of the type without what is written around it (a reference, a
      pointer), spelled as the data keys it where the language writes the
      type without a name (`[T]`); its type arguments as written, in the
      order of the type's `params`; and its named type arguments (`Item =
      T`) by name, ``{}`` where the language has none. Text it cannot read
      has the name ``""``. In the data a name is a key of `types`, a name
      of `scope`, a parameter of the type the text belongs to or `Self`,
      or `P::Name` for the named type argument that a base of the type a
      parameter `P` stands for is written with.
    - ``key(name, scope_id)`` says what a type or function name written
      in the source of ``scope_id`` is called in the data: through the
      file's imports, through `scope` unless the project declares the
      name, or as it stands. A name the data lacks is no standard type.
    """

    def __init__(self, lang, parts, key):
        self.lang = lang
        self.parts = parts
        self.key = key

    @property
    def data(self):
        return _stdlib.load(self.lang)

    def named(self, written, scope_id):
        """Key in the data of the type ``written`` in the source of
        ``scope_id``; None for a type the data does not have."""
        key = self.key(self.parts(written)[0], scope_id)
        return key if key in self.data["types"] else None

    def value(self, text, bound):
        """Value of the type the data writes as ``text``. ``bound`` holds
        the values of `Self` and of the type parameters ``text`` may
        name."""
        name, arguments, _ = self.parts(text)
        owner, _, associated = name.partition("::")
        if owner in bound:
            if associated:
                return self.associated(bound[owner], associated)
            return bound[owner]
        name = self.data["scope"].get(name) or self.data["scope"].get(
            re.split(r"[.:]", name)[-1], name
        )
        if name not in self.data["types"]:
            return None
        return ("outside", name, tuple(self.value(a, bound) for a in arguments))

    def _entry(self, value, receiver):
        """The data's entry for the instance ``value``, and the values of
        its type parameters and of `Self`: ``receiver``, the instance whose
        bases led to ``value``."""
        entry = self.data["types"][value[1]]
        return entry, {**dict(zip(entry["params"], value[2])), "Self": receiver}

    def associated(self, value, name, base=None, receiver=None, seen=()):
        """The type that a base of the instance ``value`` is written with
        as its named type argument ``name``: the base ``base``, or any."""
        if not isinstance(value, tuple) or len(value) < 3 or value[1] in seen:
            return None
        entry, bound = self._entry(value, receiver or value)
        for written in entry["bases"]:
            base_name, _, associated = self.parts(written)
            bound_type = associated.get(name, f"Self::{name}")
            named = base in (None, self.data["scope"].get(base_name, base_name))
            # A base that hands on the type's own `Self::Item` sets nothing.
            if named and bound_type != f"Self::{name}":
                return self.value(bound_type, bound)
            inherited = self.value(written, bound)
            found = self.associated(
                inherited, name, base, bound["Self"], seen + (value[1],)
            )
            if found is not None:
                return found
        return None

    def _where(self, value, name, receiver, table, seen=()):
        """``(key, bound)`` of the first type among the instance ``value``
        and its bases, in their order, whose ``table`` has ``name``, with
        the values of its parameters; ``(None, value)`` where none has:
        ``value`` is then the first base that is no standard type (what a
        pointer points at), itself where every base is one."""
        if len(value) < 3:
            return None, value
        if value[1] in seen:
            return None, value
        entry, bound = self._entry(value, receiver or value)
        if name in entry[table]:
            return value[1], bound
        for base in entry["bases"]:
            inherited = self.value(base, bound)
            if not isinstance(inherited, tuple) or inherited[0] != "outside":
                return None, inherited
            key, found = self._where(
                inherited, name, bound["Self"], table, seen + (value[1],)
            )
            if key or found != inherited:
                return key, found
        return None, value

    def member(self, value, name, receiver=None, table="methods", results=None):
        """Look ``name`` up in ``table``, `methods` or `statics`, of the
        instance ``value`` and of its bases in their order. Returns
        ``(True, the value the member returns)``, or ``(False, value)``
        where the data has no such member: ``value`` is then the first
        base that is no standard type (what a pointer points at), itself
        where every base is one. ``receiver`` is the instance `Self`
        stands for where that is not ``value``: a project type calling a
        member of a standard base. ``results`` gives, by argument
        position, the value that a callable passed to the member returns."""
        key, bound = self._where(value, name, receiver, table)
        if not key and bound == value and self.lang in _ROOT:
            root = ("outside", _ROOT[self.lang], ())
            key, found = self._where(root, name, receiver or value, table)
            bound = found if key else bound
        if not key:
            return False, bound
        callbacks = self.data.get("callbacks", {}).get(key, {}).get(name, {})
        for position, callback in callbacks.items():
            bound[callback["returns"]] = (results or {}).get(position)
        returns = self.data["types"][key][table][name]
        return True, returns and self.value(returns, bound)

    def callbacks(self, value, name):
        """``{argument position: values of the parameters}`` of each
        callable that the method ``name`` of the instance ``value`` takes."""
        key, bound = self._where(value, name, None, "methods")
        callbacks = self.data.get("callbacks", {}).get(key, {}).get(name, {})
        return {
            position: [self.value(written, bound) for written in callback["params"]]
            for position, callback in callbacks.items()
        }

    def held(self, value, variant, field):
        """Value that the variant ``variant`` of the instance ``value``
        holds at ``field``, a position or a name."""
        entry, bound = self._entry(value, value)
        fields = self.data.get("variants", {}).get(value[1], {}).get(variant)
        if fields is None:
            # An alias has the variants of what it stands for.
            for base in entry["bases"]:
                inherited = self.value(base, bound)
                if (
                    isinstance(inherited, tuple)
                    and len(inherited) > 2
                    and inherited[1] != value[1]
                ):
                    found = self.held(inherited, variant, field)
                    if found is not None:
                        return found
            return None
        named = field in fields if isinstance(fields, dict) else field < len(fields)
        return self.value(fields[field], bound) if named else None

    def function(self, name, scope_id):
        """``(True, the value returned)`` where ``name``, written in the
        source of ``scope_id``, is a function of the data."""
        functions = self.data["functions"]
        key = self.key(name, scope_id)
        return key in functions, functions.get(key) and self.value(functions[key], {})


class _ClangLibrary(StandardLibrary):
    """The C and C++ standard libraries read together: `c.json` names the
    C functions, `cpp.json` the types and functions of `std`."""

    def __init__(self, parts, key):
        super().__init__("clang", parts, key)
        self._data = None

    @property
    def data(self):
        if self._data is None:
            out = {
                "source": "c + cpp",
                "types": {},
                "functions": {},
                "scope": {},
            }
            for lang in ("c", "cpp"):
                data = _stdlib.load(lang)
                for kind in ("types", "functions", "scope"):
                    out[kind] |= data.get(kind, {})
            self._data = out
        return self._data


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
        self.c_spaces = set()  # names of the C++ namespaces that macros open
        for node_id, node in self.nodes.items():
            parent, sep, name = node_id.rpartition("::")
            self.c_spaces.update(node["metadata"].get("opens", ()))
            if sep:
                lang = self.lang(node_id)
                self.by_name.setdefault((lang, name), []).append(node_id)
                self.members.setdefault(parent, {})[name] = node_id
                declared = self._name(node_id)
                if declared != name and node["type"] == "Class":
                    # A type is looked up by the name it was declared under.
                    self.renamed_types.setdefault((lang, declared), []).append(node_id)
                    self.companions[f"{parent}::{declared}"] = node_id
        # Name -> the C# extension methods declared under it
        self.extensions = {}
        for node_id, node in self.nodes.items():
            if "extends" in node["metadata"]:
                self.extensions.setdefault(self._name(node_id), []).append(node_id)
        # class ID -> base type names, in Python the origin path of each
        self.bases = {}
        # index of an INHERITS edge -> the base as written, qualifier and all
        self.base_names = {}
        # class ID -> its bases as written, with their type arguments
        self.written_bases = {}
        self.requires = {}  # Ruby file ID -> the paths it requires
        for index, edge in enumerate(builder.edges):
            if edge["relation"] == "IMPORTS" and self.lang(edge["source"]) == "ruby":
                path = edge["target"].removeprefix("MODULE:")
                self.requires.setdefault(edge["source"], []).append(path)
            if edge["relation"] == "INHERITS":
                # A base the parser gave as a node is known by its name too.
                base = written = edge["target"].removeprefix("REF:")
                self.written_bases.setdefault(edge["source"], []).append(written)
                # A PHP base is written by its full name, which names one
                # type; a Ruby constant is looked up by its whole path.
                lang = self.lang(edge["source"])
                if lang not in ("php", "ruby"):
                    qualifier, base = _split_callee(base)
                    written = ".".join([*qualifier, base])
                origin = edge.get("metadata", {}).get("origin", "")
                if lang == "python":
                    base = written = origin or base
                elif origin.startswith("="):
                    # A type parameter names no type.
                    base = written = ""
                # A base is looked up as written, its qualifier with it.
                self.bases.setdefault(edge["source"], []).append(written)
                self.base_names[index] = written
        # A Go struct has the methods of the types it embeds, listed under
        # None among its fields.
        for type_id, fields in builder.fields.items():
            if None in fields:
                self.bases[type_id] = fields[None]
        for node_id, node in self.nodes.items():
            if node["type"] == "Class" and self.lang(node_id) == "java":
                root = _JAVA_ROOTS.get(node["metadata"].get("kind"), _ROOT["java"])
                self.bases.setdefault(node_id, []).append(root)
            elif node["type"] == "Class" and self.lang(node_id) == "python":
                root = f"@{_ROOT['python']}"
                self.bases.setdefault(node_id, []).append(root)
        self.containers = {}  # (language, module or package name) -> file IDs
        self.go_packages = {}  # directory -> Go file IDs
        self.go_modules = sorted(  # (go.mod directory, module path), longest first
            builder.go_modules.items(), key=lambda item: -len(item[1])
        )
        self.java_types = {}  # dotted class name -> class ID
        self.java_scoped = {}  # (scope, type name) -> the type it names there
        self.java_fields = {}  # (class ID, name) -> the field read by it there
        self.c_files = {}  # path a C/C++ file can be included by -> file IDs
        # dotted module name -> {directory its topmost package is in: file ID}
        self.py_modules = {}
        self.py_root_of = {}  # Python file ID -> that directory
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
                    # A directory without `__init__.py` below a package is
                    # a namespace package of it.
                    outer = next(
                        (
                            at - 1
                            for at in range(1, top)
                            if f"FILE:{'/'.join(parts[:at])}/__init__.py" in files
                        ),
                        top,
                    )
                    self.py_module_of[file_id] = ".".join(parts[outer:])
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
            elif lang == "clang":
                # A file is included by any suffix of its path.
                path = file_id[5:].split("/")
                for start in range(len(path)):
                    self.c_files.setdefault("/".join(path[start:]), []).append(file_id)
            elif lang == "java" and f"{file_id}::{parts[-1]}" in self.nodes:
                # A public class is importable by any dotted suffix of its path.
                for start in range(len(parts)):
                    self.java_types.setdefault(
                        ".".join(parts[start:]), f"{file_id}::{parts[-1]}"
                    )
        # A module is importable by its path below its topmost package and,
        # through namespace packages, by any longer path no other module owns.
        for top, parts, file_id in py_files:
            root = "/".join(parts[:top])
            self.py_root_of[file_id] = root
            self.py_modules.setdefault(".".join(parts[top:]), {}).setdefault(
                root, file_id
            )
        owned = set(self.py_modules)
        for top, parts, file_id in py_files:
            for start in range(top):
                if (name := ".".join(parts[start:])) not in owned:
                    held = self.py_modules.setdefault(name, {})
                    held.setdefault("/".join(parts[:start]), file_id)
        self.py_roots = {module.split(".")[0] for module in self.py_modules}
        # C/C++ file ID -> IDs of the project files it includes, of those
        # that include it, and of those it is compiled with
        self.includes, self.includers, self.units = {}, {}, {}
        for edge in builder.edges:
            source = edge["source"]
            if edge["relation"] == "IMPORTS" and self.lang(source) == "clang":
                path = posixpath.normpath(edge["target"][len("MODULE:") :])
                beside = posixpath.join(posixpath.dirname(source[5:]), path)
                beside = f"FILE:{posixpath.normpath(beside)}"
                found = [beside] if beside in self.nodes else self.c_files.get(path, [])
                self.includes.setdefault(source, set()).update(found)
                for included in found:
                    self.includers.setdefault(included, set()).add(source)
        # Language -> its standard library, for those whose data is read
        self.standard = {
            "rust": StandardLibrary("rust", _rust_type_parts, self._rust_std_key),
            "go": StandardLibrary("go", _go_type_parts, self._go_std_key),
            "typescript": StandardLibrary(
                "typescript", _ts_type_parts, self._ts_std_key
            ),
            "java": StandardLibrary("java", _java_type_parts, self._java_std_key),
            "python": StandardLibrary("python", _py_type_parts, self._py_std_key),
            "php": StandardLibrary("php", _php_type_parts, self._php_std_key),
            "dotnet": StandardLibrary(
                "dotnet", _dotnet_type_parts, self._dotnet_std_key
            ),
            "ruby": StandardLibrary("ruby", _ruby_type_parts, self._ruby_std_key),
            "dart": StandardLibrary("dart", _dart_type_parts, self._dart_std_key),
            "clang": _ClangLibrary(_clang_type_parts, self._clang_std_key),
        }
        # Types from outside that the project implements traits for, by
        # name; under "" the blanket impls, which are for a type parameter.
        self.impl_types = {}
        for node_id, node in self.nodes.items():
            name = self._name(node_id)
            if node["metadata"].get("kind") == "impl" and not self._declares_type(name):
                blanket = "bounds" in node["metadata"]
                self.impl_types.setdefault("" if blanket else name, []).append(node_id)
            if node["metadata"].get("kind") == "extension":
                # A Dart extension gives its members to the type it is on.
                for base in self.bases.get(node_id, ()):
                    self.impl_types.setdefault(_split_callee(base)[1], []).append(
                        node_id
                    )
        self.dart_parts = {}  # Dart file ID -> the library file it is a part of
        for file_id, directives in builder.dart_directives.items():
            for keyword, uris, *_ in directives:
                for uri in uris if keyword == "part" else ():
                    if part := self._dart_file(uri, file_id):
                        self.dart_parts[part] = file_id
        self.shared_declaration = {}  # candidate node IDs -> node ID or None
        # (Python module or class ID, name) -> value of that variable of it
        self.variables = {}
        # (Python class ID, origin path of a base) -> see ``_type_named``
        self.base_types = {}
        self.heirs = None  # path of a Ruby type -> the types it is a base of
        self.constants = None  # see _constant
        self.loaded = {}  # Ruby file ID -> the files it loads
        self.reaches = {}  # file ID -> the files its includes reach (C/C++)
        self.returned = {}  # edge index of a call -> the value it returns
        # edge index of a call -> {argument position: values of the
        # parameters of the callable passed there}
        self.callables = {}
        # edge index -> (callee, its parameters as written)
        self.declared_params = {}
        self._callables = {}  # those of the call being resolved
        self.this_args = {}  # call index -> {argument: `ThisType` value}
        self._this_args = {}  # those of the call being resolved
        # The function whose call is being kept, the value of each of its
        # type parameters there and the signature the call is of
        self._bound = None, {}, None
        # edge index of a call -> (scope, standard type, method, {argument
        # position: origin of what the callable passed there returns}),
        # until those callables' bodies are resolved
        self.deferred = {}
        self._deferred = None
        # edge index of a call -> (function, call site, scope, receiver),
        # until the calls it is passed the results of are resolved
        self.late = {}
        # (function, type it is called on) -> the value its body returns
        self.body_values = {}
        # (function, arguments passed, type arguments written) -> the
        # signature a call of it is of
        self.overloads = {}
        self._result = None  # value of the call being resolved, where known
        self._receiver = None  # type the method being resolved is called on
        self.arguments = None  # what that call passes, where its parser counted
        # Signed `#if` atoms the line of the call being resolved is under
        self.call_guards = None
        # Language -> the last name segments of the types and functions its
        # standard library's data has, and the names of their members.
        self._standard_names = {}
        # the definitions the call being resolved may reach, where several fit
        self._candidates_now = None
        # what a name-only resolution returns is not to be trusted
        self._name_only = False
        self._forwarding = set()  # methods whose returned call is being followed
        # A Ruby block that defines methods has what the type around it
        # has. One run in a class a constant names has what that class has
        # instead, and the class what the block defines.
        for scope_id, node in self.nodes.items():
            if node["metadata"].get("kind") != "block":
                continue
            around = self._enclosing_class(scope_id.rpartition("::")[0])
            opened = node["metadata"].get("reopens")
            if opened:
                self.bases[scope_id] = [opened]
                reopened = self._constant(opened.split("::"), scope_id)
                if reopened not in (None, _EXTERNAL):
                    path = scope_id.partition("::")[2]
                    self.bases.setdefault(reopened, []).append(path)
            elif around:
                self.bases[scope_id] = [around.partition("::")[2]]

    def _name(self, node_id):
        """The name ``node_id`` was declared under, which its ID may prefix."""
        declared = self.nodes[node_id]["metadata"].get("name")
        return declared or node_id.rpartition("::")[2]

    def keep(self, index, target):
        """Remember what the call at edge ``index``, just resolved to
        ``target`` if it was, returns and passes to its callables."""
        edge = self.builder.edges[index]
        site = edge.get("metadata", {}).pop("site", {})
        self._bound = None, {}, None
        if target in self.builder.signatures:
            self._bind_call(target, site, edge["source"])
            self.declared_params[index] = target, self._bound[2]["params"]
            self._callables = {
                position: [
                    self._returned_type(written, target, self._receiver)
                    for written in types
                ]
                for position, types in self._bound[2]["callbacks"].items()
            }
            for position, parameter in enumerate(self._bound[2]["params"]):
                if found := re.search(r"ThisType<([^<>]*)>", parameter):
                    written = re.sub(r"::", ".", found[1].strip())
                    self._this_args[str(position)] = self._returned_type(
                        written, target, self._receiver
                    )
            passed = [a for a in site.get("arguments") or () if isinstance(a, str)]
            calls = (re.match(r"#(\d+)", origin) for origin in passed)
            if any(call and int(call[1]) > index for call in calls):
                self.late[index] = (target, site, edge["source"], self._receiver)
        elif target and self.lang(edge["source"]) == "rust":
            # A closure an argument takes has the parameters of the `Fn`
            # trait the parameter is declared with.
            declared = self.nodes[target]["metadata"].get("param_types") or ()
            declared = [t for t in declared if not t.endswith("self")]
            self._callables = {
                str(position): [
                    None if p == "_" else self._returned_type(p, target) for p in params
                ]
                for position, written in enumerate(declared)
                if (found := _RUST_FN_TYPE_RE.search(written))
                and (params := _bracketed_parts(found[1] + ")"))
            }
        elif target and self.lang(edge["source"]) == "java":
            # A lambda an argument position of the method is typed by the
            # functional interface that position declares.
            declared = self.nodes[target]["metadata"].get("params") or ()
            self._callables = {
                str(position): list(params)
                for position, params in (
                    (position, self._functional_params(written, edge["source"]))
                    for position, written in enumerate(declared)
                )
                if params
            }
        self.returned[index] = self.result(target)
        if site.get("types") and target and self._is_class(target):
            # A generic type called with its type arguments makes a value
            # that holds them.
            made = self._returned_type(f"{self._name(target)}{site['types']}", target)
            self.returned[index] = made or self.returned[index]
        self.callables[index] = self._callables
        if self._this_args:
            self.this_args[index] = self._this_args
        if target is None and self._candidates_now:
            # Unresolved: the call reaches one of these.
            edge["metadata"]["candidates"] = self._candidates_now
        if self._deferred:
            self.deferred[index] = self._deferred

    def _bind_call(self, target, site, source_id):
        """Make the call ``site`` of the function ``target``, written in
        ``source_id``, the one whose signature and type arguments are read."""
        self._bound = None, {}, None
        signature = self._overload(target, site)
        bound = self._type_arguments(signature, target, site, source_id)
        self._bound = target, bound, signature

    def _overload(self, function_id, site):
        """The signature of ``function_id`` that the call ``site`` is of.
        Of its overloads, those that take as many arguments, where the
        number passed is known, and as many type arguments as the call is
        written with: the first, where nothing but the types of the
        arguments could choose another and all take the same types, or
        any where all return one known type. With none to choose, one
        that has no parameters and returns nothing."""
        declared = self.builder.signatures[function_id]
        # The signature of the body is not one to call where others are.
        fitting = [one for one in declared if not one["body"]] or declared
        if len(fitting) == 1:
            return fitting[0]
        arguments = site.get("arguments")
        passed = None if arguments is None else len(arguments)
        key = function_id, passed, site.get("types")
        if key in self.overloads:
            return self.overloads[key]
        if arguments is not None:
            counted = []
            for one in fitting:
                params = one["params"]
                needed = [
                    param
                    for param in params
                    if not re.match(r"(?:\.\.\.|[\w$]+\?)|.*=(?!>)", param)
                ]
                rest = any(param.startswith("...") for param in params)
                most = len(arguments) if rest else len(params)
                if len(needed) <= len(arguments) <= most:
                    counted.append(one)
            fitting = counted or fitting
        written = len(_bracketed_parts(site.get("types", "<")[1:]))
        fitting = [one for one in fitting if len(one["generics"]) >= written] or fitting
        taken = {
            tuple(param.partition(":")[2].strip() for param in one["params"])
            for one in fitting
        }
        returned = [self._returned_type(one["returns"], function_id) for one in fitting]
        agreed = returned[0] is not None and returned.count(returned[0]) == len(fitting)
        chosen = {"generics": {}, "params": [], "returns": "", "callbacks": {}}
        types = [value and value[:2] for value in returned]
        if not agreed and types[0] and types.count(types[0]) == len(fitting):
            # One type with arguments that differ: that type, its arguments
            # not known.
            chosen = {**fitting[0], "returns": _strip_generics(fitting[0]["returns"])}
        self.overloads[key] = fitting[0] if len(taken) == 1 or agreed else chosen
        return self.overloads[key]

    def _plain_value(self, origin, source_id):
        """Value of an origin path that names no member: what a call
        returned, a parameter, or an instance of a declared type."""
        value, steps, _ = self._root_value(origin, source_id, set())
        for step in steps:
            if step != "()" or not isinstance(value, tuple):
                return None
            if value[0] in ("def", "cls"):
                value = self._called(value)
        return value

    def _type_arguments(self, signature, target, site, source_id):
        """Value of each type parameter of ``signature``, one of the
        function ``target``, in the call ``site`` written in ``source_id``:
        of the type argument the call is written with, else of the
        argument passed for a parameter declared as it, else of the one
        that the type the call's result is declared as has in its place in
        the declared return type, else, where no argument is passed for it,
        of its bound."""
        names = list(signature["generics"])
        if not names:
            return {}
        written = dict(zip(names, _bracketed_parts(site.get("types", "<")[1:])))
        values = {}
        for passed, param in zip(site.get("arguments") or (), signature["params"]):
            taken = re.fullmatch(r"[\w$]+\??\s*:\s*([\w$]+)", param)
            # What the argument's type gives, not the parameters of a
            # function it is declared as
            typed = re.sub(r"\([^()]*\)\s*=>", "", param.partition(":")[2])
            for name in () if written else names:
                if re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", typed):
                    # The argument gives it: one not written as a name, or
                    # of a type it is only part of, leaves it unknown.
                    plain = taken and taken[1] == name and isinstance(passed, str)
                    held = plain and self._plain_value(passed, source_id)
                    values.setdefault(name, held or None)
        if not written and "declared" in site:
            name, arguments, _ = _ts_type_parts(signature["returns"])
            declared = site["declared"]
            returned = self._returned_type(signature["returns"], target)
            given = self._returned_type(declared, source_id)
            if name in names:
                written = {name: declared}
            elif (
                isinstance(returned, tuple)
                and isinstance(given, tuple)
                and returned[:2] == given[:2]
                and (returned[0] == "self" or len(returned) == 3)
            ):
                written = dict(zip(arguments, _ts_type_parts(declared)[1]))
        for name, bound in signature["generics"].items():
            given = (written[name], source_id) if name in written else (bound, target)
            if name in written or name not in values:
                values[name] = self._returned_type(*given) if given[0] else None
        return values

    def resolve(self, callee, source_id, origin=None, arguments=None):
        """``arguments`` is how many arguments the call passes, where the
        parser counted them."""
        self._result = self._receiver = self._deferred = None
        self._callables = {}
        self._this_args = {}
        self._candidates_now = None
        self._name_only = False
        self._arguments = arguments
        if self.lang(source_id) in _GLOBAL_OBJECT:
            callee = callee.removeprefix(_GLOBAL_OBJECT[self.lang(source_id)])
        # `new Type<arguments>`: the call constructs the type written,
        # whatever the callables it is passed return.
        constructed = None
        if origin and origin.startswith("new "):
            constructed, origin = re.sub(r"\{\d+=.*", "", origin[4:]), None
        verdict = self._verdict(callee, source_id, origin)
        if verdict[0] and not self._accepts(verdict[0]):
            # A definition that cannot take what the call passes is not
            # the one called: the receiver is another object than it seems.
            verdict = None, _UNRESOLVED
        if verdict[0] and self._candidate_only(verdict[0], source_id):
            # What resolved is a definition an `#if` arm alone builds, and
            # nothing about the call site selects that arm.
            self._candidates_now = self._candidates_now or [verdict[0]]
            verdict = None, _UNRESOLVED
        rust_path = "::" in callee and "." not in callee
        lang = self.lang(source_id)
        # A Go callee without an origin is a function, or one of a package,
        # as a C++ `std::` one is a function of the standard library and a
        # Rust path one of a type or a module.
        function = origin is None and (
            rust_path
            if lang == "rust"
            else lang == "go" or (lang == "clang" and "::" in callee)
        )
        if verdict[1] == _EXTERNAL and function:
            # What a function of the standard library returns is in its data.
            known, value = self._std_call(callee, source_id)
            if known:
                self._result = value or _OPAQUE
        if verdict[1] == _EXTERNAL and constructed:
            # A standard type gives an instance of itself; a variable the
            # library declares as another type gives what is not known.
            if self.standard[lang].key(callee, source_id) == callee:
                held = self._returned_type(constructed, source_id)
                self._result = held or self._result
        elif verdict[1] == _EXTERNAL and not origin and lang == "typescript":
            known, value = self._js_global(callee, source_id)
            if known:
                self._result = value or _OPAQUE
        name = _split_callee(callee)[1]
        if verdict[1] == _EXTERNAL and origin and name in self.extensions:
            # A call on a value from outside may reach a C# extension method
            # written for a type from outside, unless that type has the
            # member itself. `Type.Method(...)` is no call on a value.
            written = (
                self.nodes[e]["metadata"]["extends"] for e in self.extensions[name]
            )
            path = re.fullmatch(r"(?!self\.)([A-Za-z_]\w*)(?:\.\w+)+", origin)
            static = path and not self._inherited(path[1], source_id)
            on_value = "." in origin and not static
            # The receiver's declared library type having the member says
            # the call is the library's — no extension of it is meant.
            typed = self._result is not None and self._result not in (
                _OPAQUE,
                _EXTERNAL,
            )
            if (
                on_value
                and not typed
                and not all(self._types("dotnet", w) for w in written)
            ):
                self._result = _OPAQUE
                return None, _UNRESOLVED
        unlisted = self.lang(source_id) in _UNLISTED_LIBRARY_LANGS
        if verdict[1] == _EXTERNAL and self._result is None and not unlisted:
            # What a call returns is from outside too when its receiver or
            # its path is, unless the call has type arguments, `name<>`,
            # one of which may be what it returns.
            self._result = _OPAQUE if callee.endswith("<>") else _EXTERNAL
        return verdict

    def _accepts(self, target):
        """Whether the definition ``target`` can take the arguments of the
        call being resolved, where its parser counted them and recorded
        what the definition takes: ``(positional arguments, whether
        keywords follow)`` against ``(required, most or None for any
        number, keywords: 0 none, 1 taken, 2 required)``. Keywords passed
        to a definition that names none are its last positional argument."""
        arity = self.nodes[target]["metadata"].get("arity")
        if not self.arguments or not arity:
            return True
        count, keywords = self.arguments
        required, most, named = arity
        count += keywords and not named
        fits = required <= count and (most is None or count <= most)
        return fits and (keywords or named < 2)

    def _verdict(self, callee, source_id, origin):
        lang = self.lang(source_id)
        qualifier, name = _split_callee(callee)
        candidates = self.by_name.get((lang, name), [])
        if lang == "typescript" and candidates:
            # A definition elsewhere of the same name is not a candidate
            # the call can reach.
            candidates = self._named_from(candidates, source_id)
        if lang == "php":
            # A PHP call reaches no property or constant of its name.
            candidates = [c for c in candidates if self.nodes[c]["type"] != "Constant"]
        scope_id = source_id
        if origin is None and lang == "go" and qualifier:
            # A package-level variable is read as what it is declared as,
            # where it is declared.
            held = self._go_variable(qualifier[0], source_id)
            if held:
                written = self.nodes[held]["metadata"]["value_type"]
                origin = ".".join([f"{written}()", *qualifier[1:], name])
                origin, scope_id = origin.replace(".", "::", written.count(".")), held
        if lang == "clang" and origin and origin[0] == "^":
            # A bare call written as an argument of a macro is qualified as
            # the macro's text qualifies that parameter: globally, or by a
            # scope or receiver nothing tells. A pasted name is not the one
            # written, and definitions of the macro that disagree leave the
            # call open.
            how = self._c_rewritten(*origin[1:].split("^"), source_id)
            origin = None
            if len(how) > 1:
                return None, _UNRESOLVED if candidates else _EXTERNAL
            if "paste" in how:
                return None, _EXTERNAL
            if "." in how:
                return self._resolve_member(candidates, lang, callee)
            callee = f"::{callee}" if how else callee
        if lang == "clang" and origin is None and name in self.builder.c_aliases:
            # A call of an alias constructs the type it names, where nothing
            # else of that name can be called.
            written = "::".join([*qualifier, name])
            reached = (
                c
                for c in candidates
                if self._is_top_level(c) and self._in_reach(c, source_id)
            )
            found = not any(reached) and self._type_named(written, source_id)
            if found and found != _EXTERNAL:
                return found, "regex"
        if lang == "clang" and origin is None and "::" in callee:
            # The global namespace and one a macro opens have no node: what
            # they hold is at the top of the files.
            qualifier = [q for q in qualifier if q not in self.c_spaces]
            if not qualifier:
                top = [c for c in candidates if c.count("::") == 1]
                file_id = source_id.split("::")[0]
                return self._resolve_bare(name, top, file_id, lang, callee)
        if origin is not None:
            verdict = self._resolve_origin(origin, scope_id)
            if verdict is None:
                verdict = self._resolve_member(candidates, lang, callee)
            if lang == "dart" and verdict[1] == _EXTERNAL and name.startswith("_"):
                # A private Dart name is declared in the library that uses it.
                verdict = self._resolve_member(candidates, lang, callee)
            if verdict[0] or candidates:
                return verdict
            # No project symbol carries the name, so the call is external;
            # what it returns is unknown unless its receiver is external too.
            if verdict[1] != _EXTERNAL:
                self._result = _OPAQUE
            return None, _EXTERNAL
        if lang == "php":
            return self._php_function(callee, candidates, source_id)
        imported = self._imported(qualifier[0] if qualifier else name, source_id)
        if imported == _EXTERNAL:
            return None, _EXTERNAL
        if imported is None and qualifier:
            # A variable in scope that has a declared type, or members of
            # its own, is a receiver as an imported name is.
            held = self._lexical(qualifier[0], source_id, skip_classes=True)
            if held and self.nodes[held]["type"] == "Constant":
                declared = self._declared_type(held)
                if declared == _EXTERNAL:
                    return None, _EXTERNAL
                if declared or held in self.members or held in self.builder.reexports:
                    imported = [held]
        if imported is not None:
            # An import names the definition itself or the scope it is in;
            # each further segment names what that scope holds in turn. The
            # scope may export the definition under another name.
            paths = [qualifier[1:]]
            bound = self.builder.imports.get(source_id.split("::")[0], {})
            if (
                lang in _NAMESPACE_RECEIVER_LANGS
                and qualifier[:1]
                and qualifier[0] in bound
            ):
                # A receiver an import binds to one export, not to the
                # module whole, is what the module holds under its name.
                head = self._exported_as(qualifier[0], source_id)
                paths = [qualifier[1:] if head == "=" else [head, *qualifier[1:]]]
            looked_up = name if qualifier else self._exported_as(name, source_id)
            for path in paths:
                scopes = imported
                for segment in path:
                    found = (self._find_member(scope, segment)[0] for scope in scopes)
                    scopes = [scope for scope in found if scope]
                hits = {
                    i for i in scopes if not qualifier and i.endswith(f"::{looked_up}")
                }
                hits |= {self._find_member(scope, looked_up)[0] for scope in scopes}
                hits.discard(None)
                if hits:
                    break
            if hits or (candidates and len(qualifier) <= 1):
                hit = self._decide(sorted(hits), source_id) if hits else None
                return (hit, "regex") if hit else (None, _UNRESOLVED)
        if not candidates:
            # No project symbol has the name; that alone does not say where
            # the receiver, and so the result, comes from.
            self._result = _OPAQUE
            owner = (
                lang == "java" and not qualifier and self._enclosing_class(source_id)
            )
            if owner:
                self._result = self._std_inherited(owner, name) or _OPAQUE
            return None, _EXTERNAL
        if not qualifier:
            return self._resolve_bare(name, candidates, source_id, lang, callee)
        receiver = qualifier[-1]
        if receiver in _SELF_RECEIVERS.get(lang, ()):
            owners = [self._enclosing_class(source_id)] if len(qualifier) == 1 else []
            source = "self_type"
        else:
            owners = self._types_in(lang, receiver, source_id)
            if lang == "clang":
                owners = [o for o in owners if self._in_reach(o, source_id)] or owners
            if lang != "rust":
                owners += self.containers.get((lang, receiver), [])
            elif callee.endswith(f"::{name}"):
                # A Rust module is named in a path, never before a `.`. The
                # first segment of a path may be the name a `use` of the
                # file binds the module under.
                uses = self.builder.imports.get(source_id.split("::")[0], {})
                bound = (
                    uses.get(receiver, receiver) if len(qualifier) == 1 else receiver
                )
                owners += self.containers.get((lang, bound.rpartition("::")[2]), [])
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
                hit = self._decide(sorted(hits), source_id) if hits else None
                return (hit, source) if hit else (None, _EXTERNAL)
        if any(owners):
            hits, complete = set(), True
            for owner in owners:
                hit, owner_complete = self._find_member(owner, name)
                # A namespace may hold more than the project puts in it,
                # unless it is a value too, whose type tells what it holds.
                metadata = self.nodes[owner]["metadata"]
                open_scope = metadata.get("kind") == "namespace" and not metadata.get(
                    "value_type"
                )
                complete = complete and owner_complete and not open_scope
                if hit:
                    hits.add(hit)
            if lang == "clang":
                hits = {h for h in hits if self._in_reach(h, source_id)} or hits
            if hits:
                hit = self._decide(sorted(hits), source_id)
                if hit and lang == "clang" and self._overloaded(hit, hits, source_id):
                    hit = None
                return (hit, source) if hit else (None, _UNRESOLVED)
            # A member a project type has from outside may return that type.
            self._result = _OPAQUE
            if len(owners) == 1:
                self._result = self._std_inherited(owners[0], name) or _OPAQUE
            return None, _UNRESOLVED if complete else _EXTERNAL
        aliased = lang == "clang" and receiver in self.builder.type_aliases
        if lang in ("rust", "clang") and "::" in callee and not aliased:
            return None, _EXTERNAL
        if lang == "typescript" and self._js_global(callee, source_id)[0]:
            return None, _EXTERNAL
        return self._resolve_member(candidates, lang, callee)

    def _php_function(self, callee, candidates, source_id):
        """Verdict on a PHP call without a receiver, which reaches a function
        only. The parser writes its name in full: `\\App\\run` for the
        function of that name, `App\\run` for the one of the caller's
        namespace or, failing that, the global `run`."""
        namespace, _, name = callee.rpartition("\\")
        exact = callee.startswith("\\")
        for space in [namespace.lstrip("\\")] if exact else [namespace, ""]:
            functions = [
                c
                for c in candidates
                if self.nodes[c]["type"] == "Function"
                and self.nodes[c]["metadata"].get("namespace") == space
            ]
            if functions:
                hit = self._decide(functions, source_id)
                return (hit, "regex") if hit else (None, _UNRESOLVED)
        # What a function of PHP's own returns is in the library's data.
        key = f"{namespace}\\{name}".strip("\\") if exact else name
        value = self.standard["php"].function(key, source_id)[1]
        self._result = value or _OPAQUE
        return None, _EXTERNAL

    def _exported_as(self, name, scope_id):
        """The name the module an import in the file of ``scope_id`` binds
        ``name`` from exports it as."""
        renamed = self.builder.import_names.get(scope_id.split("::")[0], {})
        return renamed.get(name, name)

    def _ts_alias(self, name, scope_id):
        """``(type parameters, text)`` of the TypeScript type alias ``name``
        written in ``scope_id`` stands for: one its file declares, else one
        a module it imports the name from does. None for none."""
        file_id = scope_id.split("::")[0]
        found = self.builder.ts_aliases.get((file_id, name))
        imported = None if found else self._imported(name, scope_id)
        for module in imported if isinstance(imported, list) else ():
            exported = self._exported_as(name, scope_id)
            found = found or self.builder.ts_aliases.get(
                (module.split("::")[0], exported)
            )
        return found

    def _imported(self, name, scope_id):
        """What an import in the file of ``scope_id`` binds ``name`` to: the
        IDs of the project definitions or files it names (none when they
        cannot be located), ``"external"`` for a target outside the project,
        None when no import binds it."""
        file_id = scope_id.split("::")[0]
        target = self.builder.imports.get(file_id, {}).get(name)
        lang = self.lang(file_id)
        if lang == "dart":
            return self._dart_prefixed(name, file_id)
        if target is None or lang not in ("go", "java", "typescript", "rust"):
            return None
        if lang == "typescript":
            required = name in self.builder.required_names.get(file_id, ())
            return self._js_module(target, file_id, required)
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
            if self._name_only:
                # Resolved by name: what the definition returns is not
                # evidence of what this call's receiver gives back.
                return _OPAQUE
            return self._called(("def", target), self._receiver)
        return None if self._result == _OPAQUE else self._result

    def _use_crate(self, path, file_id):
        """Where a Rust path written in ``file_id`` leads: ``(crate
        directory,)``, ``()`` for a crate outside the project, None when
        the path starts at a module or type in scope. A name that a `use`
        of the file binds leads where that `use` does."""
        root = path.split("::")[0]
        root = self.builder.imports.get(file_id, {}).get(root, root).split("::")[0]
        if root in ("crate", "self", "super"):
            return (self._unit(file_id, "rust"),)
        if root in self.builder.rust_crates:
            return (self.builder.rust_crates[root],)
        if ("rust", root) in self.containers or self._types("rust", root):
            return None
        return ()

    def _js_module(self, specifier, file_id, required=False):
        """Resolve a module specifier written in ``file_id``, by a `require`
        if ``required``: ``[file ID]`` for a project module, ``[]`` for one
        of the project's that cannot be located, ``"external"`` for a
        declared dependency, None for anything else."""
        folder = posixpath.dirname(file_id[5:])
        if specifier.startswith("."):
            found = self._js_file(posixpath.join(folder, specifier))
            return [found] if found else []
        aliased = self._js_alias(specifier, folder)
        if aliased is not None:
            return aliased
        if specifier.partition("/")[0] == "$lib":
            # SvelteKit: `$lib` is `src/lib` beside the nearest configuration.
            for above in _ancestors(folder):
                if self._js_file(posixpath.join(above, "svelte.config")):
                    library = posixpath.join(above, "src/lib", specifier[5:])
                    found = self._js_file(library)
                    return [found] if found else []
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
                # A conditional target lists one path per condition, the
                # first that applies taken: `require` to a `require` only,
                # `import` to an import.
                skipped = "import" if required else "require"
                pending = [target]
                while pending:
                    target = pending.pop(0)
                    if isinstance(target, dict):
                        pending[:0] = [v for k, v in target.items() if k != skipped]
                    elif isinstance(target, str):
                        path = target.replace("*", matched)
                        if found := self._js_file(posixpath.join(folder, path)):
                            return [found]
            return []
        package = "/".join(specifier.split("/")[: 2 if specifier[0] == "@" else 1])
        outside = package in self.builder.js_dependencies
        if outside or specifier.startswith("node:"):
            return _EXTERNAL
        # A name no package can have (`$lib`, `#internal`, `@/x`) is an alias
        # of the project's own.
        return None if _NPM_PACKAGE_RE.fullmatch(package) else []

    def _js_alias(self, specifier, folder):
        """Resolve ``specifier`` through the `paths` and `baseUrl` of the
        tsconfig nearest above ``folder``: ``[file ID]``, ``[]`` when a
        pattern matches but no target is a project module, None when the
        configuration says nothing about it."""
        configs = self.builder.ts_configs
        names = (
            posixpath.join(above, name)
            for above in _ancestors(folder)
            for name in ("tsconfig.json", "jsconfig.json")
        )
        config = next((name for name in names if name in configs), None)
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
        extension, or naming a directory with an index module; a
        declaration file where no module is there."""
        path = posixpath.normpath(path)
        if re.search(r"\.[cm]?tsx?$", path) and f"FILE:{path}" in self.nodes:
            # A TypeScript extension names the file itself.
            return f"FILE:{path}"
        stem = re.sub(r"\.[cm]?[jt]sx?$", "", path)
        for tail in ("", "/index"):
            for extension in (*_JSTS_EXTENSIONS, ".d.ts", ".d.mts", ".d.cts"):
                if f"FILE:{stem}{tail}{extension}" in self.nodes:
                    return f"FILE:{stem}{tail}{extension}"
        return None

    def _dart_file(self, uri, file_id):
        """File ID of the Dart library that ``uri`` names in a directive of
        ``file_id``; None for one outside the project or not indexed."""
        if uri.startswith("package:"):
            package, _, path = uri[len("package:") :].partition("/")
            if package not in self.builder.dart_packages:
                return None
            path = posixpath.join(self.builder.dart_packages[package], "lib", path)
        else:
            path = posixpath.normpath(posixpath.join(file_id[5:], "..", uri))
        return f"FILE:{path}" if f"FILE:{path}" in self.nodes else None

    def _dart_directives(self, library, keyword):
        """``(library files, prefix, names shown, names hidden)`` of each
        directive ``keyword`` of the Dart library ``library``: those of the
        files it names that are the project's, None where no package of the
        project is known and it names one."""
        found = []
        for kind, uris, prefix, show, hide in self.builder.dart_directives.get(
            library, ()
        ):
            if kind == keyword:
                files = [self._dart_file(uri, library) for uri in uris]
                unknown = not self.builder.dart_packages and any(
                    uri.startswith("package:") for uri in uris
                )
                found.append(
                    (None if unknown else [f for f in files if f], prefix, show, hide)
                )
        return found

    def _dart_exports(self, library, name, seen):
        """Top-level definitions the Dart library ``library`` holds under
        ``name``: its own, its parts' and those of the libraries it exports."""
        if library in seen:
            return set()
        seen.add(library)
        units = [library]
        for files, *_ in self._dart_directives(library, "part"):
            units += files or ()
        hits = {self.members.get(unit, {}).get(name) for unit in units} - {None}
        for files, _, show, hide in self._dart_directives(library, "export"):
            if name in show if show else name not in hide:
                for exported in files or ():
                    hits |= self._dart_exports(exported, name, seen)
        return hits

    def _dart_named(self, name, scope_id, prefix=None):
        """Top-level definitions that ``name`` can stand for in the Dart
        library ``scope_id`` belongs to: those the library declares, else
        those its imports bring in, or with ``prefix`` those the imports of
        that prefix do. None where an import cannot be placed in or outside
        the project, so that it may bring the name in."""
        file_id = scope_id.split("::")[0]
        library = self.dart_parts.get(file_id, file_id)
        seen = {
            exported
            for files, *_ in self._dart_directives(library, "export")
            for exported in files or ()
        }
        hits = set() if prefix else self._dart_exports(library, name, seen)
        if hits or name.startswith("_"):
            return sorted(hits)
        for files, named, show, hide in self._dart_directives(library, "import"):
            if named == prefix and (name in show if show else name not in hide):
                if files is None:
                    return None
                for imported in files:
                    hits |= self._dart_exports(imported, name, set())
        return sorted(hits)

    def _dart_prefixed(self, prefix, file_id):
        """Files of the libraries the Dart library ``file_id`` belongs to
        imports as ``prefix``, with those they export: ``"external"`` where
        none is the project's, None where no import has the prefix."""
        library = self.dart_parts.get(file_id, file_id)
        imports = [
            files
            for files, named, *_ in self._dart_directives(library, "import")
            if named == prefix
        ]
        if not imports or None in imports:
            return None
        pending = [imported for files in imports for imported in files]
        found = []
        while pending:
            unit = pending.pop()
            if unit not in found:
                found.append(unit)
                for keyword in ("part", "export"):
                    for files, *_ in self._dart_directives(unit, keyword):
                        pending += files or ()
        return found or _EXTERNAL

    def _dart_root(self, root, steps, scope_id):
        """Value of the name ``root`` that a Dart origin path written in
        ``scope_id`` starts with, and the steps left after it. The name is
        a type written before `()`, or is read in scope: a definition of an
        enclosing scope, a member the enclosing class has or inherits, what
        the library declares or imports, or an import prefix, which names a
        definition together with the step after it. A name none of these
        has is from outside the project."""
        name, generic, _ = root.partition("<")
        if steps[:1] == ["()"]:
            held = self._type_argument(name, scope_id, None)
            if held is not False:
                return held and ("returned", held), steps
            found = self._type_named(name, scope_id)
            made = generic and found and found != _EXTERNAL
            made = made and self._returned_type(root, scope_id)
            if made and len(made) > 2:
                # A generic type written with its arguments holds them.
                return ("def", found, made[2]), steps
            name = name.rpartition("::")[2]
            if found == _EXTERNAL and (generic or name in self.impl_types):
                # Its methods are its own or those of an extension on it;
                # what it holds may be of the type arguments it is written
                # with.
                return ("outside", name), steps
            return found and (found if found == _EXTERNAL else ("def", found)), steps
        hit = self._lexical(name, scope_id, skip_classes=False)
        owner = self._enclosing_class(scope_id)
        if not hit and owner:
            hit = self._find_member(owner, name)[0]
        if not hit:
            imported = self._imported(name, scope_id)
            if imported and imported != _EXTERNAL and steps:
                named = self._dart_named(steps[0], scope_id, name)
                steps = steps[1:]
            else:
                named = [] if imported else self._dart_named(name, scope_id)
            if named is None or len(named) > 1:
                return None, steps
            if not named:
                return _EXTERNAL, steps
            hit = named[0]
        return ("def", hit), steps

    def _exported(self, file_id, name, seen):
        """Definition a file or an object makes available as ``name``: its
        own, one it re-exports from another module or holds by a name it is
        written with, or a module it re-exports whole under that name. What
        a file exports whole, `export = X`, is its default, ``=``, and holds
        every other name it exports."""
        entries = self.builder.reexports.get(file_id, ())
        whole = [original for exported, _, original in entries if exported == "="]
        if whole:
            hit = self.members.get(file_id, {}).get(whole[-1])
            if name in ("=", "default") or not hit:
                return hit
            return self._find_member(hit, name)[0]
        hit = self.members.get(file_id, {}).get(name)
        if hit or file_id in seen:
            return hit
        seen.add(file_id)
        # `export *` passes on neither a default nor what is exported whole.
        star = "*" if name not in ("default", "=") else None
        for exported, specifier, original in entries:
            if exported not in (star, name):
                continue
            if not specifier:
                # `export { a }`, `export default a`, `{ a }` or `{ ...a }`
                held = self._written(original, file_id, seen)
                hit = (
                    held
                    if exported == name
                    else held and self._find_member(held, name)[0]
                )
            modules = specifier and self._js_module(specifier, file_id)
            for module in modules if isinstance(modules, list) else ():
                if original == "*":
                    return module
                hit = hit or self._exported(module, original or name, seen)
            if hit:
                return hit
        return None

    def _written(self, name, scope_id, seen):
        """What ``name`` written in the JS/TS ``scope_id`` stands for: the
        definition in scope, else what an import binds it to, the module
        itself for one bound whole."""
        hit = self._lexical(name, scope_id, skip_classes=True)
        imported = None if hit else self._imported(name, scope_id)
        exported = self._exported_as(name, scope_id)
        for module in imported if isinstance(imported, list) else ():
            hit = hit or self._exported(module, exported, seen)
            hit = hit or (module if exported == "=" else None)
        return hit

    def _resolve_bare(self, name, candidates, source_id, lang, callee):
        file_id = source_id.split("::")[0]
        if lang == "dotnet":
            # A bare C# call reaches a local function, a member of an
            # enclosing type or one of a type named by `using static`, and
            # no method of another type.
            hit = self._inherited(name, source_id)
            hit = hit and self._overload_by_arity(hit)
            for target in self._dotnet_imports(file_id).values():
                held = None if hit else self._type_named(target, file_id)
                if held not in (None, _EXTERNAL):
                    hit = self._find_member(held, name)[0]
                    hit = hit and self._overload_by_arity(hit, static=True)
            if not hit:
                self._result = _OPAQUE
            return (hit, "regex") if hit else (None, _EXTERNAL)
        if lang in _IMPLICIT_SELF_LANGS:
            return self._resolve_self_call(name, candidates, source_id)
        explicit = lang in _EXPLICIT_RECEIVER_LANGS
        hit = self._lexical(name, source_id, skip_classes=explicit)
        if hit and lang == "clang" and self._overloaded(hit, candidates, source_id):
            return None, _UNRESOLVED
        if hit:
            return hit, "regex"
        owner = None if explicit else self._enclosing_class(source_id)
        if owner:
            hit, complete = self._find_member(owner, name)
            if hit:
                return hit, "regex"
            if complete == _SEVERAL:
                return None, _UNRESOLVED
        inherited = False
        around = filter(self._is_class, self._scopes(owner))
        for scope in around if lang == "java" else ():
            # The innermost class around a Java call that has the method is
            # the one called; one with a supertype from outside may have it
            # from there.
            hit, complete = self._find_member(scope, name)
            if hit:
                return (None, _UNRESOLVED) if inherited else (hit, "regex")
            inherited = inherited or not complete
        if inherited:
            self._result = self._std_inherited(owner, name) or _OPAQUE
            return None, _EXTERNAL
        free = [c for c in candidates if self._is_top_level(c)]
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
            # A bare Go call stays inside its package, which is its
            # directory, and one of a variable calls the value it holds.
            package = self._unit(file_id, lang)
            free = [
                c
                for c in free
                if self._unit(c, lang) == package
                and self.nodes[c]["type"] != "Constant"
            ]
        if lang == "clang":
            free = [c for c in free if self._in_reach(c, source_id)]
        if lang == "dart":
            # A bare Dart call that reaches no member of the enclosing class
            # reaches what its library declares or imports, or leaves the
            # project.
            named = self._dart_named(name, file_id)
            free = free if named is None else named
            if not free:
                self._result = _OPAQUE
                return None, _EXTERNAL
        if free:
            hit = self._decide(free, source_id)
            if hit and lang == "clang" and self._overloaded(hit, free, source_id):
                hit = None
            return (hit, "regex") if hit else (None, _UNRESOLVED)
        if lang in _FREE_BARE_CALL_LANGS:
            self._result = _OPAQUE
            return None, _EXTERNAL
        if explicit or lang == "clang":
            # Without a receiver a method is out of reach.
            candidates = [c for c in candidates if self.nodes[c]["type"] != "Method"]
        if lang in _SCRIPT_GLOBAL_LANGS:
            # A bare name is no member of a class, and in a module no
            # global of another file.
            module = (
                file_id in self.builder.imports or file_id in self.builder.reexports
            )
            return self._resolve_member(
                [] if module else candidates, lang, callee, top_level=True
            )
        return self._resolve_member(candidates, lang, callee)

    def _resolve_self_call(self, name, candidates, source_id):
        """Resolve a call made on an implicit `self`: to a member of the
        enclosing type or of what it inherits, else to a top-level
        definition. In a module `self` is an instance of a type that
        includes it, so the call may also reach a member of one of those.
        A block may be run with another `self`, so a name that none of
        these has is not shown to come from outside the project."""
        owner = self._enclosing_class(source_id)
        if owner:
            hit = self._find_member(owner, name)[0]
            source = self.nodes[source_id]["metadata"]
            if source_id == owner or source.get("is_singleton"):
                # In the body of a class and in a singleton method `self`
                # is the class.
                hit = self._find_member(owner, f"self.{name}")[0] or hit
            if hit:
                return hit, "self_type"
            if self.nodes[owner]["metadata"].get("kind") == "module":
                heirs = self._heirs(owner, set())
                hits = {self._find_member(heir, name)[0] for heir in heirs} - {None}
                if hits:
                    return (
                        (hits.pop(), "regex") if len(hits) == 1 else (None, _UNRESOLVED)
                    )
        # A top-level definition is there once its file is loaded.
        loaded = self._loaded(source_id.split("::")[0])
        free = [
            c
            for c in candidates
            if self._is_top_level(c) and c.split("::")[0] in loaded
        ]
        if free:
            hit = self._decide(free, source_id)
            return (hit, "regex") if hit else (None, _UNRESOLVED)
        return None, _UNRESOLVED

    def _loaded(self, file_id):
        """The Ruby files that ``file_id`` loads: itself and those it
        requires, directly or through one of them. A required path names
        the file whose path, without its extension, ends with it."""
        if file_id not in self.loaded:
            paths = {
                other: other[5:].rpartition(".")[0]
                for other in self.nodes
                if "::" not in other and self.lang(other) == "ruby"
            }
            found, todo = {file_id}, [file_id]
            while todo:
                for required in self.requires.get(todo.pop(), ()):
                    required = re.sub(r"^(?:\.{1,2}/)+", "", required).removesuffix(
                        ".rb"
                    )
                    for other, path in paths.items():
                        named = path == required or path.endswith(f"/{required}")
                        if named and other not in found:
                            found.add(other)
                            todo.append(other)
            self.loaded[file_id] = found
        return self.loaded[file_id]

    def _heirs(self, type_id, seen):
        """Ruby types that have ``type_id`` as a base, directly or through
        another type."""
        if self.heirs is None:
            self.heirs = {}
            for declaration, bases in self.bases.items():
                for base in bases if self.lang(declaration) == "ruby" else ():
                    named = self._type_named(base, declaration)
                    if named not in (None, _EXTERNAL):
                        path = named.partition("::")[2]
                        self.heirs.setdefault(path, []).append(declaration)
        found = []
        for heir in self.heirs.get(type_id.partition("::")[2], ()):
            if heir not in seen:
                seen.add(heir)
                found += [heir, *self._heirs(heir, seen)]
        return found

    def _resolve_member(self, candidates, lang, callee, top_level=False):
        """Resolve a call through a receiver whose type is not known, or
        with ``top_level`` a bare call of a global."""
        pool = [
            c
            for c in candidates
            if (
                self._is_top_level(c)
                if top_level
                else self._is_class(c.rpartition("::")[0])
            )
        ]
        namespace = (
            not top_level
            and lang in _NAMESPACE_RECEIVER_LANGS
            and any(map(self._is_top_level, candidates))
        )
        # A call on a receiver whose type is not known is not pinned to the
        # one project definition of the member's name; the definitions that
        # fit stay its candidates. Nor is a bare call whose name the
        # standard library also has pinned to the project's definition, nor
        # dropped.
        if (
            not top_level
            or not pool
            or _split_callee(callee)[1] in _stdlib.members(lang)
        ):
            self._result = _OPAQUE
            if pool:
                self._candidates_now = sorted(pool)
            elif namespace:
                self._candidates_now = sorted(candidates)
            return None, _UNRESOLVED if pool or namespace else _EXTERNAL
        pool = tuple(pool)
        if pool not in self.shared_declaration:
            self.shared_declaration[pool] = self._shared_declaration(pool)
        hit = self.shared_declaration[pool]
        if hit:
            # Resolved by name alone: what the definition returns is not
            # evidence of this call's.
            self._name_only = True
        else:
            if len(pool) > 1:
                self._candidates_now = sorted(pool)
        return (hit, "regex") if hit else (None, _UNRESOLVED)

    def _property_params(self, call, argument):
        """The parameters of the function the property `A:key` of an object
        literal passed to call ``call`` holds: those of the function type
        the property has in the type of the call's parameter `A`."""
        target, declared = self.declared_params.get(call, (None, ()))
        position, *path = argument.split(":")
        if int(position) >= len(declared):
            return ()
        written, scope = declared[int(position)].partition(":")[2].strip(), target
        for key in path:
            written, scope = self._property_type(written, key, scope)
            if not written:
                return ()
        return self._functional_params(written, scope)

    def _property_type(self, written, key, scope_id):
        """``(type, scope)`` of the property ``key`` of the TypeScript type
        ``written`` in ``scope_id``: an object type's, or a member's of the
        type it names, a method's as a function type; None for none."""
        if written.startswith("{") and written.endswith("}"):
            for part in _bracketed_parts(written[1:-1].replace(";", ",")):
                method = _ts_function_property(part) or (
                    re.match(r"\s*([\w$]+)\??\s*(\(.*\))\s*:\s*(.+)", part)
                )
                if method and isinstance(method, tuple) and method[0] == key:
                    name, signature = method
                    params, _, returned = signature[len(name) :].rpartition("):")
                    return f"{params}) => {returned.strip()}", scope_id
                if method and not isinstance(method, tuple) and method[1] == key:
                    return f"{method[2]} => {method[3]}", scope_id
                name, colon, held = part.partition(":")
                if colon and name.strip().rstrip("?") == key:
                    return held.strip(), scope_id
            return None, scope_id
        type_id = self._type_named(written, scope_id)
        if type_id in (None, _EXTERNAL):
            return None, scope_id
        member = self._find_member(type_id, key)[0]
        if member in self.builder.signatures:
            signature = self.builder.signatures[member][0]
            returned = signature["returns"] or "void"
            return f"({', '.join(signature['params'])}) => {returned}", member
        return self.builder.fields.get(type_id, {}).get(key), type_id

    def _functional_params(self, written, scope_id):
        """``[values]`` of the parameters of the function ``written``'s
        type declares: the single abstract method of the project
        functional interface a Java writing names, or the parameters a
        TypeScript function type writes — ``(kind, node ID)`` values stand
        for an interface of that name — the last ``("rest", value)`` where
        that and every later parameter has the element type of a rest one.
        ``()`` where the writing is neither."""
        named = isinstance(written, str) and _ts_type_parts(written)
        if named and named[0] and self.lang(scope_id) == "typescript":
            # An alias stands for the type it names, its type arguments in
            # place of its parameters.
            alias = self._ts_alias(named[0], scope_id)
            if alias and alias[1] != written:
                given = dict(zip(alias[0], named[1]))
                text = re.sub(
                    r"(?<![\w$.])[\w$]+(?![\w$])",
                    lambda found: given.get(found[0], found[0]),
                    alias[1],
                )
                return self._functional_params(text, scope_id)
        if (
            isinstance(written, str)
            and written.startswith(("(", "<"))
            and "=>" in written
            and self.lang(scope_id) == "typescript"
        ):
            # `(...) => ...` or, where an origin path carries the writing,
            # `<...>` in place of `(...)`: the parameter list the function
            # type takes, the parentheses a type is written in aside. A
            # union or an intersection of function types, and a generic
            # one, whose type parameters a parameter's type can name, take
            # none known.
            written = written.strip()
            while True:
                left, right = written[0], ")" if written[0] == "(" else ">"
                depth, close = 0, -1
                for at, char in enumerate(written):
                    if char == ">" and written[at - 1 : at] == "=":
                        continue
                    depth += (char == left) - (char == right)
                    if not depth:
                        close = at
                        break
                rest = written[close + 1 :].strip()
                if close < 0 or rest:
                    break
                written = written[1:close].strip()
                if not written.startswith(("(", "<")):
                    return ()
            if close < 0 or not rest.startswith("=>"):
                return ()
            # A parameter's name is what its list writes before a `:` no
            # bracket holds, and its type what follows that `:`; `this`
            # names the receiver's type, no parameter, and the type of a
            # rest parameter, whose `.` the origin path writes as `::`, is
            # the list's, not that of a parameter it takes in.
            values = []
            for param in _bracketed_parts(written[1:close] + ")"):
                # An origin path writes a rest parameter's `...` as `::::::`.
                rest = param.startswith(("...", "::::::"))
                param = param.removeprefix("...").removeprefix("::::::")
                name = (_bracketed_parts(param + ")", ":") or [""])[0]
                typed = param[len(name) :].lstrip()
                if param.startswith(":") or not typed.startswith(":"):
                    typed = ":"
                typed = typed[1:].strip() or "any"
                if rest:
                    # A rest parameter takes in each later argument, as an
                    # element of its array.
                    element = _ts_type_parts(typed)
                    held = element[1][:1] if element[0] == "Array" else ()
                    value = self._returned_type(held[0], scope_id) if held else None
                    return (*values, ("rest", value))
                if name != "this":
                    values.append(self._returned_type(typed, scope_id))
            return tuple(values)
        if self.lang(scope_id) != "java":
            return ()
        arguments = ()
        type_id = None
        if isinstance(written, tuple) and len(written) > 1:
            type_id = written[1]
        elif isinstance(written, str) and written:
            name, arguments, _ = _java_type_parts(written)
            types = [
                t
                for t in self._types("java", name)
                if self.nodes[t]["metadata"].get("kind") == "interface"
            ]
            type_id = types[0] if len(types) == 1 else None
        if (
            type_id is None
            or self.nodes[type_id]["metadata"].get("kind") != "interface"
        ):
            return ()
        abstract = [
            member
            for member in self.members.get(type_id, {}).values()
            if self.nodes[member]["type"] == "Method"
            and not self.nodes[member]["metadata"].get("body")
            and not self.nodes[member]["metadata"].get("static")
            and not self.nodes[member]["metadata"].get("default")
        ]
        if len(abstract) != 1:
            return ()
        node = self.nodes[type_id]
        generics = node["metadata"].get("generics", ())
        substitute = dict(zip(generics, arguments))
        params = self.nodes[abstract[0]]["metadata"].get("params", ())
        return tuple(
            self._returned_type(substitute.get(param, param), scope_id)
            for param in params
        )

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

    def _resolve_origin(self, origin, source_id, seen=None):
        """Follow an origin path (see ``PythonParser._origin``) to a verdict.
        Returns None when the receiver of the last attribute is of unknown
        type. With ``seen``, the fixtures being evaluated, the path is read
        rather than called: the value it leads to is returned."""
        read = seen is not None
        python = self.lang(source_id) == "python"
        # `path{0=origin}`: what the callable passed as argument 0 of the
        # call the path ends with returns.
        results = {}
        head, brace, passed = origin.rpartition("{")
        if brace and re.fullmatch(r"[\d?]+=[^,=]+(?:,[\d?]+=[^,=]+)*\}", passed):
            origin = head
            results = dict(item.split("=") for item in passed[:-1].split(","))
        value, steps, source = self._root_value(
            origin, source_id, seen if read else set()
        )
        if value is None and origin.startswith("@") and not read:
            return None, _UNRESOLVED
        reads_parameter = steps[:1] and steps[0].startswith("^")
        if value is None and origin.startswith("=") and not reads_parameter:
            # A value held in a variable is no definition. What calling it
            # returns is what the function it holds declares, `>Type`, and
            # not known without that; a parameter of that function is of a
            # type not known.
            declared = origin.partition(">")[2]
            held = declared and self._returned_type(declared, source_id)
            self._result = held or _OPAQUE
            return None, _EXTERNAL
        unwraps = _UNWRAPPING_METHODS.get(self.lang(source_id), ())
        library = self.standard.get(self.lang(source_id))
        # The library's types are classes, as the project's are.
        classed = library and library.lang in ("java", "php", "ruby")
        for index, step in enumerate(steps):
            last = index == len(steps) - 1 and not read
            if value in (_OPAQUE, _EXTERNAL):
                # A call's opaque or outside-typed result: its members are
                # not known.
                break
            if isinstance(value, tuple) and value[0] == "pointer":
                # `->` and a subscript reach what a raw pointer points at, as
                # does a member read after a `*` that the path does not write.
                value = value[1]
                if step in ("?", "[]"):
                    continue
            if step == "[]" and isinstance(value, tuple):
                # A subscript gives what the type's `operator[]` returns: by
                # the data for a standard type, its member's for one of the
                # project.
                if value[0] == "outside":
                    known, held = library.member(value, "operator[]")
                    value = held if known else None
                else:
                    member = self._attr(value, "operator[]", called=True)
                    value = member and self._called(member, value[1])
                continue
            if python and step == "__getitem__" and self._is_class_value(value):
                # A class written with type arguments is that class.
                value = ("returned", value)
                continue
            if isinstance(value, tuple) and value[0] == "library":
                # A name the data has under another names that; any other
                # is a member of what the first gives.
                if f"{value[1]}.{step}" in _py_std_names():
                    value = ("library", f"{value[1]}.{step}")
                    continue
                value = ("returned", self._library_value(value[1]))
            if isinstance(value, tuple) and value[0] == "returned":
                # What a member the data has gives, read or called.
                value = value[1]
                if step == "()":
                    continue
            if python:
                value = self._class_object(value)
            if (
                python
                and step != "()"
                and isinstance(value, tuple)
                and value[0] == "def"
            ):
                # An attribute of a property, or of a function a decorator
                # is applied to, is one of what the name is bound to.
                owner, _, member = value[1].rpartition("::")
                if member in self.builder.fields.get(owner, ()):
                    value = self._variable(owner, member) or value
            if step == "*" or "@" in step:
                value = self._part(value, step, source_id)
                continue
            if (
                isinstance(value, tuple)
                and value[0] == "def"
                and step not in ("()", "?")
                and self.nodes[value[1]]["type"] != "Class"
                and self.lang(source_id) == "dart"
            ):
                # A Dart variable or getter is read as its declared type.
                value = self._declared_type(value[1]) or value
            if classed and isinstance(value, tuple) and value[0] == "def":
                # A field of a standard type is read as the value it holds.
                held = self._declared_type(value[1])
                if isinstance(held, tuple) and held[0] == "outside":
                    value = held
            if step.startswith("^"):
                # `^N`: parameter N of the single abstract method of a
                # functional type — how a Java lambda's parameters are
                # written.
                params = ()
                if isinstance(value, tuple) and value[0] == "functional":
                    params = self._functional_params(value[1], source_id)
                elif isinstance(value, tuple) and value[0] in ("def", "self"):
                    if self._is_class(value[1]):
                        params = self._functional_params(value, source_id)
                    else:
                        held = self.nodes[value[1]]["metadata"].get("value_type")
                        params = self._functional_params(held, source_id)
                number = int(step[1:]) if step[1:].isdigit() else None
                rest = (
                    params and isinstance(params[-1], tuple) and params[-1][0] == "rest"
                )
                value = None
                if number is not None and number < len(params) - bool(rest):
                    value = params[number]
                elif number is not None and rest:
                    value = params[-1][1]
                continue
            if isinstance(value, tuple) and value[0] == "literal":
                verdict = self._each(value[1], steps[index:], source_id)
                if len(value) == 2 or (verdict and verdict[0]):
                    return verdict
                # What the object leads to no definition of is read in the
                # type its function declares.
                value = value[2]
            outside = isinstance(value, tuple) and value[0] == "outside"
            if outside and step not in ("()", "?"):
                # A method of a standard type is the standard library's and
                # returns what the library's data says, unless the project
                # implements a trait with it for that type or for every
                # type. One the type has not is a method of what it
                # dereferences to.
                if python and not last and steps[index + 1 : index + 2] != ["()"]:
                    # An attribute that is read, which the data names
                    # behind a dot.
                    known, held = library.member(value, f".{step}")
                    value = held if known else None
                    continue
                name = value[1].rpartition("::")[2]
                direct = {
                    self._find_member(t, step)[0] for t in self.impl_types.get(name, [])
                }
                blanket = {self._find_member(t, step)[0] for t in self._blanket(name)}
                given = (direct | blanket) - {None}
                known, held = False, value
                if library:
                    known, held = library.member(value, step)
                rust = library and library.lang == "rust"
                if (classed or python or rust) and not known:
                    # A static method is called on the type or on a value; a
                    # Rust one on the type written with its arguments.
                    known, held = library.member(value, step, table="statics")
                if (
                    library
                    and library.lang == "dotnet"
                    and not known
                    and value[1].split("<")[0] == "System.Array"
                    and value[2]
                    and "System.Collections.Generic.IEnumerable`1"
                    in library.data["types"]
                ):
                    # An array is also an IEnumerable<T> of its element:
                    # what that interface's members give is the call's.
                    known, held = library.member(
                        (
                            "outside",
                            "System.Collections.Generic.IEnumerable`1",
                            value[2],
                        ),
                        step,
                    )
                if (
                    library
                    and library.lang == "dotnet"
                    and last
                    and not known
                    and not given
                    and value[1].split("<")[0] in library.data["types"]
                ):
                    # The library's type has no member of the name: a
                    # project extension method written for it is the call's.
                    hits = self._dotnet_extension(value, step)
                    caller_arms = self.call_guards or self.nodes[source_id][
                        "metadata"
                    ].get("guards", ())
                    if caller_arms:
                        hits = [
                            h
                            for h in hits
                            if not self._guard_conflict(
                                caller_arms,
                                self._effective_guards(h, source_id),
                            )
                        ]
                    else:
                        # One every build has is meant before one an `#if`
                        # arm alone builds.
                        guarded = [
                            h for h in hits if self._effective_guards(h, source_id)
                        ]
                        hits = [h for h in hits if h not in guarded] or [
                            h for h in guarded if not self._candidate_only(h, source_id)
                        ]
                        if not hits and guarded:
                            self._candidates_now = sorted(guarded)
                            return None, _UNRESOLVED
                    picks = {
                        h.rpartition("::")[0]: self._overload_by_arity(h, static=True)
                        or h
                        for h in hits
                    }
                    if len(picks) == 1:
                        value = ("def", next(iter(picks.values())))
                        continue
                    if picks:
                        self._candidates_now = sorted(picks.values())
                        return None, _UNRESOLVED
                    # The data has the type's every member: a name it
                    # lacks that no project extension gives is external.
                    value = _EXTERNAL
                    continue
                extended = library and library.lang == "typescript" and len(value) > 2
                if last and extended and not known:
                    # A project type may extend the standard one and be
                    # what the value is: the member may be the project's.
                    self._result = _OPAQUE
                    return None, _UNRESOLVED
                if known or given or held == value:
                    # Every Rust type has `Any`; what implements a trait
                    # has more methods than the trait's.
                    typed = known or len(value) < 3 or library.lang != "rust"
                    typed = typed or library.member(value, "type_id")[0]
                    if last and not given and typed:
                        if (
                            library
                            and library.lang == "dart"
                            and value[1] not in library.data["types"]
                        ):
                            # A Dart type the data does not have may gain
                            # the member through an extension; the name
                            # alone decides it.
                            return None
                        self._result = (known and held) or _OPAQUE
                        if known:
                            self._callables = library.callbacks(value, step)
                        if known and results:
                            self._deferred = (source_id, value, step, results)
                        return None, _EXTERNAL
                    sole = len(given) == 1 and not known
                    value = ("def", given.pop()) if sole else held
                    if sole and not direct:
                        # A trait's method found through an impl that holds
                        # for every type is no evidence of this receiver's
                        # type: what the call returns is not read.
                        self._name_only = True
                    if not known and library and library.lang in ("go", "java", "php"):
                        # The data has every method: this is a field, whose
                        # type is the library's as its type is.
                        value = _EXTERNAL
                    if python and known:
                        value = ("returned", held)
                    continue
                value = held
            if value is None or value == _EXTERNAL:
                break
            if value[0] == "outside":
                if step == "?" and library and library.lang == "typescript":
                    # What `await` gives: what a thenable hands to the first
                    # callable of its `then`, any other value itself.
                    passed = library.callbacks(value, "then").get("0")
                    value = passed[0] if passed else value
                elif step == "?" and library and library.lang == "dart":
                    # `await` gives the type a `Future` or `FutureOr` holds.
                    if (
                        value[1] in ("Future", "FutureOr")
                        and len(value) > 2
                        and value[2]
                    ):
                        value = value[2][0]
                elif step == "?":
                    if (
                        library
                        and library.lang == "clang"
                        and len(value) > 2
                        and value[2]
                    ):
                        # A `unique_ptr`/`optional`/`shared_ptr` hands out
                        # the value it wraps: its first type argument. A
                        # container iterates its `value_type` — `map`'s
                        # `pair<const K, V>`, `vector`'s `T`.
                        entry = library.data["types"].get(value[1], {})
                        spelled = (entry.get("aliases") or {}).get("value_type")
                        if spelled:
                            bound = dict(zip(entry.get("params", ()), value[2]))
                            bound["Self"] = value
                            value = library.value(spelled, bound) or value[2][0]
                        else:
                            value = value[2][0]
                    else:
                        # What Rust's `Try` says the type holds.
                        held = ("Output", "std::ops::Try")
                        value = library and library.associated(value, *held)
            elif step == "()":
                if (
                    outside
                    and library
                    and value[1].split("<")[0] in library.data["types"]
                ):
                    # A standard type called constructs a value of it.
                    pass
                else:
                    value = self._called(value, self._receiver)
            elif value[0] == "union":
                # `A | B`: a member is called on whichever alternative has
                # it — resolve only where every alternative agrees, else
                # the call is genuinely ambiguous.
                if step == "?":
                    continue
                outcomes = [
                    self._attr(one, step, called=last)
                    for one in value[1]
                    if isinstance(one, tuple)
                ]
                if (
                    outcomes
                    and len(outcomes) == len(value[1])
                    and outcomes[0] is not None
                    and all(one == outcomes[0] for one in outcomes)
                ):
                    value = outcomes[0]
                else:
                    self._candidates_now = (
                        sorted(
                            {
                                one[1]
                                for one in outcomes
                                if isinstance(one, tuple) and one[0] == "def"
                            }
                        )
                        or None
                    )
                    return None, _UNRESOLVED
            elif value[0] == "wrapped":
                unwrapped = ("self", value[1])
                if last and step in unwraps:
                    self._result = unwrapped
                    return None, _EXTERNAL
                if last and step != "?" and self.lang(source_id) in ("dotnet", "clang"):
                    # A C# type or a C++ smart pointer from outside has its
                    # own members only; what one returns may be the project
                    # type it wraps.
                    self._result = _OPAQUE
                    return None, _EXTERNAL
                # A C++ smart pointer's own members are from outside.
                own = _EXTERNAL if self.lang(source_id) == "clang" else None
                value = unwrapped if step == "?" else own
            elif step != "?":
                # A value of a generic type carries its type arguments, as
                # does the type written with them, `Boxed<Dog>.new`.
                self._receiver = value[1] if value[0] == "self" else None
                if value[0] in ("self", "def") and len(value) > 2:
                    self._receiver = ("self", *value[1:])
                value = self._attr(value, step, called=last)
                if value is None and last:
                    return None, _UNRESOLVED
        if read:
            returned = isinstance(value, tuple) and value[0] == "returned"
            return value[1] if returned else value
        if python:
            value = self._class_object(value)
        if isinstance(value, tuple) and value[0] == "library":
            self._result = self._library_value(value[1]) or _OPAQUE
            # The data has the callables a function takes under its module.
            module, _, name = value[1].rpartition(".")
            passed = library.data["callbacks"].get(module, {}).get(name, {})
            self._callables = {
                argument: [library.value(written, {}) for written in taken["params"]]
                for argument, taken in passed.items()
            }
            return None, _EXTERNAL
        if isinstance(value, tuple) and value[0] in ("def", "cls"):
            return value[1], source
        if value == _EXTERNAL:
            if not steps:
                # A type from outside constructs a value from outside.
                self._result = _EXTERNAL
            return None, _EXTERNAL
        if classed and not steps and isinstance(value, tuple) and value[0] == "outside":
            # A standard type called by its name makes an instance of it.
            self._result = value
            return None, _EXTERNAL
        if python and isinstance(value, tuple) and value[0] == "self":
            hit = self._find_member(value[1], "__call__")[0]
            if hit:
                return hit, source
        standard = isinstance(value, tuple) and value[0] == "outside"
        if python and (standard or origin.startswith(":") and not steps):
            # A value of a standard type, or of a type that is no class.
            self._result = _OPAQUE
            return None, _EXTERNAL
        # A call of a value held in a variable, or returned by another call.
        if origin.endswith(")") or "." not in origin:
            self._result = _OPAQUE
            return None, _UNRESOLVED
        return None

    def _each(self, literal, steps, source_id):
        """Verdict of a call that the origin steps ``steps`` name on an
        element of the array literal ``literal``: the one that every element
        gives, None where the type of one is not known, unresolved where
        they lead to different definitions."""
        path = "".join(step if step == "()" else f".{step}" for step in steps)
        verdicts, results = set(), set()
        # What a function returns names what is in that function's scope.
        source_id = self.builder.literal_scopes.get(literal, source_id)
        for element in self.builder.literals[literal]:
            origin = element and f"{element}{path}"
            if isinstance(element, dict):
                origin = element.get(steps[0])
                origin = origin and origin + path[len(steps[0]) + 1 :]
            verdicts.add(origin and self._resolve_origin(origin, source_id))
            results.add((self._result, self._receiver))
        if None in verdicts:
            return None
        if len(results) > 1:
            # What the call returns is known where the elements agree on it.
            self._result, self._receiver = _OPAQUE, None
            self._callables, self._deferred = {}, None
        return verdicts.pop() if len(verdicts) == 1 else (None, _UNRESOLVED)

    def _root_value(self, origin, source_id, seen):
        """Evaluate the first step of an origin path written in
        ``source_id``: an imported path behind `@`, `self`, `cls` or
        `super`, `#N` for what the call at edge index N returned, `~name`
        for what pytest passes as the parameter `name`, `=name` for a
        parameter or variable of ``source_id`` (a definition only where it
        is one of ``source_id``'s own) and `=name>Type` for one that holds a
        function declared to return `Type`, `?` for a value of no known
        origin, or a name read in scope, which in Java is read with the
        names after it where it is not called (see ``_java_path``). Returns
        its value, the steps after it, and the resolution source a call
        found through it has. ``seen`` holds the fixtures being evaluated."""
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
        elif root.startswith(":") and not root.startswith("::"):
            # `::App::User` is a name read in scope, PHP's in full.
            value = self._annotated(root[1:], source_id)
        elif root.startswith("#") and "|" in root.partition(">")[0]:
            # `#N|A|P`: parameter P of the callable passed to call N as
            # argument A, or as the only one that is a callable (`?`), or
            # given as the property `key` of an object passed as A (`A:key`);
            # `#N|A|this` is the `ThisType` parameter A declares.
            call, argument, parameter = root[1:].split("|")
            if parameter == "this":
                value = self.this_args.get(int(call), {}).get(argument)
            else:
                callables = self.callables.get(int(call), {})
                if argument == "?" and len(callables) == 1:
                    argument = next(iter(callables))
                params = callables.get(argument, ())
                if ":" in argument:
                    params = self._property_params(int(call), argument)
                value = params[int(parameter)] if int(parameter) < len(params) else None
                if steps[:1] == ["&"]:
                    # A rest parameter: an array of the parameters it gathers,
                    # where they are of one type.
                    gathered = list(params[int(parameter) :])
                    if gathered and gathered[-1] and gathered[-1][0] == "rest":
                        gathered[-1] = gathered[-1][1]
                    one = gathered and all(held == gathered[0] for held in gathered)
                    value = ("outside", "Array", (gathered[0],)) if one else None
                    steps = steps[1:]
        elif root.startswith("#"):
            # `#N>Type`: what the call at edge index N returned, read for
            # the members the type a variable it is bound to is declared
            # with has where the object itself leads to none.
            call, _, declared = root[1:].partition(">")
            if int(call) in self.deferred:
                # The closures the call was passed are resolved by now.
                scope_id, held, name, results = self.deferred.pop(int(call))
                results = {
                    argument: self._root_value(origin, scope_id, set())[0]
                    for argument, origin in results.items()
                }
                library = self.standard[self.lang(scope_id)]
                self.returned[int(call)] = library.member(held, name, results=results)[
                    1
                ]
            if int(call) in self.late:
                # The calls whose results it was passed are resolved by now.
                target, site, scope_id, receiver = self.late.pop(int(call))
                self._bind_call(target, site, scope_id)
                self.returned[int(call)] = self._called(("def", target), receiver)
            value = self.returned.get(int(call))
            if declared and (value is None or value == _EXTERNAL):
                # A call that gives nothing of the project's leaves the name
                # read as the type it is declared with.
                path = "".join(step if step == "()" else f".{step}" for step in steps)
                return self._root_value(f"{declared}(){path}", source_id, seen)
            if declared and isinstance(value, tuple) and value[0] == "literal":
                # The type reads the members the object leads to none of.
                value = (*value[:2], self._plain_value(f"{declared}()", source_id))
        elif root.startswith("~"):
            value = self._fixture_value(root[1:], source_id, seen)
        elif root.startswith(">"):
            # `>Type`: a type written without a variable, as a cast or an
            # assignment gives it — a `^N` step takes its functional
            # parameters.
            value = ("functional", root[1:])
        elif root.startswith("="):
            expr = root[1:]
            if self.lang(source_id) == "clang" and ("<" in expr or "." in expr):
                # `=Type<args>`: a written type a range-for iterates, not a
                # name bound in scope.
                hit = None
                value = self._name_value(expr, source_id, set())
            else:
                hit = self.members.get(source_id, {}).get(expr.partition(">")[0])
                value = ("def", hit) if hit else None
            if not hit and steps and self.lang(source_id) == "dart":
                # `=value.call`: a variable holding `value` is called.
                return self._root_value(origin[1:], source_id, seen)
        elif self.lang(source_id) == "java" and steps and steps[0] != "()":
            value, steps = self._java_path([root, *steps], source_id)
        elif self.lang(source_id) == "dart":
            value, steps = self._dart_root(root, steps, source_id)
        else:
            # A C# name read for a member of it is a member of an enclosing
            # type before it is a type.
            reads = steps[:1] not in ([], ["()"]) and self.lang(source_id) == "dotnet"
            hit = self._inherited(root, source_id) if reads else None
            value = ("def", hit) if hit else self._name_value(root, source_id, set())
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
            parent = posixpath.dirname(folder)
            if parent == folder:
                break
            folder = parent
        for scope in scopes:
            function_id, origin, yields = self.builder.fixtures.get(
                (scope, name), (None, None, False)
            )
            # A fixture that takes a parameter of its own name gets the one
            # it overrides.
            if function_id is None or function_id == scope_id or function_id in seen:
                continue
            seen.add(function_id)
            # What a fixture yields is not the generator it is annotated as.
            value = None if yields else self._called(("def", function_id))
            if value is None and origin:
                value = self._value(origin, function_id, seen)
            return value
        return None

    def _name_value(self, name, scope_id, seen):
        """Value of a name read inside ``scope_id``."""
        lang = self.lang(scope_id)
        if lang == "typescript":
            hit = self._written(name, scope_id, set())
        else:
            hit = self._lexical(name, scope_id, skip_classes=True)
        if hit:
            return ("def", hit)
        if lang != "python":
            # `Type<>` is a type written with type arguments, in Go `Type[]`.
            written = name
            name, generic, _ = name.partition("[" if lang == "go" else "<")
            if self._standard_type(written, scope_id):
                return self._returned_type(written, scope_id)
            package, _, member = name.rpartition("::")
            if lang == "go" and package:
                # `pkg::Name` may be a package-level variable of a project
                # package, read as what it is declared as.
                scopes = self._imported(package, scope_id)
                scopes = scopes if isinstance(scopes, list) else []
                held = (self.members.get(scope, {}).get(member) for scope in scopes)
                held = [h for h in held if h and self.nodes[h]["type"] == "Constant"]
                if held:
                    return self._declared_type(held[0])
            # A type parameter is its bound, or of no type known.
            held = self._type_argument(name, scope_id, None)
            if held is not False:
                return held and ("returned", held)
            given = generic and written.endswith(">")
            generic = generic and lang not in _OWN_METHOD_LANGS
            found = self._type_named(name, scope_id, not generic)
            if isinstance(found, tuple):
                return ("returned", found)
            name = name.rpartition("::")[2]
            if given and found and found != _EXTERNAL:
                # A generic type written with its arguments: an instance of
                # it holds them.
                made = self._returned_type(written, scope_id)
                if made and len(made) > 2:
                    return ("def", found, made[2])
            if generic and found == _EXTERNAL:
                # A wrapper from outside around what may be a project type;
                # in C# its members are its own.
                if lang == "dotnet":
                    written_name, arguments, _ = _dotnet_type_parts(written)
                    key = self._dotnet_std_key(written_name, scope_id)
                    if key:
                        args = tuple(
                            self._returned_type(a, scope_id) or _OPAQUE
                            for a in arguments
                        )
                        return "outside", key, args
                return None
            if found is None and not self._types(lang, name):
                # An alias may stand for a project type, as may what a
                # generic type wraps; one written with its arguments is
                # read as a declared type is.
                held = given and self._returned_type(written, scope_id)
                if held:
                    return ("returned", held)
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
        if name in self.builder.fields.get(file_id, {}):
            return self._variable(file_id, name)
        found, outside = self._starred(file_id, name, seen)
        if found or outside:
            return found.pop() if len(found) == 1 else None
        # Neither defined nor imported: a builtin, if the data has one.
        key = self.standard["python"].data["scope"].get(name)
        return ("library", key) if key else _EXTERNAL

    def _variable(self, owner_id, name):
        """Value of the variable ``name`` that the Python module or class
        ``owner_id`` binds, from the origin paths of what it is bound to:
        the value they agree on, or some standard value where each is one
        from outside the project."""
        key = (owner_id, name)
        if key not in self.variables:
            # A variable bound to itself has no value.
            self.variables[key] = None
            origins = self.builder.fields[owner_id][name]
            values = {self._value(origin, owner_id, set()) for origin in origins}
            outside = (v == _EXTERNAL or v and v[0] == "outside" for v in values)
            if len(values) == 1:
                self.variables[key] = values.pop()
            elif all(outside):
                self.variables[key] = ("outside", "builtins.object", ())
        return self.variables[key]

    def _starred(self, file_id, name, seen):
        """``(values, outside)``: the values that the modules ``file_id``
        imports every public name of have under ``name``, and whether one of
        those modules is not the project's."""
        found, outside = set(), False
        for target in self.builder.imports.get(file_id, {}):
            if not target.endswith(".*") or (file_id, target, name) in seen:
                continue
            seen.add((file_id, target, name))
            dots = len(target) - len(target.lstrip("."))
            names = target.strip(".").split(".")[:-1]
            module = self._import_value(names, dots, file_id)[0]
            if not isinstance(module, tuple) or module[0] != "module":
                outside = True
            elif not name.startswith("_"):
                found.add(self._attr(module, name, seen))
        return found - {None}, outside

    def _value(self, origin, scope_id, seen):
        """Value the origin path ``origin`` leads to in ``scope_id``, read
        while a call is being resolved. ``seen`` holds the fixtures being
        evaluated."""
        state = self._result, self._receiver, self._callables, self._deferred
        value = self._resolve_origin(origin, scope_id, seen)
        self._result, self._receiver, self._callables, self._deferred = state
        return value

    def _is_class_value(self, value):
        """Whether ``value`` is a class, the project's or the standard
        library's, rather than an instance of one."""
        if not isinstance(value, tuple):
            return False
        if value[0] == "library":
            return value[1] in self.standard["python"].data["types"]
        return value[0] == "def" and self._is_class(value[1])

    def _class_object(self, value):
        """The class that ``value`` is where it is an instance of
        `type[C]`: C as a project class, None for any other. Every other
        value is returned as it is."""
        if not isinstance(value, tuple) or value[:2] not in _PY_CLASS_OBJECTS:
            return value
        held = value[2][0] if value[2] else None
        project = isinstance(held, tuple) and held[0] == "self"
        return ("cls", held[1]) if project else None

    def _library_value(self, key):
        """What the name ``key`` of Python's standard library gives, called
        or read: an instance of the class, what the function returns or the
        variable holds, ``"external"`` for a module."""
        library = self.standard["python"]
        if key in library.data["types"]:
            return ("outside", key, (None,) * len(library.data["types"][key]["params"]))
        functions = library.data["functions"]
        returns = functions.get(key, functions.get(f".{key}", _EXTERNAL))
        if returns in (None, _EXTERNAL):
            return returns
        return library.value(returns, {})

    def _annotated(self, written, scope_id, receiver=None):
        """Value of an instance of the Python type ``written`` in
        ``scope_id``, as an annotation has it: of a project class, of a
        standard type that holds what its type arguments are,
        ``"external"``, or None. `Self` is ``receiver``, the type the method
        was called on, where that is known."""
        library = self.standard["python"]
        name, arguments, _ = library.parts(written)
        if name == "Self":
            owner = receiver or self._enclosing_class(scope_id)
            return owner and ("self", owner)
        held = tuple(self._annotated(a, scope_id, receiver) for a in arguments)
        if not name:
            # A union of standard types is some standard value.
            standard = (isinstance(it, tuple) and it[0] == "outside" for it in held)
            if held and all(standard):
                return ("outside", "builtins.object", ())
            return None
        value = self._value(name, scope_id, set())
        if isinstance(value, tuple) and value[0] == "library":
            if value[1] not in library.data["types"]:
                return None
            return ("outside", value[1], held)
        if isinstance(value, tuple) and value[0] == "def":
            return ("self", value[1]) if self._is_class(value[1]) else None
        return value if value == _EXTERNAL else None

    def _py_std_key(self, name, scope_id):
        """What the Python name ``name``, written in ``scope_id`` or as an
        origin path, is called in the standard library's data."""
        value = self._value(name, scope_id, set()) if name else None
        library = isinstance(value, tuple) and value[0] == "library"
        return value[1] if library else None

    def _blanket(self, name):
        """The blanket impls that are for the outside type ``name``: those
        whose every bound it may meet. A trait the project declares it meets
        only where the project implements that trait for it."""
        met = {
            _split_callee(base)[1]
            for given in self.impl_types.get(name, ())
            for base in self.bases.get(given, ())
        }
        return [
            given
            for given in self.impl_types.get("", ())
            if all(
                bound in met or not self._declares_type(bound)
                for bound in self.nodes[given]["metadata"]["bounds"]
            )
        ]

    def _declares_type(self, name):
        """Whether the project declares a Rust type ``name``, rather than
        only implementing traits for one."""
        return any(
            self.nodes[t]["metadata"].get("kind") != "impl"
            for t in self._types("rust", name)
        )

    def _rust_std_key(self, name, scope_id):
        """What the Rust name ``name`` written in ``scope_id`` is called in
        the standard library's data: from the crate its first segment is
        imported from (`io::Result` is `std::io::Result` under `use
        std::io`), or by the prelude unless the project declares a type or
        an alias of that name."""
        file_id = scope_id.split("::")[0]
        if name in self.builder.rust_aliases.get(file_id, ()):
            return None
        root, _, rest = name.partition("::")
        root = self.builder.imports.get(file_id, {}).get(root, root)
        path = f"{root}::{rest}" if rest else root
        if path == name and not self._declares_type(name):
            path = self.standard["rust"].data["scope"].get(name, name)
        return path

    def _part(self, value, step, scope_id):
        """Value of the part of ``value`` that a pattern binds a name to.
        The step `*` is what iterating ``value`` gives, `Name@key` what the
        variant or the struct `Name` holds at ``key``, a position or a
        field name, and `@key` an element of a tuple. None where the type
        of ``value`` does not say."""
        if value == _EXTERNAL and self.lang(scope_id) == "go":
            # What a Go package outside the project hands out is its own.
            return _EXTERNAL
        if (
            self.lang(scope_id) == "php"
            and isinstance(value, tuple)
            and value[0] == "def"
        ):
            # A PHP property is read as the value it holds.
            value = self._declared_type(value[1])
        if not isinstance(value, tuple) or value[0] not in ("outside", "self"):
            return None
        name, _, key = step.partition("@")
        key = int(key) if key.isdigit() else key
        if value[0] == "self":
            # A struct is matched by its own name, an enum by a variant's.
            owner = value[1]
            if name and name != self._name(owner):
                owner = self.members.get(owner, {}).get(name)
            for declaration in self._declarations(owner) if owner else ():
                written = self.builder.fields.get(declaration, {}).get(key)
                if written:
                    return self._returned_type(written, declaration, value)
            return None
        if len(value) < 3:
            return None
        library = self.standard.get(self.lang(scope_id))
        if step == "*" and value[1] == "iterable":
            # PHP's own name for what a `foreach` reads.
            return value[2][1] if value[2][1:] else None
        if step == "*" and library and library.lang in _ITERATED:
            return library.associated(value, _ITERATED[library.lang])
        if step == "*":
            # The first type argument of the iterator a type hands out.
            if not library:
                return None
            iterator = library.member(value, "[Symbol.iterator]")[1]
            return iterator[2][0] if iterator and iterator[2] else None
        if name:
            return library.held(value, name, key) if library else None
        if library and library.lang == "python":
            # A tuple has the type its items share before those of each.
            if value[:2] not in _PY_TUPLES:
                return None
            key = key + 1 if len(value[2]) > 1 else 0
        return value[2][key] if key in range(len(value[2])) else None

    def _go_variable(self, name, scope_id):
        """The package-level Go variable ``name`` that ``scope_id`` has in
        scope and that has a declared type: in any file of its package."""
        package = self._unit(scope_id, "go")
        found = [
            c
            for c in self.by_name.get(("go", name), ())
            if self.nodes[c]["metadata"].get("value_type")
            and self._is_top_level(c)
            and self._unit(c, "go") == package
        ]
        return found[0] if len(found) == 1 else None

    def _go_std_key(self, name, scope_id):
        """What the Go name ``name`` written in ``scope_id`` is called in
        the standard library's data: `http.Request` is `net/http.Request`
        under `import "net/http"`, and a predeclared name is itself unless
        the project declares a type of that name."""
        package, dot, member = name.partition(".")
        imports = self.builder.imports.get(scope_id.split("::")[0], {})
        if dot:
            return f"{imports.get(package, package)}.{member}"
        return None if self._types("go", name) else name

    def _dotnet_std_key(self, name, scope_id):
        """What the C# name ``name``, written in ``scope_id``, is called in
        the standard library's data: the type it names fully, or one of
        an enclosing namespace or of a namespace a `using` gives by that
        name, or the language alias for it — unless the name names a
        project type in scope."""
        types = self.standard["dotnet"].data["types"]
        name = name.replace("::", ".")
        if name in types:
            return name
        file_id = scope_id.split("::")[0]
        imports = self._dotnet_imports(file_id)
        head, dot, rest = name.partition(".")
        if dot:
            aliased = imports.get(head)
            return (
                f"{aliased}.{rest}"
                if aliased and f"{aliased}.{rest}" in types
                else None
            )
        aliased = imports.get(head)
        if aliased and not aliased.endswith(".*"):
            return aliased if aliased in types else None
        # A project type of the name is in scope before a standard one;
        # several it could mean still shadow it.
        if self._dotnet_type([], name, scope_id) != _EXTERNAL:
            return None
        namespace = (
            self.nodes.get(file_id, {}).get("metadata", {}).get("namespace") or ""
        )
        spaces = [
            ".".join(namespace.split(".")[:size])
            for size in range(namespace.count(".") + 1)
        ]
        imported = [k[: -len(".*")] for k in imports if k.endswith(".*")]
        for parent in [*spaces, *imported, ""]:
            key = f"{parent}.{name}".lstrip(".")
            if key in types:
                return key
        # `string`, `int` and the language's other aliases are its types'.
        return self.standard["dotnet"].data["scope"].get(name)

    def _clang_std_key(self, name, scope_id):
        """What the C or C++ name ``name``, written in ``scope_id``, is
        called in the standard library's data: `std::vector` is the type
        `std::vector`1`, `vector` the same under `using namespace std`,
        `printf` the C function of that name — unless the name names a
        project type."""
        data = self.standard["clang"].data
        name = name.lstrip(":")
        if name in data["types"] or name in data["functions"]:
            return name
        base = name.rpartition("`")[0]
        written = base or name
        if written.startswith("std::"):
            return next(
                (
                    key
                    for key in (
                        f"std::{written[5:]}`{number}" for number in range(9, -1, -1)
                    )
                    if key in data["types"]
                ),
                written if written in data["functions"] else None,
            )
        if self._types("clang", written.split("::")[-1]) or (
            written in self.builder.c_aliases or written in self.builder.type_aliases
        ):
            return None
        for key in (
            f"std::{written}",
            *(f"std::{written}`{number}" for number in range(9, -1, -1)),
        ):
            if key in data["types"]:
                return key
        scoped = data["scope"].get(name) or data["scope"].get(written)
        return (
            self._clang_std_key(scoped, scope_id)
            if scoped and scoped not in (name, written)
            else None
        )

    def _php_std_key(self, name, scope_id):
        """What the PHP class or function ``name``, written in full, is
        called in the standard library's data: itself, unless the project
        declares a class of that name."""
        declared = self._type_named(f"::{name}".replace("\\", "::"), scope_id)
        return name if declared == _EXTERNAL else None

    def _ruby_std_key(self, name, scope_id):
        """What the Ruby class or module ``name``, written bare, is called
        in the standard library's data: itself, unless the project declares
        a constant of that name in scope."""
        declared = self._constant([*name.split("::")], scope_id)
        return name if declared == _EXTERNAL else None

    def _dart_std_key(self, name, scope_id):
        """What the Dart name ``name``, written in ``scope_id``, is called
        in the standard library's data: itself, unless the project's own
        libraries or its `package:` imports declare a type of that name in
        scope."""
        named = self._dart_named(name, scope_id)
        return None if named else name

    def _ts_std_key(self, name, scope_id):
        """What the JavaScript name ``name`` written in ``scope_id`` is
        called in the standard library's data: a global is itself or the
        type it is declared as, unless its file binds the name or the
        project declares a type of that name that is named there."""
        root = name.partition(".")[0]
        bound = self.builder.js_bound.get(scope_id.split("::")[0], ())
        declared = root in self.builder.type_aliases or self._types_in(
            "typescript", root, scope_id
        )
        if root in bound or declared:
            return None
        return self.standard["typescript"].data["scope"].get(name, name)

    def _js_global(self, callee, source_id):
        """``(True, the value returned)`` where the JavaScript ``callee``,
        which has no origin, is a function of the standard library, a
        member of one of its types or a method that the type of one of its
        global objects has."""
        library = self.standard["typescript"]
        owner, _, name = callee.rpartition(".")
        root = owner.partition(".")[0]
        if not owner:
            return library.function(callee, source_id)
        key = library.named(owner, source_id)
        if key and library.key(owner, source_id) == owner:
            return True, library.member(("outside", key, ()), name, table="statics")[1]
        if key:
            return library.member(("outside", key, ()), name)
        # A path through a type, `Object.prototype.toString.call`.
        own = library.key(root, source_id) == root
        return bool(own and library.named(root, source_id)), None

    def _java_std_key(self, name, scope_id):
        """What the Java type ``name`` written in ``scope_id`` is called in
        the standard library's data: by the import of its first name
        (`Map.Entry` is `java.util.Map.Entry` under `import java.util.Map`),
        in a package or class imported on demand, or as a class of
        `java.lang`, unless a project type of that name is in scope."""
        head, dot, rest = name.partition(".")
        imports = self.builder.imports.get(scope_id.split("::")[0], {})
        if head in imports:
            return f"{imports[head]}{dot}{rest}"
        if self._java_type(head, scope_id):
            return None
        data = self.standard["java"].data
        for star, package in imports.items():
            if star.endswith("*") and f"{package}.{name}" in data["types"]:
                return f"{package}.{name}"
        return f"{data['scope'].get(head, head)}{dot}{rest}"

    def _java_type(self, name, scope_id):
        """The project type that ``name`` stands for where it is written
        in ``scope_id``, a Java type name that no import of the file binds:
        a member type of a class around ``scope_id`` or of one of its
        supertypes, a type of the same package, one of a package or class
        imported on demand, or the class of that full name."""
        file_id = scope_id.split("::")[0]
        key = (self._enclosing_class(scope_id) or file_id, name)
        if key in self.java_scoped:
            return self.java_scoped[key]
        # Inside its own lookup, for a supertype, the name stands for nothing.
        self.java_scoped[key] = None
        head, *path = name.split(".")
        package = self.nodes[file_id]["metadata"].get("java_package")
        around = filter(self._is_class, self._scopes(key[0]))
        found = [self._find_member(scope, head)[0] for scope in around]
        found += [
            t
            for t in self._types("java", head)
            if self._is_top_level(t)
            and self.nodes[t.split("::")[0]]["metadata"].get("java_package") == package
        ]
        for star, imported in self.builder.imports.get(file_id, {}).items():
            if star.endswith("*"):
                found.append(self.java_types.get(f"{imported}.{head}"))
                if imported in self.java_types:
                    found.append(self._find_member(self.java_types[imported], head)[0])
        hit = next((f for f in found if f and self._is_class(f)), None)
        for segment in path if hit else ():
            hit = hit and self._find_member(hit, segment)[0]
        hit = hit or self.java_types.get(name)
        self.java_scoped[key] = hit if hit and self._is_class(hit) else None
        return self.java_scoped[key]

    def _java_path(self, names, scope_id):
        """Split a dotted Java path read in ``scope_id``, whose first name
        no local declares, into the value it starts with and the names left
        to read from it. The first name is a field of a class around
        ``scope_id`` or of one of its supertypes, failing that a type or
        what an import binds; a longer start may spell out a class with its
        package. ``"external"`` for a path into a package of the standard
        library that names no type of its data, None for any other."""
        root = names[0]
        owner = self._enclosing_class(scope_id)
        if (owner, root) not in self.java_fields:
            # Java keeps a field that shares a method's name under `field_`.
            found = (
                self._find_member(scope, name)[0]
                for scope in filter(self._is_class, self._scopes(owner))
                for name in (root, f"field_{root}")
            )
            fields = (
                hit for hit in found if hit and self.nodes[hit]["type"] == "Constant"
            )
            self.java_fields[owner, root] = next(fields, None)
        if field := self.java_fields[owner, root]:
            # Read without a receiver, it is of the type as written, a type
            # parameter's bound for one.
            written = self.nodes[field]["metadata"]["value_type"]
            held = self._type_argument(written, field, None)
            if held is False:
                held = self._name_value(written.replace(".", "::"), field, set())
            return held, names[1:]
        # A standard type, written with the types it is nested in.
        sizes = range(len(names) - 1, 0, -1)
        nested = (
            n for n in sizes if self._standard_type(".".join(names[:n]), scope_id)
        )
        if size := next(nested, 0):
            return self._returned_type(".".join(names[:size]), scope_id), names[size:]
        imported = self._imported(root, scope_id)
        if imported == _EXTERNAL:
            return _EXTERNAL, []
        found = imported[0] if imported else self._java_type(root, scope_id)
        if found:
            return ("def", found), names[1:]
        for size in range(len(names) - 1, 1, -1):
            dotted = ".".join(names[:size])
            if dotted in self.java_types:
                return ("def", self.java_types[dotted]), names[size:]
            if dotted in self.standard["java"].data["types"]:
                return ("outside", dotted, ()), names[size:]
        return (_EXTERNAL if root in _java_packages() else None), []

    def _standard_type(self, written, scope_id):
        """Key in the standard library's data of the type ``written`` in
        ``scope_id``; None for a type the data does not have."""
        library = self.standard.get(self.lang(scope_id))
        return library and library.named(written, scope_id)

    def _std_call(self, callee, source_id):
        """``(True, the value returned)`` where ``callee``, a Rust path or
        a Go name, names a function of the standard library or of one of its
        types, or in Go a type of it, which converts to an instance."""
        library = self.standard[self.lang(source_id)]
        if library.lang == "go":
            named = library.named(callee, source_id)
            if named:
                return True, ("outside", named, ())
            return library.function(callee, source_id)
        owner, _, name = callee.rpartition("::")
        owner = library.named(owner, source_id)
        if owner:
            return library.member(("outside", owner, ()), name, table="statics")
        if library.lang == "clang":
            base, arguments, _ = library.parts(callee)
            key = library.key(base, source_id)
            functions = library.data["functions"]
            if key not in functions:
                return False, None
            returns = functions[key]
            if returns is None:
                return True, None
            # The arguments the call writes are the parameters the
            # written return names, in their order of appearance.
            params = []
            for token in re.findall(r"\b_[A-Z]\w*", returns):
                if token not in params:
                    params.append(token)
            bound = dict(
                zip(
                    params,
                    (self._returned_type(a, source_id) for a in arguments),
                )
            )
            return True, library.value(returns, bound)
        return library.function(callee, source_id)

    def _py_root(self, module, file_id):
        """The directory the module ``module`` imported in ``file_id`` is
        below: the importer's own where the module is there, else the only
        one it is below; None where it is below several others."""
        roots = self.py_modules[module]
        own = self.py_root_of.get(file_id)
        if own in roots:
            return own
        return next(iter(roots)) if len(roots) == 1 else None

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
                root = self._py_root(module, file_id)
                if root is None:
                    return None, []
                return ("module", module, root), steps[size:]
        if dots or names[0] in self.py_roots:
            return None, []
        if names[0] in _py_std_names():
            return ("library", names[0]), steps[1:]
        return _EXTERNAL, []

    def _attr(self, value, name, seen=None, called=False):
        """Value of attribute ``name`` of ``value``; ``called`` says the
        attribute is the callee itself rather than part of its receiver."""
        kind, ref = value[:2]
        if kind == "outside":
            # A standard type's member is the library's, read by none of this.
            return None
        if kind == "library":
            dotted = f"{ref}.{name}"
            return ("library", dotted) if dotted in _py_std_names() else _EXTERNAL
        if kind == "module":
            root = value[2]
            if root in self.py_modules.get(f"{ref}.{name}", ()):
                return ("module", f"{ref}.{name}", root)
            file_id = self.py_modules[ref][root]
            hit = self.members.get(file_id, {}).get(name)
            if hit:
                return ("def", hit)
            seen = set() if seen is None else seen
            # A name the module itself imports is re-exported by it.
            if name in self.builder.imports.get(file_id, {}):
                return self._name_value(name, file_id, seen)
            if name in self.builder.fields.get(file_id, {}):
                return self._variable(file_id, name)
            found = self._starred(file_id, name, seen)[0]
            return found.pop() if len(found) == 1 else None
        if kind == "def" and self.nodes[ref]["type"] != "Class":
            # A field read gives an instance of the type it is declared as;
            # one declared no type holds what it is written with.
            held = self._declared_type(ref)
            hit = not held and self._exported(ref, name, set())
            if hit:
                return ("def", hit)
            if held == _EXTERNAL:
                return _EXTERNAL
            if held and held[0] == "outside":
                # A field declared a type of the standard library has that
                # type's members; a C# field of it may extend them.
                library = self.standard.get(self.lang(ref))
                if not library:
                    return held if not called else None
                known, val = library.member(held, name)
                if known:
                    self._result = val or _OPAQUE
                    return _EXTERNAL if called else val
                if (
                    called
                    and self.lang(ref) == "dotnet"
                    and held[1].split("<")[0] in library.data["types"]
                ):
                    hits = self._dotnet_extension(held, name)
                    picks = {
                        h.rpartition("::")[0]: self._overload_by_arity(h, static=True)
                        or h
                        for h in hits
                    }
                    if len(picks) == 1:
                        return ("def", next(iter(picks.values())))
                    if picks:
                        self._candidates_now = sorted(picks.values())
                        return None
                    self._result = _OPAQUE
                    return _EXTERNAL
                return None
            if called and held and self.lang(ref) == "dotnet" and held[0] != "self":
                # A C# type from outside has its own members only; what
                # one returns may be the project type it wraps.
                self._result = _OPAQUE
                return _EXTERNAL
            if not held or held[0] != "self":
                return None
            if len(held) > 2:
                # A member of a generic type is read on the instance the
                # field holds, with its arguments.
                self._receiver = held
            return self._attr(held, name, seen, called)
        hit, complete = self._find_member(ref, name, inherited_only=kind == "super")
        declared = hit and not called and len(value) > 2 and self.nodes[hit]["metadata"]
        if declared and declared.get("value_type"):
            # A field of a generic type's value is of its type with the
            # value's arguments.
            return self._returned_type(declared["value_type"], hit, value)
        python = self.lang(ref) == "python"
        if hit and not called and self.nodes[hit]["type"] == "Method":
            # Java keeps a field that shares a method's name under `field_`;
            # a Rust one is read as its struct declares it.
            hit = self._find_member(ref, f"field_{name}")[0] or hit
            fields = (self.builder.fields.get(d, {}) for d in self._declarations(ref))
            if self.lang(ref) == "rust" and any(name in f for f in fields):
                hit = None
        if hit and called and self.lang(hit) == "dotnet":
            hit = self._overload_by_arity(hit, static=kind == "def")
            if not hit:
                return None
        if kind in ("def", "cls") and self.lang(ref) == "ruby":
            # A Ruby class that has a singleton method and an instance
            # method of one name has the first as `self.name`, and `new`
            # makes an instance of a class that defines no `new` of its own.
            hit = self._find_member(ref, f"self.{name}")[0] or hit
            made = self.nodes[ref]["metadata"].get("kind") != "module"
            if not hit and name == "new" and made:
                return ("cls", ref)
        if hit:
            return ("def", hit)
        if (
            called
            and kind == "self"
            and self.lang(ref) == "dart"
            and self.nodes[ref]["metadata"].get("kind") == "typedef"
            and name == "call"
        ):
            # A typedef is a function type; calling its `call` calls it.
            self._result = _OPAQUE
            return _EXTERNAL
        # A Go struct has, besides its methods and those of what it embeds,
        # only fields: one that is called holds a function, and the call
        # returns what the field's type, `>Type`, declares.
        if called and complete and self.lang(ref) == "go":
            for declaration in self._declarations(ref):
                if self.nodes[declaration]["metadata"].get("kind") == "struct":
                    written = self.builder.fields.get(declaration, {}).get(name) or ""
                    declared = written.startswith(">") and written[1:]
                    held = declared and self._returned_type(declared, declaration)
                    self._result = held or _OPAQUE
                    return _EXTERNAL
        for owner in self._lineage(ref, set()) if python else ():
            if name in self.builder.fields.get(owner, ()):
                # What is called is a definition the attribute holds.
                held = self._variable(owner, name)
                kept = held == _EXTERNAL or held and held[0] in ("def", "library")
                return held if kept or not called else None
        if called and name in self.extensions:
            # A C# extension method written for the type, for a type it
            # derives from, or for every type where the type has no base
            # from outside that may have the member itself; none where
            # those of several classes are.
            written = {self._name(ref), *self._ancestors(ref)}
            if complete:
                written.add("object")
            hits = [
                e
                for e in self.extensions[name]
                if self.nodes[e]["metadata"]["extends"] in written
            ]
            if hits:
                # An extension an `#if` arm alone defines can be meant
                # only under that arm.
                matching = [
                    e
                    for e in hits
                    if not self._guard_conflict(
                        self.call_guards or (),
                        self.nodes[e]["metadata"].get("guards", ()),
                    )
                ]
                if not self.call_guards:
                    matching = [
                        e for e in matching if not self._effective_guards(e, ref)
                    ] or [e for e in matching if not self._candidate_only(e, ref)]
                    if not matching:
                        self._candidates_now = sorted(hits)
                        return None
                hits = matching
            if hits:
                owners = {e.rpartition("::")[0] for e in hits}
                return ("def", hits[0]) if len(owners) == 1 else None
        # A C/C++ field that is called holds a function, or an object that
        # is called.
        if called and self.lang(ref) == "clang" and self._c_field(ref, name):
            self._result = _OPAQUE
            return _EXTERNAL
        # A method missing from a type with a base outside the project is
        # inherited from that base, and may return the type; a missing data
        # attribute need not be inherited.
        if called and not complete:
            self._result = self._std_inherited(ref, name) or _OPAQUE
            return _EXTERNAL
        if python and not complete:
            return ("returned", self._std_inherited(ref, name))
        if called and self.lang(ref) == "clang":
            # A member the type declares without a definition is defined by
            # the types derived from it: the call reaches one of theirs.
            mine = self._name(ref)
            derived = (
                hit
                for hit in self.by_name.get(("clang", name), ())
                if mine in self._ancestors(hit.rpartition("::")[0])
            )
            self._candidates_now = sorted(derived) or None
        if called or kind != "self":
            return None
        return self._part(value, f"{self._name(ref)}@{name}", ref)

    def _declared_type(self, node_id):
        """Value of the type the variable ``node_id`` is declared as, read
        as ``_returned_type`` reads a return type."""
        declared = self.nodes[node_id]["metadata"].get("value_type")
        called = self.nodes[node_id]["metadata"].get("value_call")
        if called and not declared:
            # A Dart variable without a type holds what the call that is
            # its value returns.
            root, *steps = called.split(".")
            value, steps = self._dart_root(root, steps, node_id)
            for step in steps:
                if not isinstance(value, tuple):
                    break
                if value[0] == "outside":
                    # A standard type's member is the library's.
                    known, value = self.standard["dart"].member(value, step)
                    value = value if known else None
                else:
                    value = self._attr(value, step)
            if isinstance(value, tuple) and value[0] == "outside":
                # What the library says the standard method called returns
                return value
            return self._called(value) if isinstance(value, tuple) else value
        return declared and self._returned_type(declared, node_id)

    def _type_named(self, name, scope_id, full=False):
        """The project type a possibly qualified type name written in
        ``scope_id`` refers to, ``"external"`` for one an import places
        outside the project, None when it cannot be told. With ``full`` a
        C++ alias gives the value of the type it names, its template
        arguments and pointer with it."""
        lang = self.lang(scope_id)
        if lang == "python":
            # A base class is written as its origin path.
            if (scope_id, name) not in self.base_types:
                value = self._value(name, scope_id, set())
                found = None
                if value == _EXTERNAL or value and value[0] == "library":
                    found = _EXTERNAL
                elif value and value[0] == "def" and self._is_class(value[1]):
                    found = value[1]
                self.base_types[scope_id, name] = found
            return self.base_types[scope_id, name]
        if lang == "php":
            # The parser writes a PHP class by its full name, which only the
            # class declared in that namespace has: `::App::Models::User`.
            namespace, _, name = name.removeprefix("::").rpartition("::")
            namespace = namespace.replace("::", "\\")
            types = [
                t
                for t in self._types(lang, name)
                if self.nodes[t]["metadata"].get("namespace") == namespace
            ]
            return self._decide(types, scope_id) if types else _EXTERNAL
        parts = [part for part in re.split(r"::|\.", name) if part]
        if not parts:
            # A base that is the value of an expression names no type.
            return None
        *qualifier, name = parts
        if lang == "dotnet":
            return self._dotnet_type(qualifier, name, scope_id)
        if lang == "ruby":
            return self._constant([*qualifier, name], scope_id)
        if lang == "clang":
            return self._c_type(qualifier, name, scope_id, full)
        imported = self._imported(qualifier[0] if qualifier else name, scope_id)
        if imported == _EXTERNAL:
            return _EXTERNAL
        if lang == "java" and imported is None:
            return self._java_type(".".join([*qualifier, name]), scope_id)
        types = [] if qualifier else [i for i in imported or () if self._is_class(i)]
        if not types and imported:
            # The import names the scope the type is in.
            exported = name if qualifier else self._exported_as(name, scope_id)
            found = (self._find_member(scope, exported)[0] for scope in imported)
            found = (self.companions.get(hit, hit) for hit in found)
            types = [hit for hit in found if hit and self._is_class(hit)]
        if lang == "dart" and (imported or not qualifier):
            # Only the library's own types and those it imports are in scope.
            named = self._dart_named(name, scope_id, *qualifier[:1])
            if named is not None:
                types = [hit for hit in named if self._is_class(hit)]
                if not types:
                    # `void` and `dynamic` say nothing of the value.
                    unknown = named or name in ("void", "dynamic")
                    return None if unknown else _EXTERNAL
        if not types:
            types = self._types_in(lang, name, scope_id)
            if lang == "typescript" and not types and self._types(lang, name):
                # Is none of them in reach, the name is not the project's:
                # the writing names the language's global.
                return _EXTERNAL
        if lang == "rust" and qualifier:
            crate = self._use_crate("::".join(qualifier), scope_id.split("::")[0])
            if crate == ():
                return _EXTERNAL
            if crate:
                # A crate may re-export the type from another one; which of
                # several of that name is then not told.
                inside = [t for t in types if (self._unit(t, lang),) == crate]
                if not inside and len(types) > 1:
                    return None
                types = inside or types
        hit = self._decide(types, scope_id) if types else None
        if hit is None and len({self._unit(t, lang) for t in types}) == 1:
            # Declarations of one split type, in one crate or package.
            hit = types[0]
        return hit

    def _dotnet_imports(self, file_id):
        """``{name or "<path>.*": dotted path}`` of what the `using`
        directives in force in the C# file ``file_id`` name: under a name
        what an alias stands for, under a path with `.*` the namespace or
        the type whose members are in scope."""
        folder = posixpath.dirname(file_id[len("FILE:") :])
        imports = dict(self.builder.imports.get(file_id, {}))
        for directory, usings in self.builder.dotnet_global_usings.items():
            if not directory or f"{folder}/".startswith(f"{directory}/"):
                imports.update(usings)
        return imports

    def _dotnet_path(self, type_id):
        """Dotted name of the C# type ``type_id`` from the global namespace."""
        parts = type_id.split("::")
        namespace = self.nodes["::".join(parts[:2])]["metadata"].get("namespace")
        names = [
            self._name("::".join(parts[:size])) for size in range(2, len(parts) + 1)
        ]
        return ".".join(filter(None, [namespace, *names]))

    def _dotnet_type(self, qualifier, name, scope_id):
        """The project type that the C# name ``name`` means where it is
        written behind ``qualifier`` in ``scope_id``: one nested in an
        enclosing type or in a base of one, else one of the nearest
        enclosing namespace that has one, else one of a namespace the file
        imports. ``"external"`` where the project declares no type the
        name can mean, None where it declares one that is not in scope or
        several that are."""
        file_id = scope_id.split("::")[0]
        imports = self._dotnet_imports(file_id)
        aliased = imports.get(qualifier[0] if qualifier else name)
        if aliased:
            written = [*qualifier[1:], name] if qualifier else []
            *qualifier, name = [*aliased.split("."), *written]
        paths = {t: self._dotnet_path(t) for t in self._types("dotnet", name)}
        hits = []
        if qualifier:
            written = ".".join(["", *qualifier, name])
            hits = [t for t, path in paths.items() if f".{path}".endswith(written)]
            if not hits:
                return _EXTERNAL
        if not paths:
            return _EXTERNAL
        nested = [t for t in paths if self._is_class(t.rpartition("::")[0])]
        for scope in self._scopes(scope_id):
            if not hits and self._is_class(scope):
                owners = {self._name(scope), *self._ancestors(scope)}
                hits = [
                    t for t in nested if self._name(t.rpartition("::")[0]) in owners
                ]
        if not hits:
            top = "::".join(scope_id.split("::")[:2])
            metadata = self.nodes[top if top in self.nodes else file_id]["metadata"]
            enclosing = metadata.get("namespace") or metadata.get("dotnet_namespace")
            parts = enclosing.split(".") if enclosing else []
            spaces = [".".join(parts[:size]) for size in range(len(parts), -1, -1)]
            outer = {t: p for t, p in paths.items() if self._is_top_level(t)}
            for space in [*spaces, None]:
                within = {space} if space is not None else {*imports.values()}
                hits = [
                    t for t, path in outer.items() if path.rpartition(".")[0] in within
                ]
                if hits:
                    break
        if len({paths[t] for t in hits}) == 1:
            return hits[0]
        return None

    def _c_type(self, qualifier, name, scope_id, full=False):
        """``_type_named`` for C and C++, which import no names: the type
        is the one of that name nearest in scope, else the one among the
        files ``scope_id`` is compiled with, and a name that no type or
        alias of the project carries is from outside it. An alias is the
        type it names where it is declared."""
        types = [
            t
            for t in self._types("clang", name)
            if self.nodes[t]["metadata"].get("kind") != "namespace"
        ]
        qualifier = [q for q in qualifier if q not in self.c_spaces]
        aliased = {name, *qualifier[-1:]} & self.builder.type_aliases
        aliases = self.builder.c_aliases.get(name, ())
        seen = self._sees(scope_id.split("::")[0])
        if qualifier:
            inside = [
                t for t in types if self._name(t.rpartition("::")[0]) == qualifier[-1]
            ]
            # The qualifier may be an alias of the type the member is in.
            owner = not inside and self._c_type(qualifier[:-1], qualifier[-1], scope_id)
            if owner and owner != _EXTERNAL:
                owners = self._declarations(owner)
                inside = [t for t in types if t.rpartition("::")[0] in owners]
            types = inside
            aliases = [a for a in aliases if self._name(a[0]) == qualifier[-1]]
        else:
            # A member of an enclosing scope, or of the type that an
            # enclosing scope holds the out-of-class members of.
            for scope in self._scopes(scope_id):
                scopes = self._declarations(scope) if self._is_class(scope) else [scope]
                inner = {self.members.get(s, {}).get(name) for s in scopes}
                if inner & set(types):
                    hit = (inner & set(types)).pop()
                    return None if self._overloaded(hit, types, scope_id) else hit
                written = [a for a in aliases if a[0] in scopes]
                if written:
                    types, aliases = [], written
                    break
            # A member of any other type takes that type's name to reach,
            # as does an alias of a namespace that is not around the scope.
            types = [t for t in types if self._is_top_level(t)]
            path = scope_id.split("::")[1:]
            aliases = [
                a for a in aliases if (at := a[0].split("::")[1:]) == path[: len(at)]
            ]
        known = bool(types)
        types = [t for t in types if self._in_reach(t, scope_id)]
        if not types:
            # The aliases agree on the type, whatever its arguments; one of
            # a pointer names the type it points at.
            named, values = set(), set()
            for scope, written in aliases:
                if scope.split("::")[0] in seen:
                    value = self._returned_type(written, scope)
                    values.add(value)
                    held = value[1] if value and value[0] == "pointer" else value
                    named.add(held[:2] if isinstance(held, tuple) else held)
            held = named.pop() if len(named) == 1 else None
            if full and len(values) == 1 and held and held[0] == "self":
                return values.pop()
            if held == _EXTERNAL or (held and held[0] == "self"):
                return held if held == _EXTERNAL else held[1]
            return None if aliased or known else _EXTERNAL
        # The innermost of the namespaces around the scope that has one.
        scope = scope_id.split("::")[1:]
        depth = {t: len(t.split("::")) - 2 for t in types}
        inside = [t for t in types if t.split("::")[1:-1] == scope[: depth[t]]]
        types = [t for t in inside if depth[t] == max(map(depth.get, inside))] or types
        # Two definitions are different types: specialisations of a template.
        return types[0] if len({self._unit(t, "clang") for t in types}) == 1 else None

    def _sees(self, file_id):
        """IDs of the C/C++ files ``file_id`` is compiled with: those that
        it, and every file that includes it, include in any number of
        steps."""
        if file_id not in self.units:
            units, sees = {file_id}, set()
            for edges, found in ((self.includers, units), (self.includes, sees)):
                pending = list(units)
                found.update(pending)
                while pending:
                    for other in edges.get(pending.pop(), ()):
                        if other not in found:
                            found.add(other)
                            pending.append(other)
            self.units[file_id] = sees
        return self.units[file_id]

    def _in_reach(self, node_id, scope_id):
        """Whether C/C++ code in ``scope_id`` can name ``node_id``. A type,
        a macro, a function with internal linkage and whatever a header
        defines are known only to the files compiled with the one that
        holds them."""
        node = self.nodes[node_id]
        kind = node["metadata"].get("kind")
        file_id = node_id.split("::")[0]
        local = (
            kind == "macro"
            or node["metadata"].get("linkage") == "internal"
            or (node["type"] == "Class" and kind != "namespace")
            or file_id.endswith(_C_HEADER_EXTENSIONS)
        )
        return not local or file_id in self._sees(scope_id.split("::")[0])

    def _c_field(self, owner_id, name):
        """Whether the C/C++ type ``owner_id`` declares a field ``name``,
        itself or through a macro that stands in its body for fields."""
        declared = set()
        for declaration in self._declarations(owner_id):
            declared.update(self.builder.fields.get(declaration, ()))
        pending = [key[1:-1] for key in declared if key.startswith("<")]
        while pending and name not in declared:
            for node_id in self.by_name.get(("clang", pending.pop()), ()):
                for field, macro in self.nodes[node_id]["metadata"].get("fields", ()):
                    if (field or f"<{macro}>") not in declared:
                        declared.add(field or f"<{macro}>")
                        pending += [macro] if macro else []
        return name in declared

    def _c_rewritten(self, macro, index, scope_id, seen=()):
        """How the C/C++ macros named ``macro`` that ``scope_id`` can name
        write their parameter ``index`` (see ``clang_parser._rewrites``),
        followed through the macros they pass it on to."""
        found = set()
        for node_id in self.by_name.get(("clang", macro), ()):
            rewrites = self.nodes[node_id]["metadata"].get("rewrites", ())
            if (macro, index) in seen or not self._in_reach(node_id, scope_id):
                continue
            for at, how in rewrites:
                if str(at) == index and how.endswith(")"):
                    passed = (*how[:-1].split("("), scope_id, {*seen, (macro, index)})
                    found |= self._c_rewritten(*passed)
                elif str(at) == index:
                    found.add(how)
        return found

    def _overloaded(self, hit, candidates, scope_id):
        """Whether the C++ definition ``hit`` is one of several of its
        qualified name that ``scope_id`` is compiled with: an overload or a
        specialisation in another file is among ``candidates``. Definitions
        an `#if` arm alone builds are alternatives, not overloads."""
        path = hit.split("::")[1:]
        seen = self._sees(scope_id.split("::")[0])
        defined = {c for c in candidates if "end_lineno" in self.nodes[c]["metadata"]}
        macro = self.nodes[hit]["metadata"].get("kind") == "macro"
        guards = self.call_guards or self.nodes[scope_id]["metadata"].get("guards", ())
        return (
            hit in defined
            and not macro
            and any(
                c != hit
                and c in defined
                and c.split("::")[1:] == path
                and c.split("::")[0] in seen
                and not self._guard_conflict(
                    guards, self._effective_guards(c, scope_id)
                )
                for c in candidates
            )
        )

    def _constant(self, path, scope_id, seen=frozenset()):
        """The definition that the Ruby constant ``path``, a list of names
        written in ``scope_id``, refers to. Its first name is looked up in
        each enclosing class or module, innermost first, then in what the
        innermost one inherits from, then at the top level; the rest of
        the path names what that defines. ``"external"`` for a path the
        project does not define, None when it cannot be told. ``seen``
        holds the types whose bases are being looked up."""
        if self.constants is None:
            # Path of each Ruby type and constant -> its declarations, and
            # the paths that only prefix one: `A` of a `class A::B`.
            self.constants, self.namespaces = {}, set()
            for node_id, node in self.nodes.items():
                if self.lang(node_id) == "ruby" and node["type"] in (
                    "Class",
                    "Constant",
                ):
                    written = node_id.partition("::")[2]
                    self.constants.setdefault(written, []).append(node_id)
                    while written := written.rpartition("::")[0]:
                        self.namespaces.add(written)
        namespaces = []
        scope = scope_id.partition("::")[2]
        while scope:
            namespaces.append(scope)
            scope = scope.rpartition("::")[0]
        innermost = next((n for n in namespaces if n in self.constants), None)
        inherited = [] if innermost in seen else [innermost]
        for namespace in inherited:
            for declaration in self.constants.get(namespace, ()):
                for base in self.bases.get(declaration, ()):
                    named = self._constant(
                        base.split("::"), declaration, seen | {innermost}
                    )
                    if named and named != _EXTERNAL:
                        found_path = named.partition("::")[2]
                        if found_path not in inherited and found_path not in namespaces:
                            inherited.append(found_path)
        for namespace in [*namespaces, *inherited[1:], ""]:
            root = f"{namespace}::{path[0]}" if namespace else path[0]
            if root in self.constants or root in self.namespaces:
                break
        else:
            return None if self.by_name.get(("ruby", path[0])) else _EXTERNAL
        found = self.constants.get("::".join([root, *path[1:]]))
        if found:
            # Of several declarations the one in the file is the nearest.
            own = [f for f in found if f.split("::")[0] == scope_id.split("::")[0]]
            found = (own or found)[0]
            # A constant assigned a constant is another name for it.
            alias = self.nodes[found]["metadata"].get("value")
            if alias and found not in seen:
                return self._constant(alias.split("::"), found, seen | {found})
            return found
        # A constant of a type's base is that type's too.
        open_ended = any(d in self.bases for d in self.constants.get(root, ()))
        return None if open_ended and path[1:] else _EXTERNAL

    def type_at(self, path, scope_id):
        """The project type that the Rust path ``path``, written in
        ``scope_id``, leads to; ``"external"`` where it leads outside the
        project, None where it does not tell."""
        found = self._type_named(path, scope_id)
        if found in (None, _EXTERNAL):
            return found
        # A type the project only writes impl blocks for is declared outside.
        declared = self.nodes[found]["metadata"].get("kind") != "impl"
        return found if declared else _EXTERNAL

    def base_type(self, index):
        """The project type that the base the INHERITS edge ``index`` names
        stands for where its subtype is declared. None for a type outside
        the project, where several fit and nothing tells which, and where
        the name leads back to the subtype, which is no base of itself."""
        subtype = self.builder.edges[index]["source"]
        # The types several fit are no call's candidates.
        reached = self._candidates_now
        named = self._type_named(self.base_names[index], subtype)
        self._candidates_now = reached
        return None if named in (_EXTERNAL, subtype) else named

    def type_in_scope(self, name, scope_id, origin=None):
        """The project type that ``name``, written in ``scope_id``, is shown
        to be by its origin path, by an import or by a definition in an
        enclosing scope. None where none of them shows one: a name alone
        does not."""
        if origin:
            verdict = self._resolve_origin(origin, scope_id)
            found = verdict[:1] if verdict else ()
        else:
            name = _split_callee(name)[1]
            found = [self._lexical(name, scope_id, skip_classes=True)]
            imported = self._imported(name, scope_id)
            if imported not in (None, _EXTERNAL):
                exported = self._exported_as(name, scope_id)
                found += imported
                found += [self._find_member(scope, exported)[0] for scope in imported]
        for hit in found:
            # A type the project only writes impl blocks for is declared
            # outside it.
            if (
                hit != scope_id
                and self._is_class(hit)
                and self.nodes[hit]["metadata"].get("kind") != "impl"
            ):
                return hit
        return None

    def _called(self, value, receiver=None):
        """Value returned by calling ``value``, a method of the type
        ``receiver`` where that is known."""
        kind, ref = value[:2]
        if kind == "cls" or (kind == "def" and self.nodes[ref]["type"] == "Class"):
            return ("self", ref, *value[2:])
        if kind != "def":
            return None
        returns = self.nodes[ref]["metadata"].get("return_type") or ""
        body = None
        if ref in self.builder.signatures:
            # As written, of the signature the call being kept is of.
            chosen = self._bound[2] if ref == self._bound[0] else None
            chosen = chosen or self._overload(ref, {})
            returns, body = chosen["returns"], chosen.get("returned")
        if self.lang(ref) != "python":
            # A value declared under the name of an interface constructs it,
            # unless its return type says otherwise.
            companion = self.companions.get(ref)
            found = self._returned_type(returns, ref, receiver)
            if body:
                # The object a body builds is what is returned, whatever
                # type it is declared as.
                made = self._body_value(ref, receiver)
                if not returns:
                    found = made
                elif made and made[0] == "literal":
                    found = (*made, found)
            found = found or (companion and ("self", companion)) or self._alias(ref)
            return found or self._forwarded(ref)
        # The annotation is read where the function is defined.
        scope_id = ref.rpartition("::")[0]
        held = self.builder.fields.get(ref, {})
        for maker in _PY_CONTEXT_MANAGERS:
            if maker in held:
                # The context manager the data says the decorator makes,
                # of what the function yields.
                library = self.standard["python"]
                made = library.parts(library.data["functions"][maker])[1][-1]
                return (
                    "outside",
                    library.parts(made)[0],
                    (self._variable(ref, maker),),
                )
        taken = self.nodes[ref]["metadata"].get("param_types") or [""]
        fluent = held.get("()") == ["self"]
        if self._is_class(scope_id) and (fluent or returns == taken[0] != "Any"):
            # What its `self` is, or is annotated as, is the type it is
            # called on.
            return ("self", receiver or scope_id)
        if "()" in held:
            return self._variable(ref, "()")
        return self._annotated(returns, scope_id, receiver)

    def _body_value(self, function_id, receiver):
        """Value that every `return` in the body of ``function_id``, which
        declares no return type, gives when called on ``receiver``: `this`,
        what a call or a construction returns, or a name of a declared
        type. None where one of them is not known or they differ."""
        returned = self.builder.signatures[function_id][-1]["returned"]
        key = function_id, receiver
        receiver = receiver[1] if isinstance(receiver, tuple) else receiver
        if key not in self.body_values:
            # A function that returns its own call has no value from that.
            self.body_values[key] = None
            values = []
            for callee, origin in (record or ("", "?") for record in returned):
                if origin == "self" and not callee:
                    owner = receiver or self._enclosing_class(function_id)
                    values.append(owner and ("self", owner))
                elif origin in self.builder.literals:
                    values.append(("literal", origin))
                elif not callee:
                    values.append(self._plain_value(origin, function_id))
                else:
                    state = self._result, self._receiver, self._deferred
                    state += self._callables, self._bound
                    scope = receiver if callee.startswith("this.") else None
                    target = self.resolve(callee, scope or function_id, origin)[0]
                    values.append(self.result(target))
                    self._result, self._receiver, self._deferred = state[:3]
                    self._callables, self._bound = state[3:]
            if (
                values
                and values[0] is not None
                and values.count(values[0]) == len(values)
            ):
                self.body_values[key] = values[0]
        return self.body_values.get(key)

    def _forwarded(self, function_id):
        """Value that ``function_id`` returns where its parser recorded,
        as the ``(callee, origin)`` of a call written in it, the one call
        whose result every way out of it hands back."""
        call = self.nodes[function_id]["metadata"].get("returned")
        if not call or function_id in self._forwarding:
            return None
        self._forwarding.add(function_id)
        state = self._result, self._receiver, self._callables, self._deferred
        self._result = self._receiver = None
        target, how = self._verdict(*call[:1], function_id, call[1])
        if target:
            value = self._called(("def", target), self._receiver)
        elif how == _EXTERNAL and self._result != _OPAQUE:
            value = self._result or _EXTERNAL
        else:
            value = None
        self._result, self._receiver, self._callables, self._deferred = state
        self._forwarding.discard(function_id)
        return value

    def _alias(self, constant_id):
        """Value returned by calling a constant whose value is the name of
        a function or method: what that one returns."""
        node = self.nodes[constant_id]
        path = str(node["metadata"].get("value") or "").strip().rstrip(";")
        if node["type"] != "Constant" or not re.fullmatch(r"[\w$]+(?:\.[\w$]+)*", path):
            return None
        result, receiver = self._result, self._receiver
        target = (self._resolve_origin(path, constant_id) or (None,))[0]
        self._result, self._receiver = result, receiver
        if target and self.nodes[target]["type"] != "Constant":
            return self._called(("def", target))
        return None

    def _returned_type(self, returns, function_id, receiver=None):
        """Value of the declared return type ``returns`` of ``function_id``:
        an instance of a project type, a wrapper around one (`Result<T, E>`,
        `Promise<T>`), ``"external"``, or None. `this` and `Self` are
        ``receiver``, the type the method was called on, where that is known."""
        instance = receiver if isinstance(receiver, tuple) else None
        receiver = instance[1] if instance else receiver
        if self.lang(function_id) == "python":
            return self._annotated(returns, function_id, receiver)
        if function_id == self._bound[0] and returns in self._bound[1]:
            return self._bound[1][returns]
        if self.builder.literals.get(returns):
            return ("literal", returns)
        queried = self.lang(function_id) == "typescript" and _TS_TYPEOF_RE.fullmatch(
            returns.strip()
        )
        if queried:
            # `typeof x` is the type of the value `x`: a value of it is `x`.
            root, *steps = re.split(r"\.|::", queried[1])
            held = self._written(root, function_id, set())
            for step in steps:
                held = held and self._find_member(held, step)[0]
            return ("def", held) if held else None
        compound = self.lang(function_id) == "typescript" and _ts_compound(returns)
        if compound:
            values = [
                self._returned_type(part.strip(), function_id, instance or receiver)
                for part in compound[1]
            ]
            if compound[0] == "|":
                # The value's type is one of the alternatives: members it
                # calls are looked up on each of them.
                return ("union", tuple(values))
            if compound[0] == "?":
                # A conditional type is the type its branches agree on.
                return values[0] if values.count(values[0]) == len(values) else None
            # An intersection is the operand derived from every other.
            known = [value for value in values if value is not None]
            for value in known:
                if all(
                    other == value
                    or "self" == value[0] == other[0]
                    and self._name(other[1]) in self._ancestors(value[1])
                    for other in known
                ):
                    return value
            return None
        parts = _bracketed_parts(returns) if "," in returns else []
        if self.lang(function_id) == "dotnet" and len(parts) > 1:
            # A C# method that hands values to its `out` arguments returns
            # them beside its result, which a `#N.@P` origin picks up.
            return (
                "outside",
                "()",
                tuple(
                    self._returned_type(part, function_id, instance or receiver)
                    for part in parts
                ),
            )
        if self.lang(function_id) == "clang" and re.search(r">\s*\*$", returns):
            # A raw pointer to a template's instance
            held = self._returned_type(
                returns.rstrip()[:-1], function_id, instance or receiver
            )
            return held and ("pointer", held)
        library = self.standard.get(self.lang(function_id))
        name = library and library.named(returns, function_id)
        if name:
            # A standard type holds what its type arguments are.
            arguments = library.parts(returns)[1]
            held = (
                self._returned_type(a, function_id, instance or receiver)
                for a in arguments
            )
            return ("outside", name, tuple(held))
        if self.lang(function_id) == "rust":
            returns = _RUST_BORROW_RE.sub("", returns)
        match = re.match(
            r"[&*\s]*(?:'\w+\s+)?(?:mut\s+)?([\w$.:]+)\s*"
            r"(<\s*(?:&\s*)?(?:'\w+\s+)?(?:mut\s+)?([\w$.:]+)?)?",
            returns,
        )
        if not match:
            return None
        owner = receiver or self._enclosing_class(function_id)
        held = self._type_argument(match[1], function_id, instance or receiver)
        if held is not False:
            return held
        standard = None
        for kind, name in zip(("self", "wrapped"), match.group(1, 3)):
            if name in ("Self", "this"):
                if kind == "self" and instance and not match[2]:
                    # The receiver itself, with its type arguments
                    return instance
                found = owner
            else:
                plain = kind == "self" and not match[2]
                found = name and self._type_named(name, function_id, plain)
                if isinstance(found, tuple):
                    # An alias is the type it names, written as it writes it.
                    return found
            if found == _EXTERNAL and kind == "self" and match[2]:
                # A type from outside written with a type argument: its
                # methods are its own, unless it wraps a project type.
                standard = ("outside", name)
                continue
            lang = self.lang(function_id)
            if (
                not found
                and kind == "self"
                and self._types(lang, re.split(r"::|\.", name)[-1])
            ):
                # A project type that cannot be told apart wraps nothing.
                return None
            if found == _EXTERNAL:
                return _EXTERNAL if kind == "self" else standard
            generic = found and kind == "self" and match[2]
            if generic and self.nodes[found]["metadata"].get("generics"):
                # A generic type written with its arguments holds them; one
                # no import places outside, as a function's own type
                # parameter, is not known.
                held = []
                for a in _bracketed_parts(returns[returns.index("<") + 1 :]):
                    if a.startswith("'"):
                        # A lifetime takes no type parameter's place.
                        continue
                    value = self._returned_type(a, function_id, instance or receiver)
                    imported = self._imported(a.partition("<")[0].strip(), function_id)
                    outside = value != _EXTERNAL or imported == _EXTERNAL
                    held.append(value if outside else None)
                return kind, found, tuple(held)
            if found:
                return (kind, found)
        return standard

    def _type_argument(self, name, scope_id, receiver):
        """What the type parameter ``name`` of a type around ``scope_id``
        stands for: the argument ``receiver``, a value of that type or of
        one derived from it or the ID of such a type, gives it, else its
        bound, the receiver's own type for one bound by a type of itself
        (`B extends Builder<B>`), None for none; False where no type around
        declares the parameter."""
        instance = receiver if isinstance(receiver, tuple) else None
        instance = instance or (receiver and ("self", receiver))
        for scope in self._scopes(scope_id):
            metadata = self.nodes.get(scope, {}).get("metadata", {})
            generics = list(metadata.get("generics") or ())
            if name in generics and self._is_class(scope):
                given = self._instance_arguments(scope, instance)
                at = generics.index(name)
                if at < len(given):
                    # An instance's argument, known or not: its bound only
                    # limits what it may be.
                    return given[at]
                bound = (metadata.get("type_bounds") or {}).get(name)
                where = scope
                if not bound:
                    # A Rust `impl` block bounds them for its own items.
                    held = self.nodes.get(scope_id, {}).get("metadata", {})
                    bound, where = held.get("impl_bounds", {}).get(name), scope_id
                if bound and re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", bound):
                    return instance and instance[:2]
                return self._returned_type(bound, where) if bound else None
        return False

    def _instance_arguments(self, type_id, instance):
        """The type arguments ``instance`` gives the generic type
        ``type_id``: its own where it is one, those its type writes a base
        of that type with where it derives from it."""
        if not instance:
            return ()
        own, given = instance[1], instance[2] if len(instance) > 2 else ()
        if type_id in self._declarations(own):
            return given
        reached = self._candidates_now
        for base in self.written_bases.get(own, ()):
            name, opened, written = base.partition("<")
            if opened and self._type_named(name, own) == type_id:
                self._candidates_now = reached
                held = ("self", own, given)
                return tuple(
                    self._returned_type(a, own, held) for a in _bracketed_parts(written)
                )
        self._candidates_now = reached
        return ()

    def _scopes(self, node_id):
        """Yield ``node_id`` and each scope enclosing it, out to its file."""
        while node_id:
            yield node_id
            node_id = node_id.rpartition("::")[0]

    def _is_class(self, node_id):
        node = self.nodes.get(node_id)
        # A Ruby module has members as a class has; a Rust one is a scope.
        # A callback's owner is a scope source cannot name, not a type: its
        # members are read where they are written, not through a receiver.
        return bool(
            node
            and node["type"] == "Class"
            and (
                node["metadata"].get("kind") not in ("module", "namespace", "callback")
                or self.lang(node_id) == "ruby"
            )
        )

    def _is_top_level(self, node_id):
        parent = self.nodes.get(node_id.rpartition("::")[0])
        return (
            parent is None
            or parent["type"] == "File"
            or parent["metadata"].get("kind") == "namespace"
        )

    def _named_from(self, node_ids, scope_id):
        """Those of the TypeScript definitions ``node_ids`` that are named
        where ``scope_id`` is written. An ECMA-262 declaration is
        block-scoped: one written inside a function or a callback is named
        only inside it."""
        around = set(self._scopes(scope_id))
        return [
            node_id
            for node_id in node_ids
            if all(
                scope in around
                for scope in self._scopes(node_id.rpartition("::")[0])
                if self._is_function_scope(scope)
            )
        ]

    def _is_function_scope(self, node_id):
        """Whether ``node_id`` is a scope names written inside it stay in:
        a function, a method or a callback."""
        node = self.nodes.get(node_id)
        return bool(
            node
            and (
                node["type"] in ("Function", "Method")
                or node["metadata"].get("kind") == "callback"
            )
        )

    def _enclosing_class(self, node_id):
        for scope in self._scopes(node_id):
            if self._is_class(scope):
                return scope
            # The body of a C/C++ macro definition is a member's of the type
            # that an argument of the macro names.
            words = self.nodes.get(scope, {}).get("metadata", {}).get("arguments", ())
            for word in words:
                named = self._c_type([], word, scope)
                if named and named != _EXTERNAL:
                    return named
        return None

    def _inherited(self, name, scope_id):
        """What the innermost scope enclosing ``scope_id`` that has a
        ``name`` has under it: a function local to a function, or a member
        of a type, its own or one of a base."""
        for scope in self._scopes(scope_id):
            if self._is_class(scope):
                hit = self._find_member(scope, name)[0]
            else:
                hit = self.members.get(scope, {}).get(name)
            if hit and (self._is_class(scope) or not self._is_class(hit)):
                return hit
        return None

    def _overload_by_arity(self, hit, static=None):
        """The C# method that the call being resolved, which reached the
        method ``hit``, can be a call of: the first of ``hit``, its
        overloads and then those the bases of its type have that takes as
        many arguments as the call passes and, with ``static`` given, is
        static or not as that says. ``hit`` itself where it is no method."""
        owner, _, name = hit.rpartition("::")
        seen = set()
        while hit and owner not in seen:
            seen.add(owner)
            overloads = [hit]
            while (
                more := f"{owner}::overload{len(overloads) + 1}_{name}"
            ) in self.nodes:
                overloads.append(more)
            for overload in overloads:
                metadata = self.nodes[overload]["metadata"]
                if "parameters" not in metadata:
                    return overload
                least, most = metadata["parameters"]
                passed = least if self._arguments is None else self._arguments
                takes = least <= passed and (most is None or passed <= most)
                if takes and static in (None, metadata["static"]):
                    return overload
            hit = self._find_member(owner, name, inherited_only=True)[0]
            owner = hit and hit.rpartition("::")[0]
        return None

    def _lexical(self, name, scope_id, skip_classes):
        """Definition of ``name`` in the innermost scope enclosing ``scope_id``.
        What a constant holds, and what an object literal holds, are no
        scopes: each is reached through the name it is written for."""
        for scope in self._scopes(scope_id):
            if skip_classes and self._is_class(scope):
                continue
            node = self.nodes.get(scope)
            if node and (
                node["type"] == "Constant" or node["metadata"].get("kind") == "object"
            ):
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

    def _types_in(self, lang, name, scope_id):
        """The types of ``_types`` that ``name`` can stand for where it is
        written in ``scope_id``: in TypeScript those named there, and of
        them the ones the innermost function or callback declares, whose
        declaration shadows one outside it."""
        types = self._types(lang, name)
        if lang != "typescript":
            return types
        depth = {
            t: sum(map(self._is_function_scope, self._scopes(t)))
            for t in self._named_from(types, scope_id)
        }
        return [t for t in depth if depth[t] == max(depth.values())]

    def _find_member(self, owner_id, name, inherited_only=False, seen=None):
        """Look ``name`` up in a type or module and then in the type's bases.

        Returns ``(node_id, complete)``; ``complete`` is False when a base is
        not a project type, so a miss may be a member defined outside, and
        `_SEVERAL` when the types a base may be have several members of the
        name, which the miss leaves as the candidates.
        """
        held = self.nodes[owner_id]["metadata"].get("kind") == "namespace" and (
            self._declared_type(owner_id)
        )
        # A value of a namespace's name has the members of its type the
        # namespace has not.
        if isinstance(held, tuple) and held[0] == "self":
            own = self.members.get(owner_id, {}).get(name)
            return (own, True) if own else self._find_member(held[1], name, seen=seen)
        if not self._is_class(owner_id) or (
            self.nodes[owner_id]["metadata"].get("kind") == "object"
        ):
            # A variable has the members of the type it is declared as, an
            # object those it is written with.
            held = self._declared_type(owner_id)
            if isinstance(held, tuple) and held[0] == "self":
                return self._find_member(held[1], name, seen=seen)
            return self._exported(owner_id, name, set()), True
        lang = self.lang(owner_id)
        seen = set() if seen is None else seen
        if owner_id in seen:
            return None, True
        declarations = self._declarations(owner_id)
        seen.update(declarations)
        if not inherited_only:
            hits = [self.members.get(d, {}).get(name) for d in declarations]
            hits = [hit for hit in hits if hit]
            for d in declarations if lang == "typescript" and not hits else ():
                # A field typed `typeof f` holds `f`.
                written = self.builder.fields.get(d, {}).get(name, "")
                held = _TS_TYPEOF_RE.fullmatch(written) and self._returned_type(
                    written, d
                )
                hits += [held[1]] if held else []
            # A pure virtual function that is defined as well is called as
            # its definition; any other two are alternatives of the build.
            defined = [h for h in hits if not self.nodes[h]["metadata"].get("pure")]
            hits = defined or hits
            if lang == "clang" and len(hits) > 1:
                return None, True
            if hits:
                return hits[0], True
        # A type the project only writes impl blocks for is declared outside.
        kinds = {self.nodes[d]["metadata"].get("kind") for d in declarations}
        complete = kinds != {"impl"}
        for declaration in declarations:
            for base in self.bases.get(declaration, ()):
                # The base is the type its name stands for where the
                # subtype is declared; every type of that name when that
                # cannot be told. Those types are not what the call may reach.
                reached = self._candidates_now
                named = self._type_named(base, declaration)
                self._candidates_now = reached
                bare = _split_callee(base)[1]
                if named == _EXTERNAL:
                    base_types = []
                else:
                    base_types = (
                        [named] if named else self._types_in(lang, bare, declaration)
                    )
                if not base_types:
                    # A standard trait gives only the members it has.
                    trait = self._standard_type(bare, declaration)
                    given = (trait, declaration, name, owner_id)
                    complete = (
                        complete and bool(trait) and not self._std_given(*given)[0]
                    )
                hits = set()
                for base_type in base_types:
                    hit, base_complete = self._find_member(base_type, name, seen=seen)
                    if base_complete == _SEVERAL:
                        return None, _SEVERAL
                    hits.add(hit)
                    complete = complete and base_complete
                # Where the base may be any of several types, the member is
                # the one of those that has it; where several do, the call
                # reaches one of theirs.
                hits.discard(None)
                if len(hits) == 1:
                    return hits.pop(), True
                if hits:
                    self._candidates_now = sorted(hits)
                    return None, _SEVERAL
            for trait in self.builder.derived.get(declaration, ()):
                given = (trait, declaration, name, owner_id)
                complete = complete and not self._std_given(*given)[0]
        return None, complete

    def _declarations(self, owner_id):
        """The type ``owner_id`` and its other declarations, in a language
        that spreads one type's members over several."""
        lang = self.lang(owner_id)
        if lang == "dart":
            return [owner_id, *self.impl_types.get(self._name(owner_id), ())]
        if lang not in _SPLIT_TYPE_LANGS:
            return [owner_id]
        unit = self._unit(owner_id, lang)
        return [owner_id] + [
            t
            for t in self._types(lang, self._name(owner_id))
            if t != owner_id and self._unit(t, lang) == unit
        ]

    def _std_given(self, trait, declaration, name, owner_id):
        """Look the member ``name`` up in ``trait``, a trait of the standard
        library's data that ``declaration`` implements or derives for the
        project type ``owner_id``: ``(True, the value it returns)`` where
        the trait has it, also as an attribute, which Python's data names
        behind a dot."""
        library = self.standard[self.lang(declaration)]
        tables = ("methods", name), ("statics", name), ("methods", f".{name}")
        for table, member in tables:
            receiver = ("self", owner_id)
            given = library.member(("outside", trait, ()), member, receiver, table)
            if given[0]:
                return given
        return False, None

    def _std_inherited(self, owner_id, name):
        """Value the member ``name`` returns that the project type
        ``owner_id`` has from a standard trait, where the data says."""
        for declaration in self._declarations(owner_id):
            bases = self.bases.get(declaration, ())
            named = (
                self._standard_type(_split_callee(base)[1], declaration)
                for base in bases
            )
            derived = self.builder.derived.get(declaration, ())
            for trait in [*filter(None, named), *derived]:
                known, value = self._std_given(trait, declaration, name, owner_id)
                if known:
                    return value
        return None

    def _lineage(self, class_id, seen):
        """Yield the Python class ``class_id`` and the project classes it
        inherits from, each before its own bases."""
        if class_id in seen:
            return
        seen.add(class_id)
        yield class_id
        for base in self.bases.get(class_id, ()):
            named = self._type_named(base, class_id)
            if named not in (None, _EXTERNAL):
                yield from self._lineage(named, seen)

    def _unit(self, node_id, lang):
        """What bounds one split type: the scope that declares it in
        TypeScript, the package directory in Go, the crate in Rust, the
        enclosing namespace in Ruby, in C/C++ the file that defines the
        type and its qualified name."""
        path = node_id.split("::")[0][5:]
        if lang == "ruby":
            # A scope without a name of its own is one file's.
            scope = node_id.rpartition("::")[0]
            return scope if "::<" in node_id else scope.partition("::")[2]
        if lang == "typescript":
            return node_id.rpartition("::")[0]
        if lang == "dotnet":
            # A partial type is one type in every file of its namespace,
            # by its name and the number of its type parameters.
            metadata = self.nodes[node_id]["metadata"]
            if metadata.get("partial"):
                return self._dotnet_path(node_id), metadata["arity"]
            return node_id
        if lang == "clang":
            # The node that holds a type's out-of-class members stands for
            # the one definition of its qualified name that its file is
            # compiled with; one of another namespace is another type.
            qualified = node_id.partition("::")[2]
            defined = [
                t
                for t in self._types(lang, self._name(node_id))
                if "end_lineno" in self.nodes[t]["metadata"]
                and t.partition("::")[2] == qualified
                and self._in_reach(t, node_id)
            ]
            own = "end_lineno" in self.nodes[node_id]["metadata"]
            if not own and len(defined) == 1:
                path = defined[0].split("::")[0][5:]
            return path, qualified
        if lang == "go":
            return path.rpartition("/")[0]
        if lang == "rust":
            folders = [f for f in self.builder.rust_crates.values() if f]
            inside = [f for f in folders if path.startswith(f"{f}/")]
            crate = max(inside, key=len, default="")
            # Cargo builds what is under these as crates of their own.
            target = path[len(crate) :].lstrip("/").partition("/")[0]
            if target in ("examples", "tests", "benches"):
                return f"{crate}/{target}".lstrip("/")
            return crate
        return ""

    def _ancestors(self, owner_id, seen=None):
        """Names of every type ``owner_id`` inherits from."""
        seen = set() if seen is None else seen
        lang = self.lang(owner_id)
        for base in self.bases.get(owner_id, ()):
            base = _split_callee(base)[1]
            if base not in seen:
                seen.add(base)
                for base_type in self._types(lang, base):
                    self._ancestors(base_type, seen)
        return seen

    def _decide(self, candidates, source_id):
        """The one candidate scope and build evidence leave of several
        that fit — None where nothing in the language decides between
        them. The survivors are remembered as the call's candidates; where
        a pick by file path once stood in for evidence, none is made now."""
        fitting = self._fitting(candidates, source_id)
        if len(fitting) == 1 and not self._candidate_only(fitting[0], source_id):
            return fitting[0]
        # Those the rules did not rule out may be meant; where they ruled
        # all out, all stay possible but none is pinned.
        self._candidates_now = sorted(fitting or candidates)
        return None

    def _fitting(self, candidates, source_id):
        """The candidates the call's own file can mean: a definition it
        names per the language's scoping — its own declarations, its
        imports, the package or namespace it shares them with, what its
        `use`/`require`/`#include` reach — and, of those, the ones whose
        build can run with its (Go build constraints, C/C++ `#if` guards)."""
        lang = self.lang(source_id)
        # The file's own declaration of the name is what the name binds
        # to in it — only where the file declares none do its imports,
        # package, crate or includes decide.
        own = [c for c in candidates if c.split("::")[0] == source_id.split("::")[0]]
        scoped = own or [c for c in candidates if self._in_scope(c, source_id)]
        if scoped:
            candidates = scoped
        if lang == "go":
            return self._go_fitting(candidates, source_id)
        if lang in ("clang", "dotnet"):
            guards = self.call_guards or self.nodes[source_id]["metadata"].get(
                "guards", ()
            )
            if guards:
                return [
                    c
                    for c in candidates
                    if not self._guard_conflict(
                        guards, self._effective_guards(c, source_id)
                    )
                ]
            # A definition an `#if` arm alone builds is a candidate, never
            # what an unguarded call is pinned to.
            plain = [c for c in candidates if not self._effective_guards(c, source_id)]
            return plain or list(candidates)
        return list(candidates)

    def _candidate_only(self, candidate, source_id):
        """Whether the sole fitting definition still may not be pinned to
        the call: it is built only under an `#if` arm the call site does
        not select, and the standard library has the name for the builds
        that leave the arm out."""
        lang = self.lang(source_id)
        if lang not in ("clang", "dotnet"):
            return False
        caller = self.call_guards or self.nodes[source_id]["metadata"].get("guards", ())
        if caller or not self._effective_guards(candidate, source_id):
            return False
        if lang not in self._standard_names:
            data = self.standard[lang].data
            self._standard_names[lang] = (
                {
                    re.split(r"::|\.", key.partition("`")[0])[-1]
                    for key in (*data["types"], *data["functions"])
                },
                {
                    name
                    for held in data["types"].values()
                    for name in (*held["methods"], *held["statics"])
                },
            )
        names, members = self._standard_names[lang]
        name = self.nodes[candidate]["metadata"]["name"]
        owner = candidate.rpartition("::")[0]
        if self._is_class(owner):
            # A member: of a type the library names too, by a name its
            # types have.
            return self.nodes[owner]["metadata"]["name"] in names and name in members
        return name in names

    def _effective_guards(self, candidate, source_id):
        """The guards a definition is built under: its own where it has
        them — or, for a C/C++ function, the arm a prototype in a header
        the caller includes declares it in."""
        guards = self.nodes[candidate]["metadata"].get("guards", ())
        if guards or self.lang(candidate) != "clang":
            return guards
        name = candidate.rpartition("::")[2]
        declared = [
            arms
            for file in self._include_reach(source_id.split("::")[0])
            for name_guard, arms in (self.builder.declares.get(file) or {}).items()
            if name_guard == name
        ]
        if not declared or any(not arms for arms in declared):
            return ()
        return frozenset().union(*declared)

    def _go_fitting(self, candidates, source_id):
        """The candidates in files a build with the caller's file holds:
        a `_test.go` definition is visible to the package's test files
        only, and platform constraints that contradict the caller's rule
        a definition out."""
        caller = source_id.split("::")[0]
        req, excl, caller_test = self._go_build(caller)
        kept = []
        for candidate in candidates:
            cfile = candidate.split("::")[0]
            c_req, c_excl, c_test = self._go_build(cfile)
            if c_test and not (
                caller_test and self._unit(cfile, "go") == self._unit(caller, "go")
            ):
                continue
            if not _go_builds_meet((req, excl), (c_req, c_excl)):
                continue
            kept.append(candidate)
        return kept

    def _go_build(self, file_id):
        """``(requires, excludes, is_test)`` of the Go file ``file_id``."""
        build = self.nodes.get(file_id, {}).get("metadata", {}).get("build", ())
        req = frozenset(build[0]) if build else frozenset()
        excl = frozenset(build[1]) if build else frozenset()
        test = bool(build and build[2])
        return req, excl, test

    @staticmethod
    def _guard_conflict(caller, candidate):
        """A signed `#if` atom the caller holds that the candidate's guards
        negate, or the other way around."""
        for sign in caller:
            if _flip_sign(sign) in candidate:
                return True
        for sign in candidate:
            if _flip_sign(sign) in caller:
                return True
        return False

    def _dotnet_extension(self, value, name):
        """The project extension methods a call under ``name`` on the
        outside-typed ``value`` may mean — written for the type or one it
        is of, per the library's data."""
        types = self.standard["dotnet"].data["types"]
        scope = self.standard["dotnet"].data.get("scope", {})
        # The type and, through its bases, every type it is of.
        given, todo = set(), [value[1].split("<")[0]]
        while todo:
            seen = todo.pop()
            if seen in given:
                continue
            given.add(seen)
            todo += types.get(seen, {}).get("bases", ())
        hits = []
        for ext in self.extensions.get(name, ()):
            written = self.nodes[ext]["metadata"]["extends"]
            named, _, _ = _dotnet_type_parts(written)
            key = (
                scope.get(written)
                or scope.get(named)
                or self._dotnet_std_key(named, ext.split("::")[0])
                or next(
                    (
                        t
                        for t in given
                        if t.rpartition(".")[2].rpartition("`")[0]
                        == named.rpartition("`")[0]
                    ),
                    named,
                )
            )
            if key in given or (
                key == "System.Object" and value[1].split("<")[0] in types
            ):
                hits.append(ext)
        return hits

    def _include_reach(self, file_id):
        """The files ``file_id``'s `#include`s reach, transitively."""
        if file_id not in self.reaches:
            found, todo = {file_id}, [file_id]
            while todo:
                for nxt in self.includes.get(todo.pop(), ()):
                    if nxt not in found:
                        found.add(nxt)
                        todo.append(nxt)
            self.reaches[file_id] = found
        return self.reaches[file_id]

    def _in_scope(self, candidate, source_id):
        """``candidate`` is a definition the file of ``source_id`` can mean
        for a name it writes: declared in that file, imported into it, or
        in the package, crate, module or namespace the language groups it
        with. What the file cannot name is not the call's."""
        c_file = candidate.split("::")[0]
        s_file = source_id.split("::")[0]
        if c_file == s_file:
            return True
        lang = self.lang(source_id)
        if lang != self.lang(candidate):
            return True
        owner = candidate.rpartition("::")[0]
        owner_name = owner.rpartition("::")[2] if owner != c_file else ""
        imports = self.builder.imports.get(s_file, {})
        # The name an import binds is the type's for a member, the
        # definition's own for a type or a top-level one.
        named = owner_name or candidate.rpartition("::")[2]
        if named and named in imports:
            return True
        if lang == "go":
            if self._unit(candidate, "go") == self._unit(source_id, "go"):
                return True
            for target in imports.values():
                for folder, module in self.go_modules:
                    if target == module or target.startswith(f"{module}/"):
                        path = "/".join(
                            p for p in (folder, target[len(module) + 1 :]) if p
                        )
                        if c_file in self.go_packages.get(path, ()):
                            return True
            return False
        if lang == "rust":
            # Within a crate a name needs a `use` or a module that shares
            # it; another crate's names need one too.
            if owner_name and owner_name in imports:
                return True
            return self._unit(candidate, "rust") == self._unit(
                source_id, "rust"
            ) and posixpath.dirname(c_file) == posixpath.dirname(s_file)
        if lang == "java":
            own = self.nodes[c_file]["metadata"].get("java_package")
            here = self.nodes[s_file]["metadata"].get("java_package")
            if own and own == here:
                return True
            if owner_name and owner_name in imports:
                return True
            # `import pkg.*` makes every type of the package nameable.
            return any(
                key.endswith("*") and isinstance(target, str) and own == target
                for key, target in imports.items()
            )
        if lang == "dotnet":
            own = self.nodes[owner if owner.startswith("FILE:") else c_file][
                "metadata"
            ].get("namespace")
            here = self.nodes[s_file]["metadata"].get("namespace")
            if own and own == here:
                return True
            usings = self._dotnet_imports(s_file)
            return own in usings.values() or (owner_name and owner_name in usings)
        if lang == "php":
            own = self.nodes[c_file]["metadata"].get("namespace")
            here = self.nodes[s_file]["metadata"].get("namespace")
            return (own and own == here) or bool(owner_name and owner_name in imports)
        if lang == "python":
            if owner_name and owner_name in imports:
                return True
            return any(
                c_file in self.py_modules.get(target, {}).values()
                for target in imports.values()
                if isinstance(target, str)
            )
        if lang == "typescript":
            if owner_name and owner_name in imports:
                return True
            required = self.builder.required_names.get(s_file, ())
            return any(
                c_file in (self._js_module(target, s_file, name in required) or ())
                for name, target in imports.items()
                if isinstance(target, str)
            )
        if lang == "dart":
            if self.dart_parts.get(c_file, c_file) == self.dart_parts.get(
                s_file, s_file
            ):
                return True
            return owner_name and owner_name in imports
        if lang == "ruby":
            return c_file in self._loaded(s_file)
        if lang == "clang":
            if c_file in self._include_reach(s_file):
                return True
            # A prototype in an included header promises a definition
            # somewhere in the build: every file that defines the name
            # the caller can name.
            name = candidate.rpartition("::")[2]
            return any(
                name in self.builder.declares.get(file, ())
                for file in self._include_reach(s_file)
            )
        return True


class _OverTime(BaseException):
    """The block `_time_limit` runs has used up its time. No `Exception`, so
    that no handler inside the block takes it for a failure of its own."""


@contextlib.contextmanager
def _time_limit(seconds):
    """Raise `_OverTime` in the block once it has run for ``seconds``. There
    is no limit for 0, off the main thread, or on a platform without
    interval timers."""
    main = threading.current_thread() is threading.main_thread()
    if not (seconds and main and hasattr(signal, "setitimer")):
        yield
        return

    def over(_signum, _frame):
        raise _OverTime

    previous = signal.signal(signal.SIGALRM, over)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class CodeGraphBuilder:
    def __init__(
        self,
        root_dir,
        excluded_dirs=None,
        rel_base=None,
        max_file_bytes=0,
        max_parse_seconds=0,
        respect_gitignore=False,
    ):
        self.root_dir = Path(root_dir).resolve()
        # Larger source files are skipped; 0 walks files of any size.
        self.max_file_bytes = max_file_bytes
        # A file is parsed for no longer than this; 0 sets no limit.
        self.max_parse_seconds = max_parse_seconds
        # Whether the files git ignores are left out.
        self.respect_gitignore = respect_gitignore
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
        # JS/TS file or object ID -> [(name exported or held, "*" for each
        # name of what it names or "=" for the file's whole, module specifier
        # or "" for a name written in the file, the name there)]
        self.reexports = {}
        # JS/TS file ID -> {local name: the name its module exports it as},
        # for an import that renames what it binds, ``default`` for one of
        # the default and ``=`` for one of the module whole
        self.import_names = {}
        self.required_names = {}  # JS/TS file ID -> the names a `require` binds
        # (defining file or class ID, fixture name) -> (function ID, origin
        # path of the value the fixture returns or yields, if it has one,
        # whether it yields it)
        self.fixtures = {}
        # Names declared as type aliases, which are not graph nodes, and for
        # TypeScript ``(file ID, name) -> (type parameters, text)`` of each
        # written on one line
        self.type_aliases = set()
        self.ts_aliases = {}
        # C/C++ file ID -> the function names a prototype in it declares:
        # the definition may sit in any file linked in.
        self.declares = {}
        # Type or variant ID -> {field name or position: its declared type};
        # Python module or class ID -> {variable: origin paths of its values},
        # function ID -> {`()`: origin paths of what it returns}
        self.fields = {}
        # Type ID -> the standard traits it derives, as its language's
        # standard-library data names them
        self.derived = {}
        self.rust_aliases = {}  # Rust file ID -> names of the aliases it declares
        self.dart_packages = {}  # Dart package name -> its directory
        # Dart file ID -> the (`import`, `export` or `part`, URIs, prefix,
        # names shown, names hidden) of each of its directives
        self.dart_directives = {}
        # C/C++ alias name -> [(ID of the scope that declares it, its type)]
        self.c_aliases = {}
        self.js_bound = {}  # JS/TS file ID -> the names its declarations bind
        # Function ID -> its signatures as written, one for each overload:
        # the bound of each type parameter, the parameters, the return type
        # and, by argument position, the declared types of the parameters
        # of the function it is passed there
        self.signatures = {}
        # `[N]`, which an origin path names an array literal by -> the origin
        # path of each of its elements, for an object literal those of its
        # properties by name; None where they are not known
        self.literals = {}
        # The literal a function returns -> the function, in whose scope the
        # names in it are read
        self.literal_scopes = {}
        # Directory -> the `global using` directives of its C# files, which
        # hold for every file below it, as ``imports`` has a file's own
        self.dotnet_global_usings = {}

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
        return path.relative_to(self.rel_base).as_posix()

    def _git_ignored(self) -> set[str]:
        """Paths under the root that git ignores, relative to it: files, and
        directories ignored whole. Empty outside a git repository, without
        git, and where ignored files are to be indexed."""
        if not self.respect_gitignore:
            return set()
        from descry.git_history import GitError, GitHistoryAnalyzer

        listing = "ls-files -z --others --ignored --exclude-standard --directory"
        try:
            listed = GitHistoryAnalyzer(self.root_dir)._run_git(listing.split())
        except (GitError, OSError):
            return set()
        return {path.rstrip("/") for path in listed.split("\0") if path}

    def process_directory(self):
        # Resolve root once; reject entering any child directory whose resolved
        # path escapes the project root (defends against malicious symlinks).
        root_real = Path(self.root_dir).resolve()
        ignored = self._git_ignored()

        def kept(root, names):
            """``names`` in directory ``root`` without those git ignores."""
            under = Path(root).relative_to(self.root_dir)
            return [name for name in names if (under / name).as_posix() not in ignored]

        for root, dirs, files in os.walk(self.root_dir, followlinks=False):
            dirs.sort()
            files.sort()
            # Filter by name first (excluded dirs, dotfiles), and by what git
            # ignores.
            dirs[:] = [
                d
                for d in kept(root, dirs)
                if not d.startswith(".") and d not in self.excluded_dirs
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
            for file in kept(root, files):
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
                # Each file is read whole, so one over the limit is skipped
                # before it is opened, as is anything but a regular file: a
                # FIFO would hold the read open and a device never end it.
                try:
                    st = file_path.stat()
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                size = st.st_size
                if self.max_file_bytes and size > self.max_file_bytes:
                    logger.warning(
                        "Skipping %s — %.1f MiB exceeds [code_files] max_file_mb",
                        rel_path,
                        size / (1024 * 1024),
                    )
                    continue
                # O_NOFOLLOW on the final open so a post-check symlink swap
                # can't redirect us outside the project root, and O_NONBLOCK
                # so one to a FIFO doesn't wait. Mirrors the api_source
                # hardening in web/server.py.
                try:
                    flags = (
                        os.O_RDONLY
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_NONBLOCK", 0)
                    )
                    fd = os.open(str(file_path), flags)
                    with os.fdopen(fd, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except OSError as e:
                    logger.warning(f"Cannot read {rel_path}: {e.strerror}")
                    continue
                except ValueError as e:
                    logger.warning(f"Error reading {rel_path}: {e}")
                    continue
                # A file one parser chokes on, or does not finish in time,
                # must not abort the whole index.
                try:
                    with _time_limit(self.max_parse_seconds):
                        self._parse(file, file_path, rel_path, content)
                except _OverTime:
                    logger.warning(
                        "Stopped parsing %s after %g s ([code_files] "
                        "max_parse_seconds); it is indexed as far as it got",
                        rel_path,
                        self.max_parse_seconds,
                    )
                except Exception as e:  # noqa: BLE001 — boundary over every parser
                    logger.warning(f"Failed to parse {rel_path}: {e!r}")

    def _parse(self, file, file_path, rel_path, content):
        """Parse ``content``, the text of ``file``, with its language's parser."""
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
        elif file == "pubspec.yaml":
            if match := _PUBSPEC_NAME_RE.search(content):
                folder = rel_path.rpartition("/")[0]
                self.dart_packages[match.group(1)] = folder
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
            # Iterate all <script> blocks and shift each block's
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
                TSParser(self).parse(file_path, rel_path, body, line_offset=line_offset)
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
        or drops it as external. A base type is the one CallResolver finds
        its name to stand for where the subtype is declared. Other
        references are matched by name within the source's language:
        qualified name (Type::method), then same file, then any match.

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
                arguments = edge.get("metadata", {}).pop("args", None)
                call_resolver.arguments = (
                    arguments if isinstance(arguments, list) else None
                )
                call_resolver.call_guards = edge.get("metadata", {}).get("guards")

                # A Rust trait written with a path is the type the path
                # leads to; one the path does not place is read by its name.
                placed = None
                if not is_call and source_lang == "rust" and origin:
                    placed, origin = call_resolver.type_at(origin, source_id), None
                if placed == _EXTERNAL:
                    edges_to_remove.add(idx)
                    continue
                if placed:
                    resolved, resolution_source = placed, "regex"
                # The parser writes a PHP type by its full name, which no
                # other type of that name answers to.
                elif not is_call and source_lang == "php":
                    resolved = call_resolver.type_in_scope(
                        ref_name, source_id, ref_name
                    )
                    if not resolved:
                        continue
                    resolution_source = "regex"
                # A type reference to a stdlib/library name is one to a
                # project type only where its scope shows that.
                elif not is_call and is_non_project_call(ref_name):
                    resolved = call_resolver.type_in_scope(ref_name, source_id, origin)
                    if not resolved:
                        edges_to_remove.add(idx)
                        continue
                    resolution_source = "regex"

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
                        # What SCIP saw go to code outside the project is
                        # no call of a project definition of that name.
                        if not found and scip_index.outside_at(
                            ref_name, source_file, lineno
                        ):
                            resolution_source = _EXTERNAL
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
                        # This prevents spurious matches like EnvFilter -> RequestFilter
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
                    if not resolved and resolution_source != _EXTERNAL:
                        resolved, resolution_source = call_resolver.resolve(
                            ref_name, source_id, origin, arguments
                        )
                    call_resolver.keep(idx, resolved)
                    if resolution_source == _EXTERNAL:
                        edges_to_remove.add(idx)
                        stats["external"] += 1
                        continue
                elif not resolved and edge["relation"] == "INHERITS":
                    resolved = call_resolver.base_type(idx)
                    if not resolved:
                        continue
                    resolution_source = "regex"

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
        fd, tmp_path = create_sibling(output_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
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
        "path",
        nargs="?",
        help="Directory to index (default: the project root, else .)",
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

    # Progress is logged at INFO unless the environment or -v sets a level.
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ["DESCRY_LOG_LEVEL"] = "DEBUG"
    else:
        os.environ.setdefault("DESCRY_LOG_LEVEL", "INFO")

    target = args.path or args.project_root or "."

    from descry.handlers import DescryConfig

    if args.no_scip:
        os.environ["DESCRY_NO_SCIP"] = "1"

    project_root = Path(args.project_root or target).resolve()
    if not Path(target).resolve().is_relative_to(project_root):
        parser.error(f"{target!r} is outside the project root {project_root}")
    # The root comes from the arguments, not from the working directory.
    config = DescryConfig.from_env(project_root)
    if error := config.cache_dir_error():
        parser.error(error)

    # Passed always, so non-TOML projects still get the full default
    # exclusion set.
    config_excluded_dirs = config.excluded_dirs

    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build the graph
    builder = CodeGraphBuilder(
        target,
        excluded_dirs=config_excluded_dirs,
        rel_base=project_root,
        max_file_bytes=config.max_file_bytes,
        max_parse_seconds=config.max_parse_seconds,
        respect_gitignore=config.respect_gitignore,
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
