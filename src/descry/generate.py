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

# Refuse to index individual source files above this size. Matches the
# /api/source cap in web/server.py so the indexer and viewer share a budget.
_MAX_SOURCE_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB

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
        "last",
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
        "String",
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
        "contains",
        "matches",
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
        "from",
        "of",
        "entries",
        "keys",
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
        "createElement",
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
        "Map",
        "Set",
        "HashMap",
        "HashSet",
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
        "File",
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
        "FileReader",
        "FileWriter",
        "InputStreamReader",
        "OutputStreamWriter",
        "ByteBuffer",
        "CharBuffer",
        "Charset",
        "StandardCharsets",
        "String",
        "StringBuilder",
        "StringBuffer",
        "StringJoiner",
        "Integer",
        "Long",
        "Double",
        "Float",
        "Boolean",
        "Byte",
        "Short",
        "Character",
        "Number",
        "BigInteger",
        "BigDecimal",
        "Math",
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
        "Error",
        "Instant",
        "LocalDate",
        "LocalDateTime",
        "LocalTime",
        "ZonedDateTime",
        "OffsetDateTime",
        "Duration",
        "Period",
        "ChronoUnit",
        "TimeUnit",
        "Locale",
        "UUID",
        "Random",
        "SecureRandom",
        "URI",
        "URL",
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
        "toString",
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
        "remove",
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
        "abort",
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
        "assert",
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
        "abs",
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
        "sleep",
        "usleep",
        "nanosleep",
        "open",
        "close",
        "read",
        "write",
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
        "bind",
        "listen",
        "accept",
        "connect",
        "send",
        "recv",
        "sendto",
        "recvfrom",
        "setsockopt",
        "getsockopt",
        "gethostbyname",
        "select",
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
        "map",
        "unordered_map",
        "set",
        "unordered_set",
        "list",
        "deque",
        "stack",
        "queue",
        "priority_queue",
        "array",
        "pair",
        "tuple",
        "optional",
        "variant",
        "any",
        "unique_ptr",
        "shared_ptr",
        "weak_ptr",
        "make_unique",
        "make_shared",
        "make_pair",
        "make_tuple",
        "move",
        "forward",
        "swap",
        "copy",
        "copy_n",
        "copy_if",
        "fill",
        "fill_n",
        "transform",
        "find",
        "find_if",
        "find_if_not",
        "all_of",
        "any_of",
        "none_of",
        "count",
        "count_if",
        "equal",
        "mismatch",
        "search",
        "sort",
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
        "reduce",
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
        "flush",
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
        "async",
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
        "Debug",
        "Trace",
        "Task",
        "ValueTask",
        "Thread",
        "Timer",
        "CancellationToken",
        "CancellationTokenSource",
        "Environment",
        "List",
        "Dictionary",
        "HashSet",
        "SortedDictionary",
        "SortedSet",
        "Queue",
        "Stack",
        "LinkedList",
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
        "StringBuilder",
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
        "File",
        "Directory",
        "FileInfo",
        "DirectoryInfo",
        "Path",
        "Stream",
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
        "JsonNode",
        "Enumerable",
        "Queryable",
        "IEnumerable",
        "Math",
        "MathF",
        "Random",
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
        "Exception",
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
        "IOException",
        "FileNotFoundException",
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
        "StringComparer",
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
        "JsonIgnore",
        "JsonProperty",
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
        "TearDown",
        "OneTimeSetUp",
        "OneTimeTearDown",
        # ── Moq / NSubstitute ──
        "Mock",
        "It",
        "Setup",
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
        "count",
        "sizeof",
        "is_array",
        "is_string",
        "is_int",
        "is_integer",
        "is_numeric",
        "is_float",
        "is_double",
        "is_bool",
        "is_null",
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
        "fopen",
        "fclose",
        "fread",
        "fwrite",
        "fgets",
        "fputs",
        "feof",
        "fseek",
        "ftell",
        "flock",
        "rewind",
        "file_exists",
        "file_get_contents",
        "file_put_contents",
        "is_file",
        "is_dir",
        "is_link",
        "is_readable",
        "is_writable",
        "is_executable",
        "realpath",
        "dirname",
        "basename",
        "pathinfo",
        "tempnam",
        "tmpfile",
        "chmod",
        "chown",
        "chgrp",
        "mkdir",
        "rmdir",
        "rename",
        "unlink",
        "copy",
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
        "time",
        "mktime",
        "gmmktime",
        "date",
        "gmdate",
        "strtotime",
        "getenv",
        "putenv",
        "setlocale",
        "strlen",
        "mb_strlen",
        "strpos",
        "mb_strpos",
        "strrpos",
        "mb_strrpos",
        "strstr",
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
        "join",
        "trim",
        "ltrim",
        "rtrim",
        "strtolower",
        "strtoupper",
        "ucfirst",
        "ucwords",
        "lcfirst",
        "mb_strtolower",
        "mb_strtoupper",
        "substr",
        "mb_substr",
        "substr_count",
        "substr_replace",
        "sprintf",
        "printf",
        "vsprintf",
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
        "hash",
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
        "get_class",
        "get_parent_class",
        "get_class_methods",
        "get_class_vars",
        "get_object_vars",
        "class_exists",
        "interface_exists",
        "trait_exists",
        "enum_exists",
        "method_exists",
        "property_exists",
        "function_exists",
        "defined",
        "define",
        "constant",
        "call_user_func",
        "call_user_func_array",
        "func_get_args",
        "func_num_args",
        "func_get_arg",
        "min",
        "max",
        "abs",
        "round",
        "ceil",
        "floor",
        "pow",
        "sqrt",
        "sort",
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
        "range",
        "array_push",
        "array_pop",
        "array_shift",
        "array_unshift",
        "current",
        "next",
        "prev",
        "reset",
        "end",
        "key",
        "each",
        "fopen",
        "fclose",
        "fread",
        "fwrite",
        "fgets",
        "fgetcsv",
        "fputcsv",
        "feof",
        "file_get_contents",
        "file_put_contents",
        "file_exists",
        "is_file",
        "is_dir",
        "is_readable",
        "is_writable",
        "is_writeable",
        "mkdir",
        "rmdir",
        "unlink",
        "rename",
        "copy",
        "realpath",
        "dirname",
        "basename",
        "pathinfo",
        "glob",
        "scandir",
        "date",
        "time",
        "mktime",
        "strtotime",
        "microtime",
        "date_create",
        "date_format",
        "strftime",
        "gmdate",
        "checkdate",
        "usleep",
        "sleep",
        "trigger_error",
        "error_reporting",
        "ini_get",
        "ini_set",
        "getenv",
        "putenv",
        "env",
        "getcwd",
        "chdir",
        "die",
        "exit",
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
        "WeakMap",
        "WeakReference",
        "ArrayObject",
        "ArrayIterator",
        "Generator",
        "Iterator",
        "IteratorAggregate",
        "Traversable",
        "Countable",
        "ArrayAccess",
        "Stringable",
        "Throwable",
        "Exception",
        "Error",
        "TypeError",
        "ValueError",
        "RuntimeException",
        "LogicException",
        "InvalidArgumentException",
        "OutOfBoundsException",
        "OutOfRangeException",
        "UnexpectedValueException",
        "DomainException",
        "BadFunctionCallException",
        "BadMethodCallException",
        "DateTime",
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
        "Stringable",
        # ── PHP binary / encoding / output stdlib ──
        "pack",
        "unpack",
        "bin2hex",
        "hex2bin",
        "ord",
        "chr",
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
        "flush",
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
        "OutOfRangeException",
        "UnderflowException",
        "OverflowException",
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
        "id",
        "shortDescription",
        "setUpClass",
        "tearDownClass",
        "assertMultiLineEqual",
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
        "string",
        "bool",
        "any",
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
        "Condition",
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
        "get_flag",
        "get_one",
        "get_many",
        "get_count",
        "get_occurrences",
        "try_get_one",
        "try_get_many",
        "contains_id",
        "remove_one",
        "remove_many",
        "ids",
        "index_of",
        "value_source",
        # Rust rand crate (universal random number generator).
        "random_range",
        "gen_range",
        "gen_bool",
        "gen_ratio",
        "sample",
        "sample_iter",
        "shuffle",
        "partial_shuffle",
        "choose",
        "choose_multiple",
        "choose_weighted",
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
        "write_fmt",
        "pad",
        "finish",
        "field",
        "entry",
        "key",
        "value",
        # ── Rust future/async surface ──
        "poll_fn",
        "from_waker",
        "from_raw",
        "noop_waker",
        # ── mockall ──
        "in_sequence",
        "expect_call",
        "returning",
        "times",
        # ── Criterion benchmark framework surface ──
        "bench_function",
        "bench_with_input",
        "benchmark_group",
        "measurement_time",
        "measure",
        "warm_up_time",
        "sample_size",
        "configure_from_args",
        "throughput",
        "iter_batched",
        "iter_custom",
        "iter_with_setup",
        "iter_with_large_drop",
        "black_box",
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
        "times",
        "makePartial",
        "withArgs",
        "with",
        "mock",
        "spy",
        "instanceMock",
        "namedMock",
        # ── Laravel framework helpers ──
        "collect",
        "tap",
        "rescue",
        "retry",
        "value",
        "data_get",
        "data_set",
        "head",
        "last",
        "with",
        "tap",
        "dd",
        "dump",
        "logger",
        "report",
        "now",
        "today",
        "faker",
        "blank",
        "filled",
        "str",
        "Str",
        "Arr",
        "Collection",
        "Carbon",
        "URL",
        "Route",
        "DB",
        "Schema",
        "Cache",
        "Config",
        "Log",
        "Http",
        "Storage",
        "Queue",
        "Event",
        "Mail",
        "Notification",
        "Auth",
        "Request",
        "Response",
        "View",
        "Session",
        "Redirect",
        "Validator",
        "Hash",
        "Crypt",
        "Cookie",
        "Bus",
        "App",
        "Artisan",
        # ── Ruby stdlib / Kernel / Object reflection ──
        # These are never defined in user code so they must be filtered
        # even when called bare (`send(...)`, `sprintf(...)`).
        "send",
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
        "hash",
        "eql?",
        "equal?",
        "freeze",
        "frozen?",
        "tap",
        "then",
        "yield_self",
        "dup",
        "clone",
        "inspect",
        "to_proc",
        "proc",
        "lambda",
        "block_given?",
        "binding",
        "eval",
        "sprintf",
        "printf",
        "format",
        "gets",
        "getc",
        "readline",
        "readlines",
        "warn",
        "abort",
        "exit",
        "exit!",
        "at_exit",
        "caller",
        "caller_locations",
        "__method__",
        "__callee__",
        "__dir__",
        "object_id",
        "gem",
        "rand",
        "srand",
        "sleep",
        "system",
        "spawn",
        "fork",
        "exec",
        "trap",
        "Array",
        "Hash",
        "Integer",
        "Float",
        "String",
        "Rational",
        "Complex",
        # ── Ruby String / Enumerable / IO core methods (called bare on
        #    `self` or on any value — never project symbols). ──
        "gsub",
        "gsub!",
        "sub",
        "sub!",
        "scan",
        "match",
        "match?",
        "index",
        "rindex",
        "split",
        "partition",
        "rpartition",
        "chomp",
        "chomp!",
        "chop",
        "chop!",
        "strip",
        "strip!",
        "lstrip",
        "lstrip!",
        "rstrip",
        "rstrip!",
        "squeeze",
        "squeeze!",
        "tr",
        "tr!",
        "tr_s",
        "tr_s!",
        "delete",
        "delete!",
        "count",
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
        "reverse",
        "reverse!",
        "concat",
        "prepend",
        "replace",
        "hex",
        "oct",
        "to_sym",
        "to_str",
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
        "bytes",
        "chars",
        "codepoints",
        "lines",
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
        "each",
        "each_with_index",
        "each_with_object",
        "each_pair",
        "each_entry",
        "each_slice",
        "each_cons",
        "map",
        "map!",
        "collect",
        "collect!",
        "flat_map",
        "collect_concat",
        "select",
        "select!",
        "filter",
        "filter!",
        "filter_map",
        "reject",
        "reject!",
        "reduce",
        "inject",
        "find",
        "detect",
        "find_all",
        "find_index",
        "group_by",
        "chunk",
        "chunk_while",
        "slice_when",
        "slice_before",
        "slice_after",
        "partition",
        "zip",
        "take",
        "take_while",
        "drop",
        "drop_while",
        "first",
        "last",
        "min",
        "max",
        "min_by",
        "max_by",
        "minmax",
        "minmax_by",
        "sort",
        "sort!",
        "sort_by",
        "sort_by!",
        "tally",
        "uniq",
        "uniq!",
        "compact",
        "compact!",
        "flatten",
        "flatten!",
        "rotate",
        "rotate!",
        "sample",
        "shuffle",
        "shuffle!",
        "cycle",
        "each_index",
        "unshift",
        "shift",
        "push",
        "pop",
        "append",
        "prepend",
        "assoc",
        "rassoc",
        "fetch",
        "store",
        "dig",
        "merge",
        "merge!",
        "update",
        "invert",
        "values_at",
        "keys",
        "values",
        "pairs",
        "transform_keys",
        "transform_keys!",
        "transform_values",
        "transform_values!",
        "slice",
        "except",
        "compact_by",
        "readpartial",
        "sysread",
        "sysseek",
        "syswrite",
        "read",
        "read_nonblock",
        "write",
        "write_nonblock",
        "puts",
        "print",
        "gets",
        "each_line",
        "flush",
        "close",
        "closed?",
        "eof?",
        "tell",
        "pos",
        "seek",
        "rewind",
        "stat",
        "fileno",
        "fcntl",
        "ioctl",
        "pipe",
        "dup",
        # ── Ruby stdlib classes used as constructors/methods ──
        "Addrinfo",
        "SecureRandom",
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
        "skip",
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
        "expect",
        "be",
        "be_truthy",
        "be_falsey",
        "be_nil",
        "eq",
        "eql",
        "equal",
        "match",
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
        "File",
        "Dir",
        "IO",
        "StringIO",
        "Tempfile",
        "Pathname",
        "URI",
        "DateTime",
        "Date",
        "Time",
        "Net",
        "JSON",
        "YAML",
        "Marshal",
        "Base64",
        "Digest",
        "Logger",
        "Mutex",
        "Thread",
        "Queue",
        "Fiber",
        "Enumerator",
        "Range",
        "Set",
        "Hash",
        "Array",
        "String",
        "Integer",
        "Float",
        "Rational",
        "Complex",
        "Regexp",
        "Proc",
        "Method",
        "Symbol",
        "Struct",
        "OpenStruct",
        "Exception",
        "StandardError",
        "RuntimeError",
        "ArgumentError",
        "TypeError",
        "NameError",
        "NoMethodError",
        "IOError",
        "Errno",
        "Kernel",
        "Object",
        "BasicObject",
        "Module",
        "Class",
        "Comparable",
        "Enumerable",
        # ── JUnit / testing ──
        "assertEquals",
        "assertNotEquals",
        "assertTrue",
        "assertFalse",
        "assertNull",
        "assertNotNull",
        "assertThrows",
        "assertDoesNotThrow",
        "assertSame",
        "assertNotSame",
        "assertAll",
        "assertArrayEquals",
        "assertIterableEquals",
        "assertLinesMatch",
        "assertInstanceOf",
        "fail",
        "assumeTrue",
        "assumeFalse",
        "assumingThat",
        "BeforeEach",
        "AfterEach",
        "BeforeAll",
        "AfterAll",
        "Test",
        "Disabled",
        "DisplayName",
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
        "startsWith",
        "endsWith",
        "hasSize",
        "hasItem",
        "hasItems",
        "hasEntry",
        "hasKey",
        "hasValue",
        "contains",
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
        "mock",
        "when",
        "verify",
        "times",
        "atLeast",
        "atMost",
        "atLeastOnce",
        "atMostOnce",
        "exactly",
        "never",
        "only",
        "any",
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
        "eq",
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
        "spy",
        "inOrder",
        "verifyNoInteractions",
        "verifyNoMoreInteractions",
        "reset",
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
        "willReturn",
        "willThrow",
        "willDoNothing",
        "willAnswer",
        "willCallRealMethod",
        "shouldHaveNoMoreInteractions",
        "shouldHaveNoInteractions",
        "then",
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
        "empty",
        "error",
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
        "Returns",
        "ReturnsLazily",
        "ReturnsNextFromSequence",
        "Throws",
        "ThrowsAsync",
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
        "info",
        "warn",
        "error",
        "debug",
        "trace",
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
        "toMatchInlineSnapshot",
        "toMatchSnapshot",
        "toMatchFileSnapshot",
        "toMatch",
        "toStrictEqual",
        "toContainEqual",
        "toInclude",
        "toHaveBeenCalledTimes",
        "toHaveProperty",
        "toHaveBeenNthCalledWith",
        "toHaveBeenLastCalledWith",
        "toHaveReturnedWith",
        "toHaveLastReturnedWith",
        "toHaveNthReturnedWith",
        "toHaveReturned",
        "toHaveReturnedTimes",
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
        "runIf",
        "skipIf",
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
        "fromEntries",
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
        # ── Python stdlib module names (bare) ──
        "json",
        "time",
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
        "inspect",
        "typing",
        "dataclasses",
        "enum",
        "abc",
        "contextlib",
        "tempfile",
        "shlex",
        "signal",
        "socket",
        "select",
        "urllib",
        "http",
        "base64",
        "secrets",
        "random",
        "math",
        "statistics",
        "struct",
        "copy",
        "stat",
        "platform",
        "errno",
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
        "Exception",
        "BaseException",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
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
        "IOError",
        "OverflowError",
        "ZeroDivisionError",
        "ArithmeticError",
        "AssertionError",
        "LookupError",
        "ImportError",
        "ModuleNotFoundError",
        "NameError",
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
        "dict",
        "list",
        "set",
        "frozenset",
        "tuple",
        "str",
        "bytes",
        "bytearray",
        "int",
        "float",
        "bool",
        "complex",
        "range",
        "slice",
        "enumerate",
        "zip",
        "map",
        "filter",
        "reversed",
        "sorted",
        "type",
        "object",
        "super",
        "property",
        "staticmethod",
        "classmethod",
        # ── Python builtin functions ──
        "print",
        "repr",
        "ascii",
        "bin",
        "hex",
        "oct",
        "chr",
        "ord",
        "len",
        "abs",
        "min",
        "max",
        "sum",
        "round",
        "pow",
        "divmod",
        "all",
        "any",
        "next",
        "iter",
        "hash",
        "id",
        "vars",
        "dir",
        "callable",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "isinstance",
        "issubclass",
        "open",
        "input",
        "format",
        "globals",
        "locals",
        "eval",
        "exec",
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
        "mkdir",
        "rmdir",
        "touch",
        "rename",
        "replace",
        "hexdigest",
        "digest",
        "update",
        "startswith",
        "endswith",
        "encode",
        "decode",
        "upper",
        "lower",
        "title",
        "capitalize",
        "casefold",
        "strip",
        "lstrip",
        "rstrip",
        "split",
        "rsplit",
        "splitlines",
        "join",
        "removeprefix",
        "removesuffix",
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "reverse",
        "index",
        "count",
        "add",
        "discard",
        "clear",
        "copy",
        "keys",
        "values",
        "items",
        "get",
        "setdefault",
        "popitem",
        "find",
        "rfind",
        "rindex",
        "zfill",
        "center",
        "ljust",
        "rjust",
        "expandtabs",
        "partition",
        "rpartition",
        "translate",
        "maketrans",
        "format_map",
        "is_dir",
        "is_file",
        "is_symlink",
        "is_absolute",
        "exists",
        "stat",
        "lstat",
        "chmod",
        "unlink",
        "glob",
        "rglob",
        "iterdir",
        "resolve",
        "relative_to",
        "with_suffix",
        "with_stem",
        "with_name",
        "joinpath",
        "samefile",
        "cwd",
        "home",
        "seek",
        "tell",
        "flush",
        "close",
        "readline",
        "readlines",
        "writelines",
        "truncate",
        # Common method-chain intermediates.
        "reshape",
        "astype",
        "cpu",
        "gpu",
        "detach",
        # More stdlib method names
        "isupper",
        "islower",
        "isdigit",
        "isnumeric",
        "isalpha",
        "isalnum",
        "isspace",
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
        "getLogger",
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
        "end",
        "span",
        "submit",
        "cancel",
        "done",
        "result",
        "as_completed",
        "gather",
        "wait",
        "wait_for",
        "create_task",
        "ensure_future",
        "run_until_complete",
        "close",
        "run",
        "new_event_loop",
        "asynccontextmanager",
        "contextmanager",
        "wraps",
        "partial",
        "reduce",
        "lru_cache",
        "cache",
        "total_ordering",
        "singledispatch",
        "TYPE_CHECKING",
        "cast",
        "Any",
        "Optional",
        "Union",
        "List",
        "Dict",
        "Set",
        "Tuple",
        "Callable",
        "Iterator",
        "Iterable",
        "AsyncIterator",
        "AsyncIterable",
        "Awaitable",
        "Coroutine",
        "Generator",
        "AsyncGenerator",
        # ── Dart core / Flutter / common packages ──
        "print",
        "debugPrint",
        "assert",
        "identical",
        "identityHashCode",
        "runtimeType",
        "hashCode",
        "toString",
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
        "Stack",
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
        "Future",
        "Stream",
        "Completer",
        "Timer",
        "Duration",
        "DateTime",
        "Uri",
        "RegExp",
        "Match",
        "StringBuffer",
        "Iterable",
        "Iterator",
        "List",
        "Set",
        "Map",
        "HashMap",
        "HashSet",
        "LinkedHashMap",
        "LinkedHashSet",
        "SplayTreeMap",
        "SplayTreeSet",
        "Queue",
        "ListQueue",
        "DoubleLinkedQueue",
        "jsonDecode",
        "jsonEncode",
        "utf8",
        "ascii",
        "latin1",
        "base64",
        "base64Encode",
        "base64Decode",
        "min",
        "max",
        "sqrt",
        "pow",
        "log",
        "exp",
        "sin",
        "cos",
        "tan",
        "Random",
        "Platform",
        "Process",
        "File",
        "Directory",
        "HttpClient",
        "HttpServer",
        "WebSocket",
        "Socket",
        "ServerSocket",
        "test",
        "group",
        "setUp",
        "tearDown",
        "setUpAll",
        "tearDownAll",
        "expect",
        "expectAsync",
        "expectLater",
        "testWidgets",
        "when",
        "verify",
        "verifyNever",
        "verifyInOrder",
        "reset",
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
        "isNaN",
        "isNotNaN",
        "isTrue",
        "isFalse",
        "isNull",
        "isNotNull",
        "isZero",
        "isNonZero",
        "isNegative",
        "isNonNegative",
        "isPositive",
        "isNonPositive",
        "equals",
        "equalsIgnoringCase",
        "equalsIgnoringWhitespace",
        "contains",
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
        "Timeout",
        "Skip",
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
        "parse",
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
        "FormatException",
        "RangeError",
        "ArgumentError",
        "AssertionError",
        "ConcurrentModificationError",
        "NoSuchMethodError",
        "TypeError",
        "OutOfMemoryError",
        "StackOverflowError",
        # ── package:pub_semver (Version) ──
        "Version",
        "VersionConstraint",
        "VersionRange",
        # ── java.util / java.time / java.nio common ──
        "ofEpochMilli",
        "ofEpochSecond",
        "ofMillis",
        "ofSeconds",
        "ofNanos",
        "ofMinutes",
        "ofHours",
        "ofDays",
        "toEpochMilli",
        "toMillis",
        "toNanos",
        "toSeconds",
        "setProperty",
        "getProperty",
        "load",
        "store",
        "stringPropertyNames",
        "getAbsolutePath",
        "getCanonicalPath",
        "getAbsoluteFile",
        "getCanonicalFile",
        "toPath",
        "toFile",
        "toURI",
        "toURL",
        "exists",
        "isDirectory",
        "isFile",
        "canRead",
        "canWrite",
        "canExecute",
        "mkdir",
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
        "rewind",
        "hasRemaining",
        "remaining",
        "position",
        "limit",
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
        "Setup",
        "SetupGet",
        "SetupSet",
        "SetupSequence",
        "SetupProperty",
        "SetupAllProperties",
        "Verify",
        "VerifyGet",
        "VerifySet",
        "VerifyAll",
        "VerifyNoOtherCalls",
        "ReturnsAsync",
        "Callback",
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
        "CancellationTokenSource",
        "CancellationToken",
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
        "GetProperty",
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
        "hasSize",
        "hasEntry",
        "hasKey",
        "hasValue",
        "hasItem",
        "hasItems",
        "hasItemInArray",
        "hasProperty",
        "containsInAnyOrder",
        "containsInRelativeOrder",
        "arrayContaining",
        "arrayContainingInAnyOrder",
        "arrayWithSize",
        "emptyArray",
        "emptyIterable",
        "emptyCollectionOf",
        "emptyIterableOf",
        "iterableWithSize",
        "nullValue",
        "notNullValue",
        "isIn",
        "isOneOf",
        "instanceOf",
        "isA",
        "typeCompatibleWith",
        "comparesEqualTo",
        "greaterThan",
        "greaterThanOrEqualTo",
        "lessThan",
        "lessThanOrEqualTo",
        "closeTo",
        "either",
        "both",
        "allOf",
        "anyOf",
        "not",
        "is",
        "anything",
        "equalTo",
        "equalToObject",
        "equalToCompressingWhiteSpace",
        "equalToIgnoringCase",
        "endsWith",
        "endsWithIgnoringCase",
        "startsWith",
        "startsWithIgnoringCase",
        "containsString",
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
        "Match",
        "MatchWildcard",
        "notifyErrorListeners",
        "recoverInline",
        "consume",
        "expect",
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
        "IllegalStateException",
        "IllegalArgumentException",
        "NullPointerException",
        "NumberFormatException",
        "ArrayIndexOutOfBoundsException",
        "IndexOutOfBoundsException",
        "ClassCastException",
        "ConcurrentModificationException",
        "UnsupportedOperationException",
        "NoSuchElementException",
        "NoSuchMethodException",
        "NoSuchFieldException",
        "InterruptedException",
        "IOException",
        "FileNotFoundException",
        "EOFException",
        "SocketException",
        "SocketTimeoutException",
        "UnknownHostException",
        "MalformedURLException",
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
        "Lock",
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
        "Guid",
        "InterfaceType",
        "PreserveSig",
        "MarshalAsAttribute",
        # ── POSIX threads (pthread_* — language-general C API) ──
        "pthread_create",
        "pthread_join",
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
        "pthread_mutex_init",
        "pthread_mutex_destroy",
        "pthread_mutex_lock",
        "pthread_mutex_unlock",
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
        "pthread_cond_init",
        "pthread_cond_destroy",
        "pthread_cond_wait",
        "pthread_cond_timedwait",
        "pthread_cond_signal",
        "pthread_cond_broadcast",
        "pthread_condattr_init",
        "pthread_condattr_destroy",
        "pthread_condattr_setpshared",
        "pthread_condattr_getpshared",
        "pthread_rwlock_init",
        "pthread_rwlock_destroy",
        "pthread_rwlock_rdlock",
        "pthread_rwlock_wrlock",
        "pthread_rwlock_unlock",
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
        "gethostbyname",
        "gethostbyaddr",
        "gethostname",
        "getservbyname",
        "getservbyport",
        "socket",
        "bind",
        "listen",
        "accept",
        "connect",
        "send",
        "recv",
        "sendto",
        "recvfrom",
        "sendmsg",
        "recvmsg",
        "shutdown",
        "setsockopt",
        "getsockopt",
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
        "getMethod",
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
        "getAttribute",
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
        "move",
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
        "seal",
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
    # ── Common receiver-variable patterns in Python tests and code.
    # Kept conservative: removed descry-specific prefixes
    # (authenticatedPage, pptx, spinner, node_id, mcp, prs, stripped)
    # and short/generic ones (c, line, table, group, cert, np, pb)
    # that would over-match project code in other languages too.
    "pytest.",
    "subparsers.",
    "logger.",
    "response.",
    "match.",
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
    # ── Common Python local-variable receiver patterns ──
    # ArgumentParser / subparser common receiver names.
    "parser.",
    "sub.",
    "subparsers.",
    "subcommands.",
    "args.",
    "ns.",
    "namespace.",
    # Path / file / graph receiver names commonly used across descry code.
    "path_obj.",
    "p.",
    "cfg.",
    "config.",
    "gp.",
    "fd.",
    "fh.",
    "result.",
    "proc.",
    "res.",
    "req.",
    "resp.",
    "request.",
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
    "TYPE",
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


def is_non_project_call(callee: str) -> bool:
    """Check if a callee is a non-project call (stdlib or third-party library) that should be filtered."""
    # Direct match
    if callee in STDLIB_FILTER:
        return True

    # Extract last component for method calls. Split on `.` (most langs),
    # `::` (Rust/C++/Ruby), and `->` (PHP / C struct pointer). We want
    # the trailing bare method name regardless of how the language
    # qualifies it, so `headers->set` matches `set` in the filter.
    last_part = callee.split(".")[-1].split("::")[-1].split("->")[-1]

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
            # D.1: a function is a method iff its enclosing parent node is a Class.
            # Previous logic used string matching on parent_id which incorrectly
            # classified nested functions inside other functions as Methods.
            parent_node = next(
                (n for n in self.builder.nodes if n["id"] == parent_id), None
            )
            is_method = parent_node is not None and parent_node["type"] == "Class"
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
        re_struct = re.compile(r"^\s*(?:pub\s+)?(?:struct|trait)\s+([a-zA-Z0-9_]+)")
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
        current_trait_impl = (
            None  # Track trait name when in "impl Trait for Struct" block
        )
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
                    type_id,
                    "Class",
                    name=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
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
                    type_id,
                    "Class",
                    name=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring,
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
            if not self.builder.use_ast_grep and parent_id != file_id:
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
            if "::" in callee:
                parts = callee.split("::")
                if len(parts) >= 2:
                    method_name = parts[-1]
                    struct_name = parts[-2]
                    # Match constructor patterns (exact or prefix for with_*)
                    is_constructor = (
                        method_name in constructor_patterns
                        or method_name.startswith("with_")
                        or method_name.startswith("from_")
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
        if self.builder.use_ast_grep and extract_imports_typescript is not None:
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
        # D.4: merged alternation to avoid ReDoS. Previous pattern
        # (?:[^)]*|[^=]*) had overlapping unbounded alternatives which could
        # produce catastrophic backtracking on pathological long lines.
        re_const_arrow = re.compile(
            r"^\s*(?:export\s+)?const\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?[^)=]*\s*=>"
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
                self.builder.add_node(
                    config_id,
                    "Configuration",
                    name=config_name,
                    signature=f"{client_name}.interceptors.{interceptor_type}.use(...)",
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=token_count,
                    docstring=docstring
                    or f"Configures {interceptor_type} interceptor for {client_name}",
                    config_type="interceptor",
                    target=client_name,
                )
                self.builder.add_edge(file_id, config_id, "DEFINES")
            elif match := re_middleware.search(line):
                middleware_name = match.group(2).strip()
                if middleware_name and not middleware_name.startswith("("):
                    config_name = f"middleware_{middleware_name.replace('/', '_').replace('.', '_')}"
                    config_id = f"{file_id}::{config_name}"
                    end_lineno = (
                        self._find_block_end(lines, i) if "{" in line else lineno
                    )
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
                end_lineno = (
                    self._find_block_end(lines, i)
                    if "{" in line or "=>" in line
                    else lineno
                )
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
                    docstring=docstring
                    or f"Handles '{event_name}' event on {emitter_name}",
                    config_type="event_handler",
                    target=emitter_name,
                    event=event_name,
                )
                self.builder.add_edge(file_id, config_id, "DEFINES")

            # Calls (regex fallback - only used if ast-grep unavailable)
            if not self.builder.use_ast_grep and parent_id != file_id:
                for match in re_call_candidate.finditer(line):
                    callee = match.group(1)
                    # JS/TS control-flow keywords look like calls (`if(x)`,
                    # `while (y)`). Reject them by simple name — language
                    # general, not codebase specific.
                    simple_name = callee.split(".")[-1]
                    if simple_name in _JSTS_CONTROL_KEYWORDS:
                        continue
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
        if self.builder.use_ast_grep:
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


class CodeGraphBuilder:
    # ast-grep is invoked as a subprocess **per file** for TS/JS parsing
    # (imports + calls). On macOS each `fork+exec` costs ~10ms, so a
    # 20k-file monorepo (next.js, TypeScript compiler) spends 10+
    # minutes in fork overhead alone before any real work happens. On
    # corpora above this threshold we fall back to the regex parser
    # only — accuracy is slightly lower for TS imports but the index
    # actually completes. Overridable via ``DESCRY_AST_GREP_MAX_FILES``.
    _AST_GREP_MAX_FILES_DEFAULT = 5000

    def __init__(self, root_dir, excluded_dirs=None):
        self.root_dir = Path(root_dir).resolve()
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
            return path.relative_to(self.root_dir).as_posix()
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
                    if fn.endswith((".ts", ".tsx", ".js", ".jsx")):
                        ts_js_count += 1
                        if ts_js_count > self._AST_GREP_MAX_FILES_DEFAULT * 10:
                            break
                if ts_js_count > self._AST_GREP_MAX_FILES_DEFAULT * 10:
                    break
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
                except Exception as e:
                    logger.warning(f"Error reading {rel_path}: {e}")
                    continue
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
                elif file.endswith((".rb", ".rake", ".gemspec")):
                    from descry.ruby_parser import RubyParser

                    RubyParser(self).parse(file_path, rel_path, content)
                elif file.endswith(".php"):
                    from descry.php_parser import PhpParser

                    PhpParser(self).parse(file_path, rel_path, content)
                elif file.endswith((".cs", ".vb")):
                    from descry.dotnet_parser import DotnetParser

                    DotnetParser(self).parse(file_path, rel_path, content)
                elif file.endswith(
                    (".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx")
                ):
                    from descry.clang_parser import ClangParser

                    ClangParser(self).parse(file_path, rel_path, content)
                elif file.endswith(".dart"):
                    from descry.dart_parser import DartParser

                    DartParser(self).parse(file_path, rel_path, content)
                elif (
                    file.endswith(".ts")
                    or file.endswith(".tsx")
                    or file.endswith(".js")
                ):
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
                        TSParser(self).parse(file_path, rel_path, body)
                        # Shift newly-added node linenos by the offset.
                        for node in self.nodes[before:]:
                            meta = node.get("metadata", {})
                            for k in ("lineno", "end_lineno"):
                                v = meta.get(k)
                                if isinstance(v, int):
                                    meta[k] = v + line_offset
                        pos = close + len("</script>")

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
        elif file_part.endswith((".java", ".kt", ".scala")):
            return "java"
        elif file_part.endswith(".go"):
            return "go"
        elif file_part.endswith((".rb", ".rake", ".gemspec")):
            return "ruby"
        elif file_part.endswith(".php"):
            return "php"
        elif file_part.endswith((".cs", ".vb")):
            return "dotnet"
        elif file_part.endswith(
            (".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx")
        ):
            return "clang"
        elif file_part.endswith(".dart"):
            return "dart"
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
                    scip_resolved = scip_index.resolve(ref_name, source_file, lineno)
                    if scip_resolved:
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
                if not resolved and "::" in base_ref and base_ref in qualified_lookup:
                    matches = [
                        m for m in qualified_lookup[base_ref] if m[1] == source_lang
                    ]
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
                            matches = [
                                m
                                for m in qualified_lookup[qualified_self]
                                if m[1] == source_lang
                            ]
                            if matches:
                                resolved = matches[0][0]
                                resolution_source = "self_type"
                        # Also try via node_lookup with struct preference
                        if not resolved and self_method in node_lookup:
                            matches = [
                                m
                                for m in node_lookup[self_method]
                                if m[1] == source_lang
                            ]
                            struct_matches = [
                                m
                                for m in matches
                                if f"::{source_struct}::{self_method}" in m[0]
                            ]
                            if struct_matches:
                                resolved = struct_matches[0][0]
                                resolution_source = "self_type"

                # Strategy 2: For self.method, prefer same-struct methods
                if not resolved and ref_name.startswith("self."):
                    method_name = ref_name.replace("self.", "")
                    if method_name in node_lookup:
                        matches = [
                            m for m in node_lookup[method_name] if m[1] == source_lang
                        ]
                        if matches:
                            # Extract struct name from source (FILE:path::Struct::method)
                            source_parts = source_id.split("::")
                            if len(source_parts) >= 3:
                                source_struct = source_parts[-2]
                                # Prefer methods in the same struct
                                struct_matches = [
                                    m
                                    for m in matches
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
                    if edge["relation"] == "CALLS":
                        edge["relation"] = "CALLS_RESOLVED"
                        stats[resolution_source] += 1
                    elif edge["relation"] == "INSTANTIATES":
                        edge["relation"] = "INSTANTIATES_RESOLVED"
                elif edge["relation"] == "CALLS":
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

    # Load config: auto-detect from target dir, apply .descry.toml, then
    # apply env vars (DESCRY_CACHE_DIR, DESCRY_NO_SCIP, DESCRY_NO_EMBEDDINGS,
    # etc.). Calling from_env() directly would use Path.cwd() which may
    # differ from the target directory, so we replicate its logic explicitly.
    from descry.handlers import DescryConfig, _env

    # Handle SCIP opt-out CLI flag before from_env reads the env.
    if args.no_scip:
        os.environ["DESCRY_NO_SCIP"] = "1"

    config = DescryConfig(project_root=Path(target).resolve())
    toml_data = DescryConfig._load_toml(config.project_root)
    config._apply_toml(toml_data)

    # Apply env var overrides (mirrors DescryConfig.from_env).
    cache_dir_str = _env("DESCRY_CACHE_DIR")
    if cache_dir_str:
        config.cache_dir = Path(cache_dir_str)
    if _env("DESCRY_NO_SCIP").lower() in ("1", "true", "yes"):
        config.enable_scip = False
    if _env("DESCRY_NO_EMBEDDINGS").lower() in ("1", "true", "yes"):
        config.enable_embeddings = False

    # B.3: always pass config.excluded_dirs so non-TOML projects still get
    # the full default exclusion set.
    config_excluded_dirs = config.excluded_dirs

    # B.3: respect config.cache_dir instead of hardcoded ".descry_cache"
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build the graph
    builder = CodeGraphBuilder(target, excluded_dirs=config_excluded_dirs)
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
                Path(target).resolve(),
                scip_extra_args=config.scip_extra_args,
                scip_skip_crates=config.scip_skip_crates,
                scip_toolchain=config.scip_rust_toolchain,
                scip_timeout_minutes=config.scip_timeout_minutes,
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
            logger.info(
                "SCIP: Unavailable (no indexers found: install rust-analyzer and/or scip-typescript)"
            )

    # Export with optional SCIP resolution
    graph_path = cache_dir / "codebase_graph.json"
    builder.export(str(graph_path), scip_index=scip_index)

    # Generate embeddings for semantic search (if dependencies available and enabled)
    if config.enable_embeddings:
        try:
            from descry.embeddings import embeddings_available, SemanticSearcher

            if embeddings_available():
                logger.info("Generating embeddings for semantic search...")
                # B.3: respect config.embedding_model
                searcher = SemanticSearcher(
                    str(graph_path),
                    force_rebuild=True,
                    model_name=config.embedding_model,
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
        except Exception as e:
            logger.warning(f"Embeddings: Failed to generate ({e})")


if __name__ == "__main__":
    main()
