"""Descry CLI — standalone command-line interface for codebase knowledge graph tools.

Delegates all business logic to DescryService.

Usage:
    descry health
    descry search foo bar --limit 5
    descry context FILE:src/lib.rs::main --full
"""

import argparse
import asyncio
import sys

from descry.handlers import DescryService, DescryConfig


def _make_service() -> DescryService:
    config = DescryConfig.from_env()
    return DescryService(config)


def _run(coro) -> str:
    return asyncio.run(coro)


def _print_result(coro):
    """Run a service coroutine, print the result, and translate service-side
    errors to a nonzero exit code so shell chains like `descry callers X &&
    next_step` don't silently advance on "ERROR: Graph not found".
    """
    result = _run(coro)
    print(result)
    if isinstance(result, str) and result.lstrip().upper().startswith("ERROR:"):
        sys.exit(2)


def cmd_health(_args):
    svc = _make_service()
    _print_result(svc.health())


def cmd_status(_args):
    svc = _make_service()
    _print_result(svc.status())


def cmd_ensure(args):
    svc = _make_service()
    _print_result(svc.ensure(max_age_hours=args.max_age_hours))


def cmd_index(args):
    svc = _make_service()
    path = args.path if args.path else "."
    _print_result(svc.index(path))


def cmd_callers(args):
    svc = _make_service()
    _print_result(svc.callers(name=args.name, limit=args.limit))


def cmd_callees(args):
    svc = _make_service()
    _print_result(svc.callees(name=args.name, limit=args.limit))


def cmd_context(args):
    svc = _make_service()
    _print_result(
        svc.context(
            node_id=args.node_id,
            brief=args.brief,
            full=args.full,
            expand_callees=args.expand_callees,
            deduplicate=args.deduplicate,
            depth=args.depth,
            max_tokens=args.max_tokens,
            callee_budget=args.callee_budget,
            head_lines=args.head_lines,
            max_output_tokens=args.max_output_tokens,
        )
    )


def cmd_flow(args):
    svc = _make_service()
    _print_result(
        svc.flow(
            start=args.start,
            direction=args.direction,
            depth=args.depth,
            target=args.target,
            inline_threshold=args.inline_threshold,
        )
    )


def cmd_search(args):
    svc = _make_service()
    _print_result(
        svc.search(
            terms=args.terms,
            compact=args.compact,
            limit=args.limit,
            lang=args.lang,
            crate=args.crate,
            symbol_type=args.type,
            exclude_tests=args.exclude_tests,
        )
    )


def cmd_structure(args):
    svc = _make_service()
    _print_result(svc.structure(filename=args.filename))


def cmd_flatten(args):
    svc = _make_service()
    _print_result(svc.flatten(class_node_id=args.class_node_id))


def cmd_semantic(args):
    svc = _make_service()
    _print_result(svc.semantic(query=args.query, limit=args.limit))


def cmd_quick(args):
    svc = _make_service()
    _print_result(svc.quick(name=args.name, full=args.full, brief=args.brief))


def cmd_impls(args):
    svc = _make_service()
    _print_result(svc.impls(method=args.method, trait_name=args.trait_name))


def cmd_path(args):
    svc = _make_service()
    _print_result(
        svc.path(
            start=args.start,
            end=args.end,
            max_depth=args.max_depth,
            direction=args.direction,
        )
    )


def cmd_cross_lang(args):
    svc = _make_service()
    _print_result(
        svc.cross_lang(
            mode=args.mode,
            method=args.method,
            path=args.path,
            tag=args.tag,
        )
    )


def cmd_churn(args):
    svc = _make_service()
    _print_result(
        svc.churn(
            time_range=args.time_range,
            path_filter=args.path_filter,
            limit=args.limit,
            mode=args.mode,
            exclude_generated=args.exclude_generated,
        )
    )


def cmd_evolution(args):
    svc = _make_service()
    _print_result(
        svc.evolution(
            name=args.name,
            time_range=args.time_range,
            limit=args.limit,
            show_diff=args.show_diff,
            crate=args.crate,
        )
    )


def cmd_changes(args):
    svc = _make_service()
    _print_result(
        svc.changes(
            commit_range=args.commit_range,
            time_range=args.time_range,
            path_filter=args.path_filter,
            show_callers=args.show_callers,
            limit=args.limit,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        prog="descry",
        description="Codebase knowledge graph tools",
    )
    subparsers = parser.add_subparsers(dest="command")

    # health
    sub = subparsers.add_parser("health", help="Quick diagnostic check")
    sub.set_defaults(func=cmd_health)

    # status
    sub = subparsers.add_parser("status", help="Check graph existence and freshness")
    sub.set_defaults(func=cmd_status)

    # ensure
    sub = subparsers.add_parser("ensure", help="Ensure graph exists and is fresh")
    sub.add_argument(
        "--max-age-hours",
        type=float,
        default=24,
        help="Max graph age in hours (default: 24)",
    )
    sub.set_defaults(func=cmd_ensure)

    # index
    sub = subparsers.add_parser("index", help="Regenerate the codebase graph")
    sub.add_argument(
        "path", nargs="?", default=None, help="Path to index (default: project root)"
    )
    sub.set_defaults(func=cmd_index)

    # callers
    sub = subparsers.add_parser("callers", help="Find all callers of a symbol")
    sub.add_argument("name", help="Symbol name to look up")
    sub.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    sub.set_defaults(func=cmd_callers)

    # callees
    sub = subparsers.add_parser("callees", help="Find what a symbol calls")
    sub.add_argument("name", help="Symbol name to look up")
    sub.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    sub.set_defaults(func=cmd_callees)

    # context
    sub = subparsers.add_parser(
        "context", help="Get full context for a symbol by node ID"
    )
    sub.add_argument("node_id", help="Graph node ID (e.g. FILE:src/lib.rs::main)")
    sub.add_argument("--brief", action="store_true", help="Brief output")
    sub.add_argument("--full", action="store_true", help="Full source output")
    sub.add_argument(
        "--expand-callees", action="store_true", help="Include callee source"
    )
    sub.add_argument(
        "--deduplicate", action="store_true", help="Deduplicate repeated lookups"
    )
    sub.add_argument(
        "--depth", type=int, default=1, help="Traversal depth (default: 1)"
    )
    sub.add_argument(
        "--max-tokens", type=int, default=2000, help="Max token budget (default: 2000)"
    )
    sub.add_argument(
        "--callee-budget",
        type=int,
        default=2000,
        help="Token budget for expanded callees (default: 2000)",
    )
    sub.add_argument(
        "--head-lines",
        type=int,
        default=None,
        help="Truncate source head to N lines (default: full)",
    )
    sub.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Overall output token cap (default: unlimited)",
    )
    sub.set_defaults(func=cmd_context)

    # flow
    sub = subparsers.add_parser("flow", help="Trace call flow from a starting symbol")
    sub.add_argument("start", help="Starting symbol name")
    sub.add_argument(
        "--direction",
        default="forward",
        choices=["forward", "backward"],
        help="Trace direction (default: forward)",
    )
    sub.add_argument("--depth", type=int, default=3, help="Trace depth (default: 3)")
    sub.add_argument("--target", default=None, help="Target symbol to reach")
    sub.add_argument(
        "--inline-threshold",
        type=int,
        default=100,
        help="Inline callee source below this token count (default: 100)",
    )
    sub.set_defaults(func=cmd_flow)

    # search
    sub = subparsers.add_parser("search", help="Search symbol names and docstrings")
    sub.add_argument("terms", nargs="+", help="Search terms")
    sub.add_argument(
        "--compact", action="store_true", default=True, help="Compact output (default)"
    )
    sub.add_argument(
        "--no-compact", dest="compact", action="store_false", help="Detailed output"
    )
    sub.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    sub.add_argument("--lang", default=None, help="Filter by language")
    sub.add_argument("--crate", default=None, help="Filter by crate")
    sub.add_argument("--type", default=None, help="Filter by symbol type")
    sub.add_argument(
        "--exclude-tests", action="store_true", help="Exclude test symbols"
    )
    sub.set_defaults(func=cmd_search)

    # structure
    sub = subparsers.add_parser("structure", help="Show file structure")
    sub.add_argument("filename", help="Filename to inspect")
    sub.set_defaults(func=cmd_structure)

    # flatten
    sub = subparsers.add_parser(
        "flatten", help="Show effective API of a class including inherited methods"
    )
    sub.add_argument("class_node_id", help="Class node ID")
    sub.set_defaults(func=cmd_flatten)

    # semantic
    sub = subparsers.add_parser(
        "semantic", help="Pure semantic search using embeddings"
    )
    sub.add_argument("query", help="Natural language query")
    sub.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    sub.set_defaults(func=cmd_semantic)

    # quick
    sub = subparsers.add_parser(
        "quick", help="Find symbol and show full context in one step"
    )
    sub.add_argument("name", help="Symbol name")
    sub.add_argument("--full", action="store_true", help="Full source output")
    sub.add_argument("--brief", action="store_true", help="Brief output")
    sub.set_defaults(func=cmd_quick)

    # impls
    sub = subparsers.add_parser(
        "impls", help="Find all implementations of a trait method"
    )
    sub.add_argument("method", help="Method name")
    sub.add_argument("--trait-name", default=None, help="Filter by trait name")
    sub.set_defaults(func=cmd_impls)

    # path
    sub = subparsers.add_parser(
        "path", help="Find shortest call path between two symbols"
    )
    sub.add_argument("start", help="Starting symbol")
    sub.add_argument("end", help="Ending symbol")
    sub.add_argument(
        "--max-depth", type=int, default=10, help="Max search depth (default: 10)"
    )
    sub.add_argument(
        "--direction",
        default="forward",
        choices=["forward", "backward"],
        help="Search direction (default: forward)",
    )
    sub.set_defaults(func=cmd_path)

    # cross-lang
    sub = subparsers.add_parser(
        "cross-lang", help="Trace API calls from frontend to backend via OpenAPI"
    )
    sub.add_argument(
        "--mode",
        default="endpoint",
        choices=["endpoint", "list", "stats"],
        help="Trace mode (default: endpoint)",
    )
    sub.add_argument("--method", default=None, help="HTTP method (e.g. GET, POST)")
    sub.add_argument("--path", default=None, help="API path (e.g. /api/v1/deployments)")
    sub.add_argument("--tag", default=None, help="Filter by tag")
    sub.set_defaults(func=cmd_cross_lang)

    # churn
    sub = subparsers.add_parser("churn", help="Find code churn hotspots")
    sub.add_argument("--time-range", default=None, help='Time range (e.g. "30 days")')
    sub.add_argument("--path-filter", default=None, help='Path filter (e.g. "src/")')
    sub.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    sub.add_argument(
        "--mode",
        default="symbols",
        choices=["symbols", "files", "co-change"],
        help="Churn mode (default: symbols)",
    )
    sub.add_argument(
        "--include-generated",
        dest="exclude_generated",
        action="store_false",
        default=True,
        help="Include generated/lockfiles (default: excluded)",
    )
    sub.set_defaults(func=cmd_churn)

    # evolution
    sub = subparsers.add_parser(
        "evolution", help="Track how a symbol has changed over time"
    )
    sub.add_argument("name", help="Symbol name")
    sub.add_argument("--time-range", default=None, help='Time range (e.g. "90 days")')
    sub.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    sub.add_argument("--show-diff", action="store_true", help="Show diffs")
    sub.add_argument("--crate", default=None, help="Filter by crate")
    sub.set_defaults(func=cmd_evolution)

    # changes
    sub = subparsers.add_parser(
        "changes", help="Analyze change impact for a commit range"
    )
    sub.add_argument(
        "--commit-range", default=None, help="Commit range (e.g. HEAD~3..HEAD)"
    )
    sub.add_argument("--time-range", default=None, help='Time range (e.g. "7 days")')
    sub.add_argument("--path-filter", default=None, help='Path filter (e.g. "src/")')
    sub.add_argument(
        "--show-callers",
        action="store_true",
        default=True,
        help="Show callers of changed symbols (default)",
    )
    sub.add_argument(
        "--no-show-callers",
        dest="show_callers",
        action="store_false",
        help="Hide callers",
    )
    sub.add_argument("--limit", type=int, default=50, help="Max results (default: 50)")
    sub.set_defaults(func=cmd_changes)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
