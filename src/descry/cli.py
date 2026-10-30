"""Descry CLI — standalone command-line interface for codebase knowledge graph tools.

Delegates all business logic to DescryService.

Usage:
    descry health
    descry search foo bar --limit 5
    descry context FILE:src/lib.rs::main --full
"""

import argparse
import asyncio
import json
import sys

from descry._answers import fail
from descry._streams import write_utf8
from descry.handlers import (
    ChurnMode,
    CrossLangMode,
    DescryConfig,
    DescryService,
    Direction,
    SymbolType,
    public,
)


def _make_service() -> DescryService:
    config = DescryConfig.from_env()
    # A process that ends with its answer waits for the embeddings it loads.
    config.embedding_timeout = None
    return DescryService(config)


def _print_answer(args, answer, render):
    """Print a tool's answer: with `--json` the answer itself, else the text
    `render` makes of it. A failed answer exits 2, so shell chains like
    `descry callers X && next_step` stop there."""

    async def run():
        found = await answer
        return found, None if args.json else await render(found)

    found, text = asyncio.run(run())
    print(json.dumps(public(found), indent=2) if args.json else text)
    if not found["ok"]:
        sys.exit(2)


def cmd_health(args):
    svc = _make_service()
    _print_answer(args, svc.health_answer(), svc.health_text)


def cmd_status(args):
    svc = _make_service()
    _print_answer(args, svc.status_answer(), svc.status_text)


def cmd_ensure(args):
    svc = _make_service()
    _print_answer(args, svc.ensure_answer(args.max_age_hours), svc.ensure_text)


def cmd_index(args):
    svc = _make_service()
    path = args.path if args.path else "."
    _print_answer(args, svc.index_answer(path), svc.index_text)


def cmd_callers(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.callers_answer(args.symbol, args.limit),
        lambda a: svc.listing_text(a, "callers", args.limit),
    )


def cmd_callees(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.callees_answer(args.symbol, args.limit),
        lambda a: svc.listing_text(a, "callees", args.limit),
    )


def cmd_context(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.context_answer(
            args.symbol,
            brief=args.brief,
            full=args.full,
            expand_callees=args.expand_callees,
            deduplicate=args.deduplicate,
            depth=args.depth,
            max_tokens=args.max_tokens,
            callee_budget=args.callee_budget,
            head_lines=args.head_lines,
            max_output_tokens=args.max_output_tokens,
        ),
        svc.context_text,
    )


def cmd_flow(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.flow_answer(
            args.start, args.direction, args.depth, args.target, args.inline_threshold
        ),
        svc.flow_text,
    )


def cmd_search(args):
    svc = _make_service()
    filters = (args.lang, args.crate, args.type)
    _print_answer(
        args,
        svc.search_answer(args.terms, args.limit, *filters, args.exclude_tests),
        lambda a: svc.search_text(
            a, args.terms, args.compact, *filters, args.exclude_tests
        ),
    )


def cmd_structure(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.structure_answer(args.filename),
        lambda a: svc.structure_text(a, args.filename),
    )


def cmd_flatten(args):
    svc = _make_service()
    _print_answer(args, svc.flatten_answer(args.symbol), svc.flatten_text)


def cmd_embedding_models(args):
    svc = _make_service()
    _print_answer(args, svc.embedding_models_answer(), svc.health_text)


def cmd_semantic(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.semantic_answer(args.query, args.limit),
        lambda a: svc.semantic_text(a, args.query),
    )


def cmd_quick(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.quick_answer(args.symbol, args.full, args.brief),
        lambda a: svc.quick_text(a, args.symbol, args.full, args.brief),
    )


def cmd_impls(args):
    svc = _make_service()
    _print_answer(args, svc.impls_answer(args.method, args.trait_name), svc.impls_text)


def cmd_path(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.path_answer(args.start, args.end, args.max_depth, args.direction),
        lambda a: svc.path_text(a, args.start, args.end),
    )


def cmd_cross_lang(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.cross_lang_answer(args.mode, args.method, args.path, args.tag),
        lambda a: svc.cross_lang_text(a, args.method, args.path, args.tag),
    )


def cmd_churn(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.churn_answer(
            args.time_range,
            args.path_filter,
            args.limit,
            args.mode,
            args.exclude_generated,
        ),
        svc.churn_text,
    )


def cmd_evolution(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.evolution_answer(
            args.symbol, args.time_range, args.limit, args.show_diff, args.crate
        ),
        svc.evolution_text,
    )


def cmd_changes(args):
    svc = _make_service()
    _print_answer(
        args,
        svc.changes_answer(
            args.commit_range,
            args.time_range,
            args.path_filter,
            args.show_callers,
            args.limit,
        ),
        svc.changes_text,
    )


class _Parser(argparse.ArgumentParser):
    """A parser that, asked for `--json`, answers an argument it cannot read
    as every failure is answered, `invalid`, rather than with its usage."""

    def error(self, message):
        if "--json" in sys.argv[1:]:
            print(json.dumps(public(fail("invalid", message)), indent=2))
            sys.exit(2)
        super().error(message)


def build_parser():
    """The `descry` command line's parser, every command a subparser."""
    parser = _Parser(
        prog="descry",
        description="Codebase knowledge graph tools",
    )
    subparsers = parser.add_subparsers(dest="command")
    answers = argparse.ArgumentParser(add_help=False)
    answers.add_argument(
        "--json", action="store_true", help="Print the structured answer as JSON"
    )

    sub = subparsers.add_parser(
        "health", parents=[answers], help="Quick diagnostic check"
    )
    sub.set_defaults(func=cmd_health)

    sub = subparsers.add_parser(
        "status", parents=[answers], help="Check graph existence and freshness"
    )
    sub.set_defaults(func=cmd_status)

    sub = subparsers.add_parser(
        "ensure", parents=[answers], help="Ensure graph exists and is fresh"
    )
    sub.add_argument(
        "--max-age-hours",
        type=float,
        default=None,
        help="Max graph age in hours (default: [project] max_stale_hours)",
    )
    sub.set_defaults(func=cmd_ensure)

    sub = subparsers.add_parser(
        "index", parents=[answers], help="Regenerate the codebase graph"
    )
    sub.add_argument(
        "path", nargs="?", default=None, help="Path to index (default: project root)"
    )
    sub.set_defaults(func=cmd_index)

    sub = subparsers.add_parser(
        "callers", parents=[answers], help="Find all callers of a symbol"
    )
    sub.add_argument("symbol", help="A node ID, or a name")
    sub.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    sub.set_defaults(func=cmd_callers)

    sub = subparsers.add_parser(
        "callees", parents=[answers], help="Find what a symbol calls"
    )
    sub.add_argument("symbol", help="A node ID, or a name")
    sub.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    sub.set_defaults(func=cmd_callees)

    sub = subparsers.add_parser(
        "context", parents=[answers], help="Get full context for a symbol"
    )
    sub.add_argument("symbol", help="A node ID, or a name (e.g. FILE:src/lib.rs::main)")
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

    sub = subparsers.add_parser(
        "flow", parents=[answers], help="Trace call flow from a starting symbol"
    )
    sub.add_argument("start", help="Starting symbol name")
    sub.add_argument(
        "--direction",
        default="forward",
        choices=[d.value for d in Direction],
        help="Trace direction (default: forward)",
    )
    sub.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Levels of calls below the start (default: 3)",
    )
    sub.add_argument(
        "--target", default=None, help="Symbol to reach, by name or node ID"
    )
    sub.add_argument(
        "--inline-threshold",
        type=int,
        default=100,
        help="Inline callee source below this token count (default: 100)",
    )
    sub.set_defaults(func=cmd_flow)

    sub = subparsers.add_parser(
        "search", parents=[answers], help="Search symbol names and docstrings"
    )
    sub.add_argument("terms", nargs="+", help="Search terms")
    sub.add_argument(
        "--compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compact output; --no-compact for detailed",
    )
    sub.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    sub.add_argument("--lang", default=None, help="Filter by language")
    sub.add_argument("--crate", default=None, help="Filter by crate")
    sub.add_argument(
        "--type",
        default=None,
        choices=[t.value for t in SymbolType],
        help="Filter by symbol type",
    )
    sub.add_argument(
        "--exclude-tests", action="store_true", help="Exclude test symbols"
    )
    sub.set_defaults(func=cmd_search)

    sub = subparsers.add_parser(
        "structure", parents=[answers], help="Show file structure"
    )
    sub.add_argument("filename", help="Filename to inspect")
    sub.set_defaults(func=cmd_structure)

    sub = subparsers.add_parser(
        "flatten",
        parents=[answers],
        help="Show effective API of a class including inherited methods",
    )
    sub.add_argument("symbol", help="A class's node ID, or its name")
    sub.set_defaults(func=cmd_flatten)

    sub = subparsers.add_parser(
        "semantic", parents=[answers], help="Pure semantic search using embeddings"
    )
    sub.add_argument("query", help="Natural language query")
    sub.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    sub.set_defaults(func=cmd_semantic)

    sub = subparsers.add_parser(
        "embedding-models",
        parents=[answers],
        help="List embedding models descry ships pinned configurations for",
    )
    sub.set_defaults(func=cmd_embedding_models)

    sub = subparsers.add_parser(
        "quick", parents=[answers], help="Find symbol and show full context in one step"
    )
    sub.add_argument("symbol", help="A node ID, or a name")
    sub.add_argument("--full", action="store_true", help="Full source output")
    sub.add_argument("--brief", action="store_true", help="Brief output")
    sub.set_defaults(func=cmd_quick)

    sub = subparsers.add_parser(
        "impls",
        parents=[answers],
        help="Find all implementations of a Rust trait method",
    )
    sub.add_argument("method", help="Method name")
    sub.add_argument("--trait-name", default=None, help="Filter by trait name")
    sub.set_defaults(func=cmd_impls)

    sub = subparsers.add_parser(
        "path", parents=[answers], help="Find shortest call path between two symbols"
    )
    sub.add_argument("start", help="Starting symbol")
    sub.add_argument("end", help="Ending symbol")
    sub.add_argument(
        "--max-depth", type=int, default=10, help="Max search depth (default: 10)"
    )
    sub.add_argument(
        "--direction",
        default="forward",
        choices=[d.value for d in Direction],
        help="Search direction (default: forward)",
    )
    sub.set_defaults(func=cmd_path)

    sub = subparsers.add_parser(
        "cross-lang",
        parents=[answers],
        help="Trace API calls from frontend to backend via OpenAPI",
    )
    sub.add_argument(
        "--mode",
        default="endpoint",
        choices=[m.value for m in CrossLangMode],
        help="Trace mode (default: endpoint)",
    )
    sub.add_argument("--method", default=None, help="HTTP method (e.g. GET, POST)")
    sub.add_argument("--path", default=None, help="API path (e.g. /api/v1/deployments)")
    sub.add_argument("--tag", default=None, help="Filter by tag")
    sub.set_defaults(func=cmd_cross_lang)

    sub = subparsers.add_parser(
        "churn", parents=[answers], help="Find code churn hotspots"
    )
    sub.add_argument("--time-range", default=None, help='Time range (e.g. "30 days")')
    sub.add_argument("--path-filter", default=None, help='Path filter (e.g. "src/")')
    sub.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    sub.add_argument(
        "--mode",
        default="symbols",
        choices=[m.value for m in ChurnMode],
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

    sub = subparsers.add_parser(
        "evolution", parents=[answers], help="Track how a symbol has changed over time"
    )
    sub.add_argument("symbol", help="A node ID, or a name")
    sub.add_argument("--time-range", default=None, help='Time range (e.g. "90 days")')
    sub.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    sub.add_argument("--show-diff", action="store_true", help="Show diffs")
    sub.add_argument("--crate", default=None, help="Filter by crate")
    sub.set_defaults(func=cmd_evolution)

    sub = subparsers.add_parser(
        "changes", parents=[answers], help="Analyze change impact for a commit range"
    )
    sub.add_argument(
        "--commit-range", default=None, help="Commit range (e.g. HEAD~3..HEAD)"
    )
    sub.add_argument("--time-range", default=None, help='Time range (e.g. "7 days")')
    sub.add_argument("--path-filter", default=None, help='Path filter (e.g. "src/")')
    sub.add_argument(
        "--show-callers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show callers of changed symbols",
    )
    sub.add_argument("--limit", type=int, default=50, help="Max results (default: 50)")
    sub.set_defaults(func=cmd_changes)

    return parser


def main():
    write_utf8()
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
