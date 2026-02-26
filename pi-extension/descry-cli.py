#!/usr/bin/env python3
"""
Descry CLI Bridge for Pi Extension

Thin CLI that imports from the descry package and exposes each tool
as a subcommand with text output.

Usage:
    python descry-cli.py <subcommand> [args...]

Subcommands:
    health, ensure, status, callers, callees, context, search, structure,
    cross_lang, churn, quick, index, semantic, evolution, changes, flow, path,
    impls, flatten
"""

import argparse
import asyncio
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Descry CLI Bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # health
    subparsers.add_parser("health")

    # ensure
    p = subparsers.add_parser("ensure")
    p.add_argument("--max-age-hours", type=float, default=24)

    # status
    subparsers.add_parser("status")

    # callers
    p = subparsers.add_parser("callers")
    p.add_argument("--symbol", required=True)
    p.add_argument("--limit", type=int, default=20)

    # callees
    p = subparsers.add_parser("callees")
    p.add_argument("--symbol", required=True)
    p.add_argument("--limit", type=int, default=20)

    # context
    p = subparsers.add_parser("context")
    p.add_argument("--node-id", required=True)
    p.add_argument("--brief", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--expand-callees", action="store_true")
    p.add_argument("--deduplicate", action="store_true")
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=2000)

    # search
    p = subparsers.add_parser("search")
    p.add_argument("--query", required=True)
    p.add_argument("--no-compact", action="store_true")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--lang", default=None)
    p.add_argument("--crate", default=None)
    p.add_argument("--type", default=None)
    p.add_argument("--exclude-tests", action="store_true")

    # structure
    p = subparsers.add_parser("structure")
    p.add_argument("--path", required=True)

    # flatten
    p = subparsers.add_parser("flatten")
    p.add_argument("--node-id", required=True)

    # cross_lang
    p = subparsers.add_parser("cross_lang")
    p.add_argument("--mode", default="endpoint")
    p.add_argument("--method", default=None)
    p.add_argument("--path", default=None)
    p.add_argument("--tag", default=None)

    # churn
    p = subparsers.add_parser("churn")
    p.add_argument("--time-range", default=None)
    p.add_argument("--path-filter", default=None)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--mode", default="symbols")
    p.add_argument("--exclude-generated", action="store_true", default=True)

    # quick
    p = subparsers.add_parser("quick")
    p.add_argument("--name", required=True)
    p.add_argument("--full", action="store_true")
    p.add_argument("--brief", action="store_true")

    # impls
    p = subparsers.add_parser("impls")
    p.add_argument("--method", required=True)
    p.add_argument("--trait-name", default=None)

    # index
    p = subparsers.add_parser("index")
    p.add_argument("--path", default=".")

    # semantic
    p = subparsers.add_parser("semantic")
    p.add_argument("--query", required=True)
    p.add_argument("--limit", type=int, default=10)

    # evolution
    p = subparsers.add_parser("evolution")
    p.add_argument("--name", required=True)
    p.add_argument("--time-range", default=None)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--show-diff", action="store_true")
    p.add_argument("--crate", default=None)

    # changes
    p = subparsers.add_parser("changes")
    p.add_argument("--commit-range", default=None)
    p.add_argument("--time-range", default=None)
    p.add_argument("--path-filter", default=None)
    p.add_argument("--no-show-callers", action="store_true")
    p.add_argument("--limit", type=int, default=50)

    # flow
    p = subparsers.add_parser("flow")
    p.add_argument("--start", required=True)
    p.add_argument("--direction", default="forward")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--target", default=None)

    # path
    p = subparsers.add_parser("path")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--direction", default="forward")

    args = parser.parse_args()
    result = asyncio.run(dispatch(args))
    print(result)


async def dispatch(args):
    """Import from descry package and call the appropriate service method."""
    from descry.handlers import DescryService

    service = DescryService()
    cmd = args.command

    try:
        if cmd == "health":
            return await service.health()
        elif cmd == "ensure":
            return await service.ensure(args.max_age_hours)
        elif cmd == "status":
            return await service.status()
        elif cmd == "callers":
            return await service.callers(args.symbol, args.limit)
        elif cmd == "callees":
            return await service.callees(args.symbol, args.limit)
        elif cmd == "context":
            return await service.context(
                args.node_id,
                brief=args.brief,
                full=args.full,
                expand_callees=args.expand_callees,
                deduplicate=args.deduplicate,
                depth=args.depth,
                max_tokens=args.max_tokens,
            )
        elif cmd == "search":
            terms = args.query.split()
            compact = not args.no_compact
            return await service.search(
                terms,
                compact=compact,
                limit=args.limit,
                lang=args.lang,
                crate=args.crate,
                symbol_type=args.type,
                exclude_tests=args.exclude_tests,
            )
        elif cmd == "structure":
            return await service.structure(args.path)
        elif cmd == "flatten":
            return await service.flatten(args.node_id)
        elif cmd == "cross_lang":
            return await service.cross_lang(
                mode=args.mode,
                method=args.method,
                path=args.path,
                tag=args.tag,
            )
        elif cmd == "churn":
            return await service.churn(
                time_range=args.time_range,
                path_filter=args.path_filter,
                limit=args.limit,
                mode=args.mode,
                exclude_generated=args.exclude_generated,
            )
        elif cmd == "quick":
            return await service.quick(args.name, full=args.full, brief=args.brief)
        elif cmd == "impls":
            return await service.impls(args.method, trait_name=args.trait_name)
        elif cmd == "index":
            return await service.index(args.path)
        elif cmd == "semantic":
            return await service.semantic(args.query, limit=args.limit)
        elif cmd == "evolution":
            return await service.evolution(
                name=args.name,
                time_range=args.time_range,
                limit=args.limit,
                show_diff=args.show_diff,
                crate=args.crate,
            )
        elif cmd == "changes":
            show_callers = not args.no_show_callers
            return await service.changes(
                commit_range=args.commit_range,
                time_range=args.time_range,
                path_filter=args.path_filter,
                show_callers=show_callers,
                limit=args.limit,
            )
        elif cmd == "flow":
            return await service.flow(
                start=args.start,
                direction=args.direction,
                depth=args.depth,
                target=args.target,
            )
        elif cmd == "path":
            return await service.path(
                start=args.start,
                end=args.end,
                max_depth=args.max_depth,
                direction=args.direction,
            )
        else:
            return json.dumps({"error": f"Unknown command: {cmd}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    main()
