"""
Descry Web UI Server

Browser-based interface for codebase knowledge graph tools.
Reuses the same underlying modules as the MCP server.

Usage:
    uv run descry-web [--port 8787] [--host 127.0.0.1]
"""

import argparse
import asyncio
import base64
import contextlib
import errno
import hashlib
import json
import logging
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

try:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.datastructures import Headers
    from starlette.middleware import Middleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse, StreamingResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles
except ModuleNotFoundError as e:
    # The descry-web console script imports this module before main() runs.
    sys.exit(
        f"descry-web needs the 'web' extra ({e.name} is not installed). "
        "Install it with: pip install 'descry-codegraph[web]'"
    )

from descry import __version__

# Configure logging
_log_level = os.environ.get("DESCRY_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, _log_level.upper(), logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("descry_web")

SERVER_VERSION = __version__

# --- Import shared modules ---

from descry.handlers import DescryConfig, DescryService
from descry.query import GraphQuerier, _get_syntax_lang

# --- Project config (lazy-loaded via DescryConfig) ---

WEB_DIR = Path(__file__).parent / "web"

_config: DescryConfig | None = None
_service: DescryService | None = None


def _get_config() -> DescryConfig:
    global _config
    if _config is None:
        _config = DescryConfig.from_env()
    return _config


def _get_service() -> DescryService:
    global _service
    if _service is None:
        _service = DescryService(_get_config())
    return _service


# --- Cached instances (delegate to DescryService) ---

_graph_meta = {"mtime": 0, "nodes": 0, "edges": 0}


async def _get_querier() -> GraphQuerier | None:
    """Get GraphQuerier via DescryService (lock-disciplined, async)."""
    return await _get_service()._get_querier()


def _int_param(request: Request, name: str, default: int) -> int:
    """Parse an integer query parameter. Non-integer values fall back to default.

    Web endpoints were using `int(request.query_params.get(name, "N"))` which
    raises ValueError on garbage input (e.g. ``?limit=foo``) and surfaced as
    a 500. This helper returns the default instead, so malformed user input
    degrades quietly rather than crashing the request.
    """
    raw = request.query_params.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bad_direction(request: Request) -> JSONResponse | None:
    """A 400 if `direction` is given and is neither "forward" nor "backward"."""
    direction = request.query_params.get("direction", "forward")
    if direction in ("forward", "backward"):
        return None
    return JSONResponse(
        {"error": "'direction' must be 'forward' or 'backward'"}, status_code=400
    )


async def _service_error(result: str) -> JSONResponse | None:
    """An error response if `result` is rendered text reporting `ERROR:`, else None."""
    if not result.startswith("ERROR:"):
        return None
    status = 404 if await _get_querier() else 503
    return JSONResponse(
        {"error": result.removeprefix("ERROR:").strip()}, status_code=status
    )


def _git_failure(data: dict) -> JSONResponse:
    """The error response for a failed git-history result from the service."""
    status = 503 if data["reason"] == "unavailable" else 500
    return JSONResponse({"error": data["error"]}, status_code=status)


def _update_graph_meta():
    global _graph_meta
    cfg = _get_config()
    if cfg.graph_path.exists():
        mtime = cfg.graph_path.stat().st_mtime
        if mtime != _graph_meta["mtime"]:
            # Best-effort graph-metadata refresh: a corrupt / schema-
            # mismatched graph shouldn't break status polling. The old
            # cached meta stays in place until the next successful load.
            try:
                from descry._graph import GraphSchemaError, load_graph_with_schema

                data = load_graph_with_schema(cfg.graph_path)
                _graph_meta = {
                    "mtime": mtime,
                    "nodes": len(data.get("nodes", [])),
                    "edges": len(data.get("edges", [])),
                }
            except (OSError, ValueError, GraphSchemaError):
                pass


def _graph_status() -> dict:
    cfg = _get_config()
    if not cfg.graph_path.exists():
        return {
            "exists": False,
            "age_str": "N/A",
            "age_hours": None,
            "nodes": 0,
            "edges": 0,
        }
    mtime = cfg.graph_path.stat().st_mtime
    age_hours = (time.time() - mtime) / 3600
    _update_graph_meta()
    return {
        "exists": True,
        "age_str": f"{age_hours:.1f}h ago",
        "age_hours": round(age_hours, 2),
        "nodes": _graph_meta["nodes"],
        "edges": _graph_meta["edges"],
        "stale": age_hours > cfg.max_stale_hours,
    }


def _node_to_dict(node: dict) -> dict:
    """Convert a graph node to a JSON-serializable dict for the frontend."""
    meta = node.get("metadata", {})
    node_id = node.get("id", "")
    file_path = ""
    if node_id.startswith("FILE:"):
        file_path = node_id.split("::")[0].replace("FILE:", "")
    lineno = meta.get("lineno")
    return {
        "id": node_id,
        "type": node.get("type", "?"),
        "name": meta.get("name", "unknown"),
        "parent_name": meta.get("parent_name", ""),
        "file_path": file_path,
        "lineno": lineno,
        "location": f"{file_path}:{lineno}" if file_path and lineno else file_path,
        "signature": meta.get("signature", ""),
        "docstring": meta.get("docstring", ""),
        "token_count": meta.get("token_count", 0),
        "in_degree": meta.get("in_degree", 0),
    }


def _caller_to_dict(caller_id: str, q: GraphQuerier) -> dict:
    """Convert a caller node ID to a dict with location info."""
    result = {"id": caller_id, "name": caller_id}
    file_path = ""
    if caller_id.startswith("FILE:"):
        parts = caller_id.split("::")
        file_path = parts[0].replace("FILE:", "")
        if len(parts) > 1:
            result["name"] = parts[-1]
    node_info = q.get_node_info(caller_id)
    if node_info:
        meta = node_info.get("metadata", {})
        lineno = meta.get("lineno")
        result["file_path"] = file_path
        result["lineno"] = lineno
        result["location"] = f"{file_path}:{lineno}" if lineno else file_path
        result["type"] = node_info.get("type", "?")
        result["signature"] = meta.get("signature", "")
    else:
        result["file_path"] = file_path
        result["location"] = file_path
    return result


# --- API Endpoint Handlers ---


async def api_health(_request: Request) -> JSONResponse:
    health = json.loads(await _get_service().health())
    features = health["features"]
    features["cross_lang"] = _get_service().cross_lang_error() is None
    return JSONResponse(
        {
            "status": health["status"],
            "version": SERVER_VERSION,
            "project_root": str(_get_config().project_root),
            "graph": _graph_status(),
            "features": features,
        }
    )


async def api_status(_request: Request) -> JSONResponse:
    status = _graph_status()
    return JSONResponse(status)


# /api/index, /api/index/stream and /api/ensure write the same cache
# directory, so only one of them indexes at a time; the others are refused
# rather than queued.
_index_lock = threading.Lock()
_INDEX_BUSY = "An index run is already in progress"


def _index_busy() -> JSONResponse:
    return JSONResponse({"error": _INDEX_BUSY}, status_code=409)


async def api_ensure(request: Request) -> JSONResponse:
    raw = await request.body()
    try:
        body = json.loads(raw) if raw.strip() else {}
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "Request body must be a JSON object"}, status_code=400
        )
    max_age = body.get("max_age_hours", _get_config().max_stale_hours)
    if isinstance(max_age, bool) or not isinstance(max_age, (int, float)):
        return JSONResponse(
            {"error": "'max_age_hours' must be a number"}, status_code=400
        )
    status = _graph_status()
    health = json.loads(await _get_service().health())

    if not status["exists"] or health["status"] == "invalid_graph":
        result = await _run_index()
        if result is None:
            return _index_busy()
        return JSONResponse(
            {"action": "generated", "result": result, "graph": _graph_status()}
        )

    if status["age_hours"] and status["age_hours"] > max_age:
        result = await _run_index()
        if result is None:
            return _index_busy()
        return JSONResponse(
            {"action": "refreshed", "result": result, "graph": _graph_status()}
        )

    return JSONResponse({"action": "ready", "graph": status})


async def _run_index() -> str | None:
    """Regenerate the graph via DescryService, which applies `[timeouts] index_minutes`.

    Returns None, without indexing, if another run is in progress.
    """
    if not _index_lock.acquire(blocking=False):
        return None
    try:
        return await _get_service().index()
    finally:
        _index_lock.release()


async def api_index(_request: Request) -> JSONResponse:
    # H.2: path body param removed; always index cfg.project_root.
    result = await _run_index()
    if result is None:
        return _index_busy()
    return JSONResponse({"result": result, "graph": _graph_status()})


async def api_index_stream(_request: Request) -> StreamingResponse:
    """Streaming reindex endpoint using Server-Sent Events for live output."""

    async def event_stream():
        if not _index_lock.acquire(blocking=False):
            yield f"data: {json.dumps({'type': 'done', 'success': False, 'message': _INDEX_BUSY})}\n\n"
            return
        try:
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting reindex...'})}\n\n"
            # Closing this generator (the client went away) closes the
            # service's stream, which kills the indexer.
            async with contextlib.aclosing(_get_service().index_stream()) as events:
                async for event in events:
                    if event["type"] == "done" and event["success"]:
                        event["graph"] = _graph_status()
                    yield f"data: {json.dumps(event)}\n\n"
        finally:
            _index_lock.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def api_search(request: Request) -> JSONResponse:
    terms_str = request.query_params.get("terms", "")
    terms = [t.strip() for t in terms_str.split(",") if t.strip()]
    if not terms:
        return JSONResponse({"error": "Missing 'terms' parameter"}, status_code=400)

    data = await _get_service().search_data(
        terms,
        limit=_int_param(request, "limit", 10),
        lang=request.query_params.get("lang"),
        crate=request.query_params.get("crate"),
        symbol_type=request.query_params.get("type"),
        exclude_tests=request.query_params.get("exclude_tests", "").lower() == "true",
    )
    if not data["ok"]:
        return JSONResponse({"error": data["error"]}, status_code=503)

    return JSONResponse(
        {
            "results": [_node_to_dict(n) for n in data["results"]],
            "method": data["method"],
            "query": " ".join(terms),
            "total": len(data["results"]),
        }
    )


async def api_semantic(request: Request) -> JSONResponse:
    query = request.query_params.get("query", "")
    if not query:
        return JSONResponse({"error": "Missing 'query' parameter"}, status_code=400)
    limit = _int_param(request, "limit", 10)

    data = await _get_service().semantic_data(query, limit=limit)
    if not data["ok"]:
        return JSONResponse({"error": data["error"]}, status_code=503)
    results = data["results"]

    return JSONResponse(
        {
            "results": [
                {**_node_to_dict(node), "score": round(score, 4)}
                for node, score in results
            ],
            "query": query,
            "total": len(results),
        }
    )


async def api_callers(request: Request) -> JSONResponse:
    name = request.query_params.get("name", "")
    if not name:
        return JSONResponse({"error": "Missing 'name' parameter"}, status_code=400)

    data = await _get_service().callers_data(
        name, limit=_int_param(request, "limit", 20)
    )
    if not data["ok"]:
        return JSONResponse({"error": data["error"]}, status_code=503)

    q = await _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found"}, status_code=503)
    return JSONResponse(
        {
            "symbol": data["symbol"],
            "fuzzy": data["fuzzy"],
            "total": data["total"],
            "callers": [_caller_to_dict(c, q) for c in data["callers"]],
        }
    )


async def api_callees(request: Request) -> JSONResponse:
    name = request.query_params.get("name", "")
    if not name:
        return JSONResponse({"error": "Missing 'name' parameter"}, status_code=400)

    data = await _get_service().callees_data(
        name, limit=_int_param(request, "limit", 20)
    )
    if not data["ok"]:
        return JSONResponse({"error": data["error"]}, status_code=503)
    if not data["found"]:
        return JSONResponse(
            {
                "symbol": name,
                "fuzzy": False,
                "total": 0,
                "callees": [],
                "error": "Symbol not found",
            }
        )

    q = await _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found"}, status_code=503)
    node = data["node"]
    return JSONResponse(
        {
            "symbol": node.get("metadata", {}).get("name", name),
            "node_id": node["id"],
            "fuzzy": data["fuzzy"],
            "total": data["total"],
            "callees": [_caller_to_dict(c, q) for c in data["callees"]],
        }
    )


async def api_context(request: Request) -> JSONResponse:
    node_id = request.query_params.get("node_id", "")
    if not node_id:
        return JSONResponse({"error": "Missing 'node_id' parameter"}, status_code=400)

    full = request.query_params.get("full", "").lower() == "true"
    brief = request.query_params.get("brief", "").lower() == "true"
    expand_callees = request.query_params.get("expand_callees", "").lower() == "true"
    depth = _int_param(request, "depth", 1)
    max_tokens = _int_param(request, "max_tokens", 2000)

    result = await _get_service().context(
        node_id,
        depth=depth,
        max_tokens=max_tokens,
        full=full,
        brief=brief,
        expand_callees=expand_callees,
    )

    return await _service_error(result) or JSONResponse(
        {"node_id": node_id, "markdown": result}
    )


async def api_structure(request: Request) -> JSONResponse:
    filename = request.query_params.get("filename", "")
    if not filename:
        return JSONResponse({"error": "Missing 'filename' parameter"}, status_code=400)

    data = await _get_service().structure_data(filename)
    if not data["ok"]:
        return JSONResponse({"error": data["error"]}, status_code=503)
    if not data["found"]:
        return JSONResponse({"error": f"File '{filename}' not found"}, status_code=404)

    grouped = {}
    for type_name in ["Constant", "Class", "Function", "Configuration"]:
        items = [
            {
                "name": d["metadata"].get("name", ""),
                "lineno": d["metadata"].get("lineno"),
                "signature": d["metadata"].get("signature", ""),
                "id": d.get("id", ""),
            }
            for d in data["definitions"]
            if d["type"] == type_name
        ]
        if items:
            grouped[type_name.lower()] = sorted(
                items, key=lambda x: x.get("lineno") or 0
            )

    return JSONResponse(
        {
            "file": data["file"],
            "imports": data["imports"],
            "definitions": grouped,
        }
    )


async def api_flatten(request: Request) -> JSONResponse:
    class_node_id = request.query_params.get("class_node_id", "")
    if not class_node_id:
        return JSONResponse(
            {"error": "Missing 'class_node_id' parameter"}, status_code=400
        )

    result = await _get_service().flatten(class_node_id)
    return await _service_error(result) or JSONResponse(
        {"class_node_id": class_node_id, "markdown": result}
    )


async def api_impls(request: Request) -> JSONResponse:
    method = request.query_params.get("method", "")
    if not method:
        return JSONResponse({"error": "Missing 'method' parameter"}, status_code=400)
    trait_name = request.query_params.get("trait_name") or None

    data = await _get_service().impls_data(method, trait_name)
    if not data["ok"]:
        return JSONResponse({"error": data["error"]}, status_code=503)

    impls = []
    for node in data["implementations"]:
        meta = node.get("metadata", {})
        node_id = node.get("id", "")
        parts = node_id.split("::")
        struct_name = parts[-2] if len(parts) >= 2 else "?"
        file_path = (
            node_id.split("::")[0].replace("FILE:", "")
            if node_id.startswith("FILE:")
            else ""
        )
        impls.append(
            {
                "id": node_id,
                "struct_name": struct_name,
                "trait_name": meta.get("trait_impl", "unknown"),
                "file_path": file_path,
                "lineno": meta.get("lineno"),
                "signature": meta.get("signature", ""),
            }
        )

    return JSONResponse(
        {
            "method": method,
            "trait_filter": trait_name,
            "total": len(impls),
            "implementations": impls,
        }
    )


async def api_flow(request: Request) -> JSONResponse:
    start = request.query_params.get("start", "")
    if not start:
        return JSONResponse({"error": "Missing 'start' parameter"}, status_code=400)

    if bad_direction := _bad_direction(request):
        return bad_direction
    direction = request.query_params.get("direction", "forward")
    depth = _int_param(request, "depth", 3)
    target = request.query_params.get("target") or None
    inline_threshold = _int_param(request, "inline_threshold", 100)
    structured = request.query_params.get("format", "markdown") == "tree"

    data = await _get_service().flow_data(
        start,
        direction=direction,
        depth=depth,
        target=target,
        inline_threshold=inline_threshold,
        structured=structured,
    )
    if not data["ok"]:
        return JSONResponse({"error": data["error"]}, status_code=503)

    return JSONResponse(
        {
            "start": start,
            "direction": direction,
            "depth": depth,
            "tree" if structured else "markdown": data["result"],
        }
    )


async def api_path(request: Request) -> JSONResponse:
    start = request.query_params.get("start", "")
    end = request.query_params.get("end", "")
    if not start or not end:
        return JSONResponse(
            {"error": "Missing 'start' and/or 'end' parameter"}, status_code=400
        )

    max_depth = _int_param(request, "max_depth", 10)
    if bad_direction := _bad_direction(request):
        return bad_direction
    direction = request.query_params.get("direction", "forward")

    data = await _get_service().path_data(
        start, end, max_depth=max_depth, direction=direction
    )
    if not data["ok"]:
        return JSONResponse({"error": data["error"]}, status_code=503)

    if data["status"] != "found":
        return JSONResponse(
            {
                "start": start,
                "end": end,
                "status": data["status"],
                "hops": 0,
                "path": [],
                "markdown": data["message"],
            }
        )

    path = data["path"]
    # Build structured path + markdown
    hops = []
    md_lines = [
        f"### Call Path: `{start}` -> `{end}` ({len(path)} hop{'s' if len(path) != 1 else ''})\n"
    ]
    for i, hop in enumerate(path, 1):
        caller_name = hop.get("caller_name", "?")
        callee_name = hop.get("callee_name", "?")
        file_path = hop.get("file_path", "")
        call_line = hop.get("call_line")
        snippet = hop.get("call_snippet", "")
        hops.append(
            {
                "step": i,
                "caller": caller_name,
                "callee": callee_name,
                "file_path": file_path,
                "line": call_line,
                "snippet": snippet,
            }
        )
        md_lines.append(f"**{i}. {caller_name}** -> **{callee_name}**")
        if file_path and call_line:
            md_lines.append(f"   {file_path}:{call_line}")
        if snippet:
            lang = _get_syntax_lang(file_path) if file_path else ""
            md_lines.append(f"```{lang}")
            md_lines.append(snippet)
            md_lines.append("```")
        md_lines.append("")

    return JSONResponse(
        {
            "start": start,
            "end": end,
            "status": data["status"],
            "hops": len(path),
            "path": hops,
            "markdown": "\n".join(md_lines),
        }
    )


async def api_cross_lang(request: Request) -> JSONResponse:
    mode = request.query_params.get("mode", "endpoint")
    method = request.query_params.get("method")
    path = request.query_params.get("path")
    tag = request.query_params.get("tag")

    data = await _get_service().cross_lang_data(mode, method=method, path=path, tag=tag)
    if not data["ok"]:
        if data["reason"] == "invalid":
            return JSONResponse({"error": data["error"]}, status_code=400)
        return JSONResponse({"not_configured": True, "message": data["error"]})

    result = data["result"]
    if mode == "list":
        return JSONResponse({"tag": tag, "total": len(result), "endpoints": result})
    if result is None:
        return JSONResponse(
            {"error": f"No handler for {method.upper()} {path}"}, status_code=404
        )
    return JSONResponse(result)


async def api_churn(request: Request) -> JSONResponse:
    time_range = request.query_params.get("time_range")
    path_filter = request.query_params.get("path_filter")
    limit = _int_param(request, "limit", 20)
    mode = request.query_params.get("mode", "symbols")
    exclude_generated = (
        request.query_params.get("exclude_generated", "true").lower() == "true"
    )
    structured = request.query_params.get("format", "markdown") == "structured"

    data = await _get_service().churn_data(
        time_range=time_range,
        path_filter=path_filter,
        limit=limit,
        mode=mode,
        exclude_generated=exclude_generated,
        structured=structured,
    )
    if not data["ok"]:
        return _git_failure(data)
    if structured:
        return JSONResponse({"data": data["result"]})
    return JSONResponse({"mode": mode, "markdown": data["result"]})


async def api_evolution(request: Request) -> JSONResponse:
    name = request.query_params.get("name", "")
    if not name:
        return JSONResponse({"error": "Missing 'name' parameter"}, status_code=400)

    time_range = request.query_params.get("time_range")
    limit = _int_param(request, "limit", 10)
    show_diff = request.query_params.get("show_diff", "").lower() == "true"
    crate = request.query_params.get("crate")

    data = await _get_service().evolution_data(
        name, time_range=time_range, limit=limit, show_diff=show_diff, crate=crate
    )
    if not data["ok"]:
        return _git_failure(data)
    return JSONResponse({"name": name, "markdown": data["result"]})


async def api_changes(request: Request) -> JSONResponse:
    commit_range = request.query_params.get("commit_range")
    time_range = request.query_params.get("time_range")
    path_filter = request.query_params.get("path_filter")
    show_callers = request.query_params.get("show_callers", "true").lower() == "true"
    limit = _int_param(request, "limit", 50)
    structured = request.query_params.get("format", "markdown") == "structured"

    data = await _get_service().changes_data(
        commit_range=commit_range,
        time_range=time_range,
        path_filter=path_filter,
        show_callers=show_callers,
        limit=limit,
        structured=structured,
    )
    if not data["ok"]:
        return _git_failure(data)
    if structured:
        return JSONResponse({"data": data["result"]})
    return JSONResponse({"markdown": data["result"]})


async def api_quick(request: Request) -> JSONResponse:
    name = request.query_params.get("name", "")
    if not name:
        return JSONResponse({"error": "Missing 'name' parameter"}, status_code=400)

    full = request.query_params.get("full", "").lower() == "true"
    brief = request.query_params.get("brief", "").lower() == "true"

    data = await _get_service().quick_data(name, full=full, brief=brief)
    if not data["ok"]:
        return JSONResponse({"error": data["error"]}, status_code=503)
    if not data["found"]:
        return JSONResponse({"error": f"No symbol found for '{name}'"}, status_code=404)

    return await _service_error(data["context"]) or JSONResponse(
        {
            "symbol": _node_to_dict(data["node"]),
            "other_matches": data["other_matches"],
            "markdown": data["context"],
        }
    )


_MAX_SOURCE_BYTES = 10 * 1024 * 1024  # 10 MiB

# Defense-in-depth: even with project-root containment, refuse to serve files
# that commonly hold credentials or private keys. Matches path segments case-
# insensitively. The web UI is local-only, but DNS-rebinding / cross-origin
# bugs in browsers have historically exposed localhost services; make the
# worst-case blast radius smaller than "every text file in the repo".
_SENSITIVE_NAME_RE = re.compile(
    r"(?ix)"
    r"(^|/)\.env(\.|/|$)"
    r"|(^|/)\.envrc$"
    r"|(^|/)\.git(/|$)"
    r"|(^|/)\.ssh/"
    r"|(^|/)\.aws/"
    r"|(^|/)\.gnupg/"
    r"|(^|/)\.netrc$"
    r"|(^|/)\.npmrc$"
    r"|(^|/)\.pypirc$"
    r"|(^|/)id_(rsa|dsa|ecdsa|ed25519)(\.|$)"
    r"|\.(pem|key|p12|pfx|jks|keystore|asc|gpg)$"
)


def _is_sensitive_path(rel: str) -> bool:
    """True if `rel` (project-root-relative POSIX path) matches a known secret pattern."""
    return bool(_SENSITIVE_NAME_RE.search(rel))


def _is_text_sample(sample: bytes) -> bool:
    """Heuristic check: first N bytes look like text (UTF-8 or UTF-16, low NUL ratio)."""
    if not sample:
        return True
    # Strip BOM if present.
    if sample.startswith(b"\xef\xbb\xbf"):
        sample = sample[3:]
    elif sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        # UTF-16 BOM; try decode before NUL check.
        try:
            sample.decode("utf-16")
            return True
        except UnicodeDecodeError:
            return False
    nul_count = sample.count(b"\x00")
    if nul_count > len(sample) * 0.01:  # > 1% NUL
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        try:
            sample.decode("utf-16")
            return True
        except UnicodeDecodeError:
            return False


def _open_beneath(root: Path, parts: tuple[str, ...]) -> int:
    """Open `root/parts...` read-only, following no symlink below `root`.

    `O_NOFOLLOW` on a single open only covers the last component, so each
    directory on the way down is opened relative to its parent's fd.
    `O_NONBLOCK` keeps the open from waiting on a FIFO.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if os.open not in os.supports_dir_fd:  # Windows has no openat()
        return os.open(root.joinpath(*parts), flags)

    dir_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd
            )
            os.close(dir_fd)
            dir_fd = next_fd
        return os.open(parts[-1], flags, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


async def api_source(request: Request) -> JSONResponse:
    file_param = request.query_params.get("file", "")
    if not file_param:
        return JSONResponse({"error": "Missing 'file' parameter"}, status_code=400)

    target_line = _int_param(request, "line", 0) or None

    # A.6 / H.1 pipeline: resolve → containment → S_ISREG → size → text-check → read
    cfg = _get_config()
    try:
        file_path = (cfg.project_root / file_param).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as e:
        return JSONResponse({"error": f"Invalid path: {e}"}, status_code=400)

    try:
        if not file_path.is_relative_to(cfg.resolved_project_root):
            return JSONResponse({"error": "Path outside project root"}, status_code=400)
    except ValueError:
        return JSONResponse({"error": "Path outside project root"}, status_code=400)

    # Defense-in-depth secret blocklist. Check both the requested path and the
    # resolved path so a file symlinked to a sensitive target is still refused.
    try:
        rel_requested = file_param.replace("\\", "/")
        rel_resolved = file_path.relative_to(cfg.resolved_project_root).as_posix()
    except ValueError:
        rel_resolved = rel_requested
    if _is_sensitive_path(rel_requested) or _is_sensitive_path(rel_resolved):
        return JSONResponse(
            {"error": "Refusing to serve sensitive file"}, status_code=403
        )

    if not file_path.exists():
        return JSONResponse({"error": f"File not found: {file_param}"}, status_code=404)

    try:
        st = file_path.stat()
    except OSError as e:
        return JSONResponse({"error": f"Cannot stat file: {e}"}, status_code=400)

    if not stat.S_ISREG(st.st_mode):
        return JSONResponse({"error": "Not a regular file"}, status_code=400)
    if st.st_size > _MAX_SOURCE_BYTES:
        return JSONResponse({"error": "File too large"}, status_code=413)

    # Open without following symlinks so a swap between the earlier
    # containment check and this read cannot redirect us outside the project,
    # then hold the opened file itself to the type and size limits.
    root = cfg.resolved_project_root
    try:
        fd = _open_beneath(root, file_path.relative_to(root).parts)
    except OSError as e:
        return JSONResponse({"error": f"Cannot open file: {e}"}, status_code=400)
    try:
        with os.fdopen(fd, "rb") as f:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return JSONResponse({"error": "Not a regular file"}, status_code=400)
            sample = f.read(8192)
            if not _is_text_sample(sample):
                return JSONResponse({"error": "Not a text file"}, status_code=415)
            f.seek(0)
            raw = f.read(_MAX_SOURCE_BYTES + 1)
    except OSError as e:
        return JSONResponse({"error": f"Cannot read file: {e}"}, status_code=500)
    if len(raw) > _MAX_SOURCE_BYTES:
        return JSONResponse({"error": "File too large"}, status_code=413)
    content = raw.decode("utf-8", errors="replace")

    lang = _get_syntax_lang(str(file_path))
    lines = content.split("\n")

    # If a target line is specified, return a window around it
    context_lines = 50
    if target_line and target_line > 0:
        start = max(0, target_line - context_lines - 1)
        end = min(len(lines), target_line + context_lines)
        visible_lines = lines[start:end]
        return JSONResponse(
            {
                "file": file_param,
                "language": lang,
                "total_lines": len(lines),
                "start_line": start + 1,
                "end_line": end,
                "target_line": target_line,
                "content": "\n".join(visible_lines),
            }
        )

    # Return full file (capped at 2000 lines for safety)
    cap = 2000
    truncated = len(lines) > cap
    return JSONResponse(
        {
            "file": file_param,
            "language": lang,
            "total_lines": len(lines),
            "start_line": 1,
            "end_line": min(len(lines), cap),
            "target_line": None,
            "truncated": truncated,
            "content": "\n".join(lines[:cap]),
        }
    )


async def api_examples(_request: Request) -> JSONResponse:
    """Return dynamic example symbols derived from the indexed graph.

    Picks interesting symbols (high in-degree, classes, good flow candidates)
    so the UI "Try:" links always work with the current project.
    """
    q = await _get_querier()
    if not q:
        return JSONResponse({"available": False})

    nodes = q.data.get("nodes", [])
    if not nodes:
        return JSONResponse({"available": False})

    edges = q.data.get("edges", [])

    # Categorize nodes
    functions = []
    classes = []
    methods = []
    for n in nodes:
        meta = n.get("metadata", {})
        ntype = n.get("type", "")
        name = meta.get("name", "")
        # Skip test nodes and private/dunder names
        if not name or name.startswith("_") or "test" in n.get("id", "").lower():
            continue
        entry = {
            "name": name,
            "id": n["id"],
            "type": ntype,
            "in_degree": meta.get("in_degree", 0),
            "token_count": meta.get("token_count", 0),
            "file": n["id"].replace("FILE:", "").split("::")[0]
            if "::" in n["id"]
            else "",
        }
        if ntype == "Function":
            functions.append(entry)
        elif ntype in ("Class", "Struct"):
            classes.append(entry)
        elif ntype == "Method":
            methods.append(entry)

    # Sort by in_degree (most-called first) for relevance
    functions.sort(key=lambda x: x["in_degree"], reverse=True)
    methods.sort(key=lambda x: x["in_degree"], reverse=True)

    # Build outgoing edge counts for flow examples
    out_degree = {}
    for e in edges:
        src = e.get("source", "")
        out_degree[src] = out_degree.get(src, 0) + 1

    # Pick "interesting" symbols for each tool
    # Search: a popular function, a class, and a method
    search_examples = []
    if functions:
        search_examples.append(functions[0]["name"])
    if classes:
        search_examples.append(classes[0]["name"])
    if methods and len(search_examples) < 3:
        search_examples.append(methods[0]["name"])

    # Quick/callers/callees: functions with high in-degree (many callers)
    popular = [f for f in functions if f["in_degree"] >= 2][:3]
    popular_names = [f["name"] for f in popular] or [f["name"] for f in functions[:3]]

    # Flow: find a function with good fanout (interesting forward trace)
    flow_candidates = []
    for f in functions[:20]:
        od = out_degree.get(f["id"], 0)
        if od >= 2:
            flow_candidates.append({"name": f["name"], "out_degree": od})
    flow_candidates.sort(key=lambda x: x["out_degree"], reverse=True)
    flow_fwd = (
        flow_candidates[0]["name"]
        if flow_candidates
        else (popular_names[0] if popular_names else "")
    )
    # Backward: pick a popular callee that differs from the forward symbol
    flow_bwd = ""
    for pn in popular_names:
        if pn != flow_fwd:
            flow_bwd = pn
            break
    if not flow_bwd:
        flow_bwd = popular_names[0] if popular_names else ""

    # Path: pick two symbols likely connected
    path_start = flow_fwd
    path_end = popular_names[0] if popular_names else ""
    if path_start == path_end and len(popular_names) > 1:
        path_end = popular_names[1]

    # Structure: pick files that have multiple symbols
    file_counts = {}
    for n in nodes:
        nid = n.get("id", "")
        if "::" in nid:
            f = nid.replace("FILE:", "").split("::")[0]
            file_counts[f] = file_counts.get(f, 0) + 1
    structure_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
    structure_examples = [f for f, _ in structure_files[:3]]

    # Flatten: pick a class
    flatten_example = classes[0]["id"] if classes else ""

    # Impls: pick method names that might have multiple implementations
    impl_candidates = []
    seen_names = {}
    for m in methods:
        n = m["name"]
        seen_names[n] = seen_names.get(n, 0) + 1
    impl_candidates = sorted(
        [(name, count) for name, count in seen_names.items() if count >= 2],
        key=lambda x: x[1],
        reverse=True,
    )
    impl_examples = [name for name, _ in impl_candidates[:3]]

    return JSONResponse(
        {
            "available": True,
            "search": search_examples,
            "quick": popular_names[:3],
            "callers": popular_names[:3],
            "callees": popular_names[:2],
            "flow_forward": flow_fwd,
            "flow_backward": flow_bwd,
            "path_start": path_start,
            "path_end": path_end,
            "structure": structure_examples,
            "flatten": flatten_example,
            "impls": impl_examples,
            "evolution": popular_names[:2],
        }
    )


# --- App entrypoint ---


def _content_security_policy(html: str) -> str:
    """Content-Security-Policy for the page whose source is `html`.

    Scripts load only from the two CDNs the page pins its assets on, and
    inline scripts run only if they hash to one of the page's own blocks.
    Alpine.js compiles directive expressions with the Function constructor,
    hence 'unsafe-eval'. Styles allow 'unsafe-inline' for the page's style
    attributes and the <style> element Mermaid puts in each diagram.
    """
    hashes = " ".join(
        f"'sha256-{base64.b64encode(hashlib.sha256(block.encode()).digest()).decode()}'"
        for block in re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    )
    return (
        "default-src 'none'; "
        "script-src https://cdn.jsdelivr.net https://cdnjs.cloudflare.com "
        f"'unsafe-eval' {hashes}; "
        "style-src 'self' https://cdnjs.cloudflare.com 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


# read_text() normalizes newlines to LF, as a browser does before hashing.
_CSP = _content_security_policy((WEB_DIR / "index.html").read_text(encoding="utf-8"))


async def index_page(_request: Request) -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


routes = [
    Route("/", index_page),
    Route("/api/examples", api_examples),
    Route("/api/health", api_health),
    Route("/api/status", api_status),
    Route("/api/ensure", api_ensure, methods=["POST"]),
    Route("/api/index", api_index, methods=["POST"]),
    Route("/api/index/stream", api_index_stream, methods=["POST"]),
    Route("/api/search", api_search),
    Route("/api/semantic", api_semantic),
    Route("/api/quick", api_quick),
    Route("/api/callers", api_callers),
    Route("/api/callees", api_callees),
    Route("/api/context", api_context),
    Route("/api/structure", api_structure),
    Route("/api/flatten", api_flatten),
    Route("/api/impls", api_impls),
    Route("/api/flow", api_flow),
    Route("/api/path", api_path),
    Route("/api/cross-lang", api_cross_lang),
    Route("/api/churn", api_churn),
    Route("/api/evolution", api_evolution),
    Route("/api/changes", api_changes),
    Route("/api/source", api_source),
    Mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static"),
]

# The UI is same-origin with its API — no CORS headers are needed. Omitting
# CORSMiddleware means browsers enforce same-origin by default, so a tab on
# evil.com cannot read /api/source or any other response. It can still send
# requests, so _SameOriginMiddleware refuses cross-origin writes such as
# POST /api/index. TrustedHostMiddleware rejects requests whose Host header
# isn't loopback, defeating DNS-rebinding attacks that would otherwise bypass
# the 127.0.0.1 bind.
_ALLOWED_HOSTS = ["127.0.0.1", "localhost"]


class _LoopbackHostMiddleware(TrustedHostMiddleware):
    """TrustedHostMiddleware that also accepts the IPv6 loopback literal.

    The base class takes the host to be whatever precedes the first colon, so
    `[::1]:8787` is read as `[` and no allowed-hosts entry can match it.
    """

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and re.fullmatch(
            r"\[::1\](:\d+)?", Headers(scope=scope).get("host", "")
        ):
            await self.app(scope, receive, send)
        else:
            await super().__call__(scope, receive, send)


class _SameOriginMiddleware:
    """Refuse state-changing requests that a browser sent from another origin.

    Browsers label them: `Sec-Fetch-Site` is set on every request and `Origin`
    on every cross-origin or non-GET one, and page script can forge neither.
    Clients that send neither header (curl, scripts) are let through.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] not in (
            "GET",
            "HEAD",
            "OPTIONS",
        ):
            headers = Headers(scope=scope)
            site = headers.get("sec-fetch-site", "same-origin")
            origin = headers.get("origin")
            if site not in ("same-origin", "none") or (
                origin is not None
                and urlsplit(origin).netloc != headers.get("host", "")
            ):
                response = JSONResponse(
                    {"error": "Cross-origin request refused"}, status_code=403
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class _ContentSecurityPolicyMiddleware:
    """Send the page's CSP on every response.

    The page is reachable at `/` and through the static mount; the policy is
    inert on the JSON responses.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def send_with_policy(message):
            if message["type"] == "http.response.start":
                message["headers"] = [
                    *message["headers"],
                    (b"content-security-policy", _CSP.encode()),
                ]
            await send(message)

        await self.app(scope, receive, send_with_policy)


middleware = [
    Middleware(_ContentSecurityPolicyMiddleware),
    Middleware(_LoopbackHostMiddleware, allowed_hosts=_ALLOWED_HOSTS),
    Middleware(_SameOriginMiddleware),
]

app = Starlette(routes=routes, middleware=middleware)


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _loopback_host(value: str) -> str:
    """argparse type: only accept loopback addresses for --host.

    The web UI has no authentication and is designed for single-user local
    use. Binding to a non-loopback address (e.g. `0.0.0.0`) would expose every
    endpoint — including /api/source — to the LAN. See SECURITY.md and the
    'Web UI is local-only' section of the README.
    """
    if value not in _LOOPBACK_HOSTS:
        raise argparse.ArgumentTypeError(
            f"--host must be a loopback address ({', '.join(sorted(_LOOPBACK_HOSTS))}); "
            f"got {value!r}. The web UI is unauthenticated by design; use a reverse "
            f"proxy if you need remote access."
        )
    return value


def main():
    parser = argparse.ArgumentParser(description="Descry Web UI")
    parser.add_argument("--port", type=int, default=8787, help="Port (default: 8787)")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        type=_loopback_host,
        help="Host (default: 127.0.0.1; only loopback addresses accepted)",
    )
    args = parser.parse_args()

    cfg = _get_config()
    logger.info(f"Starting Descry Web UI on http://{args.host}:{args.port}")
    logger.info(f"Project root: {cfg.project_root}")
    logger.info(f"Graph path: {cfg.graph_path}")

    # Paths in the graph and in requests are relative to the project root.
    os.chdir(cfg.project_root)

    # Pre-warm the graph via a one-off event loop (main() is sync; uvicorn
    # starts its own loop afterwards).
    if cfg.graph_path.exists():
        logger.info("Pre-warming graph...")

        async def _prewarm():
            await _get_querier()

        asyncio.run(_prewarm())
        _update_graph_meta()
        logger.info(
            f"Graph ready: {_graph_meta['nodes']:,}n / {_graph_meta['edges']:,}e"
        )

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            print(
                f"Port {args.port} already in use — pass --port N to override.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
