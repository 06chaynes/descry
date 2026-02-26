#!/usr/bin/env python3
"""
Descry Web UI Server

Browser-based interface for codebase knowledge graph tools.
Reuses the same underlying modules as the MCP server.

Usage:
    uv run descry-web [--port 8787] [--host 127.0.0.1]
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse, Response, StreamingResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

# Configure logging — check DESCRY_LOG_LEVEL first, fall back to CODEGRAPH_LOG_LEVEL
_log_level = os.environ.get("DESCRY_LOG_LEVEL") or os.environ.get("CODEGRAPH_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, _log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("descry_web")

SERVER_VERSION = "1.0.0"

# --- Import shared modules ---

from descry.query import GraphQuerier, _get_syntax_lang

try:
    from descry.cross_lang import CrossLangTracer
    CROSS_LANG_AVAILABLE = True
except ImportError:
    CROSS_LANG_AVAILABLE = False
    CrossLangTracer = None

try:
    from descry.embeddings import embeddings_available, SemanticSearcher, get_embeddings_status
    SEMANTIC_AVAILABLE = embeddings_available()
except ImportError:
    SEMANTIC_AVAILABLE = False
    SemanticSearcher = None

    def get_embeddings_status(*args, **kwargs):
        return {"available": False}

try:
    from descry.scip.support import scip_available, get_scip_status
except ImportError:
    def scip_available():
        return False

    def get_scip_status():
        return {"available": False}

# docs_search is not ported (project-specific)
DOCS_SEARCH_LOADED = False
DocsSearcher = None
DOC_COLLECTIONS = {}
DOCS_EMBEDDINGS_AVAILABLE = False

try:
    from descry.git_history import GitHistoryAnalyzer, GitError
    GIT_HISTORY_AVAILABLE = True
except ImportError:
    GIT_HISTORY_AVAILABLE = False
    GitHistoryAnalyzer = None
    GitError = Exception


# --- Project paths ---

from descry.handlers import DescryConfig, _DEFAULT_PROJECT_MARKERS


def _find_project_root() -> Path:
    """Find project root using the same markers as DescryConfig."""
    markers = _DEFAULT_PROJECT_MARKERS
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd(), script_dir, script_dir.parent, script_dir.parent.parent]
    for start in candidates:
        for path in [start] + list(start.parents):
            for marker in markers:
                if (path / marker).exists():
                    return path
            if path == Path.home():
                break
    return script_dir.parent.parent


PROJECT_ROOT = _find_project_root()
_cache_dir_env = os.environ.get("DESCRY_CACHE_DIR") or os.environ.get("CODEGRAPH_CACHE_DIR")
CACHE_DIR = Path(_cache_dir_env) if _cache_dir_env else PROJECT_ROOT / ".descry_cache"
GRAPH_PATH = CACHE_DIR / "codebase_graph.json"
MAX_STALE_HOURS = 24
WEB_DIR = Path(__file__).parent / "web"


# --- Helper functions (pure functions) ---

def is_natural_language_query(terms: list[str]) -> bool:
    text = " ".join(terms).lower()
    nl_indicators = [
        "how to", "what is", "where is", "where are", "find the",
        "show me", "get the", "look for", "search for", "related to",
        "that handles", "that does", "responsible for", "used for", "deals with",
    ]
    if any(p in text for p in nl_indicators):
        return True
    if terms and terms[0].lower() in ("how", "what", "where", "why", "which", "find"):
        return True
    code_patterns = [r"[a-z]+_[a-z]+", r"[a-z]+[A-Z][a-z]+", r"[A-Z][a-z]+[A-Z]", r"::"]
    for pattern in code_patterns:
        if re.search(pattern, text):
            return False
    return len(terms) >= 3


def reciprocal_rank_fusion(tfidf_results: list, semantic_results: list, k: int = 60) -> list:
    rrf_scores = defaultdict(float)
    node_lookup = {}
    for rank, node in enumerate(tfidf_results):
        node_id = node["id"]
        rrf_scores[node_id] += 1.0 / (k + rank + 1)
        node_lookup[node_id] = node
    for rank, (node, _) in enumerate(semantic_results):
        node_id = node["id"]
        rrf_scores[node_id] += 1.0 / (k + rank + 1)
        node_lookup[node_id] = node
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [(node_lookup[nid], rrf_scores[nid]) for nid in sorted_ids]


# --- Cached instances ---

_querier_cache = {"mtime": 0, "instance": None}
_semantic_cache = {"mtime": 0, "instance": None}
_docs_cache: dict = {"instance": None}
_git_cache: dict = {"analyzer": None, "graph_mtime": None}
_graph_meta = {"mtime": 0, "nodes": 0, "edges": 0}


def _get_querier() -> GraphQuerier | None:
    global _querier_cache
    if not GRAPH_PATH.exists():
        _querier_cache = {"mtime": 0, "instance": None}
        return None
    mtime = GRAPH_PATH.stat().st_mtime
    if mtime != _querier_cache["mtime"]:
        _querier_cache = {"mtime": mtime, "instance": GraphQuerier(str(GRAPH_PATH))}
    return _querier_cache["instance"]


def _get_semantic_searcher() -> "SemanticSearcher | None":
    global _semantic_cache
    if not SEMANTIC_AVAILABLE or not GRAPH_PATH.exists():
        return None
    mtime = GRAPH_PATH.stat().st_mtime
    if mtime != _semantic_cache["mtime"] or _semantic_cache["instance"] is None:
        _semantic_cache = {"mtime": mtime, "instance": SemanticSearcher(str(GRAPH_PATH))}
    return _semantic_cache["instance"]


def _get_git_analyzer() -> "GitHistoryAnalyzer | None":
    global _git_cache
    if not GIT_HISTORY_AVAILABLE:
        return None
    current_mtime = GRAPH_PATH.stat().st_mtime if GRAPH_PATH.exists() else None
    if _git_cache["graph_mtime"] != current_mtime or _git_cache["analyzer"] is None:
        q = _get_querier()
        _git_cache["analyzer"] = GitHistoryAnalyzer(str(PROJECT_ROOT), graph_querier=q)
        _git_cache["graph_mtime"] = current_mtime
    return _git_cache["analyzer"]


def _get_docs_searcher() -> "DocsSearcher | None":
    global _docs_cache
    if not DOCS_SEARCH_LOADED or not DOCS_EMBEDDINGS_AVAILABLE:
        return None
    if _docs_cache["instance"] is None:
        searcher = DocsSearcher(PROJECT_ROOT, CACHE_DIR / "docs")
        searcher.index()
        _docs_cache["instance"] = searcher
    return _docs_cache["instance"]


def _update_graph_meta():
    global _graph_meta
    if GRAPH_PATH.exists():
        mtime = GRAPH_PATH.stat().st_mtime
        if mtime != _graph_meta["mtime"]:
            try:
                with open(GRAPH_PATH) as f:
                    data = json.load(f)
                _graph_meta = {
                    "mtime": mtime,
                    "nodes": len(data.get("nodes", [])),
                    "edges": len(data.get("edges", [])),
                }
            except (json.JSONDecodeError, KeyError):
                pass


def _graph_status() -> dict:
    if not GRAPH_PATH.exists():
        return {"exists": False, "age_str": "N/A", "age_hours": None, "nodes": 0, "edges": 0}
    mtime = GRAPH_PATH.stat().st_mtime
    age_hours = (time.time() - mtime) / 3600
    _update_graph_meta()
    return {
        "exists": True,
        "age_str": f"{age_hours:.1f}h ago",
        "age_hours": round(age_hours, 2),
        "nodes": _graph_meta["nodes"],
        "edges": _graph_meta["edges"],
        "stale": age_hours > MAX_STALE_HOURS,
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

async def api_health(request: Request) -> JSONResponse:
    status = _graph_status()
    return JSONResponse({
        "status": "ok" if status["exists"] and not status.get("stale") else ("stale" if status.get("stale") else "no_graph"),
        "version": SERVER_VERSION,
        "project_root": str(PROJECT_ROOT),
        "graph": status,
        "features": {
            "scip": scip_available(),
            "embeddings": SEMANTIC_AVAILABLE,
            "git_history": GIT_HISTORY_AVAILABLE,
            "cross_lang": CROSS_LANG_AVAILABLE,
            "docs_search": DOCS_SEARCH_LOADED and DOCS_EMBEDDINGS_AVAILABLE,
        },
    })


async def api_status(request: Request) -> JSONResponse:
    status = _graph_status()
    return JSONResponse(status)


async def api_ensure(request: Request) -> JSONResponse:
    body = await request.json() if request.method == "POST" else {}
    max_age = body.get("max_age_hours", 24)
    status = _graph_status()

    if not status["exists"]:
        result = await _run_index(".")
        return JSONResponse({"action": "generated", "result": result, "graph": _graph_status()})

    if status["age_hours"] and status["age_hours"] > max_age:
        result = await _run_index(".")
        return JSONResponse({"action": "refreshed", "result": result, "graph": _graph_status()})

    return JSONResponse({"action": "ready", "graph": status})


async def _run_index(path: str) -> str:
    import subprocess
    script_path = Path(__file__).parent.parent / "generate.py"
    index_path = str(PROJECT_ROOT) if path == "." else path
    try:
        result = subprocess.run(
            ["uv", "run", str(script_path), index_path],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=600,
        )
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Timed out after 10 minutes"
    except Exception as e:
        return f"Error: {e}"


def _reset_caches():
    """Reset all caches after reindexing."""
    global _querier_cache, _semantic_cache, _git_cache
    _querier_cache = {"mtime": 0, "instance": None}
    _semantic_cache = {"mtime": 0, "instance": None}
    _git_cache = {"analyzer": None, "graph_mtime": None}


async def api_index(request: Request) -> JSONResponse:
    body = await request.json() if request.method == "POST" else {}
    path = body.get("path", ".")
    result = await _run_index(path)
    _reset_caches()
    return JSONResponse({"result": result, "graph": _graph_status()})


async def api_index_stream(request: Request) -> StreamingResponse:
    """Streaming reindex endpoint using Server-Sent Events for live output."""
    script_path = Path(__file__).parent.parent / "generate.py"

    async def event_stream():
        yield f"data: {json.dumps({'type': 'start', 'message': 'Starting reindex...'})}\n\n"

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", str(script_path), str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            limit=10 * 1024 * 1024,  # 10MB line buffer (indexer can emit large lines)
        )

        try:
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=660)
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    yield f"data: {json.dumps({'type': 'output', 'line': text})}\n\n"
        except asyncio.TimeoutError:
            proc.kill()
            yield f"data: {json.dumps({'type': 'error', 'message': 'Timed out after 11 minutes'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'success': False})}\n\n"
            return

        await proc.wait()
        success = proc.returncode == 0

        if success:
            _reset_caches()
            graph = _graph_status()
            yield f"data: {json.dumps({'type': 'done', 'success': True, 'graph': graph})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'done', 'success': False, 'message': f'Process exited with code {proc.returncode}'})}\n\n"

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

    limit = int(request.query_params.get("limit", "10"))
    lang = request.query_params.get("lang")
    crate = request.query_params.get("crate")
    symbol_type = request.query_params.get("type")
    exclude_tests = request.query_params.get("exclude_tests", "").lower() == "true"

    q = _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found. Run ensure first."}, status_code=503)

    # TF-IDF search
    tfidf_results = q.search_docs(
        terms, lang=lang if lang != "all" else None,
        crate=crate, symbol_type=symbol_type if symbol_type != "all" else None,
        exclude_tests=exclude_tests,
    )[:limit * 2]

    # Semantic search if available
    semantic_results = []
    search_method = "keyword"
    if SEMANTIC_AVAILABLE and GRAPH_PATH.exists():
        if is_natural_language_query(terms) or len(tfidf_results) < 3:
            try:
                searcher = _get_semantic_searcher()
                if searcher:
                    query = " ".join(terms)
                    semantic_results = searcher.search(query, limit=limit * 2, min_score=0.25)
                    search_method = "hybrid"
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

    # Combine
    if semantic_results and tfidf_results:
        combined = reciprocal_rank_fusion(tfidf_results, semantic_results)
        results = [node for node, _ in combined[:limit]]
        search_method = "hybrid"
    elif tfidf_results:
        results = tfidf_results[:limit]
    else:
        results = []

    return JSONResponse({
        "results": [_node_to_dict(n) for n in results],
        "method": search_method,
        "query": " ".join(terms),
        "total": len(results),
    })


async def api_semantic(request: Request) -> JSONResponse:
    query = request.query_params.get("query", "")
    if not query:
        return JSONResponse({"error": "Missing 'query' parameter"}, status_code=400)
    limit = int(request.query_params.get("limit", "10"))

    if not SEMANTIC_AVAILABLE:
        return JSONResponse({"error": "Semantic search not available"}, status_code=503)

    searcher = _get_semantic_searcher()
    if not searcher:
        return JSONResponse({"error": "Graph not found"}, status_code=503)

    try:
        results = searcher.search(query, limit=limit)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({
        "results": [
            {**_node_to_dict(node), "score": round(score, 4)}
            for node, score in results
        ],
        "query": query,
        "total": len(results),
    })


async def api_callers(request: Request) -> JSONResponse:
    name = request.query_params.get("name", "")
    if not name:
        return JSONResponse({"error": "Missing 'name' parameter"}, status_code=400)
    limit = int(request.query_params.get("limit", "20"))

    q = _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found"}, status_code=503)

    all_callers = q.get_callers(name)
    fuzzy = False
    if not all_callers:
        all_callers = q.get_callers(name, fuzzy=True)
        fuzzy = bool(all_callers)

    callers = sorted(all_callers)[:limit]
    return JSONResponse({
        "symbol": name,
        "fuzzy": fuzzy,
        "total": len(all_callers),
        "callers": [_caller_to_dict(c, q) for c in callers],
    })


async def api_callees(request: Request) -> JSONResponse:
    name = request.query_params.get("name", "")
    if not name:
        return JSONResponse({"error": "Missing 'name' parameter"}, status_code=400)
    limit = int(request.query_params.get("limit", "20"))

    q = _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found"}, status_code=503)

    matches = q.find_nodes_by_name(name)
    func_matches = [m for m in matches if m["type"] in ("Function", "Method")]
    fuzzy = False
    if not func_matches:
        matches = q.find_nodes_by_name(name, fuzzy=True)
        func_matches = [m for m in matches if m["type"] in ("Function", "Method")]
        fuzzy = bool(func_matches)

    if not func_matches:
        return JSONResponse({"symbol": name, "fuzzy": False, "total": 0, "callees": [], "error": "Symbol not found"})

    node = func_matches[0]
    callees = sorted(q.get_callees(node["id"]))[:limit]

    return JSONResponse({
        "symbol": node.get("metadata", {}).get("name", name),
        "node_id": node["id"],
        "fuzzy": fuzzy,
        "total": len(callees),
        "callees": [_caller_to_dict(c, q) for c in callees],
    })


async def api_context(request: Request) -> JSONResponse:
    node_id = request.query_params.get("node_id", "")
    if not node_id:
        return JSONResponse({"error": "Missing 'node_id' parameter"}, status_code=400)

    full = request.query_params.get("full", "").lower() == "true"
    brief = request.query_params.get("brief", "").lower() == "true"
    expand_callees = request.query_params.get("expand_callees", "").lower() == "true"
    depth = int(request.query_params.get("depth", "1"))
    max_tokens = int(request.query_params.get("max_tokens", "2000"))

    q = _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found"}, status_code=503)

    result = q.get_context_prompt(
        node_id, depth=depth, max_tokens=max_tokens,
        full=full, brief=brief, expand_callees=expand_callees,
    )

    return JSONResponse({"node_id": node_id, "markdown": result})


async def api_structure(request: Request) -> JSONResponse:
    filename = request.query_params.get("filename", "")
    if not filename:
        return JSONResponse({"error": "Missing 'filename' parameter"}, status_code=400)

    q = _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found"}, status_code=503)

    matches = q.find_nodes_by_name(filename)
    file_matches = [m for m in matches if m["type"] == "File"]

    if not file_matches:
        return JSONResponse({"error": f"File '{filename}' not found"}, status_code=404)

    node_id = file_matches[0]["id"]
    defs = []
    imports = set()
    for edge in q.outgoing[node_id]:
        if edge["relation"] == "DEFINES":
            target = q.nodes.get(edge["target"])
            if target:
                defs.append(target)
        elif edge["relation"] == "IMPORTS":
            target = edge["target"]
            imports.add(target.replace("MODULE:", "") if target.startswith("MODULE:") else target)

    grouped = {}
    for type_name in ["Constant", "Class", "Function", "Configuration"]:
        items = [
            {"name": d["metadata"]["name"], "lineno": d["metadata"].get("lineno"),
             "signature": d["metadata"].get("signature", ""), "id": d.get("id", "")}
            for d in defs if d["type"] == type_name
        ]
        if items:
            grouped[type_name.lower()] = sorted(items, key=lambda x: x.get("lineno") or 0)

    return JSONResponse({
        "file": node_id,
        "imports": sorted(imports),
        "definitions": grouped,
    })


async def api_flatten(request: Request) -> JSONResponse:
    class_node_id = request.query_params.get("class_node_id", "")
    if not class_node_id:
        return JSONResponse({"error": "Missing 'class_node_id' parameter"}, status_code=400)

    q = _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found"}, status_code=503)

    result = q.flatten_class(class_node_id)
    return JSONResponse({"class_node_id": class_node_id, "markdown": result})


async def api_impls(request: Request) -> JSONResponse:
    method = request.query_params.get("method", "")
    if not method:
        return JSONResponse({"error": "Missing 'method' parameter"}, status_code=400)
    trait_name = request.query_params.get("trait_name") or None

    q = _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found"}, status_code=503)

    results = q.find_trait_impls(method, trait_name)
    impls = []
    for node in results:
        meta = node.get("metadata", {})
        node_id = node.get("id", "")
        parts = node_id.split("::")
        struct_name = parts[-2] if len(parts) >= 2 else "?"
        file_path = node_id.split("::")[0].replace("FILE:", "") if node_id.startswith("FILE:") else ""
        impls.append({
            "id": node_id,
            "struct_name": struct_name,
            "trait_name": meta.get("trait_impl", "unknown"),
            "file_path": file_path,
            "lineno": meta.get("lineno"),
            "signature": meta.get("signature", ""),
        })

    return JSONResponse({"method": method, "trait_filter": trait_name, "total": len(impls), "implementations": impls})


async def api_flow(request: Request) -> JSONResponse:
    start = request.query_params.get("start", "")
    if not start:
        return JSONResponse({"error": "Missing 'start' parameter"}, status_code=400)

    direction = request.query_params.get("direction", "forward")
    depth = int(request.query_params.get("depth", "3"))
    target = request.query_params.get("target") or None
    inline_threshold = int(request.query_params.get("inline_threshold", "100"))
    fmt = request.query_params.get("format", "markdown")

    q = _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found"}, status_code=503)

    base = {"start": start, "direction": direction, "depth": depth}

    if fmt == "tree":
        tree = q.trace_flow_structured(
            start_name=start, direction=direction, depth=depth,
            target=target, inline_threshold=inline_threshold,
        )
        return JSONResponse({**base, "tree": tree})

    result = q.trace_flow(
        start_name=start, direction=direction, depth=depth,
        target=target, inline_threshold=inline_threshold,
    )
    return JSONResponse({**base, "markdown": result})


async def api_path(request: Request) -> JSONResponse:
    start = request.query_params.get("start", "")
    end = request.query_params.get("end", "")
    if not start or not end:
        return JSONResponse({"error": "Missing 'start' and/or 'end' parameter"}, status_code=400)

    max_depth = int(request.query_params.get("max_depth", "10"))
    direction = request.query_params.get("direction", "forward")

    q = _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found"}, status_code=503)

    path = q.find_call_path(start, end, max_depth=max_depth, direction=direction)
    if not path:
        return JSONResponse({"start": start, "end": end, "hops": 0, "path": [], "markdown": f"No path found from '{start}' to '{end}'"})

    # Build structured path + markdown
    hops = []
    md_lines = [f"### Call Path: `{start}` -> `{end}` ({len(path)} hop{'s' if len(path) != 1 else ''})\n"]
    for i, hop in enumerate(path, 1):
        caller_name = hop.get("caller_name", "?")
        callee_name = hop.get("callee_name", "?")
        file_path = hop.get("file_path", "")
        call_line = hop.get("call_line")
        snippet = hop.get("call_snippet", "")
        hops.append({
            "step": i, "caller": caller_name, "callee": callee_name,
            "file_path": file_path, "line": call_line, "snippet": snippet,
        })
        md_lines.append(f"**{i}. {caller_name}** -> **{callee_name}**")
        if file_path and call_line:
            md_lines.append(f"   {file_path}:{call_line}")
        if snippet:
            lang = _get_syntax_lang(file_path) if file_path else ""
            md_lines.append(f"```{lang}")
            md_lines.append(snippet)
            md_lines.append("```")
        md_lines.append("")

    return JSONResponse({"start": start, "end": end, "hops": len(path), "path": hops, "markdown": "\n".join(md_lines)})


async def api_cross_lang(request: Request) -> JSONResponse:
    if not CROSS_LANG_AVAILABLE:
        return JSONResponse({"error": "Cross-language tracing not available"}, status_code=503)

    mode = request.query_params.get("mode", "endpoint")
    method = request.query_params.get("method")
    path = request.query_params.get("path")
    tag = request.query_params.get("tag")

    openapi_path = PROJECT_ROOT / "public" / "api" / "latest.json"
    if not openapi_path.exists():
        openapi_path = PROJECT_ROOT / "public" / "api" / "openapi.json"
    if not openapi_path.exists():
        return JSONResponse({"error": "OpenAPI spec not found"}, status_code=404)

    graph_path = str(GRAPH_PATH) if GRAPH_PATH.exists() else None
    tracer = CrossLangTracer(str(openapi_path), graph_path)

    if mode == "stats":
        return JSONResponse(tracer.get_stats())
    elif mode == "list":
        endpoints = tracer.list_endpoints(tag=tag)
        return JSONResponse({"tag": tag, "total": len(endpoints), "endpoints": endpoints})
    elif mode == "endpoint":
        if not method or not path:
            return JSONResponse({"error": "Endpoint mode requires 'method' and 'path'"}, status_code=400)
        info = tracer.get_handler_info(method.upper(), path)
        if not info:
            return JSONResponse({"error": f"No handler for {method.upper()} {path}"}, status_code=404)
        return JSONResponse(info)
    else:
        return JSONResponse({"error": f"Unknown mode '{mode}'"}, status_code=400)


async def api_churn(request: Request) -> JSONResponse:
    if not GIT_HISTORY_AVAILABLE:
        return JSONResponse({"error": "Git history not available"}, status_code=503)

    time_range = request.query_params.get("time_range")
    path_filter = request.query_params.get("path_filter")
    limit = int(request.query_params.get("limit", "20"))
    mode = request.query_params.get("mode", "symbols")
    exclude_generated = request.query_params.get("exclude_generated", "true").lower() == "true"

    fmt = request.query_params.get("format", "markdown")

    try:
        analyzer = _get_git_analyzer()
        if fmt == "structured":
            result = await asyncio.to_thread(
                analyzer.get_churn_structured, time_range=time_range,
                path_filter=path_filter, limit=limit, mode=mode,
                exclude_generated=exclude_generated,
            )
            return JSONResponse({"data": result})
        else:
            result = await asyncio.to_thread(
                analyzer.get_churn, time_range=time_range, path_filter=path_filter,
                limit=limit, mode=mode, exclude_generated=exclude_generated,
            )
            return JSONResponse({"mode": mode, "markdown": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_evolution(request: Request) -> JSONResponse:
    name = request.query_params.get("name", "")
    if not name:
        return JSONResponse({"error": "Missing 'name' parameter"}, status_code=400)

    if not GIT_HISTORY_AVAILABLE:
        return JSONResponse({"error": "Git history not available"}, status_code=503)

    time_range = request.query_params.get("time_range")
    limit = int(request.query_params.get("limit", "10"))
    show_diff = request.query_params.get("show_diff", "").lower() == "true"
    crate = request.query_params.get("crate")

    try:
        analyzer = _get_git_analyzer()
        result = await asyncio.to_thread(
            analyzer.get_evolution, name=name, time_range=time_range,
            limit=limit, show_diff=show_diff, crate=crate,
        )
        return JSONResponse({"name": name, "markdown": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_changes(request: Request) -> JSONResponse:
    if not GIT_HISTORY_AVAILABLE:
        return JSONResponse({"error": "Git history not available"}, status_code=503)

    commit_range = request.query_params.get("commit_range")
    time_range = request.query_params.get("time_range")
    path_filter = request.query_params.get("path_filter")
    show_callers = request.query_params.get("show_callers", "true").lower() == "true"
    limit = int(request.query_params.get("limit", "50"))

    fmt = request.query_params.get("format", "markdown")

    try:
        analyzer = _get_git_analyzer()
        if fmt == "structured":
            result = await asyncio.to_thread(
                analyzer.get_changes_structured, commit_range=commit_range,
                time_range=time_range, path_filter=path_filter,
                show_callers=show_callers, limit=limit,
            )
            return JSONResponse({"data": result})
        else:
            result = await asyncio.to_thread(
                analyzer.get_changes, commit_range=commit_range, time_range=time_range,
                path_filter=path_filter, show_callers=show_callers, limit=limit,
            )
            return JSONResponse({"markdown": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_docs_search(request: Request) -> JSONResponse:
    query = request.query_params.get("query", "")
    if not query:
        return JSONResponse({"error": "Missing 'query' parameter"}, status_code=400)

    limit = int(request.query_params.get("limit", "5"))
    collection = request.query_params.get("collection")

    searcher = _get_docs_searcher()
    if not searcher:
        return JSONResponse({"error": "Docs search not available"}, status_code=503)

    try:
        results = searcher.search(query, limit=limit, collection=collection)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({
        "query": query,
        "total": len(results),
        "results": results,
    })


async def api_quick(request: Request) -> JSONResponse:
    name = request.query_params.get("name", "")
    if not name:
        return JSONResponse({"error": "Missing 'name' parameter"}, status_code=400)

    full = request.query_params.get("full", "").lower() == "true"
    brief = request.query_params.get("brief", "").lower() == "true"

    q = _get_querier()
    if not q:
        return JSONResponse({"error": "Graph not found"}, status_code=503)

    matches = q.find_nodes_by_name(name)

    def type_priority(node):
        t = node.get("type", "")
        if t in ("Function", "Method"):
            return 0
        if t == "Class":
            return 1
        return 2

    matches.sort(key=type_priority)
    if not matches:
        matches = q.find_nodes_by_name(name, fuzzy=True)
        matches.sort(key=type_priority)

    if not matches:
        return JSONResponse({"error": f"No symbol found for '{name}'"}, status_code=404)

    best = matches[0]
    node_id = best["id"]
    context = q.get_context_prompt(node_id, full=full, brief=brief)

    return JSONResponse({
        "symbol": _node_to_dict(best),
        "other_matches": len(matches) - 1,
        "markdown": context,
    })


async def api_source(request: Request) -> JSONResponse:
    file_param = request.query_params.get("file", "")
    if not file_param:
        return JSONResponse({"error": "Missing 'file' parameter"}, status_code=400)

    line = request.query_params.get("line")
    target_line = int(line) if line else None

    # Resolve the file path relative to project root
    file_path = PROJECT_ROOT / file_param
    if not file_path.exists():
        return JSONResponse({"error": f"File not found: {file_param}"}, status_code=404)

    try:
        content = file_path.read_text(errors="replace")
    except Exception as e:
        return JSONResponse({"error": f"Cannot read file: {e}"}, status_code=500)

    lang = _get_syntax_lang(str(file_path))
    lines = content.split("\n")

    # If a target line is specified, return a window around it
    context_lines = 50
    if target_line and target_line > 0:
        start = max(0, target_line - context_lines - 1)
        end = min(len(lines), target_line + context_lines)
        visible_lines = lines[start:end]
        return JSONResponse({
            "file": file_param,
            "language": lang,
            "total_lines": len(lines),
            "start_line": start + 1,
            "end_line": end,
            "target_line": target_line,
            "content": "\n".join(visible_lines),
        })

    # Return full file (capped at 2000 lines for safety)
    cap = 2000
    truncated = len(lines) > cap
    return JSONResponse({
        "file": file_param,
        "language": lang,
        "total_lines": len(lines),
        "start_line": 1,
        "end_line": min(len(lines), cap),
        "target_line": None,
        "truncated": truncated,
        "content": "\n".join(lines[:cap]),
    })


# --- App entrypoint ---

async def index_page(request: Request) -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


routes = [
    Route("/", index_page),
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
    Route("/api/docs/search", api_docs_search),
    Route("/api/source", api_source),
    Mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static"),
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
]

app = Starlette(routes=routes, middleware=middleware)


def main():
    parser = argparse.ArgumentParser(description="Descry Web UI")
    parser.add_argument("--port", type=int, default=8787, help="Port (default: 8787)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    logger.info(f"Starting Descry Web UI on http://{args.host}:{args.port}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Graph path: {GRAPH_PATH}")

    # GraphQuerier resolves source file paths relative to CWD, so we must
    # run from the project root (same as the MCP server does).
    os.chdir(PROJECT_ROOT)

    # Pre-warm the graph
    if GRAPH_PATH.exists():
        logger.info("Pre-warming graph...")
        _get_querier()
        _update_graph_meta()
        logger.info(f"Graph ready: {_graph_meta['nodes']:,}n / {_graph_meta['edges']:,}e")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
