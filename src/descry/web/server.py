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

from descry._answers import fail

# Configure logging; uvicorn logs at the level asked of descry-web itself
# A name that is no level is WARNING, as the CLI reads it.
_level = logging.getLevelNamesMapping().get(
    os.environ.get("DESCRY_LOG_LEVEL", "INFO").upper(), logging.WARNING
)
logging.basicConfig(
    level=_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
_level_name = logging.getLevelName(_level).lower()
# uvicorn has no NOTSET; logging everything is its debug.
_level_name = "debug" if _level_name == "notset" else _level_name
logger = logging.getLogger("descry_web")

from descry.handlers import (
    DescryConfig,
    DescryService,
    Direction,
    public,
)
from descry.query import _get_syntax_lang

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


class _Malformed(ValueError):
    """A query parameter the request writes that its tool cannot take."""


def _int_param(request: Request, name: str, default: int | None) -> int | None:
    """The integer query parameter ``name``, ``default`` where it is not
    given; `_Malformed` where it is no integer."""
    raw = request.query_params.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise _Malformed(f"'{name}' must be an integer, got {raw!r}") from None


def _flag(request: Request, name: str, default: str = "false") -> bool:
    """The boolean query parameter ``name``: "true" or "false"; anything
    else is malformed."""
    raw = request.query_params.get(name, default).lower()
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise _Malformed(
        f"'{name}' must be 'true' or 'false', got {request.query_params.get(name)!r}"
    )


# The web status of each failure reason, as CONTRACT.md lists them
_STATUS = {
    "invalid": 400,
    "ambiguous": 400,
    "not_found": 404,
    "busy": 409,
    "no_graph": 503,
    "unavailable": 503,
    "loading": 503,
    "failed": 500,
}


def _failure(answer: dict) -> JSONResponse:
    """A failed answer as the web gives it: the answer, under its reason's
    status."""
    return JSONResponse(public(answer), status_code=_STATUS[answer["reason"]])


def _invalid(error: str) -> JSONResponse:
    """The failure of a request its tool cannot take."""
    return _failure(fail("invalid", error))


def _refused(reason: str, error: str, status: int | None = None) -> JSONResponse:
    """A request the server itself refuses, in the shape of every failure,
    under ``status`` where the reason's is not the HTTP one (413, 415, 403)."""
    return JSONResponse(
        public(fail(reason, error)), status_code=status or _STATUS[reason]
    )


def _answered(answer: dict, **page) -> JSONResponse:
    """A tool's answer as the web gives it, with what the page shows besides
    (`page`), or its failure."""
    if not answer["ok"]:
        return _failure(answer)
    return JSONResponse({**public(answer), **page})


def _bad_direction(request: Request) -> JSONResponse | None:
    """The failure for a `direction` that is neither "forward" nor "backward"."""
    if request.query_params.get("direction", "forward") in {d.value for d in Direction}:
        return None
    return _invalid("'direction' must be 'forward' or 'backward'")


def _page(markdown: str) -> str:
    """A tool's text without the graph header line it opens with: the page
    shows the graph's state itself."""
    head, sep, body = markdown.partition("\n\n")
    return body if sep and head.startswith("[") and head.endswith("]") else markdown


async def _cards(refs: list) -> list:
    """`_node_ref` entries with the signature and docstring the page shows
    on each result card."""
    q = await _get_service()._get_querier()
    shown = []
    for ref in refs:
        meta = ((q.nodes.get(ref["id"]) if q else None) or {}).get("metadata", {})
        shown.append(
            {
                **ref,
                "signature": meta.get("signature", ""),
                "docstring": meta.get("docstring", ""),
            }
        )
    return shown


async def api_health(_request: Request) -> JSONResponse:
    return _answered(await _get_service().health_answer())


async def api_status(_request: Request) -> JSONResponse:
    return _answered(await _get_service().status_answer())


async def api_ensure(request: Request) -> JSONResponse:
    raw = await request.body()
    try:
        body = json.loads(raw) if raw.strip() else {}
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return _invalid("Request body must be a JSON object")
    max_age = body.get("max_age_hours")
    if "max_age_hours" in body and (
        isinstance(max_age, bool) or not isinstance(max_age, (int, float))
    ):
        return _invalid("'max_age_hours' must be a number")
    answer = await _get_service().ensure_answer(max_age)
    report = answer.get("_report")
    return _answered(answer, **({"markdown": report} if report is not None else {}))


async def api_index(_request: Request) -> JSONResponse:
    # Always indexes cfg.project_root.
    answer = await _get_service().index_answer()
    return _answered(
        answer, **({"markdown": answer["_report"]} if answer["ok"] else {})
    )


async def api_index_stream(_request: Request) -> StreamingResponse:
    """Streaming reindex endpoint using Server-Sent Events for live output."""

    async def event_stream():
        # Closing this generator (the client went away) closes the
        # service's stream, which kills the indexer.
        async with contextlib.aclosing(_get_service().index_stream()) as events:
            async for event in events:
                if event["type"] == "done" and event["success"]:
                    event["graph"] = public(await _get_service().status_answer())
                yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def api_search(request: Request) -> JSONResponse:
    terms_str = request.query_params.get("terms", "")
    terms = [t.strip() for t in terms_str.split(",") if t.strip()]
    if not terms:
        return _invalid("Missing 'terms' parameter")
    answer = await _get_service().search_answer(
        terms,
        limit=_int_param(request, "limit", 10),
        lang=request.query_params.get("lang"),
        crate=request.query_params.get("crate"),
        symbol_type=request.query_params.get("type"),
        exclude_tests=_flag(request, "exclude_tests"),
    )
    if not answer["ok"]:
        return _failure(answer)
    return _answered(answer, results=await _cards(answer["results"]))


async def api_semantic(request: Request) -> JSONResponse:
    query = request.query_params.get("query", "")
    if not query:
        return _invalid("Missing 'query' parameter")
    answer = await _get_service().semantic_answer(
        query, limit=_int_param(request, "limit", 10)
    )
    if not answer["ok"]:
        return _failure(answer)
    return _answered(answer, results=await _cards(answer["results"]))


async def api_callers(request: Request) -> JSONResponse:
    name = request.query_params.get("symbol", "")
    if not name:
        return _invalid("Missing 'symbol' parameter")
    return _answered(
        await _get_service().callers_answer(
            name, limit=_int_param(request, "limit", 20)
        )
    )


async def api_callees(request: Request) -> JSONResponse:
    name = request.query_params.get("symbol", "")
    if not name:
        return _invalid("Missing 'symbol' parameter")
    return _answered(
        await _get_service().callees_answer(
            name, limit=_int_param(request, "limit", 20)
        )
    )


async def api_context(request: Request) -> JSONResponse:
    symbol = request.query_params.get("symbol", "")
    if not symbol:
        return _invalid("Missing 'symbol' parameter")
    svc = _get_service()
    answer = await svc.context_answer(
        symbol,
        brief=_flag(request, "brief"),
        full=_flag(request, "full"),
        expand_callees=_flag(request, "expand_callees"),
        deduplicate=_flag(request, "deduplicate"),
        depth=_int_param(request, "depth", 1),
        max_tokens=_int_param(request, "max_tokens", 2000),
        callee_budget=_int_param(request, "callee_budget", 2000),
        head_lines=_int_param(request, "head_lines", None),
        max_output_tokens=_int_param(request, "max_output_tokens", None),
    )
    return _answered(answer, markdown=await svc.context_text(answer))


async def api_structure(request: Request) -> JSONResponse:
    filename = request.query_params.get("filename", "")
    if not filename:
        return _invalid("Missing 'filename' parameter")
    return _answered(await _get_service().structure_answer(filename))


async def api_flatten(request: Request) -> JSONResponse:
    symbol = request.query_params.get("symbol", "")
    if not symbol:
        return _invalid("Missing 'symbol' parameter")
    svc = _get_service()
    answer = await svc.flatten_answer(symbol)
    return _answered(answer, markdown=await svc.flatten_text(answer))


async def api_impls(request: Request) -> JSONResponse:
    method = request.query_params.get("method", "")
    if not method:
        return _invalid("Missing 'method' parameter")
    trait_name = request.query_params.get("trait_name") or None
    return _answered(await _get_service().impls_answer(method, trait_name))


async def api_flow(request: Request) -> JSONResponse:
    start = request.query_params.get("start", "")
    if not start:
        return _invalid("Missing 'start' parameter")
    if bad_direction := _bad_direction(request):
        return bad_direction
    svc = _get_service()
    answer = await svc.flow_answer(
        start,
        direction=request.query_params.get("direction", "forward"),
        depth=_int_param(request, "depth", 3),
        target=request.query_params.get("target") or None,
        inline_threshold=_int_param(request, "inline_threshold", 100),
    )
    if not answer["ok"]:
        return _failure(answer)
    # The page shows the code the tree inlines
    if request.query_params.get("format", "markdown") == "tree":
        return _answered(answer, tree=answer["_flow"])
    return _answered(answer, markdown=_page(await svc.flow_text(answer)))


async def api_path(request: Request) -> JSONResponse:
    start = request.query_params.get("start", "")
    end = request.query_params.get("end", "")
    if not start or not end:
        return _invalid("Missing 'start' and/or 'end' parameter")
    if bad_direction := _bad_direction(request):
        return bad_direction
    svc = _get_service()
    answer = await svc.path_answer(
        start,
        end,
        max_depth=_int_param(request, "max_depth", 10),
        direction=request.query_params.get("direction", "forward"),
    )
    return _answered(answer, markdown=_page(await svc.path_text(answer, start, end)))


async def api_cross_lang(request: Request) -> JSONResponse:
    answer = await _get_service().cross_lang_answer(
        request.query_params.get("mode", "endpoint"),
        method=request.query_params.get("method"),
        path=request.query_params.get("path"),
        tag=request.query_params.get("tag"),
    )
    if not answer["ok"]:
        return _failure(answer)
    # The page shows the spec's summaries, by operation
    result = answer["_result"]
    found = result if isinstance(result, list) else [result] if result else []
    summaries = {e["operationId"]: e.get("summary", "") for e in found if "path" in e}
    return _answered(answer, summaries=summaries)


async def api_churn(request: Request) -> JSONResponse:
    svc = _get_service()
    answer = await svc.churn_answer(
        time_range=request.query_params.get("time_range"),
        path_filter=request.query_params.get("path_filter"),
        limit=_int_param(request, "limit", 20),
        mode=request.query_params.get("mode", "symbols"),
        exclude_generated=_flag(request, "exclude_generated", "true"),
    )
    return _answered(answer, markdown=_page(await svc.churn_text(answer)))


async def api_evolution(request: Request) -> JSONResponse:
    name = request.query_params.get("symbol", "")
    if not name:
        return _invalid("Missing 'symbol' parameter")
    svc = _get_service()
    answer = await svc.evolution_answer(
        name,
        time_range=request.query_params.get("time_range"),
        limit=_int_param(request, "limit", 10),
        show_diff=_flag(request, "show_diff"),
        crate=request.query_params.get("crate"),
    )
    return _answered(answer, markdown=_page(await svc.evolution_text(answer)))


async def api_changes(request: Request) -> JSONResponse:
    svc = _get_service()
    answer = await svc.changes_answer(
        commit_range=request.query_params.get("commit_range"),
        time_range=request.query_params.get("time_range"),
        path_filter=request.query_params.get("path_filter"),
        show_callers=_flag(request, "show_callers", "true"),
        limit=_int_param(request, "limit", 50),
    )
    return _answered(answer, markdown=_page(await svc.changes_text(answer)))


async def api_quick(request: Request) -> JSONResponse:
    name = request.query_params.get("symbol", "")
    if not name:
        return _invalid("Missing 'symbol' parameter")
    full, brief = _flag(request, "full"), _flag(request, "brief")
    svc = _get_service()
    answer = await svc.quick_answer(name, full, brief)
    return _answered(
        answer, markdown=await svc.quick_text(answer, name, full=full, brief=brief)
    )


def _open_beneath(root: Path, parts: tuple[str, ...]) -> int:
    """Open `root/parts...` read-only, following no symlink below `root`.

    `O_NOFOLLOW` on a single open only covers the last component, so each
    directory on the way down is opened relative to its parent's fd.
    `O_NONBLOCK` keeps the open from waiting on a FIFO.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if os.open not in os.supports_dir_fd:
        # Windows has no openat(): what was opened must be the file the path
        # resolves to afterwards, and that inside `root`.
        path = root.joinpath(*parts)
        fd = os.open(path, flags)
        try:
            real = path.resolve(strict=True)
            if not real.is_relative_to(root) or not os.path.samestat(
                os.fstat(fd), os.stat(real)
            ):
                raise OSError(errno.ELOOP, "the path changed while it was opened")
        except BaseException:
            os.close(fd)
            raise
        return fd

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
        return _refused("invalid", "Missing 'file' parameter")

    if "\0" in file_param:
        return _refused("invalid", "Invalid path: embedded null byte")
    target_line = _int_param(request, "line", 0) or None

    # Pipeline: resolve → containment → S_ISREG → size → text-check → read
    cfg = _get_config()
    try:
        file_path = (cfg.project_root / file_param).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as e:
        return _refused("invalid", f"Invalid path: {e}")

    if not file_path.is_relative_to(cfg.resolved_project_root):
        return _refused("invalid", "Path outside project root")

    if not file_path.exists():
        return _refused("not_found", f"File not found: {file_param}")

    try:
        st = file_path.stat()
    except OSError as e:
        return _refused("invalid", f"Cannot stat file: {e}")

    if not stat.S_ISREG(st.st_mode):
        return _refused("invalid", "Not a regular file")
    limit = cfg.max_file_bytes
    if limit and st.st_size > limit:
        return _refused("invalid", "File too large", 413)

    # Open without following symlinks so a swap between the earlier
    # containment check and this read cannot redirect us outside the project,
    # then hold the opened file itself to the type and size limits.
    root = cfg.resolved_project_root
    try:
        fd = _open_beneath(root, file_path.relative_to(root).parts)
    except OSError as e:
        return _refused("invalid", f"Cannot open file: {e}")
    try:
        with os.fdopen(fd, "rb") as f:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return _refused("invalid", "Not a regular file")
            sample = f.read(8192)
            if b"\0" in sample:
                return _refused("invalid", "Not a text file", 415)
            f.seek(0)
            raw = f.read(limit + 1 if limit else -1)
    except OSError as e:
        return _refused("failed", f"Cannot read file: {e}")
    if limit and len(raw) > limit:
        return _refused("invalid", "File too large", 413)
    content = raw.decode("utf-8", errors="replace")

    lang = _get_syntax_lang(str(file_path), cfg.syntax_lang_map)
    lines = re.split(r"\r\n?|\n", content)

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
    """Example symbols for the UI's "Try:" links, from the indexed graph."""
    data = await _get_service().examples_data()
    if not data["ok"] or data["examples"] is None:
        return JSONResponse({"available": False})
    return JSONResponse({"available": True, **data["examples"]})


def _content_security_policy(html: str) -> str:
    """Content-Security-Policy for the page whose source is `html`.

    Remote scripts and stylesheets load only from the URLs the page itself
    names, each of which it pins with an integrity hash, and inline scripts
    run only if they hash to one of the page's own blocks. Alpine.js compiles
    directive expressions with the Function constructor, hence 'unsafe-eval'.
    Styles allow 'unsafe-inline' for the page's style attributes and the
    <style> element Mermaid puts in each diagram.
    """
    hashes = " ".join(
        f"'sha256-{base64.b64encode(hashlib.sha256(block.encode()).digest()).decode()}'"
        for block in re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    )
    scripts = " ".join(re.findall(r'<script\b[^>]*\bsrc="(https://[^"]+)"', html))
    styles = " ".join(re.findall(r'<link\b[^>]*\bhref="(https://[^"]+)"', html))
    return (
        "default-src 'none'; "
        f"script-src {scripts} 'unsafe-eval' {hashes}; "
        f"style-src 'self' {styles} 'unsafe-inline'; "
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
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class _LoopbackHostMiddleware(TrustedHostMiddleware):
    """TrustedHostMiddleware that also accepts the IPv6 loopback literal, and
    refuses another host as every failure is answered.

    The base class takes the host to be whatever precedes the first colon, so
    `[::1]:8787` is read as `[` and no allowed-hosts entry can match it.
    """

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await super().__call__(scope, receive, send)
            return
        host = Headers(scope=scope).get("host", "")
        if re.fullmatch(r"\[::1\](:\d+)?", host) or (
            host.split(":")[0] in self.allowed_hosts
        ):
            await self.app(scope, receive, send)
        else:
            await _refused("invalid", "Invalid host header")(scope, receive, send)


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
            try:
                cross_origin = site not in ("same-origin", "none") or (
                    origin is not None
                    and urlsplit(origin).netloc != headers.get("host", "")
                )
            # An Origin that is not a URL is not this page's.
            except ValueError:
                cross_origin = True
            if cross_origin:
                response = _refused("invalid", "Cross-origin request refused", 403)
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
    Middleware(_LoopbackHostMiddleware, allowed_hosts=_LOOPBACK_HOSTS),
    Middleware(_SameOriginMiddleware),
]

app = Starlette(
    routes=routes,
    middleware=middleware,
    exception_handlers={_Malformed: lambda _request, exc: _invalid(str(exc))},
)


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
            await _get_service()._get_querier()
            return public(await _get_service().status_answer())

        status = asyncio.run(_prewarm())
        logger.info(f"Graph ready: {status['nodes']:,}n / {status['edges']:,}e")

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=_level_name)
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
