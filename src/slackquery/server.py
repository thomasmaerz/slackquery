"""Official MCP FastMCP Streamable HTTP server."""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from slackquery.embedding import LocalEmbeddingClient
from slackquery.models import SearchFilters, SearchMode
from slackquery.retrieval import SearchEngine
from slackquery.settings import Settings

GUIDE = """Use search_slack in hybrid mode by default. Narrow workspace, channel,
author, and time filters before raising limits. Use lexical mode for exact IDs,
URLs, filenames, quotes, and errors; semantic mode for paraphrases. Cite workspace,
channel, author, timestamp, and document_id. Expand only promising results with
get_slack_thread. Cursors are opaque and artifact/query-bound. No result does not
prove absence because archive coverage and filters may be incomplete."""


class Metrics:
    """Thread-safe, dependency-free Prometheus metrics collector."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.requests = 0
        self.errors = 0
        self.latency_seconds = 0.0
        self.routes: Counter[str] = Counter()
        self.modes: Counter[str] = Counter()
        self.route_errors: Counter[str] = Counter()
        self.latency_buckets: Counter[float] = Counter()
        self.bucket_bounds = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        self.lock = threading.Lock()

    def observe(
        self,
        elapsed: float,
        *,
        error: bool = False,
        route: str = "http",
        mode: str = "none",
    ) -> None:
        with self.lock:
            self.requests += 1
            self.errors += int(error)
            self.latency_seconds += elapsed
            self.routes[route] += 1
            self.modes[mode] += 1
            if error:
                self.route_errors[route] += 1
            for bound in self.bucket_bounds:
                if elapsed <= bound:
                    self.latency_buckets[bound] += 1

    def render(self) -> str:
        artifact_count = len(list(self.settings.artifact_dir.glob("search-*.duckdb")))
        document_count = 0
        build_id = "none"
        if self.settings.current_link.exists():
            try:
                import duckdb

                connection = duckdb.connect(
                    str(self.settings.current_link.resolve()), read_only=True
                )
                document_row = connection.execute(
                    "SELECT count(*) FROM search_documents"
                ).fetchone()
                assert document_row is not None
                document_count = document_row[0]
                build_row = connection.execute("SELECT build_id FROM artifact_metadata").fetchone()
                if build_row:
                    build_id = str(build_row[0])
                connection.close()
            except Exception:
                pass
        with self.lock:
            lines = [
                        "# TYPE slackquery_requests_total counter",
                        f"slackquery_requests_total {self.requests}",
                        "# TYPE slackquery_errors_total counter",
                        f"slackquery_errors_total {self.errors}",
                        "# TYPE slackquery_request_latency_seconds_sum counter",
                        f"slackquery_request_latency_seconds_sum {self.latency_seconds:.9f}",
                        f'slackquery_backend_info{{mode="{self.settings.embedding_backend}"}} 1',
                        f'slackquery_artifact_build_info{{build_id="{build_id}"}} 1',
                        "# TYPE slackquery_route_requests_total counter",
                        "# TYPE slackquery_mode_requests_total counter",
                        f"slackquery_artifacts {artifact_count}",
                        f"slackquery_documents {document_count}",
                    ]
            lines.extend(
                f'slackquery_route_requests_total{{route="{route}"}} {count}'
                for route, count in sorted(self.routes.items())
            )
            lines.extend(
                f'slackquery_mode_requests_total{{mode="{mode}"}} {count}'
                for mode, count in sorted(self.modes.items())
            )
            lines.extend(
                f'slackquery_route_errors_total{{route="{route}"}} {count}'
                for route, count in sorted(self.route_errors.items())
            )
            lines.extend(
                f'slackquery_request_latency_seconds_bucket{{le="{bound}"}} '
                f"{self.latency_buckets[bound]}"
                for bound in self.bucket_bounds
            )
            lines.append(f'slackquery_request_latency_seconds_bucket{{le="+Inf"}} {self.requests}')
            return "\n".join(lines) + "\n"


class MetricsMiddleware:
    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = 500
        path = str(scope.get("path", "unknown"))
        route = "mcp" if path.startswith("/mcp") else path.strip("/") or "root"

        async def wrapped(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, wrapped)
        except Exception:
            self.metrics.observe(time.perf_counter() - started, error=True, route=route)
            raise
        self.metrics.observe(time.perf_counter() - started, error=status >= 500, route=route)


class BearerAuthMiddleware:
    """Require a configured bearer token while leaving health probes public."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") not in {"/healthz", "/readyz", "/metrics"}:
            headers = dict(scope.get("headers", []))
            supplied = headers.get(b"authorization", b"").decode("latin-1")
            if not secrets.compare_digest(supplied, self.expected):
                response = JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_mcp(
    settings: Settings | None = None,
    *,
    embedding: LocalEmbeddingClient | None = None,
    engine: SearchEngine | None = None,
    metrics: Metrics | None = None,
) -> FastMCP[Any]:
    settings = settings or Settings()
    embedding = embedding or LocalEmbeddingClient(settings)
    engine = engine or SearchEngine(settings, embedding)
    mcp: FastMCP[Any] = FastMCP(
        "slackquery",
        instructions=GUIDE,
        host=settings.host,
        port=settings.port,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        max_request_body_size=1_048_576,
        log_level=settings.log_level.upper(),  # type: ignore[arg-type]
    )

    @mcp.tool()
    async def search_slack(
        query: str,
        mode: SearchMode = "hybrid",
        workspace_ids: list[str] | None = None,
        channel_ids: list[str] | None = None,
        author_ids: list[str] | None = None,
        document_kinds: list[str] | None = None,
        start_ts_us: int | None = None,
        end_ts_us: int | None = None,
        limit: int = 10,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Search archived Slack messages using lexical, semantic, or hybrid retrieval."""
        filters = SearchFilters(
            workspace_ids=workspace_ids or [],
            channel_ids=channel_ids or [],
            author_ids=author_ids or [],
            start_ts_us=start_ts_us,
            end_ts_us=end_ts_us,
            document_kinds=document_kinds or ["message", "file_chunk"],
        )
        response = await engine.search(
            query, mode=mode, filters=filters, limit=limit, cursor=cursor
        )
        if metrics is not None:
            with metrics.lock:
                metrics.modes[mode] += 1
        return response.model_dump(mode="json")

    @mcp.tool()
    async def get_slack_message(
        document_id: str, context_before: int = 0, context_after: int = 0
    ) -> list[dict[str, Any]]:
        """Fetch one exact Slack message with bounded same-channel context."""
        messages = await asyncio.to_thread(
            engine.get_message,
            document_id,
            before=context_before,
            after=context_after,
        )
        return [item.model_dump(mode="json") for item in messages]

    @mcp.tool()
    async def get_slack_thread(thread_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Fetch a promising thread in chronological order."""
        messages = await asyncio.to_thread(engine.get_thread, thread_id, limit=limit)
        return [item.model_dump(mode="json") for item in messages]

    @mcp.tool()
    async def list_slack_scopes(
        workspace_id: str | None = None,
        channel_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """List stable workspace/channel IDs and archive date coverage."""
        scopes = await asyncio.to_thread(
            engine.list_scopes, workspace_id, channel_name, limit=limit
        )
        return [item.model_dump(mode="json") for item in scopes]

    @mcp.resource("slackquery://guide")
    def guide() -> str:
        """Short portable usage contract."""
        return GUIDE

    return mcp


def create_asgi_app(settings: Settings | None = None) -> ASGIApp:
    settings = settings or Settings()
    embedding = LocalEmbeddingClient(settings)
    engine = SearchEngine(settings, embedding)
    metrics = Metrics(settings)
    mcp = create_mcp(settings, embedding=embedding, engine=engine, metrics=metrics)

    async def health(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "healthy": True,
                "embedding_backend": settings.embedding_backend,
                "embedding_base_url": settings.embedding_base_url,
            }
        )

    async def readiness(_: Request) -> JSONResponse:
        payload = await asyncio.to_thread(engine.readiness)
        return JSONResponse(payload, status_code=200 if payload["ready"] else 503)

    async def prometheus(_: Request) -> PlainTextResponse:
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            await embedding.close()

    app: ASGIApp = Starlette(
        routes=[
            Route("/healthz", health),
            Route("/readyz", readiness),
            Route("/metrics", prometheus),
            Mount("/", app=mcp.streamable_http_app()),
        ],
        lifespan=lifespan,
    )
    app = MetricsMiddleware(app, metrics)
    if settings.bearer_token is None:
        return app
    return BearerAuthMiddleware(app, settings.bearer_token.get_secret_value())


app = create_asgi_app()
