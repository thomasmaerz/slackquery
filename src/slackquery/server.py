"""Official MCP FastMCP Streamable HTTP server."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

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


class BearerAuthMiddleware:
    """Require a configured bearer token while leaving health probes public."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") not in {"/healthz", "/readyz"}:
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
        start_ts_us: int | None = None,
        end_ts_us: int | None = None,
        limit: int = 10,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Search archived Slack messages using lexical, semantic, or hybrid retrieval."""
        filters = SearchFilters(
            workspace_ids=workspace_ids or [], channel_ids=channel_ids or [],
            author_ids=author_ids or [], start_ts_us=start_ts_us, end_ts_us=end_ts_us,
        )
        response = await engine.search(
            query, mode=mode, filters=filters, limit=limit, cursor=cursor
        )
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
        return [
            item.model_dump(mode="json")
            for item in messages
        ]

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
        return [
            item.model_dump(mode="json")
            for item in scopes
        ]

    @mcp.resource("slackquery://guide")
    def guide() -> str:
        """Short portable usage contract."""
        return GUIDE

    return mcp


def create_asgi_app(settings: Settings | None = None) -> ASGIApp:
    settings = settings or Settings()
    embedding = LocalEmbeddingClient(settings)
    engine = SearchEngine(settings, embedding)
    mcp = create_mcp(settings, embedding=embedding, engine=engine)

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
            Mount("/", app=mcp.streamable_http_app()),
        ],
        lifespan=lifespan,
    )
    if settings.bearer_token is None:
        return app
    return BearerAuthMiddleware(app, settings.bearer_token.get_secret_value())


app = create_asgi_app()
