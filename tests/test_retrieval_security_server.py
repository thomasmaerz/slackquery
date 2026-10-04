from __future__ import annotations

import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient

from slackquery.models import SearchFilters
from slackquery.retrieval import _decode_cursor, _filter_sql, _fts_query
from slackquery.server import create_asgi_app, create_mcp
from slackquery.settings import Settings


def test_fts_query_drops_sql_punctuation() -> None:
    value = _fts_query("x'); DROP TABLE search_documents; -- ERR-42")
    assert "'" not in value and ";" not in value
    assert "ERR-42" in value


def test_filters_are_parameterized() -> None:
    sql, params = _filter_sql(
        SearchFilters(workspace_ids=["W1' OR true --"], channel_ids=["C1"]), 10
    )
    assert "W1" not in sql
    assert params == ["W1' OR true --", "C1", "message", "file_chunk"]


def test_filter_hard_cap() -> None:
    with pytest.raises(ValueError, match="too many"):
        _filter_sql(SearchFilters(channel_ids=[str(i) for i in range(4)]), 3)


def test_bad_cursor_rejected() -> None:
    with pytest.raises(ValueError, match="cursor"):
        _decode_cursor("not-a-cursor", "artifact", "fingerprint")


@pytest.mark.asyncio
async def test_mcp_contract_names(settings: Settings) -> None:
    tools = await create_mcp(settings).list_tools()
    assert {tool.name for tool in tools} == {
        "search_slack",
        "get_slack_message",
        "get_slack_thread",
        "list_slack_scopes",
    }


def test_health_and_unready(settings: Settings) -> None:
    with TestClient(create_asgi_app(settings)) as client:
        assert client.get("/healthz").json() == {
            "healthy": True,
            "embedding_backend": "ollama",
            "embedding_base_url": settings.embedding_base_url,
        }
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["ready"] is False


def test_bearer_auth_protects_mcp_but_not_health(settings: Settings) -> None:
    protected = settings.model_copy(update={"bearer_token": SecretStr("secret")})
    with TestClient(create_asgi_app(protected)) as client:
        assert client.get("/healthz").status_code == 200
        unauthorized = client.post("/mcp")
        assert unauthorized.status_code == 401
        assert unauthorized.headers["www-authenticate"] == "Bearer"
        authorized = client.post("/mcp", headers={"Authorization": "Bearer secret"})
        assert authorized.status_code != 401
