from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient

from slackquery.embedding import LocalEmbeddingClient
from slackquery.embedding_server import (
    NATIVE_DIMENSION,
    EmbeddingServerSettings,
    create_embedding_app,
)
from slackquery.settings import Settings


class FakeEncoder:
    model_id = "nomic-ai/nomic-embed-text-v1.5"
    device = "cuda"
    gpu_name = "Test GPU"

    def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        assert batch_size == 2
        return [
            [float(index + 1)] + [0.0] * (NATIVE_DIMENSION - 1)
            for index, _ in enumerate(texts)
        ]


@pytest.fixture
def server_settings() -> EmbeddingServerSettings:
    return EmbeddingServerSettings(
        api_key="a-secure-test-key",
        model_revision="a" * 40,
        code_revision="b" * 40,
        batch_size=2,
        max_items=2,
    )


@pytest.fixture
def client(server_settings: EmbeddingServerSettings) -> TestClient:
    return TestClient(create_embedding_app(server_settings, FakeEncoder()))


def test_server_requires_api_key(client: TestClient) -> None:
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/api/embed", json={"input": "hello"}).status_code == 401


def test_health_and_tags_are_compatible(
    client: TestClient, server_settings: EmbeddingServerSettings
) -> None:
    headers = {"Authorization": f"Bearer {server_settings.api_key}"}
    health = client.get("/health", headers=headers)
    tags = client.get("/api/tags", headers=headers)
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model": FakeEncoder.model_id,
        "ollama_name": server_settings.model_alias,
        "model_revision": "a" * 40,
        "code_revision": "b" * 40,
        "device": "cuda",
        "gpu": "Test GPU",
        "native_dimension": 768,
    }
    assert tags.json()["models"][0]["digest"] == "a" * 40
    assert tags.json()["models"][0]["details"]["embedding_length"] == 768
    assert tags.json()["models"][0]["details"]["code_revision"] == "b" * 40


def test_ollama_and_openai_embedding_shapes(
    client: TestClient, server_settings: EmbeddingServerSettings
) -> None:
    headers = {"Authorization": f"Bearer {server_settings.api_key}"}
    payload = {"model": server_settings.model_alias, "input": ["one", "two"]}
    ollama = client.post("/api/embed", headers=headers, json=payload)
    openai = client.post("/v1/embeddings", headers=headers, json=payload)
    assert ollama.status_code == 200
    assert len(ollama.json()["embeddings"]) == 2
    assert len(ollama.json()["embeddings"][0]) == 768
    assert openai.status_code == 200
    assert [item["index"] for item in openai.json()["data"]] == [0, 1]
    assert len(openai.json()["data"][0]["embedding"]) == 768


def test_server_rejects_unknown_model_and_oversized_batch(
    client: TestClient, server_settings: EmbeddingServerSettings
) -> None:
    headers = {"Authorization": f"Bearer {server_settings.api_key}"}
    unknown = client.post(
        "/api/embed", headers=headers, json={"model": "other", "input": "one"}
    )
    oversized = client.post(
        "/api/embed", headers=headers, json={"input": ["one", "two", "three"]}
    )
    assert unknown.status_code == 404
    assert oversized.status_code == 400


@pytest.mark.parametrize(
    "option",
    [
        {"dimensions": 512},
        {"encoding_format": "base64"},
        {"truncate": False},
    ],
)
def test_server_rejects_unsupported_options(
    client: TestClient,
    server_settings: EmbeddingServerSettings,
    option: dict[str, object],
) -> None:
    headers = {"Authorization": f"Bearer {server_settings.api_key}"}
    response = client.post("/api/embed", headers=headers, json={"input": "one", **option})
    assert response.status_code == 400


def test_server_limits_streamed_body_without_content_length() -> None:
    settings = EmbeddingServerSettings(
        api_key="a-secure-test-key",
        model_revision="a" * 40,
        code_revision="b" * 40,
        max_chars=512,
        max_body_bytes=600,
    )
    client = TestClient(create_embedding_app(settings, FakeEncoder()))
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    response = client.post(
        "/api/embed",
        headers=headers,
        content=(chunk for chunk in [b"{" + b"x" * 400, b"y" * 400 + b"}"]),
    )
    assert response.status_code == 413


def test_server_settings_require_strong_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACKQUERY_EMBEDDING_SERVER_API_KEY", "short")
    with pytest.raises(ValueError, match="at least 16"):
        EmbeddingServerSettings.from_env()


def test_server_settings_reject_non_ascii_key() -> None:
    with pytest.raises(ValueError, match="printable ASCII"):
        EmbeddingServerSettings(
            api_key="not-ascii-key-cafe\N{LATIN SMALL LETTER E WITH ACUTE}",
            model_revision="a" * 40,
            code_revision="b" * 40,
        )


@pytest.mark.asyncio
async def test_main_client_integrates_with_authenticated_server(
    server_settings: EmbeddingServerSettings,
) -> None:
    app = create_embedding_app(server_settings, FakeEncoder())
    settings = Settings(
        _env_file=None,
        embedding_backend="pytorch",
        embedding_base_url_override="http://embedding.test",
        embedding_api_key=SecretStr(server_settings.api_key),
        embedding_model=server_settings.model_alias,
        embedding_model_revision=server_settings.model_revision,
        embedding_code_revision=server_settings.code_revision,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://embedding.test"
    ) as http:
        embedding_client = LocalEmbeddingClient(settings, http)
        vectors = await embedding_client.embed(["hello"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 512
    assert embedding_client.status()["verified"] is True
