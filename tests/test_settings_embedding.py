from __future__ import annotations

import math

import httpx
import pytest

from slackquery.embedding import (
    BatchEmbeddingValidationError,
    EmbeddingValidationError,
    LocalEmbeddingClient,
    OllamaClient,
    normalize_embedding,
)
from slackquery.settings import Settings


def test_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACKQUERY_PORT", "9001")
    monkeypatch.setenv("SLACKQUERY_EMBEDDING_BASE_URL", "http://ollama:11434/")
    settings = Settings()
    assert settings.port == 9001
    assert settings.embedding_base_url == "http://ollama:11434"
    assert settings.embedding_backend == "pytorch"
    assert settings.embedding_dim == 512


def test_exact_embedding_env_names_take_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://exact:11434/")
    monkeypatch.setenv("EMBEDDING_MODEL", "exact-model")
    monkeypatch.setenv("EMBEDDING_DIM", "512")
    monkeypatch.setenv("SLACKQUERY_EMBEDDING_BASE_URL", "http://alias:11434")
    settings = Settings()
    assert settings.embedding_base_url == "http://exact:11434"
    assert settings.embedding_model == "exact-model"
    assert settings.embedding_dim == 512


def test_default_runtime_paths_urls_and_optional_digest() -> None:
    settings = Settings(_env_file=None)
    assert str(settings.state_db).startswith("/srv/slackquery/")
    assert settings.embedding_backend == "pytorch"
    assert settings.embedding_base_url == "http://localhost:11435"
    assert settings.duckdb_extension_dir == __import__("pathlib").Path(
        "/srv/slackquery/extensions"
    )
    assert settings.embedding_model_revision is None


def test_backend_flag_switches_url_without_changing_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("SLACKQUERY_EMBEDDING_BASE_URL", raising=False)
    pytorch = Settings(_env_file=None)
    monkeypatch.setenv("EMBEDDING_BACKEND", "ollama")
    ollama = Settings(_env_file=None)
    assert pytorch.embedding_base_url == "http://localhost:11435"
    assert ollama.embedding_base_url == "http://localhost:11434"
    assert pytorch.generation_id == ollama.generation_id


def test_backend_specific_urls_and_common_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_EMBEDDING_BASE_URL", "http://torch:11435/")
    monkeypatch.setenv("OLLAMA_EMBEDDING_BASE_URL", "http://ollama:11434/")
    assert Settings(_env_file=None).embedding_base_url == "http://torch:11435"
    monkeypatch.setenv("EMBEDDING_BACKEND", "ollama")
    assert Settings(_env_file=None).embedding_base_url == "http://ollama:11434"
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://override:9000/")
    assert Settings(_env_file=None).embedding_base_url == "http://override:9000"


def test_ollama_client_symbol_is_backward_compatible() -> None:
    assert OllamaClient is LocalEmbeddingClient


def test_normalize_truncates_before_l2() -> None:
    values = [0.0] * 768
    values[0] = 3.0
    values[1] = 4.0
    values[600] = 10_000.0
    result = normalize_embedding(values)
    assert len(result) == 512
    assert result[:2] == pytest.approx([0.6, 0.8])
    assert math.sqrt(sum(value * value for value in result)) == pytest.approx(1.0)


@pytest.mark.parametrize("values", [[1.0] * 511, [0.0] * 768, [math.nan] * 768])
def test_normalize_rejects_bad_vectors(values: list[float]) -> None:
    with pytest.raises(EmbeddingValidationError):
        normalize_embedding(values)


@pytest.mark.asyncio
async def test_ollama_batch_and_prefixes(settings: Settings) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        requests.append(payload)
        vectors = []
        for index, _ in enumerate(payload["input"]):
            vector = [0.0] * 768
            vector[index] = 1.0
            vectors.append(vector)
        return httpx.Response(200, json={"embeddings": vectors})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama"
    ) as http:
        client = OllamaClient(settings, http)
        result = await client.embed(["one", "two"])
        await client.embed(["question"], query=True)
    assert len(result) == 2 and len(result[0]) == 512
    assert requests[0]["input"] == ["search_document: one", "search_document: two"]
    assert requests[1]["input"] == ["search_query: question"]
    assert requests[0]["truncate"] is True
    assert requests[0]["keep_alive"] == "5m"


@pytest.mark.asyncio
async def test_ollama_verifies_digest(settings: Settings) -> None:
    settings = settings.model_copy(update={"embedding_verify_model": True})

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": settings.embedding_model,
                        "digest": "wrong",
                        "details": {"embedding_length": 768},
                    }
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama"
    ) as http:
        with pytest.raises(EmbeddingValidationError, match="digest mismatch"):
            await OllamaClient(settings, http).embed(["one"])


@pytest.mark.asyncio
async def test_pytorch_verifies_health_cuda_tags_digest_and_dimension(settings: Settings) -> None:
    settings = settings.model_copy(
        update={"embedding_backend": "pytorch", "embedding_verify_model": True}
    )
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "model": "nomic-ai/nomic-embed-text-v1.5",
                    "ollama_name": settings.embedding_model,
                    "device": "cuda",
                    "native_dimension": 768,
                },
            )
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [{
                        "name": settings.embedding_model,
                        "digest": settings.embedding_model_revision,
                        "details": {"embedding_length": 768},
                    }]
                },
            )
        return httpx.Response(200, json={"embeddings": [[1.0] * 768]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://pytorch"
    ) as http:
        client = LocalEmbeddingClient(settings, http)
        await client.embed(["one"])
        status = client.status()
    assert paths == ["/health", "/api/tags", "/api/embed"]
    assert status["backend"] == "pytorch"
    assert status["device"] == "cuda"
    assert status["native_dimension"] == 768
    assert status["generation_id"] == settings.generation_id


@pytest.mark.asyncio
async def test_pytorch_rejects_non_cuda_health(settings: Settings) -> None:
    settings = settings.model_copy(
        update={"embedding_backend": "pytorch", "embedding_verify_model": True}
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "model": settings.embedding_model,
                "device": "cpu",
                "native_dimension": 768,
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://pytorch"
    ) as http:
        with pytest.raises(EmbeddingValidationError, match="must use CUDA"):
            await LocalEmbeddingClient(settings, http).verify_model()


@pytest.mark.asyncio
async def test_ollama_reports_per_item_validation(settings: Settings) -> None:
    good = [1.0] * 768
    bad = [1.0] * 12

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [good, bad]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama"
    ) as http:
        with pytest.raises(BatchEmbeddingValidationError) as raised:
            await OllamaClient(settings, http).embed(["good", "bad"])
    assert isinstance(raised.value.outcomes[0], list)
    assert isinstance(raised.value.outcomes[1], EmbeddingValidationError)
