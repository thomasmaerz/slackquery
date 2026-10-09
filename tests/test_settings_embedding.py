from __future__ import annotations

import asyncio
import math

import httpx
import pytest
from pydantic import SecretStr

from slackquery.embedding import (
    BatchEmbeddingValidationError,
    EmbeddingRequestError,
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


def test_embedding_api_key_loads_as_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-embedding-api-key")
    settings = Settings(_env_file=None)
    assert settings.embedding_api_key is not None
    assert settings.embedding_api_key.get_secret_value() == "test-embedding-api-key"
    assert "test-embedding-api-key" not in repr(settings.embedding_api_key)


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


def test_code_revision_changes_generation_identity() -> None:
    baseline = Settings(_env_file=None)
    pinned = Settings(_env_file=None, embedding_code_revision="code-commit")
    assert baseline.generation_id != pinned.generation_id
    assert ":code-code-commit:" in pinned.generation_id


def test_pytorch_uses_separate_batch_cap() -> None:
    pytorch = Settings(
        _env_file=None,
        embedding_backend="pytorch",
        embedding_batch_size=32,
        pytorch_embedding_batch_size=8,
    )
    ollama = pytorch.model_copy(update={"embedding_backend": "ollama"})
    assert pytorch.effective_embedding_batch_size == 8
    assert ollama.effective_embedding_batch_size == 32


def test_busy_backoff_cap_must_cover_base() -> None:
    with pytest.raises(ValueError, match="backoff_cap_seconds"):
        Settings(
            _env_file=None,
            embedding_busy_backoff_base_seconds=2,
            embedding_busy_backoff_cap_seconds=1,
        )


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
                    "code_revision": settings.embedding_code_revision,
                },
            )
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [{
                        "name": settings.embedding_model,
                        "digest": settings.embedding_model_revision,
                        "details": {
                            "embedding_length": 768,
                            "code_revision": settings.embedding_code_revision,
                        },
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
async def test_pytorch_sends_api_key_without_leaking_it_from_status(settings: Settings) -> None:
    secret = "test-embedding-api-key"
    settings = settings.model_copy(
        update={
            "embedding_backend": "pytorch",
            "embedding_api_key": SecretStr(secret),
            "embedding_verify_model": False,
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {secret}"
        return httpx.Response(200, json={"embeddings": [[1.0] * 768]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://pytorch"
    ) as http:
        client = LocalEmbeddingClient(settings, http)
        await client.embed(["one"])
        assert secret not in repr(client.status())


@pytest.mark.asyncio
async def test_ollama_does_not_receive_pytorch_api_key(settings: Settings) -> None:
    settings = settings.model_copy(
        update={"embedding_api_key": SecretStr("test-embedding-api-key")}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"embeddings": [[1.0] * 768]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama"
    ) as http:
        await LocalEmbeddingClient(settings, http).embed(["one"])


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


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    async def sleep(self, delay: float) -> None:
        self.value += delay


def busy_response(
    settings: Settings,
    *,
    retry_after: str = "2",
    code: str = "MODEL_BUSY",
    retryable: bool = True,
    requested_model: str | None = None,
) -> httpx.Response:
    return httpx.Response(
        503,
        headers={"Retry-After": retry_after},
        json={
            "error": {
                "code": code,
                "retryable": retryable,
                "requested_model": requested_model or "nomic",
            }
        },
    )


@pytest.mark.asyncio
async def test_authenticated_model_busy_retries_then_succeeds(settings: Settings) -> None:
    secret = "test-embedding-api-key"
    settings = settings.model_copy(
        update={
            "embedding_backend": "pytorch",
            "embedding_api_key": SecretStr(secret),
            "embedding_verify_model": False,
            "embedding_busy_backoff_base_seconds": 0.25,
            "embedding_busy_backoff_cap_seconds": 1.0,
        }
    )
    clock = FakeClock()
    delays: list[float] = []
    requests: list[httpx.Request] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        await clock.sleep(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) < 3:
            return busy_response(settings)
        return httpx.Response(200, json={"embeddings": [[1.0] * 768]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://pytorch"
    ) as http:
        client = LocalEmbeddingClient(
            settings,
            http,
            sleep=sleep,
            monotonic=clock,
            uniform=lambda _low, high: high,
        )
        result = await client.embed(["synthetic retry fixture"])

    assert len(result[0]) == 512
    assert delays == [2.0, 2.0]
    assert len({request.content for request in requests}) == 1
    assert all(request.headers["Authorization"] == f"Bearer {secret}" for request in requests)


@pytest.mark.asyncio
async def test_model_busy_uses_capped_full_jitter(settings: Settings) -> None:
    settings = settings.model_copy(
        update={
            "embedding_backend": "pytorch",
            "embedding_api_key": SecretStr("test-embedding-api-key"),
            "embedding_verify_model": False,
            "embedding_busy_backoff_base_seconds": 0.5,
            "embedding_busy_backoff_cap_seconds": 1.0,
        }
    )
    clock = FakeClock()
    calls = 0
    bounds: list[tuple[float, float]] = []
    delays: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 4:
            return busy_response(settings, retry_after="0")
        return httpx.Response(200, json={"embeddings": [[1.0] * 768]})

    def uniform(low: float, high: float) -> float:
        bounds.append((low, high))
        return high / 2

    async def sleep(delay: float) -> None:
        delays.append(delay)
        await clock.sleep(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://pytorch"
    ) as http:
        await LocalEmbeddingClient(
            settings, http, sleep=sleep, monotonic=clock, uniform=uniform
        ).embed(["one"])

    assert bounds == [(0.0, 0.5), (0.0, 1.0), (0.0, 1.0)]
    assert delays == [0.25, 0.5, 0.5]


@pytest.mark.asyncio
async def test_model_busy_attempt_limit_is_bounded_and_redacted(settings: Settings) -> None:
    secret = "test-embedding-api-key"
    input_text = "synthetic text must not appear in errors"
    settings = settings.model_copy(
        update={
            "embedding_backend": "pytorch",
            "embedding_api_key": SecretStr(secret),
            "embedding_verify_model": False,
            "embedding_busy_max_attempts": 3,
        }
    )
    clock = FakeClock()
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return busy_response(settings, retry_after="0")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://pytorch"
    ) as http:
        with pytest.raises(EmbeddingRequestError) as raised:
            await LocalEmbeddingClient(
                settings, http, sleep=clock.sleep, monotonic=clock
            ).embed([input_text])

    assert calls == 3
    assert raised.value.retries == 2
    assert "after 2 retries: attempt limit reached" in str(raised.value)
    assert settings.embedding_model in str(raised.value)
    assert input_text not in str(raised.value)
    assert secret not in str(raised.value)


@pytest.mark.asyncio
async def test_model_busy_total_deadline_stops_before_sleep(settings: Settings) -> None:
    settings = settings.model_copy(
        update={
            "embedding_backend": "pytorch",
            "embedding_api_key": SecretStr("test-embedding-api-key"),
            "embedding_verify_model": False,
            "embedding_busy_deadline_seconds": 1.0,
        }
    )
    clock = FakeClock()
    slept = False

    async def sleep(_: float) -> None:
        nonlocal slept
        slept = True

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: busy_response(settings, retry_after="2")),
        base_url="http://pytorch",
    ) as http:
        with pytest.raises(EmbeddingRequestError, match="retry deadline exceeded") as raised:
            await LocalEmbeddingClient(
                settings, http, sleep=sleep, monotonic=clock
            ).embed(["one"])
    assert raised.value.retries == 0
    assert not slept


@pytest.mark.asyncio
async def test_cancellation_during_busy_sleep_propagates(settings: Settings) -> None:
    settings = settings.model_copy(
        update={
            "embedding_backend": "pytorch",
            "embedding_api_key": SecretStr("test-embedding-api-key"),
            "embedding_verify_model": False,
        }
    )
    sleeping = asyncio.Event()

    async def sleep(_: float) -> None:
        sleeping.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: busy_response(settings)),
        base_url="http://pytorch",
    ) as http:
        task = asyncio.create_task(LocalEmbeddingClient(settings, http, sleep=sleep).embed(["one"]))
        await sleeping.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_cancellation_during_embedding_http_call_propagates(settings: Settings) -> None:
    settings = settings.model_copy(
        update={
            "embedding_backend": "pytorch",
            "embedding_api_key": SecretStr("test-embedding-api-key"),
            "embedding_verify_model": False,
        }
    )
    requested = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        requested.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://pytorch"
    ) as http:
        task = asyncio.create_task(LocalEmbeddingClient(settings, http).embed(["one"]))
        await requested.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.parametrize(
    ("status", "body", "headers"),
    [
        (
            503,
            {
                "error": {
                    "code": "OTHER",
                    "retryable": True,
                    "requested_model": "nomic-embed-text:v1.5",
                }
            },
            {"Retry-After": "1"},
        ),
        (
            503,
            {
                "error": {
                    "code": "MODEL_BUSY",
                    "retryable": False,
                    "requested_model": "nomic-embed-text:v1.5",
                }
            },
            {"Retry-After": "1"},
        ),
        (
            503,
            {
                "error": {
                    "code": "MODEL_BUSY",
                    "retryable": True,
                    "requested_model": "wrong-model",
                }
            },
            {"Retry-After": "1"},
        ),
        (
            503,
            {
                "error": {
                    "code": "MODEL_BUSY",
                    "retryable": True,
                    "requested_model": "nomic-embed-text:v1.5",
                }
            },
            {},
        ),
        (503, None, {"Retry-After": "1"}),
        (
            500,
            {
                "error": {
                    "code": "MODEL_BUSY",
                    "retryable": True,
                    "requested_model": "nomic-embed-text:v1.5",
                }
            },
            {"Retry-After": "1"},
        ),
        (
            429,
            {
                "error": {
                    "code": "MODEL_BUSY",
                    "retryable": True,
                    "requested_model": "nomic-embed-text:v1.5",
                }
            },
            {"Retry-After": "1"},
        ),
        (
            401,
            {
                "error": {
                    "code": "MODEL_BUSY",
                    "retryable": True,
                    "requested_model": "nomic-embed-text:v1.5",
                }
            },
            {"Retry-After": "1"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_non_busy_errors_are_not_immediately_retried(
    settings: Settings,
    status: int,
    body: dict[str, object] | None,
    headers: dict[str, str],
) -> None:
    settings = settings.model_copy(
        update={
            "embedding_backend": "pytorch",
            "embedding_api_key": SecretStr("test-embedding-api-key"),
            "embedding_verify_model": False,
        }
    )
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if body is None:
            return httpx.Response(status, headers=headers, content=b"not-json")
        return httpx.Response(status, headers=headers, json=body)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://pytorch"
    ) as http:
        with pytest.raises(EmbeddingRequestError):
            await LocalEmbeddingClient(settings, http).embed(["one"])
    assert calls == 1


@pytest.mark.parametrize("backend", ["pytorch", "ollama"])
@pytest.mark.asyncio
async def test_busy_retry_requires_authenticated_pytorch(
    settings: Settings, backend: str
) -> None:
    settings = settings.model_copy(
        update={
            "embedding_backend": backend,
            "embedding_api_key": None
            if backend == "pytorch"
            else SecretStr("test-embedding-api-key"),
            "embedding_verify_model": False,
        }
    )
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return busy_response(settings)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://embedding"
    ) as http:
        with pytest.raises(EmbeddingRequestError):
            await LocalEmbeddingClient(settings, http).embed(["one"])
    assert calls == 1
