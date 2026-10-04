"""Local embedding client and durable resumable embedding worker."""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import duckdb
import httpx

from slackquery.db import connect_state
from slackquery.models import EmbedStats
from slackquery.settings import Settings

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingValidationError(ValueError):
    """The embedding server returned structurally invalid data."""


class EmbeddingRequestError(RuntimeError):
    """A classified embedding request failure."""

    def __init__(
        self, message: str, *, retryable: bool, retry_after_seconds: float | None = None
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class BatchEmbeddingValidationError(EmbeddingValidationError):
    """Some vectors in an otherwise valid batch failed validation."""

    def __init__(self, outcomes: list[list[float] | EmbeddingValidationError]) -> None:
        super().__init__("one or more embeddings failed validation")
        self.outcomes = outcomes


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def normalize_embedding(values: list[float], dimension: int = 512) -> list[float]:
    """Validate native output, first-dimension truncate, and L2 normalize."""
    if len(values) < dimension:
        raise EmbeddingValidationError(
            f"native embedding dimension {len(values)} is below required {dimension}"
        )
    truncated = [float(value) for value in values[:dimension]]
    if not all(math.isfinite(value) for value in truncated):
        raise EmbeddingValidationError("embedding contains non-finite values")
    norm = math.sqrt(sum(value * value for value in truncated))
    if norm <= 0:
        raise EmbeddingValidationError("embedding has zero L2 norm")
    return [value / norm for value in truncated]


class LocalEmbeddingClient:
    """Strict adapter for compatible local `/api/embed` backends."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owned = client is None
        self._model_verified = False
        self.native_dimension: int | None = None
        self.device: str | None = None
        self.health_status: str | None = None
        self.client = client or httpx.AsyncClient(
            base_url=settings.embedding_base_url,
            timeout=settings.embedding_timeout_seconds,
        )

    async def close(self) -> None:
        if self._owned:
            await self.client.aclose()

    async def verify_model(self) -> None:
        """Verify backend health, model digest, and native dimension once."""
        if self._model_verified or not self.settings.embedding_verify_model:
            return
        if self.settings.embedding_backend == "pytorch":
            await self._verify_pytorch_health()
        await self._verify_tags()
        self._model_verified = True

    async def _get(self, path: str, purpose: str) -> httpx.Response:
        try:
            response = await self.client.get(path, headers=self._authentication_headers())
        except httpx.TransportError as error:
            raise EmbeddingRequestError(type(error).__name__, retryable=True) from error
        if response.status_code >= 400:
            raise EmbeddingRequestError(
                f"{self.settings.embedding_backend} HTTP {response.status_code} while {purpose}",
                retryable=response.status_code == 429 or response.status_code >= 500,
                retry_after_seconds=_retry_after(response.headers.get("Retry-After")),
            )
        return response

    def _authentication_headers(self) -> dict[str, str]:
        key = self.settings.embedding_api_key
        if self.settings.embedding_backend != "pytorch" or key is None:
            return {}
        return {"Authorization": f"Bearer {key.get_secret_value()}"}

    async def _verify_pytorch_health(self) -> None:
        response = await self._get("/health", "checking health")
        try:
            health = response.json()
            status = str(health["status"])
            model = str(health.get("ollama_name", health["model"]))
            device = str(health["device"])
            native_dimension = int(health["native_dimension"])
            code_revision = health.get("code_revision")
        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingValidationError("invalid PyTorch /health response") from error
        if status.lower() not in {"ok", "healthy"}:
            raise EmbeddingValidationError(f"PyTorch backend is unhealthy: {status}")
        if model != self.settings.embedding_model:
            raise EmbeddingValidationError(
                "PyTorch health model mismatch: expected "
                f"{self.settings.embedding_model}, got {model}"
            )
        if device.lower() != "cuda":
            raise EmbeddingValidationError(f"PyTorch backend must use CUDA, got {device}")
        if native_dimension != 768:
            raise EmbeddingValidationError(
                f"PyTorch native dimension must be 768, got {native_dimension}"
            )
        expected_code_revision = self.settings.embedding_code_revision
        if expected_code_revision is not None and code_revision != expected_code_revision:
            raise EmbeddingValidationError(
                "PyTorch code revision mismatch: expected "
                f"{expected_code_revision}, got {code_revision}"
            )
        self.health_status = status
        self.device = device
        self.native_dimension = native_dimension

    async def _verify_tags(self) -> None:
        response = await self._get("/api/tags", "inspecting models")
        try:
            models = response.json()["models"]
            model = next(
                item
                for item in models
                if item.get("name") == self.settings.embedding_model
                or item.get("model") == self.settings.embedding_model
            )
            digest = model["digest"]
            native_dimension = model["details"]["embedding_length"]
            code_revision = model["details"].get("code_revision")
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            raise EmbeddingValidationError(
                f"model {self.settings.embedding_model!r} is absent from /api/tags"
            ) from error
        expected_digest = self.settings.embedding_model_revision
        if expected_digest is not None and digest != expected_digest:
            raise EmbeddingValidationError(
                "model digest mismatch: expected "
                f"{expected_digest}, got {digest}"
            )
        if not isinstance(native_dimension, int) or native_dimension != 768:
            raise EmbeddingValidationError(
                f"embedding native dimension must be 768, got {native_dimension!r}"
            )
        if self.native_dimension is not None and self.native_dimension != native_dimension:
            raise EmbeddingValidationError(
                "PyTorch /health and /api/tags native dimensions disagree: "
                f"{self.native_dimension} != {native_dimension}"
            )
        expected_code_revision = self.settings.embedding_code_revision
        if expected_code_revision is not None and code_revision != expected_code_revision:
            raise EmbeddingValidationError(
                "model code revision mismatch: expected "
                f"{expected_code_revision}, got {code_revision}"
            )
        self.native_dimension = native_dimension

    def status(self) -> dict[str, object]:
        """Return transport metadata that is excluded from generation identity."""
        return {
            "backend": self.settings.embedding_backend,
            "base_url": self.settings.embedding_base_url,
            "model": self.settings.embedding_model,
            "model_revision": self.settings.embedding_model_revision,
            "code_revision": self.settings.embedding_code_revision,
            "native_dimension": self.native_dimension,
            "stored_dimension": self.settings.embedding_dim,
            "device": self.device,
            "health_status": self.health_status,
            "generation_id": self.settings.generation_id,
            "verified": self._model_verified,
        }

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        await self.verify_model()
        prefix = QUERY_PREFIX if query else DOCUMENT_PREFIX
        payload = {
            "model": self.settings.embedding_model,
            "input": [prefix + text for text in texts],
            "truncate": True,
            "keep_alive": self.settings.embedding_keep_alive,
        }
        try:
            response = await self.client.post(
                "/api/embed", json=payload, headers=self._authentication_headers()
            )
        except httpx.TransportError as error:
            raise EmbeddingRequestError(type(error).__name__, retryable=True) from error
        if response.status_code >= 400:
            retryable = response.status_code == 429 or response.status_code >= 500
            raise EmbeddingRequestError(
                f"{self.settings.embedding_backend} HTTP {response.status_code}",
                retryable=retryable,
                retry_after_seconds=_retry_after(response.headers.get("Retry-After")),
            )
        try:
            data = response.json()
            raw = data["embeddings"]
        except (ValueError, KeyError, TypeError) as error:
            raise EmbeddingValidationError("invalid embedding response") from error
        if not isinstance(raw, list) or len(raw) != len(texts):
            raise EmbeddingValidationError(
                f"embedding count mismatch: expected {len(texts)}, got "
                f"{len(raw) if isinstance(raw, list) else 'non-list'}"
            )
        outcomes: list[list[float] | EmbeddingValidationError] = []
        for vector in raw:
            try:
                if isinstance(vector, list):
                    if self.native_dimension is None:
                        self.native_dimension = len(vector)
                    elif len(vector) != self.native_dimension:
                        raise EmbeddingValidationError(
                            f"inconsistent native embedding dimension: expected "
                            f"{self.native_dimension}, got {len(vector)}"
                        )
                outcomes.append(normalize_embedding(vector, self.settings.embedding_dim))
            except (EmbeddingValidationError, TypeError, ValueError) as error:
                if not isinstance(error, EmbeddingValidationError):
                    error = EmbeddingValidationError("embedding is not a numeric vector")
                outcomes.append(error)
        if any(isinstance(outcome, EmbeddingValidationError) for outcome in outcomes):
            raise BatchEmbeddingValidationError(outcomes)
        return [outcome for outcome in outcomes if isinstance(outcome, list)]


# Backward-compatible import/API name. Both backends expose the Ollama API shape.
OllamaClient = LocalEmbeddingClient


@dataclass(frozen=True)
class ClaimedDocument:
    document_id: str
    source_version: str
    text_hash: str
    text: str
    attempt_count: int


class EmbeddingWorker:
    """Checkpointed batch embedding state machine."""

    def __init__(self, settings: Settings, client: LocalEmbeddingClient) -> None:
        self.settings = settings
        self.client = client

    def _ensure_generation(self, connection: duckdb.DuckDBPyConnection) -> None:
        model_revision = self.settings.embedding_model_revision or "unpinned"
        config_hash = hashlib.sha256(
            f"{self.settings.embedding_model}|{model_revision}|"
            f"{self.settings.embedding_code_revision or 'none'}|512|v1".encode()
        ).hexdigest()
        connection.execute(
            """
            INSERT INTO embedding_generations VALUES
              (?, 'ollama-compatible', ?, ?, ?, 512, ?, ?, true, 'cosine', 'message-v1',
                ?, 'active', current_timestamp, NULL)
            ON CONFLICT (generation_id) DO UPDATE SET
              native_dimension = coalesce(excluded.native_dimension,
                                          embedding_generations.native_dimension)
            """,
            [
                self.settings.generation_id,
                self.settings.embedding_model,
                model_revision,
                self.client.native_dimension,
                DOCUMENT_PREFIX,
                QUERY_PREFIX,
                config_hash,
            ],
        )

    def _claim(self, connection: duckdb.DuckDBPyConnection, limit: int) -> list[ClaimedDocument]:
        lease_id = str(uuid.uuid4())
        lease_until = datetime.now(UTC) + timedelta(seconds=self.settings.lease_seconds)
        connection.execute("BEGIN")
        self._ensure_generation(connection)
        connection.execute(
            """
            INSERT INTO document_embeddings
              (document_id, generation_id, source_version, text_hash, state,
               attempt_count, updated_at)
            SELECT p.document_id, ?, p.source_version, p.text_hash, 'pending', 0,
                   current_timestamp
            FROM document_projection p
            LEFT JOIN document_embeddings e
              ON e.document_id = p.document_id AND e.generation_id = ?
                 AND e.text_hash = p.text_hash
            WHERE p.is_active AND e.document_id IS NULL
            """,
            [self.settings.generation_id, self.settings.generation_id],
        )
        rows = connection.execute(
            """
            SELECT e.document_id, e.source_version, e.text_hash, p.text_embedding,
                   e.attempt_count
            FROM document_embeddings e
            JOIN document_projection p USING (document_id, source_version, text_hash)
            WHERE e.generation_id = ? AND p.is_active AND (
              e.state = 'pending'
              OR (e.state = 'retryable_failed' AND
                  coalesce(e.next_attempt_at, current_timestamp) <= current_timestamp)
              OR (e.state = 'in_progress' AND e.lease_expires_at < current_timestamp)
            )
            ORDER BY e.updated_at, e.document_id
            LIMIT ?
            """,
            [self.settings.generation_id, limit],
        ).fetchall()
        claimed: list[ClaimedDocument] = []
        used_chars = 0
        for row in rows:
            if claimed and used_chars + len(row[3]) > self.settings.embedding_max_chars:
                break
            claimed.append(ClaimedDocument(*row))
            used_chars += len(row[3])
        if claimed:
            connection.executemany(
                """
                UPDATE document_embeddings
                SET state = 'in_progress', lease_id = ?, lease_expires_at = ?,
                    attempt_count = attempt_count + 1, updated_at = current_timestamp,
                    error_class = NULL, error_message = NULL
                WHERE document_id = ? AND generation_id = ? AND text_hash = ?
                """,
                [
                    [lease_id, lease_until, item.document_id, self.settings.generation_id,
                     item.text_hash]
                    for item in claimed
                ],
            )
        connection.execute("COMMIT")
        return claimed

    async def run(self, max_items: int | None = None) -> EmbedStats:
        await self.client.verify_model()
        connection = connect_state(self.settings.state_db)
        claimed_count = succeeded = retryable_failed = terminal_failed = 0
        remaining = max_items
        try:
            while remaining is None or remaining > 0:
                size = self.settings.embedding_batch_size
                if remaining is not None:
                    size = min(size, remaining)
                batch = self._claim(connection, size)
                if not batch:
                    break
                claimed_count += len(batch)
                try:
                    vectors = await self.client.embed([item.text for item in batch])
                except BatchEmbeddingValidationError as error:
                    for item, outcome in zip(batch, error.outcomes, strict=True):
                        if isinstance(outcome, EmbeddingValidationError):
                            connection.execute(
                                """
                                UPDATE document_embeddings SET state = 'terminal_failed',
                                  lease_id = NULL, lease_expires_at = NULL,
                                  error_class = ?, error_message = ?,
                                  updated_at = current_timestamp
                                WHERE document_id = ? AND generation_id = ? AND text_hash = ?
                                """,
                                [
                                    type(outcome).__name__,
                                    str(outcome)[:500],
                                    item.document_id,
                                    self.settings.generation_id,
                                    item.text_hash,
                                ],
                            )
                            terminal_failed += 1
                        else:
                            connection.execute(
                                """
                                UPDATE document_embeddings SET state = 'succeeded', embedding = ?,
                                  native_dimension = ?, embedded_at = current_timestamp,
                                  lease_id = NULL, lease_expires_at = NULL,
                                  next_attempt_at = NULL, error_class = NULL,
                                  error_message = NULL, updated_at = current_timestamp
                                WHERE document_id = ? AND generation_id = ? AND text_hash = ?
                                """,
                                [
                                    outcome,
                                    self.client.native_dimension or len(outcome),
                                    item.document_id,
                                    self.settings.generation_id,
                                    item.text_hash,
                                ],
                            )
                            succeeded += 1
                except (EmbeddingRequestError, EmbeddingValidationError) as error:
                    is_retryable = isinstance(error, EmbeddingRequestError) and error.retryable
                    for item in batch:
                        terminal = (
                            not is_retryable
                            or item.attempt_count + 1 >= self.settings.embedding_max_attempts
                        )
                        state = "terminal_failed" if terminal else "retryable_failed"
                        retry_after = (
                            error.retry_after_seconds
                            if isinstance(error, EmbeddingRequestError)
                            else None
                        )
                        delay = retry_after or (
                            min(300, 2 ** min(item.attempt_count, 8)) + random.random()
                        )
                        connection.execute(
                            """
                            UPDATE document_embeddings SET state = ?, next_attempt_at = ?,
                              lease_id = NULL, lease_expires_at = NULL, error_class = ?,
                              error_message = ?, updated_at = current_timestamp
                            WHERE document_id = ? AND generation_id = ? AND text_hash = ?
                            """,
                            [
                                state,
                                datetime.now(UTC) + timedelta(seconds=delay),
                                type(error).__name__,
                                str(error)[:500],
                                item.document_id,
                                self.settings.generation_id,
                                item.text_hash,
                            ],
                        )
                        if terminal:
                            terminal_failed += 1
                        else:
                            retryable_failed += 1
                else:
                    connection.executemany(
                        """
                        UPDATE document_embeddings SET state = 'succeeded', embedding = ?,
                          native_dimension = ?, embedded_at = current_timestamp,
                          lease_id = NULL, lease_expires_at = NULL, next_attempt_at = NULL,
                          error_class = NULL, error_message = NULL, updated_at = current_timestamp
                        WHERE document_id = ? AND generation_id = ? AND text_hash = ?
                        """,
                        [
                            [
                                vector,
                                self.client.native_dimension or len(vector),
                                item.document_id,
                                self.settings.generation_id,
                                item.text_hash,
                            ]
                            for item, vector in zip(batch, vectors, strict=True)
                        ],
                    )
                    succeeded += len(batch)
                if remaining is not None:
                    remaining -= len(batch)
            return EmbedStats(
                claimed=claimed_count,
                succeeded=succeeded,
                retryable_failed=retryable_failed,
                terminal_failed=terminal_failed,
            )
        finally:
            connection.close()


async def embed_query(settings: Settings, query: str) -> list[float]:
    client = LocalEmbeddingClient(settings)
    try:
        return (await client.embed([query], query=True))[0]
    finally:
        await client.close()


def run_worker(settings: Settings, max_items: int | None = None) -> EmbedStats:
    client = LocalEmbeddingClient(settings)

    async def execute() -> EmbedStats:
        try:
            return await EmbeddingWorker(settings, client).run(max_items)
        finally:
            await client.close()

    return asyncio.run(execute())
