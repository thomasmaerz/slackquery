from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from slackquery.artifact import build_artifact, publish_artifact, validate_artifact
from slackquery.db import connect_state
from slackquery.embedding import EmbeddingWorker, OllamaClient
from slackquery.models import SearchFilters
from slackquery.projection import project_documents
from slackquery.retrieval import SearchEngine
from slackquery.settings import Settings


@pytest.mark.asyncio
async def test_full_small_pipeline(
    settings: Settings, canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projection = project_documents(canonical, settings.state_db, settings)
    assert projection.projected == 5
    assert projection.active == 4

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        vectors = []
        for text in payload["input"]:
            values = [0.0] * 768
            if "timeout" in text.lower() or "database" in text.lower():
                values[0] = 1.0
            elif "connection" in text.lower():
                values[1] = 1.0
            else:
                values[2] = 1.0
            vectors.append(values)
        return httpx.Response(200, json={"embeddings": vectors})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama"
    ) as http:
        client = OllamaClient(settings, http)
        first = await EmbeddingWorker(settings, client).run()
        second = await EmbeddingWorker(settings, client).run()
        assert first.succeeded == 4
        assert second.claimed == 0
        state = connect_state(settings.state_db)
        try:
            assert state.execute(
                "SELECT min(native_dimension), max(native_dimension) FROM document_embeddings"
            ).fetchone() == (768, 768)
        finally:
            state.close()
        result = build_artifact(settings)
        assert result.document_count == 4
        checks = validate_artifact(Path(result.artifact_path), verify_checksum=True)
        assert checks["artifact_id"] == result.build_id
        assert checks["document_count"] == 4
        assert checks["vector_count"] == 3
        assert checks["valid"] is True
        publish_artifact(settings, Path(result.artifact_path))
        engine = SearchEngine(settings, client)
        lexical = await engine.search("ERR-42", mode="lexical")
        semantic = await engine.search("database issue", mode="semantic")
        hybrid = await engine.search("database timeout", mode="hybrid")
        filtered = await engine.search(
            "Database", mode="lexical", filters=SearchFilters(channel_ids=["C2"])
        )
        first_page = await engine.search("Database", mode="hybrid", limit=1)
        second_page = await engine.search(
            "Database", mode="hybrid", limit=1, cursor=first_page.next_cursor
        )
        paged_ids = [item.document_id for item in first_page.results + second_page.results]
        assert len(paged_ids) == len(set(paged_ids))
        replacement = build_artifact(settings)
        publish_artifact(settings, Path(replacement.artifact_path))
        reloaded = await engine.search("ERR-42", mode="lexical")
    assert calls == 6  # two document batches and four semantic query embeddings
    assert lexical.results[0].document_id == "W1:C1:1.000001"
    assert semantic.results[0].document_id == "W1:C1:1.000001"
    assert hybrid.results[0].lexical_rank == 1
    assert hybrid.results[0].semantic_rank == 1
    assert filtered.results == []
    assert first_page.next_cursor is not None
    assert {first_page.results[0].document_id, second_page.results[0].document_id} == {
        "W1:C1:1.000001",
        "W1:C1:2.000001",
    }
    assert reloaded.artifact_id == replacement.build_id
    assert reloaded.artifact_id != result.build_id
    assert settings.current_link.is_symlink()

    projected = (
        connect_state(settings.state_db)
        .execute(
            """
        SELECT document_id, thread_id, thread_root_id, permalink
        FROM document_projection WHERE document_id IN ('W1:C1:1.000001', 'W1:C1:2.000001')
        ORDER BY ts_us
        """
        )
        .fetchall()
    )
    assert projected[0][1:3] == ("W1:C1:1.000001", "W1:C1:1.000001")
    assert projected[1][1:3] == ("W1:C1:1.000001", "W1:C1:1.000001")
    assert projected[1][3].endswith("/p2000001?thread_ts=1.000001&cid=C1")

    thread = engine.get_thread("W1:C1:1.000001")
    assert [item.document_id for item in thread] == ["W1:C1:1.000001", "W1:C1:2.000001"]
    message = engine.get_message("W1:C1:2.000001", before=1)
    assert len(message) == 2
    scopes = engine.list_scopes()
    assert {(item.channel_id, item.message_count) for item in scopes} == {("C1", 2), ("C2", 1)}


def test_projection_deduplicates_user_keys(settings: Settings, canonical: Path) -> None:
    import duckdb

    connection = duckdb.connect(str(canonical))
    connection.execute(
        "INSERT INTO users VALUES ('W1', 'U1', 'duplicate', 'Duplicate', 'Duplicate')"
    )
    connection.close()
    stats = project_documents(canonical, settings.state_db)
    assert stats.projected == 4
    state = connect_state(settings.state_db)
    try:
        assert state.execute("SELECT count(*) FROM document_projection").fetchone() == (4,)
    finally:
        state.close()


def test_artifact_retry_supersedes_failed_build(
    settings: Settings, canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_documents(canonical, settings.state_db)
    connection = connect_state(settings.state_db)
    rows = connection.execute(
        "SELECT document_id, source_version, text_hash FROM document_projection WHERE is_active"
    ).fetchall()
    connection.executemany(
        """
        INSERT INTO document_embeddings
          (document_id, generation_id, source_version, text_hash, embedding,
           native_dimension, state, attempt_count, updated_at)
        VALUES (?, ?, ?, ?, ?, 768, 'succeeded', 1, current_timestamp)
        """,
        [
            [document_id, settings.generation_id, version, text_hash, [1.0] + [0.0] * 511]
            for document_id, version, text_hash in rows
        ],
    )
    watermark = connection.execute("SELECT source_watermark FROM projection_watermarks").fetchone()[
        0
    ]
    connection.execute(
        """
        INSERT INTO search_builds VALUES
          ('failed-build', ?, ?, 3, 3, '{}', current_timestamp, current_timestamp,
           NULL, '/tmp/failed.duckdb', 'failed', NULL, '0.1.0', 1, 'old failure')
        """,
        [watermark, settings.generation_id],
    )
    connection.close()

    build_artifact(settings)
    connection = connect_state(settings.state_db)
    try:
        assert connection.execute(
            "SELECT status FROM search_builds WHERE build_id = 'failed-build'"
        ).fetchone() == ("superseded",)
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_retry_state_is_resumable(settings: Settings, canonical: Path) -> None:
    project_documents(canonical, settings.state_db)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama"
    ) as http:
        stats = await EmbeddingWorker(settings, OllamaClient(settings, http)).run(max_items=2)
    assert stats.retryable_failed == 2
    connection = connect_state(settings.state_db)
    rows = dict(
        connection.execute(
            "SELECT state, count(*) FROM document_embeddings GROUP BY state"
        ).fetchall()
    )
    connection.close()
    assert rows == {"retryable_failed": 2, "pending": 1}


def test_projection_is_incremental(settings: Settings, canonical: Path) -> None:
    first = project_documents(canonical, settings.state_db)
    second = project_documents(canonical, settings.state_db)
    assert first.watermark == second.watermark
    assert second.projected == 4
