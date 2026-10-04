"""Read-only lexical, exact semantic, and RRF retrieval."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from slackquery.db import connect_readonly
from slackquery.embedding import LocalEmbeddingClient
from slackquery.models import (
    MessageRecord,
    ScopeRecord,
    SearchFilters,
    SearchHit,
    SearchMode,
    SearchResponse,
)
from slackquery.settings import Settings

_FTS_TOKEN = re.compile(r"[\w@./:+#-]+", re.UNICODE)


def _fts_query(value: str) -> str:
    """Create a bounded FTS expression with no SQL metacharacters."""
    tokens = _FTS_TOKEN.findall(value[:2000])[:64]
    if not tokens:
        raise ValueError("query has no searchable tokens")
    return " ".join(tokens)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _filter_sql(filters: SearchFilters, max_values: int) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    fields = {
        "workspace_id": filters.workspace_ids,
        "channel_id": filters.channel_ids,
        "author_id": filters.author_ids,
        "document_kind": filters.document_kinds,
    }
    for field, values in fields.items():
        if len(values) > max_values:
            raise ValueError(f"too many {field} filter values")
        if values:
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"d.{field} IN ({placeholders})")
            parameters.extend(values)
    if filters.start_ts_us is not None:
        clauses.append("d.ts_us >= ?")
        parameters.append(filters.start_ts_us)
    if filters.end_ts_us is not None:
        clauses.append("d.ts_us <= ?")
        parameters.append(filters.end_ts_us)
    return (" AND " + " AND ".join(clauses)) if clauses else "", parameters


def _cursor(artifact_id: str, fingerprint: str, offset: int) -> str:
    payload = json.dumps(
        {"a": artifact_id, "q": fingerprint, "o": offset}, separators=(",", ":")
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(value: str, artifact_id: str, fingerprint: str) -> int:
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        payload = json.loads(raw)
        if payload["a"] != artifact_id or payload["q"] != fingerprint:
            raise ValueError
        offset = int(payload["o"])
    except (ValueError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cursor does not match this artifact/query") from error
    if offset < 0 or offset > 10_000:
        raise ValueError("cursor offset is out of range")
    return offset


class SearchEngine:
    """Artifact-bound, read-only search API."""

    def __init__(
        self, settings: Settings, embedding_client: LocalEmbeddingClient | None = None
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client

    def _artifact(self) -> Path:
        path = self.settings.current_link
        if not path.exists():
            raise RuntimeError("no published search artifact")
        return path.resolve()

    @staticmethod
    def _metadata(connection: duckdb.DuckDBPyConnection) -> tuple[str, str]:
        row = connection.execute(
            "SELECT build_id, generation_id FROM artifact_metadata"
        ).fetchone()
        if row is None:
            raise RuntimeError("artifact metadata missing")
        return row[0], row[1]

    async def search(
        self,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        filters: SearchFilters | None = None,
        limit: int = 10,
        cursor: str | None = None,
    ) -> SearchResponse:
        query = query.strip()
        if not query or len(query) > self.settings.max_query_chars:
            raise ValueError("query length is out of range")
        if limit < 1 or limit > self.settings.max_results:
            raise ValueError(f"limit must be between 1 and {self.settings.max_results}")
        filters = filters or SearchFilters()
        artifact = await asyncio.to_thread(self._artifact)
        vector: list[float] | None = None
        if mode in ("semantic", "hybrid"):
            if self.embedding_client is None:
                raise RuntimeError("semantic search requires an embedding client")
            vector = (await self.embedding_client.embed([query], query=True))[0]
        return await asyncio.to_thread(
            self._search_sync, artifact, query, mode, filters, limit, cursor, vector
        )

    def _search_sync(
        self,
        artifact: Path,
        query: str,
        mode: SearchMode,
        filters: SearchFilters,
        limit: int,
        cursor: str | None,
        vector: list[float] | None,
    ) -> SearchResponse:
        connection = connect_readonly(
            artifact, extension_dir=self.settings.duckdb_extension_dir
        )
        try:
            artifact_id, _ = self._metadata(connection)
            fingerprint = hashlib.sha256(
                json.dumps(
                    {"query": query, "mode": mode, "filters": filters.model_dump()},
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            offset = _decode_cursor(cursor, artifact_id, fingerprint) if cursor else 0
            window = min(self.settings.candidate_window, 1000)
            filter_sql, filter_params = _filter_sql(filters, self.settings.max_filter_values)
            lexical: list[tuple[str, float]] = []
            semantic: list[tuple[str, float]] = []
            if mode in ("lexical", "hybrid"):
                fts_query = _sql_literal(_fts_query(query))
                # DuckDB FTS's generated macro cannot reliably bind its query argument;
                # only the sanitized/quoted value above is interpolated. Filters stay bound.
                rows = connection.execute(
                    f"""
                    SELECT d.document_id,
                      fts_main_search_documents.match_bm25(
                        d.document_id, {fts_query}
                      ) AS score
                    FROM search_documents d
                    WHERE score IS NOT NULL {filter_sql}
                    ORDER BY score DESC, d.document_id
                    LIMIT ?
                    """,
                    [*filter_params, window],
                ).fetchall()
                lexical = [(row[0], float(row[1])) for row in rows]
            if mode in ("semantic", "hybrid"):
                assert vector is not None
                rows = connection.execute(
                    f"""
                    SELECT d.document_id, array_cosine_similarity(v.embedding, ?::FLOAT[512]) score
                    FROM search_documents d JOIN document_vectors v USING (document_id)
                    WHERE true {filter_sql}
                    ORDER BY score DESC, d.document_id
                    LIMIT ?
                    """,
                    [vector, *filter_params, window],
                ).fetchall()
                semantic = [(row[0], float(row[1])) for row in rows]
            lex_rank = {document_id: rank for rank, (document_id, _) in enumerate(lexical, 1)}
            sem_rank = {document_id: rank for rank, (document_id, _) in enumerate(semantic, 1)}
            lex_score = dict(lexical)
            sem_score = dict(semantic)
            document_ids = set(lex_rank) | set(sem_rank)

            def fused(document_id: str) -> float:
                score = 0.0
                if document_id in lex_rank:
                    score += 1.0 / (self.settings.rrf_k + lex_rank[document_id])
                if document_id in sem_rank:
                    score += 1.0 / (self.settings.rrf_k + sem_rank[document_id])
                return score

            ordered = sorted(document_ids, key=lambda item: (-fused(item), item))
            page = ordered[offset : offset + limit]
            records: dict[str, tuple[Any, ...]] = {}
            if page:
                placeholders = ", ".join("?" for _ in page)
                for row in connection.execute(
                    f"""
                    SELECT document_id, document_kind, workspace_id, workspace_name,
                      workspace_slug, channel_id, channel_name, author_id, author_name,
                      ts_us, timestamp, thread_id, thread_root_id, text_display, permalink
                    FROM search_documents WHERE document_id IN ({placeholders})
                    """,
                    page,
                ).fetchall():
                    records[row[0]] = row
            results = [
                SearchHit(
                    document_id=records[item][0], document_kind=records[item][1],
                    workspace_id=records[item][2], workspace_name=records[item][3],
                    workspace_slug=records[item][4], channel_id=records[item][5],
                    channel_name=records[item][6], author_id=records[item][7],
                    author_name=records[item][8], ts_us=records[item][9],
                    timestamp=records[item][10], thread_id=records[item][11],
                    thread_root_id=records[item][12], text=records[item][13],
                    permalink=records[item][14], lexical_rank=lex_rank.get(item),
                    lexical_score=lex_score.get(item), semantic_rank=sem_rank.get(item),
                    semantic_score=sem_score.get(item), fused_score=fused(item),
                )
                for item in page
            ]
            next_cursor = None
            if offset + limit < len(ordered):
                next_cursor = _cursor(artifact_id, fingerprint, offset + limit)
            return SearchResponse(
                query=query, mode=mode, artifact_id=artifact_id,
                results=results, next_cursor=next_cursor,
            )
        finally:
            connection.close()

    def get_message(
        self, document_id: str, *, before: int = 0, after: int = 0
    ) -> list[MessageRecord]:
        if before < 0 or after < 0 or before + after > 100:
            raise ValueError("context bounds must be non-negative and total at most 100")
        connection = connect_readonly(
            self._artifact(), extension_dir=self.settings.duckdb_extension_dir
        )
        try:
            target = connection.execute(
                """
                SELECT workspace_id, channel_id, ts_us
                FROM search_documents WHERE document_id = ?
                """,
                [document_id],
            ).fetchone()
            if target is None:
                return []
            rows = connection.execute(
                """
                WITH target AS (SELECT ?::BIGINT AS ts_us), context AS (
                  (SELECT * FROM search_documents
                   WHERE workspace_id = ? AND channel_id = ? AND ts_us < (SELECT ts_us FROM target)
                   ORDER BY ts_us DESC LIMIT ?)
                  UNION ALL
                  (SELECT * FROM search_documents WHERE document_id = ?)
                  UNION ALL
                  (SELECT * FROM search_documents
                   WHERE workspace_id = ? AND channel_id = ? AND ts_us > (SELECT ts_us FROM target)
                   ORDER BY ts_us LIMIT ?)
                )
                SELECT document_id, workspace_id, workspace_slug, channel_id, channel_name,
                  author_id, author_name, ts_us, timestamp, thread_id, thread_root_id,
                  text_display, permalink FROM context ORDER BY ts_us
                """,
                [target[2], target[0], target[1], before, document_id,
                 target[0], target[1], after],
            ).fetchall()
            return [
                MessageRecord.model_validate(
                    dict(zip(MessageRecord.model_fields, row, strict=True))
                )
                for row in rows
            ]
        finally:
            connection.close()

    def get_thread(self, thread_id: str, *, limit: int = 200) -> list[MessageRecord]:
        if limit < 1 or limit > 500:
            raise ValueError("thread limit must be between 1 and 500")
        connection = connect_readonly(
            self._artifact(), extension_dir=self.settings.duckdb_extension_dir
        )
        try:
            rows = connection.execute(
                """
                SELECT document_id, workspace_id, workspace_slug, channel_id, channel_name,
                  author_id, author_name, ts_us, timestamp, thread_id, thread_root_id,
                  text_display, permalink FROM search_documents
                WHERE thread_id = ? ORDER BY ts_us LIMIT ?
                """,
                [thread_id, limit],
            ).fetchall()
            return [
                MessageRecord.model_validate(
                    dict(zip(MessageRecord.model_fields, row, strict=True))
                )
                for row in rows
            ]
        finally:
            connection.close()

    def list_scopes(
        self, workspace_id: str | None = None, channel_name: str | None = None,
        *, limit: int = 200,
    ) -> list[ScopeRecord]:
        if limit < 1 or limit > 500:
            raise ValueError("scope limit must be between 1 and 500")
        clauses: list[str] = []
        params: list[Any] = []
        if workspace_id:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if channel_name:
            clauses.append("lower(channel_name) = lower(?)")
            params.append(channel_name)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        connection = connect_readonly(
            self._artifact(), extension_dir=self.settings.duckdb_extension_dir
        )
        try:
            rows = connection.execute(
                f"""
                SELECT workspace_id, workspace_slug, workspace_name, channel_id,
                  channel_name, count(*), min(ts_us), max(ts_us)
                FROM search_documents {where}
                GROUP BY ALL ORDER BY workspace_slug, channel_name, channel_id LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
            return [
                ScopeRecord.model_validate(
                    dict(zip(ScopeRecord.model_fields, row, strict=True))
                )
                for row in rows
            ]
        finally:
            connection.close()

    def readiness(self) -> dict[str, Any]:
        try:
            connection = connect_readonly(
                self._artifact(), extension_dir=self.settings.duckdb_extension_dir
            )
            artifact_id, generation_id = self._metadata(connection)
            count_row = connection.execute("SELECT count(*) FROM search_documents").fetchone()
            assert count_row is not None
            count = count_row[0]
            connection.close()
            return {"ready": True, "artifact_id": artifact_id,
                    "generation_id": generation_id, "documents": count}
        except (OSError, RuntimeError, duckdb.Error) as error:
            return {"ready": False, "error": str(error)}


def timestamp_us(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1_000_000)
