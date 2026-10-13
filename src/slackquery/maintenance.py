"""Audited state migrations and conservative embedding garbage collection."""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import Any

import duckdb

from slackquery.embedding_identity import DOCUMENT_PREFIX
from slackquery.projection import FILE_RECIPE_VERSION, MESSAGE_RECIPE_VERSION
from slackquery.settings import Settings


def _exclusive_connection(path: Path) -> tuple[duckdb.DuckDBPyConnection, Any]:
    lock_path = path.with_suffix(path.suffix + ".maintenance.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        connection = duckdb.connect(str(path))
        connection.execute("SET threads=1")
    except Exception:
        lock.close()
        raise
    return connection, lock


def migrate_embedding_content_hashes(
    settings: Settings, *, apply: bool = False
) -> dict[str, int]:
    """Copy current vectors from payload hashes to exact model-input hashes.

    Historical vectors are not relabeled because their original model input is
    not retained. The operation is idempotent and can run before or after the
    stable-hash projection change.
    """
    connection, lock = _exclusive_connection(settings.state_db)
    try:
        generation = settings.generation_id
        counts = connection.execute(
            """
            WITH eligible AS (
              SELECT p.*, sha256(? || p.text_embedding) AS target_hash
              FROM document_projection p
              WHERE p.is_active AND p.document_kind IN ('message', 'file_chunk')
            )
            SELECT count(*),
              count(*) FILTER (WHERE NOT EXISTS (
                                  SELECT 1 FROM document_embeddings e
                                  WHERE e.document_id=eligible.document_id
                                    AND e.source_version=eligible.source_version
                                    AND (e.text_hash=eligible.text_hash
                                         OR (eligible.document_kind='message'
                                             AND e.text_hash=eligible.source_revision)
                                         OR e.text_hash=eligible.target_hash)
                                    AND e.generation_id=? AND e.state='succeeded'
                                    AND e.embedding IS NOT NULL
                                )),
              count(*) FILTER (WHERE eligible.text_hash <> eligible.target_hash
                                AND EXISTS (
                                  SELECT 1 FROM document_embeddings e
                                  WHERE e.document_id=eligible.document_id
                                    AND e.source_version=eligible.source_version
                                    AND (e.text_hash=eligible.text_hash
                                         OR (eligible.document_kind='message'
                                             AND e.text_hash=eligible.source_revision))
                                    AND e.generation_id=? AND e.state='succeeded'
                                    AND e.embedding IS NOT NULL
                                )
                                AND NOT EXISTS (
                                  SELECT 1 FROM document_embeddings e
                                  WHERE e.document_id=eligible.document_id
                                    AND e.source_version=eligible.source_version
                                    AND e.text_hash=eligible.target_hash
                                    AND e.generation_id=? AND e.state='succeeded'
                                    AND e.embedding IS NOT NULL
                                ))
            FROM eligible
            """,
            [DOCUMENT_PREFIX, generation, generation, generation],
        ).fetchone()
        assert counts is not None
        eligible, missing, migrate = (int(value) for value in counts)
        result = {"eligible": eligible, "missing": missing, "migrated": migrate, "applied": 0}
        if not apply:
            return result
        if missing:
            raise RuntimeError(f"cannot migrate with {missing} active embedding(s) missing")

        connection.execute("BEGIN")
        try:
            connection.execute(
                """
                INSERT INTO document_embeddings
                  (document_id, generation_id, source_version, text_hash, embedding,
                   native_dimension, request_id, state, attempt_count, embedded_at,
                   updated_at)
                SELECT e.document_id, e.generation_id, e.source_version,
                  sha256(? || p.text_embedding), e.embedding, e.native_dimension,
                  e.request_id, 'succeeded', e.attempt_count, e.embedded_at,
                  current_timestamp
                FROM document_projection p
                JOIN document_embeddings e
                  ON e.document_id=p.document_id AND e.source_version=p.source_version
                 AND (e.text_hash=p.text_hash
                      OR (p.document_kind='message' AND e.text_hash=p.source_revision))
                WHERE p.is_active AND p.document_kind IN ('message', 'file_chunk')
                  AND e.generation_id=? AND e.state='succeeded' AND e.embedding IS NOT NULL
                  AND p.text_hash <> sha256(? || p.text_embedding)
                ON CONFLICT DO NOTHING
                """,
                [DOCUMENT_PREFIX, generation, DOCUMENT_PREFIX],
            )
            connection.execute(
                """
                UPDATE document_projection
                SET source_revision=CASE
                      WHEN document_kind='message' THEN text_hash ELSE source_revision END,
                    text_hash=sha256(? || text_embedding),
                    embedding_recipe=CASE
                      WHEN document_kind='message' THEN ? ELSE ? END
                WHERE is_active AND document_kind IN ('message', 'file_chunk')
                """,
                [DOCUMENT_PREFIX, MESSAGE_RECIPE_VERSION, FILE_RECIPE_VERSION],
            )
            covered = connection.execute(
                """
                SELECT count(*) FROM document_projection p
                JOIN document_embeddings e USING (document_id, source_version, text_hash)
                WHERE p.is_active AND p.document_kind IN ('message', 'file_chunk')
                  AND e.generation_id=? AND e.state='succeeded' AND e.embedding IS NOT NULL
                """,
                [generation],
            ).fetchone()
            assert covered is not None
            if int(covered[0]) != eligible:
                raise RuntimeError(
                    f"post-migration coverage mismatch: {covered[0]}/{eligible}"
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        result["applied"] = migrate
        return result
    finally:
        connection.close()
        lock.close()


def prune_embeddings(settings: Settings, *, apply: bool = False) -> dict[str, int]:
    """Delete current-generation vectors not used by publishable documents."""
    connection, lock = _exclusive_connection(settings.state_db)
    try:
        generation = settings.generation_id
        coverage = connection.execute(
            """
            SELECT count(*), count(e.document_id),
              count(*) FILTER (WHERE e.embedding IS NULL)
            FROM document_projection p
            LEFT JOIN document_embeddings e
              ON e.document_id=p.document_id AND e.source_version=p.source_version
             AND e.text_hash=p.text_hash AND e.generation_id=? AND e.state='succeeded'
            WHERE p.is_active AND p.document_kind IN ('message', 'file_chunk')
            """,
            [generation],
        ).fetchone()
        assert coverage is not None
        eligible, covered, invalid = (int(value) for value in coverage)
        if covered != eligible or invalid:
            raise RuntimeError(
                f"refusing garbage collection: coverage={covered}/{eligible}, invalid={invalid}"
            )
        stale = connection.execute(
            """
            SELECT count(*) FROM document_embeddings e
            WHERE e.generation_id=? AND NOT EXISTS (
              SELECT 1 FROM document_projection p
              WHERE p.is_active AND p.document_kind IN ('message', 'file_chunk')
                AND p.document_id=e.document_id AND p.source_version=e.source_version
                AND p.text_hash=e.text_hash
            )
            """,
            [generation],
        ).fetchone()
        assert stale is not None
        result = {
            "eligible": eligible,
            "covered": covered,
            "stale": int(stale[0]),
            "deleted": 0,
        }
        if not apply:
            return result
        connection.execute("BEGIN")
        try:
            connection.execute(
                """
                DELETE FROM document_embeddings e
                WHERE e.generation_id=? AND NOT EXISTS (
                  SELECT 1 FROM document_projection p
                  WHERE p.is_active AND p.document_kind IN ('message', 'file_chunk')
                    AND p.document_id=e.document_id AND p.source_version=e.source_version
                    AND p.text_hash=e.text_hash
                )
                """,
                [generation],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        connection.execute("CHECKPOINT")
        result["deleted"] = result["stale"]
        return result
    finally:
        connection.close()
        lock.close()
