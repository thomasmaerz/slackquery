"""Immutable search artifact build, validation, and publication."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from slackquery import __version__
from slackquery.db import SQL_ROOT, connect_readonly, connect_state, path_sql, sha256_file
from slackquery.models import BuildResult
from slackquery.settings import Settings

ARTIFACT_SCHEMA_VERSION = 1
FTS_CONFIG = {"stemmer": "none", "stopwords": "none", "lower": True, "strip_accents": True}


def _build_id(watermark: str, generation_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{watermark[:8]}-{uuid.uuid4().hex[:8]}"


def _manifest_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(".manifest.json")


def build_artifact(settings: Settings, *, dagster_run_id: str | None = None) -> BuildResult:
    """Build, index, validate, checkpoint, and seal an immutable artifact."""
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    state = connect_state(settings.state_db)
    row = state.execute(
        "SELECT source_watermark FROM projection_watermarks ORDER BY projected_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        state.close()
        raise RuntimeError("no projection watermark; run project first")
    watermark = row[0]
    counts = state.execute(
        """
        SELECT count(*), count(e.embedding)
        FROM document_projection p
        LEFT JOIN document_embeddings e ON e.document_id = p.document_id
          AND e.source_version = p.source_version AND e.text_hash = p.text_hash
          AND e.generation_id = ? AND e.state = 'succeeded'
        WHERE p.is_active
        """,
        [settings.generation_id],
    ).fetchone()
    assert counts is not None
    document_count, vector_count = counts
    if document_count == 0:
        state.close()
        raise RuntimeError("cannot build an empty artifact")
    if vector_count != document_count:
        state.close()
        raise RuntimeError(f"embedding coverage incomplete: {vector_count}/{document_count}")
    build_id = _build_id(watermark, settings.generation_id)
    tmp_path = settings.artifact_dir / f"search-{build_id}.duckdb.tmp"
    final_path = settings.artifact_dir / f"search-{build_id}.duckdb"
    manifest_path = _manifest_path(final_path)
    state.execute(
        """
        UPDATE search_builds SET status = 'superseded'
        WHERE source_watermark = ? AND generation_id = ? AND status = 'failed'
        """,
        [watermark, settings.generation_id],
    )
    state.execute(
        """
        INSERT INTO search_builds VALUES (?, ?, ?, ?, ?, ?, current_timestamp, NULL,
          NULL, ?, 'building', ?, ?, ?, NULL)
        """,
        [build_id, watermark, settings.generation_id, document_count, vector_count,
         json.dumps(FTS_CONFIG, sort_keys=True), str(final_path), dagster_run_id,
         __version__, ARTIFACT_SCHEMA_VERSION],
    )
    state.close()
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        settings.duckdb_extension_dir.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(
            str(tmp_path),
            config={"extension_directory": str(settings.duckdb_extension_dir)},
        )
        connection.execute("INSTALL fts")
        connection.execute("LOAD fts")
        connection.execute((SQL_ROOT / "001_artifact.sql").read_text())
        connection.execute(f"ATTACH {path_sql(settings.state_db)} AS state (READ_ONLY)")
        connection.execute(
            """
            INSERT INTO search_documents
            SELECT document_id, document_kind, workspace_id, workspace_name,
              workspace_slug, channel_id, channel_name, author_id, author_name,
              ts_us, timestamp, thread_id, thread_root_id, is_thread_parent,
              is_thread_reply, is_thread_broadcast, text_display, text_lexical,
              text_embedding, source_version, metadata_json, permalink
            FROM state.document_projection WHERE is_active ORDER BY document_id
            """
        )
        connection.execute(
            """
            INSERT INTO document_vectors
            SELECT e.document_id, e.generation_id, e.embedding
            FROM state.document_embeddings e
            JOIN state.document_projection p USING (document_id, source_version, text_hash)
            WHERE p.is_active AND e.generation_id = ? AND e.state = 'succeeded'
            ORDER BY e.document_id
            """,
            [settings.generation_id],
        )
        connection.execute(
            """
            INSERT INTO artifact_metadata VALUES
              (?, ?, ?, ?, ?, ?, current_timestamp, ?)
            """,
            [build_id, ARTIFACT_SCHEMA_VERSION, settings.generation_id, watermark,
             document_count, vector_count, json.dumps(FTS_CONFIG, sort_keys=True)],
        )
        # PRAGMA has known prepared-statement restrictions; all values are constants.
        connection.execute(
            "PRAGMA create_fts_index('search_documents', 'document_id', 'text_lexical', "
            "stemmer='none', stopwords='none', lower=1, strip_accents=1, overwrite=1)"
        )
        connection.execute("CHECKPOINT")
        connection.close()
        connection = None
        validate_artifact(tmp_path)
        os.replace(tmp_path, final_path)
        final_path.chmod(0o444)
        checksum = sha256_file(final_path)
        manifest: dict[str, Any] = {
            "artifact_id": build_id,
            "artifact_file": final_path.name,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "generation_id": settings.generation_id,
            "source_watermark": watermark,
            "document_count": document_count,
            "vector_count": vector_count,
            "checksum_sha256": checksum,
            "fts_config": FTS_CONFIG,
            "built_at": datetime.now(UTC).isoformat(),
            "code_version": __version__,
        }
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_manifest, manifest_path)
        state = connect_state(settings.state_db)
        state.execute(
            """
            UPDATE search_builds SET completed_at = current_timestamp, checksum = ?,
              status = 'validated' WHERE build_id = ?
            """,
            [checksum, build_id],
        )
        state.close()
        return BuildResult(
            build_id=build_id,
            artifact_path=str(final_path),
            manifest_path=str(manifest_path),
            checksum=checksum,
            document_count=document_count,
        )
    except Exception as error:
        if connection is not None:
            connection.close()
        tmp_path.unlink(missing_ok=True)
        state = connect_state(settings.state_db)
        state.execute(
            "UPDATE search_builds SET status = 'failed', error_message = ? WHERE build_id = ?",
            [str(error)[:1000], build_id],
        )
        state.close()
        raise


def validate_artifact(path: Path, *, verify_checksum: bool = False) -> dict[str, Any]:
    """Run structural and data quality checks through a fresh read-only connection."""
    connection = connect_readonly(path)
    try:
        metadata = connection.execute("SELECT * FROM artifact_metadata").fetchone()
        if metadata is None:
            raise RuntimeError("artifact metadata missing")
        document_row = connection.execute("SELECT count(*) FROM search_documents").fetchone()
        vector_row = connection.execute("SELECT count(*) FROM document_vectors").fetchone()
        duplicate_row = connection.execute(
            "SELECT count(*) - count(DISTINCT document_id) FROM search_documents"
        ).fetchone()
        missing_row = connection.execute(
            """
            SELECT count(*) FROM search_documents d
            FULL OUTER JOIN document_vectors v USING (document_id)
            WHERE d.document_id IS NULL OR v.document_id IS NULL
            """
        ).fetchone()
        invalid_row = connection.execute(
            """
            SELECT count(*) FROM document_vectors
            WHERE len(embedding) <> 512
               OR abs(array_cosine_similarity(embedding, embedding) - 1) > 1e-5
            """
        ).fetchone()
        assert document_row and vector_row and duplicate_row and missing_row and invalid_row
        document_count = document_row[0]
        vector_count = vector_row[0]
        duplicate_count = duplicate_row[0]
        missing = missing_row[0]
        invalid_vectors = invalid_row[0]
        if duplicate_count or missing or invalid_vectors:
            raise RuntimeError(
                f"artifact validation failed: duplicates={duplicate_count}, "
                f"key_mismatch={missing}, invalid_vectors={invalid_vectors}"
            )
        if document_count != metadata[4] or vector_count != metadata[5]:
            raise RuntimeError("artifact metadata row counts disagree")
        result = {
            "artifact_id": metadata[0],
            "document_count": document_count,
            "vector_count": vector_count,
            "valid": True,
        }
    finally:
        connection.close()
    if verify_checksum:
        manifest = json.loads(_manifest_path(path).read_text())
        if sha256_file(path) != manifest["checksum_sha256"]:
            raise RuntimeError("artifact checksum mismatch")
    return result


def publish_artifact(settings: Settings, artifact_path: Path) -> Path:
    """Atomically replace the current symlink after full validation."""
    artifact_path = artifact_path.resolve()
    if artifact_path.parent != settings.artifact_dir.resolve():
        raise ValueError("artifact must be in artifact_dir")
    validate_artifact(artifact_path, verify_checksum=True)
    settings.current_link.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.current_link.with_name(f".{settings.current_link.name}.{uuid.uuid4().hex}")
    temporary.symlink_to(artifact_path.name)
    os.replace(temporary, settings.current_link)
    # A directory fsync makes the atomic link replacement durable across power loss.
    directory_fd = os.open(settings.current_link.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    state = connect_state(settings.state_db)
    state.execute(
        "UPDATE search_builds SET status = 'published' WHERE artifact_path = ?",
        [str(artifact_path)],
    )
    state.close()
    prune_artifacts(settings)
    return settings.current_link


def prune_artifacts(settings: Settings) -> None:
    current = settings.current_link.resolve() if settings.current_link.exists() else None
    artifacts = sorted(
        settings.artifact_dir.glob("search-*.duckdb"), key=lambda p: p.stat().st_mtime
    )
    removable = [item for item in artifacts if item.resolve() != current]
    while len(artifacts) > settings.retain_artifacts and removable:
        victim = removable.pop(0)
        victim.unlink(missing_ok=True)
        _manifest_path(victim).unlink(missing_ok=True)
        artifacts.remove(victim)


def copy_artifact(source: Path, destination: Path) -> None:
    """Explicit helper for operators restoring a known-good immutable artifact."""
    shutil.copy2(source, destination)
