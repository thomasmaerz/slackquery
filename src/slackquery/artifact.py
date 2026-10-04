"""Immutable search artifact build, validation, and publication."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from slackquery import __version__
from slackquery.db import SQL_ROOT, connect_readonly, connect_state, path_sql, sha256_file
from slackquery.models import BuildResult
from slackquery.settings import Settings

ARTIFACT_SCHEMA_VERSION = 2
FTS_CONFIG = {"stemmer": "none", "stopwords": "none", "lower": True, "strip_accents": True}


def _build_id(watermark: str, generation_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{watermark[:8]}-{uuid.uuid4().hex[:8]}"


def _manifest_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(".manifest.json")


def _lock_state(settings: Settings) -> Any:
    path = settings.state_db.with_suffix(settings.state_db.suffix + ".maintenance.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise RuntimeError("state database is in use") from error
    return stream


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        SELECT count(*) FILTER (WHERE p.document_kind <> 'thread_context'),
               count(e.embedding) FILTER (WHERE p.document_kind <> 'thread_context')
        FROM document_projection p
        LEFT JOIN document_embeddings e ON e.document_id = p.document_id
          AND e.source_version = p.source_version AND e.text_hash = p.text_hash
          AND e.generation_id = ? AND e.state = 'succeeded'
        WHERE p.is_active
        """,
        [settings.generation_id],
    ).fetchone()
    assert counts is not None
    required_vector_count, vector_count = counts
    document_count_row = state.execute(
        "SELECT count(*) FROM document_projection WHERE is_active"
    ).fetchone()
    assert document_count_row is not None
    document_count = document_count_row[0]
    if document_count == 0:
        state.close()
        raise RuntimeError("cannot build an empty artifact")
    if vector_count != required_vector_count:
        state.close()
        excluded = state.execute(
            """
            SELECT document_kind, count(*) FROM document_projection p
            LEFT JOIN document_embeddings e ON e.document_id=p.document_id
              AND e.source_version=p.source_version AND e.text_hash=p.text_hash
              AND e.generation_id=? AND e.state='succeeded'
            WHERE p.is_active AND p.document_kind <> 'thread_context' AND e.document_id IS NULL
            GROUP BY document_kind ORDER BY document_kind
            """,
            [settings.generation_id],
        ).fetchall()
        raise RuntimeError(
            f"embedding coverage incomplete: {vector_count}/{required_vector_count}; "
            f"missing_by_kind={dict(excluded)}"
        )
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
        [
            build_id,
            watermark,
            settings.generation_id,
            document_count,
            vector_count,
            json.dumps(FTS_CONFIG, sort_keys=True),
            str(final_path),
            dagster_run_id,
            __version__,
            ARTIFACT_SCHEMA_VERSION,
        ],
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
            WHERE p.is_active AND p.document_kind <> 'thread_context'
              AND e.generation_id = ? AND e.state = 'succeeded'
            ORDER BY e.document_id
            """,
            [settings.generation_id],
        )
        connection.execute(
            """
            INSERT INTO artifact_metadata VALUES
              (?, ?, ?, ?, ?, ?, current_timestamp, ?)
            """,
            [
                build_id,
                ARTIFACT_SCHEMA_VERSION,
                settings.generation_id,
                watermark,
                document_count,
                vector_count,
                json.dumps(FTS_CONFIG, sort_keys=True),
            ],
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
        if metadata[1] != ARTIFACT_SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported artifact schema version: {metadata[1]} "
                f"(expected {ARTIFACT_SCHEMA_VERSION})"
            )
        document_row = connection.execute("SELECT count(*) FROM search_documents").fetchone()
        vector_row = connection.execute("SELECT count(*) FROM document_vectors").fetchone()
        duplicate_row = connection.execute(
            "SELECT count(*) - count(DISTINCT document_id) FROM search_documents"
        ).fetchone()
        missing_row = connection.execute(
            """
            SELECT count(*) FROM search_documents d
            FULL OUTER JOIN document_vectors v USING (document_id)
            WHERE d.document_id IS NULL
               OR (v.document_id IS NULL AND d.document_kind <> 'thread_context')
               OR (v.document_id IS NOT NULL AND d.document_kind = 'thread_context')
            """
        ).fetchone()
        invalid_row = connection.execute(
            """
            SELECT count(*) FROM document_vectors
            WHERE len(embedding) <> 512
               OR NOT isfinite(array_inner_product(embedding, embedding))
               OR abs(sqrt(array_inner_product(embedding, embedding)) - 1) > 1e-5
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
        expected_manifest = {
            "artifact_id": result["artifact_id"],
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "document_count": result["document_count"],
            "vector_count": result["vector_count"],
        }
        if "artifact_file" in manifest:
            expected_manifest["artifact_file"] = path.name
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected_manifest.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"artifact manifest mismatch: {mismatches}")
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
    return settings.current_link


def retain_artifacts(settings: Settings, *, now: float | None = None) -> dict[str, Any]:
    """Keep current, N newest artifacts, and artifacts inside the minimum-age window."""
    current = settings.current_link.resolve() if settings.current_link.exists() else None
    artifacts = sorted(
        settings.artifact_dir.glob("search-*.duckdb"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    timestamp = time.time() if now is None else now
    keep = set(artifacts[: settings.retain_artifacts])
    if current is not None:
        keep.add(current)
    keep.update(
        item
        for item in artifacts
        if timestamp - item.stat().st_mtime < settings.retention_min_age_seconds
    )
    removed: list[str] = []
    for artifact in artifacts:
        if artifact not in keep:
            artifact.unlink(missing_ok=True)
            _manifest_path(artifact).unlink(missing_ok=True)
            removed.append(str(artifact))
    return {"kept": [str(item) for item in artifacts if item in keep], "removed": removed}


def prune_artifacts(settings: Settings) -> None:
    """Backward-compatible retention wrapper."""
    retain_artifacts(settings)


def rollback_artifact(settings: Settings, build_id_or_path: str) -> Path:
    candidate = Path(build_id_or_path)
    if not candidate.is_absolute():
        candidate = settings.artifact_dir / f"search-{build_id_or_path}.duckdb"
    candidate = candidate.resolve()
    if candidate.parent != settings.artifact_dir.resolve():
        raise ValueError("rollback artifact must be in artifact_dir")
    checks = validate_artifact(candidate, verify_checksum=True)
    if checks["artifact_id"] != candidate.stem.removeprefix("search-"):
        raise ValueError("rollback build ID does not match artifact metadata")
    return publish_artifact(settings, candidate)


def backup_state(settings: Settings, destination: Path) -> dict[str, Any]:
    lock = _lock_state(settings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = duckdb.connect(str(settings.state_db))
        try:
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(settings.state_db, temporary)
        _fsync_path(temporary)
        validation = connect_readonly(temporary)
        try:
            table_count = validation.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema='main'"
            ).fetchone()
            assert table_count is not None
        finally:
            validation.close()
        previous = destination.with_suffix(destination.suffix + ".previous")
        if destination.exists():
            os.replace(destination, previous)
        os.replace(temporary, destination)
        _fsync_path(destination.parent)
        return {
            "backup_path": str(destination),
            "checksum": sha256_file(destination),
            "table_count": table_count[0],
        }
    finally:
        lock.close()


def restore_state(settings: Settings, backup: Path) -> dict[str, Any]:
    lock = _lock_state(settings)
    validation = connect_readonly(backup.resolve())
    try:
        required = validation.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name IN "
            "('document_projection','document_embeddings','search_builds')"
        ).fetchone()
        assert required is not None
    finally:
        validation.close()
    try:
        if required[0] != 3:
            raise RuntimeError("backup does not contain required state tables")
        if settings.state_db.exists():
            try:
                exclusive = duckdb.connect(str(settings.state_db))
                exclusive.execute("CHECKPOINT")
                exclusive.close()
            except duckdb.Error as error:
                raise RuntimeError("state database is in use") from error
        settings.state_db.parent.mkdir(parents=True, exist_ok=True)
        temporary = settings.state_db.with_suffix(settings.state_db.suffix + ".restore.tmp")
        shutil.copy2(backup, temporary)
        _fsync_path(temporary)
        verify = connect_readonly(temporary)
        verify.close()
        previous = settings.state_db.with_suffix(settings.state_db.suffix + ".previous")
        if settings.state_db.exists():
            os.replace(settings.state_db, previous)
        os.replace(temporary, settings.state_db)
        _fsync_path(settings.state_db.parent)
        return {
            "restored_path": str(settings.state_db),
            "checksum": sha256_file(settings.state_db),
            "valid": True,
        }
    finally:
        lock.close()


def gold_validate(settings: Settings, artifact_path: Path | None = None) -> dict[str, Any]:
    """Run structural and deterministic Gold invariants without relevance claims."""
    from slackquery.retrieval import route_query, weighted_rrf

    artifact = artifact_path or settings.current_link.resolve()
    artifact = artifact.resolve()
    base = validate_artifact(artifact, verify_checksum=True)
    connection = connect_readonly(artifact, extension_dir=settings.duckdb_extension_dir)
    try:
        kinds = dict(
            connection.execute(
                "SELECT document_kind, count(*) FROM search_documents "
                "GROUP BY document_kind ORDER BY document_kind"
            ).fetchall()
        )
        vector = connection.execute(
            "SELECT count(*), min(array_length(embedding)), "
            "max(array_length(embedding)), "
            "min(sqrt(array_inner_product(embedding, embedding))), "
            "max(sqrt(array_inner_product(embedding, embedding))) "
            "FROM document_vectors"
        ).fetchone()
        fts_keys = connection.execute(
            """
            SELECT count(*) FROM search_documents d
            FULL OUTER JOIN fts_main_search_documents.docs f ON f.name=d.document_id
            WHERE d.document_id IS NULL OR f.name IS NULL
            """
        ).fetchone()
        artifact_keys = {
            row[0] for row in connection.execute(
                "SELECT document_id FROM search_documents"
            ).fetchall()
        }
        assert vector is not None and fts_keys is not None
    finally:
        connection.close()
    state = connect_state(settings.state_db)
    try:
        active_keys = {
            row[0]
            for row in state.execute(
                "SELECT document_id FROM document_projection WHERE is_active"
            ).fetchall()
        }
        report_row = state.execute(
            "SELECT extraction_json FROM projection_reports ORDER BY projected_at DESC LIMIT 1"
        ).fetchone()
    finally:
        state.close()
    extraction = json.loads(report_row[0]) if report_row else {}
    rankings = {"lexical": ["a", "b"], "semantic": ["b", "a"], "context": []}
    first = weighted_rrf(rankings, {"lexical": 1.0, "semantic": 0.5, "context": 0.0}, 60)
    checks = {
        "artifact": base["valid"],
        "fts_key_coverage": fts_keys[0] == 0,
        "active_key_equality": artifact_keys == active_keys,
        "no_stale_keys": not (artifact_keys - active_keys),
        "file_presence": extraction.get("extracted", 0) == 0
        or kinds.get("file_chunk", 0) > 0,
        "thread_presence": extraction.get("thread_contexts", 0) == 0
        or kinds.get("thread_context", 0) > 0,
        "vector_coverage": vector[0]
        == base["document_count"] - kinds.get("thread_context", 0),
        "vector_dimension": vector[1] == vector[2] == 512,
        "vector_norm": vector[3] is None
        or (abs(vector[3] - 1.0) <= 1e-5 and abs(vector[4] - 1.0) <= 1e-5),
        "routing": [route_query("ERR-42"), route_query("how did the incident unfold over time")]
        == ["exact", "conceptual"],
        "rrf_deterministic": first
        == weighted_rrf(rankings, {"lexical": 1.0, "semantic": 0.5, "context": 0.0}, 60),
        "readonly_open": True,
        # The canonical archive is attached READ_ONLY during projection and is
        # never opened by artifact validation. This check records that contract.
        "canonical_read_only": True,
    }
    return {
        **base,
        "valid": all(checks.values()),
        "checks": checks,
        "document_kinds": kinds,
        "extraction": extraction,
    }


def copy_artifact(source: Path, destination: Path) -> None:
    """Explicit helper for operators restoring a known-good immutable artifact."""
    shutil.copy2(source, destination)
