"""Deterministic projection from canonical Slackpipe data."""

from __future__ import annotations

import hashlib
from contextlib import suppress
from pathlib import Path

import duckdb

from slackquery.db import connect_state, path_sql
from slackquery.models import ProjectionStats

CANONICAL_SCHEMA_VERSION = 2
TEXT_RECIPE_VERSION = "message-v1"


def _watermark(connection: duckdb.DuckDBPyConnection) -> str:
    values = connection.execute(
        """
        SELECT count(*), coalesce(max(last_changed_at)::VARCHAR, ''),
               coalesce(max(latest_run_id), ''), coalesce(max(payload_hash), '')
        FROM canonical.messages
        """
    ).fetchone()
    assert values is not None
    return hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()


def project_documents(canonical_path: Path, state_path: Path) -> ProjectionStats:
    """Upsert all canonical messages and tombstone no-longer-active documents."""
    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)
    connection = connect_state(state_path)
    try:
        connection.execute(f"ATTACH {path_sql(canonical_path)} AS canonical (READ_ONLY)")
        version_row = connection.execute(
            "SELECT max(schema_version) FROM canonical.schema_metadata"
        ).fetchone()
        assert version_row is not None
        version = version_row[0]
        if version != CANONICAL_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported canonical schema version: {version}")
        watermark = _watermark(connection)
        now_row = connection.execute("SELECT current_timestamp").fetchone()
        assert now_row is not None
        now = now_row[0]
        connection.execute("BEGIN")
        connection.execute(
            """
            INSERT OR REPLACE INTO document_projection
            WITH canonical_users AS (
              SELECT workspace_id, user_id, any_value(username) AS username,
                     any_value(real_name) AS real_name, any_value(display_name) AS display_name
              FROM canonical.users GROUP BY workspace_id, user_id
            )
            SELECT
              m.message_key, 'message', m.workspace_id, w.team_name, w.workspace_slug,
              w.workspace_url, m.channel_id, c.name, m.user_id,
              coalesce(nullif(u.display_name, ''), nullif(u.real_name, ''), u.username),
              m.ts, m.ts_us, to_timestamp(m.ts_us / 1000000.0),
              CASE WHEN m.thread_ts IS NOT NULL OR m.is_thread_parent
                   THEN m.workspace_id || ':' || m.channel_id || ':' || coalesce(m.thread_ts, m.ts)
                   ELSE NULL END,
              CASE WHEN m.thread_ts IS NOT NULL OR m.is_thread_parent
                   THEN coalesce(root.message_key, m.message_key)
                   ELSE NULL END,
              m.is_thread_parent, m.is_thread_reply, m.is_thread_broadcast,
              coalesce(m.text, ''), m.search_text,
              concat('workspace: ', w.workspace_slug, '\nchannel: ', coalesce(c.name, m.channel_id),
                     '\nauthor: ', coalesce(nullif(u.display_name, ''), nullif(u.real_name, ''),
                                             u.username, m.user_id, 'unknown'),
                     '\nmessage: ', coalesce(m.text, '')),
              m.payload_hash,
              sha256(concat('message-v1', '\x1f', concat('workspace: ', w.workspace_slug,
                     '\nchannel: ', coalesce(c.name, m.channel_id), '\nauthor: ',
                     coalesce(nullif(u.display_name, ''), nullif(u.real_name, ''), u.username,
                              m.user_id, 'unknown'), '\nmessage: ', coalesce(m.text, '')))),
              json_object('subtype', m.subtype, 'edited', m.is_edited),
              CASE WHEN w.workspace_url IS NULL THEN NULL ELSE
                   rtrim(w.workspace_url, '/') || '/archives/' || m.channel_id || '/p' ||
                   replace(m.ts, '.', '') ||
                   CASE WHEN m.is_thread_reply THEN '?thread_ts=' || m.thread_ts ||
                        '&cid=' || m.channel_id ELSE '' END END,
              NOT m.is_deleted AND length(trim(m.search_text)) > 0,
              ?
            FROM canonical.messages m
            JOIN canonical.workspaces w USING (workspace_id)
            LEFT JOIN canonical.channels c USING (workspace_id, channel_id)
            LEFT JOIN canonical_users u
              ON u.workspace_id = m.workspace_id AND u.user_id = m.user_id
            LEFT JOIN canonical.messages root
              ON root.workspace_id = m.workspace_id AND root.channel_id = m.channel_id
             AND root.ts = coalesce(m.thread_ts, m.ts) AND root.is_thread_parent
            """,
            [now],
        )
        connection.execute(
            """
            UPDATE document_projection SET is_active = false, projected_at = ?
            WHERE document_id NOT IN (SELECT message_key FROM canonical.messages)
            """,
            [now],
        )
        active_row = connection.execute(
            "SELECT count(*) FROM document_projection WHERE is_active"
        ).fetchone()
        total_row = connection.execute("SELECT count(*) FROM document_projection").fetchone()
        assert active_row is not None and total_row is not None
        active = active_row[0]
        total = total_row[0]
        connection.execute(
            """
            INSERT OR REPLACE INTO projection_watermarks VALUES (?, ?, ?, ?, ?)
            """,
            [str(canonical_path.resolve()), version, watermark, now, active],
        )
        connection.execute("COMMIT")
        return ProjectionStats(
            projected=total, active=active, tombstoned=total - active, watermark=watermark
        )
    except Exception:
        with suppress(duckdb.TransactionException):
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
