from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from slackquery.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    artifacts = tmp_path / "artifacts"
    return Settings(
        _env_file=None,
        canonical_db=tmp_path / "canonical.duckdb",
        state_db=tmp_path / "state.duckdb",
        artifact_dir=artifacts,
        current_link=artifacts / "current.duckdb",
        attachment_root=tmp_path / "attachments",
        duckdb_extension_dir=tmp_path / "extensions",
        embedding_backend="ollama",
        embedding_batch_size=2,
        embedding_max_chars=10_000,
        embedding_model_revision="test",
        embedding_verify_model=False,
    )


@pytest.fixture
def canonical(settings: Settings) -> Path:
    connection = duckdb.connect(str(settings.canonical_db))
    connection.execute(
        """
        CREATE TABLE schema_metadata(schema_version INTEGER, installed_at TIMESTAMPTZ);
        INSERT INTO schema_metadata VALUES (2, current_timestamp);
        CREATE TABLE workspaces(
          workspace_id VARCHAR, workspace_slug VARCHAR, team_name VARCHAR,
          workspace_url VARCHAR
        );
        CREATE TABLE channels(workspace_id VARCHAR, channel_id VARCHAR, name VARCHAR);
        CREATE TABLE users(
          workspace_id VARCHAR, user_id VARCHAR, username VARCHAR,
          real_name VARCHAR, display_name VARCHAR
        );
        CREATE TABLE messages(
          message_key VARCHAR, workspace_id VARCHAR, channel_id VARCHAR, ts VARCHAR,
          ts_us BIGINT, user_id VARCHAR, subtype VARCHAR, text VARCHAR,
          search_text VARCHAR, thread_ts VARCHAR, is_thread_parent BOOLEAN,
          is_thread_reply BOOLEAN, is_thread_broadcast BOOLEAN, is_edited BOOLEAN,
          is_deleted BOOLEAN, payload_hash VARCHAR, latest_run_id VARCHAR,
          last_changed_at TIMESTAMPTZ
        );
        INSERT INTO workspaces VALUES ('W1', 'acme', 'Acme', 'https://acme.slack.com');
        INSERT INTO channels VALUES ('W1', 'C1', 'incidents'), ('W1', 'C2', 'general');
        INSERT INTO users VALUES
          ('W1', 'U1', 'ada', 'Ada Lovelace', 'Ada'),
          ('W1', 'U2', 'grace', 'Grace Hopper', 'Grace');
        INSERT INTO messages VALUES
          ('W1:C1:1.000001', 'W1', 'C1', '1.000001', 1000001, 'U1', NULL,
           'Database timeout ERR-42', 'Database timeout ERR-42', '1.000001', true,
           false, false, false, false, 'h1', 'run1', current_timestamp),
          ('W1:C1:2.000001', 'W1', 'C1', '2.000001', 2000001, 'U2', NULL,
           'Restarted the connection pool', 'Restarted the connection pool', '1.000001',
           false, true, false, false, false, 'h2', 'run1', current_timestamp),
          ('W1:C2:3.000001', 'W1', 'C2', '3.000001', 3000001, 'U1', NULL,
           'Welcome everyone', 'Welcome everyone', NULL, false, false, false, false,
           false, 'h3', 'run1', current_timestamp),
          ('W1:C2:4.000001', 'W1', 'C2', '4.000001', 4000001, 'U1', NULL,
           'deleted', 'deleted', NULL, false, false, false, false, true, 'h4', 'run1',
           current_timestamp);
        """
    )
    connection.close()
    return settings.canonical_db


def vector(index: int) -> list[float]:
    values = [0.0] * 768
    values[index] = 2.0
    return values
