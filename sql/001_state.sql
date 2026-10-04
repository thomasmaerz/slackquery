CREATE TABLE IF NOT EXISTS schema_metadata (
    schema_version INTEGER PRIMARY KEY,
    installed_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS document_projection (
    document_id VARCHAR PRIMARY KEY,
    document_kind VARCHAR NOT NULL,
    workspace_id VARCHAR NOT NULL,
    workspace_name VARCHAR,
    workspace_slug VARCHAR NOT NULL,
    workspace_url VARCHAR,
    channel_id VARCHAR NOT NULL,
    channel_name VARCHAR,
    author_id VARCHAR,
    author_name VARCHAR,
    ts VARCHAR NOT NULL,
    ts_us BIGINT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    thread_id VARCHAR,
    thread_root_id VARCHAR,
    is_thread_parent BOOLEAN NOT NULL,
    is_thread_reply BOOLEAN NOT NULL,
    is_thread_broadcast BOOLEAN NOT NULL,
    text_display VARCHAR NOT NULL,
    text_lexical VARCHAR NOT NULL,
    text_embedding VARCHAR NOT NULL,
    source_version VARCHAR NOT NULL,
    text_hash VARCHAR NOT NULL,
    metadata_json JSON NOT NULL,
    permalink VARCHAR,
    is_active BOOLEAN NOT NULL,
    projected_at TIMESTAMPTZ NOT NULL
);

ALTER TABLE document_projection ADD COLUMN IF NOT EXISTS source_identity VARCHAR;
ALTER TABLE document_projection ADD COLUMN IF NOT EXISTS source_revision VARCHAR;
ALTER TABLE document_projection ADD COLUMN IF NOT EXISTS embedding_recipe VARCHAR;
UPDATE document_projection
SET source_identity = coalesce(source_identity, source_version),
    source_revision = coalesce(source_revision, source_version),
    embedding_recipe = coalesce(
        embedding_recipe,
        CASE WHEN document_kind = 'message' THEN 'message-v1' ELSE document_kind || '-v1' END
    );

CREATE TABLE IF NOT EXISTS embedding_generations (
    generation_id VARCHAR PRIMARY KEY,
    endpoint_class VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    model_revision VARCHAR NOT NULL,
    native_dimension INTEGER,
    stored_dimension INTEGER NOT NULL,
    document_prefix VARCHAR NOT NULL,
    query_prefix VARCHAR NOT NULL,
    normalized BOOLEAN NOT NULL,
    distance_metric VARCHAR NOT NULL,
    text_recipe_version VARCHAR NOT NULL,
    config_hash VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS document_embeddings (
    document_id VARCHAR NOT NULL,
    generation_id VARCHAR NOT NULL,
    source_version VARCHAR NOT NULL,
    text_hash VARCHAR NOT NULL,
    embedding FLOAT[512],
    native_dimension INTEGER,
    request_id VARCHAR,
    state VARCHAR NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    lease_id VARCHAR,
    lease_expires_at TIMESTAMPTZ,
    next_attempt_at TIMESTAMPTZ,
    embedded_at TIMESTAMPTZ,
    error_class VARCHAR,
    error_message VARCHAR,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (document_id, generation_id, text_hash)
);

CREATE INDEX IF NOT EXISTS embedding_state_idx
ON document_embeddings(generation_id, state, next_attempt_at);

CREATE TABLE IF NOT EXISTS projection_watermarks (
    source_path VARCHAR PRIMARY KEY,
    source_schema_version INTEGER NOT NULL,
    source_watermark VARCHAR NOT NULL,
    projected_at TIMESTAMPTZ NOT NULL,
    active_count BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS projection_reports (
    source_identity VARCHAR PRIMARY KEY,
    source_watermark VARCHAR NOT NULL,
    projected_at TIMESTAMPTZ NOT NULL,
    projected_by_kind_json JSON NOT NULL,
    active_by_kind_json JSON NOT NULL,
    extraction_json JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS search_builds (
    build_id VARCHAR PRIMARY KEY,
    source_watermark VARCHAR NOT NULL,
    generation_id VARCHAR NOT NULL,
    document_count BIGINT NOT NULL,
    embedding_count BIGINT NOT NULL,
    fts_config_json JSON NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    checksum VARCHAR,
    artifact_path VARCHAR,
    status VARCHAR NOT NULL,
    dagster_run_id VARCHAR,
    code_version VARCHAR NOT NULL,
    schema_version INTEGER NOT NULL,
    error_message VARCHAR
);

INSERT INTO schema_metadata(schema_version) VALUES (3) ON CONFLICT DO NOTHING;
