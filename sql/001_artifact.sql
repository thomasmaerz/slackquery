CREATE TABLE search_documents (
    document_id VARCHAR PRIMARY KEY,
    document_kind VARCHAR NOT NULL,
    workspace_id VARCHAR NOT NULL,
    workspace_name VARCHAR,
    workspace_slug VARCHAR NOT NULL,
    channel_id VARCHAR NOT NULL,
    channel_name VARCHAR,
    author_id VARCHAR,
    author_name VARCHAR,
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
    metadata_json JSON NOT NULL,
    permalink VARCHAR
);
CREATE TABLE document_vectors (
    document_id VARCHAR PRIMARY KEY,
    generation_id VARCHAR NOT NULL,
    embedding FLOAT[512] NOT NULL
);
CREATE TABLE artifact_metadata (
    build_id VARCHAR PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    generation_id VARCHAR NOT NULL,
    source_watermark VARCHAR NOT NULL,
    document_count BIGINT NOT NULL,
    vector_count BIGINT NOT NULL,
    built_at TIMESTAMPTZ NOT NULL,
    fts_config_json JSON NOT NULL
);
