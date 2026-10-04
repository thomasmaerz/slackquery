"""Shared domain and API models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SearchMode = Literal["lexical", "semantic", "hybrid"]


class SearchFilters(BaseModel):
    workspace_ids: list[str] = Field(default_factory=list)
    channel_ids: list[str] = Field(default_factory=list)
    author_ids: list[str] = Field(default_factory=list)
    start_ts_us: int | None = None
    end_ts_us: int | None = None
    document_kinds: list[str] = Field(default_factory=lambda: ["message", "file_chunk"])

    @model_validator(mode="after")
    def valid_range(self) -> SearchFilters:
        if (
            self.start_ts_us is not None
            and self.end_ts_us is not None
            and self.start_ts_us > self.end_ts_us
        ):
            raise ValueError("start_ts_us must not exceed end_ts_us")
        return self


class SearchHit(BaseModel):
    document_id: str
    document_kind: str
    workspace_id: str
    workspace_name: str | None
    workspace_slug: str
    channel_id: str
    channel_name: str | None
    author_id: str | None
    author_name: str | None
    ts_us: int
    timestamp: datetime
    thread_id: str | None
    thread_root_id: str | None
    text: str
    permalink: str | None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    contextual_rank: int | None = None
    exact_boost: float = 0.0
    fused_score: float


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    artifact_id: str
    build_id: str | None = None
    generation_id: str | None = None
    route: Literal["exact", "conceptual", "mixed"] = "mixed"
    effective_weights: dict[str, float] = Field(default_factory=dict)
    partial_warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    results: list[SearchHit]
    next_cursor: str | None = None


class MessageRecord(BaseModel):
    document_id: str
    workspace_id: str
    workspace_slug: str
    channel_id: str
    channel_name: str | None
    author_id: str | None
    author_name: str | None
    ts_us: int
    timestamp: datetime
    thread_id: str | None
    thread_root_id: str | None
    text: str
    permalink: str | None


class ScopeRecord(BaseModel):
    workspace_id: str
    workspace_slug: str
    workspace_name: str | None
    channel_id: str
    channel_name: str | None
    message_count: int
    first_ts_us: int
    last_ts_us: int


class ProjectionStats(BaseModel):
    projected: int
    active: int
    tombstoned: int
    watermark: str
    projected_by_kind: dict[str, int] = Field(default_factory=dict)
    active_by_kind: dict[str, int] = Field(default_factory=dict)
    extraction: dict[str, int] = Field(default_factory=dict)


class EmbedStats(BaseModel):
    claimed: int
    succeeded: int
    retryable_failed: int
    terminal_failed: int


class BuildResult(BaseModel):
    build_id: str
    artifact_path: str
    manifest_path: str
    checksum: str
    document_count: int
