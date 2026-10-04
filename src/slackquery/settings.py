"""Validated runtime configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from `SLACKQUERY_*` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="SLACKQUERY_", env_file=".env", extra="ignore", case_sensitive=False
    )
    canonical_db: Path = Path("/path/to/slackpipe.duckdb")
    state_db: Path = Path("/srv/slackquery/state/slackquery.duckdb")
    artifact_dir: Path = Path("/srv/slackquery/artifacts")
    current_link: Path = Path("/srv/slackquery/artifacts/current.duckdb")
    attachment_root: Path = Path("/opt/slackpipe/slackdump")
    attachment_search_roots: list[Path] = Field(default_factory=list)
    attachment_workspace_map: dict[str, str] = Field(default_factory=dict)
    duckdb_extension_dir: Path = Path("/srv/slackquery/extensions")
    embedding_backend: Literal["pytorch", "ollama"] = Field(
        default="pytorch",
        validation_alias=AliasChoices("EMBEDDING_BACKEND", "SLACKQUERY_EMBEDDING_BACKEND"),
    )
    embedding_base_url_override: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "EMBEDDING_BASE_URL", "SLACKQUERY_EMBEDDING_BASE_URL", "embedding_base_url"
        ),
    )
    pytorch_embedding_base_url: str = Field(
        default="http://localhost:11435",
        validation_alias=AliasChoices(
            "PYTORCH_EMBEDDING_BASE_URL", "SLACKQUERY_PYTORCH_EMBEDDING_BASE_URL"
        ),
    )
    ollama_embedding_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices(
            "OLLAMA_EMBEDDING_BASE_URL", "SLACKQUERY_OLLAMA_EMBEDDING_BASE_URL"
        ),
    )
    embedding_model: str = Field(
        default="nomic-embed-text:v1.5",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "SLACKQUERY_EMBEDDING_MODEL"),
    )
    embedding_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "EMBEDDING_API_KEY", "SLACKQUERY_EMBEDDING_API_KEY"
        ),
    )
    embedding_model_revision: str | None = None
    embedding_code_revision: str | None = None
    embedding_dim: int = Field(
        default=512,
        validation_alias=AliasChoices("EMBEDDING_DIM", "SLACKQUERY_EMBEDDING_DIM"),
    )
    embedding_keep_alive: str = "5m"
    embedding_verify_model: bool = True
    embedding_timeout_seconds: float = Field(default=120.0, ge=1, le=1800)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    pytorch_embedding_batch_size: int = Field(default=1, ge=1, le=256)
    embedding_max_chars: int = Field(default=64_000, ge=512, le=1_000_000)
    embedding_max_attempts: int = Field(default=5, ge=1, le=20)
    lease_seconds: int = Field(default=900, ge=30, le=86_400)
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    bearer_token: SecretStr | None = None
    poll_seconds: float = Field(default=5.0, ge=0.2, le=300)
    max_query_chars: int = Field(default=2_000, ge=1, le=20_000)
    max_results: int = Field(default=50, ge=1, le=200)
    max_filter_values: int = Field(default=100, ge=1, le=1_000)
    candidate_window: int = Field(default=100, ge=1, le=1_000)
    rrf_k: int = Field(default=60, ge=1, le=1_000)
    lexical_rrf_weight_exact: float = Field(default=1.8, ge=0, le=10)
    semantic_rrf_weight_exact: float = Field(default=0.5, ge=0, le=10)
    lexical_rrf_weight_conceptual: float = Field(default=0.7, ge=0, le=10)
    semantic_rrf_weight_conceptual: float = Field(default=1.5, ge=0, le=10)
    lexical_rrf_weight_mixed: float = Field(default=1.0, ge=0, le=10)
    semantic_rrf_weight_mixed: float = Field(default=1.0, ge=0, le=10)
    context_rrf_weight: float = Field(default=0.65, ge=0, le=10)
    exact_phrase_boost: float = Field(default=0.02, ge=0, le=10)
    exact_token_boost: float = Field(default=0.002, ge=0, le=10)
    diversity_max_per_thread: int | None = Field(default=3, ge=1, le=100)
    diversity_max_per_channel: int | None = Field(default=None, ge=1, le=1_000)
    thread_context_max_chars: int = Field(default=24_000, ge=256, le=1_000_000)
    thread_context_max_messages: int = Field(default=100, ge=1, le=10_000)
    attachment_max_bytes: int = Field(default=25_000_000, ge=1_024, le=1_000_000_000)
    file_chunk_tokens: int = Field(default=512, ge=32, le=8_192)
    file_chunk_overlap_tokens: int = Field(default=50, ge=0, le=2_048)
    token_char_approximation: int = Field(default=4, ge=1, le=16)
    retain_artifacts: int = Field(default=3, ge=2, le=100)
    retention_min_age_seconds: int = Field(default=86_400, ge=0, le=31_536_000)
    log_level: str = "INFO"

    @field_validator(
        "embedding_base_url_override",
        "pytorch_embedding_base_url",
        "ollama_embedding_base_url",
    )
    @classmethod
    def normalize_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith(("http://", "https://")):
            raise ValueError("must be an http(s) URL")
        return value.rstrip("/")

    @field_validator("embedding_dim")
    @classmethod
    def require_stored_dimension(cls, value: int) -> int:
        if value != 512:
            raise ValueError("embedding_dim must be exactly 512")
        return value

    @model_validator(mode="after")
    def validate_search_windows(self) -> Settings:
        if self.candidate_window < self.max_results:
            raise ValueError("candidate_window must be at least max_results")
        if self.file_chunk_overlap_tokens >= self.file_chunk_tokens:
            raise ValueError("file_chunk_overlap_tokens must be less than file_chunk_tokens")
        return self

    @property
    def embedding_base_url(self) -> str:
        """Resolve the common override or the selected backend's URL."""
        if self.embedding_base_url_override is not None:
            return self.embedding_base_url_override
        if self.embedding_backend == "pytorch":
            return self.pytorch_embedding_base_url
        return self.ollama_embedding_base_url

    @property
    def generation_id(self) -> str:
        revision = (self.embedding_model_revision or "unpinned").replace("/", "-")
        code = self.embedding_code_revision
        code_identity = f":code-{code.replace('/', '-')}" if code else ""
        return (
            f"{self.embedding_model}@{revision}{code_identity}:"
            f"d{self.embedding_dim}:recipe-v1"
        )

    @property
    def effective_embedding_batch_size(self) -> int:
        """Use a separately capped request size for the CUDA backend."""
        if self.embedding_backend == "pytorch":
            return min(self.embedding_batch_size, self.pytorch_embedding_batch_size)
        return self.embedding_batch_size
