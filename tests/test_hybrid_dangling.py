"""Regression test for https://github.com/thomasmaerz/slackquery/issues/1.

hybrid search must not raise KeyError when a thread_context references a
thread_root_id with no row in search_documents (deleted/inactive root).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import duckdb
import pytest

from slackquery.retrieval import SearchEngine
from slackquery.settings import Settings


class _FakeEmbedding:
    async def embed(self, texts: list[str], query: bool = False) -> list[list[float]]:
        return [[1.0] + [0.0] * 511 for _ in texts]


def _build_artifact(tmp_path: Path, ext_dir: Path) -> Path:
    artifact = tmp_path / "search-dangling.duckdb"
    connection = duckdb.connect(str(artifact), config={"extension_directory": str(ext_dir)})
    try:
        connection.execute("INSTALL fts")
        connection.execute("LOAD fts")
        connection.execute((Path("sql/001_artifact.sql")).read_text())
        connection.execute(
            """
            INSERT INTO search_documents VALUES
            ('msg-reply-1','message','W1','Acme','acme','C1','general','U1','Ada',
             1000001,'1970-01-01 00:00:01+00','W1:C1:root-ts','dangling-root-id',
             false,true,false,
             'hiring remote private job listings','hiring remote private job listings',
             'hiring remote','v1','{}',NULL),
            ('msg-root-valid','message','W1','Acme','acme','C1','general','U1','Ada',
             900001,'1970-01-01 00:00:00+00','W1:C1:other','msg-root-valid',
             true,false,false,
             'hiring engineers','hiring engineers','hiring engineers','v1','{}',NULL),
            ('thread:W1:C1:root-ts','thread_context','W1','Acme','acme','C1','general',
             'U1','Ada',1000001,'1970-01-01 00:00:01+00','W1:C1:root-ts',
             'dangling-root-id',true,false,false,
             'hiring remote private job listings hiring',
             'hiring remote private job listings hiring','hiring remote','v1','{}',NULL),
            ('thread:W1:C1:other','thread_context','W1','Acme','acme','C1','general',
             'U1','Ada',900001,'1970-01-01 00:00:00+00','W1:C1:other',
             'msg-root-valid',true,false,false,
             'hiring engineers context','hiring engineers context',
             'hiring engineers','v1','{}',NULL)
            """
        )
        connection.execute(
            "INSERT INTO artifact_metadata VALUES "
            "('test-build',2,'gen1','wm',4,2,current_timestamp,'{}')"
        )
        vector = [1.0] + [0.0] * 511
        connection.execute(
            "INSERT INTO document_vectors VALUES "
            "('msg-reply-1','gen1', ?::FLOAT[512]), "
            "('msg-root-valid','gen1', ?::FLOAT[512])",
            [vector, vector],
        )
        connection.execute(
            "PRAGMA create_fts_index('search_documents','document_id','text_lexical',"
            "stemmer='none',stopwords='none',lower=1,strip_accents=1,overwrite=1)"
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    return artifact


def test_hybrid_skips_dangling_thread_root(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir(parents=True, exist_ok=True)
    artifact = _build_artifact(tmp_path, ext_dir)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    engine_settings = settings.model_copy(
        update={"artifact_dir": tmp_path, "current_link": tmp_path / "current.duckdb"}
    )
    engine_settings.current_link.symlink_to(artifact)
    engine_settings.duckdb_extension_dir = ext_dir
    engine = SearchEngine(engine_settings, _FakeEmbedding())  # type: ignore[arg-type]

    response = asyncio.run(engine.search("hiring", mode="hybrid", limit=10))

    assert all(hit.document_id != "dangling-root-id" for hit in response.results)
    assert {hit.document_id for hit in response.results} >= {"msg-reply-1", "msg-root-valid"}
    # Inner-join fix filters the dangling root in SQL, before fusion, so the
    # normal path must produce no warnings (warning path tested separately).
    assert response.partial_warnings == []


def test_hybrid_defense_drops_missing_fused_id(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-fusion filter drops injected missing IDs with a warning."""
    from slackquery import retrieval as retrieval_module

    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir(parents=True, exist_ok=True)
    artifact = _build_artifact(tmp_path, ext_dir)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    engine_settings = settings.model_copy(
        update={"artifact_dir": tmp_path, "current_link": tmp_path / "current.duckdb"}
    )
    engine_settings.current_link.symlink_to(artifact)
    engine_settings.duckdb_extension_dir = ext_dir
    engine = SearchEngine(engine_settings, _FakeEmbedding())  # type: ignore[arg-type]

    real_rrf = retrieval_module.weighted_rrf

    def _inject_missing(
        rankings: dict[str, list[str]], weights: dict[str, float], k: int
    ) -> dict[str, float]:
        fused = real_rrf(rankings, weights, k)
        fused["missing-ghost-id"] = 999.0
        return fused

    monkeypatch.setattr(retrieval_module, "weighted_rrf", _inject_missing)
    response = asyncio.run(engine.search("hiring", mode="hybrid", limit=10))

    assert all(hit.document_id != "missing-ghost-id" for hit in response.results)
    assert len(response.partial_warnings) == 1
    assert "missing-ghost-id" in response.partial_warnings[0]


def test_hybrid_context_still_resolves_valid_root(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir(parents=True, exist_ok=True)
    artifact = _build_artifact(tmp_path, ext_dir)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    engine_settings = settings.model_copy(
        update={"artifact_dir": tmp_path, "current_link": tmp_path / "current.duckdb"}
    )
    engine_settings.current_link.symlink_to(artifact)
    engine_settings.duckdb_extension_dir = ext_dir
    engine = SearchEngine(engine_settings, _FakeEmbedding())  # type: ignore[arg-type]

    response = asyncio.run(engine.search("hiring engineers", mode="hybrid", limit=10))
    ids = {hit.document_id for hit in response.results}
    assert "msg-root-valid" in ids
    assert "dangling-root-id" not in ids
    # The valid root must come through the contextual arm, not just
    # lexical/semantic retrieval.
    valid_hit = next(hit for hit in response.results if hit.document_id == "msg-root-valid")
    assert valid_hit.contextual_rank is not None
