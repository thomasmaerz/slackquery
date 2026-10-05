from __future__ import annotations

from slackquery.cli import main, parser
from slackquery.definitions import (
    candidate_integrity,
    definitions,
    event_timestamp_seconds,
    triggers_reconcile,
)


def test_cli_commands_are_present() -> None:
    help_text = parser().format_help()
    for command in (
        "project",
        "embed",
        "embedding-status",
        "build",
        "publish",
        "run",
        "validate",
        "benchmark",
        "retain",
        "rollback",
        "backup",
        "restore",
        "gold-validate",
    ):
        assert command in help_text


def test_cli_reports_missing_canonical(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("SLACKQUERY_CANONICAL_DB", str(tmp_path / "missing.duckdb"))
    monkeypatch.setenv("SLACKQUERY_STATE_DB", str(tmp_path / "state.duckdb"))
    assert main(["project"]) == 1
    assert "missing.duckdb" in capsys.readouterr().err


def test_dagster_asset_contract() -> None:
    keys = {key.to_user_string() for key in definitions.resolve_all_asset_keys()}
    assert keys == {
        "search/document_projection",
        "search/message_embeddings",
        "search/artifact_candidate",
        "search/published_artifact",
    }
    assert next(iter(candidate_integrity.check_specs)).blocking is True
    assert definitions.get_job_def("slackquery_reconcile").tags["lane"] == "slackquery"
    assert (
        definitions.get_sensor_def("reconcile_on_ingestion").name
        == "reconcile_on_ingestion"
    )
    assert (
        definitions.get_sensor_def("reconcile_stuck_reaper").name
        == "reconcile_stuck_reaper"
    )


def test_ingestion_trigger_predicate() -> None:
    assert triggers_reconcile("leadership_incremental", {"slackpipe/mode": "incremental"})
    assert triggers_reconcile("leadership_ingest_once", {"slackpipe/mode": "initial"})
    assert triggers_reconcile("all_workspaces_canonical", {"slackpipe/mode": "full"})
    assert not triggers_reconcile(
        "all_workspaces_extract", {"slackpipe/mode": "incremental"}
    )
    assert not triggers_reconcile("yyjtech_attachments_once", {"slackpipe/mode": "attachments"})
    assert not triggers_reconcile("slackquery_reconcile", {"lane": "slackquery"})


def test_event_timestamp_normalization() -> None:
    assert event_timestamp_seconds(1791162000.0) == 1791162000.0
    assert event_timestamp_seconds(1791162000123.0) == 1791162000.123
