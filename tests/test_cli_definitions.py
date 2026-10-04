from __future__ import annotations

from slackquery.cli import main, parser
from slackquery.definitions import candidate_integrity, definitions


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
    assert definitions.get_schedule_def("slackquery_hourly_reconciliation").name == (
        "slackquery_hourly_reconciliation"
    )
