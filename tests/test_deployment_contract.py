from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_code_server_image_contains_dagster_cli_and_source() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "dagster==1.13.20" in project["project"]["dependencies"]
    assert "COPY --chown=10001:10001 src ./src" in dockerfile
    assert "PYTHONPATH=/app/src" in dockerfile
    assert "EXPOSE 8080 4001" in dockerfile
    assert "/srv/slackquery/extensions" in dockerfile
    assert 'connection.execute("INSTALL fts")' in dockerfile


def test_compose_mounts_writable_extension_directory() -> None:
    compose = (ROOT / "compose.yaml").read_text()
    assert ":/srv/slackquery/extensions" in compose


def test_code_location_exports_expected_definitions_attribute() -> None:
    from slackquery import definitions as definitions_module

    assert definitions_module.definitions is not None
