"""Dagster assets and checks owned by the Slackquery code location."""

from pathlib import Path

import dagster as dg

from slackquery.artifact import (
    build_artifact,
    gold_validate,
    publish_artifact,
    retain_artifacts,
    validate_artifact,
)
from slackquery.db import connect_state
from slackquery.embedding import run_worker
from slackquery.projection import project_documents
from slackquery.settings import Settings


class SlackqueryResource(dg.ConfigurableResource):  # type: ignore[type-arg]
    canonical_db: str | None = None
    state_db: str | None = None
    artifact_dir: str | None = None
    current_link: str | None = None

    def settings(self) -> Settings:
        settings = Settings()
        overrides = {
            name: Path(value)
            for name, value in {
                "canonical_db": self.canonical_db,
                "state_db": self.state_db,
                "artifact_dir": self.artifact_dir,
                "current_link": self.current_link,
            }.items()
            if value is not None
        }
        return settings.model_copy(update=overrides)


@dg.asset(group_name="search", key_prefix=["search"])
def document_projection(
    context: dg.AssetExecutionContext, slackquery: SlackqueryResource
) -> dg.MaterializeResult:  # type: ignore[type-arg]
    del context
    settings = slackquery.settings()
    stats = project_documents(settings.canonical_db, settings.state_db, settings)
    return dg.MaterializeResult(metadata=stats.model_dump())


@dg.asset(group_name="search", key_prefix=["search"], deps=[document_projection])
def message_embeddings(
    context: dg.AssetExecutionContext, slackquery: SlackqueryResource
) -> dg.MaterializeResult:  # type: ignore[type-arg]
    del context
    stats = run_worker(slackquery.settings())
    return dg.MaterializeResult(metadata=stats.model_dump())


@dg.asset(group_name="search", key_prefix=["search"], deps=[message_embeddings])
def artifact_candidate(
    context: dg.AssetExecutionContext, slackquery: SlackqueryResource
) -> dg.MaterializeResult:  # type: ignore[type-arg]
    result = build_artifact(slackquery.settings(), dagster_run_id=context.run_id)
    return dg.MaterializeResult(
        metadata={
            **result.model_dump(),
            "artifact_path": dg.MetadataValue.path(result.artifact_path),
        }
    )


@dg.asset(group_name="search", key_prefix=["search"], deps=[artifact_candidate])
def published_artifact(
    context: dg.AssetExecutionContext, slackquery: SlackqueryResource
) -> dg.MaterializeResult:  # type: ignore[type-arg]
    settings = slackquery.settings()
    artifact = _candidate_for_run(settings, context.run_id)
    checks = gold_validate(settings, artifact)
    if not checks["valid"]:
        raise RuntimeError(f"Gold validation failed: {checks['checks']}")
    current = publish_artifact(settings, artifact)
    retention = retain_artifacts(settings)
    return dg.MaterializeResult(
        metadata={
            "current": dg.MetadataValue.path(current),
            "build_id": checks["artifact_id"],
            "document_count": checks["document_count"],
            "vector_count": checks["vector_count"],
            "document_kinds": dg.MetadataValue.json(checks["document_kinds"]),
            "retention": dg.MetadataValue.json(retention),
        }
    )


def _candidate_for_run(settings: Settings, run_id: str) -> Path:
    connection = connect_state(settings.state_db)
    try:
        row = connection.execute(
            """
            SELECT artifact_path FROM search_builds
            WHERE dagster_run_id = ? AND status IN ('validated', 'published')
            ORDER BY completed_at DESC LIMIT 1
            """,
            [run_id],
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError(f"no validated candidate for Dagster run {run_id}")
    return Path(row[0])


@dg.asset_check(asset=artifact_candidate, blocking=True)
def candidate_integrity(
    context: dg.AssetCheckExecutionContext, slackquery: SlackqueryResource
) -> dg.AssetCheckResult:
    settings = slackquery.settings()
    artifact = _candidate_for_run(settings, context.run.run_id)
    checks = validate_artifact(artifact, verify_checksum=True)
    return dg.AssetCheckResult(passed=checks["valid"], metadata=checks)


@dg.asset_check(asset=artifact_candidate, blocking=True)
def gold_readiness(
    context: dg.AssetCheckExecutionContext, slackquery: SlackqueryResource
) -> dg.AssetCheckResult:
    settings = slackquery.settings()
    artifact = _candidate_for_run(settings, context.run.run_id)
    report = gold_validate(settings, artifact)
    return dg.AssetCheckResult(passed=report["valid"], metadata=report)


reconcile_job = dg.define_asset_job(
    "slackquery_reconcile",
    selection=dg.AssetSelection.assets(
        document_projection, message_embeddings, artifact_candidate, published_artifact
    ),
)

reconcile_schedule = dg.ScheduleDefinition(
    name="slackquery_hourly_reconciliation",
    cron_schedule="0 * * * *",
    target=reconcile_job,
)


definitions = dg.Definitions(
    assets=[document_projection, message_embeddings, artifact_candidate, published_artifact],
    asset_checks=[candidate_integrity, gold_readiness],
    jobs=[reconcile_job],
    schedules=[reconcile_schedule],
    resources={"slackquery": SlackqueryResource()},
)
