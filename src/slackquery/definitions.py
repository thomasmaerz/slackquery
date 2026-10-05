"""Dagster assets and checks owned by the Slackquery code location."""

import time
from collections.abc import Mapping
from pathlib import Path

import dagster as dg

from slackquery.artifact import (
    build_artifact,
    gold_validate,
    publish_artifact,
    retain_artifacts,
    validate_artifact,
)
from slackquery.db import connect_readonly
from slackquery.embedding import run_worker
from slackquery.projection import project_documents
from slackquery.settings import Settings

LANE_TAG = "lane"
LANE = "slackquery"

# Runs that completed ingestion and therefore may have produced fresh
# canonical data. `all_workspaces_extract` is excluded: it only lands raw
# archives, the canonical sweep (or a workspace incremental/initial flow)
# follows it and triggers reconcile then.
TRIGGER_MODES = frozenset({"initial", "incremental", "full"})
TRIGGER_MODE_TAG = "slackpipe/mode"
RAW_EXTRACT_JOB = "all_workspaces_extract"

# A reconcile emitting no events for this long is hung, not slow: healthy
# runs log at least every few seconds while draining the backlog.
STUCK_SILENCE_SECONDS = 90 * 60


def triggers_reconcile(job_name: str, tags: Mapping[str, str]) -> bool:
    """Decide whether a successful run produced fresh canonical data."""
    if job_name == RAW_EXTRACT_JOB:
        return False
    return tags.get(TRIGGER_MODE_TAG) in TRIGGER_MODES


def event_timestamp_seconds(timestamp: float) -> float:
    """Normalize an event timestamp that may be seconds or milliseconds."""
    return timestamp / 1000.0 if timestamp > 1e12 else timestamp


def _emit(context: dg.AssetExecutionContext | dg.AssetCheckExecutionContext, message: str) -> None:
    """Emit progress to both the event log and the captured stdout tab."""
    context.log.info(message)
    print(message, flush=True)


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
    settings = slackquery.settings()
    _emit(context, f"projecting documents from {settings.canonical_db}")
    stats = project_documents(settings.canonical_db, settings.state_db, settings)
    _emit(context, f"projection finished: {stats.model_dump()}")
    return dg.MaterializeResult(metadata=stats.model_dump())


@dg.asset(group_name="search", key_prefix=["search"], deps=[document_projection])
def message_embeddings(
    context: dg.AssetExecutionContext, slackquery: SlackqueryResource
) -> dg.MaterializeResult:  # type: ignore[type-arg]
    _emit(context, "embedding worker started")
    stats = run_worker(slackquery.settings(), status=lambda message: _emit(context, message))
    _emit(context, f"embedding worker finished: {stats.model_dump()}")
    return dg.MaterializeResult(metadata=stats.model_dump())


@dg.asset(group_name="search", key_prefix=["search"], deps=[message_embeddings])
def artifact_candidate(
    context: dg.AssetExecutionContext, slackquery: SlackqueryResource
) -> dg.MaterializeResult:  # type: ignore[type-arg]
    _emit(context, "building artifact candidate")
    result = build_artifact(slackquery.settings(), dagster_run_id=context.run_id)
    _emit(context, f"artifact candidate built: {result.artifact_path}")
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
    _emit(context, f"validating gold readiness for {artifact}")
    checks = gold_validate(settings, artifact)
    if not checks["valid"]:
        raise RuntimeError(f"Gold validation failed: {checks['checks']}")
    current = publish_artifact(settings, artifact)
    retention = retain_artifacts(settings)
    _emit(context, f"published artifact, current={current}")
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
    connection = connect_readonly(settings.state_db)
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
    _emit(context, f"checking candidate integrity for {artifact}")
    checks = validate_artifact(artifact, verify_checksum=True)
    _emit(context, f"candidate integrity valid={checks['valid']}")
    return dg.AssetCheckResult(passed=checks["valid"], metadata=checks)


@dg.asset_check(asset=artifact_candidate, blocking=True)
def gold_readiness(
    context: dg.AssetCheckExecutionContext, slackquery: SlackqueryResource
) -> dg.AssetCheckResult:
    settings = slackquery.settings()
    artifact = _candidate_for_run(settings, context.run.run_id)
    _emit(context, f"checking gold readiness for {artifact}")
    report = gold_validate(settings, artifact)
    _emit(context, f"gold readiness valid={report['valid']}")
    return dg.AssetCheckResult(passed=report["valid"], metadata=report)


reconcile_job = dg.define_asset_job(
    "slackquery_reconcile",
    selection=dg.AssetSelection.assets(
        document_projection, message_embeddings, artifact_candidate, published_artifact
    ),
    tags={LANE_TAG: LANE},
)


@dg.run_status_sensor(
    run_status=dg.DagsterRunStatus.SUCCESS,
    monitor_all_code_locations=True,
    request_job=reconcile_job,
    minimum_interval_seconds=60,
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def reconcile_on_ingestion(
    context: dg.RunStatusSensorContext,
) -> dg.SkipReason | dg.RunRequest:
    """Reconcile search right after an ingestion run lands fresh data.

    Replaces the old hourly schedule: reconcile only runs when canonical
    data may actually have changed, and never overlaps itself.
    """
    run = context.dagster_run
    if not triggers_reconcile(run.job_name, run.tags):
        context.log.info(f"ignoring successful {run.job_name}, no canonical data produced")
        return dg.SkipReason(f"job {run.job_name} does not produce canonical data")
    active = context.instance.get_runs(
        filters=dg.RunsFilter(
            job_name=reconcile_job.name,
            statuses=[dg.DagsterRunStatus.QUEUED, dg.DagsterRunStatus.STARTED],
        ),
        limit=1,
    )
    if active:
        context.log.info(
            f"reconcile already active ({active[0].run_id[:8]}), skipping "
            f"trigger from {run.run_id[:8]}"
        )
        return dg.SkipReason(
            f"reconcile already active ({active[0].run_id[:8]}), skipping "
            f"trigger from {run.run_id[:8]}"
        )
    context.log.info(f"triggering reconcile after successful {run.job_name}")
    return dg.RunRequest(
        run_key=f"ingestion-{run.run_id}",
        tags={LANE_TAG: LANE},
    )


@dg.sensor(
    minimum_interval_seconds=300,
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def reconcile_stuck_reaper(context: dg.SensorEvaluationContext) -> dg.SkipReason:
    """Terminate reconcile runs that stopped emitting events.

    A hung run holds its concurrency-lane slot forever and blocks the
    queue behind it; silence well past any legitimate quiet stretch
    (startup, a slow batch) means it will never finish on its own.
    Termination is safe: embedding progress checkpoints per batch in
    DuckDB and the next run reclaims expired leases.
    """
    stuck: list[str] = []
    now = time.time()
    active = context.instance.get_runs(
        filters=dg.RunsFilter(
            job_name=reconcile_job.name,
            statuses=[dg.DagsterRunStatus.STARTED],
        )
    )
    for run in active:
        records = context.instance.get_records_for_run(
            run.run_id, limit=1, ascending=False
        )
        if not records.records:
            continue
        silence = now - event_timestamp_seconds(records.records[0].timestamp)
        if silence > STUCK_SILENCE_SECONDS:
            context.log.warning(
                f"terminating silent reconcile run {run.run_id[:8]} "
                f"(no events for {silence / 60:.0f}m)"
            )
            context.instance.report_run_canceling(run)
            stuck.append(run.run_id[:8])
    if stuck:
        context.log.info(f"terminated silent runs: {', '.join(stuck)}")
        return dg.SkipReason(f"terminated silent runs: {', '.join(stuck)}")
    context.log.info(f"checked {len(active)} active reconcile runs, none silent")
    return dg.SkipReason(
        f"no silent reconcile runs ({len(active)} active checked)"
    )


definitions = dg.Definitions(
    assets=[document_projection, message_embeddings, artifact_candidate, published_artifact],
    asset_checks=[candidate_integrity, gold_readiness],
    jobs=[reconcile_job],
    sensors=[reconcile_on_ingestion, reconcile_stuck_reaper],
    resources={"slackquery": SlackqueryResource()},
)
