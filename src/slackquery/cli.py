"""Slackquery operator CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import uvicorn

from slackquery.artifact import (
    backup_state,
    build_artifact,
    gold_validate,
    publish_artifact,
    restore_state,
    retain_artifacts,
    rollback_artifact,
    validate_artifact,
)
from slackquery.embedding import EmbeddingWorker, LocalEmbeddingClient
from slackquery.projection import project_documents
from slackquery.retrieval import SearchEngine
from slackquery.server import create_asgi_app
from slackquery.settings import Settings


def _print(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="slackquery")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("project", help="project canonical messages into state")
    embed = commands.add_parser("embed", help="resume batch embedding")
    embed.add_argument("--max-items", type=int)
    commands.add_parser("embedding-status", help="verify and report the active embedding backend")
    commands.add_parser("build", help="build an immutable candidate")
    publish = commands.add_parser("publish", help="atomically publish a candidate")
    publish.add_argument("artifact", type=Path)
    validate = commands.add_parser("validate", help="validate an artifact")
    validate.add_argument("artifact", type=Path)
    validate.add_argument("--checksum", action="store_true")
    run = commands.add_parser("run", help="serve MCP Streamable HTTP")
    run.add_argument("--host")
    run.add_argument("--port", type=int)
    benchmark = commands.add_parser("benchmark", help="benchmark lexical queries")
    benchmark.add_argument("queries", type=Path, help="one query per line")
    benchmark.add_argument("--iterations", type=int, default=1)
    commands.add_parser("retain", help="apply artifact retention policy")
    rollback = commands.add_parser("rollback", help="atomically publish a prior build")
    rollback.add_argument("build_id_or_path")
    backup = commands.add_parser("backup", help="checkpoint and back up state DB")
    backup.add_argument("destination", type=Path)
    restore = commands.add_parser("restore", help="validate and restore state DB backup")
    restore.add_argument("backup", type=Path)
    gold = commands.add_parser("gold-validate", help="run automated Gold invariants")
    gold.add_argument("artifact", type=Path, nargs="?")
    return result


async def _embed(settings: Settings, max_items: int | None) -> object:
    client = LocalEmbeddingClient(settings)
    try:
        return await EmbeddingWorker(settings, client).run(max_items)
    finally:
        await client.close()


async def _embedding_status(settings: Settings) -> dict[str, object]:
    client = LocalEmbeddingClient(settings)
    try:
        await client.verify_model()
        return client.status()
    finally:
        await client.close()


async def _benchmark(settings: Settings, path: Path, iterations: int) -> dict[str, object]:
    queries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not queries:
        raise ValueError("benchmark query file is empty")
    engine = SearchEngine(settings)
    durations: list[float] = []
    for _ in range(iterations):
        for query in queries:
            started = time.perf_counter()
            await engine.search(query, mode="lexical", limit=10)
            durations.append((time.perf_counter() - started) * 1000)
    ordered = sorted(durations)
    return {
        "queries": len(durations),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[max(0, int(len(ordered) * 0.95) - 1)],
        "max_ms": max(ordered),
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = Settings()
    try:
        if args.command == "project":
            _print(project_documents(settings.canonical_db, settings.state_db, settings))
        elif args.command == "embed":
            _print(asyncio.run(_embed(settings, args.max_items)))
        elif args.command == "embedding-status":
            _print(asyncio.run(_embedding_status(settings)))
        elif args.command == "build":
            _print(build_artifact(settings))
        elif args.command == "publish":
            _print({"current": str(publish_artifact(settings, args.artifact))})
        elif args.command == "validate":
            _print(validate_artifact(args.artifact, verify_checksum=args.checksum))
        elif args.command == "retain":
            _print(retain_artifacts(settings))
        elif args.command == "rollback":
            _print({"current": str(rollback_artifact(settings, args.build_id_or_path))})
        elif args.command == "backup":
            _print(backup_state(settings, args.destination))
        elif args.command == "restore":
            _print(restore_state(settings, args.backup))
        elif args.command == "gold-validate":
            report = gold_validate(settings, args.artifact)
            _print(report)
            if not report["valid"]:
                return 1
        elif args.command == "benchmark":
            _print(asyncio.run(_benchmark(settings, args.queries, args.iterations)))
        elif args.command == "run":
            uvicorn.run(
                create_asgi_app(settings),
                host=args.host or settings.host,
                port=args.port or settings.port,
                log_level=settings.log_level.lower(),
            )
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
