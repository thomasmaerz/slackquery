<div align="center">

# Slackquery

### Private-by-design hybrid search for Slack archives

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.5-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![Dagster](https://img.shields.io/badge/Dagster-1.13-654FF0?logo=dagster&logoColor=white)](https://dagster.io/)
[![MCP](https://img.shields.io/badge/MCP-Streamable_HTTP-5A45FF)](https://modelcontextprotocol.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](compose.yaml)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<img src="https://skillicons.dev/icons?i=python,docker,linux" alt="Python, Docker, and Linux" height="44" />
&nbsp;
<img src="https://cdn.simpleicons.org/duckdb/F5F5F5" alt="DuckDB" height="42" />
&nbsp;
<img src="https://cdn.simpleicons.org/dagster/654FF0" alt="Dagster" height="42" />

</div>

Slackquery turns a canonical Slack archive in DuckDB into an immutable search
artifact and serves it through Model Context Protocol (MCP). It combines BM25
lexical retrieval with exact cosine similarity and reciprocal rank fusion (RRF),
while keeping ingestion data read-only and deployment under your control.

## Why Slackquery

- **Hybrid retrieval:** exact terms through DuckDB FTS, semantic matches through
  local embeddings, and deterministic RRF for the default ranking.
- **Immutable serving:** build and validate a new artifact before atomically
  publishing it; readers never share the mutable pipeline database.
- **Local model support:** switch between a batched PyTorch-compatible server and
  Ollama with one environment variable.
- **Resumable enrichment:** content-addressed vectors, leases, retries, and
  durable checkpoints make embedding work incremental and idempotent.
- **Read-only MCP tools:** bounded search, exact message lookup, thread expansion,
  scope discovery, health probes, and optional bearer authentication.
- **Dagster-native operations:** assets, checks, and a reconciliation schedule are
  included without coupling the MCP process to the writer.

## Architecture

```mermaid
flowchart LR
    C[(Canonical Slack DuckDB<br/>read-only)]
    P[Deterministic projection]
    S[(Pipeline state<br/>documents + vectors)]
    E{Embedding backend}
    PT[PyTorch-compatible<br/>embedding server]
    OL[Ollama]
    B[Build + validate<br/>immutable artifact]
    A[(Published DuckDB<br/>FTS + vectors)]
    M[MCP Streamable HTTP]
    U[Agents and clients]

    C --> P --> S
    S --> E
    E --> PT
    E --> OL
    PT --> S
    OL --> S
    S --> B --> A --> M --> U
```

The canonical database is attached `READ_ONLY`. Projection, embeddings, build
metadata, and publication state live in Slackquery-owned storage. The serving
process opens only the selected artifact.

## Quickstart

### Prerequisites

- Python 3.11+
- a Slackpipe-compatible DuckDB archive
- an Ollama-compatible `/api/embed` endpoint, using either Ollama itself or the
  supported PyTorch transport
- the DuckDB FTS extension (installed automatically by the container image)

### Local installation

```bash
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Edit the ignored `.env` for your database, storage directories, embedding
endpoint, and model. Then run the pipeline:

```bash
slackquery project
slackquery embedding-status
slackquery embed
slackquery build
slackquery validate /srv/slackquery/artifacts/search-*.duckdb --checksum
slackquery publish /srv/slackquery/artifacts/search-*.duckdb
slackquery run
```

Connect an MCP client to `http://mcp-host:8181/mcp`. Liveness and readiness are
available at `/healthz` and `/readyz`.

### Docker Compose

Copy `.env.example` to `.env`, set the host mount variables, then run:

```bash
docker compose up --build -d
curl -fsS http://localhost:8181/healthz
curl -fsS http://localhost:8181/readyz
```

Compose starts the read-only MCP service. Run projection, embedding, build, and
publication from a writer process or through the included Dagster definitions.

## Embedding backends

Both transports use the same `/api/embed` request shape. Switching transport is
safe only when the model weights, native dimensions, prefixes, 512-dimensional
truncation, and L2 normalization are identical.

```dotenv
EMBEDDING_BACKEND=pytorch
PYTORCH_EMBEDDING_BASE_URL=http://embedding-host:11435
OLLAMA_EMBEDDING_BASE_URL=http://ollama-host:11434
EMBEDDING_MODEL=nomic-embed-text:v1.5
```

Switch to Ollama without changing source code:

```dotenv
EMBEDDING_BACKEND=ollama
```

Model digests are intentionally not hard-coded. For reproducible generations,
read the digest from your server and pin it locally:

```dotenv
SLACKQUERY_EMBEDDING_MODEL_REVISION=sha256-digest-from-your-server
```

Without a revision, Slackquery labels the generation `unpinned`. Pinning is
recommended for production because it detects silent model replacement.

## MCP tools

| Tool | Purpose |
|---|---|
| `search_slack` | Search in `hybrid`, `lexical`, or `semantic` mode with scope and time filters. |
| `get_slack_message` | Fetch one document with bounded same-channel context. |
| `get_slack_thread` | Expand a result into a chronological thread. |
| `list_slack_scopes` | Discover archived workspaces, channels, and coverage. |

Responses include stable source identity, component ranks, artifact identity,
resolved filters, and opaque pagination cursors. Fused scores are ranking values,
not probabilities.

## Project status

| Tier | Capability |
|---|---|
| **Stable** | Projection, resumable embeddings, immutable builds, validation, atomic publication, lexical search, exact semantic search, hybrid RRF, MCP tools, health probes. |
| **Beta** | Dagster integration, remote multi-client operation, PyTorch-compatible transport, operational benchmarks. |
| **Planned** | Human-judged relevance suites, optional approximate vector indexes at larger scale, file/chunk retrieval, richer observability. |

## Documentation

- [Architecture and design plan](PLAN.md)
- [Operations runbook](docs/operations.md)
- [MCP client setup](docs/clients/README.md)
- [Representative benchmarks](docs/benchmarks.md)
- [Agent usage skill](SKILL.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Development

```bash
pytest
ruff check .
mypy
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development expectations. Slackquery
is available under the [MIT License](LICENSE).
