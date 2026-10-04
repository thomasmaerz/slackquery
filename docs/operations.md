# Operations runbook

This runbook covers generic deployment and recovery procedures. All paths and
hosts are examples; keep installation-specific values in the ignored `.env` or a
secret/configuration manager.

## Runtime topology

| Component | Example | Access contract |
|---|---|---|
| Canonical archive | `/path/to/slackpipe.duckdb` | Read-only to Slackquery. |
| State database | `/srv/slackquery/state/slackquery.duckdb` | Writable by the pipeline only. |
| Artifact directory | `/srv/slackquery/artifacts` | Writable by builder/publisher; read-only to serving where practical. |
| Published selector | `/srv/slackquery/artifacts/current.duckdb` | Atomic symlink to a validated artifact. |
| PyTorch endpoint | `http://embedding-host:11435` | Pipeline and semantic-query access. |
| Ollama endpoint | `http://ollama-host:11434` | Alternative compatible transport. |
| MCP endpoint | `http://mcp-host:8181/mcp` | Client-facing Streamable HTTP. |

Keep the writer and reader responsibilities separate. The MCP process does not
need the canonical or state databases. The writer must not mutate the canonical
archive.

## Configuration

Copy the public template and edit the ignored file:

```bash
cp .env.example .env
chmod 600 .env
```

Minimum settings:

```dotenv
SLACKQUERY_CANONICAL_DB=/path/to/slackpipe.duckdb
SLACKQUERY_STATE_DB=/srv/slackquery/state/slackquery.duckdb
SLACKQUERY_ARTIFACT_DIR=/srv/slackquery/artifacts
SLACKQUERY_CURRENT_LINK=/srv/slackquery/artifacts/current.duckdb
SLACKQUERY_DUCKDB_EXTENSION_DIR=/srv/slackquery/extensions
EMBEDDING_BACKEND=pytorch
PYTORCH_EMBEDDING_BASE_URL=http://embedding-host:11435
OLLAMA_EMBEDDING_BASE_URL=http://ollama-host:11434
EMBEDDING_MODEL=nomic-embed-text:v1.5
EMBEDDING_DIM=512
# Required when using the bundled authenticated PyTorch server.
EMBEDDING_API_KEY=replace-with-the-server-key
SLACKQUERY_EMBEDDING_MODEL_REVISION=40-character-model-commit
SLACKQUERY_EMBEDDING_CODE_REVISION=40-character-code-commit
```

### Pinning the model

Public defaults do not contain a model digest. Ollama reports its content digest.
The bundled PyTorch server instead requires reviewed Hugging Face commits for
both the model repository and its remote-code repository. Confirm the server
metadata and native dimension, then set the applicable values:

```dotenv
SLACKQUERY_EMBEDDING_MODEL_REVISION=sha256-digest-from-your-server
# PyTorch only:
SLACKQUERY_EMBEDDING_CODE_REVISION=40-character-code-commit
```

With a pin, `slackquery embedding-status` fails if the server silently replaces
the model. Without it, the generation is marked `unpinned`; this is convenient
for evaluation but not recommended for production reproducibility.

## Initial deployment

Create writable directories and assign them to the service account:

```bash
sudo install -d -o slackquery -g slackquery /srv/slackquery/state
sudo install -d -o slackquery -g slackquery /srv/slackquery/artifacts
sudo install -d -o slackquery -g slackquery /srv/slackquery/extensions
```

Run each pipeline stage explicitly:

```bash
slackquery project
slackquery embedding-status
slackquery embed
slackquery build
slackquery validate /srv/slackquery/artifacts/search-*.duckdb --checksum
slackquery publish /srv/slackquery/artifacts/search-*.duckdb
```

Use the exact candidate emitted by `slackquery build` when multiple artifacts
match the shell pattern.

Start serving and check probes:

```bash
slackquery run --host 0.0.0.0 --port 8181
curl -fsS http://mcp-host:8181/healthz
curl -fsS http://mcp-host:8181/readyz
```

## Docker Compose

The Compose file mounts the canonical database read-only and stores state,
artifacts, and extensions in operator-selected host directories. Set these in
`.env`:

```dotenv
SLACKQUERY_CANONICAL_DB_HOST=/path/to/slackpipe.duckdb
SLACKQUERY_STATE_DIR_HOST=/srv/slackquery/state
SLACKQUERY_ARTIFACT_DIR_HOST=/srv/slackquery/artifacts
SLACKQUERY_EXTENSION_DIR_HOST=/srv/slackquery/extensions
SLACKQUERY_BIND_ADDRESS=127.0.0.1
SLACKQUERY_MCP_PORT=8181
```

Then run:

```bash
docker compose up --build -d
docker compose ps
```

The example binds to loopback by default. Use a reverse proxy for TLS and remote
access rather than exposing unauthenticated plain HTTP.

## systemd

[`deploy/slackquery.service`](../deploy/slackquery.service) is a hardened example.
Install the application under `/srv/slackquery/app`, put settings in
`/etc/slackquery/slackquery.env`, correct `ReadOnlyPaths` for the canonical file,
and then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now slackquery
sudo systemctl status slackquery
sudo journalctl -u slackquery --since today
```

Do not put tokens directly in the unit. Restrict the environment file to the
service account and administrators.

## Dagster

The package exports `slackquery.definitions:definitions`, containing:

1. `search/document_projection`
2. `search/message_embeddings`
3. `search/artifact_candidate`
4. blocking candidate integrity checks
5. `search/published_artifact`

The included hourly schedule runs the asset job. In another Dagster deployment,
load the module as a code location and inject local paths/endpoints through its
environment. Keep only one writer active for a given state database and artifact
directory.

## Backend switching

Select the transport with:

```dotenv
EMBEDDING_BACKEND=ollama
```

or:

```dotenv
EMBEDDING_BACKEND=pytorch
```

Run `slackquery embedding-status` after every switch. Reuse existing vectors only
when model revision, native dimension, prefixes, truncation, normalization, and
text recipe are identical. A different generation must be re-embedded and
published as a new artifact.

## Routine reconciliation

Run the following sequence manually or through Dagster:

```bash
slackquery project
slackquery embed
slackquery build
slackquery validate /srv/slackquery/artifacts/CANDIDATE.duckdb --checksum
slackquery publish /srv/slackquery/artifacts/CANDIDATE.duckdb
```

Expected properties:

- unchanged source rows do not create new embedding work;
- transient failures remain retryable and visible in state;
- a failed build or validation does not alter `current.duckdb`;
- publication retains prior artifacts according to `SLACKQUERY_RETAIN_ARTIFACTS`.

## Health and troubleshooting

### Embedding endpoint

```bash
slackquery embedding-status
curl -fsS -H "Authorization: Bearer $EMBEDDING_API_KEY" \
  http://embedding-host:11435/health
curl -fsS -H "Authorization: Bearer $EMBEDDING_API_KEY" \
  http://embedding-host:11435/api/tags
```

The bundled PyTorch service is an unaudited sandbox component. Keep it behind a
host firewall on a trusted network even when API-key authentication is enabled.
See the [dedicated deployment guide](pytorch-embedding-server.md).

Common failures:

| Symptom | Likely cause | Action |
|---|---|---|
| Connection or timeout error | Backend unavailable or URL wrong | Check DNS, routing, endpoint URL, and service logs. |
| Model absent | Configured model is not loaded | Pull/load the model or change `EMBEDDING_MODEL`. |
| Digest mismatch | Server model differs from the pin | Investigate replacement; update the pin only after validation. |
| Native dimension mismatch | Incompatible model/server metadata | Do not publish; use a compatible model. |
| Non-CUDA PyTorch health | PyTorch server is not using its required device | Fix server device configuration or use Ollama. |

### MCP readiness

```bash
curl -i http://mcp-host:8181/healthz
curl -i http://mcp-host:8181/readyz
```

If liveness succeeds but readiness fails:

1. verify `SLACKQUERY_CURRENT_LINK` exists and resolves;
2. verify the service account can read the target and parent directories;
3. run `slackquery validate` against the selected artifact;
4. confirm the DuckDB FTS extension is available;
5. inspect application logs for the exact open/validation error.

### Projection or state failure

- Confirm the canonical file exists and is readable.
- Confirm no migration or writer is replacing it during projection.
- Check free disk space and ownership for state and artifacts.
- Do not repair source tables from Slackquery; resolve canonical issues in the
  owning ingestion system.

## Rollback

Stop publication while investigating. Validate a retained artifact, atomically
point `current.duckdb` at it, and restart or wait for readers to reopen:

```bash
slackquery validate /srv/slackquery/artifacts/KNOWN_GOOD.duckdb --checksum
slackquery publish /srv/slackquery/artifacts/KNOWN_GOOD.duckdb
```

Never replace an artifact in place. Build and publish a new immutable file.

## Backup and retention

- Back up state to preserve completed vectors and avoid expensive recomputation.
- Back up at least one validated artifact independently of the live selector.
- Treat the canonical archive according to its owning project's backup policy.
- Exclude bearer tokens and environment files from repository and artifact
  backups unless the backup system provides secret handling.
- Periodically test restoration and checksum validation.

## Security checklist

- canonical database mounted/readable only as needed;
- dedicated unprivileged service account;
- bearer token enabled for non-isolated networks;
- TLS terminated before traffic leaves a trusted network;
- ingress restricted to intended clients;
- state database unavailable to the MCP process where practical;
- `.env` and logs free of committed secrets;
- dependencies and base image updated regularly;
- published artifact contents treated as sensitive archive data.
