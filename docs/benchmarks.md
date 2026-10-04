# Benchmarks

These reference measurements predate thread-context and attachment-chunk
projection and use a representative **46k-message corpus** with exact cosine scan
over normalized 512-dimensional vectors. They are not current Gold deployment
numbers or hardware guarantees; reproduce them in your environment before
setting service-level objectives.

## Embedding throughput

A local CUDA-capable PyTorch embedding server was measured with synthetic short
document inputs after warmup:

| Batch | Elapsed | Throughput |
|---:|---:|---:|
| 1 | 36.3 ms | 27.6 docs/s |
| 32 | 155.6 ms | 205.7 docs/s |
| 128 | 602.6 ms | 212.4 docs/s |

Those larger synthetic batches are not a safe production default for every GPU.
Use `SLACKQUERY_PYTORCH_EMBEDDING_BATCH_SIZE` to cap CUDA requests independently
and validate sustained operation on the target device.

The corpus had approximately 46k successful content-addressed vectors. Switching
transport between compatible Ollama and PyTorch servers did not invalidate them
because model weights, prefixes, first-512 Matryoshka truncation, and L2
normalization remained identical.

## Retrieval latency

Eight representative queries, three iterations each:

| Mode | Requests | p50 | p95 | Maximum |
|---|---:|---:|---:|---:|
| Lexical BM25 | 24 | 70.62 ms | 79.10 ms | 84.05 ms |
| Semantic exact cosine | 24 | 219.67 ms | 259.97 ms | 301.60 ms |
| Hybrid RRF | 24 | 266.94 ms | 309.69 ms | 329.19 ms |

These values measure latency, not relevance. Query embedding is included in
semantic and hybrid timings. Network topology, model server concurrency, storage,
DuckDB version, and document length can materially change results.

## Artifact build

Building and indexing the representative message-only immutable DuckDB artifact
took 6.75 seconds. Checksum validation and a fresh read-only open completed before
publication.

## Reproducing results

Create a UTF-8 file with one query per line and run:

```bash
slackquery benchmark queries.txt --iterations 3
```

The CLI benchmark currently measures lexical retrieval. For complete comparisons,
record query embedding and database timings separately for semantic and hybrid
modes, warm each service first, and report:

- corpus size rounded to a suitable public precision;
- vector dimension and normalization policy;
- batch size and input-length distribution;
- warmup and iteration counts;
- p50, p95, and maximum latency;
- relevant software versions;
- generalized compute class, without machine identifiers.

Use a human-judged query set before drawing relevance conclusions or replacing
exact scan with approximate indexing.
