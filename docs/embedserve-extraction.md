# Embedserve extraction

SlackQuery retains its original embedding-server implementation at
`src/slackquery/embedding_server.py` as a frozen legacy copy. It remains available for
provenance and compatibility review, but it is no longer authoritative for deployment.

The independent replacement is:

- local: `/Users/tmaerz/projects/embedserve`
- GitHub: <https://github.com/thomasmaerz/embedserve>
- wiki: <https://github.com/thomasmaerz/embedserve/wiki>

The `embedserve` repository is authoritative for server code, deployment, operations,
and future model scheduling. SlackQuery remains authoritative for its client behavior:
Nomic task prefixes, first-512-dimension Matryoshka truncation, and L2 normalization.

The extraction deliberately did not remove, rewrite, or reformat the legacy server,
its original tests, service unit, or guide. Their pre-extraction SHA-256 values are
recorded in Embedserve's provenance and release evidence.

SlackQuery retries only authenticated `503 MODEL_BUSY` embedding responses. Retries
honor `Retry-After` as a minimum, add bounded exponential full jitter, preserve
cancellation, and stop at both an attempt limit and total deadline. Other HTTP failures
remain subject only to the existing durable worker retry policy.
