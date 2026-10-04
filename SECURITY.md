# Security policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or data exposure. Use
the repository host's private vulnerability-reporting feature. If that feature
is unavailable, contact the maintainers through a private channel listed by the
repository host.

Include affected versions, impact, reproduction steps, and any proposed
mitigation. Do not include real Slack content, credentials, private endpoints,
or database files. Maintainers will acknowledge reports as capacity permits and
coordinate disclosure after a fix is available.

## Supported versions

Security fixes target the latest release and the default branch. Deployments
should track current Python, DuckDB, MCP SDK, Dagster, and container security
updates.

## Deployment responsibilities

Slackquery can expose sensitive archived communications. Operators must:

- restrict network access and terminate TLS for remote connections;
- enable bearer authentication outside a strictly isolated network;
- protect `.env`, state databases, artifacts, backups, and logs;
- run under an unprivileged account with the canonical database read-only;
- review archive access, retention, privacy, and legal requirements;
- rotate credentials and update dependencies regularly.

Health endpoints are intentionally unauthenticated and should reveal only service
status. See [docs/operations.md](docs/operations.md) for hardening guidance.
