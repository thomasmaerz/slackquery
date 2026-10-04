# MCP client setup

Slackquery is a remote MCP server using **Streamable HTTP**. This guide uses the
placeholder endpoint:

```text
http://mcp-host:8181/mcp
```

Replace `mcp-host` with your DNS name or reverse-proxy address. Do not configure
Slackquery as a local stdio executable. Client products use different names and
schemas for MCP settings, so consult the client's current documentation while
preserving the transport concepts below.

## Portable configuration model

Every client configuration needs:

- a server name, such as `slackquery`;
- transport type `streamable-http`, `http`, or the client's equivalent;
- URL `http://mcp-host:8181/mcp`;
- an optional `Authorization` header when bearer authentication is enabled.

Conceptual JSON—not guaranteed to match a specific client's schema:

```json
{
  "mcpServers": {
    "slackquery": {
      "transport": "streamable-http",
      "url": "http://mcp-host:8181/mcp"
    }
  }
}
```

With authentication, reference a secret rather than committing a token:

```json
{
  "mcpServers": {
    "slackquery": {
      "transport": "streamable-http",
      "url": "http://mcp-host:8181/mcp",
      "headers": {
        "Authorization": "Bearer ${SLACKQUERY_BEARER_TOKEN}"
      }
    }
  }
}
```

Environment interpolation syntax is client-specific. If unsupported, use the
client's keychain, secret store, managed policy, or protected local settings.

## Connectivity checks

Verify the endpoint before changing client configuration:

```bash
curl -fsS http://mcp-host:8181/healthz
curl -fsS http://mcp-host:8181/readyz
```

- `/healthz` returning `200` means the HTTP process is alive.
- `/readyz` returning `200` means the published artifact can be opened.
- `/readyz` returning `503` usually means no artifact is published, the selector
  is stale, permissions are wrong, or the artifact cannot be validated.

If bearer authentication is enabled, health probes remain public while MCP calls
require the header:

```bash
curl -H "Authorization: Bearer $SLACKQUERY_BEARER_TOKEN" \
  http://mcp-host:8181/mcp
```

## Recommended client workflow

1. Call `list_slack_scopes` when valid workspace or channel filters are unknown.
2. Call `search_slack` with `mode="hybrid"` and the narrowest known filters.
3. Retry exact identifiers, errors, filenames, URLs, or quotes in `lexical` mode.
4. Use `semantic` mode for paraphrased concepts when lexical matching is weak.
5. Expand only promising results with `get_slack_message` or `get_slack_thread`.
6. Treat cursors as opaque and do not reuse them with another query or filter set.
7. Cite source metadata and permalinks when available.

## Network security

Plain HTTP is appropriate only on a trusted local network. For access across
hosts or networks:

- terminate TLS in a reverse proxy or service mesh;
- set `SLACKQUERY_BEARER_TOKEN` from a secret manager;
- restrict ingress to intended clients;
- avoid logging authorization headers;
- rotate tokens and test readiness after proxy changes.

See [the operations runbook](../operations.md) for server configuration.
