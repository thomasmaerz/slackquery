# Contributing

Contributions are welcome through issues and pull requests.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the required checks before submitting a change:

```bash
pytest
ruff check .
mypy
```

## Guidelines

- Keep the canonical archive read-only.
- Preserve immutable artifact publication and rollback behavior.
- Add tests for behavior changes, especially retrieval filters, validation, and
  failure recovery.
- Keep MCP tools bounded and read-only; do not expose arbitrary SQL.
- Avoid committing archives, database files, credentials, private endpoints,
  personal information, machine identifiers, or local deployment state.
- Use environment settings for deployment-specific paths and endpoints.
- Update public documentation when configuration or tool contracts change.

## Pull requests

Keep pull requests focused. Explain the problem, design tradeoffs, tests run, and
any migration or compatibility impact. Changes to embedding generation identity,
projection recipes, artifact schemas, or cursor behavior require explicit
upgrade notes.

By contributing, you agree that your contributions are licensed under the MIT
License and that you will follow the [Code of Conduct](CODE_OF_CONDUCT.md).
