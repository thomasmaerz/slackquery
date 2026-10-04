FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY sql ./sql
RUN uv sync --frozen --no-dev

FROM python:3.13-slim
RUN useradd --system --uid 10001 --create-home slackquery \
    && mkdir -p /srv/slackquery/extensions \
    && chown -R 10001:10001 /srv/slackquery/extensions
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --chown=10001:10001 src ./src
COPY --chown=10001:10001 sql ./sql
USER 10001
ENV PATH=/app/.venv/bin:$PATH PYTHONPATH=/app/src PYTHONUNBUFFERED=1
RUN python - <<'PY'
import duckdb

connection = duckdb.connect(
    config={"extension_directory": "/srv/slackquery/extensions"}
)
connection.execute("INSTALL fts")
connection.close()
PY
EXPOSE 8080 4001
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["slackquery", "run"]
