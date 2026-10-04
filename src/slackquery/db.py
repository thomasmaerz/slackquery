"""DuckDB connection and schema helpers."""

from __future__ import annotations

import fcntl
import hashlib
from pathlib import Path
from typing import Any, BinaryIO, cast

import duckdb

SQL_ROOT = Path(__file__).resolve().parent / "sql"
if not SQL_ROOT.exists():  # Editable source tree.
    SQL_ROOT = Path(__file__).resolve().parents[2] / "sql"


def path_sql(path: Path) -> str:
    """Return a safely quoted DuckDB path literal for ATTACH only."""
    return "'" + str(path.resolve()).replace("'", "''") + "'"


class _LockedStateConnection:
    def __init__(self, connection: duckdb.DuckDBPyConnection, lock: BinaryIO) -> None:
        self._connection = connection
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def close(self) -> None:
        try:
            self._connection.close()
        finally:
            self._lock.close()


def connect_state(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".maintenance.lock")
    lock = lock_path.open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        connection = duckdb.connect(str(path))
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET threads=1")
        connection.execute((SQL_ROOT / "001_state.sql").read_text())
    except Exception:
        lock.close()
        raise
    return cast(duckdb.DuckDBPyConnection, _LockedStateConnection(connection, lock))


def connect_readonly(
    path: Path, *, extension_dir: Path | None = None
) -> duckdb.DuckDBPyConnection:
    if not path.exists():
        raise FileNotFoundError(path)
    if extension_dir is None:
        return duckdb.connect(str(path), read_only=True)
    return duckdb.connect(
        str(path), read_only=True, config={"extension_directory": str(extension_dir)}
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
