"""Stable identities for text sent to the embedding model."""

from __future__ import annotations

import hashlib

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def embedding_content_hash(text: str) -> str:
    """Hash the exact UTF-8 document input consumed by the model."""
    return hashlib.sha256((DOCUMENT_PREFIX + text).encode("utf-8")).hexdigest()
