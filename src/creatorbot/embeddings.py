"""Optional dense embeddings.

The bot works without this — SQLite FTS5/BM25 handles the lexical case well.
Setting VOYAGE_API_KEY layers semantic recall on top, which mostly helps when
someone asks a paraphrased question ("who should I put on my rainbow team")
that shares no keywords with the transcript.

Voyage is used because it's Anthropic's recommended embeddings partner and is a
plain HTTP call — no torch, no model download, works on any Python.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol

import httpx

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_VOYAGE_MODEL = "voyage-3.5"
# Voyage caps batch size; stay well under it.
BATCH_SIZE = 96


class Embedder(Protocol):
    model: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class NullEmbedder:
    """No-op embedder — retrieval falls back to lexical-only."""

    model = "none"
    enabled = False

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return []

    def embed_query(self, text: str) -> list[float] | None:
        return None


class VoyageEmbedder:
    """Voyage AI embeddings over HTTP."""

    enabled = True

    def __init__(self, api_key: str, model: str = DEFAULT_VOYAGE_MODEL, timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = list(texts[i : i + BATCH_SIZE])
            resp = self._client.post(
                VOYAGE_URL,
                json={"input": batch, "model": self.model, "input_type": input_type},
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            # The API may return out of order; index is authoritative.
            ordered = sorted(data, key=lambda d: d.get("index", 0))
            out.extend(d["embedding"] for d in ordered)
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "document") if texts else []

    def embed_query(self, text: str) -> list[float] | None:
        vecs = self._embed([text], "query")
        return vecs[0] if vecs else None

    def close(self) -> None:
        self._client.close()


def get_embedder(model: str | None = None) -> Embedder:
    """Return a Voyage embedder if a key is present and numpy is installed."""
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        return NullEmbedder()
    try:
        import numpy  # noqa: F401
    except ImportError:
        # Without numpy we can store vectors but can't search them; don't
        # pretend dense retrieval is on.
        return NullEmbedder()
    return VoyageEmbedder(api_key, model or os.getenv("VOYAGE_MODEL", DEFAULT_VOYAGE_MODEL))
