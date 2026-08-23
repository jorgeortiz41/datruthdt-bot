"""Turning long text into retrievable chunks.

Transcripts get special handling: chunks carry the timestamp they start at, so
a citation can deep-link to the exact moment in the video.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

from .store import Chunk

DEFAULT_CHUNK_CHARS = 1100
DEFAULT_OVERLAP_CHARS = 180

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def make_uid(doc_id: str, ordinal: int, text: str) -> str:
    """Stable per-(document, position, content) id, so re-ingest is idempotent."""
    h = hashlib.sha1(f"{doc_id}:{ordinal}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:{ordinal}:{h}"


def chunk_text(
    doc_id: str,
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    base_meta: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split prose on sentence boundaries into overlapping windows."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    sentences = _SENTENCE_END.split(text)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        body = " ".join(buf).strip()
        if body:
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    uid=make_uid(doc_id, ordinal, body),
                    doc_id=doc_id,
                    ordinal=ordinal,
                    text=body,
                    meta=dict(base_meta or {}),
                )
            )
        # Carry the tail forward so context isn't severed at the seam.
        tail, tail_len = [], 0
        for s in reversed(buf):
            if tail_len >= overlap_chars:
                break
            tail.insert(0, s)
            tail_len += len(s) + 1
        buf, buf_len = tail, tail_len

    for sentence in sentences:
        if buf_len + len(sentence) > chunk_chars and buf:
            flush()
        buf.append(sentence)
        buf_len += len(sentence) + 1

    # Final flush without overlap carry-over.
    if buf:
        body = " ".join(buf).strip()
        if body:
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    uid=make_uid(doc_id, ordinal, body),
                    doc_id=doc_id,
                    ordinal=ordinal,
                    text=body,
                    meta=dict(base_meta or {}),
                )
            )
    return chunks


def chunk_transcript(
    doc_id: str,
    snippets: Sequence[dict[str, Any]],
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_seconds: float = 12.0,
    base_meta: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Group timed transcript snippets into chunks that remember their start time.

    `snippets` is a sequence of {"text": str, "start": float, "duration": float},
    which is what both youtube-transcript-api and our yt-dlp fallback produce.
    """
    cleaned = []
    for s in snippets:
        t = re.sub(r"\s+", " ", (s.get("text") or "")).strip()
        # Auto-captions sprinkle these in.
        if not t or t in {"[Music]", "[Applause]", "[Laughter]", "[__]"}:
            continue
        cleaned.append({"text": t, "start": float(s.get("start") or 0.0)})

    if not cleaned:
        return []

    chunks: list[Chunk] = []
    buf: list[dict[str, Any]] = []
    buf_len = 0

    def flush(carry: bool = True) -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        body = " ".join(s["text"] for s in buf).strip()
        if body:
            ordinal = len(chunks)
            meta = dict(base_meta or {})
            meta["start"] = round(buf[0]["start"], 1)
            meta["end"] = round(buf[-1]["start"], 1)
            meta["timestamp"] = _hms(buf[0]["start"])
            chunks.append(
                Chunk(
                    uid=make_uid(doc_id, ordinal, body),
                    doc_id=doc_id,
                    ordinal=ordinal,
                    text=body,
                    meta=meta,
                )
            )
        if not carry:
            buf, buf_len = [], 0
            return
        cutoff = buf[-1]["start"] - overlap_seconds
        tail = [s for s in buf if s["start"] >= cutoff]
        buf = tail
        buf_len = sum(len(s["text"]) + 1 for s in tail)

    for snippet in cleaned:
        if buf_len + len(snippet["text"]) > chunk_chars and buf:
            flush()
        buf.append(snippet)
        buf_len += len(snippet["text"]) + 1

    flush(carry=False)
    return chunks


def _hms(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
