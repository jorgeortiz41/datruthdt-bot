"""Corpus storage and hybrid retrieval, on nothing but stdlib SQLite.

Design notes
------------
* Lexical search uses SQLite's built-in FTS5 (BM25). It needs no extra
  dependency, no server and no model download, and it is genuinely good at the
  thing this bot does most: matching proper nouns like "Ultra Instinct Goku".
* Dense search is optional. If a `VOYAGE_API_KEY` is set and numpy is
  installed, embeddings are stored alongside and fused with the lexical
  results via Reciprocal Rank Fusion.
* Both paths return the same `Chunk` objects, so the rest of the codebase never
  branches on which one is active.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    url          TEXT NOT NULL DEFAULT '',
    published_at TEXT,
    fetched_at   TEXT NOT NULL,
    meta         TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    rowid   INTEGER PRIMARY KEY,
    uid     TEXT UNIQUE NOT NULL,
    doc_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text    TEXT NOT NULL,
    meta    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Keep the FTS index in lockstep with the chunks table.
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_uid TEXT PRIMARY KEY REFERENCES chunks(uid) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vec       BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# FTS5 treats these as operators; strip them out of user-supplied text.
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class Document:
    id: str
    source: str
    title: str = ""
    url: str = ""
    published_at: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    uid: str
    doc_id: str
    ordinal: int
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    # Populated on retrieval.
    score: float = 0.0
    doc_title: str = ""
    doc_url: str = ""
    doc_published_at: str | None = None

    def citation(self) -> str:
        """A human-readable pointer back to the source."""
        url = self.deep_link()
        if self.doc_title and url:
            return f"{self.doc_title} — {url}"
        return self.doc_title or url or self.doc_id

    def deep_link(self) -> str:
        """Video URL with a `?t=` jump to where this chunk starts, when known."""
        start = self.meta.get("start")
        if self.doc_url and isinstance(start, (int, float)) and "youtube.com" in self.doc_url:
            sep = "&" if "?" in self.doc_url else "?"
            return f"{self.doc_url}{sep}t={int(start)}"
        return self.doc_url


def fts_query(text: str, max_terms: int = 24) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    Each token is quoted (so `"goku"` can't be read as a column filter or an
    operator) and joined with OR, letting BM25 rank by how many matched.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    tokens = [t for t in tokens if len(t) > 1][:max_terms]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


class CorpusStore:
    """SQLite-backed document/chunk store with hybrid search."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._vec_cache: tuple[list[str], Any] | None = None

    def close(self) -> None:
        self.conn.close()

    # -- writing --------------------------------------------------------------

    def upsert_document(self, doc: Document, chunks: Sequence[Chunk]) -> None:
        """Replace a document and all of its chunks atomically."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE id = ?", (doc.id,))
            self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.id,))
            self.conn.execute(
                "INSERT INTO documents (id, source, title, url, published_at, fetched_at, meta)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.id,
                    doc.source,
                    doc.title,
                    doc.url,
                    doc.published_at,
                    now,
                    json.dumps(doc.meta, ensure_ascii=False),
                ),
            )
            self.conn.executemany(
                "INSERT INTO chunks (uid, doc_id, ordinal, text, meta) VALUES (?, ?, ?, ?, ?)",
                [
                    (c.uid, doc.id, c.ordinal, c.text, json.dumps(c.meta, ensure_ascii=False))
                    for c in chunks
                ],
            )
        self._vec_cache = None

    def has_document(self, doc_id: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,))
        return cur.fetchone() is not None

    def document_ids(self, source: str | None = None) -> set[str]:
        if source:
            cur = self.conn.execute("SELECT id FROM documents WHERE source = ?", (source,))
        else:
            cur = self.conn.execute("SELECT id FROM documents")
        return {r["id"] for r in cur.fetchall()}

    def store_embeddings(
        self, model: str, items: Iterable[tuple[str, Sequence[float]]]
    ) -> int:
        import array

        rows = []
        for uid, vec in items:
            buf = array.array("f", vec)
            rows.append((uid, model, len(vec), buf.tobytes()))
        if not rows:
            return 0
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (chunk_uid, model, dim, vec)"
                " VALUES (?, ?, ?, ?)",
                rows,
            )
        self._vec_cache = None
        return len(rows)

    def chunks_missing_embeddings(self, limit: int | None = None) -> list[Chunk]:
        sql = (
            "SELECT c.uid, c.doc_id, c.ordinal, c.text, c.meta FROM chunks c"
            " LEFT JOIN embeddings e ON e.chunk_uid = c.uid"
            " WHERE e.chunk_uid IS NULL ORDER BY c.rowid"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [self._row_to_chunk(r) for r in self.conn.execute(sql).fetchall()]

    def set_state(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, value)
            )

    def get_state(self, key: str, default: str | None = None) -> str | None:
        cur = self.conn.execute("SELECT value FROM state WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default

    # -- reading --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        def one(sql: str) -> int:
            return int(self.conn.execute(sql).fetchone()[0])

        by_source = {
            r["source"]: r["n"]
            for r in self.conn.execute(
                "SELECT source, COUNT(*) AS n FROM documents GROUP BY source"
            ).fetchall()
        }
        return {
            "documents": one("SELECT COUNT(*) FROM documents"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "embeddings": one("SELECT COUNT(*) FROM embeddings"),
            "documents_by_source": by_source,
            "db_path": str(self.path),
            "db_size_mb": round(self.path.stat().st_size / 1e6, 2)
            if self.path.exists()
            else 0.0,
        }

    def sample_chunks(self, n: int, source: str | None = None, min_chars: int = 400) -> list[Chunk]:
        """Random-ish chunks, used to build the style profile and exemplars."""
        sql = (
            "SELECT c.uid, c.doc_id, c.ordinal, c.text, c.meta,"
            " d.title AS doc_title, d.url AS doc_url, d.published_at"
            " FROM chunks c JOIN documents d ON d.id = c.doc_id"
            " WHERE LENGTH(c.text) >= ?"
        )
        params: list[Any] = [min_chars]
        if source:
            sql += " AND d.source = ?"
            params.append(source)
        sql += " ORDER BY RANDOM() LIMIT ?"
        params.append(n)
        return [self._row_to_chunk(r) for r in self.conn.execute(sql, params).fetchall()]

    def lexical_search(
        self, query: str, limit: int = 20, source: str | None = None
    ) -> list[Chunk]:
        match = fts_query(query)
        if not match:
            return []
        sql = (
            "SELECT c.uid, c.doc_id, c.ordinal, c.text, c.meta,"
            " d.title AS doc_title, d.url AS doc_url, d.published_at,"
            " bm25(chunks_fts) AS rank"
            " FROM chunks_fts"
            " JOIN chunks c ON c.rowid = chunks_fts.rowid"
            " JOIN documents d ON d.id = c.doc_id"
            " WHERE chunks_fts MATCH ?"
        )
        params: list[Any] = [match]
        if source:
            sql += " AND d.source = ?"
            params.append(source)
        # FTS5 bm25() is negative, more-negative = better.
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # Malformed MATCH despite sanitising — degrade to no results rather
            # than take the bot down.
            return []

        out = []
        for r in rows:
            c = self._row_to_chunk(r)
            c.score = -float(r["rank"])
            out.append(c)
        return out

    def dense_search(
        self, query_vec: Sequence[float], limit: int = 20, source: str | None = None
    ) -> list[Chunk]:
        """Brute-force cosine over stored vectors. Requires numpy."""
        try:
            import numpy as np
        except ImportError:
            return []

        uids, matrix = self._load_vectors()
        if matrix is None or not len(uids):
            return []

        q = np.asarray(query_vec, dtype="float32")
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm

        scores = matrix @ q
        k = min(limit * 3 if source else limit, len(uids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[scores[top].argsort()[::-1]]

        ranked = [(uids[i], float(scores[i])) for i in top]
        chunks = self._chunks_by_uid([u for u, _ in ranked], source=source)
        score_by_uid = dict(ranked)
        for c in chunks:
            c.score = score_by_uid.get(c.uid, 0.0)
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:limit]

    def hybrid_search(
        self,
        query: str,
        limit: int = 8,
        source: str | None = None,
        query_vec: Sequence[float] | None = None,
        candidates: int = 30,
    ) -> list[Chunk]:
        """Lexical + optional dense, fused with Reciprocal Rank Fusion."""
        lex = self.lexical_search(query, limit=candidates, source=source)
        dense = (
            self.dense_search(query_vec, limit=candidates, source=source)
            if query_vec is not None
            else []
        )
        if not dense:
            return lex[:limit]
        if not lex:
            return dense[:limit]

        k = 60  # standard RRF damping
        fused: dict[str, float] = {}
        best: dict[str, Chunk] = {}
        for ranking in (lex, dense):
            for rank, chunk in enumerate(ranking):
                fused[chunk.uid] = fused.get(chunk.uid, 0.0) + 1.0 / (k + rank + 1)
                best.setdefault(chunk.uid, chunk)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for uid, score in ordered[:limit]:
            c = best[uid]
            c.score = score
            out.append(c)
        return out

    def neighbours(self, chunk: Chunk, window: int = 1) -> list[Chunk]:
        """Adjacent chunks, so a quote isn't cut off mid-thought."""
        lo, hi = chunk.ordinal - window, chunk.ordinal + window
        rows = self.conn.execute(
            "SELECT c.uid, c.doc_id, c.ordinal, c.text, c.meta,"
            " d.title AS doc_title, d.url AS doc_url, d.published_at"
            " FROM chunks c JOIN documents d ON d.id = c.doc_id"
            " WHERE c.doc_id = ? AND c.ordinal BETWEEN ? AND ? ORDER BY c.ordinal",
            (chunk.doc_id, lo, hi),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    # -- internals ------------------------------------------------------------

    def _chunks_by_uid(self, uids: Sequence[str], source: str | None = None) -> list[Chunk]:
        if not uids:
            return []
        placeholders = ",".join("?" for _ in uids)
        sql = (
            "SELECT c.uid, c.doc_id, c.ordinal, c.text, c.meta,"
            " d.title AS doc_title, d.url AS doc_url, d.published_at"
            f" FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.uid IN ({placeholders})"
        )
        params = list(uids)
        if source:
            sql += " AND d.source = ?"
            params.append(source)
        return [self._row_to_chunk(r) for r in self.conn.execute(sql, params).fetchall()]

    def _load_vectors(self):
        """Load and L2-normalise every stored vector once, then cache."""
        if self._vec_cache is not None:
            return self._vec_cache
        try:
            import numpy as np
        except ImportError:
            self._vec_cache = ([], None)
            return self._vec_cache

        rows = self.conn.execute(
            "SELECT chunk_uid, dim, vec FROM embeddings ORDER BY chunk_uid"
        ).fetchall()
        if not rows:
            self._vec_cache = ([], None)
            return self._vec_cache

        dim = rows[0]["dim"]
        uids, vectors = [], []
        for r in rows:
            if r["dim"] != dim:  # a model changed mid-corpus; skip the strays
                continue
            uids.append(r["chunk_uid"])
            vectors.append(np.frombuffer(r["vec"], dtype="float32"))

        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self._vec_cache = (uids, matrix)
        return self._vec_cache

    @staticmethod
    def _row_to_chunk(r: sqlite3.Row) -> Chunk:
        keys = r.keys()
        return Chunk(
            uid=r["uid"],
            doc_id=r["doc_id"],
            ordinal=r["ordinal"],
            text=r["text"],
            meta=json.loads(r["meta"]) if r["meta"] else {},
            doc_title=r["doc_title"] if "doc_title" in keys else "",
            doc_url=r["doc_url"] if "doc_url" in keys else "",
            doc_published_at=r["published_at"] if "published_at" in keys else None,
        )
