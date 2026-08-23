"""The creator's own videos.

Listing uses yt-dlp's flat playlist extraction (cheap — one request for the
whole channel). Transcripts come from youtube-transcript-api, falling back to
yt-dlp's subtitle download when that fails.

Only the text is stored. No video or audio is downloaded.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from ..chunking import chunk_transcript
from ..store import Chunk, Document
from .base import Source, ToolResult

log = logging.getLogger(__name__)

_VTT_TS = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
_TAG = re.compile(r"<[^>]+>")


class YouTubeTranscriptSource(Source):
    ingests = True

    @property
    def name(self) -> str:
        return "youtube"

    # -- config ---------------------------------------------------------------

    @property
    def _yt(self) -> dict[str, Any]:
        return self.persona.creator.get("youtube", {}) or {}

    @property
    def channel_url(self) -> str:
        yt = self._yt
        if yt.get("channel_id"):
            return f"https://www.youtube.com/channel/{yt['channel_id']}"
        if yt.get("channel_url"):
            return yt["channel_url"].rstrip("/")
        if yt.get("handle"):
            return f"https://www.youtube.com/{yt['handle'].lstrip('@') and yt['handle']}"
        raise ValueError("persona.creator.youtube needs channel_id, channel_url or handle")

    @property
    def languages(self) -> list[str]:
        return list(self._yt.get("transcript_languages") or ["en"])

    # -- access options -------------------------------------------------------
    #
    # YouTube rate-limits and IP-blocks unauthenticated transcript requests,
    # especially from datacenter IPs. Two supported ways through, both of which
    # are ordinary documented usage rather than anything sneaky:
    #
    #   YOUTUBE_COOKIES_FROM_BROWSER=chrome   use your own logged-in session
    #   YOUTUBE_COOKIES_FILE=/path/cookies.txt
    #   YOUTUBE_PROXY=http://user:pass@host:port
    #
    # See docs/TROUBLESHOOTING.md.

    def _ytdlp_auth_opts(self) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        browser = os.getenv("YOUTUBE_COOKIES_FROM_BROWSER", "").strip()
        cookie_file = os.getenv("YOUTUBE_COOKIES_FILE", "").strip()
        proxy = os.getenv("YOUTUBE_PROXY", "").strip()
        if browser:
            # yt-dlp wants a tuple: (browser, profile, keyring, container)
            opts["cookiesfrombrowser"] = (browser, None, None, None)
        elif cookie_file:
            opts["cookiefile"] = cookie_file
        if proxy:
            opts["proxy"] = proxy
        return opts

    def _transcript_api_client(self):
        """Build a YouTubeTranscriptApi honouring proxy/cookie settings."""
        from youtube_transcript_api import YouTubeTranscriptApi

        kwargs: dict[str, Any] = {}
        proxy = os.getenv("YOUTUBE_PROXY", "").strip()
        webshare_user = os.getenv("WEBSHARE_PROXY_USERNAME", "").strip()
        cookie_file = os.getenv("YOUTUBE_COOKIES_FILE", "").strip()

        try:
            if webshare_user:
                from youtube_transcript_api.proxies import WebshareProxyConfig

                kwargs["proxy_config"] = WebshareProxyConfig(
                    proxy_username=webshare_user,
                    proxy_password=os.getenv("WEBSHARE_PROXY_PASSWORD", ""),
                )
            elif proxy:
                from youtube_transcript_api.proxies import GenericProxyConfig

                kwargs["proxy_config"] = GenericProxyConfig(
                    http_url=proxy, https_url=proxy
                )
            if cookie_file:
                kwargs["cookie_path"] = cookie_file
        except (ImportError, TypeError) as exc:
            log.debug("proxy/cookie config unsupported by this version: %s", exc)
            kwargs = {}

        try:
            return YouTubeTranscriptApi(**kwargs)
        except TypeError:
            # Older/newer signature — fall back to the default constructor.
            return YouTubeTranscriptApi()

    # -- ingestion ------------------------------------------------------------

    def list_videos(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Newest-first video metadata for the channel, without downloading."""
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("yt-dlp is required: pip install yt-dlp") from exc

        max_videos = limit or int(self._yt.get("max_videos", 200))
        opts = {
            "extract_flat": "in_playlist",
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
            "playlistend": max_videos,
            **self._ytdlp_auth_opts(),
        }
        url = f"{self.channel_url}/videos"
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

        entries = (info or {}).get("entries") or []
        min_duration = int(self._yt.get("min_duration_seconds", 0))

        videos = []
        for e in entries:
            if not e or not e.get("id"):
                continue
            duration = e.get("duration") or 0
            if min_duration and duration and duration < min_duration:
                continue
            videos.append(
                {
                    "id": e["id"],
                    "title": e.get("title") or "",
                    "url": f"https://www.youtube.com/watch?v={e['id']}",
                    "duration": duration,
                    "view_count": e.get("view_count"),
                    "upload_date": e.get("upload_date"),  # often absent when flat
                }
            )
        return videos[:max_videos]

    def fetch_transcript(self, video_id: str) -> list[dict[str, Any]]:
        """Timed snippets for a video, or [] if it has no usable captions."""
        snippets = self._transcript_via_api(video_id)
        if snippets:
            return snippets
        return self._transcript_via_ytdlp(video_id)

    def _transcript_via_api(self, video_id: str) -> list[dict[str, Any]]:
        try:
            api = self._transcript_api_client()
            fetched = api.fetch(video_id, languages=self.languages)
            return [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in fetched
            ]
        except ImportError:
            return []
        except Exception as exc:  # library raises a wide family of errors
            name = type(exc).__name__
            if name in {"IpBlocked", "RequestBlocked"}:
                # Surface this loudly once — it's the difference between "this
                # video has no captions" and "every video will fail".
                self._blocked = True
            log.debug("transcript api failed for %s: %s", video_id, name)
            return []

    def _transcript_via_ytdlp(self, video_id: str) -> list[dict[str, Any]]:
        """Fallback: ask yt-dlp for the caption track and parse the VTT."""
        try:
            import httpx
            from yt_dlp import YoutubeDL
        except ImportError:
            return []

        opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": self.languages,
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
            **self._ytdlp_auth_opts(),
        }
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}", download=False
                )
            if not info:
                return []
            tracks = {**(info.get("subtitles") or {}), **(info.get("automatic_captions") or {})}
            url = None
            for lang in self.languages:
                for candidate in tracks.get(lang, []):
                    if candidate.get("ext") in {"vtt", "srt"}:
                        url = candidate["url"]
                        break
                if url:
                    break
            if not url:
                return []
            text = httpx.get(url, timeout=30.0, follow_redirects=True).text
            return _parse_vtt(text)
        except Exception as exc:
            log.debug("yt-dlp transcript fallback failed for %s: %s", video_id, exc)
            return []

    #: Set when YouTube blocks the IP rather than when a video lacks captions.
    _blocked: bool = False

    def ingest(self, *, limit: int | None = None, refresh: bool = False) -> dict[str, Any]:
        videos = self.list_videos(limit=limit)
        existing = self.store.document_ids(source=self.name)

        added = skipped = no_captions = 0
        for i, video in enumerate(videos, 1):
            doc_id = f"yt:{video['id']}"
            if doc_id in existing and not refresh:
                skipped += 1
                continue

            snippets = self.fetch_transcript(video["id"])
            if not snippets:
                no_captions += 1
                log.info("[%d/%d] no captions: %s", i, len(videos), video["title"][:70])
                continue

            published = None
            if video.get("upload_date") and len(str(video["upload_date"])) == 8:
                d = str(video["upload_date"])
                published = f"{d[:4]}-{d[4:6]}-{d[6:]}"

            doc = Document(
                id=doc_id,
                source=self.name,
                title=video["title"],
                url=video["url"],
                published_at=published,
                meta={
                    "video_id": video["id"],
                    "duration": video.get("duration"),
                    "view_count": video.get("view_count"),
                },
            )
            chunks = chunk_transcript(
                doc_id,
                snippets,
                base_meta={"video_id": video["id"], "video_title": video["title"]},
            )
            if not chunks:
                no_captions += 1
                continue

            self.store.upsert_document(doc, chunks)
            added += 1
            log.info(
                "[%d/%d] %s (%d chunks)", i, len(videos), video["title"][:70], len(chunks)
            )
            time.sleep(0.2)  # be gentle with YouTube

        summary = {
            "source": self.name,
            "videos_listed": len(videos),
            "added": added,
            "skipped_existing": skipped,
            "no_captions": no_captions,
        }
        if self._blocked and added == 0:
            summary["ERROR"] = (
                "YouTube is blocking transcript requests from this IP, so nothing "
                "was ingested. This is not a bug in the bot. Fix it by using your "
                "own logged-in session or a proxy:\n"
                "      YOUTUBE_COOKIES_FROM_BROWSER=chrome   (in .env)\n"
                "      YOUTUBE_COOKIES_FILE=/path/to/cookies.txt\n"
                "      YOUTUBE_PROXY=http://user:pass@host:port\n"
                "    See docs/TROUBLESHOOTING.md."
            )
        return summary

    # -- question-time tool ---------------------------------------------------

    def tool_spec(self) -> dict[str, Any]:
        return {
            "name": self.tool_name,
            "description": self.cfg.description
            or f"Search {self.persona.display_name}'s video transcripts.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to look for. Use the specific names a viewer would "
                            "say out loud (unit names, event names, banner names)."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many passages to return (1-10, default 6).",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        }

    def run_tool(self, tool_input: dict[str, Any]) -> ToolResult:
        query = (tool_input.get("query") or "").strip()
        if not query:
            return ToolResult("No query supplied.", is_error=True)
        limit = max(1, min(int(tool_input.get("limit") or 6), 10))

        query_vec = None
        embedder = getattr(self, "_embedder", None)
        if embedder is not None and getattr(embedder, "enabled", False):
            try:
                query_vec = embedder.embed_query(query)
            except Exception as exc:
                log.warning("query embedding failed, using lexical only: %s", exc)

        hits: list[Chunk] = self.store.hybrid_search(
            query, limit=limit, source=self.name, query_vec=query_vec
        )
        if not hits:
            return ToolResult(
                f"No passages in the ingested transcripts match {query!r}. "
                "The corpus may not cover this topic — say so rather than inventing a quote."
            )

        blocks, citations = [], []
        for hit in hits:
            when = hit.meta.get("timestamp", "?")
            date = f", {hit.doc_published_at}" if hit.doc_published_at else ""
            blocks.append(
                f'--- "{hit.doc_title}" at {when}{date}\n'
                f"{hit.text}\n"
                f"link: {hit.deep_link()}"
            )
            citations.append(f"[{hit.doc_title} @ {when}]({hit.deep_link()})")

        header = (
            f"{len(hits)} transcript passage(s). These are verbatim — quote or "
            f"paraphrase them, but do not attribute anything else to him.\n\n"
        )
        return ToolResult(header + "\n\n".join(blocks), citations=citations)

    def health(self) -> str:
        n = len(self.store.document_ids(source=self.name))
        return f"{n} videos in corpus" if n else "no videos ingested yet — run `creatorbot ingest`"


def _parse_vtt(text: str) -> list[dict[str, Any]]:
    """Minimal WebVTT/SRT parser -> timed snippets."""
    snippets: list[dict[str, Any]] = []
    current_start: float | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf, current_start
        if current_start is not None and buf:
            body = " ".join(buf).strip()
            if body:
                snippets.append({"text": body, "start": current_start, "duration": 0.0})
        buf = []

    for raw in text.splitlines():
        line = raw.strip()
        m = _VTT_TS.search(line)
        if m:
            flush()
            h, mm, s, ms = (int(m.group(i)) for i in range(1, 5))
            current_start = h * 3600 + mm * 60 + s + ms / 1000.0
            continue
        if not line or line in {"WEBVTT"} or line.isdigit():
            continue
        if line.startswith(("Kind:", "Language:", "NOTE")):
            continue
        buf.append(_TAG.sub("", line))

    flush()

    # Auto-captions repeat the previous cue as a rolling window; drop exact dupes.
    deduped: list[dict[str, Any]] = []
    for s in snippets:
        if deduped and s["text"] == deduped[-1]["text"]:
            continue
        deduped.append(s)
    return deduped
