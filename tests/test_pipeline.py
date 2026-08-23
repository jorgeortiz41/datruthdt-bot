"""End-to-end checks that don't need network or API keys."""

from __future__ import annotations

import re

import pytest

from creatorbot.chunking import chunk_text, chunk_transcript
from creatorbot.config import load_persona
from creatorbot.persona import build_system_prompt
from creatorbot.store import CorpusStore, Document, fts_query

TRANSCRIPT = [
    {"text": "yo what is up everybody it's your boy back with another one", "start": 0.0},
    {"text": "and today we are summoning on the brand new LR Gogeta banner", "start": 5.0},
    {"text": "bro this unit is absolutely busted let me tell you why", "start": 10.0},
    {"text": "he has a 200% lead for pure saiyans which is insane", "start": 15.0},
    {"text": "and the additional super attacks are guaranteed after turn three", "start": 20.0},
    {"text": "honestly though the banner itself is kind of a skip", "start": 25.0},
    {"text": "you are better off saving your stones for the anniversary", "start": 30.0},
    {"text": "that is just my opinion but I really think you should wait", "start": 35.0},
]


@pytest.fixture
def store(tmp_path):
    s = CorpusStore(tmp_path / "test.sqlite3")
    yield s
    s.close()


def test_fts_query_neutralises_operators():
    # FTS5 would choke on these if passed through raw.
    q = fts_query('goku OR NOT "cell" * (max)')
    assert q
    assert "*" not in q
    assert q.count('"') % 2 == 0


def test_fts_query_empty():
    assert fts_query("!!! ???") == ""


def test_chunk_transcript_carries_timestamps():
    chunks = chunk_transcript("yt:abc", TRANSCRIPT, chunk_chars=120)
    assert len(chunks) > 1
    assert all(c.meta["start"] >= 0 for c in chunks)
    assert chunks[0].meta["timestamp"] == "0:00"
    # Ordinals must be dense and ordered — retrieval relies on it.
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunk_text_overlaps():
    text = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = chunk_text("doc:1", text, chunk_chars=300, overlap_chars=80)
    assert len(chunks) > 3
    assert all(len(c.text) <= 500 for c in chunks)


def test_upsert_and_lexical_search(store):
    doc = Document(
        id="yt:abc",
        source="youtube",
        title="LR GOGETA SUMMONS",
        url="https://www.youtube.com/watch?v=abc",
        published_at="2026-08-01",
    )
    chunks = chunk_transcript("yt:abc", TRANSCRIPT, chunk_chars=150)
    store.upsert_document(doc, chunks)

    stats = store.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == len(chunks)

    hits = store.lexical_search("gogeta banner busted", limit=5)
    assert hits, "expected a lexical hit"
    assert hits[0].doc_title == "LR GOGETA SUMMONS"
    assert "t=" in hits[0].deep_link()


def test_upsert_is_idempotent(store):
    doc = Document(id="yt:abc", source="youtube", title="T", url="u")
    chunks = chunk_transcript("yt:abc", TRANSCRIPT, chunk_chars=150)
    store.upsert_document(doc, chunks)
    store.upsert_document(doc, chunks)
    assert store.stats()["chunks"] == len(chunks)


def test_hybrid_falls_back_to_lexical_without_vectors(store):
    doc = Document(id="yt:abc", source="youtube", title="T", url="u")
    store.upsert_document("yt:abc" and doc, chunk_transcript("yt:abc", TRANSCRIPT, chunk_chars=150))
    hits = store.hybrid_search("anniversary stones", limit=3, query_vec=None)
    assert hits


def test_search_no_results_is_empty_not_error(store):
    assert store.lexical_search("zzzzquux", limit=3) == []


def test_persona_loads_and_prompt_builds():
    persona = load_persona("datruthdt")
    assert persona.display_name == "DaTruthDT"
    assert persona.model == "claude-opus-5"

    prompt = build_system_prompt(persona, ["search_videos", "search_wiki"])
    # The honesty layer must survive prompt assembly.
    assert "never invent" in prompt.lower()
    assert "EZA" in prompt
    assert "search_wiki" in prompt


def test_enabled_sources_present():
    persona = load_persona("datruthdt")
    types = {s.type for s in persona.enabled_sources()}
    assert {"youtube_transcripts", "mediawiki", "web_search"} <= types


def test_card_site_reference_mode_never_fetches(tmp_path):
    from creatorbot.sources.cardsite import CardSiteSource

    persona = load_persona("datruthdt")
    store = CorpusStore(tmp_path / "t.sqlite3")
    cfg = persona.source_of_type("dokkaninfo")
    src = CardSiteSource(cfg, persona, store)
    try:
        assert src.mode == "reference"
        result = src.run_tool({"card_id": "1029341"})
        assert "dokkaninfo.com/cards/1029341" in result.text
        assert not result.is_error
        # Reference mode must tell the model it did not read the page.
        assert "not fetched" in result.text.lower()
    finally:
        store.close()


@pytest.mark.parametrize(
    "text",
    [
        "\n\n".join("word " * 120 for _ in range(10)),   # long paragraphs, no periods
        "x" * 5000,                                       # no whitespace at all
        ". ".join("Sentence here" for _ in range(400)),   # sentences only
        "short",
        "para one\n\npara two",
    ],
)
def test_discord_message_splitting_never_exceeds_limit(text):
    from creatorbot.bot import _split_message

    parts = _split_message(text, limit=500)
    assert all(len(p) <= 500 for p in parts), [len(p) for p in parts]
    assert all(p.strip() for p in parts)
    # Nothing may be silently dropped. Chunk boundaries consume the separator
    # they split on, so compare ignoring whitespace.
    strip_ws = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    assert strip_ws("".join(parts)) == strip_ws(text)


def test_discord_splitting_prefers_paragraph_boundaries():
    from creatorbot.bot import _split_message

    text = "\n\n".join(["A" * 200, "B" * 200, "C" * 200])
    parts = _split_message(text, limit=450)
    assert len(parts) == 2
    assert parts[0].startswith("A")


def test_unknown_source_type_raises_readable_error(tmp_path):
    from creatorbot.sources import build_sources

    persona = load_persona("datruthdt")
    persona.raw["sources"] = [{"type": "not_a_real_source", "enabled": True}]
    store = CorpusStore(tmp_path / "t.sqlite3")
    try:
        with pytest.raises(ValueError, match="Unknown source type"):
            build_sources(persona, store)
    finally:
        persona.raw.pop("sources")
        store.close()


def test_citations_dedupe_by_video_not_by_timestamp():
    """Several passages from one video must collapse to a single source link."""
    from creatorbot.engine import _citation_key, _dedupe

    v1a = "[Big Video @ 6:41](https://www.youtube.com/watch?v=ScAnlIojuKc&t=401)"
    v1b = "[Big Video @ 9:58](https://www.youtube.com/watch?v=ScAnlIojuKc&t=598)"
    v2 = "[Other @ 29:51](https://www.youtube.com/watch?v=ldTKTXs54dA&t=1791)"

    assert _citation_key(v1a) == _citation_key(v1b)
    assert _citation_key(v1a) != _citation_key(v2)

    # limit=2 here: this test is about grouping by video, not the cap.
    out = _dedupe([v1a, v1b, v2], limit=2)
    assert out == [v1a, v2], out


def test_dedupe_keeps_first_and_respects_limit():
    from creatorbot.engine import _dedupe

    items = [f"[v{i}](https://x.test/w?v={i})" for i in range(5)]
    assert _dedupe(items, limit=2) == items[:2]


def test_citation_key_handles_plain_and_malformed():
    from creatorbot.engine import _citation_key

    # No markdown link — fall back to the whole string rather than crashing.
    assert _citation_key("just some text") == "just some text"
    # ?t= as the only param leaves no dangling separator.
    assert _citation_key("[a](https://x.test/w?t=5)") == "https://x.test/w"


def test_reply_rules_demand_brevity_and_no_clarifying_questions():
    persona = load_persona("datruthdt")
    prompt = build_system_prompt(persona, ["search_videos"]).lower()
    assert "chat message, not a transcript" in prompt
    assert "never ask a clarifying question" in prompt
    assert "target ~3 sentences" in prompt


def test_clean_reply_unwraps_cite_markup():
    """<cite index="..."> leaks from the model and Discord renders it literally."""
    from creatorbot.engine import clean_reply

    raw = 'He said <cite index="1-1">"Mini SS3 is our superstar, dude"</cite> and meant it.'
    out = clean_reply(raw)
    assert "<cite" not in out and "</cite>" not in out
    assert '"Mini SS3 is our superstar, dude"' in out
    assert "  " not in out  # no doubled spaces where the tags were


def test_clean_reply_leaves_normal_text_alone():
    from creatorbot.engine import clean_reply

    assert clean_reply("  nah that's a skip, bro  ") == "nah that's a skip, bro"
    # Angle brackets that aren't tags survive (e.g. suppressed-embed syntax).
    assert clean_reply("go to <https://x.test>") == "go to <https://x.test>"


def test_only_one_citation_survives():
    from creatorbot.engine import _dedupe

    cites = [f"[v{i}](https://www.youtube.com/watch?v=abc{i})" for i in range(4)]
    assert len(_dedupe(cites)) == 1


def test_bleep_token_is_normalised_and_kept():
    """Censored profanity is evidence, not noise — it must survive chunking."""
    from creatorbot.chunking import BLEEP_TOKEN, chunk_transcript

    snippets = [
        {"text": "this kit is [ __ ] terrible", "start": 0.0},
        {"text": "[Music]", "start": 3.0},
        {"text": "absolute [__] garbage man", "start": 5.0},
    ]
    text = " ".join(c.text for c in chunk_transcript("yt:x", snippets, chunk_chars=500))
    assert text.count(BLEEP_TOKEN) == 2, text
    assert "[Music]" not in text
    assert "_" not in text


def test_overrides_render_after_and_outrank_the_profile(tmp_path, monkeypatch):
    import creatorbot.persona as P

    persona = load_persona("datruthdt")
    monkeypatch.setattr(P, "load_style_profile", lambda _p: "He always hedges.")
    monkeypatch.setattr(P, "load_exemplars", lambda _p: [])

    prompt = P.build_system_prompt(persona, ["search_videos"])
    assert "Corrections to the learned style guide" in prompt
    # Must come after the profile, or "these win" is meaningless.
    assert prompt.index("He always hedges.") < prompt.index("these win")
    assert "cocky" in prompt.lower()


def test_profanity_floor_survives_uncensored():
    persona = load_persona("datruthdt")
    assert persona.voice.get("profanity") == "uncensored"
    prompt = build_system_prompt(persona, ["t"]).lower()
    assert "never slurs" in prompt
    assert "never a sincere personal attack" in prompt
