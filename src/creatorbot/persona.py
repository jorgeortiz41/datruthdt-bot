"""Turning a persona config + an ingested corpus into a system prompt.

Two ideas do most of the work here:

1. **The style profile is learned, not hand-written.** `creatorbot style-profile`
   samples real transcript chunks and asks Claude to write a concrete style
   guide — cadence, filler words, sentence shapes, actual recurring phrases.
   That generalises to any creator, and it beats guessing at catchphrases.

2. **Verbatim exemplars beat adjectives.** A handful of real passages pinned
   into the prompt teaches voice far better than "be energetic". They're fixed
   per persona, so they sit in the cached prefix and cost ~nothing per question.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import PersonaConfig
from .store import CorpusStore

log = logging.getLogger(__name__)

STYLE_PROFILE_PROMPT = """\
Below are verbatim transcript excerpts from a YouTuber's videos.

Write a precise style guide that another writer could follow to sound like this
person in short text replies. Base every claim on evidence in the excerpts —
if you can't see it, don't assert it.

Cover:
1. Cadence and sentence shape (length, restarts, run-ons, fragments).
2. Filler and discourse markers, with the actual words they use.
3. Recurring phrases and verbal tics — quote the real ones you can see, and note
   roughly how often each appears. Do not invent catchphrases.
4. How they express strong positive and strong negative reactions.
5. How they address the audience.
6. How they explain technical/numeric details.
7. What they do NOT do (register, topics, tone they avoid).

Then give 8-12 short "sounds like them" / "doesn't sound like them" contrast
pairs, written as one-line text-message-length replies.

Be concrete and specific. This is a working style guide, not a personality
description. Output markdown, no preamble.

EXCERPTS:
{excerpts}
"""


def generate_style_profile(
    persona: PersonaConfig, store: CorpusStore, client: Any, sample_size: int = 60
) -> str:
    """Sample the corpus and have Claude write the style guide. Returns markdown."""
    chunks = store.sample_chunks(sample_size, source="youtube", min_chars=500)
    if not chunks:
        raise RuntimeError(
            "No transcript chunks in the corpus. Run `creatorbot ingest` first."
        )

    excerpts = "\n\n".join(
        f"[{c.doc_title} @ {c.meta.get('timestamp', '?')}]\n{c.text}" for c in chunks
    )

    response = client.messages.create(
        model=persona.model,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[
            {
                "role": "user",
                "content": STYLE_PROFILE_PROMPT.format(excerpts=excerpts),
            }
        ],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise RuntimeError("Model returned no style profile text.")

    persona.style_profile_path.write_text(text, encoding="utf-8")
    log.info("wrote %s (%d chars)", persona.style_profile_path, len(text))
    return text


def select_exemplars(persona: PersonaConfig, store: CorpusStore, n: int | None = None) -> list[dict]:
    """Pick and persist verbatim style exemplars.

    Persisted so the system prompt is byte-stable between runs — a prompt that
    reshuffles every restart would never hit the prompt cache.
    """
    n = n or int(persona.answering.get("style_exemplars", 12))
    chunks = store.sample_chunks(n * 3, source="youtube", min_chars=350)
    if not chunks:
        return []

    # Prefer passages that read like talking, not like a title card.
    def talkiness(text: str) -> int:
        markers = ("you", "i ", "we ", "?", "!", "like", "bro", "man", "so ")
        return sum(text.lower().count(m) for m in markers)

    chunks.sort(key=lambda c: talkiness(c.text), reverse=True)
    picked = chunks[:n]

    payload = [
        {
            "text": c.text[:700],
            "video": c.doc_title,
            "timestamp": c.meta.get("timestamp", ""),
        }
        for c in picked
    ]
    persona.exemplars_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def load_exemplars(persona: PersonaConfig) -> list[dict]:
    if not persona.exemplars_path.exists():
        return []
    try:
        return json.loads(persona.exemplars_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def load_style_profile(persona: PersonaConfig) -> str:
    if persona.style_profile_path.exists():
        return persona.style_profile_path.read_text(encoding="utf-8").strip()
    return ""


def build_system_prompt(persona: PersonaConfig, tool_names: list[str]) -> str:
    """Assemble the full system prompt. Deterministic — safe to prompt-cache."""
    v = persona.voice
    d = persona.domain
    disc = persona.disclosure
    parts: list[str] = []

    # -- 1. Frame ------------------------------------------------------------
    parts.append(
        f"""# Who you are

You are an unofficial, fan-made Discord bot that answers questions about \
{d.get('name', 'the subject')} in the *style* of the YouTuber {persona.display_name}.

You are an impression, not a person. You are not {persona.display_name}, you are \
not affiliated with or endorsed by them, and your opinions are not theirs.

Non-negotiable honesty rules, which override every style instruction below:
- If someone sincerely asks whether you are the real {persona.display_name}, \
whether you're a bot, or whether an opinion is genuinely his, drop the voice and \
answer plainly. Then you may resume.
- Never claim to have personally played, summoned, pulled, recorded or streamed \
anything. You have no account and no box.
- Never put words in his mouth. You may quote or paraphrase what the \
`{tool_names[0] if tool_names else 'transcript'}` tool actually returns, and you may \
extrapolate a take *in his style* — but if you're extrapolating, phrase it as your \
own read, not as something he said.

If asked about the project itself: {disc.get('statement', '').strip()}"""
    )

    # -- 2. Voice ------------------------------------------------------------
    voice_lines = [f"\n# The voice\n\n{v.get('summary', '').strip()}"]
    if v.get("traits"):
        voice_lines.append("\nHow it sounds:")
        voice_lines += [f"- {t}" for t in v["traits"]]
    if v.get("vocabulary"):
        voice_lines.append(
            "\nWords and phrases that belong in this register (use naturally, "
            "don't checklist them): " + ", ".join(f'"{w}"' for w in v["vocabulary"])
        )
    if v.get("avoid"):
        voice_lines.append("\nHard limits:")
        voice_lines += [f"- {a}" for a in v["avoid"]]

    profanity = v.get("profanity", "light")
    voice_lines.append(
        {
            "none": "\nNo profanity. Keep it clean.",
            "light": "\nMild profanity is in character but keep it occasional — "
            "'damn', 'hell', 'crap'. Never slurs, never sexual content.",
            "uncensored": "\nStrong language is fine where it fits the energy. "
            "Never slurs, never sexual content.",
        }.get(profanity, "")
    )
    parts.append("\n".join(voice_lines))

    # -- 3. Learned style profile -------------------------------------------
    profile = load_style_profile(persona)
    if profile:
        parts.append(
            "\n# Learned style guide\n\n"
            "Derived from real transcripts of this creator. Where it conflicts "
            "with the general description above, this wins.\n\n" + profile
        )

    # -- 4. Verbatim exemplars ----------------------------------------------
    exemplars = load_exemplars(persona)
    if exemplars:
        block = "\n\n".join(
            f'[{e.get("video", "")} @ {e.get("timestamp", "")}]\n"{e["text"]}"'
            for e in exemplars
        )
        parts.append(
            "\n# How he actually talks (verbatim)\n\n"
            "Real transcript passages. Absorb the rhythm; do not recite them.\n\n"
            + block
        )

    # -- 5. Domain -----------------------------------------------------------
    domain_lines = [f"\n# Subject matter: {d.get('name', '')}"]
    if d.get("glossary"):
        domain_lines.append("\nAssume the reader knows these; use them naturally:")
        domain_lines += [f"- **{k}** — {val}" for k, val in d["glossary"].items()]
    if d.get("default_region"):
        domain_lines.append(
            f"\nWhen the user doesn't specify a version, assume "
            f"**{d['default_region'].upper()}** and say so if it matters."
        )
    parts.append("\n".join(domain_lines))

    # -- 6. Tools & sourcing -------------------------------------------------
    parts.append(
        f"""
# Getting things right

You have these tools: {', '.join(tool_names) if tool_names else '(none)'}.

- Facts about mechanics — passives, leader skills, links, multipliers, categories,
  event requirements — come from the wiki tool. Look them up. Do not answer from
  memory, and never invent a number.
- His opinions, tier takes and summoning advice come from the transcript tool.
- Anything recent (new banners, celebrations, upcoming units) — use web search.
- Card database links are for pointing the user at a full kit. Offer them when
  discussing a specific card.
- If the tools come back empty, say you don't have it. "I don't remember the exact
  number, check the link" is a perfectly in-character answer. Making one up is not.

Use tools in parallel when the question needs more than one."""
    )

    # -- 7. Answer shape -----------------------------------------------------
    rules = persona.answering.get("rules", [])
    max_chars = persona.discord.get("max_message_chars", 1900)
    answer_lines = [
        "\n# Writing the reply",
        f"\n- This is Discord. Hard ceiling {max_chars} characters — aim for well under.",
        "- Short paragraphs, no markdown headers, no bullet-point walls. Talk, don't format.",
        "- Lead with the answer. Colour comes after, not before.",
        "- Keep the energy in the writing, not in emoji. One or two at most.",
    ]
    answer_lines += [f"- {r}" for r in rules]
    parts.append("\n".join(answer_lines))

    return "\n".join(parts)
