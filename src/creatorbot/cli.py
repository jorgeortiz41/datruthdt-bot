"""Command line interface.

creatorbot ingest          pull the creator's transcripts into the corpus
creatorbot embed           add semantic vectors (needs VOYAGE_API_KEY)
creatorbot style-profile   learn the creator's voice from the corpus
creatorbot ask "..."       ask a question in the terminal
creatorbot chat            interactive REPL
creatorbot stats           corpus size and composition
creatorbot doctor          check config, keys and source reachability
creatorbot prompt          print the assembled system prompt
creatorbot run             start the Discord bot
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv


def _setup(verbose: bool = False) -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # yt-dlp and httpx are chatty at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("discord").setLevel(logging.WARNING)


def cmd_ingest(args) -> int:
    from .config import load_persona
    from .sources import build_sources
    from .store import CorpusStore

    persona = load_persona(args.persona)
    store = CorpusStore(persona.db_path)
    try:
        any_ingested = False
        for source in build_sources(persona, store):
            if not source.ingests:
                continue
            any_ingested = True
            print(f"\n=== ingesting: {source.name} ===")
            summary = source.ingest(limit=args.limit, refresh=args.refresh)
            for k, v in summary.items():
                print(f"  {k}: {v}")
        if not any_ingested:
            print("No ingesting sources enabled in this persona.")
            return 1
        print(f"\nCorpus: {store.stats()['chunks']:,} chunks in {persona.db_path}")
    finally:
        store.close()
    return 0


def cmd_embed(args) -> int:
    from .config import load_persona
    from .embeddings import get_embedder
    from .store import CorpusStore

    persona = load_persona(args.persona)
    embedder = get_embedder()
    if not getattr(embedder, "enabled", False):
        print(
            "Dense embeddings are off. Set VOYAGE_API_KEY and install the extra:\n"
            "  pip install -e '.[dense]'\n"
            "The bot works without this — it falls back to BM25 lexical search."
        )
        return 1

    store = CorpusStore(persona.db_path)
    try:
        pending = store.chunks_missing_embeddings()
        if not pending:
            print("Every chunk already has an embedding.")
            return 0
        print(f"Embedding {len(pending):,} chunks with {embedder.model}...")
        batch = 96
        done = 0
        for i in range(0, len(pending), batch):
            window = pending[i : i + batch]
            vectors = embedder.embed_documents([c.text for c in window])
            store.store_embeddings(
                embedder.model, [(c.uid, v) for c, v in zip(window, vectors)]
            )
            done += len(window)
            print(f"  {done:,}/{len(pending):,}", end="\r", flush=True)
        print(f"\nStored {done:,} embeddings.")
    finally:
        store.close()
    return 0


def cmd_style_profile(args) -> int:
    from openai import OpenAI

    from .config import load_persona
    from .persona import generate_style_profile, select_exemplars
    from .store import CorpusStore

    persona = load_persona(args.persona)
    store = CorpusStore(persona.db_path)
    try:
        client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
        print(f"Sampling {args.sample} passages and deriving the style guide...")
        profile = generate_style_profile(
            persona, store, client, sample_size=args.sample
        )
        exemplars = select_exemplars(persona, store)
        print(f"\nWrote {persona.style_profile_path}")
        print(f"Wrote {persona.exemplars_path} ({len(exemplars)} exemplars)")
        print("\n" + "-" * 60)
        print(profile[:1500] + ("..." if len(profile) > 1500 else ""))
    finally:
        store.close()
    return 0


def cmd_ask(args) -> int:
    from .config import load_persona
    from .engine import Engine

    persona = load_persona(args.persona)
    engine = Engine(persona)
    try:
        answer = asyncio.run(engine.answer(args.question))
        print("\n" + answer.text)
        if answer.citations:
            print("\nsources: " + " · ".join(answer.citations))
        if args.verbose:
            print(f"\n[tools: {', '.join(answer.tools_used) or 'none'}]")
            print(f"[stop: {answer.stop_reason}] [usage: {answer.usage}]")
    finally:
        engine.close()
    return 0


def cmd_chat(args) -> int:
    from .config import load_persona
    from .engine import Engine

    persona = load_persona(args.persona)
    engine = Engine(persona)
    history: list[dict] = []
    print(f"{persona.display_name}Bot — Ctrl-C or 'quit' to exit.\n")
    try:
        while True:
            try:
                question = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit"}:
                break
            answer = asyncio.run(engine.answer(question, history=history))
            print(f"\nbot> {answer.text}\n")
            if answer.citations:
                print("     sources: " + " · ".join(answer.citations) + "\n")
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer.text})
            history = history[-2 * int(persona.discord.get("history_turns", 6)) :]
    finally:
        engine.close()
    return 0


def cmd_stats(args) -> int:
    from .config import load_persona
    from .store import CorpusStore

    persona = load_persona(args.persona)
    store = CorpusStore(persona.db_path)
    try:
        s = store.stats()
        print(f"persona:    {persona.display_name} ({persona.id})")
        print(f"database:   {s['db_path']} ({s['db_size_mb']} MB)")
        print(f"documents:  {s['documents']:,}")
        for src, n in sorted(s["documents_by_source"].items()):
            print(f"              {src}: {n:,}")
        print(f"chunks:     {s['chunks']:,}")
        print(f"embeddings: {s['embeddings']:,}")
        print(
            f"profile:    {'yes' if persona.style_profile_path.exists() else 'NOT GENERATED'}"
        )
    finally:
        store.close()
    return 0


def cmd_doctor(args) -> int:
    from .config import load_persona
    from .embeddings import get_embedder
    from .sources import build_sources, web_search_config
    from .store import CorpusStore

    persona = load_persona(args.persona)
    print(f"persona file:      {persona.path}")
    print(f"model:             {persona.model} (effort={persona.effort})")

    ok = True
    for var, required in (
        ("XAI_API_KEY", True),
        ("DISCORD_BOT_TOKEN", True),
        ("VOYAGE_API_KEY", False),
    ):
        present = bool(os.getenv(var))
        mark = "ok " if present else ("MISSING" if required else "not set")
        print(f"{var:<18} {mark}")
        if required and not present:
            ok = False

    embedder = get_embedder()
    print(
        f"retrieval:         {'hybrid (BM25 + ' + embedder.model + ')' if getattr(embedder, 'enabled', False) else 'lexical BM25 only'}"
    )

    store = CorpusStore(persona.db_path)
    try:
        print("\nsources:")
        for source in build_sources(persona, store):
            print(f"  {source.name:<14} {source.health()}")
        ws = web_search_config(persona)
        if ws:
            print(f"  {'web_search':<14} server-side, max_uses={ws.get('max_uses', 4)}")
        s = store.stats()
        if s["documents"] == 0:
            print("\n! corpus is empty — run `creatorbot ingest`")
            ok = False
        if not persona.style_profile_path.exists():
            print("! no style profile — run `creatorbot style-profile`")
    finally:
        store.close()

    print("\n" + ("all good" if ok else "fix the items above"))
    return 0 if ok else 1


def cmd_prompt(args) -> int:
    from .config import load_persona
    from .engine import Engine

    persona = load_persona(args.persona)
    engine = Engine(persona)
    try:
        print(engine.system_prompt)
        print(
            f"\n--- {len(engine.system_prompt):,} chars, tools: "
            f"{[t.get('name') for t in engine.tools]}"
        )
    finally:
        engine.close()
    return 0


def cmd_run(args) -> int:
    from .bot import run

    run()
    return 0


def cmd_personas(args) -> int:
    from .config import list_personas, load_persona

    for name in list_personas():
        p = load_persona(name)
        print(f"{name:<16} {p.display_name} — {p.domain.get('name', '?')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="creatorbot",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p", "--persona", help="persona name (default $CREATORBOT_PERSONA)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="pull the creator's videos into the corpus")
    p.add_argument("--limit", type=int, help="max videos this run")
    p.add_argument(
        "--refresh", action="store_true", help="re-fetch videos already stored"
    )
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("embed", help="add semantic vectors (needs VOYAGE_API_KEY)")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser(
        "style-profile", help="learn the creator's voice from the corpus"
    )
    p.add_argument("--sample", type=int, default=60, help="passages to sample")
    p.set_defaults(func=cmd_style_profile)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive REPL")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("stats", help="corpus size and composition")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("doctor", help="check config, keys and sources")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("prompt", help="print the assembled system prompt")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("personas", help="list available personas")
    p.set_defaults(func=cmd_personas)

    p = sub.add_parser("run", help="start the Discord bot")
    p.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
