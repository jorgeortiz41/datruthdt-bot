"""The Discord surface.

Responds to @mentions, to replies on its own messages, and to /ask anywhere.
Conversation state is per-channel and in-memory — restarting clears it.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from typing import Any

import discord
from discord import app_commands

from .config import PersonaConfig, load_persona
from .engine import Engine

log = logging.getLogger(__name__)


#: Progressively finer split points. The empty string is the hard fallback that
#: guarantees we never exceed the limit, even for text with no whitespace.
_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def _split_message(text: str, limit: int = 1900) -> list[str]:
    """Chunk text to Discord's per-message limit.

    Discord rejects anything over 2000 characters outright, so this must be a
    hard guarantee — hence the character-level fallback at the end of
    `_SEPARATORS`.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    for sep in _SEPARATORS:
        pieces = _atoms(text, sep, limit)
        if pieces is None:
            continue  # this separator still leaves an oversized atom

        chunks: list[str] = []
        current = ""
        joiner = sep if sep != ". " else ""  # ". " is kept on the piece itself
        for piece in pieces:
            candidate = (current + joiner + piece) if current else piece
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = piece
        if current.strip():
            chunks.append(current.strip())
        return [c for c in chunks if c]

    return [text[:limit]]  # unreachable; the "" separator always succeeds


def _atoms(text: str, sep: str, limit: int) -> list[str] | None:
    """Split on `sep`, or None if any resulting piece is still too long."""
    if sep == "":
        return [text[i : i + limit] for i in range(0, len(text), limit)]
    if sep == ". ":
        parts = [p + ". " for p in text.split(". ")]
        parts[-1] = parts[-1][:-2]  # last piece had no trailing separator
    else:
        parts = text.split(sep)
    parts = [p for p in parts if p]
    if not parts or any(len(p) > limit for p in parts):
        return None
    return parts


class CreatorBot(discord.Client):
    def __init__(self, persona: PersonaConfig):
        intents = discord.Intents.default()
        intents.message_content = True  # needed to read @mentions
        super().__init__(intents=intents)

        self.persona = persona
        self.engine = Engine(persona)
        self.tree = app_commands.CommandTree(self)

        cfg = persona.discord
        self.allowed_channels = set(cfg.get("allowed_channel_ids") or [])
        self.history_turns = int(cfg.get("history_turns", 6))
        self.max_chars = int(cfg.get("max_message_chars", 1900))
        self.show_sources = bool(cfg.get("show_sources_footer", True))

        self._history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.history_turns * 2)
        )
        self._register_commands()

    # -- lifecycle ------------------------------------------------------------

    async def setup_hook(self) -> None:
        guild_id = os.getenv("DISCORD_DEV_GUILD_ID", "").strip()
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            log.info("slash commands synced to guild %s", guild_id)
        else:
            await self.tree.sync()
            log.info("slash commands synced globally (may take up to an hour)")

    async def on_ready(self) -> None:
        log.info("logged in as %s (id %s)", self.user, self.user.id)
        log.info("in %d guild(s)", len(self.guilds))
        await self.change_presence(
            activity=discord.Game(name=self.persona.tagline or "Dokkan Battle")
        )

    # -- message handling -----------------------------------------------------

    def _should_respond(self, message: discord.Message) -> bool:
        if message.author.bot:
            return False
        if self.allowed_channels and message.channel.id not in self.allowed_channels:
            return False
        if self.user in message.mentions:
            return True
        ref = message.reference
        if ref and isinstance(ref.resolved, discord.Message):
            return ref.resolved.author.id == self.user.id
        return isinstance(message.channel, discord.DMChannel)

    async def on_message(self, message: discord.Message) -> None:
        if not self._should_respond(message):
            return

        question = message.content
        for mention in (f"<@{self.user.id}>", f"<@!{self.user.id}>"):
            question = question.replace(mention, "")
        question = question.strip()
        if not question:
            question = f"Introduce yourself in one or two lines."

        async with message.channel.typing():
            answer = await self._ask(message.channel.id, question)

        await self._send(message.channel, answer, reply_to=message)

    # -- slash commands -------------------------------------------------------

    def _register_commands(self) -> None:
        @self.tree.command(name="ask", description=f"Ask {self.persona.display_name}Bot anything")
        @app_commands.describe(question="Your question")
        async def ask(interaction: discord.Interaction, question: str) -> None:
            await interaction.response.defer(thinking=True)
            answer = await self._ask(interaction.channel_id or 0, question)
            body = self._format(answer)
            parts = _split_message(body, self.max_chars)
            for part in parts:
                await interaction.followup.send(part, suppress_embeds=True)

        @self.tree.command(name="about", description="What this bot is (and isn't)")
        async def about(interaction: discord.Interaction) -> None:
            disc = self.persona.disclosure
            stats = self.engine.store.stats()
            embed = discord.Embed(
                title=f"{self.persona.display_name}Bot",
                description=disc.get("statement", ""),
                colour=discord.Colour.orange(),
            )
            embed.add_field(
                name="Knowledge",
                value=(
                    f"{stats['documents_by_source'].get('youtube', 0)} videos ingested\n"
                    f"{stats['chunks']:,} searchable passages\n"
                    f"Live wiki + web search enabled"
                ),
                inline=False,
            )
            embed.set_footer(text="Unofficial fan project · not affiliated with the creator")
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="forget", description="Clear this channel's conversation memory")
        async def forget(interaction: discord.Interaction) -> None:
            self._history.pop(interaction.channel_id or 0, None)
            await interaction.response.send_message("Memory cleared for this channel.", ephemeral=True)

    # -- shared -------------------------------------------------------------

    async def _ask(self, channel_id: int, question: str):
        history = list(self._history[channel_id])
        answer = await self.engine.answer(question, history=history)
        self._history[channel_id].append({"role": "user", "content": question})
        self._history[channel_id].append({"role": "assistant", "content": answer.text})
        return answer

    def _format(self, answer) -> str:
        body = answer.text
        # One source, max. Every link Discord unfurls costs a thumbnail card,
        # and three of them buries a three-sentence answer.
        if self.show_sources and answer.citations:
            body += "\n\n-# " + answer.citations[0]
        return body

    async def _send(self, channel, answer, reply_to: discord.Message | None = None) -> None:
        parts = _split_message(self._format(answer), self.max_chars)
        for i, part in enumerate(parts):
            # suppress_embeds keeps a link a link, instead of a giant preview
            # card with a video thumbnail attached under every reply.
            if i == 0 and reply_to is not None:
                await reply_to.reply(part, mention_author=False, suppress_embeds=True)
            else:
                await channel.send(part, suppress_embeds=True)


def run() -> None:
    """Entry point for `creatorbot run`."""
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is not set. Copy .env.example to .env and fill it in."
        )
    if not os.getenv("ANTHROPIC_API_KEY"):
        log.warning("ANTHROPIC_API_KEY not set — relying on an `ant auth login` profile")

    persona = load_persona()
    log.info("persona: %s (%s)", persona.display_name, persona.path.name)

    bot = CreatorBot(persona)
    try:
        bot.run(token, log_handler=None)
    finally:
        bot.engine.close()
