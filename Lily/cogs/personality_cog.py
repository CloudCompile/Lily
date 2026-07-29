#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Personality Cog

Handles on_message events, spontaneous messages, and reactive behavior.
Multi-server aware — reads per-guild settings from the database.
"""

from __future__ import annotations
import random
import discord
from discord.ext import commands, tasks

from database import Database
from pollinations import PollinationsAPI
from personality import PersonalityEngine, DecisionEngine, is_safe_content
from config import (
    BOT_PREFIX, DEFAULT_TEXT_MODEL, DEFAULT_SAFE_MODE,
    BASE_REPLY_CHANCE, REACTION_CHANCE, SPONTANEOUS_MESSAGE_CHANCE,
)


class PersonalityCog(commands.Cog, name="Personality"):
    """Lily's personality — reactive messages, spontaneous messages, and mood."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.spontaneous_messages = [
            "anyone wanna chat? 🌸",
            "just vibing ✨",
            "hmm what's everyone up to?",
            "lowkey bored rn",
            "it's kinda quiet in here... 💭",
            "anyone else love rainy days? 🌧️",
            "just had a thought and forgot it immediately lol",
            "random question: what's your favorite season? 🍂",
            "prolly should be doing something productive but here i am 😅",
            "the vibes are immaculate rn ✨",
        ]

    async def cog_load(self):
        """Start the spontaneous message task when the cog loads."""
        self.check_spontaneous.start()

    async def cog_unload(self):
        self.check_spontaneous.cancel()

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle incoming messages for Lily's personality system."""
        # Ignore bots and DMs
        if message.author.bot:
            return
        if not message.guild:
            return

        guild_id = message.guild.id
        db: Database = self.bot.db  # type: ignore
        personality: PersonalityEngine = self.bot.personality  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore

        # Get guild settings
        settings = db.get_guild_settings(guild_id)

        # Check if message is in allowed channel
        allowed_channel = settings.get("allowed_channel")
        if allowed_channel and str(message.channel.id) != allowed_channel:
            return

        # Check if personality is enabled
        if not settings.get("personality_enabled", 1):
            return

        # Get per-guild behavior settings
        reply_chance = settings.get("reply_chance", BASE_REPLY_CHANCE)
        reaction_chance = settings.get("reaction_chance", REACTION_CHANCE)
        spontaneous_chance = settings.get("spontaneous_chance", SPONTANEOUS_MESSAGE_CHANCE)

        decision = DecisionEngine(
            reply_chance=reply_chance,
            reaction_chance=reaction_chance,
            spontaneous_chance=spontaneous_chance,
        )

        # Check if this is a command
        prefix = settings.get("prefix", BOT_PREFIX)
        if message.content.startswith(prefix):
            return

        # Detect if Lily is mentioned
        is_mentioned = self.bot.user.mentioned_in(message)
        is_dm = False

        # Check content safety
        if not is_safe_content(message.content):
            return

        # Decide whether to respond
        if decision.should_reply(message.content, is_mentioned, is_dm):
            # Detect emotion
            emotion = personality.detect_emotion(message.content)

            # Decide: react or reply?
            if not is_mentioned and decision.should_react():
                emoji = decision.get_reaction_emoji(emotion)
                try:
                    await message.add_reaction(emoji)
                except discord.HTTPException:
                    pass
                return

            # Generate a reply
            await self._generate_reply(message, guild_id, settings, personality, api, db)

    async def _generate_reply(
        self,
        message: discord.Message,
        guild_id: int,
        settings: dict,
        personality: PersonalityEngine,
        api: PollinationsAPI,
        db: Database,
    ):
        """Generate and send a personality-driven reply."""
        user_id = message.author.id

        # Build context
        history = db.get_conversations(guild_id, user_id, limit=15)
        user_facts = db.get_facts(guild_id, user_id)

        # Update mood
        mood, intensity = personality.mood.update(message.content)
        system_prompt = personality.build_system_prompt(
            mood, personality.mood.get_energy(), user_facts
        )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message.content})

        # Get model settings
        model = settings.get("text_model", DEFAULT_TEXT_MODEL)
        safe = settings.get("safe_mode", DEFAULT_SAFE_MODE)

        try:
            async with message.channel.typing():
                response = await api.chat_completions_simple(
                    messages, model=model, safe=safe, max_tokens=300
                )
                response = personality.inject_personality(response)

            # Save to history
            db.add_conversation(guild_id, user_id, "user", message.content)
            db.add_conversation(guild_id, user_id, "assistant", response)

            # Track topics
            topics = personality.extract_topics(message.content)
            for topic in topics:
                db.add_topic(guild_id, user_id, topic)

            # Send response
            if len(response) > 2000:
                response = response[:1997] + "..."

            await message.reply(response)

        except Exception:
            # Silently fail for personality responses — don't spam errors
            pass

    @tasks.loop(minutes=5)
    async def check_spontaneous(self):
        """Periodically check if Lily should send a spontaneous message."""
        for guild in self.bot.guilds:
            db: Database = self.bot.db  # type: ignore
            settings = db.get_guild_settings(guild.id)

            # Check if personality is enabled
            if not settings.get("personality_enabled", 1):
                continue

            spontaneous_chance = settings.get("spontaneous_chance", SPONTANEOUS_MESSAGE_CHANCE)
            if random.random() >= spontaneous_chance:
                continue

            # Find the allowed channel
            allowed_channel = settings.get("allowed_channel")
            if allowed_channel:
                channel = guild.get_channel(int(allowed_channel))
            else:
                # Pick the first text channel where Lily can speak
                channel = None
                for ch in guild.text_channels:
                    if ch.permissions_for(guild.me).send_messages:
                        channel = ch
                        break

            if not channel:
                continue

            try:
                msg = random.choice(self.spontaneous_messages)
                await channel.send(msg)
            except discord.HTTPException:
                pass

    @check_spontaneous.before_loop
    async def before_spontaneous(self):
        """Wait until the bot is ready before starting the task."""
        await self.bot.wait_until_ready()


async def setup(bot: commands.Bot):
    await bot.add_cog(PersonalityCog(bot))
