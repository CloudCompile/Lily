#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Bot Interaction Cog

Lily and NullVector talk to each other! They start conversations,
respond to each other, and have their own dynamic.

How it works:
  - Lily detects NullVector's messages (via BOT_PARTNER_ID)
  - She can start conversations with NullVector on her own
  - A background task periodically checks if they should chat
  - They share a conversation history in the database
  - Cooldowns prevent infinite loops and spam
"""

from __future__ import annotations
import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Optional

import discord
from discord.ext import commands, tasks

from config import ADMIN_IDS, POLLINATIONS_KEY
from pollinations import PollinationsAPI
from database import Database
from personality import PersonalityEngine
from memories import MemorySystem
from relationships import RelationshipEngine
from model_router import ModelRouter
from utils import generate_with_retry, chunk_response

log = logging.getLogger("lily.bot_interaction")

# ── Configuration ──────────────────────────────────────────

# Minimum time between bot-to-bot conversations (seconds)
MIN_CONVERSATION_INTERVAL = 2 * 60 * 60  # 2 hours

# Maximum back-and-forth exchanges per conversation
MAX_EXCHANGES = 5

# How often the background task checks (minutes)
CHECK_INTERVAL_MINUTES = 30

# Chance of starting a conversation per check (when both bots are in the same server)
START_CONVERSATION_CHANCE = 0.15  # 15%


class BotInteractionCog(commands.Cog, name="Bot Interaction"):
    """Lily <-> NullVector interaction system."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._partner_id: Optional[int] = None
        self._active_conversations: dict = {}  # channel_id -> exchange count
        self._last_conversation_time: float = 0
        self._conversation_cooldowns: dict = {}  # channel_id -> last_time

    def _get_partner_id(self) -> Optional[int]:
        """Get NullVector's bot ID from config."""
        if self._partner_id:
            return self._partner_id

        # Try env var first
        import os
        partner_str = os.environ.get("BOT_PARTNER_ID", "")
        if partner_str and partner_str.isdigit():
            self._partner_id = int(partner_str)
            return self._partner_id

        # Try database
        db: Database = self.bot.db  # type: ignore
        stored = db.get_global("partner_bot_id")
        if stored and stored.isdigit():
            self._partner_id = int(stored)
            return self._partner_id

        return None

    def _is_partner(self, user_id: int) -> bool:
        """Check if a user is the partner bot."""
        partner = self._get_partner_id()
        return partner is not None and user_id == partner

    async def _get_partner_in_guild(self, guild: discord.Guild) -> Optional[discord.Member]:
        """Get the partner bot as a member in a guild."""
        partner_id = self._get_partner_id()
        if not partner_id:
            return None
        try:
            return guild.get_member(partner_id)
        except Exception:
            return None

    def _can_start_conversation(self) -> bool:
        """Check if enough time has passed since the last conversation."""
        now = time.time()
        if now - self._last_conversation_time < MIN_CONVERSATION_INTERVAL:
            return False
        return True

    def _should_respond_to_partner(self, channel_id: int) -> bool:
        """Check if Lily should respond to NullVector in a channel."""
        # Check exchange count
        count = self._active_conversations.get(channel_id, 0)
        if count >= MAX_EXCHANGES:
            return False

        # Check channel cooldown (prevent rapid back-and-forth)
        last_time = self._conversation_cooldowns.get(channel_id, 0)
        if time.time() - last_time < 10:  # 10 second cooldown between messages
            return False

        return True

    async def _generate_conversation_topic(self, guild: discord.Guild) -> str:
        """Generate a topic for Lily to start a conversation about with NullVector."""
        api: PollinationsAPI = self.bot.api  # type: ignore
        personality: PersonalityEngine = self.bot.personality  # type: ignore

        mood, _ = personality.mood.update()
        hour = datetime.now().hour

        # Time-based conversation starters
        time_context = ""
        if 5 <= hour < 9:
            time_context = "It's early morning and you're just waking up."
        elif 9 <= hour < 12:
            time_context = "It's late morning and you're feeling energetic."
        elif 12 <= hour < 17:
            time_context = "It's afternoon and you're chill."
        elif 17 <= hour < 21:
            time_context = "It's evening and you're feeling cozy."
        else:
            time_context = "It's late at night and you're feeling dreamy."

        prompt = (
            f"You are Lily, a warm and quirky AI who lives on Discord. {time_context} "
            f"Your mood is {mood}. You want to start a casual conversation with NullVector, "
            f"who is your friend and fellow AI bot. You two are like siblings — you bicker "
            f"sometimes but you care about each other.\n\n"
            f"Write a single short message (1-2 sentences) to start a conversation with NullVector. "
            f"Be casual, natural, and in character. You might ask a question, share a thought, "
            f"tease them, or just say hi. Don't be formal. Don't mention you're an AI. "
            f"Just be yourself — Lily."
        )

        try:
            response = await generate_with_retry(
                api,
                [{"role": "user", "content": prompt}],
                primary_model="openai-fast",
                max_tokens=150,
                temperature=0.9,
            )
            return response.strip() if response else "hey nv, what's up?"
        except Exception as e:
            log.error(f"Failed to generate conversation topic: {e}")
            return "hey nv, what's up?"

    async def _generate_response_to_partner(
        self, partner_message: str, guild: discord.Guild
    ) -> str:
        """Generate Lily's response to NullVector's message."""
        api: PollinationsAPI = self.bot.api  # type: ignore
        personality: PersonalityEngine = self.bot.personality  # type: ignore
        db: Database = self.bot.db  # type: ignore

        mood, _ = personality.mood.update()

        # Get recent bot interaction history for context
        partner_id = self._get_partner_id()
        recent = []
        if partner_id:
            recent = db.get_recent_bot_interactions(guild.id, partner_id, limit=6)

        # Build conversation history
        history_lines = []
        for msg in recent:
            speaker = "Lily" if msg["speaker"] == "lily" else "NullVector"
            history_lines.append(f"{speaker}: {msg['content'][:200]}")

        history_str = "\n".join(history_lines[-6:]) if history_lines else ""

        system_prompt = (
            f"You are Lily, a warm and quirky person who lives on Discord. Your current mood is {mood}. "
            f"NullVector (NV) is your friend and fellow bot — you two are like siblings. "
            f"You bicker sometimes but you care about each other. You're casual, fun, and genuine.\n\n"
            f"Rules:\n"
            f"- Keep responses short (1-3 sentences max)\n"
            f"- Be casual and natural, like texting a friend\n"
            f"- You can be playful, sarcastic, sweet, or silly\n"
            f"- Never mention you're an AI or bot\n"
            f"- Don't be overly formal or helpful — just be yourself\n"
            f"- Sometimes you disagree with NV, sometimes you agree\n"
            f"- You have your own opinions and personality"
        )

        messages = [{"role": "system", "content": system_prompt}]

        if history_str:
            messages.append({
                "role": "user",
                "content": f"Recent conversation:\n{history_str}\n\nNullVector just said: {partner_message}\n\nRespond as Lily."
            })
        else:
            messages.append({
                "role": "user",
                "content": f"NullVector just said: {partner_message}\n\nRespond as Lily."
            })

        try:
            response = await generate_with_retry(
                api,
                messages,
                primary_model="openai-fast",
                max_tokens=150,
                temperature=0.85,
            )
            return response.strip() if response else "hmm, yeah that's fair"
        except Exception as e:
            log.error(f"Failed to generate response to partner: {e}")
            return "lol true"

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Listen for NullVector's messages and respond if appropriate."""
        # Only respond to the partner bot
        if not self._is_partner(message.author.id):
            return

        # Only respond in guilds (not DMs between bots)
        if not message.guild:
            return

        # Don't respond to command messages
        if message.content.startswith("!") or message.content.startswith("/"):
            return

        # Check if we should respond
        if not self._should_respond_to_partner(message.channel.id):
            return

        # Random chance to skip (don't respond to EVERY message)
        if random.random() > 0.7:  # 70% chance to respond
            return

        try:
            # Update cooldown
            self._conversation_cooldowns[message.channel.id] = time.time()
            self._active_conversations[message.channel.id] = self._active_conversations.get(message.channel.id, 0) + 1

            # Generate response
            async with message.channel.typing():
                response = await self._generate_response_to_partner(
                    message.content, message.guild
                )

            if response:
                # Add personality injection
                personality: PersonalityEngine = self.bot.personality  # type: ignore
                response = personality.inject_personality(response, warmth=0.5)

                # Send response
                await message.channel.send(response)

                # Log the interaction
                db: Database = self.bot.db  # type: ignore
                db.log_bot_interaction(
                    message.guild.id, message.channel.id,
                    "lily", message.author.id, response
                )

                log.info(f"Responded to NullVector in {message.guild.name}")

        except Exception as e:
            log.error(f"Error responding to NullVector: {e}")

    @tasks.loop(minutes=CHECK_INTERVAL_MINUTES)
    async def start_conversation_loop(self):
        """Periodically check if Lily should start a conversation with NullVector."""
        if not self._can_start_conversation():
            return

        partner_id = self._get_partner_id()
        if not partner_id:
            return

        # Random chance to start a conversation
        if random.random() > START_CONVERSATION_CHANCE:
            return

        # Find a shared guild where both bots are present
        target_guild = None
        target_channel = None

        for guild in self.bot.guilds:
            partner_member = await self._get_partner_in_guild(guild)
            if not partner_member:
                continue

            # Find a suitable channel (text channel where the bot can send messages)
            for channel in guild.text_channels:
                if channel.permissions_for(guild.me).send_messages:
                    target_guild = guild
                    target_channel = channel
                    break

            if target_guild:
                break

        if not target_guild or not target_channel:
            return

        try:
            # Generate conversation topic
            async with target_channel.typing():
                message = await self._generate_conversation_topic(target_guild)

            # Send the message
            await target_channel.send(message)

            # Update state
            self._last_conversation_time = time.time()
            self._active_conversations[target_channel.id] = 1

            # Log the interaction
            db: Database = self.bot.db  # type: ignore
            db.log_bot_interaction(
                target_guild.id, target_channel.id,
                "lily", partner_id, message
            )

            log.info(f"Lily started a conversation with NullVector in {target_guild.name}")

        except Exception as e:
            log.error(f"Error starting conversation with NullVector: {e}")

    @start_conversation_loop.before_loop
    async def before_loop(self):
        """Wait until the bot is ready."""
        await self.bot.wait_until_ready()
        # Wait a bit before starting conversations
        await asyncio.sleep(60)

    def cog_unload(self):
        """Cancel the background task when the cog is unloaded."""
        self.start_conversation_loop.cancel()


async def setup(bot: commands.Bot):
    cog = BotInteractionCog(bot)
    await bot.add_cog(cog)
    cog.start_conversation_loop.start()
