#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.5 — "Lily Lives" Multi-Server AI Discord Bot

Powered by Pollinations API — Text & Image Generation.
She has feelings, memories, and she'll reach out to you.
No 3D, audio, or video — just the real Lily.

Features:
  - Proactive DMs (she'll start conversations with you)
  - Affection/warmness/dislike per user (CROSS-SERVER)
  - Memories and daily recaps (CROSS-SERVER — she carries them everywhere)
  - Dream journal (she writes dreams and can share them)
  - Smart model routing (Sana Sprint for images, free models for text)
  - Generation quotas (not unlimited willy-nilly)
  - Mood-reactive Discord status (her status changes with her mood)
  - Real person feel (typing delays, personality quirks, emotional depth)

Run: python bot.py
"""

from __future__ import annotations
import asyncio
import logging
import sys
import os
import json
import random
from datetime import datetime
from typing import Union

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    DISCORD_TOKEN, ADMIN_IDS, BOT_PREFIX,
    POLLINATIONS_KEY, POLLINATIONS_BASE_URL,
    PROACTIVE_DM_CHECK_INTERVAL, PROACTIVE_DM_ENABLED,
    DAILY_RECAP_HOUR, DAILY_RECAP_ENABLED,
    DREAM_JOURNAL_ENABLED, DREAM_JOURNAL_HOUR, DREAM_JOURNAL_MAX_PER_DAY,
    MOOD_STATUS_ENABLED, MOOD_STATUS_INTERVAL,
)
from database import Database
from pollinations import PollinationsAPI
from personality import PersonalityEngine, DecisionEngine
from relationships import RelationshipEngine
from memories import MemorySystem
from model_router import ModelRouter
from quotas import QuotaSystem

# ── Logging ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("lily")

# ── Bot setup ────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
intents.typing = True
intents.dm_messages = True


class LilyBot(commands.Bot):
    """Custom bot class with shared resources for v8.5."""

    def __init__(self):
        super().__init__(
            command_prefix=self._get_prefix,
            intents=intents,
            application_id=None,
            help_command=None,
        )
        # Core systems
        self.db = Database()
        self.api = PollinationsAPI()
        self.personality = PersonalityEngine()
        self.relationships = RelationshipEngine()
        self.memories = MemorySystem()
        self.model_router = ModelRouter()
        self.quotas = QuotaSystem()
        self.decision = DecisionEngine()

        # Model cache
        self._models_cache: list = []
        self._models_last_fetch: float = 0

        # Dream journal tracking
        self._dreams_today: int = 0
        self._last_dream_date: str = ""

        # Mood status tracking
        self._last_mood_status: str = ""

    async def _get_prefix(self, bot, message: discord.Message) -> list[str]:
        """Dynamic prefix based on guild settings.
        Note: We do NOT include <@bot_id> as a prefix here because
        on_message uses mentioned_in() to detect mentions, and
        having the mention as a prefix causes the early-return
        check to swallow @mention messages before they reach
        the AI response logic.
        """
        if not message.guild:
            return [BOT_PREFIX, ""]
        guild_id = message.guild.id
        prefix = self.db.get_guild_setting(guild_id, "prefix", BOT_PREFIX)
        return [prefix]

    async def setup_hook(self):
        """Load all cogs and start background tasks."""
        # v8.5 cogs — no 3D, audio, or video
        cog_list = [
            "cogs.core",
            "cogs.ai_chat",
            "cogs.image_gen",
            "cogs.models",
            "cogs.account",
            "cogs.admin",
            "cogs.relationships_cog",
            "cogs.memory_cog",
        ]

        for cog in cog_list:
            try:
                await self.load_extension(cog)
                log.info(f"Loaded cog: {cog}")
            except Exception as e:
                log.error(f"Failed to load cog {cog}: {e}")

        # Sync slash commands
        try:
            synced = await self.tree.sync()
            log.info(f"Synced {len(synced)} slash commands")
        except Exception as e:
            log.error(f"Failed to sync commands: {e}")

        # Fetch models for smart routing
        await self._fetch_models()

        # Start background tasks
        self.proactive_dm_loop.start()
        self.daily_recap_loop.start()
        self.model_refresh_loop.start()
        self.mood_status_loop.start()
        self.dream_journal_loop.start()

    async def _fetch_models(self):
        """Fetch and cache model data from Pollinations API."""
        try:
            models = await self.api.list_models()
            self._models_cache = models
            self._models_last_fetch = datetime.now().timestamp()
            self.model_router.update_models(models)
            log.info(f"Cached {len(models)} models from Pollinations API")
        except Exception as e:
            log.error(f"Failed to fetch models: {e}")

    async def get_model_for_task(self, task: str, guild_id: int = 0) -> str:
        """Get the best model for a task, respecting guild overrides."""
        guild_model = None
        if guild_id:
            guild_model = self.db.get_guild_setting(guild_id, "text_model", None)
        return self.model_router.route(task, guild_model)

    # ── Background Tasks ─────────────────────────────────

    @tasks.loop(seconds=PROACTIVE_DM_CHECK_INTERVAL)
    async def proactive_dm_loop(self):
        """Check if Lily should proactively DM anyone."""
        if not PROACTIVE_DM_ENABLED:
            return

        try:
            for guild in self.guilds:
                # Check if proactive DMs are enabled for this guild
                enabled = self.db.get_guild_setting(guild.id, "proactive_dm_enabled", 1)
                if not enabled:
                    continue

                # Check a random subset of members
                # Requires Server Members Intent — if not enabled, guild.members is empty
                try:
                    members = [m for m in guild.members if not m.bot and m.id not in ADMIN_IDS]
                except Exception:
                    members = []
                if not members:
                    continue

                # Check a few random members per cycle
                sample_size = min(3, len(members))
                sample = random.sample(members, sample_size)

                for member in sample:
                    try:
                        should_dm, reason = self.relationships.should_proactive_dm(
                            guild.id, member.id
                        )
                        if should_dm:
                            await self._send_proactive_dm(guild, member, reason)
                    except Exception as e:
                        log.error(f"Proactive DM error for {member.id}: {e}")

        except Exception as e:
            log.error(f"Proactive DM loop error: {e}")

    @tasks.loop(minutes=60)
    async def daily_recap_loop(self):
        """Generate daily recaps at the configured hour."""
        if not DAILY_RECAP_ENABLED:
            return

        now = datetime.now()
        if now.hour != DAILY_RECAP_HOUR:
            return

        log.info("Running daily recap generation...")
        try:
            for guild in self.guilds:
                enabled = self.db.get_guild_setting(guild.id, "daily_recap_enabled", 1)
                if not enabled:
                    continue

                # Get all users with relationships
                rels = self.db.get_all_relationships(guild.id)
                for rel_data in rels:
                    user_id = int(rel_data["user_id"])
                    # Only generate recaps for users with some warmth
                    warmth = rel_data.get("affection", 0) - rel_data.get("annoyance", 0)
                    if warmth < 0.1:
                        continue

                    try:
                        conversations = self.db.get_today_conversations(guild.id, user_id)
                        facts = self.db.get_facts(guild.id, user_id, cross_server=True)
                        recap = self.memories.generate_daily_recap(
                            guild.id, user_id, conversations, facts
                        )
                        if recap:
                            self.db.save_daily_recap(guild.id, user_id, recap)
                            # Save important memories from the recap
                            self.db.save_memory(
                                guild.id, user_id, recap,
                                memory_type="recap", emotion="reflective",
                                importance=0.7, tags=["daily_recap"],
                                is_global=True
                            )
                            log.info(f"Generated daily recap for user {user_id} in guild {guild.id}")
                    except Exception as e:
                        log.error(f"Daily recap error for {user_id}: {e}")

        except Exception as e:
            log.error(f"Daily recap loop error: {e}")

    @tasks.loop(hours=6)
    async def model_refresh_loop(self):
        """Refresh model data periodically."""
        await self._fetch_models()

    @tasks.loop(seconds=MOOD_STATUS_INTERVAL)
    async def mood_status_loop(self):
        """Update Lily's Discord status based on her mood. Mood-reactive status."""
        if not MOOD_STATUS_ENABLED:
            return

        try:
            # Update mood
            mood, intensity = self.personality.mood.update()

            # Get the status for this mood
            status_config = self.personality.mood.get_discord_status()
            status_text = status_config.get("text", "✨ daydreaming... | /help")
            status_type = status_config.get("type", "playing")

            # Don't update if it hasn't changed
            if status_text == self._last_mood_status:
                return

            # Map status type to Discord activity type
            type_map = {
                "playing": discord.ActivityType.playing,
                "listening": discord.ActivityType.listening,
                "watching": discord.ActivityType.watching,
                "streaming": discord.ActivityType.streaming,
            }
            activity_type = type_map.get(status_type, discord.ActivityType.playing)

            await self.change_presence(
                activity=discord.Activity(
                    type=activity_type,
                    name=status_text,
                )
            )
            self._last_mood_status = status_text
            log.debug(f"Updated mood status: {mood} → {status_text}")

        except Exception as e:
            log.error(f"Mood status loop error: {e}")

    @tasks.loop(minutes=60)
    async def dream_journal_loop(self):
        """Generate dream journal entries when Lily is in a dreamy/sleepy mood."""
        if not DREAM_JOURNAL_ENABLED:
            return

        now = datetime.now()

        # Only generate dreams at night (between 1 AM and 5 AM)
        if not (1 <= now.hour <= 5):
            return

        # Reset dream count if it's a new day
        today = now.strftime("%Y-%m-%d")
        if today != self._last_dream_date:
            self._dreams_today = 0
            self._last_dream_date = today

        # Don't generate too many dreams per day
        if self._dreams_today >= DREAM_JOURNAL_MAX_PER_DAY:
            return

        # Only dream if she's in a dreamy or sleepy mood
        mood, _ = self.personality.mood.update()
        if mood not in ("dreamy", "sleepy", "cozy"):
            return

        # 30% chance per check cycle
        if random.random() > 0.3:
            return

        try:
            await self._generate_dream()
        except Exception as e:
            log.error(f"Dream journal error: {e}")

    async def _generate_dream(self):
        """Generate a dream journal entry."""
        mood, _ = self.personality.mood.update()

        # Get recent memories for dream inspiration
        recent_memories = self.db.get_memories(0, 0, memory_type="long_term", limit=3, cross_server=True)
        recent_dreams = self.db.get_dreams(count=2)

        # Build inspiration from recent memories
        inspiration = ""
        if recent_memories:
            inspiration = "Recent things on your mind:\n"
            for m in recent_memories[:3]:
                inspiration += f"  - {m.get('content', '')[:80]}\n"

        dream_memories = ""
        if recent_dreams:
            dream_memories = "Your recent dreams:\n"
            for d in recent_dreams[:2]:
                dream_memories += f"  - {d.get('dream_text', '')[:80]}\n"

        # Build dream prompt
        dream_prompt = self.personality.build_dream_prompt(
            mood, inspiration, dream_memories
        )

        # Use a cheap model for dream generation
        model = await self.get_model_for_task("dream_journal")

        try:
            # Set special status while dreaming
            if MOOD_STATUS_ENABLED:
                dream_status = self.personality.mood.get_special_status("writing_dream")
                type_map = {
                    "playing": discord.ActivityType.playing,
                    "listening": discord.ActivityType.listening,
                    "watching": discord.ActivityType.watching,
                }
                activity_type = type_map.get(dream_status.get("type", "playing"), discord.ActivityType.playing)
                await self.change_presence(
                    activity=discord.Activity(
                        type=activity_type,
                        name=dream_status.get("text", "💭 writing in her dream journal..."),
                    )
                )

            dream_text = await self.api.chat_completions_simple(
                [{"role": "system", "content": dream_prompt},
                 {"role": "user", "content": "Write your dream journal entry for tonight."}],
                model=model,
                max_tokens=400,
                temperature=0.9,
            )

            # Save the dream
            self.db.save_dream(
                dream_text=dream_text,
                mood=mood,
                inspiration=inspiration[:200],
            )
            self.memories.add_dream(dream_text, mood)

            self._dreams_today += 1
            log.info(f"Lily wrote a dream journal entry (mood: {mood})")

            # Restore mood status
            if MOOD_STATUS_ENABLED:
                status_config = self.personality.mood.get_discord_status()
                type_map = {
                    "playing": discord.ActivityType.playing,
                    "listening": discord.ActivityType.listening,
                    "watching": discord.ActivityType.watching,
                }
                activity_type = type_map.get(status_config.get("type", "playing"), discord.ActivityType.playing)
                await self.change_presence(
                    activity=discord.Activity(
                        type=activity_type,
                        name=status_config.get("text", "✨ daydreaming... | /help"),
                    )
                )

        except Exception as e:
            log.error(f"Failed to generate dream: {e}")

    async def _send_proactive_dm(self, guild: discord.Guild, member: discord.Member, reason: str):
        """Send a proactive DM to a user."""
        try:
            # Get relationship context
            rel = self.relationships.get_relationship(guild.id, member.id)
            warmth = rel.warmth

            # Build the DM message
            model = await self.get_model_for_task("proactive_dm", guild.id)

            # Build context — cross-server memories
            user_facts = self.db.get_facts(guild.id, member.id, cross_server=True)
            recent_recaps = self.db.get_daily_recaps(guild.id, member.id, 3)
            memory_context = self.memories.get_memories_for_prompt(guild.id, member.id)
            relationship_context = self.relationships.get_system_prompt_addition(guild.id, member.id)
            dream_context = self.memories.get_dreams_for_prompt(member.id)

            mood, _ = self.personality.mood.update()
            system_prompt = self.personality.build_system_prompt(
                mood, self.personality.mood.get_energy(),
                user_facts, relationship_context, memory_context,
                [r.get("recap_text", "") for r in recent_recaps] if recent_recaps else None,
                dream_context,
            )

            # Generate the DM based on reason
            reason_prompts = {
                "morning_check_in": "You just thought of this person and wanted to say good morning. Keep it casual and warm, like you're texting a friend.",
                "night_check_in": "It's late and you're thinking of this person. Send a short, cozy message. Maybe ask how their day was.",
                "just_thinking": "You were just thinking about this person and wanted to say hi. Keep it short and sweet.",
                "saw_something_reminded": "Something reminded you of this person. You wanted to reach out. Keep it casual.",
                "wanted_to_chat": "You're bored and want to chat. Be casual and friendly about it.",
                "missed_you": "You haven't talked to this person in a bit and you miss them. Be genuine but not clingy.",
                "random_thought": "You had a random thought and wanted to share it with this person. Be casual and fun.",
                "felt_like_saying_hi": "You just felt like saying hi. No big reason. Keep it short.",
                "follow_up_topic": "You were thinking about something you talked about before. Follow up on it casually.",
                "haven_talked_in_a_while": "It's been a while since you talked. You wanted to check in. Be warm but not overwhelming.",
                "had_a_dream": "You had a dream about this person or something reminded you of a dream. Share it briefly.",
            }

            dm_prompt = reason_prompts.get(reason, "You wanted to say hi to this person. Keep it casual and natural.")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"[System: You're sending a DM to someone. {dm_prompt} Don't mention you're a bot. Just be yourself. Write the message you'd send.]"},
            ]

            response = await self.api.chat_completions_simple(
                messages, model=model, max_tokens=200, temperature=0.8
            )
            response = self.personality.inject_personality(response, warmth)

            # Send the DM
            dm_channel = member.dm_channel
            if dm_channel is None:
                dm_channel = await member.create_dm()

            # Add a natural typing delay
            typing_delay = self.personality.get_typing_delay(len(response), self.personality.mood.get_energy())
            async with dm_channel.typing():
                await asyncio.sleep(min(typing_delay, 3.0))

            await dm_channel.send(response)
            log.info(f"Sent proactive DM to {member.display_name} ({member.id}) — reason: {reason}")

            # Update relationship
            self.relationships.record_action(guild.id, member.id, "casual_chat")

        except discord.Forbidden:
            log.debug(f"Cannot DM {member.id} — DMs disabled")
        except Exception as e:
            log.error(f"Failed to send proactive DM to {member.id}: {e}")

    @proactive_dm_loop.before_loop
    @daily_recap_loop.before_loop
    @model_refresh_loop.before_loop
    @mood_status_loop.before_loop
    @dream_journal_loop.before_loop
    async def before_loops(self):
        """Wait until the bot is ready before starting loops."""
        await self.wait_until_ready()

    async def on_ready(self):
        """Called when the bot is fully connected."""
        log.info(f"{'='*50}")
        log.info(f"🌸 Lily v8.5 is online! (Lily Lives)")
        log.info(f"   Servers: {len(self.guilds)}")
        log.info(f"   Pollinations API: {POLLINATIONS_BASE_URL}")
        log.info(f"   API Key: {'configured' if POLLINATIONS_KEY else 'not set (free tier only)'}")
        log.info(f"   Proactive DMs: {'enabled' if PROACTIVE_DM_ENABLED else 'disabled'}")
        log.info(f"   Daily Recaps: {'enabled' if DAILY_RECAP_ENABLED else 'disabled'}")
        log.info(f"   Dream Journal: {'enabled' if DREAM_JOURNAL_ENABLED else 'disabled'}")
        log.info(f"   Mood Status: {'enabled' if MOOD_STATUS_ENABLED else 'disabled'}")
        log.info(f"   Smart Model Routing: active (Sana Sprint / openai-fast)")
        log.info(f"   Cross-Server Memories: enabled")
        log.info(f"{'='*50}")

        # Set initial mood-reactive status
        if MOOD_STATUS_ENABLED:
            try:
                status_config = self.personality.mood.get_discord_status()
                type_map = {
                    "playing": discord.ActivityType.playing,
                    "listening": discord.ActivityType.listening,
                    "watching": discord.ActivityType.watching,
                }
                activity_type = type_map.get(status_config.get("type", "playing"), discord.ActivityType.playing)
                await self.change_presence(
                    activity=discord.Activity(
                        type=activity_type,
                        name=status_config.get("text", "✨ daydreaming... | /help"),
                    )
                )
            except Exception as e:
                log.warning(f"Could not set mood status (Presence Intent may not be enabled): {e}")

    async def on_message(self, message: discord.Message):
        """Handle all messages — the heart of Lily's personality."""
        # Ignore bot messages
        if message.author.bot:
            return

        # Process commands first
        await self.process_commands(message)

        # Don't respond to command messages (only check the text prefix,
        # NOT the mention — mentions are handled by the AI below)
        if message.guild:
            guild_prefix = self.db.get_guild_setting(message.guild.id, "prefix", BOT_PREFIX)
        else:
            guild_prefix = BOT_PREFIX
        if message.content.startswith(guild_prefix):
            return

        guild_id = message.guild.id if message.guild else 0
        user_id = message.author.id
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.user.mentioned_in(message) if self.user else False

        # Detect emotion and action
        emotion = self.personality.detect_emotion(message.content)
        action = self.relationships.detect_action(message.content, is_command=False)

        # Update relationship
        self.relationships.record_action(guild_id, user_id, action)
        rel = self.relationships.get_relationship(guild_id, user_id)
        warmth = rel.warmth

        # Apply time decay
        self.relationships.apply_time_decay(guild_id, user_id)

        # Save to conversation history
        self.db.add_conversation(guild_id, user_id, "user", message.content, emotion)

        # Extract and save topics
        topics = self.personality.extract_topics(message.content)
        for topic in topics:
            self.db.add_topic(guild_id, user_id, topic)

        # Save as memory if it's important (CROSS-SERVER)
        importance = 0.5
        if emotion in ("happy", "sad", "affection", "vulnerable", "excited"):
            importance = 0.7
        if action in ("shared_personal", "nice_compliment", "insult"):
            importance = 0.8
        if importance >= 0.6:
            self.db.save_memory(
                guild_id, user_id, message.content,
                memory_type="auto", emotion=emotion,
                importance=importance, tags=topics,
                is_global=True
            )

        # Decide whether to respond
        if is_dm or is_mentioned:
            should_reply = True
        elif message.guild:
            should_reply = self.decision.should_reply(
                message.content, is_mentioned, is_dm, warmth
            )
        else:
            should_reply = False

        if not should_reply:
            # Maybe react instead
            if not is_dm and self.decision.should_react(warmth):
                emoji = self.decision.get_reaction_emoji(emotion)
                try:
                    await message.add_reaction(emoji)
                except discord.HTTPException:
                    pass
            return

        # Generate a response
        try:
            # Get model for this task
            task = "casual_chat"
            if len(message.content) > 200:
                task = "deep_conversation"
            if emotion in ("vulnerable", "sad"):
                task = "deep_conversation"
            model = await self.get_model_for_task(task, guild_id)

            # Build context — cross-server memories
            history = self.db.get_conversations(guild_id, user_id, limit=15)
            user_facts = self.db.get_facts(guild_id, user_id, cross_server=True)
            recent_recaps = self.db.get_daily_recaps(guild_id, user_id, 3)
            memory_context = self.memories.get_memories_for_prompt(guild_id, user_id, message.content)
            relationship_context = self.relationships.get_system_prompt_addition(guild_id, user_id)
            dream_context = self.memories.get_dreams_for_prompt(user_id)

            mood, _ = self.personality.mood.update(message.content)
            system_prompt = self.personality.build_system_prompt(
                mood, self.personality.mood.get_energy(),
                user_facts, relationship_context, memory_context,
                [r.get("recap_text", "") for r in recent_recaps] if recent_recaps else None,
                dream_context,
            )

            # Build messages
            api_messages = [{"role": "system", "content": system_prompt}]
            for msg in history:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            api_messages.append({"role": "user", "content": message.content})

            # Add a natural typing delay
            typing_delay = self.personality.get_typing_delay(
                len(message.content) * 3, self.personality.mood.get_energy()
            )
            async with message.channel.typing():
                await asyncio.sleep(min(typing_delay, 5.0))

            response = await self.api.chat_completions_simple(
                api_messages, model=model, max_tokens=500
            )
            response = self.personality.inject_personality(response, warmth)

            # Maybe add a thinking prefix
            thinking = self.personality.get_thinking_prefix()
            if thinking and random.random() < 0.1:
                response = thinking + " " + response

            # Save response to conversation
            self.db.add_conversation(guild_id, user_id, "assistant", response, mood)

            # Truncate if too long
            if len(response) > 2000:
                response = response[:1997] + "..."

            await message.reply(response)

        except Exception as e:
            log.error(f"Error generating response: {e}")

    async def on_guild_join(self, guild: discord.Guild):
        """Called when Lily joins a new server."""
        log.info(f"Joined guild: {guild.name} ({guild.id})")
        self.db.get_guild_settings(guild.id)

    async def on_guild_remove(self, guild: discord.Guild):
        """Called when Lily is removed from a server."""
        log.info(f"Left guild: {guild.name} ({guild.id})")

    async def on_error(self, event, *args, **kwargs):
        """Global error handler."""
        log.error(f"Error in {event}", exc_info=True)

    async def close(self):
        """Clean up on shutdown."""
        log.info("Shutting down Lily v8.5...")
        self.proactive_dm_loop.cancel()
        self.daily_recap_loop.cancel()
        self.model_refresh_loop.cancel()
        self.mood_status_loop.cancel()
        self.dream_journal_loop.cancel()
        await self.api.close()
        await super().close()


# ── Prefix commands (legacy support) ─────────────────────

bot = LilyBot()


@bot.command(name="lily_help", aliases=["help"])
async def prefix_help(ctx):
    """Show help via prefix command."""
    embed = discord.Embed(
        title="🌸 Lily v8.5 — Help",
        description="Lily is a multi-server AI bot who actually feels real. She remembers you across ALL servers, has feelings, writes dreams, and will reach out to you.",
        color=discord.Color.pink(),
    )
    embed.add_field(
        name="Commands",
        value=(
            f"`{ctx.prefix}image <prompt>` — Generate image\n"
            f"`{ctx.prefix}mood` — Check Lily's mood\n"
            f"`{ctx.prefix}status` — Bot status\n"
            f"`{ctx.prefix}facts [@user]` — See stored facts\n"
            f"`{ctx.prefix}topics [@user]` — See topics\n"
            f"`{ctx.prefix}relationship [@user]` — See relationship\n"
            f"`{ctx.prefix}memories [@user]` — See memories\n"
            f"`{ctx.prefix}dream` — Lily shares a dream\n"
            f"`{ctx.prefix}dreams` — See dream journal\n"
            f"`{ctx.prefix}quota` — Check your generation quota\n"
            f"`{ctx.prefix}reset [@user]` — Reset memory (admin)\n"
            f"`{ctx.prefix}channel <#channel>` — Set channel (admin)\n"
            f"`{ctx.prefix}settings` — Server settings (admin)\n"
        ),
        inline=False,
    )
    embed.set_footer(text="Lily v8.5 — She lives 💕 | Cross-server memories ✨")
    await ctx.send(embed=embed)


@bot.command(name="lily_image", aliases=["image"])
async def prefix_image(ctx, *, prompt: str):
    """Generate an image via prefix command."""
    async with ctx.typing():
        try:
            guild_id = ctx.guild.id if ctx.guild else 0
            db = bot.db
            api = bot.api

            model = db.get_guild_setting(guild_id, "image_model", "sana")
            safe = db.get_guild_setting(guild_id, "safe_mode", "privacy,secrets")

            # Check quota
            rel = bot.relationships.get_relationship(guild_id, ctx.author.id)
            can_gen, reason = bot.quotas.can_generate(guild_id, ctx.author.id, "image_quick", rel.relationship_tier)
            if not can_gen:
                await ctx.send(f"❌ {reason}")
                return

            image_bytes = await api.image_generate(prompt, model=model, safe=safe)

            import io
            file = discord.File(io.BytesIO(image_bytes), filename="lily_image.png")
            embed = discord.Embed(
                title="🖼️ Generated Image",
                description=prompt[:500],
                color=discord.Color.pink(),
            )
            embed.set_image(url="attachment://lily_image.png")
            embed.set_footer(text=f"Model: {model}")
            await ctx.send(embed=embed, file=file)

            # Calculate actual cost
            cost = bot.model_router.estimate_image_cost(model)
            bot.quotas.record_generation(guild_id, ctx.author.id, "image_quick", rel.relationship_tier, actual_cost=cost)
            db.log_generation(guild_id, ctx.author.id, "image", model, prompt[:50], cost_pollen=cost)

        except Exception as e:
            await ctx.send(f"❌ Failed to generate image: {str(e)[:200]}")


@bot.command(name="lily_mood", aliases=["mood"])
async def prefix_mood(ctx):
    """Check Lily's mood via prefix command."""
    personality = bot.personality
    mood_name, intensity = personality.mood.update()
    emoji = personality.mood.get_mood_emoji()
    energy = personality.mood.get_energy()
    desc = personality.mood.get_mood_description()

    await ctx.send(f"{emoji} Lily's mood: **{mood_name.capitalize()}** — {desc} (intensity: {intensity:.0%}, energy: {energy:.0%})")


@bot.command(name="lily_status", aliases=["status"])
async def prefix_status(ctx):
    """Show bot status via prefix command."""
    if ctx.author.id not in ADMIN_IDS:
        return

    guild_count = len(bot.guilds)
    total_users = sum(g.member_count or 0 for g in bot.guilds)

    embed = discord.Embed(
        title="🌸 Lily v8.5 — Status",
        color=discord.Color.green(),
    )
    embed.add_field(name="Servers", value=str(guild_count), inline=True)
    embed.add_field(name="Total Users", value=str(total_users), inline=True)
    embed.add_field(name="Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="API", value="Pollinations", inline=True)
    embed.add_field(name="API Key", value="✅" if POLLINATIONS_KEY else "❌ (free tier)", inline=True)
    embed.add_field(name="Proactive DMs", value="✅" if PROACTIVE_DM_ENABLED else "❌", inline=True)
    embed.add_field(name="Dream Journal", value="✅" if DREAM_JOURNAL_ENABLED else "❌", inline=True)
    embed.add_field(name="Mood Status", value="✅" if MOOD_STATUS_ENABLED else "❌", inline=True)
    embed.add_field(name="Cross-Server Mem", value="✅", inline=True)

    await ctx.send(embed=embed)


@bot.command(name="lily_relationship", aliases=["relationship", "rel"])
async def prefix_relationship(ctx, user: discord.Member = None):
    """See relationship with a user."""
    guild_id = ctx.guild.id if ctx.guild else 0
    target = user or ctx.author
    rel = bot.relationships.get_relationship(guild_id, target.id)

    embed = discord.Embed(
        title=f"💕 Relationship with {target.display_name}",
        color=discord.Color.pink(),
    )
    embed.add_field(name="Tier", value=rel.relationship_tier.replace("_", " ").title(), inline=True)
    embed.add_field(name="Warmth", value=f"{rel.warmth:.0%}", inline=True)
    embed.add_field(name="Affection", value=f"{rel.affection:.0%}", inline=True)
    embed.add_field(name="Trust", value=f"{rel.trust:.0%}", inline=True)
    embed.add_field(name="Familiarity", value=f"{rel.familiarity:.0%}", inline=True)
    embed.add_field(name="Interactions", value=str(rel.total_interactions), inline=True)

    if rel.first_met:
        embed.add_field(name="First Met", value=rel.first_met[:10], inline=True)

    embed.set_footer(text="Cross-server: Lily remembers you everywhere! ✨")
    await ctx.send(embed=embed)


@bot.command(name="lily_quota", aliases=["quota"])
async def prefix_quota(ctx):
    """Check your generation quota."""
    guild_id = ctx.guild.id if ctx.guild else 0
    rel = bot.relationships.get_relationship(guild_id, ctx.author.id)
    status = bot.quotas.get_status(guild_id, ctx.author.id, rel.relationship_tier)

    embed = discord.Embed(
        title="🌸 Your Generation Quota",
        color=discord.Color.pink(),
    )
    embed.add_field(name="Pollen", value=f"{status['pollen_used']:.4f}/{status['pollen_budget']} used", inline=True)
    embed.add_field(name="Text Gens", value=f"{status['text_gens']}/{status['text_limit']}", inline=True)
    embed.add_field(name="Image Gens", value=f"{status['image_gens']}/{status['image_limit']}", inline=True)
    embed.add_field(name="Tier", value=status['tier'].replace("_", " ").title(), inline=True)
    embed.set_footer(text="Sana Sprint = 0.0001 pollen/image! 💕")

    await ctx.send(embed=embed)


@bot.command(name="lily_reset", aliases=["reset"])
async def prefix_reset(ctx, user: discord.Member = None):
    """Reset a user's memory via prefix command."""
    if ctx.author.id not in ADMIN_IDS:
        return

    guild_id = ctx.guild.id if ctx.guild else 0
    if user:
        bot.db.clear_conversations(guild_id, user.id)
        bot.db.clear_facts(guild_id, user.id)
        await ctx.send(f"✅ Memory and facts cleared for {user.mention}")
    else:
        bot.db.clear_conversations(guild_id)
        await ctx.send("✅ All conversation memory cleared for this server.")


@bot.command(name="lily_facts", aliases=["facts"])
async def prefix_facts(ctx, user: discord.Member = None):
    """See stored facts via prefix command."""
    guild_id = ctx.guild.id if ctx.guild else 0
    target = user or ctx.author

    user_facts = bot.db.get_facts(guild_id, target.id, cross_server=True)

    if user_facts:
        facts_text = "\n".join(
            f"**{f.get('category', 'general')}**: {f.get('fact', '')}"
            for f in user_facts[:10]
        )
        await ctx.send(f"📝 Facts about {target.display_name} (cross-server):\n{facts_text}")
    else:
        await ctx.send(f"Lily doesn't know anything about {target.display_name} yet.")


@bot.command(name="lily_topics", aliases=["topics"])
async def prefix_topics(ctx, user: discord.Member = None):
    """See recurring topics via prefix command."""
    guild_id = ctx.guild.id if ctx.guild else 0
    target = user or ctx.author

    user_topics = bot.db.get_topics(guild_id, target.id)

    if user_topics:
        topics_text = "\n".join(
            f"**{t.get('topic', 'unknown')}**: mentioned {t.get('mentioned_count', 0)} times"
            for t in user_topics[:10]
        )
        await ctx.send(f"📊 Topics for {target.display_name}:\n{topics_text}")
    else:
        await ctx.send(f"No topics tracked for {target.display_name} yet.")


@bot.command(name="lily_memories", aliases=["memories"])
async def prefix_memories(ctx, user: discord.Member = None):
    """See memories about a user."""
    guild_id = ctx.guild.id if ctx.guild else 0
    target = user or ctx.author

    memories = bot.db.get_memories(guild_id, target.id, limit=10, cross_server=True)

    if memories:
        memories_text = "\n".join(
            f"**[{m['memory_type']}]** {m['content'][:80]}..."
            for m in memories[:10]
        )
        await ctx.send(f"🧠 Memories about {target.display_name} (cross-server):\n{memories_text}")
    else:
        await ctx.send(f"Lily hasn't formed any memories about {target.display_name} yet.")


@bot.command(name="lily_dream", aliases=["dream"])
async def prefix_dream(ctx):
    """Lily shares one of her dreams."""
    dreams = bot.db.get_dreams(count=5)
    if not dreams:
        await ctx.send("Lily hasn't had any dreams yet... she'll write some tonight! 🌙")
        return

    dream = random.choice(dreams)
    embed = discord.Embed(
        title="🌙 Lily's Dream",
        description=dream.get("dream_text", "")[:2000],
        color=discord.Color.purple(),
    )
    embed.set_footer(text=f"Dream mood: {dream.get('mood', 'dreamy')} | Dream Journal ✨")
    await ctx.send(embed=embed)


@bot.command(name="lily_dreams", aliases=["dreams"])
async def prefix_dreams(ctx):
    """See Lily's dream journal."""
    dreams = bot.db.get_dreams(count=5)
    if not dreams:
        await ctx.send("Lily hasn't had any dreams yet... she'll write some tonight! 🌙")
        return

    embed = discord.Embed(
        title="📖 Lily's Dream Journal",
        description="Lily's recent dreams... ✨",
        color=discord.Color.purple(),
    )
    for d in dreams[:5]:
        date = d.get("created_at", "unknown")[:10]
        mood = d.get("mood", "dreamy")
        text = d.get("dream_text", "")[:150]
        if len(d.get("dream_text", "")) > 150:
            text += "..."
        embed.add_field(name=f"🌙 {date} ({mood})", value=text, inline=False)

    embed.set_footer(text="Lily dreams every night... 🌙")
    await ctx.send(embed=embed)


@bot.command(name="lily_channel", aliases=["channel"])
async def prefix_channel(ctx, channel: discord.TextChannel = None):
    """Set the bot channel via prefix command."""
    if ctx.author.id not in ADMIN_IDS:
        return

    guild_id = ctx.guild.id if ctx.guild else 0

    if channel:
        bot.db.set_guild_setting(guild_id, "allowed_channel", str(channel.id))
        await ctx.send(f"✅ Bot channel set to {channel.mention}")
    else:
        await ctx.send("Usage: `!lily_channel #channel-name`")


@bot.command(name="lily_settings", aliases=["settings"])
async def prefix_settings(ctx):
    """View server settings via prefix command."""
    if ctx.author.id not in ADMIN_IDS:
        return

    guild_id = ctx.guild.id if ctx.guild else 0
    settings = bot.db.get_guild_settings(guild_id)

    embed = discord.Embed(
        title=f"⚙️ Server Settings — {ctx.guild.name}",
        color=discord.Color.blue(),
    )
    embed.add_field(
        name="Models",
        value=(
            f"Text: `{settings.get('text_model', 'openai-fast')}`\n"
            f"Image: `{settings.get('image_model', 'sana')}`\n"
        ),
        inline=True,
    )
    embed.add_field(
        name="Behavior",
        value=(
            f"Prefix: `{settings.get('prefix', '!lily')}`\n"
            f"Reply: {settings.get('reply_chance', 0.25):.0%}\n"
            f"Reaction: {settings.get('reaction_chance', 0.40):.0%}\n"
            f"Proactive DMs: {'✅' if settings.get('proactive_dm_enabled', 1) else '❌'}\n"
            f"Daily Recaps: {'✅' if settings.get('daily_recap_enabled', 1) else '❌'}\n"
            f"Dream Journal: {'✅' if settings.get('dream_journal_enabled', 1) else '❌'}\n"
        ),
        inline=True,
    )

    await ctx.send(embed=embed)


# ── Run ──────────────────────────────────────────────────

if __name__ == "__main__":
    if not DISCORD_TOKEN or DISCORD_TOKEN == "YOUR_DISCORD_BOT_TOKEN_HERE":
        print("ERROR: DISCORD_TOKEN is not set. Create a .env file from .env.example")
        sys.exit(1)

    print("🌸 Starting Lily v8.5 — Lily Lives...")
    print(f"   Pollinations API: {POLLINATIONS_BASE_URL}")
    print(f"   API Key: {'configured' if POLLINATIONS_KEY else 'not set (free tier)'}")
    print(f"   Admin IDs: {ADMIN_IDS}")
    print(f"   Proactive DMs: {'enabled' if PROACTIVE_DM_ENABLED else 'disabled'}")
    print(f"   Daily Recaps: {'enabled' if DAILY_RECAP_ENABLED else 'disabled'}")
    print(f"   Dream Journal: {'enabled' if DREAM_JOURNAL_ENABLED else 'disabled'}")
    print(f"   Mood Status: {'enabled' if MOOD_STATUS_ENABLED else 'disabled'}")
    print(f"   Cross-Server Memories: enabled")

    try:
        bot.run(DISCORD_TOKEN, log_handler=None)
    except KeyboardInterrupt:
        print("\n🌸 Lily is shutting down...")
    except discord.LoginFailure:
        print("ERROR: Invalid Discord token. Check your .env file.")
        sys.exit(1)
