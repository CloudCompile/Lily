#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Core Bot Cog

Essential commands: help, status, mood, ping, info.
v9.0: Dream journal, mood-reactive status, cross-server memories.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands

from database import Database
from pollinations import PollinationsAPI
from personality import MoodSystem, PersonalityEngine, DecisionEngine
from relationships import RelationshipEngine
from config import (
    ADMIN_IDS, BOT_PREFIX, POLLINATIONS_KEY,
    PROACTIVE_DM_ENABLED, DAILY_RECAP_ENABLED,
    DREAM_JOURNAL_ENABLED, MOOD_STATUS_ENABLED,
)


class CoreCog(commands.Cog, name="Core"):
    """Essential bot commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="help", description="Show Lily's commands and features")
    async def help_slash(self, interaction: discord.Interaction):
        """Show all available commands."""
        embed = discord.Embed(
            title="🌸 Lily v9.0 — Command Guide",
            description="Lily is a multi-server AI bot who actually feels real. She remembers you across ALL servers, has feelings, writes dreams, and will reach out to you.",
            color=discord.Color.pink(),
        )

        # Text & Chat
        embed.add_field(
            name="💬 AI Chat",
            value=(
                "`/ask <question>` — Ask Lily anything\n"
                "`/chat <message>` — Have a conversation\n"
                "`/imagine <prompt>` — Creative text generation\n"
                "`/analyze <url> <question>` — Analyze an image\n"
                "`/translate <text> <language>` — Translate text"
            ),
            inline=False,
        )

        # Image
        embed.add_field(
            name="🖼️ Image Generation",
            value=(
                "`/image <prompt>` — Generate an image (Sana Sprint!)\n"
                "`/image_advanced <prompt> ...` — Generate with full options\n"
                "`/image_edit <prompt>` — Edit an attached image"
            ),
            inline=False,
        )

        # Relationships & Memory
        embed.add_field(
            name="💕 Relationships & Memory",
            value=(
                "`/relationship` — See your relationship with Lily\n"
                "`/quota` — Check your generation quota\n"
                "`/memories` — See what Lily remembers about you\n"
                "`/recaps` — See Lily's diary entries about you\n"
                "`/remember <what>` — Tell Lily to remember something\n"
                "`/forget <what>` — Ask Lily to forget something"
            ),
            inline=False,
        )

        # Dream Journal
        embed.add_field(
            name="🌙 Dream Journal",
            value=(
                "`/dream` — Lily shares a dream with you\n"
                "`/dream_journal` — See Lily's dream journal"
            ),
            inline=False,
        )

        # Models & Info
        embed.add_field(
            name="📋 Models & Info",
            value=(
                "`/models [category]` — List available models\n"
                "`/model_info <model>` — Get model details\n"
                "`/balance` — Check API balance\n"
                "`/mood` — Check Lily's current mood"
            ),
            inline=False,
        )

        # Admin
        embed.add_field(
            name="⚙️ Server Admin",
            value=(
                "`/set_channel` — Set bot channel\n"
                "`/set_model <type> <model>` — Set default model\n"
                "`/set_prefix <prefix>` — Set command prefix\n"
                "`/server_settings` — View server settings\n"
                "`/toggle_proactive` — Toggle proactive DMs\n"
                "`/toggle_recaps` — Toggle daily recaps\n"
                "`/toggle_dreams` — Toggle dream journal\n"
                "`/reset_user <user>` — Reset user memory (admin)"
            ),
            inline=False,
        )

        embed.set_footer(text="Lily v9.0 — She lives 💕 | Cross-server memories ✨ | Sana Sprint images 🖼️")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="ping", description="Check Lily's response time")
    async def ping(self, interaction: discord.Interaction):
        """Simple ping/pong command."""
        latency = round(self.bot.latency * 1000)
        await interaction.response.send_message(
            f"🌸 Pong! {latency}ms"
        )

    @app_commands.command(name="mood", description="Check Lily's current mood")
    async def mood(self, interaction: discord.Interaction):
        """Display Lily's current mood state."""
        personality: PersonalityEngine = self.bot.personality  # type: ignore
        mood_name, intensity = personality.mood.update()
        emoji = personality.mood.get_mood_emoji()
        energy = personality.mood.get_energy()
        desc = personality.mood.get_mood_description()

        # Get the current status text
        status_config = personality.mood.get_discord_status()
        status_text = status_config.get("text", "✨ daydreaming...")

        embed = discord.Embed(
            title=f"{emoji} Lily's Mood",
            color=discord.Color.pink(),
        )
        embed.add_field(name="Mood", value=mood_name.capitalize(), inline=True)
        embed.add_field(name="Intensity", value=f"{intensity:.0%}", inline=True)
        embed.add_field(name="Energy", value=f"{energy:.0%}", inline=True)

        mood_descriptions = {
            "sleepy": "Feeling drowsy... might need a nap 💤",
            "morning": "Just woke up, feeling fresh! ☀️",
            "energetic": "Full of energy and ready to go! ⚡",
            "chill": "Vibing, just relaxing ☕",
            "cozy": "Feeling warm and cozy 🌙",
            "dreamy": "Lost in thought... ✨",
        }
        embed.description = mood_descriptions.get(mood_name, "Feeling alright!")
        embed.add_field(name="Discord Status", value=status_text, inline=False)
        embed.set_footer(text="Her mood changes with the time of day! 💕")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="info", description="Show bot information")
    async def info(self, interaction: discord.Interaction):
        """Display bot information and stats."""
        db: Database = self.bot.db  # type: ignore
        guild_count = len(self.bot.guilds)

        embed = discord.Embed(
            title="🌸 Lily v9.0 — Lily Lives",
            description="Multi-server AI Discord Bot who actually feels real. She remembers you across ALL servers, has feelings, writes dreams, and will reach out to you.",
            color=discord.Color.pink(),
        )
        embed.add_field(name="Servers", value=str(guild_count), inline=True)
        embed.add_field(name="API", value="Pollinations", inline=True)
        embed.add_field(name="Image Model", value="Sana Sprint (0.0001/gen!)", inline=True)
        embed.add_field(name="Features", value="Text, Image, Relationships, Memories, Proactive DMs, Dream Journal, Mood Status", inline=False)
        embed.add_field(name="Proactive DMs", value="✅" if PROACTIVE_DM_ENABLED else "❌", inline=True)
        embed.add_field(name="Daily Recaps", value="✅" if DAILY_RECAP_ENABLED else "❌", inline=True)
        embed.add_field(name="Dream Journal", value="✅" if DREAM_JOURNAL_ENABLED else "❌", inline=True)
        embed.add_field(name="Mood Status", value="✅" if MOOD_STATUS_ENABLED else "❌", inline=True)
        embed.add_field(name="Cross-Server Mem", value="✅", inline=True)
        embed.add_field(name="Smart Routing", value="✅", inline=True)
        embed.set_footer(text="Lily v9.0 — She lives 💕 | github.com/cloudcompile/Lily")
        await interaction.response.send_message(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(CoreCog(bot))
