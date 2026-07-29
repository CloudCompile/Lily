#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Core Bot Cog

Essential commands: help, status, mood, ping, info.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands

from database import Database
from pollinations import PollinationsAPI
from personality import MoodSystem, PersonalityEngine, DecisionEngine
from config import ADMIN_IDS, BOT_PREFIX, DEFAULT_TEXT_MODEL


class CoreCog(commands.Cog, name="Core"):
    """Essential bot commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="help", description="Show Lily's commands and features")
    async def help_slash(self, interaction: discord.Interaction):
        """Show all available commands."""
        embed = discord.Embed(
            title="🌸 Lily v8.0 — Command Guide",
            description="Lily is a multi-server AI bot powered by Pollinations API",
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
                "`/image <prompt>` — Generate an image\n"
                "`/image_advanced <prompt> ...` — Generate with full options\n"
                "`/image_edit <prompt>` — Edit an attached image"
            ),
            inline=False,
        )

        # Video
        embed.add_field(
            name="🎬 Video Generation",
            value=(
                "`/video <prompt>` — Generate a video\n"
                "`/video_advanced <prompt> ...` — Generate with full options"
            ),
            inline=False,
        )

        # Audio
        embed.add_field(
            name="🎵 Audio",
            value=(
                "`/tts <text>` — Text to speech\n"
                "`/tts_simple <text>` — Quick TTS\n"
                "`/music <prompt>` — Generate music\n"
                "`/transcribe` — Transcribe an audio file"
            ),
            inline=False,
        )

        # 3D
        embed.add_field(
            name="🧊 3D Generation",
            value="`/3d <prompt>` — Generate a 3D model (GLB)",
            inline=False,
        )

        # Models & Info
        embed.add_field(
            name="📋 Models & Info",
            value=(
                "`/models [category]` — List available models\n"
                "`/model_info <model>` — Get model details\n"
                "`/model_status` — Check model health\n"
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
                "`/reset_user <user>` — Reset user memory (admin)"
            ),
            inline=False,
        )

        # Prefix commands
        embed.add_field(
            name="⌨️ Prefix Commands",
            value=(
                f"`{BOT_PREFIX} help` — Show this help\n"
                f"`{BOT_PREFIX} status` — Bot status\n"
                f"`{BOT_PREFIX} image <prompt>` — Generate image\n"
                f"`{BOT_PREFIX} mood` — Check mood\n"
                f"`{BOT_PREFIX} reset [@user]` — Reset memory (admin)\n"
                f"`{BOT_PREFIX} facts [@user]` — See stored facts\n"
                f"`{BOT_PREFIX} topics [@user]` — See recurring topics\n"
                f"`{BOT_PREFIX} channel` — Set channel (admin)\n"
                f"`{BOT_PREFIX} settings` — Server settings (admin)"
            ),
            inline=False,
        )

        embed.set_footer(text="Powered by Pollinations API • Multi-server ready")
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
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="info", description="Show bot information")
    async def info(self, interaction: discord.Interaction):
        """Display bot information and stats."""
        db: Database = self.bot.db  # type: ignore
        guild_count = len(self.bot.guilds)

        embed = discord.Embed(
            title="🌸 Lily v8.0",
            description="Multi-server AI Discord Bot powered by Pollinations API",
            color=discord.Color.pink(),
        )
        embed.add_field(name="Servers", value=str(guild_count), inline=True)
        embed.add_field(name="API", value="Pollinations", inline=True)
        embed.add_field(name="Features", value="Text, Image, Video, Audio, 3D, Embeddings", inline=False)
        embed.add_field(
            name="Supported Models",
            value="141 Text · 54 Image · 13 Video · 18 Audio · 4 3D · 5 Embedding",
            inline=False,
        )
        embed.set_footer(text="Open-source • github.com/cloudcompile/Lily")
        await interaction.response.send_message(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(CoreCog(bot))
