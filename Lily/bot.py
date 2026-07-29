#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Multi-Server AI Discord Bot

Powered by Pollinations API — Text, Image, Video, Audio, 3D, Embeddings.
Fully multi-server capable with per-guild settings.

Run: python bot.py
"""

from __future__ import annotations
import asyncio
import logging
import sys
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    DISCORD_TOKEN, ADMIN_IDS, BOT_PREFIX,
    POLLINATIONS_KEY, POLLINATIONS_BASE_URL,
)
from database import Database
from pollinations import PollinationsAPI
from personality import PersonalityEngine, DecisionEngine

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
intents.members = True
intents.reactions = True
intents.typing = True


class LilyBot(commands.Bot):
    """Custom bot class with shared resources."""

    def __init__(self):
        super().__init__(
            command_prefix=self._get_prefix,
            intents=intents,
            application_id=None,
        )
        # Shared resources
        self.db = Database()
        self.api = PollinationsAPI()
        self.personality = PersonalityEngine()

    async def _get_prefix(self, bot, message: discord.Message) -> list[str]:
        """Dynamic prefix based on guild settings."""
        if not message.guild:
            return [BOT_PREFIX, ""]
        guild_id = message.guild.id
        prefix = self.db.get_guild_setting(guild_id, "prefix", BOT_PREFIX)
        return [prefix, f"<@{bot.user.id}> ", f"<@!{bot.user.id}> "]

    async def setup_hook(self):
        """Load all cogs and sync commands."""
        cog_list = [
            "cogs.core",
            "cogs.ai_chat",
            "cogs.image_gen",
            "cogs.video_gen",
            "cogs.audio_gen",
            "cogs.model_3d",
            "cogs.models",
            "cogs.account",
            "cogs.admin",
            "cogs.personality_cog",
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

    async def on_ready(self):
        """Called when the bot is fully connected."""
        log.info(f"{'='*50}")
        log.info(f"🌸 Lily v8.0 is online!")
        log.info(f"   Servers: {len(self.guilds)}")
        log.info(f"   Pollinations API: {POLLINATIONS_BASE_URL}")
        log.info(f"   API Key: {'configured' if POLLINATIONS_KEY else 'not set (free tier only)'}")
        log.info(f"   Admin IDs: {ADMIN_IDS}")
        log.info(f"{'='*50}")

        # Set presence
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="Pollinations AI | /help",
            )
        )

    async def on_guild_join(self, guild: discord.Guild):
        """Called when Lily joins a new server."""
        log.info(f"Joined guild: {guild.name} ({guild.id})")
        # Initialize guild settings
        self.db.get_guild_settings(guild.id)

    async def on_guild_remove(self, guild: discord.Guild):
        """Called when Lily is removed from a server."""
        log.info(f"Left guild: {guild.name} ({guild.id})")

    async def on_error(self, event, *args, **kwargs):
        """Global error handler."""
        log.error(f"Error in {event}", exc_info=True)

    async def close(self):
        """Clean up on shutdown."""
        log.info("Shutting down Lily...")
        await self.api.close()
        await super().close()


# ── Prefix commands (legacy support) ─────────────────────

bot = LilyBot()


@bot.command(name="lily_help", aliases=["help"])
async def prefix_help(ctx):
    """Show help via prefix command."""
    embed = discord.Embed(
        title="🌸 Lily v8.0 — Help",
        description="Use `/help` for the full command guide, or check these prefix commands:",
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
            f"`{ctx.prefix}reset [@user]` — Reset memory (admin)\n"
            f"`{ctx.prefix}channel <#channel>` — Set channel (admin)\n"
            f"`{ctx.prefix}settings` — Server settings (admin)\n"
        ),
        inline=False,
    )
    embed.set_footer(text="Use slash commands (/) for all features including video, audio, 3D, and more!")
    await ctx.send(embed=embed)


@bot.command(name="lily_image", aliases=["image"])
async def prefix_image(ctx, *, prompt: str):
    """Generate an image via prefix command."""
    async with ctx.typing():
        try:
            guild_id = ctx.guild.id if ctx.guild else 0
            db = bot.db
            api = bot.api

            model = db.get_guild_setting(guild_id, "image_model", "zimage")
            safe = db.get_guild_setting(guild_id, "safe_mode", "privacy,secrets")

            image_bytes = await api.image_generate(
                prompt, model=model, safe=safe
            )

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

            db.log_generation(guild_id, ctx.author.id, "image", model, prompt[:50])

        except Exception as e:
            await ctx.send(f"❌ Failed to generate image: {str(e)[:200]}")


@bot.command(name="lily_mood", aliases=["mood"])
async def prefix_mood(ctx):
    """Check Lily's mood via prefix command."""
    personality = bot.personality
    mood_name, intensity = personality.mood.update()
    emoji = personality.mood.get_mood_emoji()
    energy = personality.mood.get_energy()

    await ctx.send(f"{emoji} Lily's mood: **{mood_name.capitalize()}** (intensity: {intensity:.0%}, energy: {energy:.0%})")


@bot.command(name="lily_status", aliases=["status"])
async def prefix_status(ctx):
    """Show bot status via prefix command."""
    if ctx.author.id not in ADMIN_IDS:
        return

    guild_count = len(bot.guilds)
    total_users = sum(g.member_count or 0 for g in bot.guilds)

    embed = discord.Embed(
        title="🌸 Lily v8.0 — Status",
        color=discord.Color.green(),
    )
    embed.add_field(name="Servers", value=str(guild_count), inline=True)
    embed.add_field(name="Total Users", value=str(total_users), inline=True)
    embed.add_field(name="Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="API", value="Pollinations", inline=True)
    embed.add_field(name="API Key", value="✅" if POLLINATIONS_KEY else "❌ (free tier)", inline=True)

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

    user_facts = bot.db.get_facts(guild_id, target.id)

    if user_facts:
        facts_text = "\n".join(
            f"**{f.get('category', 'general')}**: {f.get('fact', '')}"
            for f in user_facts[:10]
        )
        await ctx.send(f"📝 Facts about {target.display_name}:\n{facts_text}")
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
            f"Text: `{settings.get('text_model', 'openai')}`\n"
            f"Image: `{settings.get('image_model', 'zimage')}`\n"
            f"Video: `{settings.get('video_model', 'veo')}`\n"
            f"TTS: `{settings.get('tts_model', 'elevenlabs')}`\n"
            f"3D: `{settings.get('model_3d', 'trellis-2-low')}`"
        ),
        inline=True,
    )
    embed.add_field(
        name="Behavior",
        value=(
            f"Prefix: `{settings.get('prefix', '!lily')}`\n"
            f"Reply: {settings.get('reply_chance', 0.25):.0%}\n"
            f"Reaction: {settings.get('reaction_chance', 0.40):.0%}\n"
            f"Safe: `{settings.get('safe_mode', 'privacy,secrets')}`"
        ),
        inline=True,
    )

    await ctx.send(embed=embed)


# ── Run ──────────────────────────────────────────────────

if __name__ == "__main__":
    if not DISCORD_TOKEN or DISCORD_TOKEN == "YOUR_DISCORD_BOT_TOKEN_HERE":
        print("ERROR: DISCORD_TOKEN is not set. Create a .env file from .env.example")
        sys.exit(1)

    print("🌸 Starting Lily v8.0...")
    print(f"   Pollinations API: {POLLINATIONS_BASE_URL}")
    print(f"   API Key: {'configured' if POLLINATIONS_KEY else 'not set (free tier)'}")
    print(f"   Admin IDs: {ADMIN_IDS}")

    try:
        bot.run(DISCORD_TOKEN, log_handler=None)
    except KeyboardInterrupt:
        print("\n🌸 Lily is shutting down...")
    except discord.LoginFailure:
        print("ERROR: Invalid Discord token. Check your .env file.")
        sys.exit(1)
