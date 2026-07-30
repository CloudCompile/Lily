#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Admin Cog

Server administration commands for guild settings.
v9.0: Dream journal toggle, updated defaults, health dashboard.
"""

from __future__ import annotations
import time
import discord
from discord.ext import commands
from discord import app_commands

from database import Database
from config import ADMIN_IDS, PROACTIVE_DM_ENABLED, DAILY_RECAP_ENABLED, DREAM_JOURNAL_ENABLED


class AdminCog(commands.Cog, name="Admin"):
    """Server administration commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._start_time = time.time()

    def _is_admin(self, user_id: int) -> bool:
        return user_id in ADMIN_IDS

    @commands.hybrid_command(name="stats", description="Lily health dashboard (admin only)")
    async def stats(self, ctx: commands.Context):
        """Comprehensive bot health dashboard."""
        if not self._is_admin(ctx.author.id):
            if ctx.interaction:
                await ctx.send("Admin only.", ephemeral=True)
            else:
                await ctx.send("Admin only.")
            return

        db: Database = self.bot.db  # type: ignore
        db_stats = db.get_stats()

        # Calculate uptime
        uptime_seconds = int(time.time() - self._start_time)
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, _ = divmod(remainder, 60)
        uptime_str = f"{days}d {hours}h {minutes}m" if days else f"{hours}h {minutes}m"

        # Get mood
        personality = self.bot.personality  # type: ignore
        mood = personality.mood.current_mood
        energy = personality.mood.get_energy()

        # Memory usage
        try:
            import psutil
            import os as _os
            process = psutil.Process(_os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            mem_str = f"{mem_mb:.1f} MB"
        except ImportError:
            mem_str = "N/A"

        # API status
        api = self.bot.api  # type: ignore
        api_status = "connected" if api._session and not api._session.closed else "not initialized"

        # Partner status
        partner_cog = self.bot.get_cog("Bot Interaction")
        partner_status = "configured" if partner_cog else "not set"

        embed = discord.Embed(
            title="🌸 Lily v9.0 — Health Dashboard",
            color=discord.Color.pink(),
        )
        embed.add_field(
            name="System",
            value=(
                f"Uptime: {uptime_str}\n"
                f"Servers: {len(self.bot.guilds)}\n"
                f"Latency: {round(self.bot.latency * 1000)}ms\n"
                f"Memory: {mem_str}\n"
                f"API: {api_status}"
            ),
            inline=True,
        )
        embed.add_field(
            name="Personality",
            value=(
                f"Mood: {mood}\n"
                f"Energy: {energy:.1f}/1.0\n"
                f"Partner: {partner_status}"
            ),
            inline=True,
        )
        embed.add_field(
            name="Memory",
            value=(
                f"Conversations: {db_stats.get('total_conversations', 0)}\n"
                f"Users: {db_stats.get('total_users', 0)}\n"
                f"Memories: {db_stats.get('total_memories', 0)}\n"
                f"Facts: {db_stats.get('total_facts', 0)}\n"
                f"Dreams: {db_stats.get('total_dreams', 0)}"
            ),
            inline=True,
        )
        embed.add_field(
            name="Generations",
            value=(
                f"Total: {db_stats.get('total_generations', 0)}\n"
                f"Today: {db_stats.get('today_generations', 0)}\n"
                f"Bot Chats: {db_stats.get('bot_interactions', 0)}\n"
                f"Pollen Spent: {db_stats.get('total_pollen_spent', 0):.4f}"
            ),
            inline=True,
        )
        embed.set_footer(text="Lily v9.0 — She lives 💕 | Retry Logic ✅ | Cross-Server Memory ✅")
        if ctx.interaction:
            await ctx.send(embed=embed, ephemeral=True)
        else:
            await ctx.send(embed=embed)

    @commands.hybrid_command(name="set_channel", description="Set the bot's allowed channel")
    @app_commands.describe(channel="The channel where Lily should respond")
    async def set_channel(
        self,
        ctx: commands.Context,
        channel: discord.TextChannel,
    ):
        """Set the bot's allowed channel."""
        if not self._is_admin(ctx.author.id):
            if ctx.interaction:
                await ctx.send("❌ Admin only.", ephemeral=True)
            else:
                await ctx.send("❌ Admin only.")
            return

        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore
        db.set_guild_setting(guild_id, "allowed_channel", str(channel.id))
        await ctx.send(f"✅ Bot channel set to {channel.mention}")

    @commands.hybrid_command(name="set_model", description="Set the default model for a generation type")
    @app_commands.describe(
        model_type="Type: text or image",
        model="Model name (e.g. openai-fast, sana, flux, tomdacatto/ling-3.0-flash)"
    )
    async def set_model(
        self,
        ctx: commands.Context,
        model_type: str,
        model: str,
    ):
        """Set the default model for a generation type."""
        if not self._is_admin(ctx.author.id):
            if ctx.interaction:
                await ctx.send("❌ Admin only.", ephemeral=True)
            else:
                await ctx.send("❌ Admin only.")
            return

        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore

        type_map = {
            "text": "text_model",
            "image": "image_model",
        }

        setting_key = type_map.get(model_type.lower())
        if not setting_key:
            if ctx.interaction:
                await ctx.send(
                    f"❌ Invalid model type. Use: {', '.join(type_map.keys())}", ephemeral=True
                )
            else:
                await ctx.send(
                    f"❌ Invalid model type. Use: {', '.join(type_map.keys())}"
                )
            return

        db.set_guild_setting(guild_id, setting_key, model)
        await ctx.send(f"✅ Default {model_type} model set to `{model}`")

    @commands.hybrid_command(name="set_prefix", description="Set the bot's command prefix")
    @app_commands.describe(prefix="New prefix (e.g. !lily, lily, !)")
    async def set_prefix(
        self,
        ctx: commands.Context,
        prefix: str,
    ):
        """Set the bot's command prefix."""
        if not self._is_admin(ctx.author.id):
            if ctx.interaction:
                await ctx.send("❌ Admin only.", ephemeral=True)
            else:
                await ctx.send("❌ Admin only.")
            return

        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore
        db.set_guild_setting(guild_id, "prefix", prefix)
        await ctx.send(f"✅ Prefix set to `{prefix}`")

    @commands.hybrid_command(name="server_settings", description="View server settings")
    async def server_settings(self, ctx: commands.Context):
        """View all server settings."""
        if not self._is_admin(ctx.author.id):
            if ctx.interaction:
                await ctx.send("❌ Admin only.", ephemeral=True)
            else:
                await ctx.send("❌ Admin only.")
            return

        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore
        settings = db.get_guild_settings(guild_id)

        guild_name = ctx.guild.name if ctx.guild else "DM"

        embed = discord.Embed(
            title=f"⚙️ Server Settings — {guild_name}",
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
            ),
            inline=True,
        )
        embed.add_field(
            name="v9.0 Features",
            value=(
                f"Proactive DMs: {'✅' if settings.get('proactive_dm_enabled', 1) else '❌'}\n"
                f"Daily Recaps: {'✅' if settings.get('daily_recap_enabled', 1) else '❌'}\n"
                f"Dream Journal: {'✅' if settings.get('dream_journal_enabled', 1) else '❌'}\n"
                f"Personality: {'✅' if settings.get('personality_enabled', 1) else '❌'}\n"
            ),
            inline=True,
        )

        if ctx.interaction:
            await ctx.send(embed=embed, ephemeral=True)
        else:
            await ctx.send(embed=embed)

    @commands.hybrid_command(name="toggle_proactive", description="Toggle proactive DMs for this server")
    @app_commands.describe(enabled="Enable or disable proactive DMs")
    async def toggle_proactive(
        self,
        ctx: commands.Context,
        enabled: bool = True,
    ):
        """Toggle proactive DMs."""
        if not self._is_admin(ctx.author.id):
            if ctx.interaction:
                await ctx.send("❌ Admin only.", ephemeral=True)
            else:
                await ctx.send("❌ Admin only.")
            return

        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore
        db.set_guild_setting(guild_id, "proactive_dm_enabled", 1 if enabled else 0)
        status = "✅ enabled" if enabled else "❌ disabled"
        await ctx.send(f"Proactive DMs are now {status} for this server.")

    @commands.hybrid_command(name="toggle_recaps", description="Toggle daily recaps for this server")
    @app_commands.describe(enabled="Enable or disable daily recaps")
    async def toggle_recaps(
        self,
        ctx: commands.Context,
        enabled: bool = True,
    ):
        """Toggle daily recaps."""
        if not self._is_admin(ctx.author.id):
            if ctx.interaction:
                await ctx.send("❌ Admin only.", ephemeral=True)
            else:
                await ctx.send("❌ Admin only.")
            return

        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore
        db.set_guild_setting(guild_id, "daily_recap_enabled", 1 if enabled else 0)
        status = "✅ enabled" if enabled else "❌ disabled"
        await ctx.send(f"Daily recaps are now {status} for this server.")

    @commands.hybrid_command(name="toggle_dreams", description="Toggle dream journal for this server")
    @app_commands.describe(enabled="Enable or disable dream journal")
    async def toggle_dreams(
        self,
        ctx: commands.Context,
        enabled: bool = True,
    ):
        """Toggle dream journal."""
        if not self._is_admin(ctx.author.id):
            if ctx.interaction:
                await ctx.send("❌ Admin only.", ephemeral=True)
            else:
                await ctx.send("❌ Admin only.")
            return

        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore
        db.set_guild_setting(guild_id, "dream_journal_enabled", 1 if enabled else 0)
        status = "✅ enabled" if enabled else "❌ disabled"
        await ctx.send(f"Dream journal is now {status} for this server.")

    @commands.hybrid_command(name="reset_user", description="Reset a user's memory and relationship")
    @app_commands.describe(user="The user to reset")
    async def reset_user(
        self,
        ctx: commands.Context,
        user: discord.Member,
    ):
        """Reset a user's memory and relationship."""
        if not self._is_admin(ctx.author.id):
            if ctx.interaction:
                await ctx.send("❌ Admin only.", ephemeral=True)
            else:
                await ctx.send("❌ Admin only.")
            return

        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore
        db.clear_conversations(guild_id, user.id)
        db.clear_facts(guild_id, user.id)
        await ctx.send(f"✅ Memory and facts cleared for {user.mention}")


async def setup(bot: commands.Bot):
    await bot.add_cog(AdminCog(bot))
