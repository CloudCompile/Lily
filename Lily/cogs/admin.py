#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.5 — Admin Cog

Server administration commands for guild settings.
v8.5: Dream journal toggle, updated defaults.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands

from database import Database
from config import ADMIN_IDS, PROACTIVE_DM_ENABLED, DAILY_RECAP_ENABLED, DREAM_JOURNAL_ENABLED


class AdminCog(commands.Cog, name="Admin"):
    """Server administration commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    def _is_admin(self, user_id: int) -> bool:
        return user_id in ADMIN_IDS

    @app_commands.command(name="set_channel", description="Set the bot's allowed channel")
    @app_commands.describe(channel="The channel where Lily should respond")
    async def set_channel(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel,
    ):
        """Set the bot's allowed channel."""
        if not self._is_admin(interaction.user.id):
            await interaction.response.send_message("❌ Admin only.", ephemeral=True)
            return

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        db.set_guild_setting(guild_id, "allowed_channel", str(channel.id))
        await interaction.response.send_message(f"✅ Bot channel set to {channel.mention}")

    @app_commands.command(name="set_model", description="Set the default model for a generation type")
    @app_commands.describe(
        model_type="Type: text or image",
        model="Model name (e.g. openai-fast, sana, flux, tomdacatto/ling-3.0-flash)"
    )
    async def set_model(
        self,
        interaction: discord.Interaction,
        model_type: str,
        model: str,
    ):
        """Set the default model for a generation type."""
        if not self._is_admin(interaction.user.id):
            await interaction.response.send_message("❌ Admin only.", ephemeral=True)
            return

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore

        type_map = {
            "text": "text_model",
            "image": "image_model",
        }

        setting_key = type_map.get(model_type.lower())
        if not setting_key:
            await interaction.response.send_message(
                f"❌ Invalid model type. Use: {', '.join(type_map.keys())}", ephemeral=True
            )
            return

        db.set_guild_setting(guild_id, setting_key, model)
        await interaction.response.send_message(f"✅ Default {model_type} model set to `{model}`")

    @app_commands.command(name="set_prefix", description="Set the bot's command prefix")
    @app_commands.describe(prefix="New prefix (e.g. !lily, lily, !)")
    async def set_prefix(
        self,
        interaction: discord.Interaction,
        prefix: str,
    ):
        """Set the bot's command prefix."""
        if not self._is_admin(interaction.user.id):
            await interaction.response.send_message("❌ Admin only.", ephemeral=True)
            return

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        db.set_guild_setting(guild_id, "prefix", prefix)
        await interaction.response.send_message(f"✅ Prefix set to `{prefix}`")

    @app_commands.command(name="server_settings", description="View server settings")
    async def server_settings(self, interaction: discord.Interaction):
        """View all server settings."""
        if not self._is_admin(interaction.user.id):
            await interaction.response.send_message("❌ Admin only.", ephemeral=True)
            return

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        settings = db.get_guild_settings(guild_id)

        embed = discord.Embed(
            title=f"⚙️ Server Settings — {interaction.guild.name}",
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
            name="v8.5 Features",
            value=(
                f"Proactive DMs: {'✅' if settings.get('proactive_dm_enabled', 1) else '❌'}\n"
                f"Daily Recaps: {'✅' if settings.get('daily_recap_enabled', 1) else '❌'}\n"
                f"Dream Journal: {'✅' if settings.get('dream_journal_enabled', 1) else '❌'}\n"
                f"Personality: {'✅' if settings.get('personality_enabled', 1) else '❌'}\n"
            ),
            inline=True,
        )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="toggle_proactive", description="Toggle proactive DMs for this server")
    @app_commands.describe(enabled="Enable or disable proactive DMs")
    async def toggle_proactive(
        self,
        interaction: discord.Interaction,
        enabled: bool = True,
    ):
        """Toggle proactive DMs."""
        if not self._is_admin(interaction.user.id):
            await interaction.response.send_message("❌ Admin only.", ephemeral=True)
            return

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        db.set_guild_setting(guild_id, "proactive_dm_enabled", 1 if enabled else 0)
        status = "✅ enabled" if enabled else "❌ disabled"
        await interaction.response.send_message(f"Proactive DMs are now {status} for this server.")

    @app_commands.command(name="toggle_recaps", description="Toggle daily recaps for this server")
    @app_commands.describe(enabled="Enable or disable daily recaps")
    async def toggle_recaps(
        self,
        interaction: discord.Interaction,
        enabled: bool = True,
    ):
        """Toggle daily recaps."""
        if not self._is_admin(interaction.user.id):
            await interaction.response.send_message("❌ Admin only.", ephemeral=True)
            return

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        db.set_guild_setting(guild_id, "daily_recap_enabled", 1 if enabled else 0)
        status = "✅ enabled" if enabled else "❌ disabled"
        await interaction.response.send_message(f"Daily recaps are now {status} for this server.")

    @app_commands.command(name="toggle_dreams", description="Toggle dream journal for this server")
    @app_commands.describe(enabled="Enable or disable dream journal")
    async def toggle_dreams(
        self,
        interaction: discord.Interaction,
        enabled: bool = True,
    ):
        """Toggle dream journal."""
        if not self._is_admin(interaction.user.id):
            await interaction.response.send_message("❌ Admin only.", ephemeral=True)
            return

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        db.set_guild_setting(guild_id, "dream_journal_enabled", 1 if enabled else 0)
        status = "✅ enabled" if enabled else "❌ disabled"
        await interaction.response.send_message(f"Dream journal is now {status} for this server.")

    @app_commands.command(name="reset_user", description="Reset a user's memory and relationship")
    @app_commands.describe(user="The user to reset")
    async def reset_user(
        self,
        interaction: discord.Interaction,
        user: discord.Member,
    ):
        """Reset a user's memory and relationship."""
        if not self._is_admin(interaction.user.id):
            await interaction.response.send_message("❌ Admin only.", ephemeral=True)
            return

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        db.clear_conversations(guild_id, user.id)
        db.clear_facts(guild_id, user.id)
        await interaction.response.send_message(f"✅ Memory and facts cleared for {user.mention}")


async def setup(bot: commands.Bot):
    await bot.add_cog(AdminCog(bot))
