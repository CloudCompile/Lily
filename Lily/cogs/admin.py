#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Admin Cog

Server administration commands: /set_channel, /set_model, /set_prefix,
/server_settings, /reset_user, /set_safe_mode, /set_reply_chance.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands

from database import Database
from config import ADMIN_IDS


def is_admin():
    """Check if the user is a bot admin or server admin."""
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.user.id in ADMIN_IDS:
            return True
        if isinstance(interaction.user, discord.Member):
            return interaction.user.guild_permissions.administrator
        return False
    return app_commands.check(predicate)


class AdminCog(commands.Cog, name="Admin"):
    """Server administration commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="set_channel", description="Set the bot channel for this server")
    @app_commands.describe(channel="Channel where Lily should be active")
    @is_admin()
    async def set_channel(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel,
    ):
        """Set the bot's active channel for this server."""
        db: Database = self.bot.db  # type: ignore
        guild_id = interaction.guild_id

        db.set_guild_setting(guild_id, "allowed_channel", str(channel.id))

        embed = discord.Embed(
            title="✅ Channel Set",
            description=f"Lily will now be active in {channel.mention}",
            color=discord.Color.green(),
        )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="set_model", description="Set the default model for a generation type")
    @app_commands.describe(
        model_type="Type of generation",
        model_name="Model ID to use as default",
    )
    @app_commands.choices(model_type=[
        app_commands.Choice(name="Text", value="text_model"),
        app_commands.Choice(name="Image", value="image_model"),
        app_commands.Choice(name="Video", value="video_model"),
        app_commands.Choice(name="TTS", value="tts_model"),
        app_commands.Choice(name="Transcription", value="transcription_model"),
        app_commands.Choice(name="3D", value="model_3d"),
    ])
    @is_admin()
    async def set_model(
        self,
        interaction: discord.Interaction,
        model_type: str,
        model_name: str,
    ):
        """Set the default model for a generation type in this server."""
        db: Database = self.bot.db  # type: ignore
        guild_id = interaction.guild_id

        db.set_guild_setting(guild_id, model_type, model_name)

        embed = discord.Embed(
            title="✅ Default Model Set",
            description=f"{model_type.replace('_', ' ').title()}: `{model_name}`",
            color=discord.Color.green(),
        )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="set_prefix", description="Set the command prefix for this server")
    @app_commands.describe(prefix="New prefix (e.g. !lily, !, ?)")
    @is_admin()
    async def set_prefix(
        self,
        interaction: discord.Interaction,
        prefix: str,
    ):
        """Set the command prefix for this server."""
        if len(prefix) > 10:
            await interaction.response.send_message(
                "❌ Prefix must be 10 characters or less.", ephemeral=True
            )
            return

        db: Database = self.bot.db  # type: ignore
        guild_id = interaction.guild_id

        db.set_guild_setting(guild_id, "prefix", prefix)

        embed = discord.Embed(
            title="✅ Prefix Set",
            description=f"Command prefix is now `{prefix}`",
            color=discord.Color.green(),
        )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="set_safe_mode", description="Set content safety level")
    @app_commands.describe(
        mode="Safety mode: true (privacy,secrets), nsfw (sexual,violence), or off"
    )
    @app_commands.choices(mode=[
        app_commands.Choice(name="Standard (privacy, secrets)", value="privacy,secrets"),
        app_commands.Choice(name="Strict (all filters)", value="privacy,secrets,sexual,violence,shield"),
        app_commands.Choice(name="NSFW (sexual, violence)", value="sexual,violence"),
        app_commands.Choice(name="Off", value="false"),
    ])
    @is_admin()
    async def set_safe_mode(
        self,
        interaction: discord.Interaction,
        mode: str,
    ):
        """Set the content safety mode for this server."""
        db: Database = self.bot.db  # type: ignore
        guild_id = interaction.guild_id

        db.set_guild_setting(guild_id, "safe_mode", mode)

        embed = discord.Embed(
            title="✅ Safe Mode Set",
            description=f"Content safety: `{mode}`",
            color=discord.Color.green(),
        )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="set_reply_chance", description="Set how often Lily responds to messages")
    @app_commands.describe(
        reply_chance="Chance to reply (0.0 - 1.0)",
        reaction_chance="Chance to react instead of reply (0.0 - 1.0)",
        spontaneous_chance="Chance for spontaneous messages (0.0 - 1.0)",
    )
    @is_admin()
    async def set_reply_chance(
        self,
        interaction: discord.Interaction,
        reply_chance: float = None,
        reaction_chance: float = None,
        spontaneous_chance: float = None,
    ):
        """Configure Lily's response behavior for this server."""
        db: Database = self.bot.db  # type: ignore
        guild_id = interaction.guild_id

        changes = []
        if reply_chance is not None:
            if 0 <= reply_chance <= 1:
                db.set_guild_setting(guild_id, "reply_chance", reply_chance)
                changes.append(f"Reply: {reply_chance:.0%}")
            else:
                await interaction.response.send_message(
                    "❌ reply_chance must be between 0.0 and 1.0", ephemeral=True
                )
                return

        if reaction_chance is not None:
            if 0 <= reaction_chance <= 1:
                db.set_guild_setting(guild_id, "reaction_chance", reaction_chance)
                changes.append(f"Reaction: {reaction_chance:.0%}")
            else:
                await interaction.response.send_message(
                    "❌ reaction_chance must be between 0.0 and 1.0", ephemeral=True
                )
                return

        if spontaneous_chance is not None:
            if 0 <= spontaneous_chance <= 1:
                db.set_guild_setting(guild_id, "spontaneous_chance", spontaneous_chance)
                changes.append(f"Spontaneous: {spontaneous_chance:.0%}")
            else:
                await interaction.response.send_message(
                    "❌ spontaneous_chance must be between 0.0 and 1.0", ephemeral=True
                )
                return

        embed = discord.Embed(
            title="✅ Response Settings Updated",
            description="\n".join(changes),
            color=discord.Color.green(),
        )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="server_settings", description="View all server settings")
    @is_admin()
    async def server_settings(self, interaction: discord.Interaction):
        """View all settings for this server."""
        db: Database = self.bot.db  # type: ignore
        guild_id = interaction.guild_id

        settings = db.get_guild_settings(guild_id)

        embed = discord.Embed(
            title=f"⚙️ Server Settings — {interaction.guild.name}",
            color=discord.Color.blue(),
        )

        # Models
        embed.add_field(
            name="🤖 Default Models",
            value=(
                f"Text: `{settings.get('text_model', 'openai')}`\n"
                f"Image: `{settings.get('image_model', 'zimage')}`\n"
                f"Video: `{settings.get('video_model', 'veo')}`\n"
                f"TTS: `{settings.get('tts_model', 'elevenlabs')}`\n"
                f"Transcription: `{settings.get('transcription_model', 'whisper')}`\n"
                f"3D: `{settings.get('model_3d', 'trellis-2-low')}`"
            ),
            inline=True,
        )

        # Behavior
        embed.add_field(
            name="🎭 Behavior",
            value=(
                f"Prefix: `{settings.get('prefix', '!lily')}`\n"
                f"Reply: {settings.get('reply_chance', 0.25):.0%}\n"
                f"Reaction: {settings.get('reaction_chance', 0.40):.0%}\n"
                f"Spontaneous: {settings.get('spontaneous_chance', 0.02):.0%}\n"
                f"Personality: {'On' if settings.get('personality_enabled', 1) else 'Off'}\n"
                f"Language: `{settings.get('language', 'en')}`"
            ),
            inline=True,
        )

        # Safety
        embed.add_field(
            name="🛡️ Safety",
            value=f"Safe Mode: `{settings.get('safe_mode', 'privacy,secrets')}`",
            inline=True,
        )

        # Channel
        channel_id = settings.get("allowed_channel")
        if channel_id:
            channel = interaction.guild.get_channel(int(channel_id))
            embed.add_field(
                name="📍 Active Channel",
                value=channel.mention if channel else f"Unknown ({channel_id})",
                inline=True,
            )
        else:
            embed.add_field(
                name="📍 Active Channel",
                value="All channels",
                inline=True,
            )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="reset_user", description="Reset a user's conversation memory")
    @app_commands.describe(user="User to reset (leave empty for all)")
    @is_admin()
    async def reset_user(
        self,
        interaction: discord.Interaction,
        user: discord.Member = None,
    ):
        """Reset a user's conversation memory and facts."""
        db: Database = self.bot.db  # type: ignore
        guild_id = interaction.guild_id

        if user:
            db.clear_conversations(guild_id, user.id)
            db.clear_facts(guild_id, user.id)
            embed = discord.Embed(
                title="✅ User Reset",
                description=f"Memory and facts cleared for {user.mention}",
                color=discord.Color.green(),
            )
        else:
            db.clear_conversations(guild_id)
            embed = discord.Embed(
                title="✅ Server Memory Reset",
                description="All conversation memory cleared for this server.",
                color=discord.Color.green(),
            )

        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="facts", description="See what Lily knows about a user")
    @app_commands.describe(user="User to check (leave empty for yourself)")
    async def facts(
        self,
        interaction: discord.Interaction,
        user: discord.Member = None,
    ):
        """See what Lily knows about a user."""
        db: Database = self.bot.db  # type: ignore
        guild_id = interaction.guild_id
        target = user or interaction.user

        user_facts = db.get_facts(guild_id, target.id)

        embed = discord.Embed(
            title=f"📝 Facts about {target.display_name}",
            color=discord.Color.pink(),
        )

        if user_facts:
            # Group by category
            categories = {}
            for f in user_facts:
                cat = f.get("category", "general")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(f.get("fact", ""))

            for cat, facts_list in categories.items():
                embed.add_field(
                    name=cat.capitalize(),
                    value="\n".join(facts_list[:5]),
                    inline=False,
                )
        else:
            embed.description = "Lily doesn't know anything about this user yet."

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="topics", description="See recurring topics for a user")
    @app_commands.describe(user="User to check (leave empty for yourself)")
    async def topics(
        self,
        interaction: discord.Interaction,
        user: discord.Member = None,
    ):
        """See recurring conversation topics for a user."""
        db: Database = self.bot.db  # type: ignore
        guild_id = interaction.guild_id
        target = user or interaction.user

        user_topics = db.get_topics(guild_id, target.id)

        embed = discord.Embed(
            title=f"📊 Topics for {target.display_name}",
            color=discord.Color.pink(),
        )

        if user_topics:
            for t in user_topics[:10]:
                embed.add_field(
                    name=t.get("topic", "unknown"),
                    value=f"Mentioned {t.get('mentioned_count', 0)} times",
                    inline=True,
                )
        else:
            embed.description = "No topics tracked yet."

        await interaction.response.send_message(embed=embed, ephemeral=True)


async def setup(bot: commands.Bot):
    await bot.add_cog(AdminCog(bot))
