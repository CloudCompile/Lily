#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.5 — Account Cog

User account and preferences commands.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands

from database import Database
from relationships import RelationshipEngine
from quotas import QuotaSystem
from config import ADMIN_IDS


class AccountCog(commands.Cog, name="Account"):
    """User account and preferences commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="my_stats", description="See your Lily stats")
    async def my_stats(self, interaction: discord.Interaction):
        """See your overall stats with Lily."""
        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore
        quotas: QuotaSystem = self.bot.quotas  # type: ignore

        user_id = interaction.user.id
        rel = rel_engine.get_relationship(guild_id, user_id)
        quota_status = quotas.get_status(guild_id, user_id, rel.relationship_tier)
        facts = db.get_facts(guild_id, user_id)
        topics = db.get_topics(guild_id, user_id)
        memories = db.get_memories(guild_id, user_id, limit=100)

        embed = discord.Embed(
            title=f"📊 Your Lily Stats",
            color=discord.Color.pink(),
        )

        # Relationship
        tier_emoji = {
            "rival": "😤", "strained": "😐", "stranger": "🤝",
            "acquaintance": "😊", "friend": "😄", "close_friend": "🥰",
            "bestie": "💕", "soulmate": "💖",
        }
        embed.add_field(
            name="Relationship",
            value=f"{tier_emoji.get(rel.relationship_tier, '✨')} {rel.relationship_tier.replace('_', ' ').title()}",
            inline=True,
        )
        embed.add_field(name="Warmth", value=f"{rel.warmth:.0%}", inline=True)
        embed.add_field(name="Interactions", value=str(rel.total_interactions), inline=True)

        # Memory stats
        memory_types = {}
        for m in memories:
            t = m.get("memory_type", "unknown")
            memory_types[t] = memory_types.get(t, 0) + 1

        embed.add_field(name="Facts Known", value=str(len(facts)), inline=True)
        embed.add_field(name="Topics", value=str(len(topics)), inline=True)
        embed.add_field(name="Total Memories", value=str(len(memories)), inline=True)

        # Quota
        embed.add_field(
            name="Pollen Budget",
            value=f"{quota_status['pollen_used']}/{quota_status['pollen_budget']}",
            inline=True,
        )
        embed.add_field(name="Text Gens", value=f"{quota_status['text_gens']}/{quota_status['text_limit']}", inline=True)
        embed.add_field(name="Image Gens", value=f"{quota_status['image_gens']}/{quota_status['image_limit']}", inline=True)

        embed.set_footer(text="Be nice to Lily and she'll warm up to you! 💕")
        await interaction.response.send_message(embed=embed, ephemeral=True)


async def setup(bot: commands.Bot):
    await bot.add_cog(AccountCog(bot))
