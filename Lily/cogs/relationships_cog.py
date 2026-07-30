#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Relationships Cog

Commands for viewing and managing relationships with Lily.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands

from relationships import RelationshipEngine
from database import Database


class RelationshipsCog(commands.Cog, name="Relationships"):
    """Commands for Lily's relationship system."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.hybrid_command(name="relationship", description="See your relationship with Lily")
    @app_commands.describe(user="Check someone else's relationship")
    async def relationship(
        self,
        ctx: commands.Context,
        user: discord.Member = None,
    ):
        """View relationship details."""
        guild_id = ctx.guild.id if ctx.guild else 0
        target = user or ctx.author
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore
        db: Database = self.bot.db  # type: ignore

        # Load from DB if available, otherwise use in-memory
        rel = rel_engine.get_relationship(guild_id, target.id)

        # Tier colors
        tier_colors = {
            "rival": discord.Color.red(),
            "strained": discord.Color.dark_red(),
            "stranger": discord.Color.light_grey(),
            "acquaintance": discord.Color.green(),
            "friend": discord.Color.blue(),
            "close_friend": discord.Color.purple(),
            "bestie": discord.Color.pink(),
            "soulmate": discord.Color.magenta(),
        }

        embed = discord.Embed(
            title=f"💕 Relationship with {target.display_name}",
            color=tier_colors.get(rel.relationship_tier, discord.Color.pink()),
        )

        tier_emoji = {
            "rival": "😤", "strained": "😐", "stranger": "🤝",
            "acquaintance": "😊", "friend": "😄", "close_friend": "🥰",
            "bestie": "💕", "soulmate": "💖",
        }

        embed.add_field(
            name="Tier",
            value=f"{tier_emoji.get(rel.relationship_tier, '✨')} {rel.relationship_tier.replace('_', ' ').title()}",
            inline=True,
        )
        embed.add_field(name="Warmth", value=f"{rel.warmth:.0%}", inline=True)
        embed.add_field(name="Dislike", value=f"{rel.dislike:.0%}", inline=True)
        embed.add_field(name="Affection", value=f"{rel.affection:.0%}", inline=True)
        embed.add_field(name="Trust", value=f"{rel.trust:.0%}", inline=True)
        embed.add_field(name="Familiarity", value=f"{rel.familiarity:.0%}", inline=True)
        embed.add_field(name="Interactions", value=str(rel.total_interactions), inline=True)
        embed.add_field(name="Positive", value=str(rel.positive_interactions), inline=True)
        embed.add_field(name="Negative", value=str(rel.negative_interactions), inline=True)

        if rel.first_met:
            embed.add_field(name="First Met", value=rel.first_met[:10], inline=True)

        # Private notes (Lily won't share these, but we can hint)
        if rel.private_notes:
            embed.add_field(
                name="Lily's Thoughts",
                value=f"*{len(rel.private_notes)} private thoughts about you*",
                inline=True,
            )

        embed.set_footer(text="Be nice to Lily and she'll warm up to you! 💕")
        await ctx.send(embed=embed)

    @commands.hybrid_command(name="quota", description="Check your generation quota")
    async def quota(self, ctx: commands.Context):
        """View your generation quota and pollen budget."""
        guild_id = ctx.guild.id if ctx.guild else 0
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore
        from quotas import QuotaSystem
        quota_system: QuotaSystem = self.bot.quotas  # type: ignore

        user_id = ctx.author.id
        rel = rel_engine.get_relationship(guild_id, user_id)
        status = quota_system.get_status(guild_id, user_id, rel.relationship_tier)

        embed = discord.Embed(
            title="🌸 Your Generation Quota",
            color=discord.Color.pink(),
        )

        # Pollen bar
        pollen_pct = status['pollen_used'] / max(status['pollen_budget'], 1)
        bar_len = 10
        filled = int(pollen_pct * bar_len)
        bar = "💜" * filled + "🤍" * (bar_len - filled)

        embed.add_field(
            name=f"Pollen Budget",
            value=f"{bar} {status['pollen_used']}/{status['pollen_budget']}",
            inline=False,
        )
        embed.add_field(name="Text Gens", value=f"{status['text_gens']}/{status['text_limit']}", inline=True)
        embed.add_field(name="Image Gens", value=f"{status['image_gens']}/{status['image_limit']}", inline=True)
        embed.add_field(name="Tier", value=status['tier'].replace("_", " ").title(), inline=True)
        embed.set_footer(text="Higher relationship tiers = more pollen! Be nice to Lily 💕")

        if ctx.interaction:
            await ctx.send(embed=embed, ephemeral=True)
        else:
            await ctx.send(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(RelationshipsCog(bot))
