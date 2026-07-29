#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.5 — Memory Cog

Commands for viewing and managing memories with Lily.
v8.5: Dream journal commands, cross-server memories.
"""

from __future__ import annotations
import random
import discord
from discord.ext import commands
from discord import app_commands

from database import Database
from memories import MemorySystem
from relationships import RelationshipEngine
from config import ADMIN_IDS


class MemoryCog(commands.Cog, name="Memory"):
    """Commands for Lily's memory system."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="memories", description="See what Lily remembers about you")
    @app_commands.describe(
        user="Check someone else's memories (admin only)",
        memory_type="Filter by type: short_term, long_term, episodic, recap, dream"
    )
    async def memories(
        self,
        interaction: discord.Interaction,
        user: discord.Member = None,
        memory_type: str = None,
    ):
        """View Lily's memories about a user (cross-server)."""
        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore

        # Only admins can view other users' memories
        target = interaction.user
        if user and interaction.user.id in ADMIN_IDS:
            target = user
        elif user and interaction.user.id not in ADMIN_IDS:
            await interaction.response.send_message("❌ Only admins can view other users' memories.", ephemeral=True)
            return

        # Get memories (cross-server)
        memories = db.get_memories(guild_id, target.id, memory_type=memory_type, limit=10, cross_server=True)

        if not memories:
            await interaction.response.send_message(
                f"Lily hasn't formed any memories about {target.display_name} yet. Talk to her more! 💕",
                ephemeral=True,
            )
            return

        # Type emoji map
        type_emoji = {
            "short_term": "💭",
            "long_term": "📌",
            "episodic": "✨",
            "recap": "📖",
            "auto": "🧠",
            "dream": "🌙",
        }

        embed = discord.Embed(
            title=f"🧠 Lily's Memories — {target.display_name}",
            description="✨ Cross-server: Lily remembers you everywhere!",
            color=discord.Color.pink(),
        )

        for m in memories[:10]:
            emoji = type_emoji.get(m.get("memory_type", ""), "💭")
            importance = m.get("importance", 0.5)
            stars = "⭐" * max(1, int(importance * 3))
            content = m.get("content", "")[:100]
            if len(m.get("content", "")) > 100:
                content += "..."
            guild_indicator = ""
            if m.get("guild_id", "0") != str(guild_id) and m.get("guild_id", "0") != "0":
                guild_indicator = " 🌐"
            embed.add_field(
                name=f"{emoji} {m.get('memory_type', 'unknown').replace('_', ' ').title()} {stars}{guild_indicator}",
                value=content,
                inline=False,
            )

        embed.set_footer(text="Lily remembers what matters to her 💕 | Cross-server ✨")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="recaps", description="See Lily's recent diary entries about you")
    async def recaps(self, interaction: discord.Interaction):
        """View Lily's daily recaps about you (cross-server)."""
        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore

        recaps = db.get_daily_recaps(guild_id, interaction.user.id, count=5)

        if not recaps:
            await interaction.response.send_message(
                "Lily hasn't written any diary entries about you yet. Give her some time! 📖",
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title="📖 Lily's Diary — Recent Entries",
            description="These are Lily's private thoughts about her conversations with you. ✨",
            color=discord.Color.pink(),
        )

        for recap in recaps:
            date = recap.get("recap_date", "unknown")
            text = recap.get("recap_text", "")[:200]
            if len(recap.get("recap_text", "")) > 200:
                text += "..."
            embed.add_field(name=f"📅 {date}", value=text, inline=False)

        embed.set_footer(text="Lily writes a diary entry every night 🌙")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="forget", description="Ask Lily to forget something (she might not)")
    @app_commands.describe(what="What you want Lily to forget")
    async def forget(self, interaction: discord.Interaction, what: str):
        """Ask Lily to forget something. She might not, though."""
        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore

        # She might not actually forget — it depends on trust
        rel = rel_engine.get_relationship(guild_id, interaction.user.id)

        if rel.trust < 0.3:
            # Low trust = she might "forget" but not really
            response = f"hmm, i'll try to forget about {what}... but no promises 👀"
        elif rel.trust < 0.6:
            # Medium trust = she'll try
            response = f"okay, i'll try to forget about {what}. no guarantees tho lol"
        else:
            # High trust = she'll actually try
            response = f"of course, i'll forget about {what}. don't worry about it 💕"
            # Actually mark it as forgotten in the memory system
            db.save_memory(
                guild_id, interaction.user.id,
                f"[Lily was asked to forget: {what}]",
                memory_type="short_term", emotion="neutral",
                importance=0.1, tags=["forgotten"],
                is_global=True
            )

        await interaction.response.send_message(response)

    @app_commands.command(name="remember", description="Tell Lily to remember something important")
    @app_commands.describe(what="What you want Lily to remember")
    async def remember(self, interaction: discord.Interaction, what: str):
        """Tell Lily to remember something important (cross-server)."""
        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore

        # Save it as a long-term memory (cross-server)
        db.save_memory(
            guild_id, interaction.user.id, what,
            memory_type="long_term", emotion="important",
            importance=0.9, tags=["user_requested", "important"],
            is_global=True
        )

        # Also save as a fact
        db.add_fact(guild_id, interaction.user.id, "important", what, confidence=0.9)

        # Update relationship
        rel_engine.record_action(guild_id, interaction.user.id, "remembered_detail")

        # Generate a personalized response
        responses = [
            f"got it! i'll remember that 💕",
            f"noted!! that's important to you so it's important to me",
            f"i'll remember that, don't worry ✨",
            f"saved!! i won't forget about that 📌",
            f"aww, i'll keep that in mind 💕",
            f"remembered! i carry this across all servers yknow ✨",
        ]

        await interaction.response.send_message(random.choice(responses))

    @app_commands.command(name="dream", description="Lily shares one of her dreams with you")
    async def dream(self, interaction: discord.Interaction):
        """Lily shares a dream from her dream journal."""
        db: Database = self.bot.db  # type: ignore

        dreams = db.get_dreams(count=5)
        if not dreams:
            await interaction.response.send_message(
                "Lily hasn't had any dreams yet... she'll write some tonight! 🌙",
                ephemeral=True,
            )
            return

        dream = random.choice(dreams)
        embed = discord.Embed(
            title="🌙 Lily's Dream",
            description=dream.get("dream_text", "")[:2000],
            color=discord.Color.purple(),
        )
        mood_emoji = {
            "dreamy": "✨", "sleepy": "😴", "cozy": "🌙",
            "energetic": "⚡", "chill": "☕", "morning": "🌅",
        }
        mood = dream.get("mood", "dreamy")
        embed.set_footer(text=f"Dream mood: {mood_emoji.get(mood, '✨')} {mood} | Dream Journal ✨")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="dream_journal", description="See Lily's dream journal")
    async def dream_journal(self, interaction: discord.Interaction):
        """View Lily's dream journal entries."""
        db: Database = self.bot.db  # type: ignore

        dreams = db.get_dreams(count=5)
        if not dreams:
            await interaction.response.send_message(
                "Lily hasn't had any dreams yet... she'll write some tonight! 🌙",
                ephemeral=True,
            )
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
        await interaction.response.send_message(embed=embed, ephemeral=True)


async def setup(bot: commands.Bot):
    await bot.add_cog(MemoryCog(bot))
