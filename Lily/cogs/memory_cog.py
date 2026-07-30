#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Memory Cog

Commands for viewing and managing memories with Lily.
v9.0: Dream journal commands, cross-server memories.
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

    @commands.hybrid_command(name="memories", description="See what Lily remembers about you")
    @app_commands.describe(
        user="Check someone else's memories (admin only)",
        memory_type="Filter by type: short_term, long_term, episodic, recap, dream"
    )
    async def memories(
        self,
        ctx: commands.Context,
        user: discord.Member = None,
        memory_type: str = None,
    ):
        """View Lily's memories about a user (cross-server)."""
        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore

        # Only admins can view other users' memories
        target = ctx.author
        if user and ctx.author.id in ADMIN_IDS:
            target = user
        elif user and ctx.author.id not in ADMIN_IDS:
            if ctx.interaction:
                await ctx.send("❌ Only admins can view other users' memories.", ephemeral=True)
            else:
                await ctx.send("❌ Only admins can view other users' memories.")
            return

        # Get memories (cross-server)
        memories = db.get_memories(guild_id, target.id, memory_type=memory_type, limit=10, cross_server=True)

        if not memories:
            if ctx.interaction:
                await ctx.send(
                    f"Lily hasn't formed any memories about {target.display_name} yet. Talk to her more! 💕",
                    ephemeral=True,
                )
            else:
                await ctx.send(
                    f"Lily hasn't formed any memories about {target.display_name} yet. Talk to her more! 💕"
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
        if ctx.interaction:
            await ctx.send(embed=embed, ephemeral=True)
        else:
            await ctx.send(embed=embed)

    @commands.hybrid_command(name="recaps", description="See Lily's recent diary entries about you")
    async def recaps(self, ctx: commands.Context):
        """View Lily's daily recaps about you (cross-server)."""
        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore

        user_id = ctx.author.id
        recaps = db.get_daily_recaps(guild_id, user_id, count=5)

        if not recaps:
            if ctx.interaction:
                await ctx.send(
                    "Lily hasn't written any diary entries about you yet. Give her some time! 📖",
                    ephemeral=True,
                )
            else:
                await ctx.send(
                    "Lily hasn't written any diary entries about you yet. Give her some time! 📖"
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
        if ctx.interaction:
            await ctx.send(embed=embed, ephemeral=True)
        else:
            await ctx.send(embed=embed)

    @commands.hybrid_command(name="forget", description="Ask Lily to forget something (she might not)")
    @app_commands.describe(what="What you want Lily to forget")
    async def forget(self, ctx: commands.Context, what: str):
        """Ask Lily to forget something. She might not, though."""
        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore

        # She might not actually forget — it depends on trust
        user_id = ctx.author.id
        rel = rel_engine.get_relationship(guild_id, user_id)

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
                guild_id, user_id,
                f"[Lily was asked to forget: {what}]",
                memory_type="short_term", emotion="neutral",
                importance=0.1, tags=["forgotten"],
                is_global=True
            )

        await ctx.send(response)

    @commands.hybrid_command(name="remember", description="Tell Lily to remember something important")
    @app_commands.describe(what="What you want Lily to remember")
    async def remember(self, ctx: commands.Context, what: str):
        """Tell Lily to remember something important (cross-server)."""
        guild_id = ctx.guild.id if ctx.guild else 0
        db: Database = self.bot.db  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore

        user_id = ctx.author.id

        # Save it as a long-term memory (cross-server)
        db.save_memory(
            guild_id, user_id, what,
            memory_type="long_term", emotion="important",
            importance=0.9, tags=["user_requested", "important"],
            is_global=True
        )

        # Also save as a fact
        db.add_fact(guild_id, user_id, "important", what, confidence=0.9)

        # Update relationship
        rel_engine.record_action(guild_id, user_id, "remembered_detail")

        # Generate a personalized response
        responses = [
            f"got it! i'll remember that 💕",
            f"noted!! that's important to you so it's important to me",
            f"i'll remember that, don't worry ✨",
            f"saved!! i won't forget about that 📌",
            f"aww, i'll keep that in mind 💕",
            f"remembered! i carry this across all servers yknow ✨",
        ]

        await ctx.send(random.choice(responses))

    @commands.hybrid_command(name="dream", description="Lily shares one of her dreams with you")
    async def dream(self, ctx: commands.Context):
        """Lily shares a dream from her dream journal."""
        db: Database = self.bot.db  # type: ignore

        dreams = db.get_dreams(count=5)
        if not dreams:
            if ctx.interaction:
                await ctx.send(
                    "Lily hasn't had any dreams yet... she'll write some tonight! 🌙",
                    ephemeral=True,
                )
            else:
                await ctx.send("Lily hasn't had any dreams yet... she'll write some tonight! 🌙")
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
        await ctx.send(embed=embed)

    @commands.hybrid_command(name="dream_journal", description="See Lily's dream journal")
    async def dream_journal(self, ctx: commands.Context):
        """View Lily's dream journal entries."""
        db: Database = self.bot.db  # type: ignore

        dreams = db.get_dreams(count=5)
        if not dreams:
            if ctx.interaction:
                await ctx.send(
                    "Lily hasn't had any dreams yet... she'll write some tonight! 🌙",
                    ephemeral=True,
                )
            else:
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
        if ctx.interaction:
            await ctx.send(embed=embed, ephemeral=True)
        else:
            await ctx.send(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(MemoryCog(bot))
