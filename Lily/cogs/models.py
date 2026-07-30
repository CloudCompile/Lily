#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Models Cog

Commands for browsing and managing AI models.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands

from database import Database
from pollinations import PollinationsAPI
from model_router import ModelRouter
from config import ADMIN_IDS


class ModelsCog(commands.Cog, name="Models"):
    """Commands for browsing and managing AI models."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.hybrid_command(name="models", description="List available AI models")
    @app_commands.describe(
        category="Model category: text, image, embedding",
        tier="Cost tier: budget, standard, premium"
    )
    async def models(
        self,
        ctx: commands.Context,
        category: str = "text",
        tier: str = None,
    ):
        """List available models from Pollinations API."""
        router: ModelRouter = self.bot.model_router  # type: ignore

        if not router._models:
            if ctx.interaction:
                await ctx.send("❌ Models not loaded yet. Try again in a moment.", ephemeral=True)
            else:
                await ctx.send("❌ Models not loaded yet. Try again in a moment.")
            return

        models = router.list_models(category=category, cost_tier=tier)

        if not models:
            if ctx.interaction:
                await ctx.send(
                    f"No models found for category `{category}`"
                    + (f" tier `{tier}`" if tier else ""),
                    ephemeral=True,
                )
            else:
                await ctx.send(
                    f"No models found for category `{category}`"
                    + (f" tier `{tier}`" if tier else "")
                )
            return

        embed = discord.Embed(
            title=f"📋 {category.title()} Models"
            + (f" — {tier.title()} Tier" if tier else ""),
            color=discord.Color.blue(),
        )

        for m in models[:20]:
            vision = " 👁️" if m.has_vision else ""
            reasoning = " 🧠" if m.has_reasoning else ""
            tools = " 🔧" if m.has_tools else ""
            cost = f"{m.completion_price:.8f}"
            embed.add_field(
                name=f"`{m.name}`{vision}{reasoning}{tools}",
                value=f"{m.title} — {m.description[:50]}\nCost: {cost} pollen/token | Context: {m.context_length:,}",
                inline=False,
            )

        if len(models) > 20:
            embed.set_footer(text=f"Showing 20 of {len(models)} models")

        if ctx.interaction:
            await ctx.send(embed=embed, ephemeral=True)
        else:
            await ctx.send(embed=embed)

    @commands.hybrid_command(name="model_info", description="Get details about a specific model")
    @app_commands.describe(model="Model name to look up")
    async def model_info(self, ctx: commands.Context, model: str):
        """Get detailed info about a specific model."""
        router: ModelRouter = self.bot.model_router  # type: ignore
        info = router.get_model_info(model)

        if not info:
            if ctx.interaction:
                await ctx.send(
                    f"❌ Model `{model}` not found. Use `/models` to see available models.",
                    ephemeral=True,
                )
            else:
                await ctx.send(
                    f"❌ Model `{model}` not found. Use `/models` to see available models."
                )
            return

        embed = discord.Embed(
            title=f"📋 {info.title}",
            description=info.description,
            color=discord.Color.blue(),
        )
        embed.add_field(name="ID", value=f"`{info.name}`", inline=True)
        embed.add_field(name="Brand", value=info.brand, inline=True)
        embed.add_field(name="Cost Tier", value=info.cost_tier.title(), inline=True)
        embed.add_field(name="Prompt Cost", value=f"{info.prompt_price:.8f} pollen/token", inline=True)
        embed.add_field(name="Completion Cost", value=f"{info.completion_price:.8f} pollen/token", inline=True)
        embed.add_field(name="Context Length", value=f"{info.context_length:,}", inline=True)
        embed.add_field(name="Vision", value="✅" if info.has_vision else "❌", inline=True)
        embed.add_field(name="Reasoning", value="✅" if info.has_reasoning else "❌", inline=True)
        embed.add_field(name="Tools", value="✅" if info.has_tools else "❌", inline=True)

        if ctx.interaction:
            await ctx.send(embed=embed, ephemeral=True)
        else:
            await ctx.send(embed=embed)

    @commands.hybrid_command(name="balance", description="Check API balance (admin only)")
    async def balance(self, ctx: commands.Context):
        """Check Pollinations API balance."""
        if ctx.author.id not in ADMIN_IDS:
            if ctx.interaction:
                await ctx.send("❌ Admin only.", ephemeral=True)
            else:
                await ctx.send("❌ Admin only.")
            return

        api: PollinationsAPI = self.bot.api  # type: ignore
        try:
            result = await api.get_balance()
            embed = discord.Embed(
                title="💰 API Balance",
                color=discord.Color.green(),
            )
            embed.add_field(name="Balance", value=str(result.get("balance", result.get("total", "N/A"))), inline=True)
            embed.add_field(name="Currency", value=result.get("currency", "pollen"), inline=True)
            if ctx.interaction:
                await ctx.send(embed=embed, ephemeral=True)
            else:
                await ctx.send(embed=embed)
        except Exception as e:
            if ctx.interaction:
                await ctx.send(f"❌ Failed to check balance: {str(e)[:200]}", ephemeral=True)
            else:
                await ctx.send(f"❌ Failed to check balance: {str(e)[:200]}")


async def setup(bot: commands.Bot):
    await bot.add_cog(ModelsCog(bot))
