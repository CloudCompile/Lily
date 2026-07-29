#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Models Cog

Commands: /models, /model_info, /model_status.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands

from pollinations import PollinationsAPI
from database import Database


class ModelsCog(commands.Cog, name="Models"):
    """Model listing and info commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="models", description="List available AI models")
    @app_commands.describe(
        category="Model category to list",
    )
    @app_commands.choices(category=[
        app_commands.Choice(name="All Models", value="all"),
        app_commands.Choice(name="Text Models", value="text"),
        app_commands.Choice(name="Image Models", value="image"),
        app_commands.Choice(name="Video Models", value="video"),
        app_commands.Choice(name="Audio Models", value="audio"),
        app_commands.Choice(name="3D Models", value="3d"),
        app_commands.Choice(name="Embedding Models", value="embeddings"),
    ])
    async def models(
        self,
        interaction: discord.Interaction,
        category: str = "all",
    ):
        """List available AI models by category."""
        await interaction.response.defer(thinking=True)

        api: PollinationsAPI = self.bot.api  # type: ignore

        try:
            if category == "all":
                data = await api.list_models()
                model_list = data if isinstance(data, list) else []
            elif category == "text":
                model_list = await api.list_text_models()
            elif category == "image":
                model_list = await api.list_image_models()
            elif category == "video":
                model_list = await api.list_video_models()
            elif category == "audio":
                model_list = await api.list_audio_models()
            elif category == "3d":
                model_list = await api.list_3d_models()
            elif category == "embeddings":
                model_list = await api.list_embedding_models()
            else:
                model_list = await api.list_models()
                if isinstance(model_list, dict):
                    model_list = model_list.get("data", [])

            if not model_list:
                await interaction.followup.send("No models found for this category.", ephemeral=True)
                return

            # Build paginated embed
            embed = discord.Embed(
                title=f"📋 {category.capitalize()} Models",
                color=discord.Color.gold(),
            )

            # Show first 25 models (Discord field limit)
            for m in model_list[:25]:
                name = m.get("name", m.get("id", "unknown"))
                title = m.get("title", name)
                cat = m.get("category", "")
                description = m.get("description", "")[:80] if m.get("description") else ""

                value = f"`{name}`"
                if cat:
                    value += f" | {cat}"
                if description:
                    value += f"\n{description}"

                embed.add_field(name=title[:256], value=value[:1024], inline=False)

            total = len(model_list)
            if total > 25:
                embed.set_footer(text=f"Showing 25 of {total} models. Use /model_info for details.")

            await interaction.followup.send(embed=embed)

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to list models: {str(e)[:200]}", ephemeral=True
            )

    @app_commands.command(name="model_info", description="Get detailed info about a specific model")
    @app_commands.describe(model="Model ID to look up")
    async def model_info(
        self,
        interaction: discord.Interaction,
        model: str,
    ):
        """Get detailed information about a specific model."""
        await interaction.response.defer(thinking=True)

        api: PollinationsAPI = self.bot.api  # type: ignore

        try:
            all_models = await api.list_models()
            if isinstance(all_models, dict):
                all_models = all_models.get("data", [])

            # Find the model
            found = None
            for m in all_models:
                if m.get("name") == model or m.get("id") == model:
                    found = m
                    break
                # Check aliases
                if model in m.get("aliases", []):
                    found = m
                    break

            if not found:
                await interaction.followup.send(
                    f"❌ Model `{model}` not found. Use `/models` to see available models.",
                    ephemeral=True,
                )
                return

            embed = discord.Embed(
                title=f"📋 {found.get('title', model)}",
                description=found.get("description", "No description available."),
                color=discord.Color.gold(),
            )

            embed.add_field(name="ID", value=f"`{found.get('name', model)}`", inline=True)
            embed.add_field(name="Category", value=found.get("category", "unknown"), inline=True)
            embed.add_field(name="Brand", value=found.get("brand", "unknown"), inline=True)

            # Input/output modalities
            input_mods = ", ".join(found.get("input_modalities", []))
            output_mods = ", ".join(found.get("output_modalities", []))
            if input_mods:
                embed.add_field(name="Input", value=input_mods, inline=True)
            if output_mods:
                embed.add_field(name="Output", value=output_mods, inline=True)

            # Capabilities
            context = found.get("context_length")
            if context:
                embed.add_field(name="Context", value=f"{context:,} tokens", inline=True)

            caps = []
            if found.get("tools"):
                caps.append("Tool Calling")
            if found.get("reasoning"):
                caps.append("Reasoning")
            if caps:
                embed.add_field(name="Capabilities", value=", ".join(caps), inline=True)

            # Aliases
            aliases = found.get("aliases", [])
            if aliases:
                embed.add_field(name="Aliases", value=", ".join(f"`{a}`" for a in aliases[:10]), inline=False)

            # Pricing
            pricing = found.get("pricing", {})
            if pricing:
                price_text = []
                if pricing.get("promptTextTokens"):
                    price_text.append(f"Input: {pricing['promptTextTokens']}/token")
                if pricing.get("completionTextTokens"):
                    price_text.append(f"Output: {pricing['completionTextTokens']}/token")
                if price_text:
                    embed.add_field(name="Pricing", value="\n".join(price_text), inline=False)

            await interaction.followup.send(embed=embed)

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to get model info: {str(e)[:200]}", ephemeral=True
            )

    @app_commands.command(name="model_status", description="Check model health status")
    async def model_status(self, interaction: discord.Interaction):
        """Check the health status of Pollinations models."""
        await interaction.response.defer(thinking=True)

        api: PollinationsAPI = self.bot.api  # type: ignore

        try:
            status = await api.model_health_status()

            if not status:
                await interaction.followup.send("No health data available.")
                return

            embed = discord.Embed(
                title="📊 Model Health Status",
                color=discord.Color.green(),
            )

            # Show first 20 models
            healthy = 0
            degraded = 0
            for entry in status[:20]:
                model_name = entry.get("model", "unknown")
                success_rate = entry.get("success_rate", 0)
                avg_latency = entry.get("avg_latency_ms", 0)

                if success_rate > 0.9:
                    status_emoji = "🟢"
                    healthy += 1
                elif success_rate > 0.5:
                    status_emoji = "🟡"
                    degraded += 1
                else:
                    status_emoji = "🔴"
                    degraded += 1

                embed.add_field(
                    name=f"{status_emoji} {model_name}",
                    value=f"Success: {success_rate:.0%} | Latency: {avg_latency:.0f}ms",
                    inline=True,
                )

            embed.set_footer(text=f"🟢 {healthy} healthy | 🟡🔴 {degraded} issues")
            await interaction.followup.send(embed=embed)

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to check model status: {str(e)[:200]}", ephemeral=True
            )


async def setup(bot: commands.Bot):
    await bot.add_cog(ModelsCog(bot))
