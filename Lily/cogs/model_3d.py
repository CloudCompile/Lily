#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — 3D Model Generation Cog

Command: /3d
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands
import io

from database import Database
from pollinations import PollinationsAPI
from config import DEFAULT_3D_MODEL, DEFAULT_SAFE_MODE


class Model3DCog(commands.Cog, name="3D Generation"):
    """3D model generation commands using Pollinations API."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="3d", description="Generate a 3D model from a prompt")
    @app_commands.describe(
        prompt="Description of the 3D model to generate",
        model="3D model to use (trellis-2-low, trellis-2-medium, trellis-2-high, hyper3d-rodin)",
        image_url="Reference image URL for image-to-3D",
        seed="Seed for varied generations",
    )
    async def generate_3d(
        self,
        interaction: discord.Interaction,
        prompt: str,
        model: str = None,
        image_url: str = None,
        seed: int = None,
    ):
        """Generate a 3D model (GLB format) from a text prompt."""
        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore

        use_model = model or db.get_guild_setting(guild_id, "model_3d", DEFAULT_3D_MODEL)
        safe = db.get_guild_setting(guild_id, "safe_mode", DEFAULT_SAFE_MODE)

        await interaction.followup.send(
            f"🧊 Generating 3D model with `{use_model}`... This may take a few minutes.",
        )

        try:
            kwargs = {}
            if image_url:
                kwargs["image"] = image_url
            if seed is not None:
                kwargs["seed"] = seed

            model_bytes = await api.generate_3d(
                prompt,
                model=use_model,
                safe=safe,
                **kwargs,
            )

            # Check file size
            max_size = 25 * 1024 * 1024
            if len(model_bytes) > max_size:
                await interaction.channel.send(
                    f"⚠️ 3D model too large ({len(model_bytes) / 1024 / 1024:.1f}MB). "
                    f"Try a lower quality model."
                )
                return

            file = discord.File(io.BytesIO(model_bytes), filename="lily_3d.glb")
            embed = discord.Embed(
                title="🧊 Generated 3D Model",
                description=prompt[:500],
                color=discord.Color.teal(),
            )
            details = f"Model: {use_model}"
            if seed is not None:
                details += f" | Seed: {seed}"
            if image_url:
                details += " | Image-to-3D"
            embed.set_footer(text=details)

            await interaction.channel.send(embed=embed, file=file)
            db.log_generation(guild_id, interaction.user.id, "3d", use_model, prompt[:50])

        except Exception as e:
            await interaction.channel.send(
                f"❌ Failed to generate 3D model: {str(e)[:200]}"
            )


async def setup(bot: commands.Bot):
    await bot.add_cog(Model3DCog(bot))
