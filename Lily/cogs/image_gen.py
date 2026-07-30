#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Image Generation Cog

Commands: /image, /image_advanced, /image_edit.
v9.0: Sana Sprint by default (0.0001/gen!), actual cost tracking.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands
import io

from database import Database
from pollinations import PollinationsAPI
from relationships import RelationshipEngine
from quotas import QuotaSystem
from model_router import ModelRouter
from config import DEFAULT_IMAGE_MODEL, DEFAULT_SAFE_MODE


class ImageGenCog(commands.Cog, name="Image Generation"):
    """Image generation commands using Pollinations API."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="image", description="Generate an image from a prompt")
    @app_commands.describe(
        prompt="Description of the image to generate",
        model="Image model (sana=cheap, flux=quality, gptimage=pro)",
        width="Width in pixels (default: 1024)",
        height="Height in pixels (default: 1024)",
    )
    async def image(
        self,
        interaction: discord.Interaction,
        prompt: str,
        model: str = None,
        width: int = 1024,
        height: int = 1024,
    ):
        """Generate an image from a text prompt. Sana Sprint by default!"""
        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore
        quotas: QuotaSystem = self.bot.quotas  # type: ignore

        user_id = interaction.user.id
        rel = rel_engine.get_relationship(guild_id, user_id)

        # Determine gen type based on model
        use_model = model or db.get_guild_setting(guild_id, "image_model", DEFAULT_IMAGE_MODEL)
        if use_model == "sana":
            gen_type = "image_quick"     # 0.0001 pollen
        elif use_model in ("flux",):
            gen_type = "image_standard"  # 0.003 pollen
        else:
            gen_type = "image_pro"       # 0.01 pollen

        # Check quota
        can_gen, reason = quotas.can_generate(guild_id, user_id, gen_type, rel.relationship_tier)
        if not can_gen:
            await interaction.followup.send(f"❌ {reason}", ephemeral=True)
            return

        safe = db.get_guild_setting(guild_id, "safe_mode", DEFAULT_SAFE_MODE)

        # Record action
        rel_engine.record_action(guild_id, user_id, "used_command")

        try:
            image_bytes = await api.image_generate(
                prompt,
                model=use_model,
                width=width,
                height=height,
                safe=safe,
            )

            file = discord.File(io.BytesIO(image_bytes), filename="lily_image.png")
            embed = discord.Embed(
                title="🖼️ Generated Image",
                description=prompt[:500],
                color=discord.Color.pink(),
            )
            embed.set_image(url="attachment://lily_image.png")

            # Show actual cost
            actual_cost = self.bot.model_router.estimate_image_cost(use_model)
            embed.set_footer(text=f"Model: {use_model} | {width}x{height} | Cost: {actual_cost:.4f} pollen")

            await interaction.followup.send(embed=embed, file=file)

            # Log and quota with actual cost
            quotas.record_generation(guild_id, user_id, gen_type, rel.relationship_tier, actual_cost=actual_cost)
            db.log_generation(guild_id, user_id, "image", use_model, prompt[:50], cost_pollen=actual_cost)

            # Save as a memory (cross-server)
            db.save_memory(
                guild_id, user_id, f"Generated image: {prompt[:100]}",
                memory_type="episodic", emotion="creative",
                importance=0.4, tags=["image_generation"],
                is_global=True
            )

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to generate image: {str(e)[:200]}", ephemeral=True
            )

    @app_commands.command(name="image_advanced", description="Generate image with advanced options")
    @app_commands.describe(
        prompt="Description of the image",
        model="Image model to use (sana, flux, gptimage, kontext)",
        width="Width in pixels",
        height="Height in pixels",
        seed="Seed for reproducible results (-1 for random)",
        quality="Image quality (low/medium/high/hd)",
        enhance="Enhance the prompt with AI",
    )
    async def image_advanced(
        self,
        interaction: discord.Interaction,
        prompt: str,
        model: str = "sana",
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        quality: str = "medium",
        enhance: bool = False,
    ):
        """Generate an image with all available options."""
        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore
        quotas: QuotaSystem = self.bot.quotas  # type: ignore

        user_id = interaction.user.id
        rel = rel_engine.get_relationship(guild_id, user_id)

        # Determine gen type based on model and quality
        if model == "sana":
            gen_type = "image_quick"
        elif quality in ("high", "hd") or model in ("gptimage",):
            gen_type = "image_pro"
        else:
            gen_type = "image_standard"

        can_gen, reason = quotas.can_generate(guild_id, user_id, gen_type, rel.relationship_tier)
        if not can_gen:
            await interaction.followup.send(f"❌ {reason}", ephemeral=True)
            return

        safe = db.get_guild_setting(guild_id, "safe_mode", DEFAULT_SAFE_MODE)

        try:
            image_bytes = await api.image_generate(
                prompt,
                model=model,
                width=width,
                height=height,
                seed=seed,
                safe=safe,
                quality=quality,
                enhance=enhance,
            )

            file = discord.File(io.BytesIO(image_bytes), filename="lily_image.png")
            embed = discord.Embed(
                title="🖼️ Generated Image",
                description=prompt[:500],
                color=discord.Color.pink(),
            )
            embed.set_image(url="attachment://lily_image.png")
            actual_cost = self.bot.model_router.estimate_image_cost(model)
            details = f"Model: {model} | {width}x{height} | Seed: {seed} | Cost: {actual_cost:.4f} pollen"
            if quality != "medium":
                details += f" | Quality: {quality}"
            if enhance:
                details += " | Enhanced"
            embed.set_footer(text=details)

            await interaction.followup.send(embed=embed, file=file)

            quotas.record_generation(guild_id, user_id, gen_type, rel.relationship_tier, actual_cost=actual_cost)
            db.log_generation(guild_id, user_id, "image", model, prompt[:50], cost_pollen=actual_cost)

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to generate image: {str(e)[:200]}", ephemeral=True
            )

    @app_commands.command(name="image_edit", description="Edit an image with a prompt (attach an image)")
    @app_commands.describe(
        prompt="What to change about the image",
        model="Image editing model (kontext, gptimage)",
    )
    async def image_edit(
        self,
        interaction: discord.Interaction,
        prompt: str,
        model: str = "kontext",
    ):
        """Edit an attached image using AI."""
        if not interaction.message or not interaction.message.attachments:
            await interaction.response.send_message(
                "❌ Please attach an image to edit!", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore
        quotas: QuotaSystem = self.bot.quotas  # type: ignore

        user_id = interaction.user.id
        rel = rel_engine.get_relationship(guild_id, user_id)

        can_gen, reason = quotas.can_generate(guild_id, user_id, "image_edit", rel.relationship_tier)
        if not can_gen:
            await interaction.followup.send(f"❌ {reason}", ephemeral=True)
            return

        try:
            attachment = interaction.message.attachments[0]
            image_bytes = await attachment.read()

            result = await api.image_edit(
                image_bytes,
                prompt,
                model=model,
                filename=attachment.filename or "image.png",
            )

            if "data" in result and result["data"]:
                data = result["data"][0]
                if "b64_json" in data:
                    import base64
                    img_data = base64.b64decode(data["b64_json"])
                    file = discord.File(io.BytesIO(img_data), filename="lily_edited.png")
                    embed = discord.Embed(
                        title="🖼️ Edited Image",
                        description=prompt[:500],
                        color=discord.Color.pink(),
                    )
                    embed.set_image(url="attachment://lily_edited.png")
                    embed.set_footer(text=f"Model: {model}")
                    await interaction.followup.send(embed=embed, file=file)
                elif "url" in data:
                    embed = discord.Embed(
                        title="🖼️ Edited Image",
                        description=prompt[:500],
                        color=discord.Color.pink(),
                    )
                    embed.set_image(url=data["url"])
                    embed.set_footer(text=f"Model: {model}")
                    await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send("❌ No image was returned from the API.", ephemeral=True)

            actual_cost = self.bot.model_router.estimate_image_cost(model)
            quotas.record_generation(guild_id, user_id, "image_edit", rel.relationship_tier, actual_cost=actual_cost)
            db.log_generation(guild_id, user_id, "image_edit", model, prompt[:50], cost_pollen=actual_cost)

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to edit image: {str(e)[:200]}", ephemeral=True
            )


async def setup(bot: commands.Bot):
    await bot.add_cog(ImageGenCog(bot))
