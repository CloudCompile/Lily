#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Image Generation Cog

Commands: /image, /image_advanced, /image_edit.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands
import io

from database import Database
from pollinations import PollinationsAPI
from config import DEFAULT_IMAGE_MODEL, DEFAULT_SAFE_MODE


class ImageGenCog(commands.Cog, name="Image Generation"):
    """Image generation commands using Pollinations API."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="image", description="Generate an image from a prompt")
    @app_commands.describe(
        prompt="Description of the image to generate",
        model="Image model to use (flux, zimage, gptimage, seedream5, etc.)",
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
        """Generate an image from a text prompt."""
        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore

        use_model = model or db.get_guild_setting(guild_id, "image_model", DEFAULT_IMAGE_MODEL)
        safe = db.get_guild_setting(guild_id, "safe_mode", DEFAULT_SAFE_MODE)

        try:
            image_bytes = await api.image_generate(
                prompt,
                model=use_model,
                width=width,
                height=height,
                safe=safe,
            )

            # Send the image
            file = discord.File(io.BytesIO(image_bytes), filename="lily_image.png")
            embed = discord.Embed(
                title="🖼️ Generated Image",
                description=prompt[:500],
                color=discord.Color.pink(),
            )
            embed.set_image(url="attachment://lily_image.png")
            embed.set_footer(text=f"Model: {use_model} | {width}x{height}")

            await interaction.followup.send(embed=embed, file=file)

            # Log generation
            db.log_generation(guild_id, interaction.user.id, "image", use_model, prompt[:50])

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to generate image: {str(e)[:200]}", ephemeral=True
            )

    @app_commands.command(name="image_advanced", description="Generate image with advanced options")
    @app_commands.describe(
        prompt="Description of the image",
        model="Image model to use",
        width="Width in pixels",
        height="Height in pixels",
        seed="Seed for reproducible results (-1 for random)",
        quality="Image quality (low/medium/high/hd)",
        image_url="Reference image URL for editing",
        transparent="Generate with transparent background (gptimage only)",
    )
    async def image_advanced(
        self,
        interaction: discord.Interaction,
        prompt: str,
        model: str = "zimage",
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        quality: str = "medium",
        image_url: str = None,
        transparent: bool = False,
    ):
        """Generate an image with all available options."""
        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore
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
                image=image_url,
                transparent=transparent,
            )

            file = discord.File(io.BytesIO(image_bytes), filename="lily_image.png")
            embed = discord.Embed(
                title="🖼️ Generated Image",
                description=prompt[:500],
                color=discord.Color.pink(),
            )
            embed.set_image(url="attachment://lily_image.png")
            details = f"Model: {model} | {width}x{height} | Seed: {seed}"
            if quality != "medium":
                details += f" | Quality: {quality}"
            if transparent:
                details += " | Transparent"
            embed.set_footer(text=details)

            await interaction.followup.send(embed=embed, file=file)
            db.log_generation(guild_id, interaction.user.id, "image", model, prompt[:50])

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to generate image: {str(e)[:200]}", ephemeral=True
            )

    @app_commands.command(name="image_edit", description="Edit an image with a prompt (attach an image)")
    @app_commands.describe(
        prompt="What to change about the image",
        model="Image editing model (kontext, gptimage, etc.)",
    )
    async def image_edit(
        self,
        interaction: discord.Interaction,
        prompt: str,
        model: str = "kontext",
    ):
        """Edit an attached image using AI."""
        # Check for attachment
        if not interaction.message or not interaction.message.attachments:
            await interaction.response.send_message(
                "❌ Please attach an image to edit!", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore

        try:
            attachment = interaction.message.attachments[0]
            image_bytes = await attachment.read()

            result = await api.image_edit(
                image_bytes,
                prompt,
                model=model,
                filename=attachment.filename or "image.png",
            )

            # Handle response
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

            db.log_generation(guild_id, interaction.user.id, "image_edit", model, prompt[:50])

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to edit image: {str(e)[:200]}", ephemeral=True
            )


async def setup(bot: commands.Bot):
    await bot.add_cog(ImageGenCog(bot))
