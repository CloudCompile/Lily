#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Video Generation Cog

Commands: /video, /video_advanced.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands
import io

from database import Database
from pollinations import PollinationsAPI
from config import DEFAULT_VIDEO_MODEL, DEFAULT_SAFE_MODE


class VideoGenCog(commands.Cog, name="Video Generation"):
    """Video generation commands using Pollinations API."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="video", description="Generate a video from a prompt")
    @app_commands.describe(
        prompt="Description of the video to generate",
        model="Video model to use (veo, seedance, wan, etc.)",
        duration="Duration in seconds",
    )
    async def video(
        self,
        interaction: discord.Interaction,
        prompt: str,
        model: str = None,
        duration: int = None,
    ):
        """Generate a video from a text prompt."""
        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore

        use_model = model or db.get_guild_setting(guild_id, "video_model", DEFAULT_VIDEO_MODEL)
        safe = db.get_guild_setting(guild_id, "safe_mode", DEFAULT_SAFE_MODE)

        # Video generation can take a while
        await interaction.followup.send(
            f"🎬 Generating video with `{use_model}`... This may take a few minutes.",
        )

        try:
            kwargs = {}
            if duration:
                kwargs["duration"] = duration

            video_bytes = await api.video_generate(
                prompt,
                model=use_model,
                safe=safe,
                **kwargs,
            )

            # Check Discord file size limit (25MB for most servers)
            max_size = 25 * 1024 * 1024
            if len(video_bytes) > max_size:
                await interaction.channel.send(
                    f"⚠️ Video is too large for Discord ({len(video_bytes) / 1024 / 1024:.1f}MB). "
                    f"Try a shorter duration or different model."
                )
                return

            file = discord.File(io.BytesIO(video_bytes), filename="lily_video.mp4")
            embed = discord.Embed(
                title="🎬 Generated Video",
                description=prompt[:500],
                color=discord.Color.purple(),
            )
            embed.set_footer(text=f"Model: {use_model}")

            await interaction.channel.send(embed=embed, file=file)
            db.log_generation(guild_id, interaction.user.id, "video", use_model, prompt[:50])

        except Exception as e:
            await interaction.channel.send(
                f"❌ Failed to generate video: {str(e)[:200]}"
            )

    @app_commands.command(name="video_advanced", description="Generate video with full options")
    @app_commands.describe(
        prompt="Description of the video",
        model="Video model to use",
        width="Width in pixels",
        height="Height in pixels",
        duration="Duration in seconds",
        seed="Seed for reproducible results",
        aspect_ratio="Aspect ratio (16:9 or 9:16)",
        audio="Include audio in the video",
        image_url="Reference image URL (start frame)",
    )
    async def video_advanced(
        self,
        interaction: discord.Interaction,
        prompt: str,
        model: str = "veo",
        width: int = 1024,
        height: int = 1024,
        duration: int = None,
        seed: int = 0,
        aspect_ratio: str = None,
        audio: bool = False,
        image_url: str = None,
    ):
        """Generate a video with all available options."""
        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore
        safe = db.get_guild_setting(guild_id, "safe_mode", DEFAULT_SAFE_MODE)

        await interaction.followup.send(
            f"🎬 Generating video with `{model}`... This may take a few minutes.",
        )

        try:
            kwargs = {}
            if duration:
                kwargs["duration"] = duration
            if aspect_ratio:
                kwargs["aspect_ratio"] = aspect_ratio
            if image_url:
                kwargs["image"] = image_url

            video_bytes = await api.video_generate(
                prompt,
                model=model,
                width=width,
                height=height,
                seed=seed,
                safe=safe,
                audio=audio,
                **kwargs,
            )

            max_size = 25 * 1024 * 1024
            if len(video_bytes) > max_size:
                await interaction.channel.send(
                    f"⚠️ Video too large ({len(video_bytes) / 1024 / 1024:.1f}MB). "
                    f"Try shorter duration or different model."
                )
                return

            file = discord.File(io.BytesIO(video_bytes), filename="lily_video.mp4")
            embed = discord.Embed(
                title="🎬 Generated Video",
                description=prompt[:500],
                color=discord.Color.purple(),
            )
            details = f"Model: {model} | {width}x{height} | Seed: {seed}"
            if duration:
                details += f" | {duration}s"
            if audio:
                details += " | Audio"
            embed.set_footer(text=details)

            await interaction.channel.send(embed=embed, file=file)
            db.log_generation(guild_id, interaction.user.id, "video", model, prompt[:50])

        except Exception as e:
            await interaction.channel.send(
                f"❌ Failed to generate video: {str(e)[:200]}"
            )


async def setup(bot: commands.Bot):
    await bot.add_cog(VideoGenCog(bot))
