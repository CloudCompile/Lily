#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Audio Generation Cog

Commands: /tts, /tts_simple, /music, /transcribe.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands
import io

from database import Database
from pollinations import PollinationsAPI
from config import DEFAULT_TTS_MODEL, DEFAULT_TRANSCRIPTION_MODEL, DEFAULT_SAFE_MODE


class AudioGenCog(commands.Cog, name="Audio Generation"):
    """Audio generation commands using Pollinations API."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="tts", description="Convert text to speech")
    @app_commands.describe(
        text="Text to convert to speech",
        voice="Voice to use (alloy, echo, fable, onyx, nova, shimmer, etc.)",
        model="TTS model to use",
        format="Output format (mp3, opus, aac, flac, wav)",
    )
    async def tts(
        self,
        interaction: discord.Interaction,
        text: str,
        voice: str = "nova",
        model: str = None,
        format: str = "mp3",
    ):
        """Generate speech from text using OpenAI-compatible TTS."""
        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore

        use_model = model or db.get_guild_setting(guild_id, "tts_model", DEFAULT_TTS_MODEL)

        try:
            audio_bytes = await api.tts(
                text,
                voice=voice,
                model=use_model,
                response_format=format,
            )

            ext = format if format in ("mp3", "flac", "wav") else "mp3"
            file = discord.File(io.BytesIO(audio_bytes), filename=f"lily_tts.{ext}")

            embed = discord.Embed(
                title="🔊 Text to Speech",
                description=text[:500],
                color=discord.Color.blurple(),
            )
            embed.set_footer(text=f"Voice: {voice} | Model: {use_model} | Format: {format}")

            await interaction.followup.send(embed=embed, file=file)
            db.log_generation(guild_id, interaction.user.id, "tts", use_model, text[:50])

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to generate speech: {str(e)[:200]}", ephemeral=True
            )

    @app_commands.command(name="tts_simple", description="Quick text-to-speech via GET")
    @app_commands.describe(
        text="Text to convert to speech",
        voice="Voice to use",
        model="TTS model (or music model for music)",
    )
    async def tts_simple(
        self,
        interaction: discord.Interaction,
        text: str,
        voice: str = "alloy",
        model: str = None,
    ):
        """Simple TTS via GET endpoint."""
        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        api: PollinationsAPI = self.bot.api  # type: ignore

        try:
            audio_bytes = await api.tts_simple(text, voice=voice, model=model)

            file = discord.File(io.BytesIO(audio_bytes), filename="lily_tts.mp3")
            embed = discord.Embed(
                title="🔊 Text to Speech",
                description=text[:500],
                color=discord.Color.blurple(),
            )
            embed.set_footer(text=f"Voice: {voice}")

            await interaction.followup.send(embed=embed, file=file)
            db: Database = self.bot.db  # type: ignore
            db.log_generation(guild_id, interaction.user.id, "tts", model or "default", text[:50])

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to generate speech: {str(e)[:200]}", ephemeral=True
            )

    @app_commands.command(name="music", description="Generate music from a prompt")
    @app_commands.describe(
        prompt="Description of the music to generate",
        model="Music model (elevenmusic, lyria-3-clip, stable-audio-3-medium, etc.)",
        duration="Duration in seconds",
        instrumental="Generate instrumental only (no vocals)",
    )
    async def music(
        self,
        interaction: discord.Interaction,
        prompt: str,
        model: str = "elevenmusic",
        duration: int = None,
        instrumental: bool = False,
    ):
        """Generate music from a text description."""
        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore
        safe = db.get_guild_setting(guild_id, "safe_mode", DEFAULT_SAFE_MODE)

        await interaction.followup.send(
            f"🎵 Generating music with `{model}`... This may take a moment.",
        )

        try:
            kwargs = {}
            if duration:
                kwargs["duration"] = duration
            if instrumental:
                kwargs["instrumental"] = True

            audio_bytes = await api.music_generate(
                prompt,
                model=model,
                safe=safe,
                **kwargs,
            )

            file = discord.File(io.BytesIO(audio_bytes), filename="lily_music.mp3")
            embed = discord.Embed(
                title="🎵 Generated Music",
                description=prompt[:500],
                color=discord.Color.dark_purple(),
            )
            details = f"Model: {model}"
            if duration:
                details += f" | {duration}s"
            if instrumental:
                details += " | Instrumental"
            embed.set_footer(text=details)

            await interaction.channel.send(embed=embed, file=file)
            db.log_generation(guild_id, interaction.user.id, "music", model, prompt[:50])

        except Exception as e:
            await interaction.channel.send(
                f"❌ Failed to generate music: {str(e)[:200]}"
            )

    @app_commands.command(name="transcribe", description="Transcribe an audio file to text")
    @app_commands.describe(
        language="Language hint (e.g. en, fr, es)",
        model="Transcription model (whisper, scribe, universal-2, universal-3-pro)",
    )
    async def transcribe(
        self,
        interaction: discord.Interaction,
        language: str = None,
        model: str = None,
    ):
        """Transcribe an attached audio file to text."""
        # Check for attachment
        if not interaction.message or not interaction.message.attachments:
            await interaction.response.send_message(
                "❌ Please attach an audio file to transcribe! "
                "Supported: mp3, mp4, wav, m4a, webm",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)

        guild_id = interaction.guild_id or 0
        db: Database = self.bot.db  # type: ignore
        api: PollinationsAPI = self.bot.api  # type: ignore

        use_model = model or db.get_guild_setting(guild_id, "transcription_model", DEFAULT_TRANSCRIPTION_MODEL)

        try:
            attachment = interaction.message.attachments[0]
            audio_bytes = await attachment.read()

            kwargs = {}
            if language:
                kwargs["language"] = language

            result = await api.transcribe(
                audio_bytes,
                model=use_model,
                filename=attachment.filename or "audio.mp3",
                **kwargs,
            )

            text = result.get("text", "No transcription returned.")

            embed = discord.Embed(
                title="📝 Transcription",
                color=discord.Color.green(),
            )
            embed.add_field(name="File", value=attachment.filename, inline=True)
            embed.add_field(name="Model", value=use_model, inline=True)
            if language:
                embed.add_field(name="Language", value=language, inline=True)

            # Truncate if too long
            if len(text) > 1024:
                text = text[:1021] + "..."
            embed.add_field(name="Transcription", value=text, inline=False)

            # Include segments if available
            if "segments" in result:
                segments = result["segments"][:5]
                seg_text = "\n".join(
                    f"[{s.get('start', 0):.1f}s - {s.get('end', 0):.1f}s] {s.get('speaker', 'Speaker')}: {s.get('text', '')}"
                    for s in segments
                )
                if len(seg_text) > 1024:
                    seg_text = seg_text[:1021] + "..."
                embed.add_field(name="Segments (first 5)", value=seg_text, inline=False)

            embed.set_footer(text=f"File: {attachment.filename}")
            await interaction.followup.send(embed=embed)
            db.log_generation(guild_id, interaction.user.id, "transcribe", use_model, "audio")

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to transcribe: {str(e)[:200]}", ephemeral=True
            )


async def setup(bot: commands.Bot):
    await bot.add_cog(AudioGenCog(bot))
