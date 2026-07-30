#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — AI Chat Cog

Text generation commands with smart model routing, relationship awareness,
and memory integration. v9.0: Cross-server memories, cheap models by default.
"""

from __future__ import annotations
import asyncio
import random
import discord
from discord.ext import commands
from discord import app_commands

from database import Database
from pollinations import PollinationsAPI
from personality import PersonalityEngine
from relationships import RelationshipEngine
from memories import MemorySystem
from model_router import ModelRouter
from quotas import QuotaSystem
from utils import generate_with_retry, chunk_response, send_chunked
from config import DEFAULT_TEXT_MODEL, DEFAULT_SAFE_MODE


class AIChatCog(commands.Cog, name="AI Chat"):
    """AI text generation commands — the heart of Lily."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.hybrid_command(name="ask", description="Ask Lily a question")
    @app_commands.describe(question="Your question", model="AI model to use (leave empty for smart routing)")
    async def ask(
        self,
        ctx: commands.Context,
        question: str,
        model: str = None,
    ):
        """Ask Lily anything using Pollinations text generation."""
        await ctx.defer(thinking=True)

        guild_id = ctx.guild.id if ctx.guild else 0
        api: PollinationsAPI = self.bot.api  # type: ignore
        personality: PersonalityEngine = self.bot.personality  # type: ignore
        db: Database = self.bot.db  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore
        memories: MemorySystem = self.bot.memories  # type: ignore
        quotas: QuotaSystem = self.bot.quotas  # type: ignore

        user_id = ctx.author.id

        # Get relationship
        rel = rel_engine.get_relationship(guild_id, user_id)
        warmth = rel.warmth

        # Check quota (casual chat is free with openai-fast)
        can_gen, reason = quotas.can_generate(guild_id, user_id, "text_casual", rel.relationship_tier)
        if not can_gen:
            if ctx.interaction:
                await ctx.send(f"❌ {reason}", ephemeral=True)
            else:
                await ctx.send(f"❌ {reason}")
            return

        # Smart model routing
        use_model = model or await self.bot.get_model_for_task("casual_chat", guild_id)
        safe = db.get_guild_setting(guild_id, "safe_mode", DEFAULT_SAFE_MODE)

        # Build context — cross-server memories (DMs get cross-server history)
        is_dm = not ctx.guild
        if is_dm:
            history = db.get_conversations_cross_server(user_id, limit=15)
        else:
            history = db.get_conversations(guild_id, user_id, limit=15)
        user_facts = db.get_facts(guild_id, user_id, cross_server=True)
        recent_recaps = db.get_daily_recaps(guild_id, user_id, 3)
        memory_context = memories.get_memories_for_prompt(guild_id, user_id, question)
        relationship_context = rel_engine.get_system_prompt_addition(guild_id, user_id)
        dream_context = memories.get_dreams_for_prompt(user_id)

        # Update mood
        mood, intensity = personality.mood.update(question)
        system_prompt = personality.build_system_prompt(
            mood, personality.mood.get_energy(),
            user_facts, relationship_context, memory_context,
            [r.get("recap_text", "") for r in recent_recaps] if recent_recaps else None,
            dream_context,
        )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        # Detect action for relationship
        action = rel_engine.detect_action(question, is_command=True)
        rel_engine.record_action(guild_id, user_id, action)

        try:
            # Add a natural typing delay
            typing_delay = personality.get_typing_delay(len(question) * 3, personality.mood.get_energy())
            await asyncio.sleep(min(typing_delay, 3.0))

            response = await api.chat_completions_simple(
                messages, model=use_model, safe=safe, max_tokens=500
            )
            response = personality.inject_personality(response, warmth)

            # Maybe add thinking prefix
            thinking = personality.get_thinking_prefix()
            if thinking and random.random() < 0.1:
                response = thinking + " " + response

            # Save to conversation history
            db.add_conversation(guild_id, user_id, "user", question)
            db.add_conversation(guild_id, user_id, "assistant", response, mood)

            # Track topics
            topics = personality.extract_topics(question)
            for topic in topics:
                db.add_topic(guild_id, user_id, topic)

            # Save important memories (cross-server)
            emotion = personality.detect_emotion(question)
            if emotion in ("happy", "sad", "affection", "vulnerable", "excited"):
                db.save_memory(guild_id, user_id, question, memory_type="auto",
                             emotion=emotion, importance=0.7, tags=topics, is_global=True)

            # Record generation with actual cost
            cost = self.bot.model_router.estimate_cost(use_model, len(question.split()) * 2, len(response.split()) * 2)
            quotas.record_generation(guild_id, user_id, "text_casual", rel.relationship_tier, actual_cost=cost)
            db.log_generation(guild_id, user_id, "text", use_model, question[:50], cost_pollen=cost)

            # Use chunked response for long text
            if len(response) > 1900:
                await send_chunked(ctx, response)
            else:
                await ctx.send(response)

        except Exception as e:
            if ctx.interaction:
                await ctx.send(f"❌ Failed to generate response: {str(e)[:200]}", ephemeral=True)
            else:
                await ctx.send(f"❌ Failed to generate response: {str(e)[:200]}")

    @commands.hybrid_command(name="chat", description="Have a conversation with Lily")
    @app_commands.describe(message="Your message", model="AI model to use")
    async def chat(
        self,
        ctx: commands.Context,
        message: str,
        model: str = None,
    ):
        """Have a conversation with Lily using full context."""
        await ctx.defer(thinking=True)

        guild_id = ctx.guild.id if ctx.guild else 0
        api: PollinationsAPI = self.bot.api  # type: ignore
        personality: PersonalityEngine = self.bot.personality  # type: ignore
        db: Database = self.bot.db  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore
        memories: MemorySystem = self.bot.memories  # type: ignore
        quotas: QuotaSystem = self.bot.quotas  # type: ignore

        user_id = ctx.author.id
        rel = rel_engine.get_relationship(guild_id, user_id)
        warmth = rel.warmth

        # Check quota
        can_gen, reason = quotas.can_generate(guild_id, user_id, "text_casual", rel.relationship_tier)
        if not can_gen:
            if ctx.interaction:
                await ctx.send(f"❌ {reason}", ephemeral=True)
            else:
                await ctx.send(f"❌ {reason}")
            return

        # Route to appropriate model
        task = "deep_conversation" if len(message) > 200 else "casual_chat"
        use_model = model or await self.bot.get_model_for_task(task, guild_id)
        safe = db.get_guild_setting(guild_id, "safe_mode", DEFAULT_SAFE_MODE)

        # Build context — cross-server (DMs get cross-server history)
        is_dm = not ctx.guild
        if is_dm:
            history = db.get_conversations_cross_server(user_id, limit=15)
        else:
            history = db.get_conversations(guild_id, user_id, limit=15)
        user_facts = db.get_facts(guild_id, user_id, cross_server=True)
        recent_recaps = db.get_daily_recaps(guild_id, user_id, 3)
        memory_context = memories.get_memories_for_prompt(guild_id, user_id, message)
        relationship_context = rel_engine.get_system_prompt_addition(guild_id, user_id)
        dream_context = memories.get_dreams_for_prompt(user_id)

        mood, _ = personality.mood.update(message)
        system_prompt = personality.build_system_prompt(
            mood, personality.mood.get_energy(),
            user_facts, relationship_context, memory_context,
            [r.get("recap_text", "") for r in recent_recaps] if recent_recaps else None,
            dream_context,
        )

        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        api_messages.append({"role": "user", "content": message})

        # Detect action
        action = rel_engine.detect_action(message, is_command=False)
        rel_engine.record_action(guild_id, user_id, action)

        try:
            typing_delay = personality.get_typing_delay(len(message) * 3, personality.mood.get_energy())
            await asyncio.sleep(min(typing_delay, 3.0))

            response = await api.chat_completions_simple(
                api_messages, model=use_model, safe=safe, max_tokens=500
            )
            response = personality.inject_personality(response, warmth)

            db.add_conversation(guild_id, user_id, "user", message)
            db.add_conversation(guild_id, user_id, "assistant", response, mood)

            topics = personality.extract_topics(message)
            for topic in topics:
                db.add_topic(guild_id, user_id, topic)

            emotion = personality.detect_emotion(message)
            if emotion in ("happy", "sad", "affection", "vulnerable", "excited"):
                db.save_memory(guild_id, user_id, message, memory_type="auto",
                             emotion=emotion, importance=0.7, tags=topics, is_global=True)

            cost = self.bot.model_router.estimate_cost(use_model, len(message.split()) * 2, len(response.split()) * 2)
            quotas.record_generation(guild_id, user_id, "text_casual", rel.relationship_tier, actual_cost=cost)
            db.log_generation(guild_id, user_id, "text", use_model, message[:50], cost_pollen=cost)

            # Use chunked response for long text
            if len(response) > 1900:
                await send_chunked(ctx, response)
            else:
                await ctx.send(response)

        except Exception as e:
            if ctx.interaction:
                await ctx.send(f"❌ Failed to generate response: {str(e)[:200]}", ephemeral=True)
            else:
                await ctx.send(f"❌ Failed to generate response: {str(e)[:200]}")

    @commands.hybrid_command(name="imagine", description="Generate creative text from a prompt")
    @app_commands.describe(
        prompt="What to imagine",
        model="AI model to use",
        temperature="Creativity (0.0-2.0)",
    )
    async def imagine(
        self,
        ctx: commands.Context,
        prompt: str,
        model: str = None,
        temperature: float = None,
    ):
        """Creative text generation."""
        await ctx.defer(thinking=True)

        guild_id = ctx.guild.id if ctx.guild else 0
        api: PollinationsAPI = self.bot.api  # type: ignore
        db: Database = self.bot.db  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore
        quotas: QuotaSystem = self.bot.quotas  # type: ignore

        user_id = ctx.author.id
        rel = rel_engine.get_relationship(guild_id, user_id)

        can_gen, reason = quotas.can_generate(guild_id, user_id, "text_standard", rel.relationship_tier)
        if not can_gen:
            if ctx.interaction:
                await ctx.send(f"❌ {reason}", ephemeral=True)
            else:
                await ctx.send(f"❌ {reason}")
            return

        use_model = model or await self.bot.get_model_for_task("creative_writing", guild_id)
        safe = db.get_guild_setting(guild_id, "safe_mode", DEFAULT_SAFE_MODE)

        try:
            response = await api.text_simple(
                prompt,
                model=use_model,
                system="You are a creative writer. Be vivid and imaginative. Write in a casual, engaging style.",
                temperature=temperature or 0.9,
                safe=safe,
            )

            cost = self.bot.model_router.estimate_cost(use_model, len(prompt.split()) * 2, len(response.split()) * 2)
            quotas.record_generation(guild_id, user_id, "text_standard", rel.relationship_tier, actual_cost=cost)
            db.log_generation(guild_id, user_id, "text", use_model, prompt[:50], cost_pollen=cost)

            if len(response) > 1900:
                await send_chunked(ctx, response)
            else:
                await ctx.send(response)

        except Exception as e:
            if ctx.interaction:
                await ctx.send(f"❌ Failed to generate: {str(e)[:200]}", ephemeral=True)
            else:
                await ctx.send(f"❌ Failed to generate: {str(e)[:200]}")

    @commands.hybrid_command(name="analyze", description="Analyze an image using AI vision")
    @app_commands.describe(
        image_url="URL of the image to analyze",
        question="What do you want to know about the image?",
        model="Vision-capable model to use",
    )
    async def analyze(
        self,
        ctx: commands.Context,
        image_url: str,
        question: str = "What is in this image?",
        model: str = None,
    ):
        """Analyze an image using a vision-capable model."""
        await ctx.defer(thinking=True)

        guild_id = ctx.guild.id if ctx.guild else 0
        api: PollinationsAPI = self.bot.api  # type: ignore
        db: Database = self.bot.db  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore
        quotas: QuotaSystem = self.bot.quotas  # type: ignore

        user_id = ctx.author.id
        rel = rel_engine.get_relationship(guild_id, user_id)

        can_gen, reason = quotas.can_generate(guild_id, user_id, "text_standard", rel.relationship_tier)
        if not can_gen:
            if ctx.interaction:
                await ctx.send(f"❌ {reason}", ephemeral=True)
            else:
                await ctx.send(f"❌ {reason}")
            return

        use_model = model or await self.bot.get_model_for_task("image_analysis", guild_id)

        try:
            response = await api.analyze_image(image_url, question, model=use_model)

            cost = self.bot.model_router.estimate_cost(use_model, 500, 200)
            quotas.record_generation(guild_id, user_id, "text_standard", rel.relationship_tier, actual_cost=cost)
            db.log_generation(guild_id, user_id, "image_analysis", use_model, question[:50], cost_pollen=cost)

            if len(response) > 1900:
                desc = response[:1900]
            else:
                desc = response

            embed = discord.Embed(
                title="🔍 Image Analysis",
                description=response,
                color=discord.Color.blue(),
            )
            embed.set_thumbnail(url=image_url)
            embed.set_footer(text=f"Model: {use_model}")
            await ctx.send(embed=embed)

        except Exception as e:
            if ctx.interaction:
                await ctx.send(f"❌ Failed to analyze image: {str(e)[:200]}", ephemeral=True)
            else:
                await ctx.send(f"❌ Failed to analyze image: {str(e)[:200]}")

    @commands.hybrid_command(name="translate", description="Translate text to another language")
    @app_commands.describe(
        text="Text to translate",
        language="Target language (e.g. Spanish, French, Japanese)",
        model="AI model to use",
    )
    async def translate(
        self,
        ctx: commands.Context,
        text: str,
        language: str,
        model: str = None,
    ):
        """Translate text using AI."""
        await ctx.defer(thinking=True)

        guild_id = ctx.guild.id if ctx.guild else 0
        api: PollinationsAPI = self.bot.api  # type: ignore
        db: Database = self.bot.db  # type: ignore
        rel_engine: RelationshipEngine = self.bot.relationships  # type: ignore
        quotas: QuotaSystem = self.bot.quotas  # type: ignore

        user_id = ctx.author.id
        rel = rel_engine.get_relationship(guild_id, user_id)

        can_gen, reason = quotas.can_generate(guild_id, user_id, "translation", rel.relationship_tier)
        if not can_gen:
            if ctx.interaction:
                await ctx.send(f"❌ {reason}", ephemeral=True)
            else:
                await ctx.send(f"❌ {reason}")
            return

        use_model = model or await self.bot.get_model_for_task("translation", guild_id)

        system_prompt = f"You are a professional translator. Translate the following text to {language}. Only output the translation, nothing else."

        try:
            response = await api.chat_completions_simple(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                model=use_model,
                max_tokens=1000,
            )

            cost = self.bot.model_router.estimate_cost(use_model, len(text.split()) * 2, len(response.split()) * 2)
            quotas.record_generation(guild_id, user_id, "translation", rel.relationship_tier, actual_cost=cost)
            db.log_generation(guild_id, user_id, "translation", use_model, text[:50], cost_pollen=cost)

            embed = discord.Embed(
                title=f"🌐 Translation to {language}",
                color=discord.Color.green(),
            )
            embed.add_field(name="Original", value=text[:1024], inline=False)
            embed.add_field(name="Translation", value=response[:1024], inline=False)
            embed.set_footer(text=f"Model: {use_model}")
            await ctx.send(embed=embed)

        except Exception as e:
            if ctx.interaction:
                await ctx.send(f"❌ Failed to translate: {str(e)[:200]}", ephemeral=True)
            else:
                await ctx.send(f"❌ Failed to translate: {str(e)[:200]}")


async def setup(bot: commands.Bot):
    await bot.add_cog(AIChatCog(bot))
