#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Model Router

Routes different tasks to different models based on pricing and capability.
Uses the models from the user's screenshots:
  - Sana Sprint 1.6B: Image gen at 0.0001/gen (insanely cheap)
  - Ling 3.0 flash: Text at 0.1/M tokens (budget)
  - Nemotron 3 Ultra (Free): Free text model
  - MiniMax M3: Text with reasoning at 0.12/M in, 0.48/M out

Smart about spending pollen — uses cheap models for casual chat, better ones for complex tasks.
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

log = logging.getLogger("lily.model_router")


# ── Model Definitions ─────────────────────────────────────

@dataclass
class ModelInfo:
    """Info about a Pollinations model."""
    name: str
    title: str
    category: str        # "text" or "image"
    brand: str
    description: str
    prompt_price: float      # pollen per million tokens (text) or per gen (image)
    completion_price: float  # pollen per million tokens (text) or per gen (image)
    context_length: int
    has_vision: bool = False
    has_reasoning: bool = False
    has_tools: bool = False
    has_search: bool = False   # Can search the web
    input_modalities: List[str] = None
    output_modalities: List[str] = None
    is_free: bool = False

    @property
    def cost_tier(self) -> str:
        """Categorize model by cost."""
        if self.is_free or self.completion_price == 0:
            return "free"
        if self.category == "image":
            if self.completion_price <= 0.001:
                return "budget"
            elif self.completion_price <= 0.01:
                return "standard"
            else:
                return "premium"
        else:  # text
            if self.completion_price <= 0.15:
                return "budget"
            elif self.completion_price <= 0.5:
                return "standard"
            elif self.completion_price <= 2.0:
                return "premium"
            else:
                return "luxury"


class ModelRouter:
    """Routes tasks to the right model based on complexity and cost."""

    # ── Hardcoded models from the user's screenshots & API ──
    # These are the known good models with their actual pricing

    KNOWN_MODELS = {
        # ── Free models ──
        "openai-fast": ModelInfo(
            name="openai-fast", title="GPT-OSS 20B (Fast)", category="text",
            brand="OpenAI", description="Fast reasoning model, free tier",
            prompt_price=0.0, completion_price=0.0,
            context_length=400000, has_vision=True, has_reasoning=True,
            has_tools=True, input_modalities=["text", "image"],
            output_modalities=["text"], is_free=True,
        ),
        "gpt-oss": ModelInfo(
            name="gpt-oss", title="GPT-OSS 20B Reasoning", category="text",
            brand="OpenAI", description="Reasoning model, free tier",
            prompt_price=0.0, completion_price=0.0,
            context_length=131072, has_vision=False, has_reasoning=True,
            has_tools=True, input_modalities=["text"],
            output_modalities=["text"], is_free=True,
        ),

        # ── Budget text models (from screenshots) ──
        "tomdacatto/ling-3.0-flash": ModelInfo(
            name="tomdacatto/ling-3.0-flash", title="Ling 3.0 Flash", category="text",
            brand="Ling", description="Cheap fast model, multimodal",
            prompt_price=0.1, completion_price=0.1,
            context_length=250000, has_vision=True, has_reasoning=False,
            has_tools=True, input_modalities=["text", "image", "audio"],
            output_modalities=["text", "audio"], is_free=False,
        ),
        "MarcosFRG/minimax-m3": ModelInfo(
            name="MarcosFRG/minimax-m3", title="MiniMax M3", category="text",
            brand="MiniMax", description="Autonomous coding with tool use and task decomposition",
            prompt_price=0.12, completion_price=0.48,
            context_length=200000, has_vision=False, has_reasoning=True,
            has_tools=True, input_modalities=["text"],
            output_modalities=["text"], is_free=False,
        ),
        "vendouble/nemotron-3-ultra:free": ModelInfo(
            name="vendouble/nemotron-3-ultra:free", title="Nemotron 3 Ultra (Free)", category="text",
            brand="NVIDIA", description="Free text model, Alpha tier",
            prompt_price=0.0, completion_price=0.0,
            context_length=128000, has_vision=False, has_reasoning=False,
            has_tools=True, input_modalities=["text"],
            output_modalities=["text"], is_free=True,
        ),

        # ── Standard text models ──
        "openai": ModelInfo(
            name="openai", title="GPT-5.4 Nano", category="text",
            brand="OpenAI", description="Good all-rounder with vision",
            prompt_price=0.15, completion_price=0.6,
            context_length=400000, has_vision=True, has_reasoning=False,
            has_tools=True, input_modalities=["text", "image"],
            output_modalities=["text"], is_free=False,
        ),
        "openai-mini": ModelInfo(
            name="openai-mini", title="GPT-5.4 Mini", category="text",
            brand="OpenAI", description="Balanced model with reasoning",
            prompt_price=0.2, completion_price=0.8,
            context_length=400000, has_vision=True, has_reasoning=False,
            has_tools=True, input_modalities=["text", "image"],
            output_modalities=["text"], is_free=False,
        ),
        "mistral-small-3.2": ModelInfo(
            name="mistral-small-3.2", title="Mistral Small 3.2", category="text",
            brand="Mistral", description="Cheap with vision",
            prompt_price=0.1, completion_price=0.3,
            context_length=128000, has_vision=True, has_reasoning=False,
            has_tools=True, input_modalities=["text", "image"],
            output_modalities=["text"], is_free=False,
        ),

        # ── Premium text models ──
        "gpt-5.4-mini": ModelInfo(
            name="gpt-5.4-mini", title="GPT-5.4 Mini", category="text",
            brand="OpenAI", description="Strong reasoning model",
            prompt_price=0.5, completion_price=2.0,
            context_length=400000, has_vision=True, has_reasoning=True,
            has_tools=True, input_modalities=["text", "image"],
            output_modalities=["text"], is_free=False,
        ),
        "gpt-5.4": ModelInfo(
            name="gpt-5.4", title="GPT-5.4", category="text",
            brand="OpenAI", description="Deep reasoning, top tier",
            prompt_price=2.0, completion_price=8.0,
            context_length=1050000, has_vision=True, has_reasoning=True,
            has_tools=True, input_modalities=["text", "image"],
            output_modalities=["text"], is_free=False,
        ),
        "openai-large": ModelInfo(
            name="openai-large", title="GPT-5.5", category="text",
            brand="OpenAI", description="Best overall model",
            prompt_price=3.0, completion_price=12.0,
            context_length=1050000, has_vision=True, has_reasoning=True,
            has_tools=True, input_modalities=["text", "image"],
            output_modalities=["text"], is_free=False,
        ),
        "gemini-3-flash": ModelInfo(
            name="gemini-3-flash", title="Gemini 3 Flash", category="text",
            brand="Google", description="Fast multimodal model",
            prompt_price=0.1, completion_price=0.4,
            context_length=1000000, has_vision=True, has_reasoning=False,
            has_tools=True, input_modalities=["text", "image"],
            output_modalities=["text"], is_free=False,
        ),
        "qwen-coder": ModelInfo(
            name="qwen-coder", title="Qwen Coder", category="text",
            brand="Alibaba", description="Code specialist",
            prompt_price=0.15, completion_price=0.5,
            context_length=262144, has_vision=False, has_reasoning=False,
            has_tools=True, input_modalities=["text"],
            output_modalities=["text"], is_free=False,
        ),

        # ── Search-capable models ──
        "gemini-search": ModelInfo(
            name="gemini-search", title="Gemini Search", category="text",
            brand="Google", description="Web search-capable model for research",
            prompt_price=0.1, completion_price=0.4,
            context_length=1000000, has_vision=True, has_reasoning=False,
            has_tools=True, has_search=True, input_modalities=["text", "image"],
            output_modalities=["text"], is_free=False,
        ),

        # ── Fast specialized models ──
        "claude-fast": ModelInfo(
            name="claude-fast", title="Claude Fast", category="text",
            brand="Anthropic", description="Fast coding and analysis model",
            prompt_price=0.15, completion_price=0.6,
            context_length=200000, has_vision=False, has_reasoning=False,
            has_tools=True, input_modalities=["text"],
            output_modalities=["text"], is_free=False,
        ),

        # ── Deep reasoning models ──
        "deepseek": ModelInfo(
            name="deepseek", title="DeepSeek R1", category="text",
            brand="DeepSeek", description="Deep reasoning model for complex tasks",
            prompt_price=0.2, completion_price=0.8,
            context_length=131072, has_vision=False, has_reasoning=True,
            has_tools=True, input_modalities=["text"],
            output_modalities=["text"], is_free=False,
        ),

        # ── Image models ──
        "sana": ModelInfo(
            name="sana", title="Sana Sprint 1.6B", category="image",
            brand="NVIDIA", description="Near-instant images at rock-bottom cost; simpler detail than premium models",
            prompt_price=0.0001, completion_price=0.0001,
            context_length=0, is_free=False,
            input_modalities=["text"], output_modalities=["image"],
        ),
        "flux": ModelInfo(
            name="flux", title="Flux", category="image",
            brand="Black Forest Labs", description="Standard quality image generation",
            prompt_price=0.003, completion_price=0.003,
            context_length=0, is_free=False,
            input_modalities=["text"], output_modalities=["image"],
        ),
        "kontext": ModelInfo(
            name="kontext", title="Kontext", category="image",
            brand="Black Forest Labs", description="Image editing model",
            prompt_price=0.005, completion_price=0.005,
            context_length=0, is_free=False,
            input_modalities=["text", "image"], output_modalities=["image"],
        ),
        "gptimage": ModelInfo(
            name="gptimage", title="GPT Image", category="image",
            brand="OpenAI", description="GPT-powered image generation",
            prompt_price=0.01, completion_price=0.01,
            context_length=0, is_free=False,
            input_modalities=["text", "image"], output_modalities=["image"],
        ),
    }

    # Aliases — short names that map to full model IDs
    ALIASES = {
        "ling": "tomdacatto/ling-3.0-flash",
        "ling-3.0-flash": "tomdacatto/ling-3.0-flash",
        "minimax": "MarcosFRG/minimax-m3",
        "minimax-m3": "MarcosFRG/minimax-m3",
        "nemotron": "vendouble/nemotron-3-ultra:free",
        "nemotron-free": "vendouble/nemotron-3-ultra:free",
        "sana-sprint": "sana",
        "sana-sprint-1.6b": "sana",
    }

    # Task → model requirements
    TASK_PROFILES = {
        # Casual chat — free or ultra-cheap
        "casual_chat": {
            "cost_tier": "free",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 32000,
            "fallback": "openai-fast",
        },
        # Greeting — free
        "greeting": {
            "cost_tier": "free",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 16000,
            "fallback": "openai-fast",
        },
        # Proactive DM — cheap but thoughtful
        "proactive_dm": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 32000,
            "fallback": "tomdacatto/ling-3.0-flash",
        },
        # Deep conversation — needs good context
        "deep_conversation": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 64000,
            "fallback": "tomdacatto/ling-3.0-flash",
        },
        # Image analysis — needs vision
        "image_analysis": {
            "cost_tier": "budget",
            "needs_vision": True,
            "needs_reasoning": False,
            "min_context": 64000,
            "fallback": "openai",
        },
        # Creative writing — standard quality
        "creative_writing": {
            "cost_tier": "standard",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 32000,
            "fallback": "openai",
        },
        # Translation — cheap and fast
        "translation": {
            "cost_tier": "free",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 16000,
            "fallback": "openai-fast",
        },
        # Research — needs web search
        "research": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "needs_search": True,
            "min_context": 64000,
            "fallback": "gemini-search",
        },
        # Coding — needs code specialist
        "coding": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 64000,
            "fallback": "claude-fast",
        },
        # Complex reasoning — needs reasoning models
        "complex_reasoning": {
            "cost_tier": "premium",
            "needs_vision": False,
            "needs_reasoning": True,
            "min_context": 64000,
            "fallback": "deepseek",
        },
        # Daily recap — cheap
        "daily_recap": {
            "cost_tier": "free",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 32000,
            "fallback": "openai-fast",
        },
        # Memory extraction — cheap
        "memory_extraction": {
            "cost_tier": "free",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 16000,
            "fallback": "openai-fast",
        },
        # Emotion detection — ultra cheap
        "emotion_detection": {
            "cost_tier": "free",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 8000,
            "fallback": "openai-fast",
        },
        # Dream journal — creative but cheap
        "dream_journal": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 32000,
            "fallback": "tomdacatto/ling-3.0-flash",
        },
    }

    # Image model routing
    IMAGE_TASK_PROFILES = {
        "quick_image": {
            "preferred": "sana",          # Sana Sprint: 0.0001/gen — DIRT CHEAP
            "fallback": "flux",
        },
        "quality_image": {
            "preferred": "flux",          # Flux for quality
            "fallback": "sana",
        },
        "image_edit": {
            "preferred": "kontext",
            "fallback": "gptimage",
        },
        "pro_image": {
            "preferred": "gptimage",
            "fallback": "flux",
        },
    }

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._fetched = False
        # Seed with known models
        self._models.update(self.KNOWN_MODELS)

    def _resolve_alias(self, model_name: str) -> str:
        """Resolve a model alias to its full name."""
        if model_name in self.KNOWN_MODELS:
            return model_name
        return self.ALIASES.get(model_name, model_name)

    def update_models(self, models_data: List[Dict]) -> None:
        """Update the model registry from the Pollinations API."""
        # Don't overwrite our known models — they have accurate pricing
        for m in models_data:
            name = m.get("name", "")
            if name in self._models:
                continue  # Keep our hardcoded data

            # Only add text models we don't already know about
            if m.get("category") == "text":
                pricing = m.get("pricing", {})
                prompt_price = float(pricing.get("promptTextTokens", 0))
                completion_price = float(pricing.get("completionTextTokens", 0))

                info = ModelInfo(
                    name=name,
                    title=m.get("title", name),
                    category="text",
                    brand=m.get("brand", ""),
                    description=m.get("description", ""),
                    prompt_price=prompt_price,
                    completion_price=completion_price,
                    context_length=m.get("context_length", 32000) or 32000,
                    has_vision="image" in m.get("input_modalities", []),
                    has_reasoning=m.get("reasoning", False),
                    has_tools=m.get("tools", False),
                    has_search=m.get("search", False),
                    input_modalities=m.get("input_modalities", []),
                    output_modalities=m.get("output_modalities", []),
                    is_free=completion_price == 0,
                )
                self._models[name] = info

        self._fetched = True
        log.info(f"Model router has {len(self._models)} models (including {sum(1 for m in self._models.values() if m.is_free)} free)")

    def route(self, task: str, guild_model_override: str = None) -> str:
        """Route a task to the best model. Returns model name."""
        # If guild has a specific model set, use that for chat tasks
        if guild_model_override and task in ("casual_chat", "deep_conversation", "greeting"):
            resolved = self._resolve_alias(guild_model_override)
            if resolved in self._models:
                return resolved

        profile = self.TASK_PROFILES.get(task)
        if not profile:
            return "openai-fast"  # Default to free

        # Find the best model for this task
        candidates = []
        for name, info in self._models.items():
            if info.category != "text":
                continue

            # Check vision requirement
            if profile.get("needs_vision") and not info.has_vision:
                continue

            # Check reasoning requirement
            if profile.get("needs_reasoning") and not info.has_reasoning:
                continue

            # Check search requirement
            if profile.get("needs_search") and not info.has_search:
                continue

            # Check context length
            if info.context_length < profile.get("min_context", 32000):
                continue

            # Score: prefer cheaper models, then higher context
            score = 0

            # Free models get a big bonus
            if info.is_free:
                score += 50

            # Prefer models in the right cost tier
            tier_order = ["free", "budget", "standard", "premium", "luxury"]
            task_tier_idx = tier_order.index(profile["cost_tier"]) if profile["cost_tier"] in tier_order else 1
            model_tier_idx = tier_order.index(info.cost_tier) if info.cost_tier in tier_order else 1

            if model_tier_idx == task_tier_idx:
                score += 30
            elif model_tier_idx == task_tier_idx + 1:
                score += 15  # One tier above is okay
            elif model_tier_idx > task_tier_idx + 1:
                continue  # Too expensive

            # Vision bonus
            if profile.get("needs_vision") and info.has_vision:
                score += 10

            # Reasoning bonus
            if profile.get("needs_reasoning") and info.has_reasoning:
                score += 10

            # Search bonus
            if profile.get("needs_search") and info.has_search:
                score += 20

            # Context bonus
            score += min(info.context_length / 100000, 5)

            # Known good models bonus
            if name in self.KNOWN_MODELS:
                score += 5

            candidates.append((score, name))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        # Fallback
        return profile.get("fallback", "openai-fast")

    def route_image(self, task: str) -> str:
        """Route an image generation task. Default: Sana Sprint (cheapest!)."""
        profile = self.IMAGE_TASK_PROFILES.get(task, self.IMAGE_TASK_PROFILES["quick_image"])
        return profile["preferred"]

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get info about a specific model."""
        resolved = self._resolve_alias(model_name)
        return self._models.get(resolved)

    def estimate_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate pollen cost for a generation (text models only)."""
        info = self._models.get(model_name)
        if not info:
            return 0.0
        if info.category == "image":
            return info.completion_price  # Per-gen cost
        # Text: per million tokens
        prompt_cost = (prompt_tokens / 1_000_000) * info.prompt_price
        completion_cost = (completion_tokens / 1_000_000) * info.completion_price
        return prompt_cost + completion_cost

    def estimate_image_cost(self, model_name: str) -> float:
        """Estimate cost for a single image generation."""
        info = self._models.get(model_name)
        if not info or info.category != "image":
            return 0.003  # Default flux cost
        return info.completion_price

    def list_models(self, category: str = "text", cost_tier: str = None) -> List[ModelInfo]:
        """List available models, optionally filtered."""
        models = [m for m in self._models.values() if m.category == category]
        if cost_tier:
            models = [m for m in models if m.cost_tier == cost_tier]
        return sorted(models, key=lambda m: m.completion_price)

    def get_cheapest_vision_model(self) -> str:
        """Get the cheapest model that supports vision."""
        vision_models = [m for m in self._models.values() if m.has_vision and m.category == "text"]
        if not vision_models:
            return "openai"  # fallback
        return sorted(vision_models, key=lambda m: m.completion_price)[0].name

    def get_cheapest_reasoning_model(self) -> str:
        """Get the cheapest model that supports reasoning."""
        reasoning_models = [m for m in self._models.values() if m.has_reasoning and m.category == "text"]
        if not reasoning_models:
            return "gpt-oss"  # fallback
        return sorted(reasoning_models, key=lambda m: m.completion_price)[0].name

    def get_all_model_names(self) -> List[str]:
        """Get all model names."""
        return list(self._models.keys())
