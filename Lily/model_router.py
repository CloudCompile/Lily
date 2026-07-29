#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.5 — Model Router

Routes different tasks to different models based on pricing and capability.
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
    category: str
    brand: str
    description: str
    prompt_price: float      # pollen per token
    completion_price: float  # pollen per token
    context_length: int
    has_vision: bool = False
    has_reasoning: bool = False
    has_tools: bool = False
    input_modalities: List[str] = None
    output_modalities: List[str] = None

    @property
    def cost_tier(self) -> str:
        """Categorize model by cost."""
        if self.completion_price <= 0.0000003:
            return "budget"
        elif self.completion_price <= 0.000002:
            return "standard"
        elif self.completion_price <= 0.00001:
            return "premium"
        else:
            return "luxury"


class ModelRouter:
    """Routes tasks to the right model based on complexity and cost."""

    # Task → model requirements
    TASK_PROFILES = {
        # Casual chat — cheap and fast
        "casual_chat": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 32000,
            "fallback": "openai-fast",
        },
        # Greeting — ultra cheap
        "greeting": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 16000,
            "fallback": "openai-fast",
        },
        # Proactive DM — casual but thoughtful
        "proactive_dm": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 32000,
            "fallback": "openai",
        },
        # Deep conversation — needs good context
        "deep_conversation": {
            "cost_tier": "standard",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 64000,
            "fallback": "openai-mini",
        },
        # Image analysis — needs vision
        "image_analysis": {
            "cost_tier": "standard",
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
            "fallback": "openai-mini",
        },
        # Translation — cheap and fast
        "translation": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 16000,
            "fallback": "openai-fast",
        },
        # Complex reasoning — needs reasoning models
        "complex_reasoning": {
            "cost_tier": "premium",
            "needs_vision": False,
            "needs_reasoning": True,
            "min_context": 64000,
            "fallback": "gpt-5.4-mini",
        },
        # Daily recap — cheap
        "daily_recap": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 32000,
            "fallback": "openai-fast",
        },
        # Memory extraction — cheap
        "memory_extraction": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 16000,
            "fallback": "openai-fast",
        },
        # Emotion detection — ultra cheap
        "emotion_detection": {
            "cost_tier": "budget",
            "needs_vision": False,
            "needs_reasoning": False,
            "min_context": 8000,
            "fallback": "openai-fast",
        },
    }

    # Image model routing
    IMAGE_TASK_PROFILES = {
        "quick_image": {
            "preferred": "sana",
            "fallback": "flux",
        },
        "quality_image": {
            "preferred": "nanobanana-2",
            "fallback": "flux",
        },
        "image_edit": {
            "preferred": "kontext",
            "fallback": "gptimage",
        },
        "pro_image": {
            "preferred": "nanobanana-pro",
            "fallback": "seedream5-pro",
        },
    }

    # Known good models by category (hardcoded as fallbacks)
    KNOWN_TEXT_MODELS = {
        "budget": [
            "openai-fast",     # GPT-5 Nano — ultra cheap
            "nova-fast",       # Nova Micro — very cheap
            "gpt-oss",         # GPT-OSS 20B — cheap with reasoning
        ],
        "standard": [
            "openai",          # GPT-5.4 Nano — good all-rounder
            "openai-mini",     # GPT-5.4 Mini — balanced
            "mistral-small-3.2", # Mistral — cheap with vision
        ],
        "premium": [
            "gpt-5.4-mini",    # GPT-5.4 Mini — strong reasoning
            "gpt-5.4",         # GPT-5.4 — deep reasoning
            "openai-large",    # GPT-5.5 — top tier
        ],
        "luxury": [
            "gpt-5.4",         # GPT-5.4 — best reasoning
            "openai-large",    # GPT-5.5 — best overall
        ],
    }

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._fetched = False

    def update_models(self, models_data: List[Dict]) -> None:
        """Update the model registry from the Pollinations API."""
        self._models.clear()
        for m in models_data:
            if m.get("category") != "text":
                continue
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("promptTextTokens", 999))
            completion_price = float(pricing.get("completionTextTokens", 999))
            if completion_price >= 999:
                continue  # Skip models without pricing

            info = ModelInfo(
                name=m.get("name", ""),
                title=m.get("title", ""),
                category=m.get("category", "text"),
                brand=m.get("brand", ""),
                description=m.get("description", ""),
                prompt_price=prompt_price,
                completion_price=completion_price,
                context_length=m.get("context_length", 32000) or 32000,
                has_vision="image" in m.get("input_modalities", []),
                has_reasoning="reasoning" in m.get("capabilities", []),
                has_tools=m.get("tools", False),
                input_modalities=m.get("input_modalities", []),
                output_modalities=m.get("output_modalities", []),
            )
            self._models[info.name] = info

        self._fetched = True
        log.info(f"Model router loaded {len(self._models)} text models")

    def route(self, task: str, guild_model_override: str = None) -> str:
        """Route a task to the best model. Returns model name."""
        # If guild has a specific model set, use that for chat tasks
        if guild_model_override and task in ("casual_chat", "deep_conversation", "greeting"):
            return guild_model_override

        profile = self.TASK_PROFILES.get(task)
        if not profile:
            return profile.get("fallback", "openai") if profile else "openai"

        # Find the best model for this task
        candidates = []
        for name, info in self._models.items():
            # Check cost tier
            if info.cost_tier != profile["cost_tier"]:
                # Allow one tier above
                tier_order = ["budget", "standard", "premium", "luxury"]
                task_tier_idx = tier_order.index(profile["cost_tier"]) if profile["cost_tier"] in tier_order else 1
                model_tier_idx = tier_order.index(info.cost_tier) if info.cost_tier in tier_order else 1
                if model_tier_idx > task_tier_idx + 1:
                    continue  # Too expensive

            # Check vision requirement
            if profile.get("needs_vision") and not info.has_vision:
                continue

            # Check reasoning requirement
            if profile.get("needs_reasoning") and not info.has_reasoning:
                continue

            # Check context length
            if info.context_length < profile.get("min_context", 32000):
                continue

            # Score: prefer cheaper models, then higher context
            score = 0
            # Prefer models in the right cost tier
            if info.cost_tier == profile["cost_tier"]:
                score += 10
            # Prefer models with exactly what we need
            if profile.get("needs_vision") and info.has_vision:
                score += 5
            if profile.get("needs_reasoning") and info.has_reasoning:
                score += 5
            # Prefer larger context
            score += min(info.context_length / 100000, 5)

            candidates.append((score, name))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        # Fallback
        return profile.get("fallback", "openai")

    def route_image(self, task: str) -> str:
        """Route an image generation task."""
        profile = self.IMAGE_TASK_PROFILES.get(task, self.IMAGE_TASK_PROFILES["quick_image"])
        return profile["preferred"]

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get info about a specific model."""
        return self._models.get(model_name)

    def estimate_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate pollen cost for a generation."""
        info = self._models.get(model_name)
        if not info:
            return 0.0
        return (prompt_tokens * info.prompt_price) + (completion_tokens * info.completion_price)

    def list_models(self, category: str = "text", cost_tier: str = None) -> List[ModelInfo]:
        """List available models, optionally filtered."""
        models = [m for m in self._models.values() if m.category == category]
        if cost_tier:
            models = [m for m in models if m.cost_tier == cost_tier]
        return sorted(models, key=lambda m: m.completion_price)

    def get_cheapest_vision_model(self) -> str:
        """Get the cheapest model that supports vision."""
        vision_models = [m for m in self._models.values() if m.has_vision]
        if not vision_models:
            return "openai"  # fallback
        return sorted(vision_models, key=lambda m: m.completion_price)[0].name

    def get_cheapest_reasoning_model(self) -> str:
        """Get the cheapest model that supports reasoning."""
        reasoning_models = [m for m in self._models.values() if m.has_reasoning]
        if not reasoning_models:
            return "gpt-oss"  # fallback
        return sorted(reasoning_models, key=lambda m: m.completion_price)[0].name
