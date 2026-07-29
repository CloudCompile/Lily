#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.5 — Generation Quota System

Not unlimited willy-nilly generations. Each user gets a daily pollen budget.
Rations are based on relationship tier and server settings.

v8.5 update: Costs are now based on actual model pricing.
  - Sana Sprint images: 0.0001 pollen/gen (basically free)
  - Ling 3.0 flash text: 0.1/M tokens (super cheap)
  - Free models (openai-fast, nemotron): 0 pollen
  - Premium models: cost more, used sparingly
"""

from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

log = logging.getLogger("lily.quotas")


# ── Quota Config ──────────────────────────────────────────

@dataclass
class UserQuota:
    """A user's generation quota for the current period."""
    user_id: str
    guild_id: str
    period_start: str = ""

    # Token counts (in pollen)
    pollen_spent: float = 0.0
    pollen_budget: float = 10.0  # Daily budget (much lower now — models are cheap)

    # Generation counts
    text_gens: int = 0
    image_gens: int = 0
    total_gens: int = 0

    # Limits
    max_text_per_day: int = 100
    max_image_per_day: int = 50
    max_gens_per_hour: int = 30

    # Hourly tracking
    hourly_gens: int = 0
    hourly_reset: float = 0.0


class QuotaSystem:
    """Manages generation quotas. No unlimited willy-nilly generations."""

    # Base budgets by relationship tier
    # With Sana Sprint at 0.0001/gen and free text models, we can be generous
    # but still cap it so it's not unlimited
    TIER_BUDGETS = {
        "rival":        {"pollen": 2.0,   "text": 20,  "image": 10},
        "strained":     {"pollen": 3.0,   "text": 30,  "image": 15},
        "stranger":     {"pollen": 5.0,   "text": 50,  "image": 25},
        "acquaintance": {"pollen": 7.0,   "text": 70,  "image": 30},
        "friend":       {"pollen": 10.0,  "text": 100, "image": 40},
        "close_friend": {"pollen": 15.0,  "text": 150, "image": 50},
        "bestie":       {"pollen": 20.0,  "text": 200, "image": 60},
        "soulmate":     {"pollen": 30.0,  "text": 300, "image": 80},
    }

    # Actual costs per generation type (in pollen) based on real model pricing
    # Free models = 0 cost, budget models = tiny cost, premium = more
    GENERATION_COSTS = {
        # Text — most tasks use free models
        "text_casual":       0.0,    # Free model (openai-fast)
        "text_standard":     0.01,   # Ling 3.0 flash (~0.1/M tokens, ~100 tokens = 0.01)
        "text_premium":      0.05,   # Better model for complex tasks
        "text_reasoning":    0.1,    # MiniMax M3 or reasoning model

        # Image — Sana Sprint is insanely cheap
        "image_quick":       0.0001, # Sana Sprint: 0.0001/gen
        "image_standard":    0.003,  # Flux: 0.003/gen
        "image_pro":         0.01,   # GPT Image: 0.01/gen
        "image_edit":        0.005,  # Kontext: 0.005/gen

        # Other
        "translation":       0.0,    # Free model
        "emotion_detection": 0.0,    # Very cheap / free
        "daily_recap":       0.0,    # Free model
        "dream_journal":     0.01,   # Budget model for creative generation
    }

    # How many generations of each type per day (hard caps regardless of pollen)
    GENERATION_COUNT_LIMITS = {
        "text_casual":       200,
        "text_standard":     100,
        "text_premium":      20,
        "text_reasoning":    10,
        "image_quick":       100,
        "image_standard":    50,
        "image_pro":         15,
        "image_edit":        25,
        "translation":       50,
        "dream_journal":     5,
    }

    def __init__(self):
        self._quotas: Dict[str, UserQuota] = {}
        self._gen_counts: Dict[str, Dict[str, int]] = {}  # key -> {gen_type: count}

    def _key(self, guild_id: int | str, user_id: int | str) -> str:
        return f"{guild_id}:{user_id}"

    def get_quota(self, guild_id: int, user_id: int, relationship_tier: str = "stranger") -> UserQuota:
        """Get or create a user's quota."""
        key = self._key(guild_id, user_id)
        today = datetime.now().strftime("%Y-%m-%d")

        if key not in self._quotas:
            quota = self._create_quota(guild_id, user_id, relationship_tier)
            self._quotas[key] = quota
        else:
            quota = self._quotas[key]
            # Reset if new day
            if quota.period_start != today:
                quota = self._create_quota(guild_id, user_id, relationship_tier)
                self._quotas[key] = quota
                # Reset gen counts too
                self._gen_counts.pop(key, None)

        return quota

    def _create_quota(self, guild_id: int | str, user_id: int | str, relationship_tier: str) -> UserQuota:
        """Create a fresh daily quota based on relationship tier."""
        budget = self.TIER_BUDGETS.get(relationship_tier, self.TIER_BUDGETS["stranger"])
        return UserQuota(
            user_id=str(user_id),
            guild_id=str(guild_id),
            period_start=datetime.now().strftime("%Y-%m-%d"),
            pollen_budget=budget["pollen"],
            max_text_per_day=budget["text"],
            max_image_per_day=budget["image"],
            hourly_reset=time.time(),
        )

    def _get_gen_count(self, key: str, gen_type: str) -> int:
        """Get the count of a specific generation type today."""
        if key not in self._gen_counts:
            return 0
        return self._gen_counts[key].get(gen_type, 0)

    def _increment_gen_count(self, key: str, gen_type: str) -> None:
        """Increment the count of a specific generation type."""
        if key not in self._gen_counts:
            self._gen_counts[key] = {}
        self._gen_counts[key][gen_type] = self._gen_counts[key].get(gen_type, 0) + 1

    def can_generate(
        self, guild_id: int, user_id: int, gen_type: str,
        relationship_tier: str = "stranger"
    ) -> Tuple[bool, str]:
        """Check if a user can generate. Returns (allowed, reason)."""
        quota = self.get_quota(guild_id, user_id, relationship_tier)
        cost = self.GENERATION_COSTS.get(gen_type, 0.01)
        key = self._key(guild_id, user_id)

        # Check hourly limit
        now = time.time()
        if now - quota.hourly_reset > 3600:
            quota.hourly_gens = 0
            quota.hourly_reset = now

        if quota.hourly_gens >= quota.max_gens_per_hour:
            remaining_mins = int((3600 - (now - quota.hourly_reset)) / 60)
            return False, f"Slow down! You've hit the hourly limit. Try again in ~{remaining_mins} min."

        # Check daily pollen budget
        if quota.pollen_spent + cost > quota.pollen_budget:
            remaining = quota.pollen_budget - quota.pollen_spent
            if remaining < 0:
                remaining = 0
            return False, f"You've used up your daily pollen budget! ({quota.pollen_spent:.4f}/{quota.pollen_budget:.1f} pollen used today)"

        # Check type-specific limits
        type_limit = self.GENERATION_COUNT_LIMITS.get(gen_type, 100)
        type_count = self._get_gen_count(key, gen_type)
        if type_count >= type_limit:
            return False, f"You've hit your daily limit for {gen_type} ({type_limit}). Try again tomorrow!"

        is_text = gen_type.startswith("text") or gen_type in ("translation", "emotion_detection", "daily_recap", "dream_journal")
        is_image = gen_type.startswith("image")

        if is_text and quota.text_gens >= quota.max_text_per_day:
            return False, f"You've hit your daily text generation limit ({quota.max_text_per_day}). Try again tomorrow!"

        if is_image and quota.image_gens >= quota.max_image_per_day:
            return False, f"You've hit your daily image generation limit ({quota.max_image_per_day}). Try again tomorrow!"

        return True, ""

    def record_generation(
        self, guild_id: int, user_id: int, gen_type: str,
        relationship_tier: str = "stranger", actual_cost: float = None
    ) -> float:
        """Record that a generation was used. Returns the actual cost."""
        quota = self.get_quota(guild_id, user_id, relationship_tier)
        cost = actual_cost if actual_cost is not None else self.GENERATION_COSTS.get(gen_type, 0.01)
        key = self._key(guild_id, user_id)

        quota.pollen_spent += cost
        quota.total_gens += 1
        quota.hourly_gens += 1

        is_text = gen_type.startswith("text") or gen_type in ("translation", "emotion_detection", "daily_recap", "dream_journal")
        is_image = gen_type.startswith("image")

        if is_text:
            quota.text_gens += 1
        if is_image:
            quota.image_gens += 1

        self._increment_gen_count(key, gen_type)

        return cost

    def get_status(self, guild_id: int, user_id: int, relationship_tier: str = "stranger") -> dict:
        """Get a user's quota status for display."""
        quota = self.get_quota(guild_id, user_id, relationship_tier)
        return {
            "pollen_used": round(quota.pollen_spent, 4),
            "pollen_budget": round(quota.pollen_budget, 1),
            "pollen_remaining": round(max(0, quota.pollen_budget - quota.pollen_spent), 4),
            "text_gens": quota.text_gens,
            "text_limit": quota.max_text_per_day,
            "image_gens": quota.image_gens,
            "image_limit": quota.max_image_per_day,
            "hourly_gens": quota.hourly_gens,
            "hourly_limit": quota.max_gens_per_hour,
            "tier": relationship_tier,
        }
