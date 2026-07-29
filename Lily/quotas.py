#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.5 — Generation Quota System

Not unlimited willy-nilly generations. Each user gets a daily pollen budget.
Rations are based on relationship tier and server settings.
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
    pollen_budget: float = 100.0  # Daily budget

    # Generation counts
    text_gens: int = 0
    image_gens: int = 0
    total_gens: int = 0

    # Limits
    max_text_per_day: int = 50
    max_image_per_day: int = 20
    max_gens_per_hour: int = 15

    # Hourly tracking
    hourly_gens: int = 0
    hourly_reset: float = 0.0


class QuotaSystem:
    """Manages generation quotas. No unlimited willy-nilly generations."""

    # Base budgets by relationship tier
    TIER_BUDGETS = {
        "rival":        {"pollen": 25,   "text": 10, "image": 5},
        "strained":     {"pollen": 40,   "text": 15, "image": 8},
        "stranger":     {"pollen": 60,   "text": 25, "image": 10},
        "acquaintance": {"pollen": 80,   "text": 30, "image": 12},
        "friend":       {"pollen": 100,  "text": 40, "image": 15},
        "close_friend": {"pollen": 130,  "text": 50, "image": 18},
        "bestie":       {"pollen": 160,  "text": 60, "image": 22},
        "soulmate":     {"pollen": 200,  "text": 80, "image": 25},
    }

    # Cost estimates per generation type (in pollen)
    GENERATION_COSTS = {
        "text_casual":       0.5,    # Cheap model for casual chat
        "text_standard":     2.0,    # Standard model
        "text_premium":      8.0,    # Premium model
        "image_quick":       3.0,    # Quick image gen
        "image_standard":    5.0,    # Standard image gen
        "image_pro":         12.0,   # Pro image gen
        "image_edit":        4.0,    # Image editing
        "translation":       0.3,    # Translation is cheap
        "emotion_detection": 0.1,    # Very cheap
        "daily_recap":       0.5,    # Recap generation
    }

    def __init__(self):
        self._quotas: Dict[str, UserQuota] = {}

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

    def can_generate(
        self, guild_id: int, user_id: int, gen_type: str,
        relationship_tier: str = "stranger"
    ) -> Tuple[bool, str]:
        """Check if a user can generate. Returns (allowed, reason)."""
        quota = self.get_quota(guild_id, user_id, relationship_tier)
        cost = self.GENERATION_COSTS.get(gen_type, 1.0)

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
            return False, f"You've used up your daily pollen budget! ({quota.pollen_spent:.0f}/{quota.pollen_budget:.0f} pollen used today)"

        # Check type-specific limits
        is_text = gen_type.startswith("text") or gen_type in ("translation", "emotion_detection", "daily_recap")
        is_image = gen_type.startswith("image")

        if is_text and quota.text_gens >= quota.max_text_per_day:
            return False, f"You've hit your daily text generation limit ({quota.max_text_per_day}). Try again tomorrow!"

        if is_image and quota.image_gens >= quota.max_image_per_day:
            return False, f"You've hit your daily image generation limit ({quota.max_image_per_day}). Try again tomorrow!"

        return True, ""

    def record_generation(
        self, guild_id: int, user_id: int, gen_type: str,
        relationship_tier: str = "stranger", actual_cost: float = None
    ) -> None:
        """Record that a generation was used."""
        quota = self.get_quota(guild_id, user_id, relationship_tier)
        cost = actual_cost if actual_cost is not None else self.GENERATION_COSTS.get(gen_type, 1.0)

        quota.pollen_spent += cost
        quota.total_gens += 1
        quota.hourly_gens += 1

        is_text = gen_type.startswith("text") or gen_type in ("translation", "emotion_detection", "daily_recap")
        is_image = gen_type.startswith("image")

        if is_text:
            quota.text_gens += 1
        if is_image:
            quota.image_gens += 1

    def get_status(self, guild_id: int, user_id: int, relationship_tier: str = "stranger") -> dict:
        """Get a user's quota status for display."""
        quota = self.get_quota(guild_id, user_id, relationship_tier)
        return {
            "pollen_used": round(quota.pollen_spent, 1),
            "pollen_budget": round(quota.pollen_budget, 1),
            "pollen_remaining": round(max(0, quota.pollen_budget - quota.pollen_spent), 1),
            "text_gens": quota.text_gens,
            "text_limit": quota.max_text_per_day,
            "image_gens": quota.image_gens,
            "image_limit": quota.max_image_per_day,
            "hourly_gens": quota.hourly_gens,
            "hourly_limit": quota.max_gens_per_hour,
            "tier": relationship_tier,
        }
