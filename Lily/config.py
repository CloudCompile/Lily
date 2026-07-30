#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Configuration Module

Loads settings from environment variables / .env file.
Supports per-guild overrides stored in the database.
v9.0: Updated defaults for cheap models (Sana Sprint, openai-fast).
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ── Discord ──────────────────────────────────────────────
DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")

# ── Pollinations API ─────────────────────────────────────
POLLINATIONS_KEY: str = os.getenv("POLLINATIONS_KEY", "")
POLLINATIONS_BASE_URL: str = os.getenv("POLLINATIONS_BASE_URL", "https://gen.pollinations.ai")
POLLINATIONS_MEDIA_URL: str = os.getenv("POLLINATIONS_MEDIA_URL", "https://media.pollinations.ai")

# ── Admin IDs ────────────────────────────────────────────
_raw_admins = os.getenv("ADMIN_IDS", "")
ADMIN_IDS: list[int] = [
    int(x.strip()) for x in _raw_admins.split(",") if x.strip().isdigit()
]

# ── Bot behaviour ────────────────────────────────────────
BOT_PREFIX: str = os.getenv("BOT_PREFIX", "!lily")

# ── Default model selections (v9.0: cheap models by default) ─
# Sana Sprint for images (0.0001/gen — dirt cheap!)
# openai-fast for text (free tier!)
DEFAULT_TEXT_MODEL: str = os.getenv("DEFAULT_TEXT_MODEL", "openai-fast")
DEFAULT_IMAGE_MODEL: str = os.getenv("DEFAULT_IMAGE_MODEL", "sana")

# ── Personality defaults ─────────────────────────────────
MAX_CONV_MEMORY: int = 50
STM_MESSAGES: int = 15
BASE_REPLY_CHANCE: float = 0.25
REACTION_CHANCE: float = 0.40
SPONTANEOUS_MESSAGE_CHANCE: float = 0.02

# ── Proactive DM settings ────────────────────────────────
PROACTIVE_DM_CHECK_INTERVAL: int = int(os.getenv("PROACTIVE_DM_CHECK_INTERVAL", "300"))  # seconds
PROACTIVE_DM_ENABLED: bool = os.getenv("PROACTIVE_DM_ENABLED", "true").lower() == "true"

# ── Daily recap settings ────────────────────────────────
DAILY_RECAP_HOUR: int = int(os.getenv("DAILY_RECAP_HOUR", "23"))  # When to generate recaps
DAILY_RECAP_ENABLED: bool = os.getenv("DAILY_RECAP_ENABLED", "true").lower() == "true"

# ── Dream journal settings ───────────────────────────────
DREAM_JOURNAL_ENABLED: bool = os.getenv("DREAM_JOURNAL_ENABLED", "true").lower() == "true"
DREAM_JOURNAL_HOUR: int = int(os.getenv("DREAM_JOURNAL_HOUR", "3"))  # When Lily dreams (3 AM)
DREAM_JOURNAL_MAX_PER_DAY: int = 2  # Max dreams per day

# ── Mood-reactive status ────────────────────────────────
MOOD_STATUS_ENABLED: bool = os.getenv("MOOD_STATUS_ENABLED", "true").lower() == "true"
MOOD_STATUS_INTERVAL: int = int(os.getenv("MOOD_STATUS_INTERVAL", "300"))  # Update every 5 min

# ── Data paths ───────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "lily.db"

# ── Safety ───────────────────────────────────────────────
DEFAULT_SAFE_MODE: str = "privacy,secrets"

# ── Bot-to-bot interaction ────────────────────────────────
BOT_PARTNER_ID: int = int(os.getenv("BOT_PARTNER_ID", "0")) if os.getenv("BOT_PARTNER_ID", "").isdigit() else 0

# ── Validation ───────────────────────────────────────────
if not DISCORD_TOKEN or DISCORD_TOKEN == "YOUR_DISCORD_BOT_TOKEN_HERE":
    print("WARNING: DISCORD_TOKEN is not set. Create a .env file from .env.example")
