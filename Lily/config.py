#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Configuration Module

Loads settings from environment variables / .env file.
Supports per-guild overrides stored in the database.
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

# ── Default model selections ─────────────────────────────
DEFAULT_TEXT_MODEL: str = os.getenv("DEFAULT_TEXT_MODEL", "openai")
DEFAULT_IMAGE_MODEL: str = os.getenv("DEFAULT_IMAGE_MODEL", "zimage")
DEFAULT_VIDEO_MODEL: str = os.getenv("DEFAULT_VIDEO_MODEL", "veo")
DEFAULT_TTS_MODEL: str = os.getenv("DEFAULT_TTS_MODEL", "elevenlabs")
DEFAULT_TRANSCRIPTION_MODEL: str = os.getenv("DEFAULT_TRANSCRIPTION_MODEL", "whisper")
DEFAULT_3D_MODEL: str = os.getenv("DEFAULT_3D_MODEL", "trellis-2-low")

# ── Personality defaults ─────────────────────────────────
MAX_CONV_MEMORY: int = 50
STM_MESSAGES: int = 15
BASE_REPLY_CHANCE: float = 0.25
REACTION_CHANCE: float = 0.40
SPONTANEOUS_MESSAGE_CHANCE: float = 0.02

# ── Data paths ───────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "lily.db"

# ── Safety ───────────────────────────────────────────────
DEFAULT_SAFE_MODE: str = "privacy,secrets"  # Pollinations safe param default

# ── Validation ───────────────────────────────────────────
if not DISCORD_TOKEN or DISCORD_TOKEN == "YOUR_DISCORD_BOT_TOKEN_HERE":
    print("ERROR: DISCORD_TOKEN is not set. Create a .env file from .env.example")
    raise SystemExit(1)
