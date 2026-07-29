#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Multi-Server Database Module

SQLite database with per-guild configuration support.
Each guild can have its own settings, channels, models, and preferences.
"""

from __future__ import annotations
import sqlite3
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from config import DB_PATH, DATA_DIR


class Database:
    """Thread-safe multi-server SQLite database for Lily."""

    def __init__(self, db_path: Path = DB_PATH):
        DATA_DIR.mkdir(exist_ok=True)
        self.db_path = str(db_path)
        self._local = threading.local()
        self._init_tables()

    # ── Connection management ────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_tables(self):
        conn = self._conn()
        conn.executescript("""
            -- Guild-specific settings (multi-server core)
            CREATE TABLE IF NOT EXISTS guild_settings (
                guild_id  TEXT PRIMARY KEY,
                prefix               TEXT DEFAULT '!lily',
                allowed_channel      TEXT DEFAULT NULL,
                text_model           TEXT DEFAULT 'openai',
                image_model          TEXT DEFAULT 'zimage',
                video_model          TEXT DEFAULT 'veo',
                tts_model            TEXT DEFAULT 'elevenlabs',
                transcription_model  TEXT DEFAULT 'whisper',
                model_3d             TEXT DEFAULT 'trellis-2-low',
                safe_mode            TEXT DEFAULT 'privacy,secrets',
                reply_chance         REAL DEFAULT 0.25,
                reaction_chance      REAL DEFAULT 0.40,
                spontaneous_chance   REAL DEFAULT 0.02,
                personality_enabled  INTEGER DEFAULT 1,
                language             TEXT DEFAULT 'en',
                created_at           TEXT DEFAULT (datetime('now')),
                updated_at           TEXT DEFAULT (datetime('now'))
            );

            -- Per-user conversation memory (per-guild)
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id   TEXT NOT NULL DEFAULT '0',
                user_id    TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                emotion    TEXT DEFAULT NULL,
                topic      TEXT DEFAULT NULL,
                timestamp  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_conv_guild_user
                ON conversations(guild_id, user_id, timestamp);

            -- Learned facts about users (per-guild)
            CREATE TABLE IF NOT EXISTS user_facts (
                guild_id       TEXT NOT NULL DEFAULT '0',
                user_id        TEXT NOT NULL,
                category       TEXT NOT NULL,
                fact           TEXT NOT NULL,
                confidence     REAL DEFAULT 0.5,
                last_mentioned TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (guild_id, user_id, category, fact)
            );

            -- Recurring conversation topics (per-guild)
            CREATE TABLE IF NOT EXISTS conversation_topics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id        TEXT NOT NULL DEFAULT '0',
                user_id         TEXT NOT NULL,
                topic           TEXT NOT NULL,
                mentioned_count INTEGER DEFAULT 1,
                last_mentioned  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_topics_guild_user
                ON conversation_topics(guild_id, user_id);

            -- Intentional "forgetting" for realism
            CREATE TABLE IF NOT EXISTS lily_memory_gaps (
                guild_id        TEXT NOT NULL DEFAULT '0',
                user_id         TEXT NOT NULL,
                forgotten_detail TEXT NOT NULL,
                timestamp       TEXT DEFAULT (datetime('now'))
            );

            -- Generation history for rate limiting and analytics
            CREATE TABLE IF NOT EXISTS generation_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id    TEXT NOT NULL DEFAULT '0',
                user_id     TEXT NOT NULL,
                endpoint    TEXT NOT NULL,
                model       TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                cost_usd    REAL DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_genlog_guild
                ON generation_log(guild_id, created_at);

            -- Global key-value store for bot-wide settings
            CREATE TABLE IF NOT EXISTS global_settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

    # ── Guild settings ───────────────────────────────────

    def get_guild_settings(self, guild_id: int) -> Dict[str, Any]:
        """Get all settings for a guild, creating defaults if new."""
        gid = str(guild_id)
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM guild_settings WHERE guild_id = ?", (gid,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO guild_settings (guild_id) VALUES (?)", (gid,)
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM guild_settings WHERE guild_id = ?", (gid,)
            ).fetchone()
        return dict(row)

    def set_guild_setting(self, guild_id: int, key: str, value: Any) -> None:
        """Set a single guild setting."""
        gid = str(guild_id)
        conn = self._conn()
        # Ensure the guild exists
        conn.execute(
            "INSERT OR IGNORE INTO guild_settings (guild_id) VALUES (?)", (gid,)
        )
        conn.execute(
            f"UPDATE guild_settings SET {key} = ?, updated_at = datetime('now') WHERE guild_id = ?",
            (value, gid),
        )
        conn.commit()

    def get_guild_setting(self, guild_id: int, key: str, default: Any = None) -> Any:
        """Get a single guild setting value."""
        settings = self.get_guild_settings(guild_id)
        return settings.get(key, default)

    # ── Conversations ────────────────────────────────────

    def add_conversation(
        self, guild_id: int, user_id: int, role: str,
        content: str, emotion: str = None, topic: str = None
    ) -> None:
        """Store a conversation message."""
        conn = self._conn()
        conn.execute(
            "INSERT INTO conversations (guild_id, user_id, role, content, emotion, topic) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(guild_id), str(user_id), role, content, emotion, topic),
        )
        conn.commit()
        # Prune old messages per user per guild
        conn.execute(
            "DELETE FROM conversations WHERE guild_id = ? AND user_id = ? AND id NOT IN "
            "(SELECT id FROM conversations WHERE guild_id = ? AND user_id = ? "
            "ORDER BY timestamp DESC LIMIT 50)",
            (str(guild_id), str(user_id), str(guild_id), str(user_id)),
        )
        conn.commit()

    def get_conversations(
        self, guild_id: int, user_id: int, limit: int = 15
    ) -> List[Dict]:
        """Get recent conversation messages for a user in a guild."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT role, content, emotion, topic, timestamp FROM conversations "
            "WHERE guild_id = ? AND user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (str(guild_id), str(user_id), limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def clear_conversations(self, guild_id: int, user_id: int = None) -> None:
        """Clear conversation history for a user (or all users in a guild)."""
        conn = self._conn()
        if user_id:
            conn.execute(
                "DELETE FROM conversations WHERE guild_id = ? AND user_id = ?",
                (str(guild_id), str(user_id)),
            )
        else:
            conn.execute(
                "DELETE FROM conversations WHERE guild_id = ?", (str(guild_id),)
            )
        conn.commit()

    # ── User facts ───────────────────────────────────────

    def add_fact(
        self, guild_id: int, user_id: int, category: str, fact: str, confidence: float = 0.5
    ) -> None:
        """Store a learned fact about a user."""
        conn = self._conn()
        conn.execute(
            "INSERT INTO user_facts (guild_id, user_id, category, fact, confidence, last_mentioned) "
            "VALUES (?, ?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(guild_id, user_id, category, fact) DO UPDATE "
            "SET confidence = MAX(confidence, excluded.confidence), last_mentioned = datetime('now')",
            (str(guild_id), str(user_id), category, fact, confidence),
        )
        conn.commit()

    def get_facts(self, guild_id: int, user_id: int) -> List[Dict]:
        """Get all known facts about a user in a guild."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT category, fact, confidence, last_mentioned FROM user_facts "
            "WHERE guild_id = ? AND user_id = ? ORDER BY confidence DESC",
            (str(guild_id), str(user_id)),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_facts(self, guild_id: int, user_id: int) -> None:
        """Clear all facts about a user in a guild."""
        conn = self._conn()
        conn.execute(
            "DELETE FROM user_facts WHERE guild_id = ? AND user_id = ?",
            (str(guild_id), str(user_id)),
        )
        conn.commit()

    # ── Topics ───────────────────────────────────────────

    def add_topic(self, guild_id: int, user_id: int, topic: str) -> None:
        """Track a recurring conversation topic."""
        conn = self._conn()
        conn.execute(
            "INSERT INTO conversation_topics (guild_id, user_id, topic, mentioned_count, last_mentioned) "
            "VALUES (?, ?, ?, 1, datetime('now')) "
            "ON CONFLICT DO UPDATE SET mentioned_count = mentioned_count + 1, last_mentioned = datetime('now')",
            (str(guild_id), str(user_id), topic),
        )
        conn.commit()

    def get_topics(self, guild_id: int, user_id: int) -> List[Dict]:
        """Get recurring topics for a user in a guild."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT topic, mentioned_count, last_mentioned FROM conversation_topics "
            "WHERE guild_id = ? AND user_id = ? ORDER BY mentioned_count DESC",
            (str(guild_id), str(user_id)),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Generation log ───────────────────────────────────

    def log_generation(
        self, guild_id: int, user_id: int, endpoint: str,
        model: str, prompt_hash: str, cost_usd: float = 0
    ) -> None:
        """Log a generation for rate limiting and analytics."""
        conn = self._conn()
        conn.execute(
            "INSERT INTO generation_log (guild_id, user_id, endpoint, model, prompt_hash, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(guild_id), str(user_id), endpoint, model, prompt_hash, cost_usd),
        )
        conn.commit()

    def get_generation_count(
        self, guild_id: int, user_id: int = None, hours: int = 1
    ) -> int:
        """Get generation count for rate limiting."""
        conn = self._conn()
        if user_id:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM generation_log "
                "WHERE guild_id = ? AND user_id = ? AND created_at >= datetime('now', ?)",
                (str(guild_id), str(user_id), f"-{hours} hours"),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM generation_log "
                "WHERE guild_id = ? AND created_at >= datetime('now', ?)",
                (str(guild_id), f"-{hours} hours"),
            ).fetchone()
        return row["cnt"] if row else 0

    # ── Global settings ──────────────────────────────────

    def get_global(self, key: str, default: str = None) -> Optional[str]:
        conn = self._conn()
        row = conn.execute(
            "SELECT value FROM global_settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_global(self, key: str, value: str) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO global_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()
