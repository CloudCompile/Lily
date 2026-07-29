#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.5 — Multi-Server Database Module

SQLite database with per-guild configuration support.
v8.5: Cross-server memories (global by user_id), dream journal, mood status tracking.
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
    """Thread-safe multi-server SQLite database for Lily v8.5."""

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
                text_model           TEXT DEFAULT 'openai-fast',
                image_model          TEXT DEFAULT 'sana',
                safe_mode            TEXT DEFAULT 'privacy,secrets',
                reply_chance         REAL DEFAULT 0.25,
                reaction_chance      REAL DEFAULT 0.40,
                spontaneous_chance   REAL DEFAULT 0.02,
                proactive_dm_enabled INTEGER DEFAULT 1,
                personality_enabled  INTEGER DEFAULT 1,
                daily_recap_enabled  INTEGER DEFAULT 1,
                dream_journal_enabled INTEGER DEFAULT 1,
                language             TEXT DEFAULT 'en',
                created_at           TEXT DEFAULT (datetime('now')),
                updated_at           TEXT DEFAULT (datetime('now'))
            );

            -- Per-user conversation memory (per-guild, for context)
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

            -- Learned facts about users (CROSS-SERVER — global by user_id)
            CREATE TABLE IF NOT EXISTS user_facts (
                user_id        TEXT NOT NULL,
                guild_id       TEXT NOT NULL DEFAULT '0',
                category       TEXT NOT NULL,
                fact           TEXT NOT NULL,
                confidence     REAL DEFAULT 0.5,
                last_mentioned TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, category, fact)
            );
            CREATE INDEX IF NOT EXISTS idx_facts_user
                ON user_facts(user_id);

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

            -- v8.5: Per-user relationships (CROSS-SERVER — global by user_id)
            CREATE TABLE IF NOT EXISTS relationships (
                user_id               TEXT NOT NULL,
                guild_id              TEXT NOT NULL DEFAULT '0',
                affection             REAL DEFAULT 0.0,
                trust                 REAL DEFAULT 0.0,
                familiarity           REAL DEFAULT 0.0,
                annoyance             REAL DEFAULT 0.0,
                total_interactions    INTEGER DEFAULT 0,
                positive_interactions INTEGER DEFAULT 0,
                negative_interactions INTEGER DEFAULT 0,
                last_interaction      TEXT DEFAULT NULL,
                last_proactive_dm     TEXT DEFAULT NULL,
                first_met             TEXT DEFAULT NULL,
                relationship_tier     TEXT DEFAULT 'stranger',
                private_notes         TEXT DEFAULT '[]',
                PRIMARY KEY (user_id, guild_id)
            );
            CREATE INDEX IF NOT EXISTS idx_rel_user
                ON relationships(user_id);

            -- v8.5: Long-term memories (CROSS-SERVER — global by user_id)
            -- Memories carry across ALL servers. Lily remembers you everywhere.
            CREATE TABLE IF NOT EXISTS memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT NOT NULL,
                guild_id      TEXT NOT NULL DEFAULT '0',
                content       TEXT NOT NULL,
                memory_type   TEXT DEFAULT 'short_term',
                emotion       TEXT DEFAULT 'neutral',
                importance    REAL DEFAULT 0.5,
                tags          TEXT DEFAULT '[]',
                is_global     INTEGER DEFAULT 1,
                created_at    TEXT DEFAULT (datetime('now')),
                last_accessed TEXT DEFAULT (datetime('now')),
                access_count  INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories(user_id, memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_global
                ON memories(user_id, is_global);

            -- v8.5: Daily recaps (CROSS-SERVER — global by user_id)
            CREATE TABLE IF NOT EXISTS daily_recaps (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                guild_id   TEXT NOT NULL DEFAULT '0',
                recap_text TEXT NOT NULL,
                recap_date TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, recap_date)
            );
            CREATE INDEX IF NOT EXISTS idx_recaps_user
                ON daily_recaps(user_id);

            -- v8.5: Dream Journal (CROSS-SERVER — Lily's dreams are hers everywhere)
            CREATE TABLE IF NOT EXISTS dream_journal (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL DEFAULT '0',
                dream_text  TEXT NOT NULL,
                mood        TEXT DEFAULT 'dreamy',
                inspiration TEXT DEFAULT '',
                is_shared   INTEGER DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_dreams_user
                ON dream_journal(user_id);

            -- Generation history for rate limiting and analytics
            CREATE TABLE IF NOT EXISTS generation_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id    TEXT NOT NULL DEFAULT '0',
                user_id     TEXT NOT NULL,
                endpoint    TEXT NOT NULL,
                model       TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                cost_pollen REAL DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_genlog_guild
                ON generation_log(guild_id, created_at);

            -- v8.5: Generation quotas (per-user, per-guild, per-day)
            CREATE TABLE IF NOT EXISTS generation_quotas (
                guild_id      TEXT NOT NULL DEFAULT '0',
                user_id       TEXT NOT NULL,
                period_date   TEXT NOT NULL,
                pollen_spent  REAL DEFAULT 0,
                pollen_budget REAL DEFAULT 10,
                text_gens     INTEGER DEFAULT 0,
                image_gens    INTEGER DEFAULT 0,
                total_gens    INTEGER DEFAULT 0,
                PRIMARY KEY (guild_id, user_id, period_date)
            );

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

    def get_today_conversations(
        self, guild_id: int, user_id: int
    ) -> List[Dict]:
        """Get today's conversations for the daily recap."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT role, content, emotion, topic, timestamp FROM conversations "
            "WHERE guild_id = ? AND user_id = ? AND timestamp >= date('now') "
            "ORDER BY timestamp ASC",
            (str(guild_id), str(user_id)),
        ).fetchall()
        return [dict(r) for r in rows]

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

    # ── User facts (CROSS-SERVER) ────────────────────────

    def add_fact(
        self, guild_id: int, user_id: int, category: str, fact: str, confidence: float = 0.5
    ) -> None:
        """Store a learned fact about a user (cross-server)."""
        conn = self._conn()
        conn.execute(
            "INSERT INTO user_facts (user_id, guild_id, category, fact, confidence, last_mentioned) "
            "VALUES (?, ?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(user_id, category, fact) DO UPDATE "
            "SET confidence = MAX(confidence, excluded.confidence), last_mentioned = datetime('now')",
            (str(user_id), str(guild_id), category, fact, confidence),
        )
        conn.commit()

    def get_facts(self, guild_id: int, user_id: int, cross_server: bool = True) -> List[Dict]:
        """Get all known facts about a user. Cross-server by default."""
        conn = self._conn()
        if cross_server:
            # Get facts from ALL servers
            rows = conn.execute(
                "SELECT category, fact, confidence, last_mentioned, guild_id FROM user_facts "
                "WHERE user_id = ? ORDER BY confidence DESC",
                (str(user_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT category, fact, confidence, last_mentioned FROM user_facts "
                "WHERE guild_id = ? AND user_id = ? ORDER BY confidence DESC",
                (str(guild_id), str(user_id)),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_facts(self, guild_id: int, user_id: int) -> None:
        """Clear all facts about a user."""
        conn = self._conn()
        conn.execute(
            "DELETE FROM user_facts WHERE user_id = ?",
            (str(user_id),),
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

    # ── v8.5: Relationships (CROSS-SERVER) ──────────────

    def save_relationship(self, guild_id: int, user_id: int, rel_data: dict) -> None:
        """Save a relationship to the database."""
        conn = self._conn()
        conn.execute(
            """INSERT INTO relationships 
               (user_id, guild_id, affection, trust, familiarity, annoyance,
                total_interactions, positive_interactions, negative_interactions,
                last_interaction, last_proactive_dm, first_met, relationship_tier, private_notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, guild_id) DO UPDATE SET
                affection = excluded.affection,
                trust = excluded.trust,
                familiarity = excluded.familiarity,
                annoyance = excluded.annoyance,
                total_interactions = excluded.total_interactions,
                positive_interactions = excluded.positive_interactions,
                negative_interactions = excluded.negative_interactions,
                last_interaction = excluded.last_interaction,
                last_proactive_dm = excluded.last_proactive_dm,
                first_met = excluded.first_met,
                relationship_tier = excluded.relationship_tier,
                private_notes = excluded.private_notes
            """,
            (
                str(user_id), str(guild_id),
                rel_data.get("affection", 0.0),
                rel_data.get("trust", 0.0),
                rel_data.get("familiarity", 0.0),
                rel_data.get("annoyance", 0.0),
                rel_data.get("total_interactions", 0),
                rel_data.get("positive_interactions", 0),
                rel_data.get("negative_interactions", 0),
                rel_data.get("last_interaction"),
                rel_data.get("last_proactive_dm"),
                rel_data.get("first_met"),
                rel_data.get("relationship_tier", "stranger"),
                json.dumps(rel_data.get("private_notes", [])),
            ),
        )
        conn.commit()

    def load_relationship(self, guild_id: int, user_id: int) -> Optional[dict]:
        """Load a relationship from the database (prefers current guild, falls back to any guild)."""
        conn = self._conn()
        # First try the current guild
        row = conn.execute(
            "SELECT * FROM relationships WHERE user_id = ? AND guild_id = ?",
            (str(user_id), str(guild_id)),
        ).fetchone()
        if row is None:
            # Fall back to any guild (cross-server relationship)
            row = conn.execute(
                "SELECT * FROM relationships WHERE user_id = ? ORDER BY total_interactions DESC LIMIT 1",
                (str(user_id),),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["private_notes"] = json.loads(data.get("private_notes", "[]"))
        return data

    def get_all_relationships(self, guild_id: int) -> List[Dict]:
        """Get all relationships in a guild."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM relationships WHERE guild_id = ? ORDER BY affection DESC",
            (str(guild_id),),
        ).fetchall()
        results = []
        for row in rows:
            data = dict(row)
            data["private_notes"] = json.loads(data.get("private_notes", "[]"))
            results.append(data)
        return results

    # ── v8.5: Memories (CROSS-SERVER) ───────────────────

    def save_memory(
        self, guild_id: int, user_id: int, content: str,
        memory_type: str = "short_term", emotion: str = "neutral",
        importance: float = 0.5, tags: List[str] = None,
        is_global: bool = True
    ) -> None:
        """Save a memory to the database (cross-server by default)."""
        conn = self._conn()
        conn.execute(
            "INSERT INTO memories (user_id, guild_id, content, memory_type, emotion, importance, tags, is_global) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(user_id), str(guild_id), content, memory_type, emotion, importance,
             json.dumps(tags or []), 1 if is_global else 0),
        )
        conn.commit()

    def get_memories(
        self, guild_id: int, user_id: int,
        memory_type: str = None, limit: int = 20, cross_server: bool = True
    ) -> List[Dict]:
        """Get memories for a user. Cross-server by default (carries across all guilds)."""
        conn = self._conn()
        if cross_server:
            # Get memories from ALL guilds
            if memory_type:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE user_id = ? AND memory_type = ? "
                    "ORDER BY importance DESC, created_at DESC LIMIT ?",
                    (str(user_id), memory_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE user_id = ? "
                    "ORDER BY importance DESC, created_at DESC LIMIT ?",
                    (str(user_id), limit),
                ).fetchall()
        else:
            # Only this guild
            if memory_type:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE guild_id = ? AND user_id = ? AND memory_type = ? "
                    "ORDER BY importance DESC, created_at DESC LIMIT ?",
                    (str(guild_id), str(user_id), memory_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE guild_id = ? AND user_id = ? "
                    "ORDER BY importance DESC, created_at DESC LIMIT ?",
                    (str(guild_id), str(user_id), limit),
                ).fetchall()
        results = []
        for row in rows:
            data = dict(row)
            data["tags"] = json.loads(data.get("tags", "[]"))
            results.append(data)
        return results

    def update_memory_access(self, memory_id: int) -> None:
        """Update memory access count and timestamp."""
        conn = self._conn()
        conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = datetime('now') "
            "WHERE id = ?",
            (memory_id,),
        )
        conn.commit()

    def prune_old_memories(self, guild_id: int, user_id: int) -> None:
        """Remove old, unimportant short-term memories."""
        conn = self._conn()
        # Keep only 25 short-term memories
        conn.execute(
            "DELETE FROM memories WHERE user_id = ? AND memory_type = 'short_term' "
            "AND id NOT IN "
            "(SELECT id FROM memories WHERE user_id = ? AND memory_type = 'short_term' "
            "ORDER BY importance DESC, created_at DESC LIMIT 25)",
            (str(user_id), str(user_id)),
        )
        # Keep only 50 long-term memories
        conn.execute(
            "DELETE FROM memories WHERE user_id = ? AND memory_type = 'long_term' "
            "AND id NOT IN "
            "(SELECT id FROM memories WHERE user_id = ? AND memory_type = 'long_term' "
            "ORDER BY importance DESC, created_at DESC LIMIT 50)",
            (str(user_id), str(user_id)),
        )
        conn.commit()

    # ── v8.5: Daily Recaps (CROSS-SERVER) ──────────────

    def save_daily_recap(self, guild_id: int, user_id: int, recap_text: str, recap_date: str = None) -> None:
        """Save a daily recap (Lily's diary entry)."""
        if not recap_date:
            recap_date = datetime.now().strftime("%Y-%m-%d")
        conn = self._conn()
        conn.execute(
            "INSERT INTO daily_recaps (user_id, guild_id, recap_text, recap_date) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, recap_date) DO UPDATE SET recap_text = excluded.recap_text",
            (str(user_id), str(guild_id), recap_text, recap_date),
        )
        conn.commit()

    def get_daily_recaps(self, guild_id: int, user_id: int, count: int = 7) -> List[Dict]:
        """Get recent daily recaps (cross-server)."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM daily_recaps WHERE user_id = ? "
            "ORDER BY recap_date DESC LIMIT ?",
            (str(user_id), count),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── v8.5: Dream Journal ─────────────────────────────

    def save_dream(self, dream_text: str, mood: str = "dreamy", 
                   inspiration: str = "", user_id: int = 0) -> None:
        """Save a dream journal entry."""
        conn = self._conn()
        conn.execute(
            "INSERT INTO dream_journal (user_id, dream_text, mood, inspiration) "
            "VALUES (?, ?, ?, ?)",
            (str(user_id), dream_text, mood, inspiration),
        )
        conn.commit()

    def get_dreams(self, user_id: int = 0, count: int = 7) -> List[Dict]:
        """Get recent dream journal entries."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM dream_journal WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (str(user_id), count),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_dream(self, user_id: int = 0) -> Optional[Dict]:
        """Get the most recent dream."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM dream_journal WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (str(user_id),),
        ).fetchone()
        return dict(row) if row else None

    def share_dream(self, dream_id: int) -> None:
        """Mark a dream as shared."""
        conn = self._conn()
        conn.execute(
            "UPDATE dream_journal SET is_shared = 1 WHERE id = ?",
            (dream_id,),
        )
        conn.commit()

    # ── Generation log ───────────────────────────────────

    def log_generation(
        self, guild_id: int, user_id: int, endpoint: str,
        model: str, prompt_hash: str, cost_pollen: float = 0
    ) -> None:
        """Log a generation for rate limiting and analytics."""
        conn = self._conn()
        conn.execute(
            "INSERT INTO generation_log (guild_id, user_id, endpoint, model, prompt_hash, cost_pollen) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(guild_id), str(user_id), endpoint, model, prompt_hash, cost_pollen),
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

    # ── v8.5: Generation Quotas ──────────────────────────

    def get_quota(self, guild_id: int, user_id: int) -> Optional[Dict]:
        """Get today's quota for a user."""
        today = datetime.now().strftime("%Y-%m-%d")
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM generation_quotas WHERE guild_id = ? AND user_id = ? AND period_date = ?",
            (str(guild_id), str(user_id), today),
        ).fetchone()
        return dict(row) if row else None

    def upsert_quota(self, guild_id: int, user_id: int, quota_data: dict) -> None:
        """Update or insert a quota record."""
        today = datetime.now().strftime("%Y-%m-%d")
        conn = self._conn()
        conn.execute(
            """INSERT INTO generation_quotas 
               (guild_id, user_id, period_date, pollen_spent, pollen_budget, text_gens, image_gens, total_gens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(guild_id, user_id, period_date) DO UPDATE SET
                pollen_spent = excluded.pollen_spent,
                pollen_budget = excluded.pollen_budget,
                text_gens = excluded.text_gens,
                image_gens = excluded.image_gens,
                total_gens = excluded.total_gens
            """,
            (
                str(guild_id), str(user_id), today,
                quota_data.get("pollen_spent", 0),
                quota_data.get("pollen_budget", 10),
                quota_data.get("text_gens", 0),
                quota_data.get("image_gens", 0),
                quota_data.get("total_gens", 0),
            ),
        )
        conn.commit()

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
