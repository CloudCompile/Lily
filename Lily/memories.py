#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Memory System

Short-term, long-term, and daily recap memories.
v9.0: Cross-server memories — Lily carries memories from ALL servers.
Dream journal — she writes dreams and can share them.
She remembers what matters, forgets what doesn't, and writes a diary every night.
"""

from __future__ import annotations
import json
import random
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field


# ── Memory Types ──────────────────────────────────────────

@dataclass
class Memory:
    """A single memory entry."""
    content: str
    memory_type: str          # "short_term", "long_term", "episodic", "recap", "dream"
    guild_id: str = "0"
    user_id: str = "0"
    emotion: str = "neutral"  # What emotion was attached to this memory
    importance: float = 0.5   # 0.0 to 1.0 — how important this memory is
    created_at: str = ""
    last_accessed: str = ""
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    is_global: bool = True    # v9.0: Cross-server by default

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_accessed:
            self.last_accessed = now


class MemorySystem:
    """Lily's memory system. The thing that makes her feel real.
    v9.0: Cross-server — she carries memories from ALL servers."""

    # How many memories of each type to keep per user
    MEMORY_LIMITS = {
        "short_term": 25,    # Recent conversation context
        "long_term": 50,     # Important facts and events
        "episodic": 30,      # Specific memorable moments
        "recap": 14,         # Daily recaps (2 weeks worth)
        "dream": 30,         # Dream journal entries
    }

    # What kinds of things are worth remembering long-term
    LONG_TERM_TRIGGERS = [
        "name", "age", "birthday", "favorite", "love", "hate",
        "job", "school", "pet", "family", "friend", "partner",
        "hobby", "project", "goal", "dream", "fear", "struggle",
        "accomplishment", "milestone", "relationship",
    ]

    # What makes something episodic (a memorable moment)
    EPISODIC_TRIGGERS = [
        "first time", "finally", "omg", "i can't believe",
        "best day", "worst day", "so happy", "so sad",
        "big news", "i got", "i passed", "i failed",
        "we did it", "surprise", "i'm so proud",
    ]

    def __init__(self):
        self._memories: Dict[str, List[Memory]] = {}  # key -> list of memories
        self._recaps: Dict[str, List[Memory]] = {}     # key -> daily recaps
        self._dreams: Dict[str, List[Memory]] = {}     # key -> dream journal

    def _key(self, user_id: int | str) -> str:
        """v9.0: Key is just user_id — memories are cross-server."""
        return f"global:{user_id}"

    def add_memory(
        self,
        guild_id: int,
        user_id: int,
        content: str,
        emotion: str = "neutral",
        importance: float = 0.5,
        tags: List[str] = None,
        is_global: bool = True,
    ) -> Memory:
        """Add a memory. Automatically determines the type. Cross-server by default."""
        key = self._key(user_id)

        # Determine memory type
        memory_type = self._classify_memory(content, importance)

        memory = Memory(
            content=content,
            memory_type=memory_type,
            guild_id=str(guild_id),
            user_id=str(user_id),
            emotion=emotion,
            importance=importance,
            tags=tags or [],
            is_global=is_global,
        )

        if key not in self._memories:
            self._memories[key] = []
        self._memories[key].append(memory)

        # Enforce limits
        self._prune_memories(key, memory_type)

        return memory

    def _classify_memory(self, content: str, importance: float) -> str:
        """Determine what type of memory this should be."""
        content_lower = content.lower()

        # Check for long-term triggers
        if any(trigger in content_lower for trigger in self.LONG_TERM_TRIGGERS):
            return "long_term"

        # Check for episodic triggers
        if any(trigger in content_lower for trigger in self.EPISODIC_TRIGGERS):
            return "episodic"

        # High importance = long term
        if importance >= 0.8:
            return "long_term"

        # Default to short-term
        return "short_term"

    def _prune_memories(self, key: str, memory_type: str):
        """Keep memory counts within limits, removing least important first."""
        if key not in self._memories:
            return

        limit = self.MEMORY_LIMITS.get(memory_type, 50)
        typed = [m for m in self._memories[key] if m.memory_type == memory_type]

        if len(typed) > limit:
            # Sort by importance (ascending), then by access count, then by age
            typed.sort(key=lambda m: (m.importance, m.access_count, m.created_at))
            to_remove = set(id(m) for m in typed[:len(typed) - limit])
            self._memories[key] = [m for m in self._memories[key] if id(m) not in to_remove]

    def get_relevant_memories(
        self, guild_id: int, user_id: int, context: str = "", limit: int = 10
    ) -> List[Memory]:
        """Get memories relevant to the current context. Cross-server by default."""
        key = self._key(user_id)
        if key not in self._memories:
            return []

        # Score memories by relevance
        scored = []
        context_lower = context.lower() if context else ""
        context_words = set(context_lower.split())

        for memory in self._memories[key]:
            score = 0.0

            # Base score from importance
            score += memory.importance * 0.3

            # Recency bonus
            try:
                created = datetime.fromisoformat(memory.created_at)
                hours_old = (datetime.now() - created).total_seconds() / 3600
                if hours_old < 1:
                    score += 0.3
                elif hours_old < 24:
                    score += 0.2
                elif hours_old < 72:
                    score += 0.1
            except (ValueError, TypeError):
                pass

            # Keyword overlap
            memory_words = set(memory.content.lower().split())
            overlap = context_words & memory_words
            if overlap:
                score += len(overlap) * 0.1

            # Tag overlap
            if memory.tags:
                tag_overlap = set(t.lower() for t in memory.tags) & context_words
                if tag_overlap:
                    score += len(tag_overlap) * 0.15

            # Type priority
            if memory.memory_type == "long_term":
                score += 0.2
            elif memory.memory_type == "episodic":
                score += 0.15
            elif memory.memory_type == "recap":
                score += 0.1
            elif memory.memory_type == "dream":
                score += 0.05

            # Access count (popular memories are more relevant)
            score += min(memory.access_count * 0.02, 0.2)

            scored.append((score, memory))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top results, mark as accessed
        results = []
        for score, memory in scored[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.now().isoformat()
            results.append(memory)

        return results

    def get_memories_for_prompt(
        self, guild_id: int, user_id: int, context: str = ""
    ) -> str:
        """Format memories for inclusion in the system prompt. Cross-server."""
        memories = self.get_relevant_memories(guild_id, user_id, context, limit=8)

        if not memories:
            return ""

        sections = []

        # Long-term memories (things she knows)
        long_term = [m for m in memories if m.memory_type == "long_term"]
        if long_term:
            sections.append("Things you remember about this person:")
            for m in long_term[:5]:
                sections.append(f"  - {m.content}")

        # Episodic memories (memorable moments)
        episodic = [m for m in memories if m.memory_type == "episodic"]
        if episodic:
            sections.append("Memorable moments with them:")
            for m in episodic[:3]:
                sections.append(f"  - {m.content}")

        # Recent context (short-term)
        short_term = [m for m in memories if m.memory_type == "short_term"]
        if short_term:
            sections.append("Recent things you've talked about:")
            for m in short_term[:4]:
                sections.append(f"  - {m.content}")

        return "\n".join(sections)

    def get_dreams_for_prompt(
        self, user_id: int, count: int = 2
    ) -> str:
        """Format recent dreams for inclusion in the system prompt."""
        key = self._key(user_id)
        if key not in self._dreams or not self._dreams[key]:
            return ""

        recent = self._dreams[key][-count:]
        sections = ["Your recent dreams:"]
        for d in recent:
            sections.append(f"  - {d.content[:150]}")

        return "\n".join(sections)

    def generate_daily_recap(
        self, guild_id: int, user_id: int, conversations: List[Dict], facts: List[Dict]
    ) -> Optional[str]:
        """Generate a daily recap — Lily's diary entry about the day."""
        if not conversations and not facts:
            return None

        # Build a summary of the day's interactions
        topics_discussed = set()
        emotions_felt = set()
        key_moments = []
        new_facts = []

        for conv in conversations:
            if conv.get("emotion") and conv["emotion"] != "neutral":
                emotions_felt.add(conv["emotion"])
            if conv.get("topic"):
                topics_discussed.add(conv["topic"])
            if conv.get("role") == "user":
                content = conv.get("content", "")
                if len(content) > 50:  # Longer messages = more meaningful
                    key_moments.append(content[:100])

        for fact in facts:
            new_facts.append(f"{fact.get('category', 'general')}: {fact.get('fact', '')}")

        # Build the recap
        recap_parts = []
        today = datetime.now().strftime("%A, %B %d")

        recap_parts.append(f"Daily recap for {today}:")

        if topics_discussed:
            recap_parts.append(f"  Topics: {', '.join(list(topics_discussed)[:5])}")
        if emotions_felt:
            recap_parts.append(f"  Their mood: {', '.join(list(emotions_felt)[:3])}")
        if new_facts:
            recap_parts.append(f"  Learned: {'; '.join(new_facts[:5])}")
        if key_moments:
            recap_parts.append(f"  Key moments: {key_moments[0][:80]}")

        recap = "\n".join(recap_parts)

        # Store as a recap memory
        key = self._key(user_id)
        recap_memory = Memory(
            content=recap,
            memory_type="recap",
            guild_id=str(guild_id),
            user_id=str(user_id),
            emotion="reflective",
            importance=0.7,
            tags=["daily_recap", today],
        )

        if key not in self._recaps:
            self._recaps[key] = []
        self._recaps[key].append(recap_memory)

        # Keep only 2 weeks of recaps
        if len(self._recaps[key]) > 14:
            self._recaps[key] = self._recaps[key][-14:]

        return recap

    def add_dream(self, dream_text: str, mood: str = "dreamy", user_id: int = 0) -> Memory:
        """Add a dream journal entry."""
        key = self._key(user_id)
        dream = Memory(
            content=dream_text,
            memory_type="dream",
            guild_id="0",
            user_id=str(user_id),
            emotion=mood,
            importance=0.6,
            tags=["dream_journal", mood],
        )

        if key not in self._dreams:
            self._dreams[key] = []
        self._dreams[key].append(dream)

        # Keep only 30 dreams
        if len(self._dreams[key]) > 30:
            self._dreams[key] = self._dreams[key][-30:]

        return dream

    def get_recent_recaps(self, guild_id: int, user_id: int, count: int = 3) -> List[str]:
        """Get recent daily recaps for context."""
        key = self._key(user_id)
        if key not in self._recaps:
            return []
        return [m.content for m in self._recaps[key][-count:]]

    def get_recent_dreams(self, user_id: int = 0, count: int = 3) -> List[Memory]:
        """Get recent dream journal entries."""
        key = self._key(user_id)
        if key not in self._dreams:
            return []
        return self._dreams[key][-count:]

    def should_forget(self, memory: Memory) -> bool:
        """Simulate realistic forgetting. Some things slip away."""
        if memory.memory_type == "long_term" and memory.importance >= 0.8:
            return False  # Never forget important long-term memories
        if memory.memory_type == "dream":
            return False  # Dreams are never fully forgotten
        if memory.access_count > 5:
            return False  # Frequently accessed memories stay

        # Age-based forgetting
        try:
            created = datetime.fromisoformat(memory.created_at)
            days_old = (datetime.now() - created).days
            if days_old > 7 and memory.memory_type == "short_term":
                # 30% chance to forget short-term memories older than a week
                return random.random() < 0.3
            if days_old > 30 and memory.importance < 0.3:
                # 50% chance to forget unimportant stuff after a month
                return random.random() < 0.5
        except (ValueError, TypeError):
            pass

        return False

    def to_dict_list(self, memories: List[Memory]) -> List[dict]:
        """Serialize memories for database storage."""
        return [
            {
                "content": m.content,
                "memory_type": m.memory_type,
                "guild_id": m.guild_id,
                "user_id": m.user_id,
                "emotion": m.emotion,
                "importance": m.importance,
                "created_at": m.created_at,
                "last_accessed": m.last_accessed,
                "access_count": m.access_count,
                "tags": m.tags,
                "is_global": m.is_global,
            }
            for m in memories
        ]

    def from_dict_list(self, data: List[dict]) -> List[Memory]:
        """Deserialize memories from database."""
        return [
            Memory(
                content=d.get("content", ""),
                memory_type=d.get("memory_type", "short_term"),
                guild_id=d.get("guild_id", "0"),
                user_id=d.get("user_id", "0"),
                emotion=d.get("emotion", "neutral"),
                importance=d.get("importance", 0.5),
                created_at=d.get("created_at", ""),
                last_accessed=d.get("last_accessed", ""),
                access_count=d.get("access_count", 0),
                tags=d.get("tags", []),
                is_global=d.get("is_global", True),
            )
            for d in data
        ]
