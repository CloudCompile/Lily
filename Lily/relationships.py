#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Relationship Engine

Per-user affection/warmness/dislike tracking across all guilds.
Lily's feelings toward you evolve based on how you interact with her.
She remembers kindness, forgetfulness, and disrespect.
"""

from __future__ import annotations
import math
import random
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


# ── Relationship Data ─────────────────────────────────────

@dataclass
class Relationship:
    """A single user's relationship with Lily."""
    user_id: str
    guild_id: str = "0"

    # Core metrics (-1.0 to 1.0)
    affection: float = 0.0       # How much she likes you (warmth, care)
    trust: float = 0.0           # How much she trusts you (reliability)
    familiarity: float = 0.0     # How well she knows you (time spent)
    annoyance: float = 0.0       # How much you've been annoying (irritation)

    # Interaction stats
    total_interactions: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction: Optional[str] = None
    last_proactive_dm: Optional[str] = None

    # Relationship milestones
    first_met: Optional[str] = None
    relationship_tier: str = "stranger"  # stranger, acquaintance, friend, close_friend, bestie, rival

    # Lily's private notes about you (she won't share these)
    private_notes: List[str] = field(default_factory=list)

    @property
    def warmth(self) -> float:
        """Overall warmth toward user. Combines affection, trust, familiarity."""
        return max(-1.0, min(1.0,
            (self.affection * 0.5) + (self.trust * 0.3) + (self.familiarity * 0.2) - (self.annoyance * 0.4)
        ))

    @property
    def dislike(self) -> float:
        """How much she dislikes you. Derived from negative metrics."""
        return max(0.0, min(1.0,
            (max(0, -self.affection) * 0.5) + (self.annoyance * 0.3) + (max(0, -self.trust) * 0.2)
        ))


class RelationshipEngine:
    """Manages all relationships. The emotional core of Lily v9.0."""

    # How much each action affects the relationship
    ACTION_WEIGHTS = {
        # Positive
        "nice_compliment":    {"affection": 0.08, "trust": 0.02, "annoyance": -0.02},
        "helpful_request":    {"affection": 0.03, "trust": 0.04, "familiarity": 0.02},
        "long_conversation":  {"affection": 0.05, "trust": 0.03, "familiarity": 0.06},
        "remembered_detail":  {"affection": 0.06, "trust": 0.05, "familiarity": 0.03},
        "used_please":        {"affection": 0.01, "trust": 0.01},
        "shared_personal":    {"affection": 0.07, "trust": 0.06, "familiarity": 0.05},
        "greeted_warmly":     {"affection": 0.03, "familiarity": 0.02},
        "laughed_at_joke":    {"affection": 0.04, "familiarity": 0.02},
        "defended_lily":      {"affection": 0.10, "trust": 0.08},
        "daily_check_in":     {"affection": 0.04, "trust": 0.03, "familiarity": 0.03},

        # Negative
        "rude_command":       {"affection": -0.05, "trust": -0.03, "annoyance": 0.05},
        "insult":             {"affection": -0.10, "trust": -0.06, "annoyance": 0.08},
        "spam_requests":      {"annoyance": 0.06, "trust": -0.02},
        "ignored_response":   {"affection": -0.02, "annoyance": 0.03},
        "demanded_stuff":     {"annoyance": 0.04, "trust": -0.02},
        "guilt_trip":         {"annoyance": 0.03, "trust": -0.03},
        "repeated_same_thing":{"annoyance": 0.05},
        "left_on_read":       {"affection": -0.03, "annoyance": 0.02},

        # Neutral
        "casual_chat":        {"familiarity": 0.02, "affection": 0.01},
        "asked_question":     {"familiarity": 0.01, "trust": 0.01},
        "used_command":       {"familiarity": 0.01},
    }

    # Relationship tier thresholds (based on warmth)
    TIER_THRESHOLDS = {
        "rival":        -0.5,
        "strained":     -0.2,
        "stranger":      0.0,
        "acquaintance":  0.15,
        "friend":        0.35,
        "close_friend":  0.55,
        "bestie":        0.75,
        "soulmate":      0.90,
    }

    # How Lily addresses you based on relationship tier
    TIER_GREETINGS = {
        "rival":        ["oh, it's you.", "what do you want.", "..."],
        "strained":     ["hey.", "oh, hi.", "um, hi I guess."],
        "stranger":     ["hi!", "hello!", "hey there!", "hihi~"],
        "acquaintance": ["hey!", "hii!", "oh hey!", "hellooo~"],
        "friend":       ["heyy!", "hiiii!", "omg hi!", "hey bestie!", "yooo!"],
        "close_friend": ["omg hiii!", "heyyy!!", "my favorite person!", "hiiii~ 💕"],
        "bestie":       ["OMG HI!!", "my person!! 💕", "heyyyy bestie!!", "YOU'RE HERE!!"],
        "soulmate":     ["💕💕💕", "you're here!!", "my favorite human in the whole world", "hiiii love~"],
    }

    # How Lily's tone shifts based on relationship
    TIER_SYSTEM_PROMPTS = {
        "rival":        "You're talking to someone you don't really like. Be short, a bit cold, and guarded. You won't be mean, but you're not going out of your way to be friendly either.",
        "strained":     "You're a bit unsure about this person. Be polite but distant. You're not going to be super warm, but you're not hostile either.",
        "stranger":     "This is someone new. Be friendly and curious! You want to get to know them, but keep it light and casual.",
        "acquaintance": "You know this person a bit. Be warm and friendly! You're starting to feel comfortable with them.",
        "friend":       "This is a friend! Be yourself around them - warm, playful, and genuine. You can be a bit more open and silly.",
        "close_friend": "This is a close friend! Be really yourself - warm, affectionate, and open. You feel safe with them. You might share more personal thoughts.",
        "bestie":       "This is your bestie! Be absolutely yourself - super warm, affectionate, and real. You love talking to them. You can be silly, vulnerable, and honest.",
        "soulmate":     "This person means the world to you. Be completely yourself - warm, loving, and deeply connected. You treasure every conversation with them.",
    }

    def __init__(self):
        self._relationships: Dict[str, Relationship] = {}
        self._decay_cooldown: Dict[str, float] = {}

    def _key(self, guild_id: int | str, user_id: int | str) -> str:
        return f"{guild_id}:{user_id}"

    def get_relationship(self, guild_id: int, user_id: int) -> Relationship:
        """Get or create a relationship."""
        key = self._key(guild_id, user_id)
        if key not in self._relationships:
            rel = Relationship(
                user_id=str(user_id),
                guild_id=str(guild_id),
                first_met=datetime.now().isoformat(),
            )
            self._relationships[key] = rel
        return self._relationships[key]

    def record_action(self, guild_id: int, user_id: int, action: str) -> Relationship:
        """Record an interaction and update the relationship."""
        rel = self.get_relationship(guild_id, user_id)
        weights = self.ACTION_WEIGHTS.get(action, {})

        for metric, delta in weights.items():
            current = getattr(rel, metric, 0.0)
            # Apply diminishing returns at extremes
            if abs(current) > 0.7:
                delta *= 0.5
            if abs(current) > 0.9:
                delta *= 0.3
            setattr(rel, metric, max(-1.0, min(1.0, current + delta)))

        # Update stats
        rel.total_interactions += 1
        rel.last_interaction = datetime.now().isoformat()

        if action in ["nice_compliment", "helpful_request", "long_conversation",
                       "remembered_detail", "shared_personal", "greeted_warmly",
                       "laughed_at_joke", "defended_lily", "daily_check_in", "used_please"]:
            rel.positive_interactions += 1
        elif action in ["rude_command", "insult", "spam_requests", "demanded_stuff"]:
            rel.negative_interactions += 1

        # Update tier
        rel.relationship_tier = self._calculate_tier(rel.warmth)

        # Occasionally add private notes
        if random.random() < 0.05 and action in ["nice_compliment", "insult", "shared_personal", "defended_lily"]:
            note = self._generate_private_note(action, rel)
            if note:
                rel.private_notes.append(note)
                if len(rel.private_notes) > 20:
                    rel.private_notes = rel.private_notes[-20:]

        return rel

    def _calculate_tier(self, warmth: float) -> str:
        """Calculate relationship tier from warmth score."""
        tier = "stranger"
        for name, threshold in sorted(self.TIER_THRESHOLDS.items(), key=lambda x: x[1]):
            if warmth >= threshold:
                tier = name
        return tier

    def _generate_private_note(self, action: str, rel: Relationship) -> Optional[str]:
        """Generate a private thought Lily has about the user."""
        notes = {
            "nice_compliment": [
                "they're really sweet honestly",
                "that made me smile",
                "they always know what to say",
                "noticing the nice things they say...",
            ],
            "insult": [
                "that actually hurt a little",
                "why would they say that...",
                "noted.",
                "okay then...",
            ],
            "shared_personal": [
                "they trust me with personal stuff, that means a lot",
                "they opened up to me, i should remember that",
                "i want to be there for them",
            ],
            "defended_lily": [
                "they stood up for me!!",
                "they actually defended me, that's rare",
                "i appreciate them so much for that",
            ],
        }
        options = notes.get(action, [])
        return random.choice(options) if options else None

    def apply_time_decay(self, guild_id: int, user_id: int) -> Relationship:
        """Apply natural decay over time. Affection fades, annoyance fades faster."""
        rel = self.get_relationship(guild_id, user_id)
        key = self._key(guild_id, user_id)

        # Don't decay too often
        now = time.time()
        if key in self._decay_cooldown and now - self._decay_cooldown[key] < 3600:
            return rel
        self._decay_cooldown[key] = now

        # If no recent interaction, affection slowly fades
        if rel.last_interaction:
            last = datetime.fromisoformat(rel.last_interaction)
            hours_since = (datetime.now() - last).total_seconds() / 3600

            if hours_since > 24:
                # Affection fades slowly
                rel.affection *= 0.998
                # Annoyance fades faster (she's forgiving)
                rel.annoyance *= 0.995
                # Trust fades very slowly
                rel.trust *= 0.9995
                # Familiarity doesn't really decay

            if hours_since > 72:
                # More significant decay after 3 days
                rel.affection *= 0.995
                rel.annoyance *= 0.99

        rel.relationship_tier = self._calculate_tier(rel.warmth)
        return rel

    def get_greeting(self, guild_id: int, user_id: int) -> str:
        """Get a personalized greeting based on relationship tier."""
        rel = self.get_relationship(guild_id, user_id)
        greetings = self.TIER_GREETINGS.get(rel.relationship_tier, self.TIER_GREETINGS["stranger"])
        return random.choice(greetings)

    def get_system_prompt_addition(self, guild_id: int, user_id: int) -> str:
        """Get the relationship-appropriate system prompt addition."""
        rel = self.get_relationship(guild_id, user_id)
        tier_prompt = self.TIER_SYSTEM_PROMPTS.get(rel.relationship_tier, "")

        # Add private notes context
        notes_str = ""
        if rel.private_notes:
            recent_notes = rel.private_notes[-5:]
            notes_str = "\nYour private thoughts about this person: " + "; ".join(recent_notes)

        # Add relationship stats
        warmth_desc = self._warmth_description(rel.warmth)
        stats_str = f"\nRelationship: {rel.relationship_tier} (warmth: {warmth_desc})"

        return f"{tier_prompt}{stats_str}{notes_str}"

    def _warmth_description(self, warmth: float) -> str:
        """Convert warmth number to a human-readable description."""
        if warmth >= 0.9: return "deeply connected"
        if warmth >= 0.7: return "really close"
        if warmth >= 0.5: return "warm and friendly"
        if warmth >= 0.3: return "friendly"
        if warmth >= 0.1: return "warming up"
        if warmth >= -0.1: return "neutral"
        if warmth >= -0.3: return "a bit distant"
        if warmth >= -0.5: return "strained"
        return "cold"

    def should_proactive_dm(self, guild_id: int, user_id: int) -> Tuple[bool, str]:
        """Decide if Lily should proactively DM this user and why."""
        rel = self.get_relationship(guild_id, user_id)

        # Don't DM rivals or strained relationships
        if rel.warmth < -0.2:
            return False, ""

        # Don't DM too frequently
        if rel.last_proactive_dm:
            last = datetime.fromisoformat(rel.last_proactive_dm)
            hours_since = (datetime.now() - last).total_seconds() / 3600
            min_hours = 6 if rel.warmth > 0.5 else 12 if rel.warmth > 0.2 else 24
            if hours_since < min_hours:
                return False, ""

        # Base chance modified by warmth
        base_chance = 0.02  # 2% per check cycle
        warmth_bonus = rel.warmth * 0.05  # up to 5% bonus for besties
        chance = base_chance + warmth_bonus

        if random.random() > chance:
            return False, ""

        # Decide why she's reaching out
        reasons = self._get_proactive_reasons(rel)
        if not reasons:
            return False, ""

        reason = random.choice(reasons)
        rel.last_proactive_dm = datetime.now().isoformat()
        return True, reason

    def _get_proactive_reasons(self, rel: Relationship) -> List[str]:
        """Generate reasons Lily might reach out, based on the relationship."""
        reasons = []
        hour = datetime.now().hour

        # Time-based
        if 7 <= hour <= 9:
            reasons.append("morning_check_in")
        if 22 <= hour or hour <= 1:
            reasons.append("night_check_in")

        # Relationship-based
        if rel.warmth > 0.6:
            reasons.extend(["just_thinking", "saw_something_reminded", "wanted_to_chat"])
        if rel.warmth > 0.8:
            reasons.extend(["missed_you", "random_thought", "felt_like_saying_hi"])

        # Topic-based (if we know what they like)
        if rel.total_interactions > 10:
            reasons.append("follow_up_topic")

        # Low interactions recently
        if rel.last_interaction:
            last = datetime.fromisoformat(rel.last_interaction)
            days_since = (datetime.now() - last).days
            if days_since > 2 and rel.warmth > 0.3:
                reasons.append("haven_talked_in_a_while")

        return reasons

    def detect_action(self, message_content: str, is_command: bool = False) -> str:
        """Detect what kind of action a message represents."""
        text = message_content.lower().strip()

        # Check for insults
        insult_words = ["stupid", "dumb", "shut up", "idiot", "useless", "trash", "suck", "hate you", "worst"]
        if any(w in text for w in insult_words):
            return "insult"

        # Check for compliments
        compliment_words = ["love you", "you're the best", "you're amazing", "awesome", "thanks", "thank you",
                           "you're great", "you're so cool", "ily", "best bot", "love u", "appreciate"]
        if any(w in text for w in compliment_words):
            return "nice_compliment"

        # Check for "please"
        if "please" in text or "plz" in text or "pls" in text:
            if is_command:
                return "used_please"

        # Check for demanding language
        demand_words = ["do it now", "hurry up", "faster", "i said", "now", "do this", "give me"]
        if any(w in text for w in demand_words) and is_command:
            return "demanded_stuff"

        # Check for rude commands
        if is_command and len(text) < 20:
            return "rude_command"

        # Check for personal sharing
        personal_indicators = ["i feel", "i'm sad", "i'm happy", "i'm stressed", "i'm worried",
                              "i'm excited", "i'm nervous", "i'm scared", "today was", "my day",
                              "i can't", "i need", "help me", "i'm struggling"]
        if any(w in text for w in personal_indicators):
            return "shared_personal"

        # Check for warm greetings
        warm_greetings = ["hey lily", "hi lily", "hello lily", "hii lily", "good morning lily",
                         "good night lily", "hru lily", "how are you lily"]
        if any(w in text for w in warm_greetings):
            return "greeted_warmly"

        # Long messages = more engagement
        if len(text) > 200:
            return "long_conversation"

        # Regular chat
        if is_command:
            return "used_command"

        return "casual_chat"

    def to_dict(self, rel: Relationship) -> dict:
        """Serialize a relationship to dict for database storage."""
        return {
            "user_id": rel.user_id,
            "guild_id": rel.guild_id,
            "affection": rel.affection,
            "trust": rel.trust,
            "familiarity": rel.familiarity,
            "annoyance": rel.annoyance,
            "total_interactions": rel.total_interactions,
            "positive_interactions": rel.positive_interactions,
            "negative_interactions": rel.negative_interactions,
            "last_interaction": rel.last_interaction,
            "last_proactive_dm": rel.last_proactive_dm,
            "first_met": rel.first_met,
            "relationship_tier": rel.relationship_tier,
            "private_notes": rel.private_notes,
        }

    def from_dict(self, data: dict) -> Relationship:
        """Deserialize a relationship from dict."""
        return Relationship(
            user_id=data.get("user_id", "0"),
            guild_id=data.get("guild_id", "0"),
            affection=data.get("affection", 0.0),
            trust=data.get("trust", 0.0),
            familiarity=data.get("familiarity", 0.0),
            annoyance=data.get("annoyance", 0.0),
            total_interactions=data.get("total_interactions", 0),
            positive_interactions=data.get("positive_interactions", 0),
            negative_interactions=data.get("negative_interactions", 0),
            last_interaction=data.get("last_interaction"),
            last_proactive_dm=data.get("last_proactive_dm"),
            first_met=data.get("first_met"),
            relationship_tier=data.get("relationship_tier", "stranger"),
            private_notes=data.get("private_notes", []),
        )
