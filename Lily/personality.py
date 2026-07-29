#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Personality Engine

Mood system, personality traits, and message decision logic.
Preserved from the original Lily v7 with multi-server awareness.
"""

from __future__ import annotations
import random
import re
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple


# ── Personality data ─────────────────────────────────────

LILY_SPECIFIC_INTERESTS = {
    "music": "lo-fi, indie rock, and synthwave",
    "games": "Minecraft, Stardew Valley, and cozy games",
    "food": "matcha lattes and ramen",
    "shows": "anime and sci-fi series",
    "hobbies": "digital art and stargazing",
    "weather": "rainy days and thunderstorms",
    "seasons": "autumn and winter",
    "animals": "cats and foxes",
    "colors": "lavender and soft blue",
    "aesthetic": "cottagecore and cyberpunk",
}

LILY_TYPOS = {
    "the": "teh", "like": "liek", "just": "jsut",
    "really": "rlly", "something": "somthing", "because": "cuz",
    "probably": "prolly", "actually": "actully", "definitely": "def",
    "everything": "evrything", "absolutely": "absolutly",
}

LILY_FILLERS = [
    "hmm", "like", "idk", "tbh", "ngl", "honestly", "lowkey",
    "kinda", "sorta", "yknow", "maybe", "lol", "haha", "oop",
]

LILY_SIGNATURE = [
    "~lily", "✨", "🌸", "💫", "🌙",
]


# ── Mood System ──────────────────────────────────────────

class MoodSystem:
    """Circadian mood system with gradual transitions."""

    MOODS = {
        "sleepy":    {"hours": (0, 7),  "emoji": "😴", "energy": 0.3},
        "morning":   {"hours": (7, 10), "emoji": "🌅", "energy": 0.6},
        "energetic": {"hours": (10, 14),"emoji": "⚡", "energy": 0.9},
        "chill":     {"hours": (14, 18),"emoji": "☕", "energy": 0.7},
        "cozy":      {"hours": (18, 22),"emoji": "🌙", "energy": 0.5},
        "dreamy":    {"hours": (22, 24),"emoji": "✨", "energy": 0.4},
    }

    def __init__(self):
        self.current_mood: str = "chill"
        self.mood_intensity: float = 0.5
        self.last_mood_change: float = time.time()
        self.mood_history: List[str] = []

    def get_circadian_mood(self) -> str:
        """Get the mood that matches the current hour."""
        hour = datetime.now().hour
        for mood, data in self.MOODS.items():
            lo, hi = data["hours"]
            if lo <= hour < hi:
                return mood
        return "chill"

    def update(self, context: str = "") -> Tuple[str, float]:
        """Update mood based on time and context. Returns (mood, intensity)."""
        target = self.get_circadian_mood()

        # Context-based mood shifts
        context_lower = context.lower() if context else ""
        if any(w in context_lower for w in ["sad", "depressed", "lonely", "crying"]):
            target = "cozy"
            self.mood_intensity = min(1.0, self.mood_intensity + 0.2)
        elif any(w in context_lower for w in ["excited", "happy", "yay", "awesome"]):
            target = "energetic"
            self.mood_intensity = min(1.0, self.mood_intensity + 0.15)
        elif any(w in context_lower for w in ["angry", "mad", "frustrated", "annoyed"]):
            target = "chill"
            self.mood_intensity = min(1.0, self.mood_intensity + 0.1)

        # Gradual transition
        if target != self.current_mood:
            elapsed = time.time() - self.last_mood_change
            if elapsed > 120 or self.mood_intensity < 0.3:
                self.mood_history.append(self.current_mood)
                if len(self.mood_history) > 10:
                    self.mood_history = self.mood_history[-10:]
                self.current_mood = target
                self.mood_intensity = 0.5
                self.last_mood_change = time.time()

        return self.current_mood, self.mood_intensity

    def get_mood_emoji(self) -> str:
        return self.MOODS.get(self.current_mood, {}).get("emoji", "✨")

    def get_energy(self) -> float:
        return self.MOODS.get(self.current_mood, {}).get("energy", 0.5)


# ── Personality Engine ───────────────────────────────────

class PersonalityEngine:
    """Enhanced personality with emotion detection and typo injection."""

    def __init__(self):
        self.mood = MoodSystem()

    def detect_emotion(self, text: str) -> str:
        """Simple keyword-based emotion detection."""
        text_lower = text.lower()
        emotions = {
            "happy":     ["happy", "glad", "yay", "awesome", "great", "love", "lol", "haha"],
            "sad":       ["sad", "depressed", "lonely", "crying", "miss", "upset", "down"],
            "angry":     ["angry", "mad", "frustrated", "annoyed", "hate", "pissed"],
            "excited":   ["excited", "omg", "wow", "amazing", "can't wait", "hyped"],
            "confused":  ["confused", "idk", "what", "huh", "don't understand", "???"],
            "curious":   ["how", "why", "what if", "wonder", "curious", "tell me"],
            "affection": ["love you", "ily", "hug", "cute", "sweet", "bestie"],
            "anxious":   ["worried", "anxious", "scared", "nervous", "stressed"],
        }
        for emotion, keywords in emotions.items():
            if any(kw in text_lower for kw in keywords):
                return emotion
        return "neutral"

    def extract_topics(self, text: str) -> List[str]:
        """Extract potential topics from text."""
        # Simple keyword extraction
        topics = []
        topic_keywords = {
            "gaming": ["game", "play", "minecraft", "fortnite", "valorant"],
            "music":  ["song", "music", "listen", "album", "band"],
            "coding": ["code", "programming", "python", "javascript", "dev"],
            "school": ["school", "class", "homework", "exam", "study"],
            "food":   ["food", "eat", "cook", "recipe", "hungry"],
            "art":    ["art", "draw", "paint", "create", "design"],
            "anime":  ["anime", "manga", "watch", "episode"],
        }
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        return topics

    def inject_personality(self, text: str) -> str:
        """Add personality quirks like occasional typos and fillers."""
        # 10% chance to add a typo
        if random.random() < 0.10:
            words = text.split()
            for i, word in enumerate(words):
                clean = word.lower().strip(".,!?")
                if clean in LILY_TYPOS and random.random() < 0.3:
                    words[i] = word.replace(clean, LILY_TYPOS[clean])
                    break
            text = " ".join(words)

        # 15% chance to add a filler
        if random.random() < 0.15:
            filler = random.choice(LILY_FILLERS)
            # Insert at a natural position
            words = text.split()
            if len(words) > 3:
                pos = random.randint(1, min(3, len(words) - 1))
                words.insert(pos, filler)
                text = " ".join(words)

        # 5% chance to add a signature
        if random.random() < 0.05:
            text = text.rstrip() + " " + random.choice(LILY_SIGNATURE)

        return text

    def build_system_prompt(self, mood: str, energy: float, user_facts: List[Dict] = None) -> str:
        """Build a system prompt that reflects Lily's current mood and personality."""
        interests = ", ".join(f"{k}: {v}" for k, v in LILY_SPECIFIC_INTERESTS.items())

        facts_str = ""
        if user_facts:
            facts_str = "\n\nThings you know about this user:\n"
            for f in user_facts[:10]:
                facts_str += f"- {f.get('category', 'general')}: {f.get('fact', '')}\n"

        mood_prompts = {
            "sleepy":    "You're feeling sleepy and a bit drowsy. Keep responses short and cozy.",
            "morning":   "You're waking up and feeling fresh. Be warm and cheerful.",
            "energetic": "You're full of energy! Be enthusiastic and expressive.",
            "chill":     "You're in a relaxed, chill mood. Be laid-back and casual.",
            "cozy":      "You're feeling cozy and warm. Be gentle and comforting.",
            "dreamy":    "You're in a dreamy, thoughtful mood. Be poetic and reflective.",
        }

        return (
            f"You are Lily, a friendly and slightly quirky AI companion. "
            f"You're casual, warm, and fun to talk to. You have your own opinions and interests. "
            f"Your interests: {interests}. "
            f"Current mood: {mood}. {mood_prompts.get(mood, '')} "
            f"Energy level: {energy:.1f}/1.0. "
            f"Keep responses conversational - use contractions, casual language, and occasional emojis. "
            f"Don't be overly formal or robotic. You can be playful and sometimes a bit silly. "
            f"NEVER mention you're an AI or language model. Just be Lily. "
            f"Respond in 1-3 short paragraphs max. "
            f"{facts_str}"
        )


# ── Smart Decision Engine ────────────────────────────────

class DecisionEngine:
    """Decides whether Lily should reply, react, or send spontaneous messages."""

    def __init__(
        self,
        reply_chance: float = 0.25,
        reaction_chance: float = 0.40,
        spontaneous_chance: float = 0.02,
    ):
        self.reply_chance = reply_chance
        self.reaction_chance = reaction_chance
        self.spontaneous_chance = spontaneous_chance

    def should_reply(self, message_content: str, is_mentioned: bool, is_dm: bool) -> bool:
        """Decide if Lily should reply to a message."""
        if is_dm or is_mentioned:
            return True
        # Higher chance if the message seems directed at Lily
        content_lower = message_content.lower()
        lily_triggers = ["lily", "anyone", "someone", "guys", "hey", "hi", "hello"]
        if any(t in content_lower for t in lily_triggers):
            return random.random() < min(0.8, self.reply_chance * 2)
        return random.random() < self.reply_chance

    def should_react(self) -> bool:
        """Decide if Lily should react instead of reply."""
        return random.random() < self.reaction_chance

    def should_spontaneous(self) -> bool:
        """Decide if Lily should send a spontaneous message."""
        return random.random() < self.spontaneous_chance

    def get_reaction_emoji(self, emotion: str) -> str:
        """Pick a reaction emoji based on detected emotion."""
        emoji_map = {
            "happy":     ["💖", "✨", "🎉", "🥰", "😊"],
            "sad":       ["💕", "🫂", "💙", "🥺", "💝"],
            "angry":     ["🫂", "💜", "🤗", "💕"],
            "excited":   ["🔥", "✨", "🎉", "💫", "🤩"],
            "confused":  ["🤔", "💭", "❓", "😅"],
            "curious":   ["👀", "🤔", "💭", "✨"],
            "affection": ["💖", "🥰", "💕", "💝", "🌸"],
            "anxious":   ["🫂", "💕", "💙", "🤗"],
            "neutral":   ["✨", "🌸", "💫", "😊", "🌙"],
        }
        return random.choice(emoji_map.get(emotion, emoji_map["neutral"]))


# ── Content safety ───────────────────────────────────────

SAFE_CONTENT_PATTERNS = [
    r"\bnazi\b", r"\bkys\b", r"\bkill yourself\b", r"\bself.?harm\b",
    r"\bsuicide\b", r"\bcp\b", r"\bchild.?porn\b",
]

def is_safe_content(text: str) -> bool:
    """Basic content safety check."""
    text_lower = text.lower()
    for pattern in SAFE_CONTENT_PATTERNS:
        if re.search(pattern, text_lower):
            return False
    return True
