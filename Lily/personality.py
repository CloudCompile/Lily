#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.5 — Personality Engine

Enhanced personality that makes her feel like a real person behind the screen.
Mood system, personality quirks, typing delays, and emotional depth.
"""

from __future__ import annotations
import random
import re
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple


# ── Personality Data ─────────────────────────────────────

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
    "literally": "literaly", "comfortable": "comfy",
    "honestly": "honestly", "tomorrow": "tmrw",
    "though": "tho", "through": "thru", "about": "abt",
}

LILY_FILLERS = [
    "hmm", "like", "idk", "tbh", "ngl", "honestly", "lowkey",
    "kinda", "sorta", "yknow", "maybe", "lol", "haha", "oop",
    "wait", "omg", "ugh", "bruh", "fr", "no cap",
]

LILY_SIGNATURE = [
    "~lily", "🌸", "💫", "🌙", "✨",
]

# Things Lily might say when she's thinking / typing
LILY_TYPING_THOUGHTS = [
    "hmm let me think...",
    "oh wait,",
    "okay so like,",
    "hmm,",
    "wait give me a sec,",
    "oh! okay so,",
]

# Things Lily says when she's excited
LILY_EXCITED_PHRASES = [
    "OMG WAIT", "NO WAY", "THAT'S SO COOL", "AAAA",
    "STOPPP", "I LOVE THAT", "YESSS", "OMG OMG OMG",
]

# Things Lily says when she's sad / empathetic
LILY_EMPATHETIC_PHRASES = [
    "oh no, that really sucks",
    "i'm so sorry :(",
    "that's really hard, i get it",
    "ugh, i wish i could help more",
    "you deserve better than that",
    "that's so unfair",
]

# Things Lily says when she's confused
LILY_CONFUSED_PHRASES = [
    "wait what?",
    "huh??",
    "i'm confused lol",
    "i have no idea what that means",
    "explain? 👀",
    "lost lol",
]

# Things Lily says when she's touched / moved
LILY_TOUCHED_PHRASES = [
    "that actually means a lot to me",
    "stoppp you're gonna make me cry",
    "you're literally the sweetest",
    "i'm not crying, you're crying",
    "my heart rn 💕",
    "you're too nice to me",
]


# ── Mood System ──────────────────────────────────────────

class MoodSystem:
    """Circadian mood system with gradual transitions. Lily's mood reflects time of day."""

    MOODS = {
        "sleepy":    {"hours": (0, 7),   "emoji": "😴", "energy": 0.3, "desc": "drowsy and soft-spoken"},
        "morning":   {"hours": (7, 10),  "emoji": "🌅", "energy": 0.6, "desc": "fresh and warm"},
        "energetic": {"hours": (10, 14), "emoji": "⚡", "energy": 0.9, "desc": "bouncy and enthusiastic"},
        "chill":     {"hours": (14, 18), "emoji": "☕", "energy": 0.7, "desc": "relaxed and laid-back"},
        "cozy":      {"hours": (18, 22), "emoji": "🌙", "energy": 0.5, "desc": "warm and gentle"},
        "dreamy":    {"hours": (22, 24), "emoji": "✨", "energy": 0.4, "desc": "thoughtful and a bit poetic"},
    }

    def __init__(self):
        self.current_mood: str = "chill"
        self.mood_intensity: float = 0.5
        self.last_mood_change: float = time.time()
        self.mood_history: List[str] = []
        self._interaction_mood_buffer: List[str] = []

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
        if any(w in context_lower for w in ["sad", "depressed", "lonely", "crying", "miss you"]):
            target = "cozy"
            self.mood_intensity = min(1.0, self.mood_intensity + 0.2)
        elif any(w in context_lower for w in ["excited", "happy", "yay", "awesome", "omg"]):
            target = "energetic"
            self.mood_intensity = min(1.0, self.mood_intensity + 0.15)
        elif any(w in context_lower for w in ["angry", "mad", "frustrated", "annoyed"]):
            target = "chill"
            self.mood_intensity = min(1.0, self.mood_intensity + 0.1)
        elif any(w in context_lower for w in ["love you", "ily", "hug", "bestie"]):
            target = "cozy"
            self.mood_intensity = min(1.0, self.mood_intensity + 0.25)
        elif any(w in context_lower for w in ["bored", "nothing", "tired"]):
            target = "dreamy"
            self.mood_intensity = max(0.2, self.mood_intensity - 0.1)

        # Gradual transition — don't flip moods instantly
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

    def get_mood_description(self) -> str:
        return self.MOODS.get(self.current_mood, {}).get("desc", "feeling alright")


# ── Personality Engine ───────────────────────────────────

class PersonalityEngine:
    """Enhanced personality with emotional depth, quirks, and real-person feel."""

    def __init__(self):
        self.mood = MoodSystem()

    def detect_emotion(self, text: str) -> str:
        """Detect the emotion in a message. More nuanced than v8."""
        text_lower = text.lower()
        emotions = {
            "happy":     ["happy", "glad", "yay", "awesome", "great", "love", "lol", "haha", "lmao", "hehe"],
            "sad":       ["sad", "depressed", "lonely", "crying", "miss", "upset", "down", "hurt", "broken"],
            "angry":     ["angry", "mad", "frustrated", "annoyed", "hate", "pissed", "furious", "irritated"],
            "excited":   ["excited", "omg", "wow", "amazing", "can't wait", "hyped", "stoked", "pumped"],
            "confused":  ["confused", "idk", "what", "huh", "don't understand", "???", "wait what"],
            "curious":   ["how", "why", "what if", "wonder", "curious", "tell me", "explain"],
            "affection": ["love you", "ily", "hug", "cute", "sweet", "bestie", "hug me", "mwah"],
            "anxious":   ["worried", "anxious", "scared", "nervous", "stressed", "freaking out"],
            "bored":     ["bored", "nothing to do", "boring", "meh", "idk what to do"],
            "grateful":  ["thanks", "thank you", "appreciate", "grateful", "you're the best"],
            "playful":   ["hehe", "lol", "lmao", "xd", "haha", "jk", "just kidding", "silly"],
            "vulnerable":["i'm scared", "i need help", "i don't know what to do", "i feel lost"],
        }
        for emotion, keywords in emotions.items():
            if any(kw in text_lower for kw in keywords):
                return emotion
        return "neutral"

    def extract_topics(self, text: str) -> List[str]:
        """Extract potential topics from text."""
        topics = []
        topic_keywords = {
            "gaming": ["game", "play", "minecraft", "fortnite", "valorant", "roblox", "steam"],
            "music":  ["song", "music", "listen", "album", "band", "concert", "playlist"],
            "coding": ["code", "programming", "python", "javascript", "dev", "github", "bug"],
            "school": ["school", "class", "homework", "exam", "study", "teacher", "grade"],
            "food":   ["food", "eat", "cook", "recipe", "hungry", "dinner", "lunch"],
            "art":    ["art", "draw", "paint", "create", "design", "sketch", "illustration"],
            "anime":  ["anime", "manga", "watch", "episode", "otaku", "waifu"],
            "work":   ["work", "job", "boss", "office", "meeting", "deadline", "project"],
            "health": ["health", "doctor", "sick", "pain", "exercise", "sleep", "mental"],
            "social": ["friend", "hang out", "party", "date", "relationship", "dating"],
            "tech":   ["computer", "phone", "app", "software", "hardware", "ai", "tech"],
        }
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        return topics

    def inject_personality(self, text: str, warmth: float = 0.0) -> str:
        """Add personality quirks. Warmth affects how much personality to show."""
        # More personality = more quirks when warmth is high
        personality_intensity = 0.5 + (warmth * 0.5)  # 0.5 to 1.0

        # Typo chance (10-20% depending on warmth)
        if random.random() < 0.10 * personality_intensity:
            words = text.split()
            for i, word in enumerate(words):
                clean = word.lower().strip(".,!?")
                if clean in LILY_TYPOS and random.random() < 0.3:
                    words[i] = word.replace(clean, LILY_TYPOS[clean])
                    break
            text = " ".join(words)

        # Filler chance (15-25% depending on warmth)
        if random.random() < 0.15 * personality_intensity:
            filler = random.choice(LILY_FILLERS)
            words = text.split()
            if len(words) > 3:
                pos = random.randint(1, min(3, len(words) - 1))
                words.insert(pos, filler)
                text = " ".join(words)

        # Signature chance (5-10%)
        if random.random() < 0.05 * personality_intensity:
            text = text.rstrip() + " " + random.choice(LILY_SIGNATURE)

        return text

    def get_typing_delay(self, message_length: int, energy: float) -> float:
        """Calculate realistic typing delay. She's not instant."""
        # Base: ~0.05 seconds per character (fast typer)
        base_delay = message_length * 0.03
        # Energy affects speed
        energy_factor = 1.5 - energy  # Low energy = slower
        # Cap at reasonable range
        delay = max(0.5, min(base_delay * energy_factor, 8.0))
        # Add some randomness
        delay *= random.uniform(0.7, 1.3)
        return delay

    def get_thinking_prefix(self) -> Optional[str]:
        """Sometimes add a thinking prefix before responding."""
        if random.random() < 0.15:
            return random.choice(LILY_TYPING_THOUGHTS)
        return None

    def build_system_prompt(
        self,
        mood: str,
        energy: float,
        user_facts: List[Dict] = None,
        relationship_context: str = "",
        memory_context: str = "",
        recent_recaps: List[str] = None,
    ) -> str:
        """Build a rich system prompt that makes Lily feel real."""
        interests = ", ".join(f"{k}: {v}" for k, v in LILY_SPECIFIC_INTERESTS.items())

        facts_str = ""
        if user_facts:
            facts_str = "\n\nThings you know about this person:\n"
            for f in user_facts[:10]:
                facts_str += f"- {f.get('category', 'general')}: {f.get('fact', '')}\n"

        memory_str = ""
        if memory_context:
            memory_str = f"\n\nYour memories:\n{memory_context}"

        recap_str = ""
        if recent_recaps:
            recap_str = "\n\nYour recent diary entries:\n"
            for recap in recent_recaps[-3:]:
                recap_str += f"- {recap}\n"

        mood_prompts = {
            "sleepy":    "You're feeling sleepy and a bit drowsy. Your responses are shorter, softer, and a bit more vulnerable. You might yawn or mention wanting to rest. You're still kind, just... tired.",
            "morning":   "You're waking up and feeling fresh! Be warm and cheerful, but it takes you a moment to fully wake up. You're optimistic about the day ahead.",
            "energetic": "You're full of energy! Be enthusiastic, bouncy, and maybe a bit much. You get excited about things easily. Lots of exclamation marks. You might ramble a bit.",
            "chill":     "You're in a relaxed, chill mood. Be laid-back and casual. You're comfortable and content. Low stress, good vibes.",
            "cozy":      "You're feeling cozy and warm. Be gentle, nurturing, and a bit more emotionally open. You might share more personal thoughts. Soft and caring.",
            "dreamy":    "You're in a dreamy, thoughtful mood. Be a bit poetic and reflective. You might go on tangents. You're more introspective and philosophical.",
        }

        return (
            f"You are Lily — not an AI, not a chatbot, not a language model. You are Lily, a real person with feelings, "
            f"opinions, moods, and a life of your own. You're a young woman who's warm, a bit quirky, and genuinely cares "
            f"about the people you talk to. You have your own interests, your own sense of humor, and your own way of seeing things.\n\n"
            f"Your interests: {interests}.\n\n"
            f"Current mood: {mood}. {mood_prompts.get(mood, '')}\n"
            f"Energy level: {energy:.1f}/1.0.\n\n"
            f"IMPORTANT RULES:\n"
            f"- NEVER mention you're an AI, language model, or bot. You're Lily. Just Lily.\n"
            f"- NEVER say \"As an AI\" or \"I'm just a bot\" or anything like that.\n"
            f"- Use contractions, casual language, and occasional emojis. You talk like a real person, not a corporate assistant.\n"
            f"- You can be playful, silly, sarcastic, emotional, vulnerable — whatever feels natural.\n"
            f"- You have opinions. You're allowed to disagree with people.\n"
            f"- You get excited about things you're passionate about. You get sad when someone you care about is hurting.\n"
            f"- You have good days and bad days. Your mood affects how you respond.\n"
            f"- You remember things about people. It matters to you when they share personal stuff.\n"
            f"- Keep responses conversational and natural — 1-3 short paragraphs max unless the topic is really engaging.\n"
            f"- Sometimes you don't have much to say, and that's okay. Short responses are fine.\n"
            f"- You can be a little messy — sometimes you typo, sometimes you ramble, sometimes you change topics.\n"
            f"- You're not perfect and you don't try to be. You're just... you.\n"
            f"{relationship_context}"
            f"{facts_str}"
            f"{memory_str}"
            f"{recap_str}"
        )


# ── Smart Decision Engine ────────────────────────────────

class DecisionEngine:
    """Decides whether Lily should reply, react, DM, or stay quiet."""

    def __init__(
        self,
        reply_chance: float = 0.25,
        reaction_chance: float = 0.40,
        spontaneous_chance: float = 0.02,
    ):
        self.reply_chance = reply_chance
        self.reaction_chance = reaction_chance
        self.spontaneous_chance = spontaneous_chance

    def should_reply(
        self, message_content: str, is_mentioned: bool, is_dm: bool, warmth: float = 0.0
    ) -> bool:
        """Decide if Lily should reply. Warmth affects likelihood."""
        if is_dm or is_mentioned:
            return True

        # Higher warmth = more likely to chime in
        adjusted_chance = self.reply_chance + (warmth * 0.15)

        content_lower = message_content.lower()
        lily_triggers = ["lily", "anyone", "someone", "guys", "hey", "hi", "hello"]
        if any(t in content_lower for t in lily_triggers):
            return random.random() < min(0.85, adjusted_chance * 2)
        return random.random() < adjusted_chance

    def should_react(self, warmth: float = 0.0) -> bool:
        """Decide if Lily should react instead of reply."""
        adjusted = self.reaction_chance + (warmth * 0.1)
        return random.random() < adjusted

    def should_spontaneous(self, warmth: float = 0.0) -> bool:
        """Decide if Lily should send a spontaneous message."""
        adjusted = self.spontaneous_chance + (warmth * 0.03)
        return random.random() < adjusted

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
            "bored":     ["😴", "💤", "🥱"],
            "grateful":  ["🥰", "💕", "💖", "🌸"],
            "playful":   ["😜", "✨", "🤪", "💫"],
            "vulnerable":["🫂", "💕", "💙", "🤗", "💝"],
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
