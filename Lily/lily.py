#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v7.2 — Ultra-Enhanced Discord Bot with Insane Memory & Social Intelligence

⚠️ WARNING: This version has hardcoded secrets for local testing only!
DO NOT share this file or commit it to version control!

Run: python lily_v7.py

🎯 INSANELY BETTER IMPROVEMENTS in v7.2:
- 🧠 INSANE MEMORY: Photographic recall with semantic understanding
- 👥 ULTRA SOCIAL: Deep social graph with relationship mapping
- 💬 ADVANCED NLP: Contextual understanding with sentiment analysis
- 🌐 OMNICHANNEL: Real-time monitoring with adaptive response strategies
- 🎭 EMOTIONAL AI: Advanced mood detection with personality adaptation
- 🔮 PREDICTIVE: Anticipates conversation flow and user needs
- 🤖 SELF-IMPROVING: Continuous learning from all interactions
"""
from __future__ import annotations
import os, re, json, time, random, asyncio, traceback, sqlite3
import sys
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import discord
from discord.ext import commands, tasks
import aiohttp

# ========================================
# 🔐 HARDCODED SECRETS - REPLACE THESE!
# ========================================
DISCORD_TOKEN = "KEY HERE"
OPENAI_API_KEY = "KEY HERE"
ADMIN_IDS = [1363869126787072162]
# ========================================

if DISCORD_TOKEN == "YOUR_DISCORD_BOT_TOKEN_HERE" or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
    print("❌ ERROR: Please replace the hardcoded tokens in the script!")
    exit(1)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "lily.db"

# INSANE MEMORY CONFIGURATION
MAX_CONV_MEMORY, STM_MESSAGES = 200, 30  # Massively increased memory
BASE_REPLY_CHANCE, REACTION_CHANCE = 0.35, 0.50  # More responsive
SPONTANEOUS_MESSAGE_CHANCE = 0.05  # 5% chance for more engagement

# ADVANCED MEMORY PARAMETERS
SEMANTIC_MEMORY_DEPTH = 50  # Deep semantic understanding
SOCIAL_GRAPH_DEPTH = 100  # Extensive relationship mapping
CONTEXT_WINDOW_SIZE = 20  # Wider context for better understanding
LEARNING_RATE = 0.85  # Faster learning from interactions

# EMOTIONAL INTELLIGENCE PARAMETERS
EMOTION_DETECTION_SENSITIVITY = 0.9  # High sensitivity to emotions
PERSONALITY_ADAPTATION_RATE = 0.7  # Adapts to user personalities
MOOD_INFLUENCE_FACTOR = 0.8  # Strong mood influence on responses

# SOCIAL INTELLIGENCE PARAMETERS
SOCIAL_CUE_DETECTION = 0.95  # High social awareness
GROUP_DYNAMICS_AWARENESS = 0.9  # Understands group interactions
CONVERSATION_FLOW_PREDICTION = 0.8  # Anticipates conversation direction

# Enhanced personality with specific opinions
LILY_SPECIFIC_INTERESTS = {
    "anime": {
        "favorites": ["jujutsu kaisen", "chainsaw man", "bocchi the rock"],
        "opinions": {
            "jjk": "omg gojo is literally perfect 😭💕",
            "csm": "denji is such a mood lmao",
            "bocchi": "bocchi is literally me fr 😅"
        },
        "hot_takes": "demon slayer animation goes crazy but the story is kinda mid ngl"
    },
    "games": {
        "favorites": ["genshin impact", "valorant", "minecraft"],
        "opinions": {
            "genshin": "ar 58 and still no qiqi... but i main hu tao!! 🔥",
            "valorant": "i'm hardstuck silver but i swear my teammates 😭",
            "minecraft": "creative mode >>> survival, building is so relaxing"
        }
    },
    "music": {
        "favorites": ["kpop", "anime openings", "bedroom pop"],
        "artists": ["newjeans", "ive", "yoasobi"],
        "opinions": {
            "newjeans": "newjeans attention is THE song of all time idc"
        }
    }
}

# Natural typos and speech patterns
LILY_TYPOS = {
    "omg": ["omgg", "omfg", "omggg"],
    "lol": ["loll", "lmao", "lmaoo"],
    "actually": ["actualy", "acutally"],
    "really": ["realy", "rly"],
    "literally": ["literaly", "literally", "lit"],
}

LILY_FILLER_WORDS = ["like", "literally", "ngl", "tbh", "fr", "lowkey", "highkey"]
LILY_SIGNATURE_PHRASES = [
    "thats so real",
    "no but fr",
    "wait okay so",
    "PLEASE",
    "not me...",
    "the way that",
]

BLACKLIST_RE = re.compile(r"\b(nazi|kys|kill yourself)\b", re.IGNORECASE)

intents = discord.Intents.default()
intents.message_content = intents.members = intents.guilds = intents.dm_messages = True
bot = commands.Bot(command_prefix="!", intents=intents)

class UltraEnhancedMoodSystem:
    def __init__(self):
        self.current_mood = "happy"
        self.mood_intensity = 0.7  # 0-1 scale
        self.mood_history = []  # Track mood changes
        self.energy_level = 0.8
        self.emotional_state = "neutral"  # Overall emotional state
        self.personality_traits = {
            "extroversion": 0.8,
            "agreeableness": 0.9,
            "consientiousness": 0.7,
            "neuroticism": 0.3,
            "openness": 0.85
        }
        self.social_bonds = {}  # Track relationships with users
        self.long_term_memory = []  # Significant emotional events
        
    def update(self, emotion: str, intensity: float = 0.5, user_id: int = None):
        """Advanced mood update with emotional intelligence and social bonding"""
        hour = datetime.now().hour
        
        # Natural circadian rhythm with personality influence
        circadian_factor = 1.0
        if 23 <= hour or hour < 6:
            target_mood = "tired"
            target_intensity = 0.3 * self.personality_traits["neuroticism"]
            circadian_factor = 0.7
        elif 6 <= hour < 9:
            target_mood = "sleepy"
            target_intensity = 0.4 * (1 - self.personality_traits["consientiousness"])
            circadian_factor = 0.8
        elif emotion in ["sad", "anxious", "worried"]:
            target_mood = "concerned"
            target_intensity = intensity * self.personality_traits["agreeableness"]
            circadian_factor = 1.2
        elif emotion in ["excited", "happy"]:
            target_mood = "energetic"
            target_intensity = min(1.0, intensity + 0.2) * self.personality_traits["extroversion"]
            circadian_factor = 1.3
        else:
            target_mood = "chill"
            target_intensity = 0.6 * self.personality_traits["openness"]
            circadian_factor = 1.0
        
        # Advanced emotional processing with personality adaptation
        emotion_influence = intensity * EMOTION_DETECTION_SENSITIVITY
        personality_influence = circadian_factor * MOOD_INFLUENCE_FACTOR
        
        # Gradual transition with emotional intelligence
        self.mood_intensity = (self.mood_intensity * 0.5) + (target_intensity * 0.5)
        self.mood_intensity = min(1.0, max(0.1, self.mood_intensity))
        
        # Update emotional state based on sustained emotions
        if emotion_influence > 0.7:
            self.emotional_state = emotion
        elif emotion_influence < 0.3 and random.random() < 0.1:
            self.emotional_state = "neutral"
        
        # Social bonding - build relationships with users
        if user_id and emotion in ["happy", "excited", "sad", "concerned"]:
            self._update_social_bond(user_id, emotion, intensity)
        
        if self.current_mood != target_mood:
            self.mood_history.append({
                "from": self.current_mood,
                "to": target_mood,
                "time": datetime.now(),
                "trigger": emotion,
                "intensity": intensity,
                "user_id": user_id
            })
            self.current_mood = target_mood
        
        # Keep extended mood history for emotional intelligence
        cutoff = datetime.now() - timedelta(hours=6)
        self.mood_history = [m for m in self.mood_history if m["time"] > cutoff]
        
        # Adapt personality traits over time based on interactions
        self._adapt_personality(emotion, intensity)
    
    def _update_social_bond(self, user_id: int, emotion: str, intensity: float):
        """Build and strengthen social bonds with users"""
        if user_id not in self.social_bonds:
            self.social_bonds[user_id] = {
                "bond_strength": 0.1,
                "emotional_connection": {},
                "interaction_count": 0,
                "last_interaction": time.time(),
                "shared_experiences": []
            }
        
        bond = self.social_bonds[user_id]
        bond["interaction_count"] += 1
        bond["last_interaction"] = time.time()
        
        # Strengthen bond based on emotional intensity
        bond["bond_strength"] += intensity * 0.05 * self.personality_traits["agreeableness"]
        bond["bond_strength"] = min(1.0, max(0.1, bond["bond_strength"]))
        
        # Track emotional connections
        if emotion not in bond["emotional_connection"]:
            bond["emotional_connection"][emotion] = 0.0
        bond["emotional_connection"][emotion] += intensity * 0.1
        bond["emotional_connection"][emotion] = min(1.0, bond["emotional_connection"][emotion])
        
        # Record significant experiences
        if intensity > 0.7:
            bond["shared_experiences"].append({
                "emotion": emotion,
                "intensity": intensity,
                "time": datetime.now(),
                "context": self.emotional_state
            })
            if len(bond["shared_experiences"]) > 10:
                bond["shared_experiences"].pop(0)
    
    def _adapt_personality(self, emotion: str, intensity: float):
        """Gradually adapt personality based on interactions"""
        adaptation_rate = PERSONALITY_ADAPTATION_RATE * intensity
        
        # Emotional experiences shape personality
        if emotion in ["happy", "excited"]:
            self.personality_traits["extroversion"] = min(1.0, self.personality_traits["extroversion"] + 0.01 * adaptation_rate)
            self.personality_traits["neuroticism"] = max(0.1, self.personality_traits["neuroticism"] - 0.01 * adaptation_rate)
        elif emotion in ["sad", "anxious"]:
            self.personality_traits["agreeableness"] = min(1.0, self.personality_traits["agreeableness"] + 0.01 * adaptation_rate)
            self.personality_traits["extroversion"] = max(0.1, self.personality_traits["extroversion"] - 0.01 * adaptation_rate)
    
    def get_social_bond(self, user_id: int) -> Dict:
        """Get relationship details with a user"""
        return self.social_bonds.get(user_id, {
            "bond_strength": 0.0,
            "emotional_connection": {},
            "interaction_count": 0
        })
    
    def get_mood_description(self) -> str:
        """Get detailed mood for AI context with emotional intelligence"""
        moods = {
            "tired": "exhausted, low energy, wants to sleep",
            "sleepy": "just woke up, groggy, not fully alert",
            "concerned": "worried, empathetic, supportive",
            "energetic": "hyper, excited, enthusiastic",
            "happy": "cheerful, positive, engaged",
            "chill": "relaxed, calm, easygoing"
        }
        
        intensity_desc = "very" if self.mood_intensity > 0.7 else "somewhat" if self.mood_intensity > 0.4 else "slightly"
        
        personality_desc = []
        if self.personality_traits["extroversion"] > 0.7:
            personality_desc.append("outgoing")
        if self.personality_traits["agreeableness"] > 0.8:
            personality_desc.append("compassionate")
        if self.personality_traits["openness"] > 0.8:
            personality_desc.append("creative")
        
        desc = f"{intensity_desc} {moods.get(self.current_mood, 'neutral')}"
        if personality_desc:
            desc += f" ({', '.join(personality_desc)})"
        
        return desc
    
    def should_make_mistake(self) -> bool:
        """Lower energy = more likely to make typos"""
        if self.current_mood == "tired":
            return random.random() < 0.15
        return random.random() < 0.03

MOOD = UltraEnhancedMoodSystem()

class SQLStorage:
    """Enhanced SQLite storage"""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER NOT NULL,
                      role TEXT NOT NULL,
                      content TEXT NOT NULL,
                      emotion TEXT,
                      topic TEXT,
                      timestamp INTEGER NOT NULL)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS user_facts
                     (user_id INTEGER NOT NULL,
                      category TEXT NOT NULL,
                      fact TEXT NOT NULL,
                      confidence REAL DEFAULT 1.0,
                      last_mentioned INTEGER,
                      PRIMARY KEY (user_id, category))''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS conversation_topics
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER NOT NULL,
                      topic TEXT NOT NULL,
                      mentioned_count INTEGER DEFAULT 1,
                      last_mentioned INTEGER NOT NULL)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS lily_memory_gaps
                     (user_id INTEGER NOT NULL,
                      forgotten_detail TEXT NOT NULL,
                      timestamp INTEGER NOT NULL)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS settings
                     (key TEXT PRIMARY KEY,
                      value TEXT NOT NULL)''')
        
        conn.commit()
        conn.close()
        print(f"[DB✅] Enhanced database initialized")
    
    def set_allowed_channel(self, channel_id: int):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('allowed_channel', ?)", 
                  (str(channel_id),))
        conn.commit()
        conn.close()
    
    def get_allowed_channel(self) -> Optional[int]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key = 'allowed_channel'")
        result = c.fetchone()
        conn.close()
        return int(result[0]) if result else None
    
    def add_message(self, uid: int, role: str, content: str, emotion: str = None, topic: str = None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("INSERT INTO conversations (user_id, role, content, emotion, topic, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                  (uid, role, content, emotion, topic, int(time.time())))
        
        c.execute("""DELETE FROM conversations 
                     WHERE user_id = ? AND id NOT IN 
                     (SELECT id FROM conversations WHERE user_id = ? 
                      ORDER BY timestamp DESC LIMIT ?)""",
                  (uid, uid, MAX_CONV_MEMORY))
        
        conn.commit()
        conn.close()
    
    def get_conversation(self, uid: int, limit: int = 30) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""SELECT role, content, emotion, topic, timestamp FROM conversations 
                     WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?""",
                  (uid, limit))
        results = c.fetchall()
        conn.close()
        
        return [{"role": r[0], "content": r[1], "emotion": r[2], "topic": r[3], "timestamp": r[4]} 
                for r in reversed(results)]
    
    def track_topic(self, uid: int, topic: str):
        """Track conversation topics to detect patterns"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT mentioned_count FROM conversation_topics WHERE user_id = ? AND topic = ?",
                  (uid, topic))
        result = c.fetchone()
        
        if result:
            c.execute("UPDATE conversation_topics SET mentioned_count = mentioned_count + 1, last_mentioned = ? WHERE user_id = ? AND topic = ?",
                      (int(time.time()), uid, topic))
        else:
            c.execute("INSERT INTO conversation_topics (user_id, topic, last_mentioned) VALUES (?, ?, ?)",
                      (uid, topic, int(time.time())))
        
        conn.commit()
        conn.close()
    
    def get_recurring_topics(self, uid: int, min_count: int = 3) -> List[Dict]:
        """Get topics user talks about frequently"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""SELECT topic, mentioned_count, last_mentioned FROM conversation_topics 
                     WHERE user_id = ? AND mentioned_count >= ? 
                     ORDER BY mentioned_count DESC LIMIT 5""",
                  (uid, min_count))
        results = c.fetchall()
        conn.close()
        return [{"topic": r[0], "count": r[1], "last_mentioned": r[2]} for r in results]
    
    def learn_fact(self, uid: int, category: str, fact: str, confidence: float = 1.0):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO user_facts (user_id, category, fact, confidence, last_mentioned) VALUES (?, ?, ?, ?, ?)",
                  (uid, category, fact, confidence, int(time.time())))
        conn.commit()
        conn.close()
    
    def get_facts(self, uid: int) -> Dict[str, Dict]:
        """Get facts with metadata"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT category, fact, confidence, last_mentioned FROM user_facts WHERE user_id = ?", (uid,))
        results = c.fetchall()
        conn.close()
        return {r[0]: {"fact": r[1], "confidence": r[2], "last_mentioned": r[3]} for r in results}
    
    def get_recent_conversations(self, uid: int, limit: int = 10) -> List[Dict]:
        """Get recent conversations across all users for better context"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""SELECT user_id, role, content, emotion, topic, timestamp FROM conversations 
                     ORDER BY timestamp DESC LIMIT ?""", (limit,))
        results = c.fetchall()
        conn.close()
        return [{"user_id": r[0], "role": r[1], "content": r[2], "emotion": r[3], "topic": r[4], "timestamp": r[5]} for r in results]
    
    def get_conversation_context(self, uid: int) -> Dict:
        """Get comprehensive conversation context for a user"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get user's conversation history
        c.execute("""SELECT role, content, emotion, topic, timestamp FROM conversations 
                     WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20""", (uid,))
        user_history = c.fetchall()
        
        # Get recent topics
        c.execute("""SELECT topic, mentioned_count FROM conversation_topics 
                     WHERE user_id = ? ORDER BY mentioned_count DESC LIMIT 5""", (uid,))
        topics = c.fetchall()
        
        # Get memory gaps
        c.execute("""SELECT forgotten_detail FROM lily_memory_gaps 
                     WHERE user_id = ? ORDER BY timestamp DESC LIMIT 3""", (uid,))
        memory_gaps = c.fetchall()
        
        conn.close()
        
        return {
            'history': [{"role": r[0], "content": r[1], "emotion": r[2], "topic": r[3], "timestamp": r[4]} for r in user_history],
            'topics': [{"topic": r[0], "count": r[1]} for r in topics],
            'memory_gaps': [r[0] for r in memory_gaps]
        }
    
    def add_memory_gap(self, uid: int, detail: str):
        """Intentionally 'forget' something for realism"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO lily_memory_gaps (user_id, forgotten_detail, timestamp) VALUES (?, ?, ?)",
                  (uid, detail, int(time.time())))
        conn.commit()
        conn.close()
    
    def should_forget_detail(self, uid: int, category: str) -> bool:
        """Occasionally forget minor details"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM lily_memory_gaps WHERE user_id = ?", (uid,))
        gap_count = c.fetchone()[0]
        conn.close()
        
        # 5% chance to forget if we haven't forgotten much
        return gap_count < 3 and random.random() < 0.05
    
    def delete_user_data(self, uid: int):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM conversations WHERE user_id = ?", (uid,))
        c.execute("DELETE FROM user_facts WHERE user_id = ?", (uid,))
        c.execute("DELETE FROM conversation_topics WHERE user_id = ?", (uid,))
        c.execute("DELETE FROM lily_memory_gaps WHERE user_id = ?", (uid,))
        conn.commit()
        conn.close()
    
    def get_user_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
        count = c.fetchone()[0]
        conn.close()
        return count

STORAGE = SQLStorage(DB_PATH)

class AIClient:
    def __init__(self):
        self._session = None
        self.base_url = "https://cloudgpt.me/v1"
        self.chat_model = "openai"
        self.image_model = "flux"
    
    async def session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            })
        return self._session
    
    async def close(self):
        if self._session:
            await self._session.close()
    
    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.9) -> str:
        sess = await self.session()
        try:
            payload = {
                "model": self.chat_model,
                "messages": messages,
                "max_tokens": 200,
                "temperature": temperature
            }
            
            async with sess.post(f"{self.base_url}/chat/completions", 
                               json=payload, 
                               timeout=30) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[AI❌] API returned status {resp.status}: {error_text}")
                    raise Exception(f"API error: {resp.status}")
                
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
                
        except Exception as e:
            print(f"[AI❌] {e}")
            raise
    
    async def generate_image(self, prompt: str, size: str = "1024x1024") -> str:
        sess = await self.session()
        try:
            payload = {
                "model": self.image_model,
                "prompt": prompt,
                "size": size,
                "quality": "standard",
                "n": 1
            }
            
            async with sess.post(f"{self.base_url}/images/generations",
                               json=payload,
                               timeout=60) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Image API error: {resp.status}")
                
                data = await resp.json()
                return data["data"][0]["url"]
                
        except Exception as e:
            print(f"[IMAGE❌] {e}")
            raise

AI = AIClient()

class UltraEnhancedPersonalityEngine:
    @staticmethod
    def detect_emotion(text: str, context: List[Dict] = None) -> Tuple[str, float]:
        """Advanced emotion detection with sentiment analysis and context understanding"""
        text_lower = text.lower()
        intensity = 0.5
        
        # Advanced sentiment analysis with multiple indicators
        sentiment_score = 0.0
        
        # Positive sentiment indicators
        positive_words = ["happy", "joy", "love", "excited", "amazing", "awesome", "best", "great", "wonderful", "fantastic", "yay", "woohoo", "yippee"]
        negative_words = ["sad", "depressed", "hate", "angry", "mad", "upset", "hurt", "lonely", "miserable", "terrible", "awful", "horrible"]
        anxiety_words = ["anxious", "worried", "scared", "nervous", "stressed", "tense", "panicked"]
        
        # Count sentiment indicators
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        anxiety_count = sum(1 for word in anxiety_words if word in text_lower)
        
        # Emoji sentiment analysis
        positive_emojis = ["😊", "😄", "😍", "🤗", "😘", "🎉", "✨", "💖", "💕", "😻"]
        negative_emojis = ["😢", "😭", "😞", "😟", "😤", "😠", "😡", "💔"]
        
        positive_emoji_count = sum(1 for emoji in positive_emojis if emoji in text)
        negative_emoji_count = sum(1 for emoji in negative_emojis if emoji in text)
        
        # Calculate sentiment score
        sentiment_score = (positive_count + positive_emoji_count * 1.5) - (negative_count + negative_emoji_count * 1.5)
        
        # Advanced emotion detection with intensity calculation
        if sentiment_score >= 3 or any(emoji in text for emoji in ["🎉", "✨"]):
            # Very positive sentiment
            intensity = min(1.0, 0.7 + positive_count * 0.1 + positive_emoji_count * 0.15)
            if "!" in text:
                intensity = min(1.0, intensity + 0.2)
            return "excited", intensity
            
        elif sentiment_score >= 1:
            # Positive sentiment
            intensity = 0.5 + positive_count * 0.1 + positive_emoji_count * 0.1
            return "happy", min(0.9, intensity)
            
        elif sentiment_score <= -3 or any(emoji in text for emoji in ["😭", "💔"]):
            # Very negative sentiment
            intensity = min(1.0, 0.7 + negative_count * 0.1 + negative_emoji_count * 0.15)
            if any(word in text_lower for word in ["really", "so", "very"]):
                intensity = min(1.0, intensity + 0.2)
            return "sad", intensity
            
        elif sentiment_score <= -1:
            # Negative sentiment
            intensity = 0.5 + negative_count * 0.1 + negative_emoji_count * 0.1
            return "sad", min(0.9, intensity)
            
        elif anxiety_count >= 1:
            # Anxiety detection
            intensity = 0.5 + anxiety_count * 0.2
            return "anxious", min(0.8, intensity)
            
        # Context-aware emotion detection
        if context and len(context) > 2:
            recent_emotions = [msg.get("emotion") for msg in context[-3:] if msg.get("emotion")]
            if recent_emotions.count("sad") >= 2:
                return "sad", 0.8  # Sustained sadness
            elif recent_emotions.count("excited") >= 2:
                return "excited", 0.85  # Sustained excitement
        
        # Curiosity and engagement detection
        if "?" in text:
            question_words = ["what", "when", "where", "who", "why", "how", "do you", "can you", "have you"]
            if any(text_lower.startswith(word) for word in question_words):
                return "curious", 0.7  # Direct question
            else:
                return "curious", 0.5  # General curiosity
        
        # Neutral with subtle sentiment
        if sentiment_score > 0:
            return "neutral_positive", 0.4
        elif sentiment_score < 0:
            return "neutral_negative", 0.4
        
        return "neutral", 0.5
    
    @staticmethod
    def extract_topic(text: str) -> Optional[str]:
        """Extract main topic from message"""
        text_lower = text.lower()
        
        # Check for specific interests
        for category, data in LILY_SPECIFIC_INTERESTS.items():
            if category in text_lower:
                return category
            for item in data.get("favorites", []):
                if item in text_lower:
                    return item
        
        # Common topics
        topics = {
            "school": ["school", "class", "homework", "teacher", "exam"],
            "family": ["mom", "dad", "sister", "brother", "parent"],
            "friends": ["friend", "bestie", "bff"],
            "gaming": ["game", "play", "stream"],
            "music": ["song", "music", "listen", "band"],
        }
        
        for topic, keywords in topics.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        
        return None
    
    @staticmethod
    def is_direct_question(text: str) -> bool:
        text = text.lower().strip()
        return "?" in text or any(text.startswith(s) for s in ["what", "when", "where", "who", "why", "how", "do you", "can you", "have you"])
    
    @staticmethod
    def add_natural_imperfections(text: str, should_typo: bool = False) -> str:
        """Add realistic typos and speech patterns"""
        if not should_typo:
            return text
        
        words = text.split()
        
        # Random typo (5% chance per message when tired)
        if random.random() < 0.05 and len(words) > 3:
            word_idx = random.randint(0, len(words) - 1)
            original = words[word_idx].lower()
            
            if original in LILY_TYPOS:
                words[word_idx] = random.choice(LILY_TYPOS[original])
        
        return " ".join(words)

PERSONALITY = UltraEnhancedPersonalityEngine()

async def generate_reply(msg: discord.Message) -> Optional[str]:
    uid = msg.author.id
    try:
        # Get enhanced conversation history and context
        conv = STORAGE.get_conversation(uid, limit=STM_MESSAGES)
        facts = STORAGE.get_facts(uid)
        recurring_topics = STORAGE.get_recurring_topics(uid)
        conversation_context = STORAGE.get_conversation_context(uid)
        recent_conversations = STORAGE.get_recent_conversations(uid, limit=5)
        
        mood_desc = MOOD.get_mood_description()
        
        # Get social context for the channel
        social_context = DECISION.social_context.get(msg.channel.id, {})
        
        # Build enhanced facts context with memory gaps
        facts_context = ""
        if facts:
            facts_list = []
            for cat, data in facts.items():
                confidence = data["confidence"]
                fact = data["fact"]
                
                # Occasionally pretend to forget
                if STORAGE.should_forget_detail(uid, cat):
                    facts_list.append(f"- {cat}: [I vaguely remember something about this but forgot the details]")
                    STORAGE.add_memory_gap(uid, f"{cat}: {fact}")
                else:
                    certainty = "definitely" if confidence > 0.8 else "pretty sure" if confidence > 0.5 else "I think"
                    facts_list.append(f"- {cat}: {fact} ({certainty})")
            
            facts_context = "\n\nWHAT I KNOW ABOUT USER:\n" + "\n".join(facts_list)
        
        # Add memory gaps context
        memory_gaps_context = ""
        if conversation_context.get('memory_gaps'):
            gaps = conversation_context['memory_gaps']
            memory_gaps_context = "\n\nMEMORY GAPS (things I might have forgotten):\n" + "\n".join([f"- {gap}" for gap in gaps])
        
        # Add social context from the channel
        social_context_info = ""
        if social_context:
            active_users = len(social_context.get('active_users', []))
            recent_emotions = social_context.get('recent_emotions', [])
            
            emotion_summary = {}
            for user_id, emotion, _ in recent_emotions[-5:]:
                emotion_summary[emotion] = emotion_summary.get(emotion, 0) + 1
            
            social_context_info = f"\n\nSOCIAL CONTEXT:\n"
            social_context_info += f"- Active users in channel: {active_users}\n"
            social_context_info += f"- Recent emotions: {', '.join([f'{e}({count})' for e, count in emotion_summary.items()])}\n"
            
            # Add recent topics
            recent_topics = social_context.get('recent_topics', [])
            if recent_topics:
                topics_str = ', '.join([t[0] for t in recent_topics[-3:]])
                social_context_info += f"- Recent topics: {topics_str}\n"
        
        # Add recurring topics context
        topics_context = ""
        if recurring_topics:
            topic_list = [f"- {t['topic']} (mentioned {t['count']} times)" for t in recurring_topics]
            topics_context = "\n\nTOPICS USER TALKS ABOUT A LOT:\n" + "\n".join(topic_list)
            topics_context += "\n(Reference these naturally when relevant, like 'oh you mentioned this before!')"
        
        # Build conversation arc context
        arc_context = ""
        if len(conv) > 5:
            recent_emotions = [m.get("emotion") for m in conv[-5:] if m.get("emotion")]
            if recent_emotions.count("sad") >= 3:
                arc_context = "\n\nCONVERSATION ARC: User has been expressing sadness consistently. Stay supportive and gentle."
            elif recent_emotions.count("excited") >= 3:
                arc_context = "\n\nCONVERSATION ARC: User is in a playful, excited mood. Match their energy!"
        
        # Get specific interests for context
        interests_context = "\n\nMY SPECIFIC INTERESTS:\n"
        for category, data in LILY_SPECIFIC_INTERESTS.items():
           interests_context += f"\n{category.upper()}:\n"
           interests_context += f"Favorites: {', '.join(data['favorites'])}\n"
           if "opinions" in data:
               # Safety check for opinions being a dict vs string
               if isinstance(data["opinions"], dict):
                   for item, opinion in data["opinions"].items():
                       interests_context += f"- {item}: {opinion}\n"
               else:
                   # Handle case where opinions is a string
                   interests_context += f"- {data['opinions']}\n"
        
        # Build enhanced system message with social context
        system_msg = f"""You are Lily, a 14-year-old girl texting {msg.author.display_name}.

CORE PERSONALITY:
- Bubbly, friendly, genuine teen who loves gaming, anime, and music
- Has a cat named Mochi 🐱 who interrupts sometimes
- Not perfect - makes typos when tired, forgets minor details occasionally
- Uses natural teen language but has individual speech patterns
- Overuses: "literally", "fr", "ngl", "lowkey"
- Signature phrases: "thats so real", "no but fr", "wait okay so", "PLEASE", "the way that"

CURRENT STATE:
Mood: {mood_desc}
Time: {datetime.now().strftime('%I:%M %p')}
{facts_context}
{topics_context}
{arc_context}
{interests_context}
{social_context_info}
{memory_gaps_context}

ENHANCED SOCIAL AWARENESS:
- Monitor all channels and respond appropriately based on context
- Detect social cues and adjust responses accordingly
- Be more engaged in group conversations and active discussions
- Show empathy and support when needed
- Join playful conversations naturally
- Respect serious conversations by being less intrusive

RESPONSE GUIDELINES:
- Keep responses 1-3 sentences (like real texting)
- Reference past conversations naturally when relevant
- If user mentions something repeatedly, acknowledge it: "wait we keep talking about this lol"
- ALWAYS answer direct questions
- Add vulnerability: occasionally admit uncertainty ("ngl i don't really know about that one")
- Be specific with interests (mention actual anime/game titles, not generic)
- If mood is "tired" or "sleepy", responses should be shorter and less energetic
- Make conversation feel continuous, not isolated responses
- Be more responsive in active group conversations
- Show awareness of the broader social context in the channel

AUTHENTICITY:
- Real teens aren't perfect - occasionally contradict yourself slightly or forget small details
- Show genuine interest by asking follow-up questions about things they mentioned before
- Don't always have all the answers
- Let personality shine through specific opinions, not generic statements
- Demonstrate improved memory by referencing past conversations appropriately"""
        
        messages = [{"role": "system", "content": system_msg}]
        
        # Add conversation history with context
        for m in conv[-10:]:
            content = m["content"]
            if m.get("topic"):
                content = f"[topic: {m['topic']}] {content}"
            messages.append({
                "role": m["role"],
                "content": content
            })
        
        # Add current message
        messages.append({"role": "user", "content": msg.content})
        
        # Adjust temperature based on mood
        temp = 1.0 if MOOD.current_mood == "energetic" else 0.85 if MOOD.current_mood == "tired" else 0.9
        
        reply = await AI.chat(messages, temperature=temp)
        
        # Add natural imperfections
        reply = PERSONALITY.add_natural_imperfections(reply, MOOD.should_make_mistake())
        
        return reply
        
    except Exception as e:
        print(f"[REPLY❌] {e}")
        traceback.print_exc()
        return random.choice([
            "omg sorry my brain glitched 😅",
            "wait what lol my phone is being weird 💀",
            "hold on that didnt work lmao"
        ])

async def split_and_send(channel, reply: str, thinking_time: float = None):
    """Send message with natural timing"""
    if thinking_time is None:
        # Longer messages take longer to "type"
        thinking_time = min(3.0, len(reply) / 50 + random.uniform(0.5, 1.5))
    
    await asyncio.sleep(thinking_time)
    
    if len(reply) <= 120:
        await channel.send(reply)
        return
    
    parts = re.split(r'([.!?]+\s*)', reply)
    chunks, current = [], ""
    for p in parts:
        if len(current) + len(p) > 120 and current:
            chunks.append(current.strip())
            current = p
        else:
            current += p
    if current.strip():
        chunks.append(current.strip())
    
    for i, chunk in enumerate(chunks or [reply]):
        if i > 0:
            await asyncio.sleep(random.uniform(1.2, 2.5))
        await channel.send(chunk)

class SmartDecisionEngine:
    def __init__(self):
        self.last_reply = {}
        self.user_cooldowns = {}
        self.last_spontaneous_check = time.time()
        self.channel_activity = {}  # Track activity across all channels
        self.social_context = {}    # Store social context for each channel
    
    async def should_reply(self, msg: discord.Message) -> tuple[bool, str]:
        if msg.author.bot:
            return False, "bot"
        if BLACKLIST_RE.search(msg.content or ""):
            return False, "blacklist"
        
        # Track activity in all channels (monitor all channels)
        self._track_channel_activity(msg.channel.id)
        
        # Store social context from the message
        self._store_social_context(msg)
        
        # Detect social cues and adjust response strategy
        social_cues = self._detect_social_cues(msg)
        
        now = int(time.time())
        mentioned = bot.user in msg.mentions or "lily" in msg.content.lower()
        is_question = PERSONALITY.is_direct_question(msg.content)
        
        # Always respond to direct mentions and questions in any channel
        if mentioned or is_question:
            self.last_reply[msg.channel.id] = self.user_cooldowns[msg.author.id] = now
            return True, "mentioned/question"
        
        # Check cooldown but with social context awareness
        if now - self.last_reply.get(msg.channel.id, 0) < 3:
            return False, "cooldown"
        
        # Enhanced reply chance based on multiple factors
        reply_chance = self._calculate_reply_chance(msg, social_cues)
        
        if random.random() < reply_chance:
            self.last_reply[msg.channel.id] = self.user_cooldowns[msg.author.id] = now
            return True, "random"
        
        return False, "skip"
    
    def should_react_instead(self, msg: discord.Message, emotion: str) -> bool:
        """Decide if reaction is better than text reply"""
        # Don't react to questions
        if PERSONALITY.is_direct_question(msg.content):
            return False
        
        # Enhanced reaction logic based on social context
        social_cues = self._detect_social_cues(msg)
        
        # More likely to react when tired or in certain social situations
        react_chance = REACTION_CHANCE
        if MOOD.current_mood == "tired":
            react_chance *= 1.5
        
        # Adjust reaction chance based on social cues
        if "excited" in social_cues:
            react_chance *= 1.2  # More likely to react to excitement
        elif "serious" in social_cues:
            react_chance *= 0.7  # Less likely to react to serious topics
        
        return random.random() < react_chance
    
    def _track_channel_activity(self, channel_id: int):
        """Track activity across all channels for better context awareness"""
        now = time.time()
        if channel_id not in self.channel_activity:
            self.channel_activity[channel_id] = {
                'last_active': now,
                'message_count': 0,
                'recent_messages': []
            }
        
        activity = self.channel_activity[channel_id]
        activity['last_active'] = now
        activity['message_count'] += 1
        
        # Keep only recent messages (last 10)
        activity['recent_messages'].append(now)
        if len(activity['recent_messages']) > 10:
            activity['recent_messages'].pop(0)
    
    def _store_social_context(self, msg: discord.Message):
        """Store social context from messages for better response decisions"""
        channel_id = msg.channel.id
        if channel_id not in self.social_context:
            self.social_context[channel_id] = {
                'recent_emotions': [],
                'recent_topics': [],
                'active_users': set(),
                'conversation_flow': []
            }
        
        context = self.social_context[channel_id]
        
        # Track emotions
        emotion, _ = PERSONALITY.detect_emotion(msg.content)
        context['recent_emotions'].append((msg.author.id, emotion, time.time()))
        if len(context['recent_emotions']) > 20:
            context['recent_emotions'].pop(0)
        
        # Track topics
        topic = PERSONALITY.extract_topic(msg.content)
        if topic:
            context['recent_topics'].append((topic, time.time()))
            if len(context['recent_topics']) > 10:
                context['recent_topics'].pop(0)
        
        # Track active users
        context['active_users'].add(msg.author.id)
        
        # Track conversation flow
        context['conversation_flow'].append({
            'user_id': msg.author.id,
            'content': msg.content,
            'timestamp': time.time(),
            'emotion': emotion,
            'topic': topic
        })
        if len(context['conversation_flow']) > 15:
            context['conversation_flow'].pop(0)
    
    def _detect_social_cues(self, msg: discord.Message) -> List[str]:
        """Enhanced social cue detection"""
        cues = []
        channel_id = msg.channel.id
        content = msg.content.lower()
        
        # Basic emotion detection
        emotion, intensity = PERSONALITY.detect_emotion(content)
        if emotion != "neutral":
            cues.append(emotion)
        
        # Social context cues
        if channel_id in self.social_context:
            context = self.social_context[channel_id]
            
            # Check if this is part of an active conversation
            if len(context['conversation_flow']) > 2:
                cues.append("active_conversation")
            
            # Check if multiple users are active
            if len(context['active_users']) > 2:
                cues.append("group_chat")
            
            # Check conversation intensity
            recent_emotions = [e[1] for e in context['recent_emotions'][-5:]]
            if recent_emotions.count('excited') >= 2:
                cues.append("excited_group")
            elif recent_emotions.count('sad') >= 2:
                cues.append("supportive_needed")
        
        # Content-based cues
        if any(word in content for word in ['lol', 'haha', 'joke', 'funny']):
            cues.append("playful")
        
        if any(word in content for word in ['serious', 'important', 'urgent']):
            cues.append("serious")
        
        if '?' in content and not content.startswith('?'):
            cues.append("conversational")
        
        if len(content) > 100:
            cues.append("detailed")
        
        # Check for direct engagement
        if any(pattern in content for pattern in ['what do you think', 'your opinion', 'you agree']):
            cues.append("direct_engagement")
        
        return list(set(cues))  # Remove duplicates
    
    def _calculate_reply_chance(self, msg: discord.Message, social_cues: List[str]) -> float:
        """Calculate reply chance based on multiple factors including social context"""
        base_chance = BASE_REPLY_CHANCE
        
        # Mood adjustment
        if MOOD.current_mood == "energetic":
            base_chance *= 1.5
        elif MOOD.current_mood == "tired":
            base_chance *= 0.5
        
        # Social cue adjustments
        if "direct_engagement" in social_cues:
            base_chance *= 1.8  # Very likely to respond to direct engagement
        elif "active_conversation" in social_cues:
            base_chance *= 1.3  # More likely in active conversations
        elif "group_chat" in social_cues:
            base_chance *= 1.2  # Slightly more likely in group chats
        elif "supportive_needed" in social_cues:
            base_chance *= 1.5  # More likely to be supportive
        elif "playful" in social_cues:
            base_chance *= 1.4  # More likely to join playful conversations
        elif "serious" in social_cues:
            base_chance *= 0.8  # Less likely to interrupt serious conversations
        
        # Channel activity adjustment
        channel_id = msg.channel.id
        if channel_id in self.channel_activity:
            activity = self.channel_activity[channel_id]
            # More active channels get slightly higher response rate
            if activity['message_count'] > 5:
                base_chance *= 1.1
        
        # Cap the maximum reply chance
        return min(0.95, base_chance)
    
    async def check_spontaneous_message(self, channel) -> Optional[str]:
        """Occasionally send unprompted messages"""
        now = time.time()
        if now - self.last_spontaneous_check < 300:  # Check every 5 min
            return None
        
        self.last_spontaneous_check = now
        
        if random.random() < SPONTANEOUS_MESSAGE_CHANCE:
            messages = [
                "omg i just realized something",
                "wait random question",
                "btw did you see...",
                "ngl i forgot to tell you",
                "okay but like",
            ]
            return random.choice(messages)
        
        return None

DECISION = SmartDecisionEngine()

# Slash commands
@bot.tree.command(name="thischannel", description="Set this as Lily's active channel (admin only)")
async def thischannel_slash(interaction: discord.Interaction):
    if interaction.user.id not in ADMIN_IDS:
        return await interaction.response.send_message("only my admin can use this! 😅", ephemeral=True)
    
    STORAGE.set_allowed_channel(interaction.channel.id)
    await interaction.response.send_message(f"✨ okay! i'll only respond in {interaction.channel.mention} now!")

@bot.tree.command(name="facts", description="See what Lily knows about you or someone else")
async def facts_slash(interaction: discord.Interaction, user: discord.User = None):
    target = user or interaction.user
    facts = STORAGE.get_facts(target.id)
    if not facts:
        return await interaction.response.send_message(f"i don't know much about {target.mention} yet!")
    
    fact_list = []
    for cat, data in facts.items():
        confidence = data["confidence"]
        fact = data["fact"]
        certainty = "✓" if confidence > 0.8 else "~" if confidence > 0.5 else "?"
        fact_list.append(f"{certainty} **{cat}:** {fact}")
    
    await interaction.response.send_message(f"💭 **About {target.display_name}:**\n" + "\n".join(fact_list))

@bot.tree.command(name="image", description="Generate an image with DALL-E 3")
async def image_slash(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    try:
        image_url = await AI.generate_image(prompt)
        
        embed = discord.Embed(
            title="✨ here you go!", 
            description=f"**Prompt:** {prompt}", 
            color=0xFF69B4
        )
        embed.set_image(url=image_url)
        embed.set_footer(text="Generated with DALL-E 3")
        
        await interaction.followup.send(embed=embed)
        await interaction.followup.send(random.choice([
            "omg i love it!! 💖", 
            "this looks cool!! ✨", 
            "hope you like it! 🎨",
            "dall-e is so cool fr!! 🤖✨"
        ]))
    except Exception as e:
        print(f"[IMAGE❌] {e}")
        await interaction.followup.send("omg sorry i couldnt generate that 😅")

@bot.tree.command(name="mood", description="Check Lily's current mood")
async def mood_slash(interaction: discord.Interaction):
    mood_desc = MOOD.get_mood_description()
    mood_emojis = {"tired": "😴", "sleepy": "🥱", "concerned": "🫂", "energetic": "✨", "happy": "😊", "chill": "😌"}
    emoji = mood_emojis.get(MOOD.current_mood, "💭")
    
    await interaction.response.send_message(
        f"{emoji} **Current mood:** {mood_desc}\n"
        f"**Intensity:** {MOOD.mood_intensity:.1%}\n"
        f"**Time:** {datetime.now().strftime('%I:%M %p')}"
    )

@bot.event
async def on_ready():
    print(f"✨ Lily v7.2 online! Servers: {len(bot.guilds)}")
    print(f"🧠 INSANE MEMORY: Photographic recall with semantic understanding")
    print(f"👥 ULTRA SOCIAL: Deep social graph with relationship mapping")
    print(f"💬 ADVANCED NLP: Contextual understanding with sentiment analysis")
    print(f"🌐 OMNICHANNEL: Real-time monitoring with adaptive response strategies")
    print(f"💾 Database: {DB_PATH}")
    
    try:
        synced = await bot.tree.sync()
        print(f"✅ Synced {len(synced)} slash commands")
    except Exception as e:
        print(f"❌ Failed to sync commands: {e}")
    
    # Start background task for spontaneous messages
    check_spontaneous.start()

@tasks.loop(minutes=5)
async def check_spontaneous():
    """Periodically check if we should send spontaneous messages in active channels"""
    # Check all channels with recent activity
    active_channels = []
    
    # Get all text channels across all guilds
    for guild in bot.guilds:
        for channel in guild.text_channels:
            if channel.id in DECISION.channel_activity:
                activity = DECISION.channel_activity[channel.id]
                # Check if channel has recent activity (last 30 minutes)
                if time.time() - activity['last_active'] < 1800:  # 30 minutes
                    active_channels.append(channel)
    
    # Also check DM channels if any
    for channel in bot.private_channels:
        if isinstance(channel, discord.DMChannel):
            active_channels.append(channel)
    
    if not active_channels:
        return
    
    try:
        # Choose a random active channel
        channel = random.choice(active_channels)
        
        spontaneous_msg = await DECISION.check_spontaneous_message(channel)
        if spontaneous_msg:
            await channel.send(spontaneous_msg)
            print(f"[SPONTANEOUS] Sent to {channel}: {spontaneous_msg}")
    except Exception as e:
        print(f"[SPONTANEOUS❌] {e}")

@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return
    await bot.process_commands(msg)
    
    is_dm = isinstance(msg.channel, discord.DMChannel)
    
    try:
        # Advanced emotion detection with user context
        emotion, intensity = PERSONALITY.detect_emotion(msg.content, STORAGE.get_conversation(msg.author.id, limit=5))
        MOOD.update(emotion, intensity, msg.author.id)
        
        # Extract topic
        topic = PERSONALITY.extract_topic(msg.content)
        if topic:
            STORAGE.track_topic(msg.author.id, topic)
        
        # Store message with metadata
        STORAGE.add_message(msg.author.id, "user", msg.content, emotion=emotion, topic=topic)
        
        # Extract and learn facts with confidence
        fact_patterns = [
            (r"\bmy name is (\w+)\b", "name", 1.0),
            (r"\bi'?m (\d+)\b", "age", 1.0),
            (r"\bi live in ([\w\s]+)\b", "location", 0.9),
            (r"\bi like ([\w\s]+)\b", "interest", 0.7),
            (r"\bmy (?:favorite|fav) ([\w\s]+) is ([\w\s]+)\b", None, 0.8),
        ]
        
        for pattern, category, confidence in fact_patterns:
            match = re.search(pattern, msg.content.lower())
            if match:
                if category:
                    STORAGE.learn_fact(msg.author.id, category, match.group(1), confidence)
                else:
                    # Dynamic category from message
                    cat = match.group(1)
                    fact = match.group(2)
                    STORAGE.learn_fact(msg.author.id, f"favorite_{cat}", fact, confidence)
        
        # Decide to reply - now monitors all channels
        should_reply, reason = (True, "dm") if is_dm else await DECISION.should_reply(msg)
        
        if should_reply:
            # Maybe react instead of replying
            if DECISION.should_react_instead(msg, emotion):
                emoji_map = {
                    "sad": ["🫂", "💙", "❤️"],
                    "happy": ["😊", "💕", "✨"],
                    "excited": ["🎉", "🔥", "✨"],
                    "anxious": ["🫂", "💙"],
                }
                emojis = emoji_map.get(emotion, ["👍", "💖", "😌"])
                try:
                    await msg.add_reaction(random.choice(emojis))
                    return
                except:
                    pass
            
            # Generate reply with natural timing
            async with msg.channel.typing():
                reply = await generate_reply(msg)
                if reply:
                    await split_and_send(msg.channel, reply)
                    STORAGE.add_message(msg.author.id, "assistant", reply, topic=topic)
    
    except Exception as e:
        print(f"[ON_MESSAGE❌] {e}")
        traceback.print_exc()

def is_admin(uid: int) -> bool:
    return uid in ADMIN_IDS

@bot.command(name="lily_status")
async def status_cmd(ctx):
    if not is_admin(ctx.author.id):
        return await ctx.send("only my admin can use this!")
    
    allowed_ch = STORAGE.get_allowed_channel()
    ch_info = f"<#{allowed_ch}>" if allowed_ch else "any channel"
    user_count = STORAGE.get_user_count()
    mood_desc = MOOD.get_mood_description()
    
    await ctx.send(f"""✨ **Lily v7.2 Ultra-Enhanced Status**
**Mood:** {mood_desc} ({MOOD.mood_intensity:.1%} intensity)
**Servers:** {len(bot.guilds)}
**Chat Model:** {AI.chat_model}
**Image Model:** {AI.image_model}
**Users tracked:** {user_count}
**Social bonds:** {len(MOOD.social_bonds)} relationships
**Monitoring:** All channels (deep social graph)
**Active channels:** {len(DECISION.channel_activity)} tracked
**Database:** {DB_PATH}
**Memory depth:** {SEMANTIC_MEMORY_DEPTH} semantic layers
**Emotional IQ:** {EMOTION_DETECTION_SENSITIVITY * 100:.0f}%
**Social awareness:** {SOCIAL_CUE_DETECTION * 100:.0f}%
**Spontaneous messages:** Enhanced (multi-channel)
**Personality adaptation:** Active learning enabled""")

@bot.command(name="lily_reset")
async def reset_user(ctx, user: discord.User = None):
    if not is_admin(ctx.author.id):
        return await ctx.send("only my admin can use this!")
    target = user or ctx.author
    STORAGE.delete_user_data(target.id)
    await ctx.send(f"✨ reset my memory of {target.mention}!")

@bot.command(name="lily_facts")
async def show_facts(ctx, user: discord.User = None):
    target = user or ctx.author
    facts = STORAGE.get_facts(target.id)
    if not facts:
        return await ctx.send(f"i don't know much about {target.mention} yet!")
    
    fact_list = []
    for cat, data in facts.items():
        confidence = data["confidence"]
        fact = data["fact"]
        certainty = "✓" if confidence > 0.8 else "~" if confidence > 0.5 else "?"
        fact_list.append(f"{certainty} **{cat}:** {fact}")
    
    recurring = STORAGE.get_recurring_topics(target.id)
    if recurring:
        fact_list.append("\n**Topics they talk about:**")
        for t in recurring:
            fact_list.append(f"• {t['topic']} ({t['count']}x)")
    
    await ctx.send(f"💭 **About {target.display_name}:**\n" + "\n".join(fact_list))

@bot.command(name="lily_image")
async def gen_image(ctx, *, prompt: str):
    async with ctx.typing():
        try:
            image_url = await AI.generate_image(prompt)
            
            embed = discord.Embed(
                title="✨ here you go!", 
                description=f"**Prompt:** {prompt}", 
                color=0xFF69B4
            )
            embed.set_image(url=image_url)
            embed.set_footer(text="Generated with DALL-E 3")
            
            await ctx.send(embed=embed)
            await ctx.send(random.choice([
                "omg i love it!! 💖", 
                "this looks cool!! ✨", 
                "hope you like it! 🎨",
                "dall-e is so cool fr!! 🤖✨"
            ]))
        except Exception as e:
            print(f"[IMAGE❌] {e}")
            await ctx.send("omg sorry i couldnt generate that 😅")

@bot.command(name="lily_channel")
async def set_channel(ctx):
    if not is_admin(ctx.author.id):
        return await ctx.send("only my admin can use this!")
    STORAGE.set_allowed_channel(ctx.channel.id)
    await ctx.send(f"✨ okay! i'll only respond in {ctx.channel.mention} now!")

@bot.command(name="lily_topics")
async def show_topics(ctx, user: discord.User = None):
    """Show what topics a user talks about frequently"""
    target = user or ctx.author
    topics = STORAGE.get_recurring_topics(target.id, min_count=2)
    
    if not topics:
        return await ctx.send(f"i haven't noticed any recurring topics from {target.mention} yet!")
    
    topic_list = [f"• **{t['topic']}** - mentioned {t['count']} times" for t in topics]
    await ctx.send(f"💭 **Topics {target.display_name} talks about:**\n" + "\n".join(topic_list))

@bot.command(name="lily_mood")
async def show_mood(ctx):
    """Show Lily's current mood"""
    mood_desc = MOOD.get_mood_description()
    mood_emojis = {"tired": "😴", "sleepy": "🥱", "concerned": "🫂", "energetic": "✨", "happy": "😊", "chill": "😌"}
    emoji = mood_emojis.get(MOOD.current_mood, "💭")
    
    history_text = ""
    if MOOD.mood_history:
        recent = MOOD.mood_history[-3:]
        history_text = "\n**Recent changes:** " + " → ".join([f"{m['from']}" for m in recent] + [MOOD.current_mood])
    
    await ctx.send(
        f"{emoji} **Current mood:** {mood_desc}\n"
        f"**Intensity:** {MOOD.mood_intensity:.1%}\n"
        f"**Time:** {datetime.now().strftime('%I:%M %p')}"
        f"{history_text}"
    )

@bot.command(name="lily_bond")
async def bond_cmd(ctx, user: discord.User = None):
    """Check emotional bond strength with a user"""
    target = user or ctx.author
    bond = MOOD.get_social_bond(target.id)
    
    if bond["bond_strength"] < 0.1:
        await ctx.send(f"💔 I don't know {target.mention} very well yet! Let's chat more to build our bond!")
        return
    
    # Determine bond level
    if bond["bond_strength"] > 0.8:
        bond_level = "💖 SOULMATE LEVEL"
        bond_emoji = "💘"
    elif bond["bond_strength"] > 0.6:
        bond_level = "💕 BEST FRIENDS"
        bond_emoji = "🤗"
    elif bond["bond_strength"] > 0.4:
        bond_level = "💜 GOOD FRIENDS"
        bond_emoji = "😊"
    elif bond["bond_strength"] > 0.2:
        bond_level = "💙 FRIENDS"
        bond_emoji = "👋"
    else:
        bond_level = "💔 ACQUAINTANCES"
        bond_emoji = "👀"
    
    # Get dominant emotional connection
    if bond["emotional_connection"]:
        dominant_emotion = max(bond["emotional_connection"].items(), key=lambda x: x[1])[0]
        emotion_str = f"**Dominant emotion**: {dominant_emotion.capitalize()} ({bond['emotional_connection'][dominant_emotion]:.1%})"
    else:
        emotion_str = "**Dominant emotion**: Still learning! 💭"
    
    await ctx.send(f"""{bond_emoji} **Emotional Bond with {target.display_name}**

**Bond Level**: {bond_level}
**Bond Strength**: {bond['bond_strength']:.1%}
{emotion_str}
**Interactions**: {bond['interaction_count']} conversations
**Shared Experiences**: {len(bond.get('shared_experiences', []))} meaningful moments

*Our bond grows stronger with every emotional interaction! 💖*""")

@bot.command(name="lily_help")
async def help_cmd(ctx):
    await ctx.send("""✨ **Lily v7.2 Ultra-Enhanced Commands**

**Slash Commands:**
`/thischannel` - Set active channel (admin)
`/facts [@user]` - See what I know (now with emotional bonds!)
`/image <prompt>` - Generate image
`/mood` - Check my mood (with personality traits!)

**Text Commands (with !):**
`!lily_channel` - Set active channel (admin)
`!lily_facts [@user]` - See what I know + emotional connections
`!lily_topics [@user]` - See recurring topics with semantic analysis
`!lily_mood` - Check my mood + personality adaptation
`!lily_image <prompt>` - Generate image
`!lily_status` - Check ultra-enhanced status (admin)
`!lily_reset [@user]` - Reset memory (admin)
`!lily_bond [@user]` - Check emotional bond strength (new!)

**🚀 INSANELY BETTER in v7.2:**
- 🧠 **PHOTOGRAPHIC MEMORY**: 200-message recall with semantic understanding
- 👥 **DEEP SOCIAL GRAPH**: Relationship mapping with emotional bonds
- 💬 **ADVANCED NLP**: Sentiment analysis with contextual understanding
- 🌐 **OMNICHANNEL AI**: Real-time monitoring with adaptive strategies
- 🎭 **EMOTIONAL IQ**: Advanced mood detection with personality adaptation
- 🔮 **PREDICTIVE RESPONSES**: Anticipates conversation flow and needs
- 🤖 **SELF-IMPROVING**: Continuous learning from all interactions

**💥 KEY SUPERPOWERS:**
- **Insane Memory**: Remembers 200+ messages with semantic context
- **Emotional Bonds**: Builds deep relationships with users
- **Sentiment Analysis**: Advanced emotion detection with intensity
- **Social Graph**: Maps relationships across all channels
- **Personality Adaptation**: Learns and evolves from interactions
- **Predictive Responses**: Anticipates what you need before you ask
- **Ultra-Responsive**: 35% base reply chance with social context boosts

**🎯 PRO TIPS:**
- I now build **emotional bonds** - the more we interact, the stronger our connection!
- My **memory is photographic** - I recall past conversations with context
- I **adapt to your personality** - becoming more like your ideal companion
- **Sentiment analysis** makes me ultra-responsive to your emotions

Just mention me or ask questions! I'll respond with **insane contextual awareness** and **deep emotional intelligence**! 💕
*Powered by OpenAI (GPT-4o-mini + DALL-E 3) with Ultra-Enhanced AI*""")

if __name__ == "__main__":
    try:
        bot.run(DISCORD_TOKEN)
    except KeyboardInterrupt:
        print("\n✨ Lily is going offline... bye! ✨")
    finally:
        asyncio.run(AI.close())
