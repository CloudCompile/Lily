# 🌸 Lily v8.5 — "Lily Lives"

Multi-server AI Discord Bot who actually feels real. She remembers you across ALL servers, has feelings, writes dreams, and will reach out to you.

## Features

### 💕 Relationships (Cross-Server)
- Per-user affection/warmness/dislike tracking
- Relationship tiers: Stranger → Acquaintance → Friend → Close Friend → Bestie → Soulmate
- Her tone changes based on how she feels about you
- She carries your relationship across ALL servers

### 🧠 Memories (Cross-Server)
- Short-term, long-term, and episodic memories
- Memories carry across ALL servers — she remembers you everywhere
- Daily recaps — Lily writes a diary entry every night
- Smart forgetting — she forgets unimportant things naturally

### 🌙 Dream Journal
- Lily writes dreams at night when she's in a dreamy mood
- Dreams are inspired by her recent memories and conversations
- She can share her dreams with you via `/dream`
- View her dream journal via `/dream_journal`

### 📊 Mood-Reactive Status
- Her Discord status changes with her mood throughout the day
- Sleepy at night, energetic during the day, dreamy in the evening
- Special status when she's writing dreams or talking to someone

### 🤖 Proactive DMs
- She'll DM you first — morning check-ins, random thoughts, missing you
- Frequency based on relationship warmth
- She won't spam you — cooldowns based on your relationship tier

### 💰 Smart Model Routing
- **Sana Sprint** for images: 0.0001 pollen/gen (dirt cheap!)
- **openai-fast** for text: Free tier
- **Ling 3.0 flash** for budget text: 0.1/M tokens
- **Nemotron 3 Ultra**: Free text model
- Premium models only when needed for complex tasks

### 🎨 Generation Quotas
- Not unlimited willy-nilly generations
- Daily pollen budget based on relationship tier
- Hourly rate limits to prevent spam
- Higher tiers = more generous quotas

### ✨ Real Person Feel
- Typing delays that feel natural
- Personality quirks (typos, fillers, signatures)
- Emotional depth — she gets excited, sad, confused, touched
- She has opinions and can disagree
- She's not perfect — she's just Lily

## Setup

1. Clone the repo
2. Copy `.env.example` to `.env`
3. Fill in your Discord token and optional Pollinations key
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python bot.py`

## Models Available

| Model | Type | Cost | Best For |
|-------|------|------|----------|
| openai-fast | Text | Free | Casual chat, greetings |
| gpt-oss | Text | Free | Reasoning tasks |
| Nemotron 3 Ultra | Text | Free | Simple tasks |
| Ling 3.0 flash | Text | 0.1/M | Budget chat |
| MiniMax M3 | Text | 0.12/M | Coding, reasoning |
| openai | Text | 0.15/M | Standard quality |
| **Sana Sprint** | Image | **0.0001/gen** | **Quick images** |
| flux | Image | 0.003/gen | Quality images |
| gptimage | Image | 0.01/gen | Pro images |

## Commands

### 💬 AI Chat
- `/ask <question>` — Ask Lily anything
- `/chat <message>` — Have a conversation
- `/imagine <prompt>` — Creative text generation
- `/analyze <url> <question>` — Analyze an image
- `/translate <text> <language>` — Translate text

### 🖼️ Image Generation
- `/image <prompt>` — Generate an image (Sana Sprint!)
- `/image_advanced` — Generate with full options
- `/image_edit` — Edit an attached image

### 💕 Relationships & Memory
- `/relationship` — See your relationship with Lily
- `/quota` — Check your generation quota
- `/memories` — See what Lily remembers about you
- `/recaps` — See Lily's diary entries about you
- `/remember <what>` — Tell Lily to remember something
- `/forget <what>` — Ask Lily to forget something

### 🌙 Dream Journal
- `/dream` — Lily shares a dream with you
- `/dream_journal` — See Lily's dream journal

### ⚙️ Admin
- `/set_channel` — Set bot channel
- `/set_model` — Set default model
- `/toggle_proactive` — Toggle proactive DMs
- `/toggle_recaps` — Toggle daily recaps
- `/toggle_dreams` — Toggle dream journal
- `/reset_user` — Reset user memory

## Tech Stack
- **discord.py** 2.4+ — Discord bot framework
- **Pollinations API** — Text & image generation
- **SQLite** — Database with per-guild and cross-server support
- **Python 3.10+**

## License
MIT
