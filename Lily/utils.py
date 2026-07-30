#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Shared Utilities

Response chunking, retry logic with fallback models,
and cross-server memory helpers.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Optional, List, Dict, Any

log = logging.getLogger("lily.utils")

# ── Response Chunking ──────────────────────────────────────

MAX_DISCORD_MSG = 2000
CHUNK_OVERLAP = 50  # Characters to overlap when splitting


def chunk_response(text: str, max_len: int = MAX_DISCORD_MSG) -> List[str]:
    """Split a long response into Discord-safe chunks.

    Tries to split on paragraph breaks, then sentence breaks,
    then word breaks, then character breaks — in that order.
    Never sends an empty chunk.
    """
    if not text or not text.strip():
        return []

    if len(text) <= max_len:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        # Try to split at a paragraph break
        split_at = -1
        for sep in ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]:
            # Look for the separator within the last 200 chars of the chunk
            search_start = max(0, max_len - 200)
            idx = remaining.rfind(sep, search_start, max_len)
            if idx > 0:
                split_at = idx + len(sep)
                break

        if split_at <= 0:
            # Hard split at max_len
            split_at = max_len

        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    return chunks


async def send_chunked(
    target, text: str, max_len: int = MAX_DISCORD_MSG, **kwargs
) -> List[Any]:
    """Send a potentially long response as multiple messages.

    Works with Context, Webhook, or any object with .send().

    Args:
        target: A discord Context, Webhook, or any object with .send()
        text: The text to send
        max_len: Max characters per message
        **kwargs: Additional kwargs passed to send()

    Returns:
        List of message objects
    """
    chunks = chunk_response(text, max_len)
    messages = []

    for i, chunk in enumerate(chunks):
        try:
            if i == 0:
                msg = await target.send(chunk, **kwargs)
            else:
                msg = await target.send(chunk)
            messages.append(msg)
        except Exception as e:
            log.error(f"Failed to send chunk {i}/{len(chunks)}: {e}")
            break

    return messages


# ── Retry Logic with Fallback Models ──────────────────────

# Model fallback chain — if the primary model fails, try these in order
FALLBACK_CHAIN = [
    "openai-fast",   # Free tier — always try this first as fallback
    "openai",        # Standard — reliable
    "gpt-oss",       # Free reasoning — last resort
]

# Models that should NOT be used as fallbacks (too expensive or specialized)
NO_FALLBACK_MODELS = {"gemini-search", "perplexity-fast", "sana", "flux", "gptimage", "kontext"}


async def generate_with_retry(
    api,
    messages: List[Dict[str, Any]],
    primary_model: str,
    max_retries: int = 2,
    **kwargs,
) -> str:
    """Generate a response with automatic retry and fallback model routing.

    Args:
        api: PollinationsAPI instance
        messages: Chat messages for the API
        primary_model: The model to try first
        max_retries: How many times to retry (with fallback models)
        **kwargs: Additional kwargs for chat_completions_simple

    Returns:
        The generated text, or empty string if all attempts fail
    """
    # Build the model chain: primary first, then fallbacks
    model_chain = [primary_model]
    for fallback in FALLBACK_CHAIN:
        if fallback not in model_chain:
            model_chain.append(fallback)

    last_error = None
    attempts = 0

    for model in model_chain:
        if attempts >= max_retries + 1:  # +1 for the primary attempt
            break

        try:
            response = await api.chat_completions_simple(
                messages, model=model, **kwargs
            )

            if response and response.strip():
                if model != primary_model:
                    log.info(f"Fallback to {model} succeeded after {primary_model} failed")
                return response

            # Empty response — try next model
            log.warning(f"Empty response from model {model}, trying fallback...")
            last_error = "empty_response"

        except Exception as e:
            log.warning(f"Model {model} failed: {e}, trying fallback...")
            last_error = str(e)

        attempts += 1

    # All attempts failed
    log.error(f"All models failed for primary={primary_model}. Last error: {last_error}")
    return ""


# ── Cross-Server Memory Helpers ────────────────────────────

def build_cross_server_context(
    conversations: List[Dict],
    user_facts: List[Dict],
    memories_str: str = "",
    limit: int = 15,
) -> List[Dict[str, str]]:
    """Build a cross-server conversation context for the API.

    Merges conversations from different guilds into a single message list,
    deduplicating by timestamp and keeping the most recent ones.

    Args:
        conversations: List of conversation dicts with 'role' and 'content'
        user_facts: List of fact dicts about the user
        memories_str: Formatted memory context string
        limit: Max number of messages to include

    Returns:
        List of API-format message dicts
    """
    # Deduplicate conversations by content (avoid repeating the same message)
    seen = set()
    unique = []
    for conv in conversations:
        content = conv.get("content", "")
        key = (conv.get("role", ""), content[:100])
        if key not in seen:
            seen.add(key)
            unique.append(conv)

    # Keep only the most recent
    recent = unique[-limit:]

    return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in recent]
