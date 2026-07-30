#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v9.0 — Pollinations API Client

Focused async client for the Pollinations AI API.
Covers: Text Generation, Image Generation, Image Editing, Models, Embeddings.
NO 3D, audio, or video generation — that's not who Lily is.
"""

from __future__ import annotations
import asyncio
import hashlib
import io
import json
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

import aiohttp

from config import (
    POLLINATIONS_KEY,
    POLLINATIONS_BASE_URL,
    POLLINATIONS_MEDIA_URL,
)

log = logging.getLogger("lily.pollinations")

# ── Retry Configuration ────────────────────────────────────
MAX_RETRIES = 2          # Number of retries on failure
RETRY_DELAY = 1.0        # Seconds to wait between retries
RETRY_BACKOFF = 2.0      # Multiplier for each retry (1s, 2s, 4s...)


class PollinationsAPI:
    """Async client for the Pollinations gen API — text and image only."""

    def __init__(
        self,
        api_key: str = POLLINATIONS_KEY,
        base_url: str = POLLINATIONS_BASE_URL,
        media_url: str = POLLINATIONS_MEDIA_URL,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.media_url = media_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    # ── Session lifecycle ────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _headers(self, extra: dict = None) -> dict:
        """Build auth headers."""
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            h.update(extra)
        return h

    # ══════════════════════════════════════════════════════
    #  TEXT GENERATION
    # ══════════════════════════════════════════════════════

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        model: str = "openai",
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        safe: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        user: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v1/chat/completions — OpenAI-compatible chat completions with retry."""
        session = await self._get_session()
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if seed is not None:
            body["seed"] = seed
        if response_format is not None:
            body["response_format"] = response_format
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if safe is not None:
            body["safe"] = safe
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if user is not None:
            body["user"] = user

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers(),
                    json=body,
                ) as resp:
                    if resp.status >= 500:
                        error_text = await resp.text()
                        last_error = f"Server error {resp.status}: {error_text[:300]}"
                        log.warning(f"Chat retry {attempt + 1}/{MAX_RETRIES}: {last_error}")
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * (RETRY_BACKOFF ** attempt))
                            continue
                        raise Exception(last_error)
                    resp.raise_for_status()
                    return await resp.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = str(e)
                log.warning(f"Chat retry {attempt + 1}/{MAX_RETRIES}: Connection error: {last_error}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (RETRY_BACKOFF ** attempt))
                    continue

        raise Exception(f"Chat completions failed after {MAX_RETRIES + 1} attempts. Last error: {last_error}")

    async def chat_completions_simple(
        self,
        messages: List[Dict[str, Any]],
        model: str = "openai",
        **kwargs,
    ) -> str:
        """Simplified chat completions — returns just the assistant text."""
        result = await self.chat_completions(messages, model, **kwargs)
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return ""

    async def text_generate(
        self,
        messages: List[Dict[str, Any]],
        model: str = "openai",
        **kwargs,
    ) -> str:
        """POST /text — Returns assistant content directly."""
        session = await self._get_session()
        body: Dict[str, Any] = {"model": model, "messages": messages, **kwargs}
        async with session.post(
            f"{self.base_url}/text",
            headers=self._headers(),
            json=body,
        ) as resp:
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct:
                data = await resp.json()
                if isinstance(data, dict) and "choices" in data:
                    return data["choices"][0]["message"]["content"]
                return json.dumps(data)
            return await resp.text()

    async def text_simple(
        self,
        prompt: str,
        model: str = "openai",
        *,
        seed: int = 0,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        safe: Optional[str] = None,
    ) -> str:
        """GET /text/{prompt} — Simple text generation via GET."""
        session = await self._get_session()
        params: Dict[str, Any] = {"model": model, "seed": seed}
        if system:
            params["system"] = system
        if json_mode:
            params["json"] = "true"
        if temperature is not None:
            params["temperature"] = temperature
        if safe is not None:
            params["safe"] = safe
        if self.api_key:
            params["key"] = self.api_key

        from urllib.parse import quote
        encoded_prompt = quote(prompt, safe="")

        async with session.get(
            f"{self.base_url}/text/{encoded_prompt}",
            params=params,
        ) as resp:
            resp.raise_for_status()
            return await resp.text()

    # ══════════════════════════════════════════════════════
    #  IMAGE GENERATION
    # ══════════════════════════════════════════════════════

    async def image_generate(
        self,
        prompt: str,
        *,
        model: str = "flux",
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        nologo: bool = True,
        enhance: bool = False,
        safe: Optional[str] = None,
        quality: Optional[str] = None,
        image: Optional[str] = None,
        transparent: bool = False,
    ) -> bytes:
        """Generate an image via GET endpoint. Returns PNG bytes. Retries on failure."""
        session = await self._get_session()
        from urllib.parse import quote
        encoded_prompt = quote(prompt, safe="")

        params: Dict[str, Any] = {
            "model": model,
            "width": width,
            "height": height,
            "seed": seed,
            "nologo": nologo,
            "enhance": enhance,
            "transparent": transparent,
        }
        if safe:
            params["safe"] = safe
        if quality:
            params["quality"] = quality
        if image:
            params["image"] = image
        if self.api_key:
            params["key"] = self.api_key

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with session.get(
                    f"{self.base_url}/image/{encoded_prompt}",
                    params=params,
                ) as resp:
                    if resp.status >= 500:
                        last_error = f"Image server error {resp.status}"
                        log.warning(f"Image retry {attempt + 1}/{MAX_RETRIES}: {last_error}")
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * (RETRY_BACKOFF ** attempt))
                            continue
                        raise Exception(last_error)
                    resp.raise_for_status()
                    return await resp.read()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = str(e)
                log.warning(f"Image retry {attempt + 1}/{MAX_RETRIES}: Connection error: {last_error}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (RETRY_BACKOFF ** attempt))
                    continue

        raise Exception(f"Image generation failed after {MAX_RETRIES + 1} attempts. Last error: {last_error}")

    async def image_generate_post(
        self,
        prompt: str,
        *,
        model: str = "flux",
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        nologo: bool = True,
        enhance: bool = False,
        safe: Optional[str] = None,
        quality: Optional[str] = None,
        image: Optional[str] = None,
        transparent: bool = False,
    ) -> Dict[str, Any]:
        """POST /image — Generate image with full options. Returns JSON with image data."""
        session = await self._get_session()
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "nologo": nologo,
            "enhance": enhance,
            "transparent": transparent,
        }
        if safe:
            body["safe"] = safe
        if quality:
            body["quality"] = quality
        if image:
            body["image"] = image

        async with session.post(
            f"{self.base_url}/image",
            headers=self._headers(),
            json=body,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def image_edit(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        model: str = "kontext",
        filename: str = "image.png",
        safe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /image — Edit an existing image with a prompt."""
        session = await self._get_session()
        import aiohttp

        data = aiohttp.FormData()
        data.add_field("image", image_bytes, filename=filename, content_type="image/png")
        data.add_field("prompt", prompt)
        data.add_field("model", model)
        if safe:
            data.add_field("safe", safe)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with session.post(
            f"{self.base_url}/image",
            headers=headers,
            data=data,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ══════════════════════════════════════════════════════
    #  IMAGE ANALYSIS (via vision-capable text models)
    # ══════════════════════════════════════════════════════

    async def analyze_image(
        self,
        image_url: str,
        question: str = "What is in this image?",
        model: str = "openai",
    ) -> str:
        """Analyze an image using a vision-capable text model."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        return await self.chat_completions_simple(
            messages, model=model, max_tokens=500
        )

    # ══════════════════════════════════════════════════════
    #  EMBEDDINGS
    # ══════════════════════════════════════════════════════

    async def create_embedding(
        self,
        input_text: str,
        model: str = "openai-3-small",
    ) -> Dict[str, Any]:
        """POST /v1/embeddings — Create text embeddings."""
        session = await self._get_session()
        body = {"model": model, "input": input_text}
        async with session.post(
            f"{self.base_url}/v1/embeddings",
            headers=self._headers(),
            json=body,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ══════════════════════════════════════════════════════
    #  MODELS
    # ══════════════════════════════════════════════════════

    async def list_models(self) -> List[Dict[str, Any]]:
        """GET /models — List all available models."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/models",
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get info about a specific model."""
        models = await self.list_models()
        for m in models:
            if m.get("name") == model_id or model_id in m.get("aliases", []):
                return m
        return None

    # ══════════════════════════════════════════════════════
    #  ACCOUNT
    # ══════════════════════════════════════════════════════

    async def get_balance(self) -> Dict[str, Any]:
        """GET /v1/balance — Check API balance."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/v1/balance",
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ══════════════════════════════════════════════════════
    #  MEDIA STORAGE
    # ══════════════════════════════════════════════════════

    async def upload_media(
        self,
        file_bytes: bytes,
        filename: str = "upload.png",
        content_type: str = "image/png",
    ) -> Dict[str, Any]:
        """POST /media — Upload a file to Pollinations media storage."""
        session = await self._get_session()
        import aiohttp as aio

        data = aio.FormData()
        data.add_field("file", file_bytes, filename=filename, content_type=content_type)

        async with session.post(
            f"{self.media_url}/media",
            headers=self._headers(),
            data=data,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ══════════════════════════════════════════════════════
    #  UTILITY
    # ══════════════════════════════════════════════════════

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Create a hash of a prompt for logging/analytics."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]


