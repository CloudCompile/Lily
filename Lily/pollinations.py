#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Pollinations API Client

Full-featured async client for the Pollinations AI API.
Covers: Text, Image, Video, Audio (TTS + Music + Transcription),
3D, Embeddings, Models, Media Storage, Account, and Realtime.
"""

from __future__ import annotations
import asyncio
import hashlib
import io
import json
import logging
from typing import Optional, List, Dict, Any, AsyncIterator, Union
from pathlib import Path

import aiohttp

from config import (
    POLLINATIONS_KEY,
    POLLINATIONS_BASE_URL,
    POLLINATIONS_MEDIA_URL,
)

log = logging.getLogger("lily.pollinations")


class PollinationsAPI:
    """Async client for the Pollinations gen API."""

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

    def _safe_param(self, safe: Optional[str] = None) -> Optional[str]:
        """Return the safe query param if provided."""
        return safe if safe else None

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
        stream: bool = False,
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
        """POST /v1/chat/completions — OpenAI-compatible chat completions."""
        session = await self._get_session()
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
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

        async with session.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(),
            json=body,
        ) as resp:
            resp.raise_for_status()
            if stream:
                return resp  # Return the response for streaming
            return await resp.json()

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
        url = f"{self.base_url}/text/{quote(prompt, safe='')}"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.text()

    # ══════════════════════════════════════════════════════
    #  IMAGE GENERATION
    # ══════════════════════════════════════════════════════

    async def image_generate(
        self,
        prompt: str,
        model: str = "zimage",
        *,
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        safe: Optional[str] = None,
        quality: Optional[str] = None,
        image: Optional[str] = None,
        transparent: bool = False,
    ) -> bytes:
        """GET /image/{prompt} — Generate image, returns raw image bytes."""
        session = await self._get_session()
        params: Dict[str, Any] = {
            "model": model,
            "width": width,
            "height": height,
            "seed": seed,
            "transparent": str(transparent).lower(),
        }
        if safe is not None:
            params["safe"] = safe
        if quality is not None:
            params["quality"] = quality
        if image is not None:
            params["image"] = image
        if self.api_key:
            params["key"] = self.api_key

        from urllib.parse import quote
        url = f"{self.base_url}/image/{quote(prompt, safe='')}"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def image_generate_openai(
        self,
        prompt: str,
        model: str = "flux",
        *,
        size: str = "1024x1024",
        quality: str = "medium",
        response_format: str = "b64_json",
        safe: Optional[str] = None,
        image: Optional[Union[str, List[str]]] = None,
        user: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v1/images/generations — OpenAI-compatible image generation."""
        session = await self._get_session()
        body: Dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "size": size,
            "quality": quality,
            "response_format": response_format,
        }
        if safe is not None:
            body["safe"] = safe
        if image is not None:
            body["image"] = image
        if user is not None:
            body["user"] = user

        async with session.post(
            f"{self.base_url}/v1/images/generations",
            headers=self._headers(),
            json=body,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def image_edit(
        self,
        image_file: bytes,
        prompt: str,
        model: str = "kontext",
        *,
        size: str = "1024x1024",
        filename: str = "image.png",
    ) -> Dict[str, Any]:
        """POST /v1/images/edits — OpenAI-compatible image editing."""
        session = await self._get_session()
        data = aiohttp.FormData()
        data.add_field("image", image_file, filename=filename, content_type="image/png")
        data.add_field("prompt", prompt)
        data.add_field("model", model)
        data.add_field("size", size)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with session.post(
            f"{self.base_url}/v1/images/edits",
            headers=headers,
            data=data,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ══════════════════════════════════════════════════════
    #  VIDEO GENERATION
    # ══════════════════════════════════════════════════════

    async def video_generate(
        self,
        prompt: str,
        model: str = "veo",
        *,
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        safe: Optional[str] = None,
        image: Optional[str] = None,
        duration: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        audio: bool = False,
    ) -> bytes:
        """GET /video/{prompt} — Generate video, returns raw MP4 bytes."""
        session = await self._get_session()
        params: Dict[str, Any] = {
            "model": model,
            "width": width,
            "height": height,
            "seed": seed,
            "audio": str(audio).lower(),
        }
        if safe is not None:
            params["safe"] = safe
        if image is not None:
            params["image"] = image
        if duration is not None:
            params["duration"] = duration
        if aspect_ratio is not None:
            params["aspectRatio"] = aspect_ratio
        if self.api_key:
            params["key"] = self.api_key

        from urllib.parse import quote
        url = f"{self.base_url}/video/{quote(prompt, safe='')}"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.read()

    # ══════════════════════════════════════════════════════
    #  AUDIO — TTS, MUSIC, TRANSCRIPTION
    # ══════════════════════════════════════════════════════

    async def tts(
        self,
        text: str,
        *,
        voice: str = "alloy",
        model: str = "elevenlabs",
        response_format: str = "mp3",
        speed: Optional[float] = None,
    ) -> bytes:
        """POST /v1/audio/speech — OpenAI-compatible TTS."""
        session = await self._get_session()
        body: Dict[str, Any] = {
            "input": text,
            "voice": voice,
            "model": model,
            "response_format": response_format,
        }
        if speed is not None:
            body["speed"] = speed

        async with session.post(
            f"{self.base_url}/v1/audio/speech",
            headers=self._headers(),
            json=body,
        ) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def tts_simple(
        self,
        text: str,
        *,
        voice: str = "alloy",
        model: Optional[str] = None,
        response_format: str = "mp3",
        seed: Optional[int] = None,
        safe: Optional[str] = None,
    ) -> bytes:
        """GET /audio/{text} — Simple TTS via GET."""
        session = await self._get_session()
        params: Dict[str, Any] = {"voice": voice, "response_format": response_format}
        if model:
            params["model"] = model
        if seed is not None:
            params["seed"] = seed
        if safe is not None:
            params["safe"] = safe
        if self.api_key:
            params["key"] = self.api_key

        from urllib.parse import quote
        url = f"{self.base_url}/audio/{quote(text, safe='')}"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def music_generate(
        self,
        prompt: str,
        *,
        model: str = "elevenmusic",
        duration: Optional[int] = None,
        instrumental: bool = False,
        response_format: str = "mp3",
        seed: Optional[int] = None,
        safe: Optional[str] = None,
    ) -> bytes:
        """Generate music via GET /audio/{text} with a music model."""
        session = await self._get_session()
        params: Dict[str, Any] = {
            "model": model,
            "response_format": response_format,
            "instrumental": str(instrumental).lower(),
        }
        if duration is not None:
            params["duration"] = str(duration)
        if seed is not None:
            params["seed"] = seed
        if safe is not None:
            params["safe"] = safe
        if self.api_key:
            params["key"] = self.api_key

        from urllib.parse import quote
        url = f"{self.base_url}/audio/{quote(prompt, safe='')}"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def music_upload_reference(
        self,
        file_bytes: bytes,
        filename: str = "reference.mp3",
        extract_composition_plan: bool = False,
    ) -> Dict[str, Any]:
        """POST /v1/audio/music/upload — Upload music reference for conditioning."""
        session = await self._get_session()
        data = aiohttp.FormData()
        data.add_field("file", file_bytes, filename=filename, content_type="audio/mpeg")
        data.add_field("extract_composition_plan", str(extract_composition_plan).lower())

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with session.post(
            f"{self.base_url}/v1/audio/music/upload",
            headers=headers,
            data=data,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        model: str = "whisper",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        filename: str = "audio.mp3",
        speakers_expected: Optional[int] = None,
    ) -> Dict[str, Any]:
        """POST /v1/audio/transcriptions — Transcribe audio to text."""
        session = await self._get_session()
        data = aiohttp.FormData()
        data.add_field("file", audio_bytes, filename=filename, content_type="audio/mpeg")
        data.add_field("model", model)
        data.add_field("response_format", response_format)
        data.add_field("temperature", str(temperature))
        if language:
            data.add_field("language", language)
        if prompt:
            data.add_field("prompt", prompt)
        if speakers_expected:
            data.add_field("speakers_expected", str(speakers_expected))

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with session.post(
            f"{self.base_url}/v1/audio/transcriptions",
            headers=headers,
            data=data,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ══════════════════════════════════════════════════════
    #  3D MODEL GENERATION
    # ══════════════════════════════════════════════════════

    async def generate_3d(
        self,
        prompt: str,
        model: str = "trellis-2-low",
        *,
        image: Optional[str] = None,
        seed: Optional[int] = None,
        safe: Optional[str] = None,
    ) -> bytes:
        """GET /3d/{prompt} — Generate a 3D model (GLB)."""
        session = await self._get_session()
        params: Dict[str, Any] = {"model": model}
        if image is not None:
            params["image"] = image
        if seed is not None:
            params["seed"] = seed
        if safe is not None:
            params["safe"] = safe
        if self.api_key:
            params["key"] = self.api_key

        from urllib.parse import quote
        url = f"{self.base_url}/3d/{quote(prompt, safe='')}"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.read()

    # ══════════════════════════════════════════════════════
    #  EMBEDDINGS
    # ══════════════════════════════════════════════════════

    async def create_embeddings(
        self,
        input_text: Union[str, List[str], Dict, List[Dict]],
        *,
        model: str = "openai-3-small",
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
        task_type: Optional[str] = None,
        input_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v1/embeddings — Generate vector embeddings."""
        session = await self._get_session()
        body: Dict[str, Any] = {
            "input": input_text,
            "model": model,
            "encoding_format": encoding_format,
        }
        if dimensions is not None:
            body["dimensions"] = dimensions
        if task_type is not None:
            body["task_type"] = task_type
        if input_type is not None:
            body["input_type"] = input_type

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
        """GET /models — All models with full metadata."""
        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with session.get(
            f"{self.base_url}/models", headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def list_models_openai(self) -> Dict[str, Any]:
        """GET /v1/models — OpenAI-compatible model list."""
        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with session.get(
            f"{self.base_url}/v1/models", headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def list_text_models(self) -> List[Dict[str, Any]]:
        """GET /text/models — Text models with details."""
        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with session.get(
            f"{self.base_url}/text/models", headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def list_image_models(self) -> List[Dict[str, Any]]:
        """GET /image/models — Image & video models with details."""
        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with session.get(
            f"{self.base_url}/image/models", headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def list_video_models(self) -> List[Dict[str, Any]]:
        """GET /video/models — Video models with details."""
        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with session.get(
            f"{self.base_url}/video/models", headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def list_audio_models(self) -> List[Dict[str, Any]]:
        """GET /audio/models — Audio models with details."""
        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with session.get(
            f"{self.base_url}/audio/models", headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def list_3d_models(self) -> List[Dict[str, Any]]:
        """GET /3d/models — 3D models with details."""
        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with session.get(
            f"{self.base_url}/3d/models", headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def list_embedding_models(self) -> List[Dict[str, Any]]:
        """GET /embeddings/models — Embedding models with details."""
        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with session.get(
            f"{self.base_url}/embeddings/models", headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def model_health_status(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """GET /v1/models/status — Model health monitoring."""
        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with session.get(
            f"{self.base_url}/v1/models/status",
            headers=headers,
            params={"minutes": minutes},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ══════════════════════════════════════════════════════
    #  MEDIA STORAGE
    # ══════════════════════════════════════════════════════

    async def media_upload(
        self,
        file_bytes: bytes,
        *,
        filename: str = "upload.png",
        content_type: str = "image/png",
        tags: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """POST /upload — Upload media to Pollinations storage."""
        session = await self._get_session()
        data = aiohttp.FormData()
        data.add_field("file", file_bytes, filename=filename, content_type=content_type)
        if tags:
            tag_str = ",".join(tags) if isinstance(tags, list) else tags
            data.add_field("tags", tag_str)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with session.post(
            f"{self.media_url}/upload",
            headers=headers,
            data=data,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def media_list(
        self,
        tag: str,
        *,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """GET /media — List public gallery for a tag."""
        session = await self._get_session()
        params: Dict[str, Any] = {"tag": tag, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        async with session.get(
            f"{self.media_url}/media", params=params
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def media_delete(self, media_id: str) -> Dict[str, Any]:
        """DELETE /media/{id} — Delete a media item."""
        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with session.delete(
            f"{self.media_url}/media/{media_id}", headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def media_get(self, media_id: str) -> bytes:
        """GET /{id} — Retrieve media file by id."""
        session = await self._get_session()
        async with session.get(f"{self.media_url}/{media_id}") as resp:
            resp.raise_for_status()
            return await resp.read()

    async def media_metadata(self, media_id: str) -> Dict[str, Any]:
        """GET /{id}/metadata — Get file metadata."""
        session = await self._get_session()
        async with session.get(
            f"{self.media_url}/{media_id}/metadata"
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ══════════════════════════════════════════════════════
    #  ACCOUNT
    # ══════════════════════════════════════════════════════

    async def account_profile(self) -> Dict[str, Any]:
        """GET /account/profile — Get your account profile."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/account/profile", headers=self._headers()
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def account_balance(self) -> Dict[str, Any]:
        """GET /account/balance — Get pollen balance."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/account/balance", headers=self._headers()
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def account_usage(
        self,
        *,
        format: str = "json",
        limit: int = 100,
        days: int = 30,
    ) -> Dict[str, Any]:
        """GET /account/usage — Get usage history."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/account/usage",
            headers=self._headers(),
            params={"format": format, "limit": limit, "days": days},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def account_key_info(self) -> Dict[str, Any]:
        """GET /account/key — Get API key info."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/account/key", headers=self._headers()
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def account_quests(self) -> Dict[str, Any]:
        """GET /account/quests — Get quest status."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/account/quests", headers=self._headers()
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ══════════════════════════════════════════════════════
    #  QUESTS
    # ══════════════════════════════════════════════════════

    async def quest_catalog(self) -> Dict[str, Any]:
        """GET /quests/catalog — Get quest catalog."""
        session = await self._get_session()
        async with session.get(f"{self.base_url}/quests/catalog") as resp:
            resp.raise_for_status()
            return await resp.json()

    # ══════════════════════════════════════════════════════
    #  VISION — Image analysis via chat
    # ══════════════════════════════════════════════════════

    async def analyze_image(
        self,
        image_url: str,
        question: str = "What is in this image?",
        model: str = "openai",
    ) -> str:
        """Analyze an image using a vision-capable model."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        return await self.chat_completions_simple(messages, model=model)
