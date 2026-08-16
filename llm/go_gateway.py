"""
LLM calls go through the Go backend.

Go owns ai_providers (URL / Key / default model). Python must not decrypt keys
or talk to DeepSeek/OpenAI itself.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Optional

import httpx

from .base import LLMProvider
from .model_config import get_snapshot

logger = logging.getLogger(__name__)

SUCCESS_CODE = 2000


class GoGateway(LLMProvider):
    """LLMProvider that forwards chat to cogniforge (Go) HTTP APIs."""

    def __init__(self, api_url: Optional[str] = None, timeout: float = 120.0):
        self.api_url = (api_url or os.getenv("COGNIFORGE_API_URL", "http://localhost:8080")).rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)
        logger.info("Go LLM gateway initialized api_url=%s", self.api_url)

    @property
    def name(self) -> str:
        return "go"

    def supports_functions(self) -> bool:
        # Go chat payload currently forwards messages only (no tools).
        return False

    async def chat(self, request: dict) -> dict:
        payload = _chat_payload(request, stream=False)
        url = f"{self.api_url}/api/v1/chat/completions"
        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except Exception as e:
            logger.error("Go chat error: %s", e)
            raise

        data = _unwrap_go(body)
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected Go chat response: {body!r}")
        return data

    async def chat_stream(self, request: dict) -> AsyncIterator[dict]:
        payload = _chat_payload(request, stream=True)
        url = f"{self.api_url}/api/v1/chat/stream"
        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is None:
                        continue
                    if chunk is _DONE:
                        break
                    yield chunk
        except Exception as e:
            logger.error("Go chat stream error: %s", e)
            yield {"error": str(e)}

    async def close(self) -> None:
        await self._client.aclose()


_DONE = object()


def _chat_payload(request: dict, stream: bool) -> dict:
    payload: dict[str, Any] = {
        "messages": request.get("messages") or [],
        "stream": stream,
    }
    model = request.get("model")
    if not model:
        snap = get_snapshot()
        if snap and snap.default_model:
            model = snap.default_model
    if model:
        payload["model"] = model
    if request.get("temperature") is not None:
        payload["temperature"] = request["temperature"]
    if request.get("max_tokens") is not None:
        payload["max_tokens"] = request["max_tokens"]
    if request.get("top_p") is not None:
        payload["top_p"] = request["top_p"]
    return payload


def _unwrap_go(body: Any) -> Any:
    if not isinstance(body, dict):
        return body
    if "code" not in body:
        return body
    code = body.get("code")
    if code != SUCCESS_CODE:
        err = body.get("error") or body.get("message") or json.dumps(body, ensure_ascii=False)
        raise RuntimeError(f"Go API error code={code}: {err}")
    return body.get("data")


def _parse_sse_line(line: str):
    line = (line or "").strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return _DONE
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None
