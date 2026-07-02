"""
OpenRouter LLM Provider
"""
import logging
import os
from typing import Any, AsyncIterator, Optional

from openrouter import OpenRouter

logger = logging.getLogger(__name__)


class OpenRouterProvider:
    """OpenRouter LLM provider — routes requests through OpenRouter gateway."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        http_referer: Optional[str] = None,
        x_open_router_title: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

        self._http_referer = http_referer or os.getenv("OPENROUTER_HTTP_REFERER")
        self._title = x_open_router_title or os.getenv("OPENROUTER_TITLE")

        self.client = OpenRouter(api_key=self.api_key)
        logger.info("OpenRouter provider initialized")

    @property
    def name(self) -> str:
        return "openrouter"

    def supports_functions(self) -> bool:
        return True

    async def chat(self, request: dict) -> dict:
        """Send a non-streaming chat completion request."""
        messages = request.get("messages", [])
        model = request.get("model", os.getenv("DEFAULT_MODEL", "openai/gpt-4o"))
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 4096)
        functions = request.get("functions")

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "http_referer": self._http_referer,
            "x_open_router_title": self._title,
        }
        if functions:
            kwargs["tools"] = functions
            if request.get("tool_choice"):
                kwargs["tool_choice"] = request["tool_choice"]

        try:
            result = self.client.chat.send_async(**kwargs)
            resp = await result
            return self._normalize_response(resp)
        except Exception as e:
            logger.error(f"OpenRouter chat error: {e}")
            raise

    async def chat_stream(self, request: dict) -> AsyncIterator[dict]:
        """Send a streaming chat completion request."""
        messages = request.get("messages", [])
        model = request.get("model", os.getenv("DEFAULT_MODEL", "openai/gpt-4o"))
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 4096)
        functions = request.get("functions")

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "http_referer": self._http_referer,
            "x_open_router_title": self._title,
        }
        if functions:
            kwargs["tools"] = functions

        try:
            stream_result = self.client.chat.send_async(**kwargs)
            async for chunk in stream_result:
                yield self._normalize_chunk(chunk)
        except Exception as e:
            logger.error(f"OpenRouter stream error: {e}")
            yield {"error": str(e)}

    # ------------------------------------------------------------------
    # Internal helpers — normalize OpenRouter types to the dict shape
    # expected by the rest of the codebase.
    # ------------------------------------------------------------------

    def _normalize_response(self, resp) -> dict:
        """Convert ChatResult to the standard dict response."""
        msg = resp.choices[0].message
        return {
            "id": resp.id,
            "model": resp.model,
            "choices": [{
                "message": {
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in (msg.tool_calls or [])
                    ],
                },
                "finish_reason": resp.choices[0].finish_reason,
            }],
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            } if resp.usage else None,
        }

    def _normalize_chunk(self, chunk) -> dict:
        """Convert ChatStreamChunk to the standard dict chunk."""
        delta = chunk.choices[0].delta
        return {
            "id": chunk.id,
            "model": chunk.model,
            "choices": [{
                "delta": {
                    "role": delta.role,
                    "content": delta.content,
                    "tool_calls": [
                        {
                            "index": tc.index,
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in (delta.tool_calls or [])
                    ],
                },
                "finish_reason": chunk.choices[0].finish_reason,
            }],
        }
