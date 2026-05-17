"""
Anthropic LLM Provider
"""
import os
import logging
from typing import Any, AsyncIterator, Optional

from anthropic import AsyncAnthropic

from .base import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider using the official SDK"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        
        self.client = AsyncAnthropic(api_key=self.api_key)
        logger.info("Anthropic provider initialized")
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    def supports_functions(self) -> bool:
        return True
    
    async def chat(self, request: dict) -> dict:
        """Send a non-streaming message request"""
        messages = request.get("messages", [])
        model = request.get("model", "claude-sonnet-4-20250514")
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 4096)
        tools = request.get("tools")
        system = request.get("system")
        
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if system:
                kwargs["system"] = system
            
            if tools:
                kwargs["tools"] = tools
            
            response = await self.client.messages.create(**kwargs)
            
            return {
                "id": response.id,
                "model": response.model,
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0]),
                    },
                    "finish_reason": response.stop_reason,
                }],
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
            }
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            raise
    
    async def chat_stream(self, request: dict) -> AsyncIterator[dict]:
        """Send a streaming message request"""
        messages = request.get("messages", [])
        model = request.get("model", "claude-sonnet-4-20250514")
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 4096)
        tools = request.get("tools")
        system = request.get("system")
        
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            
            if system:
                kwargs["system"] = system
            
            if tools:
                kwargs["tools"] = tools
            
            stream = await self.client.messages.stream(**kwargs)
            
            async for chunk in stream:
                if chunk.type == "content_block_delta":
                    yield {
                        "choices": [{
                            "delta": {
                                "content": chunk.delta.text,
                            },
                            "finish_reason": None,
                        }],
                    }
                elif chunk.type == "message_delta":
                    yield {
                        "choices": [{
                            "delta": {},
                            "finish_reason": chunk.delta.stop_reason,
                        }],
                    }
        except Exception as e:
            logger.error(f"Anthropic stream error: {e}")
            yield {"error": str(e)}
    
    async def close(self):
        """Close the client connection"""
        await self.client.close()
