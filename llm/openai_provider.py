"""
OpenAI LLM Provider
"""
import os
import logging
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using the official OpenAI SDK"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        logger.info(f"OpenAI provider initialized (base_url: {self.base_url})")
    
    @property
    def name(self) -> str:
        return "openai"
    
    def supports_functions(self) -> bool:
        return True
    
    async def chat(self, request: dict) -> dict:
        """Send a non-streaming chat completion request"""
        messages = request.get("messages", [])
        model = request.get("model", os.getenv("DEFAULT_MODEL", "gpt-4o"))
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 4096)
        functions = request.get("functions")
        
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if functions:
                kwargs["tools"] = functions
                if request.get("tool_choice"):
                    kwargs["tool_choice"] = request["tool_choice"]
            
            response = await self.client.chat.completions.create(**kwargs)
            
            return {
                "id": response.id,
                "model": response.model,
                "choices": [{
                    "message": {
                        "role": response.choices[0].message.role,
                        "content": response.choices[0].message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                }
                            }
                            for tc in (response.choices[0].message.tool_calls or [])
                        ],
                    },
                    "finish_reason": response.choices[0].finish_reason,
                }],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
            }
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise
    
    async def chat_stream(self, request: dict) -> AsyncIterator[dict]:
        """Send a streaming chat completion request"""
        messages = request.get("messages", [])
        model = request.get("model", os.getenv("DEFAULT_MODEL", "gpt-4o"))
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 4096)
        functions = request.get("functions")
        
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            
            if functions:
                kwargs["tools"] = functions
            
            stream = await self.client.chat.completions.create(**kwargs)
            
            async for chunk in stream:
                choice = chunk.choices[0]
                delta = choice.delta
                
                yield {
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
                        "finish_reason": choice.finish_reason,
                    }],
                }
        except Exception as e:
            logger.error(f"OpenAI stream error: {e}")
            yield {"error": str(e)}
    
    async def close(self):
        """Close the client connection"""
        await self.client.close()
