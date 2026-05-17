"""
LLM Provider Base Interface
"""
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')"""
        pass
    
    @abstractmethod
    async def chat(self, request: dict) -> dict:
        """Send a chat completion request"""
        pass
    
    @abstractmethod
    async def chat_stream(self, request: dict) -> AsyncIterator[dict]:
        """Send a streaming chat completion request"""
        pass
    
    @abstractmethod
    def supports_functions(self) -> bool:
        """Whether this provider supports function calling"""
        pass
