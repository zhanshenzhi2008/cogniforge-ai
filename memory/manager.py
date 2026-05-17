"""
Memory Manager - Manages conversation history and memory
"""
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages conversation history and memory.
    
    Currently uses in-memory storage.
    Can be extended to use Redis for distributed memory.
    """
    
    def __init__(self):
        self._memory: Dict[str, List[dict]] = {}
        self._max_turns = 50
        logger.info("Memory manager initialized (in-memory storage)")
    
    async def save(self, request: dict) -> dict:
        """Save a message or messages to memory"""
        session_id = request.get("session_id", "default")
        messages = request.get("messages", [])
        
        if session_id not in self._memory:
            self._memory[session_id] = []
        
        if isinstance(messages, dict):
            self._memory[session_id].append(messages)
        elif isinstance(messages, list):
            self._memory[session_id].extend(messages)
        
        self._trim_memory(session_id)
        
        return {
            "saved": len(messages) if isinstance(messages, list) else 1,
            "total_turns": len(self._memory[session_id]),
        }
    
    async def get(self, session_id: str, limit: int = 10) -> List[dict]:
        """Get recent conversation history"""
        messages = self._memory.get(session_id, [])
        
        if limit > 0:
            return messages[-limit:]
        return messages
    
    async def append(self, session_id: str, messages: List[dict]) -> None:
        """Append messages to session memory"""
        if session_id not in self._memory:
            self._memory[session_id] = []
        
        if isinstance(messages, dict):
            self._memory[session_id].append(messages)
        else:
            self._memory[session_id].extend(messages)
        
        self._trim_memory(session_id)
    
    async def clear(self, session_id: str) -> None:
        """Clear session memory"""
        if session_id in self._memory:
            del self._memory[session_id]
            logger.info(f"Cleared memory for session: {session_id}")
    
    async def search(self, session_id: str, query: str) -> List[dict]:
        """Search in memory (simple keyword match)"""
        messages = self._memory.get(session_id, [])
        results = []
        
        query_lower = query.lower()
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and query_lower in content.lower():
                results.append(msg)
        
        return results
    
    def _trim_memory(self, session_id: str) -> None:
        """Trim memory to max turns"""
        if len(self._memory[session_id]) > self._max_turns:
            self._memory[session_id] = self._memory[session_id][-self._max_turns:]
    
    def list_sessions(self) -> List[str]:
        """List all active sessions"""
        return list(self._memory.keys())
    
    def set_max_turns(self, max_turns: int) -> None:
        """Set maximum turns to keep in memory"""
        self._max_turns = max_turns
