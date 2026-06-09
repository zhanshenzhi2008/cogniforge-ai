"""
Memory Service Router
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.tracing import get_current_trace_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["Memory"])

# Global memory manager (set by main.py)
_memory = None


def set_memory(memory):
    """Set the global memory manager."""
    global _memory
    _memory = memory


class SaveRequest(BaseModel):
    session_id: str = "default"
    messages: List[Dict[str, Any]]


@router.get("/health")
async def memory_health():
    """Health check for Memory service."""
    return {
        "status": "ok",
        "memory_ready": _memory is not None,
        "trace_id": get_current_trace_id(),
    }


@router.post("/save")
async def save_memory(request: SaveRequest):
    """Save conversation to memory."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    try:
        result = await _memory.save(request.model_dump())
        result["trace_id"] = get_current_trace_id()
        return result
    except Exception as e:
        logger.error(f"Memory save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}")
async def get_memory(session_id: str, limit: int = 10):
    """Get conversation history from memory."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    try:
        result = await _memory.get(session_id, limit)
        result["trace_id"] = get_current_trace_id()
        return result
    except Exception as e:
        logger.error(f"Memory get error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def clear_memory(session_id: str):
    """Clear conversation history."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    try:
        await _memory.clear(session_id)
        return {"message": "Memory cleared", "trace_id": get_current_trace_id()}
    except Exception as e:
        logger.error(f"Memory clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
