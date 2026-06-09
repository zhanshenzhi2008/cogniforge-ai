"""
Agent Service Router
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.tracing import get_current_trace_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["Agent"])

# Global executor (set by main.py)
_executor = None


def set_executor(executor):
    """Set the global agent executor."""
    global _executor
    _executor = executor


class ChatRequest(BaseModel):
    provider: str = "openai"
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    messages: List[Dict[str, Any]] = []
    tools: Optional[List[Dict[str, Any]]] = None
    session_id: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False


@router.get("/health")
async def agent_health():
    """Health check for Agent service."""
    return {
        "status": "ok",
        "executor_ready": _executor is not None,
        "trace_id": get_current_trace_id(),
    }


@router.post("/chat")
async def agent_chat(request: ChatRequest):
    """Agent chat with tools and memory."""
    if not _executor:
        raise HTTPException(status_code=503, detail="Agent executor not initialized")

    try:
        result = await _executor.execute(request.model_dump())
        result["trace_id"] = get_current_trace_id()
        return result
    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        return {"error": str(e), "trace_id": get_current_trace_id()}
