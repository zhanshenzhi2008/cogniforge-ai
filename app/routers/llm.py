"""
LLM Service Router
"""
import logging
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from llm.model_config import get_snapshot
from app.tracing import get_current_trace_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["LLM"])

# Global providers (set by main.py)
_providers: Dict[str, Any] = {}


def set_providers(providers: Dict[str, Any]):
    """Set the global LLM providers."""
    global _providers
    _providers = providers


class ChatRequest(BaseModel):
    provider: str = "default"
    model: Optional[str] = None
    messages: List[Dict[str, Any]] = []
    temperature: float = 0.7
    max_tokens: int = 4096
    functions: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None


@router.get("/health")
async def llm_health():
    """Health check for LLM service."""
    snap = get_snapshot()
    return {
        "status": "ok",
        "providers": list(_providers.keys()),
        "default_model": snap.default_model if snap else "",
        "model_cache_rev": snap.rev if snap else 0,
        "trace_id": get_current_trace_id(),
    }


@router.post("/chat")
async def llm_chat(request: ChatRequest):
    """Generic LLM chat endpoint."""
    provider = _providers.get(request.provider) or _providers.get("default")

    if not provider:
        raise HTTPException(status_code=400, detail=f"Provider '{request.provider}' not available")

    try:
        result = await provider.chat(request.model_dump())
        result["trace_id"] = get_current_trace_id()
        return result
    except Exception as e:
        logger.error(f"LLM chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
