"""
Memory Service Router
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from app.tracing import get_current_trace_id

if TYPE_CHECKING:
    from llm.base import LLMProvider
    from services.rag import DocumentProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["Memory"])

# Global dependencies (set by main.py)
_memory: Optional["MemoryManager"] = None
_llm_provider: Optional["LLMProvider"] = None
_rag_processor: Optional["DocumentProcessor"] = None

_SUMMARIZE_PROMPT = """你是一个对话摘要助手。请把以下对话片段压缩成一段简短的摘要（不超过500字），只保留关键信息和结论，不要复述对话过程。

对话片段：
{text}

摘要（不超过500字）："""


def set_memory(memory):
    """Set the global memory manager."""
    global _memory
    _memory = memory


def set_llm_provider(provider: "LLMProvider"):
    """Set the global LLM provider (for summarization)."""
    global _llm_provider
    _llm_provider = provider


def set_rag_processor(processor: "DocumentProcessor"):
    """Set the global RAG processor (for vector search in long-term memory)."""
    global _rag_processor
    _rag_processor = processor


class SaveRequest(BaseModel):
    session_id: str = "default"
    messages: List[Dict[str, Any]]


class SummarizeRequest(BaseModel):
    messages: List[Dict[str, Any]]
    existing_summary: Optional[str] = None


class SearchRequest(BaseModel):
    user_id: str
    query: str
    agent_id: Optional[str] = None
    top_k: int = 5
    min_score: float = 0.4


@router.get("/health")
async def memory_health():
    """Health check for Memory service."""
    return {
        "status": "ok",
        "memory_ready": _memory is not None,
        "llm_ready": _llm_provider is not None,
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


@router.post("/summarize")
async def summarize_memory(request: SummarizeRequest):
    """
    对被裁剪掉的旧消息生成摘要（阶段十四 14.3 滚动摘要）。

    策略：调用小模型（Go 默认模型），把旧消息合并为一段文本，
    返回 ≤500 字的摘要。摘要失败时降级为返回空字符串。
    """
    if not _llm_provider:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    if not request.messages:
        return {"summary": "", "trace_id": get_current_trace_id()}

    # 把消息合并成文本
    parts = []
    for msg in request.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(f"{role}: {content.strip()}")
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "").strip()
                    if text:
                        parts.append(f"{role}: {text}")

    raw_text = "\n".join(parts)
    if not raw_text.strip():
        return {"summary": "", "trace_id": get_current_trace_id()}

    # 拼接已有摘要（追加模式）
    prompt_text = _SUMMARIZE_PROMPT.format(text=raw_text)
    if request.existing_summary:
        prompt_text = (
            f"以下是对之前对话的摘要：\n{request.existing_summary}\n\n"
            f"新增对话片段：\n{raw_text}\n\n"
            f"请把新增片段的关键信息合并到已有摘要中，更新摘要（不超过500字）："
        )

    try:
        resp = await _llm_provider.chat({
            "messages": [
                {"role": "system", "content": "你是一个简洁的摘要助手。回复只能是摘要文本，不要有任何前缀、解释或格式。"},
                {"role": "user", "content": prompt_text},
            ],
            "max_tokens": 300,
            "temperature": 0.3,
        })
        # 非流式响应取 content
        choices = resp.get("choices", [])
        if choices:
            summary = choices[0].get("message", {}).get("content", "")
            # 截断到合理长度（最多 600 字保守上限）
            if len(summary) > 600:
                summary = summary[:600]
            return {"summary": summary.strip(), "trace_id": get_current_trace_id()}
        return {"summary": "", "trace_id": get_current_trace_id()}
    except Exception as e:
        logger.warning("Summarize failed, degrading to empty: %s", e)
        # 摘要失败降级：返回空字符串，不阻塞对话
        return {"summary": "", "trace_id": get_current_trace_id()}


@router.post("/search")
async def search_memory(request: SearchRequest):
    """
    长期记忆向量检索（阶段十四 14.4）。

    Go 调此端点，根据当前用户消息检索 chat_memories 表，
    返回 Top-K 条记忆片段，供 Go 注入 messages。
    """
    if not _rag_processor:
        # RAG processor 未初始化时降级返回空
        return {"results": [], "trace_id": get_current_trace_id()}

    try:
        results = _rag_processor.search(
            query=request.query,
            collection_name="chat_memories",
            top_k=request.top_k,
            min_score=request.min_score,
        )
        # 过滤 scope：只保留 user 全局 或 agent_id 匹配的
        filtered = []
        for r in results:
            meta = r.get("metadata", {})
            scope = meta.get("scope", "user")
            r_agent = meta.get("agent_id", "")
            if scope == "user" or (scope == "agent" and r_agent == request.agent_id):
                filtered.append({
                    "id": r.get("chunk_id", ""),
                    "content": r.get("text", ""),
                    "score": r.get("score", 0),
                    "kind": meta.get("kind", ""),
                })
        return {"results": filtered, "trace_id": get_current_trace_id()}
    except Exception as e:
        logger.warning("Memory search failed, degrading to empty: %s", e)
        return {"results": [], "trace_id": get_current_trace_id()}


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
