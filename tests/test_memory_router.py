"""阶段十四 14.3/14.4 Python memory router 单元测试"""

import sys
import os
from unittest.mock import MagicMock

mock_trace = MagicMock()
mock_trace.get_current_trace_id.return_value = "test-trace-id"
sys.modules["app.tracing"] = mock_trace
sys.modules["app"] = MagicMock()
sys.modules["app.routers"] = MagicMock()


def _reload_mem_module():
    """强制重新加载 memory 模块"""
    import importlib
    # 先从 sys.modules 清除，强制重新加载
    if "app.routers.memory" in sys.modules:
        del sys.modules["app.routers.memory"]
    spec = importlib.util.spec_from_file_location(
        "app.routers.memory",
        os.path.join(os.path.dirname(__file__), "..", "app", "routers", "memory.py"),
    )
    mem_module = importlib.util.module_from_spec(spec)
    sys.modules["app.routers.memory"] = mem_module
    spec.loader.exec_module(mem_module)
    return mem_module


mem_module = _reload_mem_module()
router = mem_module.router


def make_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _inject_llm(llm):
    mem_module._llm_provider = llm


def _inject_rag(rag):
    mem_module._rag_processor = rag


class DummyLLMProvider:
    async def chat(self, request, **kwargs):
        return {"choices": [{"message": {"content": "这是摘要内容"}}]}


class DummyRAGProcessor:
    def __init__(self, results=None):
        self.results = results or []

    def search(self, query, collection_name, top_k, min_score):
        return self.results[:top_k]


class _SearchResult:
    def __init__(self, chunk_id, text, score, metadata):
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.metadata = metadata

    def get(self, key, default=None):
        return getattr(self, key, default)


# ==================== 14.3 摘要 ====================

def test_summarize_empty_when_no_llm():
    """无 LLM provider 返回 503"""
    _inject_llm(None)
    client = make_client()
    resp = client.post(
        "/api/memory/summarize",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503


def test_summarize_empty_messages():
    """空消息列表直接返回空摘要"""
    _inject_llm(DummyLLMProvider())
    client = make_client()
    resp = client.post("/api/memory/summarize", json={"messages": []})
    assert resp.status_code == 200
    assert resp.json()["summary"] == ""


def test_summarize_text_content():
    """消息列表中的文本内容被合并发给 LLM"""
    class CapturingLLM:
        def __init__(self):
            self.received_messages = None

        async def chat(self, request, **kwargs):
            self.received_messages = request
            return {"choices": [{"message": {"content": "摘要"}}]}

    captor = CapturingLLM()
    _inject_llm(captor)
    client = make_client()
    resp = client.post(
        "/api/memory/summarize",
        json={
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！"},
                {"role": "user", "content": "帮我写代码"},
            ],
            "existing_summary": "旧摘要",
        },
    )
    assert resp.status_code == 200
    # 验证 LLM 收到了合并后的文本
    assert captor.received_messages is not None
    prompt = captor.received_messages["messages"][-1]["content"]
    assert "你好" in prompt
    assert "帮我写代码" in prompt
    assert "旧摘要" in prompt


def test_summarize_truncates_long_response():
    """超过 600 字时截断，防止过长注入"""
    class LongLLM:
        async def chat(self, request, **kwargs):
            return {"choices": [{"message": {"content": "x" * 1000}}]}

    _inject_llm(LongLLM())
    client = make_client()
    resp = client.post(
        "/api/memory/summarize",
        json={"messages": [{"role": "user", "content": "test"}]},
    )
    assert resp.status_code == 200
    assert len(resp.json()["summary"]) <= 600


# ==================== 14.4 长期记忆搜索 ====================

def test_search_returns_empty_when_no_rag():
    """无 RAG processor 返回空（降级保护）"""
    _inject_rag(None)
    client = make_client()
    resp = client.post(
        "/api/memory/search",
        json={"user_id": "user1", "query": "名字"},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_search_scope_filtering():
    """scope=user 全局记忆永远返回；scope=agent 精准匹配 agent_id"""
    _inject_rag(DummyRAGProcessor(results=[
        _SearchResult("id1", "用户叫小明", 0.85, {"scope": "user", "kind": "profile"}),
        _SearchResult("id2", "客服偏好顺丰", 0.90, {"scope": "agent", "agent_id": "agent-X", "kind": "preference"}),
    ]))
    client = make_client()

    # agent-X 查：全局 + agent-X 专属
    resp = client.post(
        "/api/memory/search",
        json={"user_id": "user1", "query": "快递", "agent_id": "agent-X"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2

    # agent-Y 查：只有全局（agent-X 专属不应出现）
    resp2 = client.post(
        "/api/memory/search",
        json={"user_id": "user1", "query": "快递", "agent_id": "agent-Y"},
    )
    ids = [r["id"] for r in resp2.json()["results"]]
    assert "id2" not in ids


def test_search_degrades_on_exception():
    """RAG 抛异常返回空（降级保护，不阻塞对话）"""
    class BrokenRAG:
        def search(self, **kwargs):
            raise RuntimeError("pgvector down")

    _inject_rag(BrokenRAG())
    client = make_client()
    resp = client.post(
        "/api/memory/search",
        json={"user_id": "user1", "query": "x"},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_search_respects_top_k():
    """top_k 参数透传给 RAG processor"""
    results = [_SearchResult(f"id{i}", f"text{i}", 0.9 - i * 0.1, {}) for i in range(10)]
    _inject_rag(DummyRAGProcessor(results=results))
    client = make_client()
    resp = client.post(
        "/api/memory/search",
        json={"user_id": "u1", "query": "x", "top_k": 3},
    )
    assert len(resp.json()["results"]) == 3


# ==================== 健康检查 ====================

def test_health_returns_status():
    """health 返回各组件就绪状态"""
    _inject_llm(DummyLLMProvider())
    _inject_rag(DummyRAGProcessor())
    client = make_client()
    resp = client.get("/api/memory/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm_ready"] is True
    assert "trace_id" in data
