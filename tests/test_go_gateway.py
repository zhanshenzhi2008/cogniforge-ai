import httpx
import pytest

from llm.go_gateway import GoGateway, _chat_payload, _unwrap_go


def test_unwrap_go_success():
    inner = {"id": "chatcmpl-1", "choices": [{"message": {"content": "hi"}}]}
    assert _unwrap_go({"code": 2000, "data": inner}) == inner


def test_unwrap_go_error():
    with pytest.raises(RuntimeError, match="4006"):
        _unwrap_go({"code": 4006, "error": "no provider"})


def test_chat_payload_omits_empty_model():
    payload = _chat_payload({"messages": [{"role": "user", "content": "hi"}], "temperature": 0.2}, False)
    assert payload["stream"] is False
    assert "model" not in payload
    assert payload["temperature"] == 0.2


@pytest.mark.asyncio
async def test_chat_unwraps_go_envelope(monkeypatch):
    gateway = GoGateway(api_url="http://cogniforge:8080")

    async def fake_post(url, json):
        assert url.endswith("/api/v1/chat/completions")
        assert json["messages"][0]["content"] == "hello"
        assert json["stream"] is False
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "code": 2000,
                "data": {
                    "id": "chatcmpl-x",
                    "model": "deepseek-chat",
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                },
            },
        )

    monkeypatch.setattr(gateway._client, "post", fake_post)
    result = await gateway.chat({"messages": [{"role": "user", "content": "hello"}]})
    assert result["choices"][0]["message"]["content"] == "hi"
    await gateway.close()


def test_sse_line_done():
    from llm.go_gateway import _DONE, _parse_sse_line

    assert _parse_sse_line("data: [DONE]") is _DONE
    chunk = _parse_sse_line('data: {"choices":[{"delta":{"content":"a"}}]}')
    assert chunk["choices"][0]["delta"]["content"] == "a"
