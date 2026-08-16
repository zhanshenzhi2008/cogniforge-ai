import httpx

from services.rag.embedding.go_embedder import GoEmbedder


def test_embed_batch_unwraps_go_envelope(monkeypatch):
    embedder = GoEmbedder(api_url="http://cogniforge:8080")

    def fake_post(url, json):
        assert url.endswith("/api/v1/embeddings")
        assert json["input"] == ["hello"]
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "code": 2000,
                "data": {
                    "object": "list",
                    "model": "deepseek-chat",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                    ],
                },
            },
        )

    monkeypatch.setattr(embedder._client, "post", fake_post)
    vectors = embedder.embed_batch(["hello"])
    assert vectors == [[0.1, 0.2]]
    assert embedder.get_dimension() == 2
