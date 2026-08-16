"""
Embeddings via Go POST /api/v1/embeddings (same ai_providers as chat).
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import httpx

from .base import BaseEmbedder

logger = logging.getLogger(__name__)

SUCCESS_CODE = 2000


class GoEmbedder(BaseEmbedder):
    """Sync embedder that calls the Go hub instead of holding API keys."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.api_url = (api_url or os.getenv("COGNIFORGE_API_URL", "http://localhost:8080")).rstrip("/")
        self.model = model or os.getenv("EMBEDDING_MODEL") or ""
        self._dimension: Optional[int] = None
        self._client = httpx.Client(timeout=timeout)
        logger.info("Go embedder initialized api_url=%s model=%s", self.api_url, self.model or "(provider default)")

    def embed(self, text: str) -> List[float]:
        results = self.embed_batch([text])
        return results[0] if results else []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        payload: dict = {"input": texts}
        if not self.model:
            try:
                from llm.model_config import get_snapshot
                snap = get_snapshot()
                if snap and snap.default_model:
                    self.model = snap.default_model
            except Exception:
                pass
        if self.model:
            payload["model"] = self.model

        url = f"{self.api_url}/api/v1/embeddings"
        logger.info("Calling Go embeddings for %s texts", len(texts))
        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json()

        data = body
        if isinstance(body, dict) and "code" in body:
            if body.get("code") != SUCCESS_CODE:
                err = body.get("error") or body.get("message") or body
                raise RuntimeError(f"Go embeddings error: {err}")
            data = body.get("data") or {}

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise RuntimeError(f"unexpected Go embeddings payload: {body!r}")

        ordered = sorted(items, key=lambda i: i.get("index", 0))
        vectors = [item["embedding"] for item in ordered]
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors

    def get_dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed("dimension-probe")
            self._dimension = len(probe) if probe else 1536
        return self._dimension
