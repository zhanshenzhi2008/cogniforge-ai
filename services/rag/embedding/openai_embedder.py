"""
OpenAI embedding client.
"""
import logging
import os
from typing import List, Optional

from .base import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding model client."""

    DEFAULT_MODEL = "text-embedding-3-small"
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or self.DEFAULT_MODEL

        if not self.api_key:
            logger.warning("OpenAI API key not provided. Embedding will not work.")
        else:
            key_preview = self.api_key[:7] + "..." if len(self.api_key) > 7 else "***"
            logger.info(f"OpenAI embedder initialized with model={self.model}, key={key_preview}")

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        results = self.embed_batch([text])
        return results[0] if results else []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required for embedding")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        logger.info(f"Calling OpenAI API for {len(texts)} embeddings with model {self.model}")

        client = OpenAI(api_key=self.api_key, timeout=30.0)

        response = client.embeddings.create(
            model=self.model,
            input=texts
        )

        logger.info(f"Received {len(response.data)} embeddings from OpenAI")
        return [item.embedding for item in response.data]

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.DIMENSIONS.get(self.model, 1536)
