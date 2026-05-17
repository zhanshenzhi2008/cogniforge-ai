"""
Local sentence transformer embedding.
"""
import logging
import os
from typing import List, Optional

from .base import BaseEmbedder

logger = logging.getLogger(__name__)


class LocalEmbedder(BaseEmbedder):
    """Local sentence transformer embedding model."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._dimension = None

        if device:
            self.device = device
        else:
            self.device = "cpu"

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        results = self.embed_batch([text])
        return results[0] if results else []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._load_model()

        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        self._load_model()
        return self._dimension
