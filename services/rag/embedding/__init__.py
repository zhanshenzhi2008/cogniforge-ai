"""
Embedding models for text vectorization.
"""
import logging

from .base import BaseEmbedder
from .go_embedder import GoEmbedder

logger = logging.getLogger(__name__)

__all__ = ['BaseEmbedder', 'GoEmbedder', 'LocalEmbedder']


def create_embedder(embedder_type: str = "go", **kwargs) -> BaseEmbedder:
    """Factory function to create an embedder.

    ``local`` 在未安装 sentence-transformers 时回退到 Go，避免生产解析/检索整条挂掉。
    """
    if embedder_type in ("go", "openai"):
        allowed = {k: kwargs[k] for k in ("api_url", "model", "timeout") if k in kwargs}
        return GoEmbedder(**allowed)
    if embedder_type == "local":
        try:
            import sentence_transformers  # noqa: F401
            from .local_embedder import LocalEmbedder
            allowed = {k: kwargs[k] for k in ("model_name", "device") if k in kwargs}
            return LocalEmbedder(**allowed)
        except Exception as e:
            logger.warning(
                "Local embedder unavailable (%s); falling back to Go embedder",
                e,
            )
            allowed = {k: kwargs[k] for k in ("api_url", "model", "timeout") if k in kwargs}
            return GoEmbedder(**allowed)
    raise ValueError(f"Unknown embedder type: {embedder_type}")
