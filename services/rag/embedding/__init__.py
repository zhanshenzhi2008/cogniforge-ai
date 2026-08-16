"""
Embedding models for text vectorization.
"""
from .base import BaseEmbedder
from .go_embedder import GoEmbedder

__all__ = ['BaseEmbedder', 'GoEmbedder', 'LocalEmbedder']


def create_embedder(embedder_type: str = "go", **kwargs) -> BaseEmbedder:
    """Factory function to create an embedder."""
    if embedder_type in ("go", "openai"):
        allowed = {k: kwargs[k] for k in ("api_url", "model", "timeout") if k in kwargs}
        return GoEmbedder(**allowed)
    if embedder_type == "local":
        from .local_embedder import LocalEmbedder
        return LocalEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder type: {embedder_type}")
