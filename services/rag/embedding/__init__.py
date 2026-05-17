"""
Embedding models for text vectorization.
"""
from .base import BaseEmbedder
from .openai_embedder import OpenAIEmbedder
from .local_embedder import LocalEmbedder

__all__ = ['BaseEmbedder', 'OpenAIEmbedder', 'LocalEmbedder']


def create_embedder(embedder_type: str = "openai", **kwargs) -> BaseEmbedder:
    """Factory function to create an embedder."""
    if embedder_type == "openai":
        return OpenAIEmbedder(**kwargs)
    elif embedder_type == "local":
        return LocalEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")
