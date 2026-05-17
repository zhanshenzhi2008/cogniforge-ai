"""
Vector stores for embedding storage and retrieval.
"""
from .base import BaseVectorStore, VectorEntry, SearchResult
from .pgvector_store import PGVectorStore

__all__ = ['BaseVectorStore', 'VectorEntry', 'SearchResult', 'PGVectorStore']


def create_vector_store(store_type: str = "pgvector", **kwargs) -> BaseVectorStore:
    """Factory function to create a vector store."""
    if store_type == "pgvector":
        return PGVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
