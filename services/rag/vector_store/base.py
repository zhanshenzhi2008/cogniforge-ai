"""
Base vector store interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class VectorEntry:
    """A vector entry with metadata."""
    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Search result with score."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the vector store."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the vector store."""
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int, **kwargs) -> None:
        """Create a collection (table/index) for vectors."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        pass

    @abstractmethod
    def insert(self, collection_name: str, entry: VectorEntry) -> str:
        """Insert a single vector entry."""
        pass

    @abstractmethod
    def insert_batch(self, collection_name: str, entries: List[VectorEntry]) -> List[str]:
        """Insert multiple vector entries."""
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, collection_name: str, entry_id: str) -> None:
        """Delete an entry by ID."""
        pass

    @abstractmethod
    def get_dimension(self, collection_name: str) -> Optional[int]:
        """Get the dimension of vectors in a collection."""
        pass
