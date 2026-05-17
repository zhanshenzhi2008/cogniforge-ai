"""
Base text splitter interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = None


class BaseSplitter(ABC):
    """Abstract base class for text splitters."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def split(self, text: str, metadata: dict = None) -> List[TextChunk]:
        pass

    def _create_chunk(self, text: str, index: int, start: int, end: int, metadata: dict = None) -> TextChunk:
        """Create a TextChunk with metadata."""
        return TextChunk(
            content=text,
            index=index,
            start_char=start,
            end_char=end,
            metadata=metadata or {}
        )
