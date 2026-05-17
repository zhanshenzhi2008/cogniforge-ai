"""
Recursive character text splitter.
"""
import logging
import re
from typing import List

from .base import BaseSplitter, TextChunk

logger = logging.getLogger(__name__)


class RecursiveCharacterSplitter(BaseSplitter):
    """
    Splits text recursively by different characters until chunks are small enough.
    """

    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        ". ",
        "。",
        "！",
        "？",
        "; ",
        "，",
        ", ",
        " ",
        "",
    ]

    def __init__(self, chunk_size: int = 512, overlap: int = 50, separators: List[str] = None):
        super().__init__(chunk_size, overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """Split text into chunks using recursive character splitting."""
        if not text or len(text) <= self.chunk_size:
            if text:
                return [self._create_chunk(text, 0, 0, len(text), metadata)]
            return []

        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                end = self._find_best_split_point(text, start, end)

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(self._create_chunk(chunk_text, index, start, end, metadata))
                index += 1

            start = end - self.overlap
            if start < 0:
                start = 0

        return chunks

    def _find_best_split_point(self, text: str, start: int, end: int) -> int:
        """Find the best point to split text within a range."""
        for separator in self.separators:
            search_start = max(start, end - 100)
            pos = text.rfind(separator, search_start, end)

            if pos != -1 and pos > start:
                return pos + len(separator)

        return end


class SentenceSplitter(BaseSplitter):
    """Splits text by sentences, trying to keep sentences together."""

    SENTENCE_PATTERNS = [
        r'[.!?。！？]\s+',
        r'([.!?。！？])+\s+',
        r'\n\n+',
        r'\n',
    ]

    def __init__(self, chunk_size: int = 512, overlap: int = 50, max_sentences: int = 10):
        super().__init__(chunk_size, overlap)
        self.max_sentences = max_sentences

    def split(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """Split text into chunks while keeping sentences together."""
        if not text:
            return []

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_size = 0
        index = 0
        chunk_start = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_size + sentence_len > self.chunk_size and current_chunk:
                chunk_text = "".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, index, chunk_start,
                                                chunk_start + len(chunk_text), metadata))
                index += 1

                overlap_text = "".join(current_chunk[-2:]) if len(current_chunk) >= 2 else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_size = len(overlap_text)
                chunk_start = chunk_start + len(chunk_text) - self.overlap

            current_chunk.append(sentence)
            current_size += sentence_len

        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, index, chunk_start,
                                            chunk_start + len(chunk_text), metadata))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        pattern = '|'.join(self.SENTENCE_PATTERNS)
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]
