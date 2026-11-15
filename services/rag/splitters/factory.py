"""Build text splitters from parse options."""
from __future__ import annotations

from typing import List, Optional

from .base import BaseSplitter, TextChunk
from .recursive_splitter import RecursiveCharacterSplitter


class FixedSizeSplitter(BaseSplitter):
    """Split by fixed character window (no soft boundary search)."""

    def split(self, text: str, metadata: dict = None) -> List[TextChunk]:
        if not text:
            return []
        chunks: List[TextChunk] = []
        start = 0
        index = 0
        n = len(text)
        overlap = max(0, min(self.overlap, self.chunk_size - 1))
        while start < n:
            end = min(start + self.chunk_size, n)
            piece = text[start:end].strip()
            if piece:
                chunks.append(self._create_chunk(piece, index, start, end, metadata))
                index += 1
            if end >= n:
                break
            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks


class SeparatorSplitter(BaseSplitter):
    """Split by a primary separator, then pack pieces up to chunk_size."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50, separator: str = "\n\n"):
        super().__init__(chunk_size, overlap)
        self.separator = separator if separator is not None else "\n\n"

    def split(self, text: str, metadata: dict = None) -> List[TextChunk]:
        if not text:
            return []
        if not self.separator:
            return FixedSizeSplitter(self.chunk_size, self.overlap).split(text, metadata)

        parts = [p.strip() for p in text.split(self.separator) if p and p.strip()]
        if not parts:
            return []

        packed: List[str] = []
        buf = ""
        for part in parts:
            candidate = part if not buf else f"{buf}{self.separator}{part}"
            if len(candidate) <= self.chunk_size:
                buf = candidate
                continue
            if buf:
                packed.append(buf)
            if len(part) <= self.chunk_size:
                buf = part
            else:
                # oversized single part → fixed split
                sub = FixedSizeSplitter(self.chunk_size, self.overlap).split(part, metadata)
                packed.extend([c.content for c in sub[:-1]])
                buf = sub[-1].content if sub else ""
        if buf:
            packed.append(buf)

        chunks: List[TextChunk] = []
        cursor = 0
        for i, content in enumerate(packed):
            start = text.find(content, cursor)
            if start < 0:
                start = cursor
            end = start + len(content)
            chunks.append(self._create_chunk(content, i, start, end, metadata))
            cursor = end
        return chunks


def create_splitter(
    split_mode: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separator: Optional[str] = None,
) -> BaseSplitter:
    size = max(64, min(int(chunk_size or 512), 4000))
    overlap = max(0, min(int(chunk_overlap or 0), size - 1))
    mode = (split_mode or "recursive").strip().lower()
    if mode in ("fixed", "fixed_size"):
        return FixedSizeSplitter(size, overlap)
    if mode in ("separator", "by_separator"):
        return SeparatorSplitter(size, overlap, separator if separator is not None else "\n\n")
    return RecursiveCharacterSplitter(size, overlap)
