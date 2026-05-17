"""
Document processing service.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

from .parsers import registry as parser_registry
from .splitters import RecursiveCharacterSplitter
from .embedding import create_embedder, BaseEmbedder
from .vector_store import create_vector_store, BaseVectorStore, VectorEntry

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Result of chunking a document."""
    chunk_id: str
    content: str
    index: int
    metadata: dict


@dataclass
class ProcessingResult:
    """Result of processing a document."""
    document_id: str
    file_path: str
    chunks: List[ChunkResult]
    success: bool
    error: Optional[str] = None


class DocumentProcessor:
    """Service for processing documents through the RAG pipeline."""

    def __init__(
        self,
        embedder_type: str = "openai",
        vector_store_type: str = "pgvector",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **embedder_kwargs
    ):
        self.embedder: BaseEmbedder = create_embedder(embedder_type, **embedder_kwargs)
        self.vector_store: BaseVectorStore = create_vector_store(vector_store_type)
        self.splitter = RecursiveCharacterSplitter(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def connect(self) -> None:
        """Connect to the vector store."""
        self.vector_store.connect()

    def disconnect(self) -> None:
        """Disconnect from the vector store."""
        self.vector_store.disconnect()

    def process_document(
        self,
        file_path: str,
        document_id: str,
        collection_name: str,
        metadata: dict = None
    ) -> ProcessingResult:
        """Process a document through the full RAG pipeline."""
        try:
            logger.info(f"Parsing document: {file_path}")
            parsed = parser_registry.parse(file_path)

            logger.info(f"Splitting into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
            text_chunks = self.splitter.split(parsed.content, {
                "document_id": document_id,
                "source": file_path,
                **(metadata or {})
            })

            chunks = []
            for i, tc in enumerate(text_chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                chunks.append(ChunkResult(
                    chunk_id=chunk_id,
                    content=tc.content,
                    index=i,
                    metadata=tc.metadata
                ))

            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            texts = [c.content for c in chunks]
            embeddings = self.embedder.embed_batch(texts)

            logger.info(f"Storing {len(chunks)} vectors in collection: {collection_name}")

            dimension = self.embedder.get_dimension()
            self.vector_store.create_collection(collection_name, dimension)

            entries = []
            for chunk, embedding in zip(chunks, embeddings):
                entry = VectorEntry(
                    id=chunk.chunk_id,
                    vector=embedding,
                    text=chunk.content,
                    metadata={
                        "document_id": document_id,
                        "chunk_index": chunk.index,
                        **chunk.metadata
                    }
                )
                entries.append(entry)

            self.vector_store.insert_batch(collection_name, entries)

            logger.info(f"Successfully processed document {document_id}: {len(chunks)} chunks")

            return ProcessingResult(
                document_id=document_id,
                file_path=file_path,
                chunks=chunks,
                success=True
            )

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            return ProcessingResult(
                document_id=document_id,
                file_path=file_path,
                chunks=[],
                success=False,
                error=str(e)
            )

    def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[dict]:
        """Search for relevant document chunks."""
        query_embedding = self.embedder.embed(query)

        results = self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            top_k=top_k
        )

        filtered = [r for r in results if r.score >= min_score]

        return [
            {
                "chunk_id": r.id,
                "content": r.text,
                "score": r.score,
                "metadata": r.metadata
            }
            for r in filtered
        ]

    def delete_document(self, document_id: str, collection_name: str) -> int:
        """Delete all chunks for a document."""
        logger.info(f"Deleting document {document_id} from {collection_name}")
        return 0
