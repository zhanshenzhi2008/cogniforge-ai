"""
RAG Service Router
"""
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

from services.rag import DocumentProcessor
from app.tracing import get_current_trace_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["RAG"])

# Global processor instance (set by main.py)
_processor: Optional[DocumentProcessor] = None


def set_processor(processor: DocumentProcessor):
    """Set the global document processor."""
    global _processor
    _processor = processor


class ProcessRequest(BaseModel):
    file_path: str
    document_id: str
    collection_name: str
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    collection_name: str
    top_k: int = 5
    min_score: float = 0.0


class SearchResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: dict


@router.get("/health")
async def rag_health():
    """Health check for RAG service."""
    return {
        "status": "ok",
        "processor_ready": _processor is not None,
        "trace_id": get_current_trace_id(),
    }


@router.post("/process")
async def process_document(request: ProcessRequest):
    """Process a document and store its vectors."""
    if not _processor:
        raise HTTPException(status_code=503, detail="RAG processor not initialized")

    try:
        result = _processor.process_document(
            file_path=request.file_path,
            document_id=request.document_id,
            collection_name=request.collection_name,
            metadata=request.metadata
        )

        return {
            "success": result.success,
            "document_id": result.document_id,
            "chunks_created": len(result.chunks),
            "error": result.error,
            "trace_id": get_current_trace_id(),
        }
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_knowledge(request: SearchRequest):
    """Search the knowledge base for relevant documents."""
    if not _processor:
        raise HTTPException(status_code=503, detail="RAG processor not initialized")

    try:
        results = _processor.search(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
            min_score=request.min_score
        )

        return {
            "query": request.query,
            "results": results,
            "total": len(results),
            "trace_id": get_current_trace_id(),
        }
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    document_id: str = Body(...),
    collection_name: str = Body(...),
):
    """Upload a file and process it."""
    if not _processor:
        raise HTTPException(status_code=503, detail="RAG processor not initialized")

    import tempfile
    import shutil

    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        result = _processor.process_document(
            file_path=temp_path,
            document_id=document_id,
            collection_name=collection_name,
            metadata={"filename": file.filename, "content_type": file.content_type}
        )

        return {
            "success": result.success,
            "document_id": result.document_id,
            "chunks_created": len(result.chunks),
            "error": result.error,
            "trace_id": get_current_trace_id(),
        }
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import os
        os.unlink(temp_path)


@router.delete("/{collection_name}/{document_id}")
async def delete_document(collection_name: str, document_id: str):
    """Delete all chunks for a document from a collection."""
    if not _processor:
        raise HTTPException(status_code=503, detail="RAG processor not initialized")

    try:
        count = _processor.delete_document(document_id, collection_name)
        return {
            "success": True,
            "document_id": document_id,
            "chunks_deleted": count,
            "trace_id": get_current_trace_id(),
        }
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
