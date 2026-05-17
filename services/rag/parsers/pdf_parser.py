"""
PDF document parser.
"""
import logging
from typing import Optional
from .base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """Parser for PDF documents."""

    SUPPORTED_EXTENSIONS = ['.pdf']

    def __init__(self):
        self._pypdf_available = False
        try:
            import pypdf
            self.pypdf = pypdf
            self._pypdf_available = True
        except ImportError:
            logger.warning("pypdf not installed. PDF parsing will be limited.")

    def parse(self, file_path: str) -> ParsedDocument:
        """Parse PDF file and extract content."""
        try:
            text = self.extract_text(file_path)
            return ParsedDocument(
                content=text,
                metadata={"source": file_path, "type": "pdf"},
                pages=self._count_pages(file_path)
            )
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise

    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF using pypdf."""
        if not self._pypdf_available:
            raise ImportError("pypdf is required for PDF parsing. Install with: pip install pypdf")

        try:
            reader = self.pypdf.PdfReader(file_path)
            text_parts = []

            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num}]\n{page_text}")

            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise

    def _count_pages(self, file_path: str) -> int:
        """Count pages in PDF."""
        if not self._pypdf_available:
            return 0
        try:
            reader = self.pypdf.PdfReader(file_path)
            return len(reader.pages)
        except Exception:
            return 0
