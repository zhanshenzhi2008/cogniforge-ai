"""
Document parser factory and registry.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Type

from .base import BaseParser, ParsedDocument
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .txt_parser import TXTParser
from .html_parser import HTMLParser

logger = logging.getLogger(__name__)


class ParserRegistry:
    """Registry for document parsers."""

    def __init__(self):
        self._parsers: Dict[str, BaseParser] = {}

        self.register('.pdf', PDFParser())
        self.register('.docx', DOCXParser())
        self.register('.txt', TXTParser())
        self.register('.md', TXTParser())
        self.register('.html', HTMLParser())
        self.register('.htm', HTMLParser())

    def register(self, extension: str, parser: BaseParser):
        """Register a parser for a file extension."""
        self._parsers[extension.lower()] = parser
        logger.debug(f"Registered parser for {extension}: {type(parser).__name__}")

    def get_parser(self, file_path: str) -> Optional[BaseParser]:
        """Get the appropriate parser for a file."""
        ext = Path(file_path).suffix.lower()
        return self._parsers.get(ext)

    def parse(self, file_path: str) -> ParsedDocument:
        """Parse a file using the appropriate parser."""
        parser = self.get_parser(file_path)
        if parser is None:
            raise ValueError(f"No parser available for file: {file_path}")
        return parser.parse(file_path)

    def extract_text(self, file_path: str) -> str:
        """Extract text from a file."""
        parser = self.get_parser(file_path)
        if parser is None:
            raise ValueError(f"No parser available for file: {file_path}")
        return parser.extract_text(file_path)

    def supported_extensions(self) -> list:
        """Get list of supported file extensions."""
        return list(self._parsers.keys())


registry = ParserRegistry()
