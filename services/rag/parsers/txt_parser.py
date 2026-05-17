"""
Plain text file parser (TXT and Markdown).
"""
import logging
import re
from pathlib import Path
from .base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class TXTParser(BaseParser):
    """Parser for plain text files."""

    SUPPORTED_EXTENSIONS = ['.txt', '.md']

    def parse(self, file_path: str) -> ParsedDocument:
        """Parse text file."""
        try:
            text = self.extract_text(file_path)
            title = self._extract_title(file_path, text)
            return ParsedDocument(
                content=text,
                metadata={
                    "source": file_path,
                    "type": Path(file_path).suffix.lstrip('.'),
                    "title": title
                },
                title=title
            )
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}")
            raise

    def extract_text(self, file_path: str) -> str:
        """Extract plain text from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            for encoding in ['gbk', 'gb2312', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Unable to decode file {file_path}")

    def _extract_title(self, file_path: str, content: str) -> str:
        """Extract title from markdown or filename."""
        if file_path.lower().endswith('.md'):
            match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if match:
                return match.group(1).strip()
        return Path(file_path).stem
