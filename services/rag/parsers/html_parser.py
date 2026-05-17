"""
HTML document parser.
"""
import logging
import re
from pathlib import Path
from .base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class HTMLParser(BaseParser):
    """Parser for HTML documents."""

    SUPPORTED_EXTENSIONS = ['.html', '.htm']

    def parse(self, file_path: str) -> ParsedDocument:
        """Parse HTML file."""
        try:
            text = self.extract_text(file_path)
            return ParsedDocument(
                content=text,
                metadata={
                    "source": file_path,
                    "type": "html"
                }
            )
        except Exception as e:
            logger.error(f"Error parsing HTML {file_path}: {e}")
            raise

    def extract_text(self, file_path: str) -> str:
        """Extract plain text from HTML."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html = f.read()
            return self._strip_html(html)
        except Exception as e:
            logger.error(f"Error reading HTML file {file_path}: {e}")
            raise

    def _strip_html(self, html: str) -> str:
        """Remove HTML tags and return plain text."""
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<(?:p|div|br|h[1-6]|li|tr)[^>]*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'<[^>]+>', '', html)
        html = re.sub(r'\n\s*\n', '\n\n', html)
        html = re.sub(r'[ \t]+', ' ', html)
        return html.strip()
