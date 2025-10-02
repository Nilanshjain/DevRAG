"""
Text Extraction Service
Handles extracting text content from various file formats
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Extracts text from various file formats

    This class handles the conversion of files (PDF, TXT, etc.)
    into plain text that our AI can understand.
    """

    def __init__(self):
        """Initialize the text extractor"""
        self.supported_formats = {
            '.txt': self._extract_text_file,
            '.md': self._extract_text_file,
            '.py': self._extract_text_file,
            '.js': self._extract_text_file,
            '.ts': self._extract_text_file,
            '.html': self._extract_text_file,
            '.css': self._extract_text_file,
            '.json': self._extract_text_file,
            '.yaml': self._extract_text_file,
            '.yml': self._extract_text_file,
        }

        # Add PDF support if PyPDF2 is available
        if PDF_AVAILABLE:
            self.supported_formats['.pdf'] = self._extract_pdf

    def extract_text(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from a file

        Args:
            file_path: Path to the file to extract text from

        Returns:
            Dictionary containing extracted text and metadata

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file extension
        extension = file_path.suffix.lower()

        # Check if format is supported
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")

        # Extract text using appropriate method
        extractor_func = self.supported_formats[extension]

        try:
            result = extractor_func(file_path)

            # Add metadata
            file_stat = file_path.stat()
            result.update({
                'file_name': file_path.name,
                'file_size': file_stat.st_size,
                'file_extension': extension,
                'extraction_method': extractor_func.__name__
            })

            return result

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise ValueError(f"Failed to extract text: {str(e)}")

    def _extract_text_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from plain text files

        Args:
            file_path: Path to text file

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()

                    return {
                        'text': content,
                        'encoding': encoding,
                        'word_count': len(content.split()),
                        'char_count': len(content),
                        'line_count': content.count('\n') + 1
                    }

                except UnicodeDecodeError:
                    continue

            # If all encodings fail
            raise ValueError("Could not decode file with any supported encoding")

        except Exception as e:
            raise ValueError(f"Error reading text file: {str(e)}")

    def _extract_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from PDF files using PyPDF2

        Args:
            file_path: Path to PDF file

        Returns:
            Dictionary with extracted text and metadata
        """
        if not PDF_AVAILABLE:
            raise ValueError("PDF extraction not available. Install PyPDF2")

        try:
            extracted_text = ""
            page_count = 0

            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)

                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += f"\n--- Page {page_num + 1} ---\n"
                            extracted_text += page_text
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                        continue

            # Clean up the text
            cleaned_text = self._clean_extracted_text(extracted_text)

            return {
                'text': cleaned_text,
                'page_count': page_count,
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text),
                'extraction_quality': self._assess_extraction_quality(cleaned_text)
            }

        except Exception as e:
            raise ValueError(f"Error extracting PDF: {str(e)}")

    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text with normalized characters
        """
        # Normalize Unicode characters (replace special bullets with standard dashes)
        import re
        import unicodedata

        # Normalize Unicode to ASCII-safe characters
        text = text.replace('\u25cf', '-')  # Replace bullet with dash
        text = text.replace('\u2022', '-')  # Replace bullet with dash
        text = text.replace('\u2023', '-')  # Replace triangular bullet with dash

        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Strip whitespace and skip empty lines
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

        # Join with single newlines
        cleaned_text = '\n'.join(cleaned_lines)

        # Remove multiple consecutive newlines
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        return cleaned_text

    def _assess_extraction_quality(self, text: str) -> str:
        """
        Assess the quality of text extraction

        Args:
            text: Extracted text

        Returns:
            Quality assessment string
        """
        if not text or len(text.strip()) < 10:
            return "poor"

        # Check for reasonable word count
        words = text.split()
        if len(words) < 5:
            return "poor"

        # Check for reasonable character distribution
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text)

        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio > 0.6:
                return "good"
            elif alpha_ratio > 0.3:
                return "fair"

        return "poor"

    def get_supported_formats(self) -> list:
        """
        Get list of supported file formats

        Returns:
            List of supported file extensions
        """
        return list(self.supported_formats.keys())


# Global instance for easy importing
text_extractor = TextExtractor()


def extract_text_from_file(file_path: Path) -> Dict[str, Any]:
    """
    Convenience function to extract text from a file

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with extracted text and metadata
    """
    return text_extractor.extract_text(file_path)


# Example usage and testing function
def test_extraction():
    """Test function to verify text extraction works"""
    test_file = Path("test_file.txt")
    if test_file.exists():
        try:
            result = extract_text_from_file(test_file)
            print(f"Successfully extracted {result['word_count']} words from {test_file.name}")
            print(f"First 100 characters: {result['text'][:100]}...")
            return True
        except Exception as e:
            print(f"Extraction failed: {str(e)}")
            return False
    else:
        print("Test file not found")
        return False


if __name__ == "__main__":
    # Run test when script is executed directly
    test_extraction()