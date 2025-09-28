"""
Document Chunking Service
Splits large documents into smaller, manageable pieces for better AI processing
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """
    Represents a chunk of text with metadata

    This class holds a piece of text along with information
    about where it came from and its characteristics.
    """
    text: str              # The actual text content
    chunk_id: str          # Unique identifier for this chunk
    source_file: str       # Original file name
    chunk_index: int       # Position in the document (0, 1, 2...)
    char_count: int        # Number of characters
    word_count: int        # Number of words
    start_char: int        # Starting character position in original text
    end_char: int          # Ending character position in original text
    metadata: Dict[str, Any] = None  # Additional information

    def __post_init__(self):
        """Calculate derived properties after initialization"""
        if self.metadata is None:
            self.metadata = {}

        # Ensure counts are accurate
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())


class DocumentChunker:
    """
    Splits documents into chunks for optimal AI processing

    Different chunking strategies work better for different use cases:
    - Fixed size: Simple, predictable chunks
    - Sentence-aware: Respects sentence boundaries
    - Paragraph-aware: Keeps related content together
    - Code-aware: Understands code structure
    """

    def __init__(
        self,
        chunk_size: int = 1000,      # Target characters per chunk
        chunk_overlap: int = 200,     # Characters to overlap between chunks
        min_chunk_size: int = 100     # Minimum chunk size to keep
    ):
        """
        Initialize the chunker with configuration

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: How much chunks should overlap
            min_chunk_size: Minimum size to consider a valid chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(
        self,
        text: str,
        source_file: str = "unknown",
        strategy: str = "smart"
    ) -> List[Chunk]:
        """
        Split text into chunks using the specified strategy

        Args:
            text: The text to chunk
            source_file: Name of the source file
            strategy: Chunking strategy ('fixed', 'sentence', 'paragraph', 'smart', 'code')

        Returns:
            List of Chunk objects
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        # Choose chunking method based on strategy
        if strategy == "fixed":
            return self._chunk_fixed_size(text, source_file)
        elif strategy == "sentence":
            return self._chunk_by_sentences(text, source_file)
        elif strategy == "paragraph":
            return self._chunk_by_paragraphs(text, source_file)
        elif strategy == "code":
            return self._chunk_code_aware(text, source_file)
        elif strategy == "smart":
            return self._chunk_smart(text, source_file)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _chunk_fixed_size(self, text: str, source_file: str) -> List[Chunk]:
        """
        Split text into fixed-size chunks

        This is the simplest strategy - just cut the text every N characters.
        Not ideal for readability but guaranteed consistent sizes.
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # If this is not the last chunk, try to find a good break point
            if end < len(text):
                # Look for space within the last 100 characters
                search_start = max(end - 100, start)
                space_pos = text.rfind(' ', search_start, end)

                if space_pos > start:
                    end = space_pos

            # Extract chunk text
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{source_file}_chunk_{chunk_index}",
                    source_file=source_file,
                    chunk_index=chunk_index,
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    start_char=start,
                    end_char=end,
                    metadata={"strategy": "fixed"}
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move to next chunk with overlap
            start = end - self.chunk_overlap

        return chunks

    def _chunk_by_sentences(self, text: str, source_file: str) -> List[Chunk]:
        """
        Split text by sentences, grouping them into chunks

        This strategy respects sentence boundaries for better readability
        and context preservation.
        """
        # Split into sentences using regex
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for i, sentence in enumerate(sentences):
            # Would adding this sentence exceed our chunk size?
            if (len(current_chunk) + len(sentence) > self.chunk_size and
                len(current_chunk) > 0):

                # Create chunk from current content
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunk = Chunk(
                        text=current_chunk.strip(),
                        chunk_id=f"{source_file}_chunk_{chunk_index}",
                        source_file=source_file,
                        chunk_index=chunk_index,
                        char_count=len(current_chunk.strip()),
                        word_count=len(current_chunk.strip().split()),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        metadata={"strategy": "sentence", "sentence_count": current_chunk.count('.')}
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with overlap (include last sentence if space allows)
                if self.chunk_overlap > 0 and len(sentence) < self.chunk_overlap:
                    current_chunk = sentence + " "
                    current_start = current_start + len(current_chunk) - len(sentence) - 1
                else:
                    current_chunk = ""
                    current_start = current_start + len(current_chunk)

            current_chunk += sentence + " "

        # Handle remaining content
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunk = Chunk(
                text=current_chunk.strip(),
                chunk_id=f"{source_file}_chunk_{chunk_index}",
                source_file=source_file,
                chunk_index=chunk_index,
                char_count=len(current_chunk.strip()),
                word_count=len(current_chunk.strip().split()),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={"strategy": "sentence", "sentence_count": current_chunk.count('.')}
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_paragraphs(self, text: str, source_file: str) -> List[Chunk]:
        """
        Split text by paragraphs, keeping related content together

        This strategy is great for documents with clear paragraph structure.
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for paragraph in paragraphs:
            # Would adding this paragraph exceed our chunk size?
            if (len(current_chunk) + len(paragraph) > self.chunk_size and
                len(current_chunk) > 0):

                # Create chunk from current content
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunk = Chunk(
                        text=current_chunk.strip(),
                        chunk_id=f"{source_file}_chunk_{chunk_index}",
                        source_file=source_file,
                        chunk_index=chunk_index,
                        char_count=len(current_chunk.strip()),
                        word_count=len(current_chunk.strip().split()),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        metadata={"strategy": "paragraph", "paragraph_count": current_chunk.count('\n\n') + 1}
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk
                current_chunk = ""
                current_start = current_start + len(current_chunk)

            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

        # Handle remaining content
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunk = Chunk(
                text=current_chunk.strip(),
                chunk_id=f"{source_file}_chunk_{chunk_index}",
                source_file=source_file,
                chunk_index=chunk_index,
                char_count=len(current_chunk.strip()),
                word_count=len(current_chunk.strip().split()),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={"strategy": "paragraph", "paragraph_count": current_chunk.count('\n\n') + 1}
            )
            chunks.append(chunk)

        return chunks

    def _chunk_code_aware(self, text: str, source_file: str) -> List[Chunk]:
        """
        Split code while respecting structure (functions, classes, etc.)

        This strategy understands code structure and tries to keep
        related code blocks together.
        """
        # Detect if this looks like code
        code_indicators = [
            'def ', 'class ', 'function ', 'var ', 'const ',
            'import ', 'from ', '#include', '#!/'
        ]

        is_code = any(indicator in text for indicator in code_indicators)

        if not is_code:
            # Fall back to smart chunking for non-code
            return self._chunk_smart(text, source_file)

        # For code, try to split by function/class boundaries
        lines = text.split('\n')
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        line_start = 0

        for i, line in enumerate(lines):
            # Check if this line starts a new major block
            is_major_boundary = (
                line.strip().startswith('def ') or
                line.strip().startswith('class ') or
                line.strip().startswith('function ') or
                line.strip().startswith('async def ')
            )

            # Would adding this line exceed our chunk size?
            if (len(current_chunk) + len(line) > self.chunk_size and
                len(current_chunk) > 0 and
                (is_major_boundary or i - line_start > 50)):  # Don't break in middle of large functions

                # Create chunk from current content
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunk = Chunk(
                        text=current_chunk.strip(),
                        chunk_id=f"{source_file}_chunk_{chunk_index}",
                        source_file=source_file,
                        chunk_index=chunk_index,
                        char_count=len(current_chunk.strip()),
                        word_count=len(current_chunk.strip().split()),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        metadata={
                            "strategy": "code",
                            "line_count": current_chunk.count('\n') + 1,
                            "is_code": True
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk
                current_chunk = ""
                current_start = current_start + len(current_chunk)
                line_start = i

            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line

        # Handle remaining content
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunk = Chunk(
                text=current_chunk.strip(),
                chunk_id=f"{source_file}_chunk_{chunk_index}",
                source_file=source_file,
                chunk_index=chunk_index,
                char_count=len(current_chunk.strip()),
                word_count=len(current_chunk.strip().split()),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={
                    "strategy": "code",
                    "line_count": current_chunk.count('\n') + 1,
                    "is_code": True
                }
            )
            chunks.append(chunk)

        return chunks

    def _chunk_smart(self, text: str, source_file: str) -> List[Chunk]:
        """
        Intelligent chunking that adapts to content type

        This strategy examines the text and chooses the best approach:
        - Code files: Use code-aware chunking
        - Documents with clear paragraphs: Use paragraph chunking
        - Other text: Use sentence-aware chunking
        """
        # Analyze the text to determine best strategy
        file_ext = Path(source_file).suffix.lower()

        # Code files
        if file_ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.html', '.css']:
            return self._chunk_code_aware(text, source_file)

        # Check paragraph structure
        paragraph_count = text.count('\n\n')
        line_count = text.count('\n')

        # If we have clear paragraph breaks (more than 10% of lines are paragraph breaks)
        if paragraph_count > 3 and paragraph_count / line_count > 0.1:
            return self._chunk_by_paragraphs(text, source_file)

        # Default to sentence-aware chunking
        return self._chunk_by_sentences(text, source_file)

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex

        This handles common sentence endings while avoiding
        false positives like abbreviations.
        """
        # Simple sentence splitting - can be improved with more sophisticated NLP
        sentence_endings = r'[.!?]+(?:\s|$)'
        sentences = re.split(sentence_endings, text)

        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def get_chunking_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Generate statistics about the chunking results

        Args:
            chunks: List of chunks to analyze

        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {"total_chunks": 0}

        char_counts = [chunk.char_count for chunk in chunks]
        word_counts = [chunk.word_count for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(char_counts),
            "total_words": sum(word_counts),
            "avg_chunk_size": sum(char_counts) / len(char_counts),
            "min_chunk_size": min(char_counts),
            "max_chunk_size": max(char_counts),
            "avg_words_per_chunk": sum(word_counts) / len(word_counts),
            "strategies_used": list(set(chunk.metadata.get("strategy", "unknown") for chunk in chunks))
        }


# Global instance for easy importing
document_chunker = DocumentChunker()


def chunk_text(text: str, source_file: str = "unknown", strategy: str = "smart") -> List[Chunk]:
    """
    Convenience function to chunk text

    Args:
        text: Text to chunk
        source_file: Source filename
        strategy: Chunking strategy

    Returns:
        List of chunks
    """
    return document_chunker.chunk_text(text, source_file, strategy)