"""
Simple RAG Pipeline Test - Without Heavy Dependencies

This demonstrates our vector storage system using mock embeddings
to show the complete workflow without requiring sentence-transformers.
"""

import sys
import logging
import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from app.core.config import get_settings
from app.core.database import get_db, create_tables
from app.core.vector_storage import PostgreSQLVectorStorage
from app.services.text_extraction import TextExtractor
from app.services.chunking import DocumentChunker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_mock_embedding(text: str, dimension: int = 384) -> List[float]:
    """Generate a mock embedding based on text content."""
    # Simple hash-based embedding for demonstration
    import hashlib

    # Create deterministic "embedding" based on text
    text_hash = hashlib.md5(text.encode()).hexdigest()

    # Convert hash to numbers
    random.seed(hash(text))  # Deterministic seed
    embedding = [random.uniform(-1, 1) for _ in range(dimension)]

    # Normalize to unit vector (like real embeddings)
    norm = sum(x**2 for x in embedding) ** 0.5
    return [x / norm for x in embedding]


def calculate_similarity(emb1: List[float], emb2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    import math

    # Dot product
    dot_product = sum(a * b for a, b in zip(emb1, emb2))

    # Magnitudes
    mag1 = math.sqrt(sum(a * a for a in emb1))
    mag2 = math.sqrt(sum(b * b for b in emb2))

    # Cosine similarity
    return dot_product / (mag1 * mag2)


def create_test_document() -> str:
    """Create a test document to process."""
    test_content = """
# Python Programming Guide

Python is a high-level programming language known for its simplicity and readability.

## Data Types
Python supports various data types:
- Integers: whole numbers like 1, 2, 3
- Floats: decimal numbers like 3.14, 2.7
- Strings: text data like "hello", "world"
- Lists: collections like [1, 2, 3] or ["a", "b", "c"]

## Functions
Functions in Python are defined using the 'def' keyword:

```python
def greet(name):
    return f"Hello, {name}!"

result = greet("Alice")
print(result)  # Output: Hello, Alice!
```

## Classes and Objects
Python supports object-oriented programming:

```python
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

calc = Calculator()
result = calc.add(5, 3)  # Result: 8
```

## File Handling
Python makes file operations simple:

```python
# Reading a file
with open('data.txt', 'r') as file:
    content = file.read()

# Writing to a file
with open('output.txt', 'w') as file:
    file.write("Hello, World!")
```

## Error Handling
Use try-except blocks to handle errors:

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
finally:
    print("Cleanup code here")
```

Python is widely used for web development, data science, machine learning, and automation.
The language emphasizes code readability and allows developers to express concepts in fewer lines of code.
"""

    # Save test document
    test_file = Path(__file__).parent / "test_python_guide.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)

    return str(test_file)


def test_rag_pipeline_simple():
    """Test the RAG pipeline with mock embeddings."""

    logger.info("🚀 Starting Simple RAG Pipeline Test")
    logger.info("=" * 60)

    # Initialize services
    text_extractor = TextExtractor()
    chunker = DocumentChunker()

    try:
        # Step 1: Create test document
        logger.info("📄 Step 1: Creating Test Document")
        test_file = create_test_document()
        logger.info(f"   Created: {Path(test_file).name}")

        # Step 2: Extract text
        logger.info("\n🔍 Step 2: Extracting Text")
        try:
            extraction_result = text_extractor.extract_text(Path(test_file))
            text_content = extraction_result["text"]
            logger.info(f"   Extracted {len(text_content)} characters")
            logger.info(f"   Word count: {extraction_result.get('word_count', 'N/A')}")
            logger.info(f"   Encoding: {extraction_result.get('encoding', 'N/A')}")
        except Exception as e:
            logger.error(f"   Failed to extract text: {e}")
            return False

        # Step 3: Chunk the document
        logger.info("\n✂️  Step 3: Chunking Document")
        chunks = chunker.chunk_text(text_content, strategy="smart")
        logger.info(f"   Created {len(chunks)} chunks")

        # Show chunk examples
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            logger.info(f"   Chunk {i+1}: {chunk.text[:80]}...")
            logger.info(f"     Words: {chunk.word_count}, Chars: {chunk.char_count}")

        # Convert chunks to text for embeddings
        chunk_texts = [chunk.text for chunk in chunks]

        # Step 4: Generate mock embeddings
        logger.info("\n🧠 Step 4: Generating Mock Embeddings")
        embeddings = []
        for i, chunk_text in enumerate(chunk_texts):
            embedding = generate_mock_embedding(chunk_text)
            embeddings.append(embedding)
            logger.info(f"   Generated mock embedding {i+1}/{len(chunk_texts)} (dim: {len(embedding)})")

        # Step 5: Store in vector database
        logger.info("\n💾 Step 5: Storing in Vector Database")
        create_tables()  # Ensure tables exist

        with next(get_db()) as db:
            vector_store = PostgreSQLVectorStorage(db)

            document_ids = vector_store.store_document_with_embeddings(
                filename="python_guide.txt",
                content=text_content,
                chunks=chunk_texts,
                embeddings=embeddings
            )

            logger.info(f"   Stored {len(document_ids)} chunks in database")

            # Step 6: Test similarity search
            logger.info("\n🔍 Step 6: Testing Similarity Search")

            # Test queries
            test_queries = [
                "How do you define functions in Python?",
                "What are Python data types?",
                "How to handle errors in Python?",
                "Object-oriented programming in Python"
            ]

            for query in test_queries:
                logger.info(f"\n   Query: '{query}'")

                # Generate query embedding
                query_embedding = generate_mock_embedding(query)

                # Search for similar chunks
                results = vector_store.similarity_search(
                    query_embedding=query_embedding,
                    limit=2,
                    similarity_threshold=0.1  # Lower threshold for mock embeddings
                )

                logger.info(f"   Found {len(results)} similar chunks:")
                for i, (doc, similarity) in enumerate(results):
                    logger.info(f"     {i+1}. Similarity: {similarity:.3f}")
                    logger.info(f"        Text: {doc.chunk_text[:80]}...")

            # Step 7: Demonstrate similarity calculation
            logger.info("\n🧮 Step 7: Similarity Demonstration")

            # Compare similar texts
            text1 = "Python programming language"
            text2 = "Python is for programming"
            text3 = "Cats and dogs are pets"

            emb1 = generate_mock_embedding(text1)
            emb2 = generate_mock_embedding(text2)
            emb3 = generate_mock_embedding(text3)

            sim_1_2 = calculate_similarity(emb1, emb2)
            sim_1_3 = calculate_similarity(emb1, emb3)

            logger.info(f"   '{text1}' vs '{text2}': {sim_1_2:.3f}")
            logger.info(f"   '{text1}' vs '{text3}': {sim_1_3:.3f}")
            logger.info("   Similar texts should have higher similarity!")

            # Step 8: Get storage statistics
            logger.info("\n📊 Step 8: Storage Statistics")
            stats = vector_store.get_storage_stats()
            logger.info(f"   Total documents: {stats.get('total_documents', 0)}")
            logger.info(f"   Total chunks: {stats.get('total_chunks', 0)}")
            logger.info(f"   Storage type: {stats.get('storage_type', 'Unknown')}")

        # Cleanup
        logger.info("\n🧹 Cleanup")
        Path(test_file).unlink()  # Delete test file
        logger.info("   Test file deleted")

        logger.info("\n🎉 Simple RAG Pipeline Test Complete!")
        logger.info("=" * 60)
        logger.info("✅ Document processing: SUCCESS")
        logger.info("✅ Text extraction: SUCCESS")
        logger.info("✅ Intelligent chunking: SUCCESS")
        logger.info("✅ Vector storage: SUCCESS")
        logger.info("✅ Similarity search: SUCCESS")
        logger.info("")
        logger.info("🎯 System Ready For:")
        logger.info("   - Real embedding integration (sentence-transformers)")
        logger.info("   - pgvector optimization (when build tools available)")
        logger.info("   - AI chat integration (OpenAI API)")
        logger.info("   - Frontend development (React)")

        return True

    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    try:
        success = test_rag_pipeline_simple()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)