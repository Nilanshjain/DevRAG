"""
Embedding Generation Service
Converts text chunks into vector embeddings for similarity search

This is where the magic happens! We convert human text into numbers
that computers can understand and compare for similarity.
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
import logging
import hashlib
import json

# For local embeddings (no API required)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# For OpenAI embeddings (API required)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# For caching embeddings
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using various providers

    Embeddings are vector representations of text that capture semantic meaning.
    Similar texts will have similar vectors, allowing us to find relevant
    content through vector similarity search.
    """

    def __init__(
        self,
        provider: str = "sentence_transformers",
        model_name: Optional[str] = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize the embedding service

        Args:
            provider: "sentence_transformers" or "openai"
            model_name: Specific model to use (None for defaults)
            cache_embeddings: Whether to cache embeddings for performance
        """
        self.provider = provider
        self.cache_embeddings = cache_embeddings
        self.model = None
        self.redis_client = None

        # Setup caching if available
        if cache_embeddings and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis caching enabled for embeddings")
            except Exception as e:
                logger.warning(f"Redis not available for caching: {e}")
                self.redis_client = None

        # Initialize the embedding model
        self._initialize_model(model_name)

    def _initialize_model(self, model_name: Optional[str]):
        """Initialize the embedding model based on provider"""

        if self.provider == "sentence_transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")

            # Default to a good general-purpose model
            model_name = model_name or "all-MiniLM-L6-v2"

            try:
                logger.info(f"Loading sentence transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")

            except Exception as e:
                logger.error(f"Failed to load sentence transformer model: {e}")
                raise

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not available. Install with: pip install openai")

            # Set default model
            self.model_name = model_name or "text-embedding-ada-002"
            self.embedding_dimension = 1536  # OpenAI ada-002 dimension

            # Check for API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required for OpenAI embeddings")

            # Initialize OpenAI client
            self.model = OpenAI(api_key=api_key)
            logger.info(f"OpenAI embeddings initialized with model: {self.model_name}")

        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Check cache first
        cache_key = None
        if self.redis_client:
            cache_key = self._get_cache_key(text)
            cached_embedding = self._get_cached_embedding(cache_key)
            if cached_embedding:
                return cached_embedding

        # Generate new embedding
        start_time = time.time()
        embedding = self._generate_embedding_impl(text)
        generation_time = time.time() - start_time

        logger.debug(f"Generated embedding in {generation_time:.3f}s for text: {text[:100]}...")

        # Cache the result
        if self.redis_client and cache_key:
            self._cache_embedding(cache_key, embedding)

        return embedding

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]

        if not valid_texts:
            return []

        # Check cache for batch items
        embeddings = []
        texts_to_generate = []
        cache_keys = []

        for text in valid_texts:
            cache_key = None
            if self.redis_client:
                cache_key = self._get_cache_key(text)
                cached_embedding = self._get_cached_embedding(cache_key)
                if cached_embedding:
                    embeddings.append(cached_embedding)
                    cache_keys.append(None)  # Already cached
                    continue

            # Need to generate this one
            embeddings.append(None)  # Placeholder
            texts_to_generate.append(text)
            cache_keys.append(cache_key)

        # Generate embeddings for uncached texts
        if texts_to_generate:
            start_time = time.time()
            new_embeddings = self._generate_embeddings_impl(texts_to_generate)
            generation_time = time.time() - start_time

            logger.info(f"Generated {len(new_embeddings)} embeddings in {generation_time:.3f}s")

            # Fill in the placeholders and cache results
            new_embedding_idx = 0
            for i, embedding in enumerate(embeddings):
                if embedding is None:  # This was a placeholder
                    new_embedding = new_embeddings[new_embedding_idx]
                    embeddings[i] = new_embedding
                    new_embedding_idx += 1

                    # Cache the new embedding
                    if self.redis_client and cache_keys[i]:
                        self._cache_embedding(cache_keys[i], new_embedding)

        return embeddings

    def _generate_embedding_impl(self, text: str) -> List[float]:
        """Implementation-specific embedding generation"""

        if self.provider == "sentence_transformers":
            # Use local sentence transformer model
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()

        elif self.provider == "openai":
            # Use OpenAI API (v1.0+ syntax)
            try:
                response = self.model.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                return response.data[0].embedding

            except Exception as e:
                logger.error(f"OpenAI embedding generation failed: {e}")
                raise

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _generate_embeddings_impl(self, texts: List[str]) -> List[List[float]]:
        """Implementation-specific batch embedding generation"""

        if self.provider == "sentence_transformers":
            # Batch processing with sentence transformers
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return [embedding.tolist() for embedding in embeddings]

        elif self.provider == "openai":
            # OpenAI batch processing (v1.0+ syntax)
            try:
                response = self.model.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                return [item.embedding for item in response.data]

            except Exception as e:
                logger.error(f"OpenAI batch embedding generation failed: {e}")
                raise

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Create hash of text + model info for cache key
        content = f"{self.provider}:{getattr(self, 'model_name', 'default')}:{text}"
        return f"embedding:{hashlib.md5(content.encode()).hexdigest()}"

    def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Retrieve embedding from cache"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None

    def _cache_embedding(self, cache_key: str, embedding: List[float]):
        """Store embedding in cache"""
        try:
            # Cache for 1 week (embeddings don't change)
            self.redis_client.setex(
                cache_key,
                7 * 24 * 60 * 60,  # 1 week in seconds
                json.dumps(embedding)
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding service"""
        return {
            "provider": self.provider,
            "model": getattr(self, 'model_name', getattr(self.model, '_model_name', 'unknown')),
            "dimension": self.embedding_dimension,
            "caching_enabled": self.redis_client is not None
        }

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1 (higher is more similar)
        """
        import math

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(a * a for a in embedding2))

        # Calculate cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


# Global instance for easy importing
# We'll use sentence transformers by default (no API key required)
embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance"""
    global embedding_service

    if embedding_service is None:
        # Try OpenAI first if API key is available (no disk space needed for PyTorch)
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            embedding_service = EmbeddingService("openai")
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            embedding_service = EmbeddingService("sentence_transformers")
        else:
            raise RuntimeError(
                "No embedding provider available. Install sentence-transformers or set OPENAI_API_KEY"
            )

    return embedding_service


def generate_embedding(text: str) -> List[float]:
    """Convenience function to generate a single embedding"""
    service = get_embedding_service()
    return service.generate_embedding(text)


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Convenience function to generate multiple embeddings"""
    service = get_embedding_service()
    return service.generate_embeddings(texts)


# Test function
def test_embeddings():
    """Test embedding generation"""
    try:
        service = get_embedding_service()
        print(f"Embedding service info: {service.get_embedding_info()}")

        # Test single embedding
        test_text = "This is a test sentence for embedding generation."
        embedding = service.generate_embedding(test_text)
        print(f"Generated embedding with {len(embedding)} dimensions")

        # Test similarity
        text1 = "The cat sat on the mat"
        text2 = "A feline was sitting on a rug"
        text3 = "The weather is nice today"

        emb1 = service.generate_embedding(text1)
        emb2 = service.generate_embedding(text2)
        emb3 = service.generate_embedding(text3)

        sim_12 = service.calculate_similarity(emb1, emb2)
        sim_13 = service.calculate_similarity(emb1, emb3)

        print(f"Similarity between '{text1}' and '{text2}': {sim_12:.3f}")
        print(f"Similarity between '{text1}' and '{text3}': {sim_13:.3f}")

        print("✅ Embedding service working correctly!")
        return True

    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False


if __name__ == "__main__":
    test_embeddings()