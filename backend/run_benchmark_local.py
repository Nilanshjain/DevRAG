"""
Run benchmark with local sentence-transformers embeddings
"""
import os
import sys

# Remove OpenAI key from environment to force local embeddings
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

# Now run the benchmark
from tests.benchmark_suite import main

if __name__ == "__main__":
    main()
