"""
Test Redis connection and caching functionality
"""

import redis
import time
from app.services.embeddings import get_embedding_service

def test_redis_connection():
    """Test basic Redis connectivity"""
    print("=" * 60)
    print("Testing Redis Connection")
    print("=" * 60)

    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis is running and accessible!")

        # Test basic operations
        r.set('test_key', 'Hello from DevRAG!')
        value = r.get('test_key')
        print(f"✅ Redis read/write test: {value}")

        # Get info
        info = r.info('server')
        print(f"✅ Redis version: {info['redis_version']}")

        return True

    except redis.exceptions.ConnectionError:
        print("❌ Redis is NOT running!")
        print("   Please start Redis using: backend/start_redis.bat")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_embedding_cache():
    """Test embedding service with Redis caching"""
    print("\n" + "=" * 60)
    print("Testing Embedding Cache")
    print("=" * 60)

    try:
        service = get_embedding_service()
        info = service.get_embedding_info()
        print(f"📊 Embedding service: {info['provider']}")
        print(f"📊 Model: {info['model']}")
        print(f"📊 Caching enabled: {info['caching_enabled']}")

        if not info['caching_enabled']:
            print("⚠️  Warning: Caching is not enabled!")
            print("   Redis might not be running.")
            return False

        # Test cache performance
        test_texts = [
            "What is machine learning?",
            "How does Python work?",
            "Explain FastAPI benefits"
        ]

        print(f"\n🔄 Generating embeddings (first time - should be SLOW)...")
        start = time.time()
        embeddings1 = service.generate_embeddings(test_texts)
        time1 = (time.time() - start) * 1000
        print(f"   ⏱️  Time: {time1:.0f}ms")

        print(f"\n🔄 Generating same embeddings (cached - should be FAST)...")
        start = time.time()
        embeddings2 = service.generate_embeddings(test_texts)
        time2 = (time.time() - start) * 1000
        print(f"   ⏱️  Time: {time2:.0f}ms")

        speedup = time1 / time2 if time2 > 0 else 0
        print(f"\n🚀 Cache speedup: {speedup:.1f}x faster!")

        if speedup > 10:
            print("✅ Excellent! Caching is working perfectly!")
        elif speedup > 5:
            print("✅ Good! Caching is working well!")
        else:
            print("⚠️  Caching might not be working optimally")

        return True

    except Exception as e:
        print(f"❌ Error testing cache: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cache_stats():
    """Check Redis cache statistics"""
    print("\n" + "=" * 60)
    print("Cache Statistics")
    print("=" * 60)

    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)

        # Count embedding keys
        embedding_keys = r.keys('embedding:*')
        print(f"📊 Cached embeddings: {len(embedding_keys)}")

        # Get memory usage
        info = r.info('memory')
        memory_mb = info['used_memory'] / (1024 * 1024)
        print(f"📊 Redis memory usage: {memory_mb:.2f} MB")

        # Show sample keys
        if embedding_keys:
            print(f"\n📝 Sample cached embeddings:")
            for key in embedding_keys[:3]:
                ttl = r.ttl(key)
                ttl_days = ttl / (24 * 60 * 60) if ttl > 0 else 0
                print(f"   {key[:50]}... (expires in {ttl_days:.1f} days)")

        return True

    except Exception as e:
        print(f"❌ Error checking stats: {e}")
        return False


if __name__ == "__main__":
    print("\n🚀 DevRAG Redis Testing Suite\n")

    # Test 1: Redis connection
    if not test_redis_connection():
        print("\n❌ Redis is not running. Please start it first!")
        print("   Run: backend\\start_redis.bat")
        exit(1)

    # Test 2: Embedding cache
    test_embedding_cache()

    # Test 3: Cache statistics
    check_cache_stats()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run benchmark: python tests/benchmark_suite.py --skip-processing")
    print("2. Check cache metrics: curl http://localhost:8000/api/v1/metrics/cache")
    print("3. View performance: curl http://localhost:8000/api/v1/metrics/performance")
