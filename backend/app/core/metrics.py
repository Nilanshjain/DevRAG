"""
Performance Metrics Module

Centralized metrics collection for monitoring system performance:
- Embedding generation times
- Vector search latency
- Cache hit rates
- API response times
- System health metrics
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class MetricStats:
    """Statistics for a specific metric"""
    count: int = 0
    total: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    recent_values: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_value(self, value: float):
        """Add a new measurement"""
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.recent_values.append(value)

    @property
    def average(self) -> float:
        """Calculate average"""
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def p50(self) -> float:
        """Calculate median (P50)"""
        if not self.recent_values:
            return 0.0
        sorted_values = sorted(self.recent_values)
        return sorted_values[len(sorted_values) // 2]

    @property
    def p95(self) -> float:
        """Calculate 95th percentile"""
        if not self.recent_values:
            return 0.0
        sorted_values = sorted(self.recent_values)
        idx = int(len(sorted_values) * 0.95)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    @property
    def p99(self) -> float:
        """Calculate 99th percentile"""
        if not self.recent_values:
            return 0.0
        sorted_values = sorted(self.recent_values)
        idx = int(len(sorted_values) * 0.99)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "count": self.count,
            "average": round(self.average, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2)
        }


class PerformanceMetrics:
    """
    Centralized performance metrics collection
    Thread-safe singleton for tracking system performance
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Embedding metrics
        self.embedding_single_time = MetricStats()
        self.embedding_batch_time = MetricStats()
        self.embedding_cache_hits = 0
        self.embedding_cache_misses = 0

        # Vector search metrics
        self.vector_search_time = MetricStats()
        self.vector_search_results = MetricStats()

        # API metrics
        self.api_upload_time = MetricStats()
        self.api_process_time = MetricStats()
        self.api_chat_time = MetricStats()

        # System metrics
        self.documents_processed = 0
        self.chunks_created = 0
        self.queries_answered = 0
        self.start_time = datetime.now()

        self._initialized = True

        logger.info("Performance metrics initialized")

    def record_embedding_time(self, time_ms: float, is_batch: bool = False, cache_hit: bool = False):
        """Record embedding generation time"""
        if cache_hit:
            self.embedding_cache_hits += 1
        else:
            self.embedding_cache_misses += 1
            if is_batch:
                self.embedding_batch_time.add_value(time_ms)
            else:
                self.embedding_single_time.add_value(time_ms)

    def record_vector_search(self, time_ms: float, num_results: int):
        """Record vector search operation"""
        self.vector_search_time.add_value(time_ms)
        self.vector_search_results.add_value(num_results)

    def record_api_call(self, endpoint: str, time_ms: float):
        """Record API endpoint call"""
        if "upload" in endpoint:
            self.api_upload_time.add_value(time_ms)
        elif "process" in endpoint:
            self.api_process_time.add_value(time_ms)
            self.documents_processed += 1
        elif "chat" in endpoint:
            self.api_chat_time.add_value(time_ms)
            self.queries_answered += 1

    def record_document_processed(self, num_chunks: int):
        """Record document processing"""
        self.chunks_created += num_chunks

    @property
    def cache_hit_rate(self) -> float:
        """Calculate embedding cache hit rate"""
        total = self.embedding_cache_hits + self.embedding_cache_misses
        return self.embedding_cache_hits / total if total > 0 else 0.0

    @property
    def uptime_seconds(self) -> float:
        """Calculate system uptime"""
        return (datetime.now() - self.start_time).total_seconds()

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            "system": {
                "uptime_seconds": round(self.uptime_seconds, 1),
                "uptime_hours": round(self.uptime_seconds / 3600, 2),
                "start_time": self.start_time.isoformat(),
                "documents_processed": self.documents_processed,
                "chunks_created": self.chunks_created,
                "queries_answered": self.queries_answered
            },
            "embedding_performance": {
                "single_embedding_ms": self.embedding_single_time.to_dict(),
                "batch_embedding_ms": self.embedding_batch_time.to_dict(),
                "cache_hit_rate": round(self.cache_hit_rate, 3),
                "cache_hits": self.embedding_cache_hits,
                "cache_misses": self.embedding_cache_misses
            },
            "vector_search_performance": {
                "search_latency_ms": self.vector_search_time.to_dict(),
                "avg_results_per_search": round(self.vector_search_results.average, 1)
            },
            "api_performance": {
                "upload_time_ms": self.api_upload_time.to_dict(),
                "process_time_ms": self.api_process_time.to_dict(),
                "chat_response_ms": self.api_chat_time.to_dict()
            }
        }

    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        with self._lock:
            self.__init__()
            self._initialized = True
            self.start_time = datetime.now()

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        summary = self.get_summary()

        # Determine health based on performance
        health_status = "healthy"
        warnings = []

        # Check response times
        if self.api_chat_time.average > 2000:  # > 2 seconds
            health_status = "degraded"
            warnings.append("High chat response time")

        if self.vector_search_time.average > 500:  # > 500ms
            warnings.append("High vector search latency")

        if self.cache_hit_rate < 0.3 and self.embedding_cache_misses > 100:
            warnings.append("Low cache hit rate")

        return {
            "status": health_status,
            "uptime_hours": summary["system"]["uptime_hours"],
            "total_queries": self.queries_answered,
            "warnings": warnings,
            "metrics_summary": {
                "avg_chat_response_ms": round(self.api_chat_time.average, 1),
                "avg_search_latency_ms": round(self.vector_search_time.average, 1),
                "cache_hit_rate": round(self.cache_hit_rate, 2)
            }
        }


# Global instance
_metrics = PerformanceMetrics()


def get_metrics() -> PerformanceMetrics:
    """Get the global metrics instance"""
    return _metrics


# Context manager for timing operations
class Timer:
    """Context manager for timing code blocks"""

    def __init__(self, metric_name: str = None):
        self.metric_name = metric_name
        self.start_time = None
        self.elapsed_ms = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000

    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return self.elapsed_ms if self.elapsed_ms is not None else 0.0


# Example usage
if __name__ == "__main__":
    metrics = get_metrics()

    # Simulate some operations
    for i in range(10):
        metrics.record_embedding_time(150 + i * 10, is_batch=False, cache_hit=(i % 3 == 0))
        metrics.record_vector_search(100 + i * 5, num_results=3)
        metrics.record_api_call("chat", 800 + i * 50)

    metrics.record_document_processed(15)

    # Print summary
    import json
    print(json.dumps(metrics.get_summary(), indent=2))
    print("\nHealth Status:")
    print(json.dumps(metrics.get_health_status(), indent=2))
