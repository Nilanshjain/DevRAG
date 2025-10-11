"""
Metrics API Endpoints
Provides performance monitoring and system health endpoints
"""

from fastapi import APIRouter
from typing import Dict, Any

from app.core.metrics import get_metrics

# Create router
router = APIRouter(
    prefix="/api/v1/metrics",
    tags=["metrics"]
)


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get comprehensive performance metrics

    Returns detailed statistics on:
    - Embedding generation performance
    - Vector search latency
    - API response times
    - Cache hit rates
    - System uptime
    """
    metrics = get_metrics()
    return metrics.get_summary()


@router.get("/health")
async def get_health_status() -> Dict[str, Any]:
    """
    Get system health status

    Returns:
    - Overall health status (healthy/degraded/unhealthy)
    - Key performance indicators
    - Active warnings
    - System uptime
    """
    metrics = get_metrics()
    return metrics.get_health_status()


@router.get("/cache")
async def get_cache_metrics() -> Dict[str, Any]:
    """
    Get cache performance metrics

    Returns:
    - Cache hit rate
    - Cache hits vs misses
    - Cache efficiency
    """
    metrics = get_metrics()

    return {
        "cache_hit_rate": round(metrics.cache_hit_rate, 3),
        "cache_hits": metrics.embedding_cache_hits,
        "cache_misses": metrics.embedding_cache_misses,
        "total_requests": metrics.embedding_cache_hits + metrics.embedding_cache_misses,
        "efficiency": f"{metrics.cache_hit_rate * 100:.1f}%"
    }


@router.get("/throughput")
async def get_throughput_metrics() -> Dict[str, Any]:
    """
    Get system throughput metrics

    Returns:
    - Documents processed
    - Chunks created
    - Queries answered
    - Rates (per hour)
    """
    metrics = get_metrics()
    uptime_hours = max(metrics.uptime_seconds / 3600, 0.01)  # Avoid division by zero

    return {
        "totals": {
            "documents_processed": metrics.documents_processed,
            "chunks_created": metrics.chunks_created,
            "queries_answered": metrics.queries_answered
        },
        "rates_per_hour": {
            "documents": round(metrics.documents_processed / uptime_hours, 2),
            "chunks": round(metrics.chunks_created / uptime_hours, 2),
            "queries": round(metrics.queries_answered / uptime_hours, 2)
        },
        "uptime_hours": round(uptime_hours, 2)
    }


@router.post("/reset")
async def reset_metrics() -> Dict[str, str]:
    """
    Reset all metrics

    WARNING: This clears all collected metrics data
    Use only for testing or when restarting monitoring
    """
    metrics = get_metrics()
    metrics.reset_metrics()

    return {
        "status": "success",
        "message": "All metrics have been reset"
    }


@router.get("/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a simplified metrics summary for dashboards

    Returns key metrics in an easy-to-display format
    """
    metrics = get_metrics()
    summary = metrics.get_summary()

    return {
        "system_health": metrics.get_health_status()["status"],
        "uptime_hours": round(metrics.uptime_seconds / 3600, 1),
        "key_metrics": {
            "queries_answered": metrics.queries_answered,
            "avg_response_time_ms": round(summary["api_performance"]["chat_response_ms"]["average"], 0),
            "cache_hit_rate": f"{metrics.cache_hit_rate * 100:.1f}%",
            "avg_search_latency_ms": round(summary["vector_search_performance"]["search_latency_ms"]["average"], 0)
        },
        "performance_percentiles": {
            "chat_p95_ms": summary["api_performance"]["chat_response_ms"]["p95"],
            "chat_p99_ms": summary["api_performance"]["chat_response_ms"]["p99"],
            "search_p95_ms": summary["vector_search_performance"]["search_latency_ms"]["p95"]
        }
    }
