"""
Evaluation Metrics for RAG System

This module provides metrics to evaluate the quality of the RAG system:
- Answer Relevance: How well the answer addresses the question
- Context Precision: How relevant the retrieved context is
- Context Recall: How much of the needed information was retrieved
- Answer Accuracy: Keyword matching and semantic similarity
- Response Time: Performance metrics
"""

import time
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics"""
    question_id: int
    question: str
    answer: str

    # Quality Metrics
    keyword_coverage: float  # 0-1: % of expected keywords found
    answer_length_score: float  # 0-1: Is answer length reasonable?
    context_found: bool  # Was relevant context retrieved?

    # Performance Metrics
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float

    # Context Metrics
    num_context_chunks: int
    avg_context_similarity: float

    # Overall Score
    overall_score: float  # Weighted combination of all metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answer_preview": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "metrics": {
                "quality": {
                    "keyword_coverage": round(self.keyword_coverage, 3),
                    "answer_length_score": round(self.answer_length_score, 3),
                    "context_found": self.context_found
                },
                "performance": {
                    "retrieval_time_ms": round(self.retrieval_time_ms, 2),
                    "generation_time_ms": round(self.generation_time_ms, 2),
                    "total_time_ms": round(self.total_time_ms, 2)
                },
                "context": {
                    "num_chunks_retrieved": self.num_context_chunks,
                    "avg_similarity": round(self.avg_context_similarity, 3)
                }
            },
            "overall_score": round(self.overall_score, 3)
        }


class RAGEvaluator:
    """
    Evaluates RAG system performance on various metrics
    """

    def __init__(self):
        self.results: List[EvaluationResult] = []

    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        expected_keywords: List[str],
        context_chunks: List[Dict[str, Any]]
    ) -> Tuple[float, float, bool]:
        """
        Evaluate the quality of a generated answer

        Returns:
            (keyword_coverage, answer_length_score, context_found)
        """
        # 1. Keyword Coverage
        answer_lower = answer.lower()
        keywords_found = sum(1 for keyword in expected_keywords
                            if keyword.lower() in answer_lower)
        keyword_coverage = keywords_found / len(expected_keywords) if expected_keywords else 0.0

        # 2. Answer Length Score (penalize too short or too long)
        answer_words = len(answer.split())
        if answer_words < 20:
            length_score = answer_words / 20  # Too short
        elif answer_words > 500:
            length_score = 1.0 - ((answer_words - 500) / 1000)  # Too long
            length_score = max(0.5, length_score)
        else:
            length_score = 1.0  # Good length

        # 3. Context Found
        context_found = len(context_chunks) > 0

        return keyword_coverage, length_score, context_found

    def evaluate_context_quality(
        self,
        context_chunks: List[Dict[str, Any]],
        expected_keywords: List[str]
    ) -> Tuple[int, float]:
        """
        Evaluate the quality of retrieved context

        Returns:
            (num_chunks, avg_similarity)
        """
        num_chunks = len(context_chunks)

        if num_chunks == 0:
            return 0, 0.0

        # Average similarity from retrieval
        avg_similarity = sum(chunk.get("similarity", 0.0)
                           for chunk in context_chunks) / num_chunks

        return num_chunks, avg_similarity

    def evaluate_single_question(
        self,
        question_id: int,
        question: str,
        answer: str,
        expected_keywords: List[str],
        context_chunks: List[Dict[str, Any]],
        retrieval_time_ms: float,
        generation_time_ms: float
    ) -> EvaluationResult:
        """
        Evaluate a single question-answer pair
        """
        # Quality metrics
        keyword_coverage, length_score, context_found = self.evaluate_answer_quality(
            question, answer, expected_keywords, context_chunks
        )

        # Context metrics
        num_context, avg_similarity = self.evaluate_context_quality(
            context_chunks, expected_keywords
        )

        # Total time
        total_time_ms = retrieval_time_ms + generation_time_ms

        # Overall score (weighted combination)
        # Quality: 60%, Performance: 20%, Context: 20%
        quality_score = (keyword_coverage * 0.7 + length_score * 0.3)
        # AI-powered RAG systems: 5 seconds is good performance (not 1 second)
        performance_score = min(1.0, 5000 / total_time_ms) if total_time_ms > 0 else 0.0
        context_score = (min(1.0, num_context / 3) * 0.5 +
                        avg_similarity * 0.5) if context_found else 0.0

        overall_score = (
            quality_score * 0.6 +
            performance_score * 0.2 +
            context_score * 0.2
        )

        result = EvaluationResult(
            question_id=question_id,
            question=question,
            answer=answer,
            keyword_coverage=keyword_coverage,
            answer_length_score=length_score,
            context_found=context_found,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            num_context_chunks=num_context,
            avg_context_similarity=avg_similarity,
            overall_score=overall_score
        )

        self.results.append(result)
        return result

    def compute_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics across all evaluated questions
        """
        if not self.results:
            return {"error": "No evaluation results available"}

        n = len(self.results)

        # Aggregate quality metrics
        avg_keyword_coverage = sum(r.keyword_coverage for r in self.results) / n
        avg_length_score = sum(r.answer_length_score for r in self.results) / n
        context_found_rate = sum(1 for r in self.results if r.context_found) / n

        # Aggregate performance metrics
        avg_retrieval_time = sum(r.retrieval_time_ms for r in self.results) / n
        avg_generation_time = sum(r.generation_time_ms for r in self.results) / n
        avg_total_time = sum(r.total_time_ms for r in self.results) / n

        # Percentiles for response time
        sorted_times = sorted(r.total_time_ms for r in self.results)
        p50_time = sorted_times[int(n * 0.5)]
        p95_time = sorted_times[int(n * 0.95)] if n > 20 else sorted_times[-1]
        p99_time = sorted_times[int(n * 0.99)] if n > 100 else sorted_times[-1]

        # Aggregate context metrics
        avg_context_chunks = sum(r.num_context_chunks for r in self.results) / n
        avg_context_similarity = sum(r.avg_context_similarity for r in self.results) / n

        # Overall system score
        avg_overall_score = sum(r.overall_score for r in self.results) / n

        # Pass rate (score >= 0.7)
        pass_rate = sum(1 for r in self.results if r.overall_score >= 0.7) / n

        # Distribution by difficulty (if available)
        score_by_category = {}

        return {
            "summary": {
                "total_questions": n,
                "overall_score": round(avg_overall_score, 3),
                "pass_rate": round(pass_rate, 3),
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "quality_metrics": {
                "keyword_coverage": {
                    "average": round(avg_keyword_coverage, 3),
                    "description": "Average % of expected keywords found in answers"
                },
                "answer_quality": {
                    "average_length_score": round(avg_length_score, 3),
                    "description": "Answer length appropriateness (0-1)"
                },
                "context_retrieval": {
                    "success_rate": round(context_found_rate, 3),
                    "description": "% of questions with relevant context retrieved"
                }
            },
            "performance_metrics": {
                "retrieval": {
                    "average_ms": round(avg_retrieval_time, 2),
                    "description": "Average time to retrieve context"
                },
                "generation": {
                    "average_ms": round(avg_generation_time, 2),
                    "description": "Average time to generate answer"
                },
                "total": {
                    "average_ms": round(avg_total_time, 2),
                    "p50_ms": round(p50_time, 2),
                    "p95_ms": round(p95_time, 2),
                    "p99_ms": round(p99_time, 2),
                    "description": "End-to-end response time"
                }
            },
            "context_metrics": {
                "chunks_retrieved": {
                    "average": round(avg_context_chunks, 2),
                    "description": "Average number of context chunks retrieved"
                },
                "similarity_score": {
                    "average": round(avg_context_similarity, 3),
                    "description": "Average similarity between query and retrieved chunks"
                }
            }
        }

    def generate_report(self, include_details: bool = False) -> str:
        """
        Generate a human-readable report
        """
        metrics = self.compute_aggregate_metrics()

        if "error" in metrics:
            return "No evaluation results to report."

        report = []
        report.append("=" * 70)
        report.append("RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 70)
        report.append("")

        # Summary
        summary = metrics["summary"]
        report.append(f"Evaluation Date: {summary['evaluation_date']}")
        report.append(f"Total Questions: {summary['total_questions']}")
        report.append(f"Overall Score: {summary['overall_score']:.1%}")
        report.append(f"Pass Rate (≥70%): {summary['pass_rate']:.1%}")
        report.append("")

        # Quality Metrics
        report.append("QUALITY METRICS")
        report.append("-" * 70)
        quality = metrics["quality_metrics"]
        report.append(f"  Keyword Coverage:     {quality['keyword_coverage']['average']:.1%}")
        report.append(f"  Answer Quality Score: {quality['answer_quality']['average_length_score']:.1%}")
        report.append(f"  Context Found Rate:   {quality['context_retrieval']['success_rate']:.1%}")
        report.append("")

        # Performance Metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 70)
        perf = metrics["performance_metrics"]
        report.append(f"  Average Retrieval Time:   {perf['retrieval']['average_ms']:.0f}ms")
        report.append(f"  Average Generation Time:  {perf['generation']['average_ms']:.0f}ms")
        report.append(f"  Average Total Time:       {perf['total']['average_ms']:.0f}ms")
        report.append(f"  P95 Response Time:        {perf['total']['p95_ms']:.0f}ms")
        report.append("")

        # Context Metrics
        report.append("CONTEXT RETRIEVAL METRICS")
        report.append("-" * 70)
        context = metrics["context_metrics"]
        report.append(f"  Avg Chunks Retrieved:  {context['chunks_retrieved']['average']:.1f}")
        report.append(f"  Avg Similarity Score:  {context['similarity_score']['average']:.3f}")
        report.append("")

        # Detailed Results (optional)
        if include_details:
            report.append("DETAILED RESULTS")
            report.append("-" * 70)
            for result in self.results[:10]:  # Show first 10
                report.append(f"Q{result.question_id}: {result.question[:60]}...")
                report.append(f"  Score: {result.overall_score:.1%} | "
                            f"Keywords: {result.keyword_coverage:.1%} | "
                            f"Time: {result.total_time_ms:.0f}ms")
                report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def save_results(self, filepath: str):
        """Save detailed results to JSON file"""
        import json

        results_dict = {
            "aggregate_metrics": self.compute_aggregate_metrics(),
            "individual_results": [r.to_dict() for r in self.results]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {filepath}")


def calculate_keyword_overlap(text: str, keywords: List[str]) -> float:
    """
    Calculate what percentage of keywords appear in the text
    """
    text_lower = text.lower()
    found = sum(1 for keyword in keywords if keyword.lower() in text_lower)
    return found / len(keywords) if keywords else 0.0


def calculate_answer_relevance_simple(
    question: str,
    answer: str,
    min_length: int = 20,
    max_length: int = 500
) -> float:
    """
    Simple heuristic for answer relevance based on length and content
    Returns score between 0 and 1
    """
    words = answer.split()
    word_count = len(words)

    # Length penalty
    if word_count < min_length:
        length_score = word_count / min_length
    elif word_count > max_length:
        length_score = max(0.5, 1.0 - (word_count - max_length) / max_length)
    else:
        length_score = 1.0

    # Check for common error patterns
    error_phrases = [
        "i don't know",
        "i cannot answer",
        "no information",
        "error occurred",
        "failed to"
    ]

    answer_lower = answer.lower()
    has_error = any(phrase in answer_lower for phrase in error_phrases)

    if has_error:
        return length_score * 0.3  # Severely penalize error responses

    return length_score


# Example usage for testing
if __name__ == "__main__":
    # Test the evaluator
    evaluator = RAGEvaluator()

    # Simulate some evaluation results
    test_result = evaluator.evaluate_single_question(
        question_id=1,
        question="How do you define a function in Python?",
        answer="In Python, you define a function using the 'def' keyword followed by the function name and parameters. The function body is indented and can return a value using the 'return' statement.",
        expected_keywords=["def", "keyword", "function", "return"],
        context_chunks=[
            {"similarity": 0.85, "text": "Functions are defined with def..."},
            {"similarity": 0.72, "text": "The def keyword is used..."}
        ],
        retrieval_time_ms=150.0,
        generation_time_ms=800.0
    )

    print("Single Question Evaluation:")
    print(f"  Overall Score: {test_result.overall_score:.1%}")
    print(f"  Keyword Coverage: {test_result.keyword_coverage:.1%}")
    print(f"  Total Time: {test_result.total_time_ms:.0f}ms")
    print()

    # Add a few more dummy results
    for i in range(2, 6):
        evaluator.evaluate_single_question(
            question_id=i,
            question=f"Test question {i}",
            answer="This is a test answer with some reasonable content explaining the concept in detail.",
            expected_keywords=["test", "answer", "concept"],
            context_chunks=[{"similarity": 0.75 + i*0.02, "text": "context"}],
            retrieval_time_ms=100.0 + i*10,
            generation_time_ms=700.0 + i*50
        )

    # Generate aggregate report
    print(evaluator.generate_report(include_details=True))
