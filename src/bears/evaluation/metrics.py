"""
Evaluation metrics calculation.

Direct copy from archive/GraphRag_hybrid_1/app/services/evaluation/metrics.py.
"""

import logging
from typing import List, Set, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


def calculate_reciprocal_rank(gold_ids: Set[str], retrieved_ids: List[str]) -> float:
    """Calculate Reciprocal Rank (standard MRR definition).

    Returns RR of the first hit among gold docs.
    """
    if not gold_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank

    return 0.0


def calculate_average_precision(gold_ids: Set[str], retrieved_ids: List[str]) -> float:
    """Calculate Average Precision (AP).

    AP = (1 / |Gold|) * sum(P@k * rel(k))
    """
    if not gold_ids:
        return 0.0

    relevant_count = 0
    precision_sum = 0.0

    for k, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            relevant_count += 1
            precision_at_k = relevant_count / k
            precision_sum += precision_at_k

    return precision_sum / len(gold_ids)


def calculate_retrieval_metrics(
    retrieved_ids: List[str],
    gold_ids: Set[str]
) -> Dict[str, Any]:
    """Calculate retrieval metrics.

    Returns dict with: hit, found_count, avg_rr, ap
    """
    found_count = sum(1 for doc_id in retrieved_ids if doc_id in gold_ids)
    hit = 1 if found_count > 0 else 0
    avg_rr = calculate_reciprocal_rank(gold_ids, retrieved_ids)
    ap = calculate_average_precision(gold_ids, retrieved_ids)

    return {
        "hit": hit,
        "found_count": found_count,
        "avg_rr": avg_rr,
        "ap": ap
    }


def compute_final_metrics(
    stats_by_source: Dict[str, Dict],
    stats_by_type: Dict[str, Dict],
    total_queries: int
) -> Dict[str, Any]:
    """Compute final aggregated metrics."""
    overall_stats = {
        "total": 0,
        "hit_count": 0,
        "found_sum": 0,
        "gold_sum": 0,
        "rr_sum": 0.0,
        "ap_sum": 0.0,
        "generation_pass": 0,
        "retrieval_time_sum": 0.0,
        "generation_time_sum": 0.0,
        "total_time_sum": 0.0
    }

    for source_stats in stats_by_source.values():
        overall_stats["total"] += source_stats["total"]
        overall_stats["hit_count"] += source_stats["hit_count"]
        overall_stats["found_sum"] += source_stats["found_sum"]
        overall_stats["gold_sum"] += source_stats["gold_sum"]
        overall_stats["rr_sum"] += source_stats["rr_sum"]
        overall_stats["ap_sum"] += source_stats.get("ap_sum", 0.0)
        overall_stats["generation_pass"] += source_stats["generation_pass"]
        overall_stats["retrieval_time_sum"] += source_stats.get("retrieval_time_sum", 0.0)
        overall_stats["generation_time_sum"] += source_stats.get("generation_time_sum", 0.0)
        overall_stats["total_time_sum"] += source_stats.get("total_time_sum", 0.0)

    overall_metrics = _compute_metrics(overall_stats)

    by_source = {}
    for source in ["drcd", "hotpotqa", "2wiki"]:
        if source in stats_by_source and stats_by_source[source]["total"] > 0:
            by_source[source] = _compute_metrics(stats_by_source[source])

    by_question_type = {}
    for qtype in ["single-hop", "multi-hop"]:
        if qtype in stats_by_type and stats_by_type[qtype]["total"] > 0:
            by_question_type[qtype] = _compute_metrics(stats_by_type[qtype])

    return {
        "overall": overall_metrics,
        "by_source": by_source,
        "by_question_type": by_question_type
    }


def _compute_metrics(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics for a single group."""
    total = stats["total"]

    if total == 0:
        return {
            "total_questions": 0,
            "hit_rate": 0.0,
            "partial_hit_rate": 0.0,
            "mrr": 0.0,
            "map": 0.0,
            "generation_pass_rate": 0.0,
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_total_time": 0.0
        }

    hit_rate = stats["hit_count"] / total
    partial_hit_rate = stats["found_sum"] / stats["gold_sum"] if stats["gold_sum"] > 0 else 0
    mrr = stats["rr_sum"] / total
    map_score = stats.get("ap_sum", 0.0) / total
    generation_pass_rate = stats["generation_pass"] / total

    avg_retrieval_time = stats.get("retrieval_time_sum", 0.0) / total
    avg_generation_time = stats.get("generation_time_sum", 0.0) / total
    avg_total_time = stats.get("total_time_sum", 0.0) / total

    return {
        "total_questions": total,
        "hit_rate": hit_rate,
        "partial_hit_rate": partial_hit_rate,
        "mrr": mrr,
        "map": map_score,
        "generation_pass_rate": generation_pass_rate,
        "avg_retrieval_time": avg_retrieval_time,
        "avg_generation_time": avg_generation_time,
        "avg_total_time": avg_total_time
    }
