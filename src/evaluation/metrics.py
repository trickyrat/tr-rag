import logging
import math
from typing import List, Set, Dict

logger = logging.getLogger(__name__)


def hit_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Hit@K: whether at least one relevant document appears in top K.

    Args:
        retrieved_ids: ordered list of retrieved chunk IDs
        relevant_ids: set of relevant chunk IDs
        k: top K to consider

    Returns:
        1.0 if hit, else 0.0
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    top_k_ids = retrieved_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top_k_ids) else 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Precision@K: proportion of relevant documents in top K.

    Args:
        retrieved_ids: ordered list of retrieved chunk IDs
        relevant_ids: set of relevant chunk IDs
        k: top K to consider

    Returns:
        precision value between 0 and 1
    """
    if not retrieved_ids:
        return 0.0
    top_k_ids = retrieved_ids[:k]
    if not top_k_ids:
        return 0.0
    hits = sum(1 for rid in top_k_ids if rid in relevant_ids)
    return hits / len(top_k_ids)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Recall@K: proportion of relevant documents retrieved in top K.

    Args:
        retrieved_ids: ordered list of retrieved chunk IDs
        relevant_ids: set of all relevant chunk IDs
        k: top K to consider

    Returns:
        recall value between 0 and 1
    """
    if not relevant_ids:
        return 0.0
    top_k_ids = retrieved_ids[:k]
    hits = sum(1 for rid in top_k_ids if rid in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Mean Reciprocal Rank (MRR): reciprocal of the rank of the first relevant document.

    Args:
        retrieved_ids: ordered list of retrieved chunk IDs
        relevant_ids: set of relevant chunk IDs

    Returns:
        MRR value (1/rank) or 0 if none found
    """
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K (binary relevance).

    Args:
        retrieved_ids: ordered list of retrieved chunk IDs
        relevant_ids: set of relevant chunk IDs
        k: top K to consider

    Returns:
        NDCG@K value between 0 and 1
    """
    top_k_ids = retrieved_ids[:k]
    # DCG: sum_{i=1}^{k} (rel_i / log2(i+1))
    dcg = 0.0
    for i, rid in enumerate(top_k_ids, start=1):
        rel = 1.0 if rid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 1)

    # Ideal DCG: all relevant at top positions
    ideal_relevant = list(relevant_ids)
    ideal_dcg = 0.0
    for i in range(1, min(k, len(ideal_relevant)) + 1):
        ideal_dcg += 1.0 / math.log2(i + 1)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def compute_all_metrics(
    retrieved_ids: List[str], relevant_ids: Set[str], k_list: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single query.

    Args:
        retrieved_ids: ordered list of retrieved chunk IDs
        relevant_ids: set of relevant chunk IDs
        k_list: list of K values for Hit@K, Precision@K, Recall@K

    Returns:
        Dictionary containing metrics
    """
    metrics = {}
    metrics["mrr"] = mean_reciprocal_rank(retrieved_ids, relevant_ids)
    for k in k_list:
        metrics[f"hit@{k}"] = hit_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    return metrics
