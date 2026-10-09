import math
from collections.abc import Sequence


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Precision@K.

    Precision@K = (relevant retrieved documents in top-K) / K
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not retrieved_ids or not relevant_ids:
        return 0.0

    retrieved = retrieved_ids[:k]
    relevant = set(relevant_ids)

    # Use unique retrieved documents to prevent skewed hits if duplicates exist
    unique_retrieved = set(retrieved)
    hits = len(unique_retrieved & relevant)

    return hits / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    Recall@K = (relevant retrieved documents in top-K) / (total relevant documents)
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant = set(relevant_ids)
    if not relevant or not retrieved_ids:
        return 0.0

    retrieved = retrieved_ids[:k]
    unique_retrieved = set(retrieved)
    hits = len(unique_retrieved & relevant)

    return hits / len(relevant)


def hit_rate_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Hit Rate@K (Success@K).

    Returns 1.0 if at least one relevant document is retrieved within the top-K,
    otherwise returns 0.0.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant = set(relevant_ids)
    if not relevant or not retrieved_ids:
        return 0.0

    retrieved = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant for doc_id in retrieved) else 0.0


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int | None = None,
) -> float:
    """
    Calculate the reciprocal rank for a single query.

    Reciprocal rank is 1 / rank of the first relevant document.
    Returns 0.0 if no relevant document is found in top-K (or in retrieved list if k is None).
    """
    if k is not None and k <= 0:
        raise ValueError("k must be greater than 0")

    relevant = set(relevant_ids)
    if not relevant or not retrieved_ids:
        return 0.0

    search_pool = retrieved_ids[:k] if k is not None else retrieved_ids

    for rank, doc_id in enumerate(search_pool, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant = set(relevant_ids)
    if not relevant or not retrieved_ids:
        return 0.0

    retrieved = retrieved_ids[:k]
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(
    results: Sequence[tuple[Sequence[str], Sequence[str]]],
    k: int | None = None,
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) across multiple queries.

    results:
        Sequence of (retrieved_ids, relevant_ids) tuples.
    """
    if not results:
        return 0.0

    scores = [
        reciprocal_rank(retrieved_ids, relevant_ids, k=k)
        for retrieved_ids, relevant_ids in results
    ]

    return sum(scores) / len(scores)