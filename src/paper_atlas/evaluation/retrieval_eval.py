from dataclasses import asdict, dataclass
from pathlib import Path

from paper_atlas.config import DB_PATH
from paper_atlas.evaluation.dataset import EvalDataset
from paper_atlas.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from paper_atlas.retriever import Retriever


@dataclass
class QueryEvalResult:
    query_id: str | None
    query: str
    precision: float
    recall: float
    hit_rate: float
    mrr: float
    ndcg: float
    retrieved_ids: list[str]
    relevant_ids: list[str]
    category: str | None = None



@dataclass
class EvaluationSummary:
    k: int
    num_samples: int
    mean_precision: float
    mean_recall: float
    mean_hit_rate: float
    mean_mrr: float
    mean_ndcg: float
    results: list[QueryEvalResult]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_retriever(
    k: int = 5,
    dataset_path: Path | str | None = None,
    db_path: Path | None = None,
    verbose: bool = True,
) -> EvaluationSummary:
    """Evaluate retrieval performance using an offline evaluation dataset.

    Args:
        k: The cutoff depth for retrieval and metric calculation.
        dataset_path: Path to the JSON evaluation dataset. Defaults to
            evaluation_dataset.json in the same directory.
        db_path: Path to ChromaDB store. Defaults to DB_PATH in config.
        verbose: Whether to print per-query and aggregate metric details.

    Returns:
        An EvaluationSummary instance containing aggregate metrics and query results.
    """
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "evaluation_dataset.json"

    dataset = EvalDataset.load(dataset_path)
    retriever = Retriever(db_path=db_path or DB_PATH)

    results: list[QueryEvalResult] = []

    for sample in dataset:
        query = sample.query
        relevant_ids = sample.relevant_doc_ids

        # Fetch k items to match the evaluation depth k
        retrieved_results = retriever.retrieve(query, n_results=k)
        retrieved_ids = [
            result.get("id") or result.get("metadata", {}).get("chunk_id", "")
            for result in retrieved_results
        ]

        p = precision_at_k(retrieved_ids, relevant_ids, k=k)
        r = recall_at_k(retrieved_ids, relevant_ids, k=k)
        hr = hit_rate_at_k(retrieved_ids, relevant_ids, k=k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids, k=k)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k=k)

        eval_res = QueryEvalResult(
            query_id=sample.query_id,
            query=query,
            precision=p,
            recall=r,
            hit_rate=hr,
            mrr=rr,
            ndcg=ndcg,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            category=sample.category,
        )
        results.append(eval_res)

        if verbose:
            qid_str = f"[{sample.query_id}] " if sample.query_id else ""
            cat_str = f"({sample.category}) " if sample.category else ""
            print(f"Query: {qid_str}{cat_str}{query}")
            print(f"  Precision@{k}: {p:.4f} | Recall@{k}: {r:.4f} | HitRate@{k}: {hr:.4f}")
            print(f"  MRR:          {rr:.4f} | NDCG@{k}:   {ndcg:.4f}")
            print("-" * 50)

    num_samples = len(results)
    if num_samples == 0:
        summary = EvaluationSummary(
            k=k,
            num_samples=0,
            mean_precision=0.0,
            mean_recall=0.0,
            mean_hit_rate=0.0,
            mean_mrr=0.0,
            mean_ndcg=0.0,
            results=[],
        )
    else:
        summary = EvaluationSummary(
            k=k,
            num_samples=num_samples,
            mean_precision=sum(res.precision for res in results) / num_samples,
            mean_recall=sum(res.recall for res in results) / num_samples,
            mean_hit_rate=sum(res.hit_rate for res in results) / num_samples,
            mean_mrr=sum(res.mrr for res in results) / num_samples,
            mean_ndcg=sum(res.ndcg for res in results) / num_samples,
            results=results,
        )

    if verbose and num_samples > 0:
        print("=" * 60)
        print(f"EVALUATION SUMMARY (k={k}, total_samples={num_samples})")
        print(f"  Mean Precision@{k}: {summary.mean_precision:.4f}")
        print(f"  Mean Recall@{k}:    {summary.mean_recall:.4f}")
        print(f"  Mean Hit Rate@{k}:  {summary.mean_hit_rate:.4f}")
        print(f"  Mean MRR:          {summary.mean_mrr:.4f}")
        print(f"  Mean NDCG@{k}:     {summary.mean_ndcg:.4f}")

        # Breakdown by category
        categories = sorted({res.category for res in results if res.category})
        if categories:
            print("-" * 60)
            print("BREAKDOWN BY SCENARIO:")
            for cat in categories:
                cat_results = [r for r in results if r.category == cat]
                cat_n = len(cat_results)
                if cat == "not_exist":
                    # For non-existent documents, any retrieved result without thresholding is a candidate false positive
                    print(f"  [{cat.upper()}] (samples={cat_n})")
                    print(f"    Queries evaluated: {cat_n} (negative control queries)")
                else:
                    c_p = sum(r.precision for r in cat_results) / cat_n
                    c_r = sum(r.recall for r in cat_results) / cat_n
                    c_hr = sum(r.hit_rate for r in cat_results) / cat_n
                    c_mrr = sum(r.mrr for r in cat_results) / cat_n
                    print(f"  [{cat.upper()}] (samples={cat_n})")
                    print(f"    Recall@{k}: {c_r:.4f} | HitRate@{k}: {c_hr:.4f} | Precision@{k}: {c_p:.4f} | MRR: {c_mrr:.4f}")
        print("=" * 60)

    return summary


def main() -> None:
    """CLI entry point for retrieval evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics.")
    parser.add_argument("-k", type=int, default=5, help="Retrieval cutoff depth K (default: 5)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation JSON dataset")
    args = parser.parse_args()

    evaluate_retriever(k=args.k, dataset_path=args.dataset)


if __name__ == "__main__":
    main()