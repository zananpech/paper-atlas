from dataclasses import dataclass

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


@dataclass
class MetricsMetadata:
    query: str
    category: str
    retriever_type: str
    reranker_model: str
    embedding_model: str
    top_k: int