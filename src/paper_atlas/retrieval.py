import logging
import re
from collections import defaultdict
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from paper_atlas.config import COLLECTION_NAME, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class Retrieval:
    """Retrieve research paper chunks using semantic search and BM25.

    Chroma provides semantic results. An in-memory BM25 index provides keyword
    results. Reciprocal rank fusion (RRF) combines the two rankings.

    Call ``refresh_index()`` after ingestion adds or deletes chunks.
    """

    def __init__(self, db_path: Path) -> None:
        """Connect to Chroma and build the initial BM25 index.

        Args:
            db_path: Directory containing the persistent Chroma database.
                This must be the same directory used by ingestion.

        Raises:
            Exception: If Chroma, the embedding model, or index initialization
                fails. The original exception is logged and re-raised.
        """
        try:
            self.chroma_client = chromadb.PersistentClient(path=str(db_path))
            embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL
                )
            )
            self.collection = self.chroma_client.get_collection(
                COLLECTION_NAME,
                embedding_function=embedding_function,
            )

            self.doc_ids: list[str] = []
            self.records_by_id: dict[str, dict] = {}
            self.bm25: BM25Okapi | None = None

            self.refresh_index()
        except Exception:
            logger.exception("Failed to initialize retrieval from %s", db_path)
            raise

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase word tokens for BM25.

        Args:
            text: Document text or search query.

        Returns:
            Lowercase tokens in their original order.
        """
        return re.findall(r"\w+", text.lower())

    def refresh_index(self) -> None:
        """Rebuild the in-memory keyword index from Chroma documents.

        Call this after ingestion adds, updates, or deletes chunks. The existing
        index remains available if rebuilding the new index fails.

        Raises:
            ValueError: If Chroma returns mismatched IDs, documents, and
                metadata lists.
            Exception: If reading Chroma or building BM25 fails. The original
                exception is logged and re-raised.
        """
        try:
            records = self.collection.get(include=["documents", "metadatas"])
            ids = records["ids"]
            documents = records["documents"] or []
            metadatas = records["metadatas"] or []

            if not (len(ids) == len(documents) == len(metadatas)):
                raise ValueError("Chroma returned mismatched record fields")

            new_records = {
                doc_id: {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata or {},
                }
                for doc_id, document, metadata in zip(
                    ids, documents, metadatas
                )
            }
            tokenized = [self._tokenize(doc or "") for doc in documents]
            new_bm25 = BM25Okapi(tokenized) if ids and any(tokenized) else None

            self.doc_ids = ids
            self.records_by_id = new_records
            self.bm25 = new_bm25

            logger.info("BM25 index refreshed with %d chunks", len(ids))
        except Exception:
            logger.exception("Failed to refresh BM25 index")
            raise

    def _semantic_search(self, query: str, limit: int) -> list[str]:
        """Find chunks whose embeddings are similar to the query.

        Args:
            query: Natural language search query.
            limit: Maximum number of candidate chunks to retrieve.

        Returns:
            Chunk IDs ordered from most to least semantically similar.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
        )
        return results["ids"][0] if results["ids"] else []

    def _keyword_search(self, query: str, limit: int) -> list[str]:
        """Find chunks containing query terms, ranked by BM25.

        Args:
            query: Search query to tokenize and match against chunk text.
            limit: Maximum number of candidate chunks to retrieve.

        Returns:
            Chunk IDs ordered by BM25 score. Returns an empty list when there
            is no BM25 index or the query has no searchable tokens.
        """
        if self.bm25 is None:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            range(len(self.doc_ids)),
            key=lambda i: scores[i],
            reverse=True,
        )

        return [
            self.doc_ids[i]
            for i in ranked
            if any(token in self.bm25.doc_freqs[i] for token in tokens)
        ][:limit]

    @staticmethod
    def _rrf(
        result_lists: list[list[str]],
        *,
        k: int = 60,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """Merge ranked chunk ID lists using reciprocal rank fusion.

        A chunk appearing near the top of either list receives a higher score.
        A chunk appearing in both lists receives contributions from both.

        Args:
            result_lists: Ranked chunk IDs from each search method.
            k: Smoothing constant that controls the effect of rank position.
            top_n: Maximum number of combined results to return.

        Returns:
            ``(chunk_id, rrf_score)`` pairs, ordered by descending RRF score.
        """
        scores: dict[str, float] = defaultdict(float)

        for results in result_lists:
            for rank, doc_id in enumerate(results, start=1):
                scores[doc_id] += 1 / (k + rank)

        return sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]

    def retrieve(self, query: str, n_results: int = 10) -> list[dict]:
        """Search for relevant paper chunks using both retrieval methods.

        Fetches more candidates than requested, merges their rankings with
        RRF, and includes each chunk's text and citation metadata.

        Args:
            query: Natural language question or search phrase.
            n_results: Maximum number of final chunks to return.

        Returns:
            Dictionaries containing ``id``, ``text``, ``metadata``, and
            ``rrf_score``, ordered by descending relevance. Returns an empty
            list for a blank query or an empty collection.

        Raises:
            ValueError: If ``n_results`` is not positive.
            RuntimeError: If Chroma returns an ID absent from the local index,
                indicating that ``refresh_index()`` is needed.
            Exception: If either search fails. The original exception is
                logged and re-raised.
        """
        if not query.strip():
            logger.debug("Skipping empty retrieval query")
            return []
        if n_results <= 0:
            raise ValueError("n_results must be positive")
        if not self.doc_ids:
            logger.info("No indexed chunks available for retrieval")
            return []

        candidate_count = min(len(self.doc_ids), max(20, n_results * 3))

        try:
            semantic_ids = self._semantic_search(query, candidate_count)
            keyword_ids = self._keyword_search(query, candidate_count)
            combined = self._rrf(
                [semantic_ids, keyword_ids],
                top_n=n_results,
            )

            missing = [
                doc_id
                for doc_id, _ in combined
                if doc_id not in self.records_by_id
            ]
            if missing:
                raise RuntimeError(
                    "Retrieval index is stale; call refresh_index() "
                    "after ingestion"
                )

            logger.info(
                "Retrieved %d chunks (%d semantic, %d keyword candidates)",
                len(combined),
                len(semantic_ids),
                len(keyword_ids),
            )

            return [
                {**self.records_by_id[doc_id], "rrf_score": score}
                for doc_id, score in combined
            ]
        except Exception:
            logger.exception("Retrieval failed for query %r", query)
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retrieval = Retrieval(db_path=Path("chroma_db"))
    print(retrieval.retrieve("model"))