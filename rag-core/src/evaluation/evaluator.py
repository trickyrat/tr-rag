from .metrics import compute_all_metrics
import json
import logging
import time
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from collections import defaultdict

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates retrieval quality using a test set."""

    def __init__(self, testset_path: str):
        self.testset_path = Path(testset_path)
        self.test_cases = self._load_testset()
        logger.info("Loaded %d test cases from %s", len(self.test_cases), testset_path)

    def _load_testset(self) -> List[Dict[str, Any]]:
        """Load test cases from JSONL file."""
        if not self.testset_path.exists():
            raise FileNotFoundError(f"Testset not found: {self.testset_path}")
        cases = []
        with open(self.testset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                    # Normalize: support both chunk_id-based and content_hash-based test cases
                    if "relevant_content_hashes" in case:
                        # content-hash mode (preferred: survives re-chunking)
                        if not isinstance(case["relevant_content_hashes"], list):
                            case["relevant_content_hashes"] = [case["relevant_content_hashes"]]
                        case["_match_mode"] = "content_hash"
                    elif "relevant_chunk_ids" in case:
                        # legacy chunk-id mode
                        if not isinstance(case["relevant_chunk_ids"], list):
                            case["relevant_chunk_ids"] = [case["relevant_chunk_ids"]]
                        case["_match_mode"] = "chunk_id"
                    else:
                        raise ValueError(
                            f"Missing 'relevant_content_hashes' or 'relevant_chunk_ids' at line {line_num}"
                        )
                    cases.append(case)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        return cases

    def evaluate(
        self,
        retrieve_fn: Callable[[str, int], List[Document]],
        top_k: int = 5,
        k_list: List[int] = [1, 3, 5],
        per_query_results: bool = False,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a retrieval function against the test set.

        Args:
            retrieve_fn: ``(query, top_k) → List[Document]``.
            top_k: Number of documents to retrieve per query.
            k_list: K values for Hit/Precision/Recall/NDCG.
            per_query_results: Include per-query details in output.
            run_metadata: Arbitrary dict attached to result (model,
                chunker, retrieval strategy, etc.).

        Returns:
            ``{"aggregate": {…}, "total_queries": int, "elapsed_seconds": float,
               "per_query": […] (if requested), "run_metadata": {…}}``
        """
        t0 = time.perf_counter()
        per_query: List[Dict[str, Any]] = []
        agg_metrics: Dict[str, List[float]] = defaultdict(list)

        for case in self.test_cases:
            query = case["query"]
            match_mode = case.get("_match_mode", "chunk_id")

            if match_mode == "content_hash":
                relevant_ids = set(case["relevant_content_hashes"])
            else:
                relevant_ids = set(case["relevant_chunk_ids"])

            try:
                retrieved_docs = retrieve_fn(query, top_k=top_k)
                # Prefer content_hash matching (survives re-chunking);
                # fall back to chunk_id matching for backward compatibility.
                if match_mode == "content_hash":
                    retrieved_ids = []
                    for doc in retrieved_docs:
                        # Support child_content_hashes from Small-to-Big expansion
                        child_hashes = doc.metadata.get("child_content_hashes")
                        if child_hashes and isinstance(child_hashes, list):
                            retrieved_ids.extend(child_hashes)
                        else:
                            cid = doc.metadata.get("content_hash") or doc.metadata.get("chunk_id")
                            if cid:
                                retrieved_ids.append(cid)
                else:
                    retrieved_ids = [
                        doc.metadata.get("chunk_id")
                        for doc in retrieved_docs
                        if doc.metadata.get("chunk_id")
                    ]
                if not all(retrieved_ids):
                    logger.warning(
                        f"Retrieved doc missing match key for query: {query[:50]}..."
                    )
                    retrieved_ids = [
                        str(hash(doc.page_content)) for doc in retrieved_docs
                    ]
            except Exception as e:
                logger.error(f"Retrieval failed for query '{query[:50]}...': {e}")
                retrieved_ids = []

            metrics = compute_all_metrics(retrieved_ids, relevant_ids, k_list)
            per_query.append(
                {
                    "query": query,
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": list(relevant_ids),
                    "metrics": metrics,
                }
            )

            for key, value in metrics.items():
                agg_metrics[key].append(value)

        # Aggregate averages
        aggregate = {key: sum(vals) / len(vals) for key, vals in agg_metrics.items()}
        aggregate["total_queries"] = len(self.test_cases)

        result: Dict[str, Any] = {
            "aggregate": aggregate,
            "total_queries": len(self.test_cases),
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }
        if run_metadata:
            result["run_metadata"] = run_metadata
        if per_query_results:
            result["per_query"] = per_query

        logger.info(
            "Evaluation complete: Hit@1=%.4f MRR=%.4f (%.1fs)",
            aggregate.get("hit@1", 0),
            aggregate.get("mrr", 0),
            result["elapsed_seconds"],
        )
        return result

    def evaluate_on_subset(
        self,
        retrieve_fn: Callable[[str, int], List[Document]],
        top_k: int = 5,
        max_queries: int = 50,
    ) -> Dict[str, Any]:
        """Run evaluation on a subset (first N queries) for quick iteration."""
        original_cases = self.test_cases
        try:
            self.test_cases = self.test_cases[:max_queries]
            result = self.evaluate(retrieve_fn, top_k, per_query_results=False)
        finally:
            self.test_cases = original_cases
        return result
