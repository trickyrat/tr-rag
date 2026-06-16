from .metrics import compute_all_metrics
import json
import logging
from typing import List, Dict, Any, Callable
from pathlib import Path
from collections import defaultdict

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates retrieval quality using a test set.
    """

    def __init__(self, testset_path: str):
        """
        Args:
            testset_path: Path to JSONL file with test cases.
                          Each line: {"query": str, "relevant_chunk_ids": list[str]}
        """
        self.testset_path = Path(testset_path)
        self.test_cases = self._load_testset()
        logger.info(f"Loaded {len(self.test_cases)} test cases from {testset_path}")

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
    ) -> Dict[str, Any]:
        """
        Evaluate a retrieval function against the test set.

        Args:
            retrieve_fn: Function that takes (query, top_k) and returns List[Document].
                         Must preserve chunk_id in metadata.
            top_k: Number of documents to retrieve per query.
            k_list: List of K values for metrics.
            per_query_results: If True, include detailed per-query metrics.

        Returns:
            Dictionary containing:
                - aggregate: dict of averaged metrics
                - per_query (optional): list of per-query results
                - total_queries: int
        """
        per_query = []
        agg_metrics = defaultdict(list)

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
                    retrieved_ids = [
                        doc.metadata.get("content_hash")
                        or doc.metadata.get("chunk_id")
                        for doc in retrieved_docs
                    ]
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

        result = {
            "aggregate": aggregate,
            "total_queries": len(self.test_cases),
        }
        if per_query_results:
            result["per_query"] = per_query

        logger.info(
            f"Evaluation complete: Hit@1={aggregate.get('hit@1', 0):.4f}, MRR={aggregate.get('mrr', 0):.4f}"
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
