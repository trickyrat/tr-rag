import logging
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document

# Import your existing classes
from src.rag import RetrievalOptimization  # adjust import as needed

from src.evaluation import RAGEvaluator

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Run multiple retrieval configurations and compare metrics.
    """

    def __init__(
        self,
        vectorstore: Chroma,
        chunks: List[Document],
        testset_path: str,
        results_dir: str = "evaluation_results",
    ):
        """
        Args:
            vectorstore: Chroma vector store instance
            chunks: List of all chunk documents
            testset_path: Path to JSONL test set
            results_dir: Directory to save results CSV and logs
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.testset_path = testset_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = RAGEvaluator(testset_path)

    def run_experiment(
        self,
        experiment_name: str,
        retriever_builder: Callable[[], RetrievalOptimization],
        top_k: int = 5,
        k_list: List[int] = [1, 3, 5],
    ) -> Dict[str, Any]:
        """
        Run a single experiment with a given retriever builder.

        Args:
            experiment_name: Unique name/ID for this experiment
            retriever_builder: Function that returns a configured RetrievalOptimization instance
            top_k: Number of documents to retrieve
            k_list: List of K values for metrics

        Returns:
            Dictionary with aggregate metrics
        """
        logger.info(f"Running experiment: {experiment_name}")
        retriever = retriever_builder()

        def retrieve_fn(query: str, k: int = top_k) -> List[Document]:
            return retriever.hybrid_search(query, top_k=k)

        results = self.evaluator.evaluate(
            retrieve_fn, top_k=top_k, k_list=k_list, per_query_results=False
        )
        aggregate = results["aggregate"]
        aggregate["experiment"] = experiment_name
        aggregate["timestamp"] = datetime.now().isoformat()

        # Save to CSV
        self._save_result(aggregate)
        return aggregate

    def _save_result(self, metrics: Dict[str, Any]) -> None:
        """Append metrics to CSV file."""
        csv_path = self.results_dir / "experiments.csv"
        file_exists = csv_path.exists()

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        logger.info(f"Results appended to {csv_path}")

    def run_grid_search(
        self, param_grid: List[Dict[str, Any]], base_retriever_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Run grid search over parameter combinations.

        Args:
            param_grid: List of dicts each specifying parameters to override.
                        e.g., [{"rrf_constant": 60, "vector_k": 5}, {"rrf_constant": 30, "vector_k": 10}]
            base_retriever_config: Base parameters for RetrievalOptimization (like vectorstore, chunks)

        Returns:
            List of result dictionaries for each configuration.
        """
        results = []
        for params in param_grid:
            # Create a new retrieval instance with modified parameters
            def builder(p=params):
                # Note: RetrievalOptimization expects vectorstore and chunks in __init__
                # Then we set attributes on its .hybrid_retriever
                ro = RetrievalOptimization(self.vectorstore, self.chunks)
                for key, value in p.items():
                    if hasattr(ro.hybrid_retriever, key):
                        setattr(ro.hybrid_retriever, key, value)
                    elif hasattr(ro, key):
                        setattr(ro, key, value)
                return ro

            exp_name = "_".join([f"{k}={v}" for k, v in params.items()])
            result = self.run_experiment(exp_name, builder)
            results.append(result)
        return results

    @staticmethod
    def load_results(results_dir: str) -> List[Dict[str, Any]]:
        """Load all experiment results from CSV."""
        csv_path = Path(results_dir) / "experiments.csv"
        if not csv_path.exists():
            return []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)


# ----- Example usage (commented) -----
# if __name__ == "__main__":
#     import logging
#     logging.basicConfig(level=logging.INFO)
#
#     # Assume you have built vectorstore and chunks already
#     # vectorstore = ... from VectorStoreBuilder
#     # chunks = ... from DocumentChunker.chunks
#
#     runner = ExperimentRunner(vectorstore, chunks, "testset.jsonl")
#
#     # Baseline: default params
#     def baseline():
#         return RetrievalOptimization(vectorstore, chunks)
#
#     baseline_metrics = runner.run_experiment("baseline", baseline)
#
#     # Grid search
#     param_grid = [
#         {"rrf_constant": 30, "vector_k": 5, "bm25_k": 5},
#         {"rrf_constant": 60, "vector_k": 8, "bm25_k": 8},
#         {"rrf_constant": 90, "vector_k": 3, "bm25_k": 3},
#     ]
#     grid_results = runner.run_grid_search(param_grid, {})
#
#     # Print best config
#     best = max(grid_results, key=lambda x: float(x.get("hit@1", 0)))
#     print(f"Best hit@1: {best['hit@1']} with params: {best['experiment']}")
