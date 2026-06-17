import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag import RetrievalOptimization

from evaluation import EvaluationStore, RAGEvaluator
from evaluation.db import get_async_engine, get_session_factory

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
    ):
        """
        Args:
            vectorstore: Chroma vector store instance
            chunks: List of all chunk documents
            testset_path: Path to JSONL test set
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.testset_path = testset_path
        self.evaluator = RAGEvaluator(testset_path)

    def run_experiment(
        self,
        experiment_name: str,
        retriever_builder: Callable[[], RetrievalOptimization],
        top_k: int = 5,
        k_list: List[int] = [1, 3, 5],
        run_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Run a single experiment with a given retriever builder.

        Args:
            experiment_name: Unique name/ID for this experiment.
            retriever_builder: Function that returns a configured
                RetrievalOptimization instance.
            top_k: Number of documents to retrieve.
            k_list: List of K values for metrics.
            run_metadata: Optional metadata dict with ``config`` sub-dict
                (same format as ``_make_run_metadata``).

        Returns:
            Dictionary with ``aggregate`` metrics, ``run_id``, and
            ``elapsed_seconds``.
        """
        logger.info("Running experiment: %s", experiment_name)
        retriever = retriever_builder()

        def retrieve_fn(query: str, k: int = top_k) -> List[Document]:
            return retriever.hybrid_search(query, top_k=k)

        results = self.evaluator.evaluate(
            retrieve_fn, top_k=top_k, k_list=k_list, per_query_results=True,
        )

        # Persist to SQLite (sync→async bridge)
        run_id = asyncio.run(
            self._save_to_db(results, run_metadata, experiment_name)
        )

        aggregate = dict(results["aggregate"])
        aggregate["experiment"] = experiment_name
        aggregate["run_id"] = run_id
        return aggregate

    async def _save_to_db(
        self,
        results: Dict[str, Any],
        run_metadata: Dict[str, Any] | None,
        experiment_name: str,
    ) -> int:
        """Async helper: persist results via EvaluationStore."""
        engine = get_async_engine()
        factory = get_session_factory(engine)
        async with factory() as session:
            meta = run_metadata or {
                "timestamp": datetime.now().isoformat(),
                "config": {"retrieval_strategy": experiment_name},
            }
            store = EvaluationStore()
            run_id = await store.save_run(
                session=session,
                run_metadata=meta,
                aggregate_metrics=results.get("aggregate", {}),
                per_query=results.get("per_query", []),
                elapsed_seconds=results.get("elapsed_seconds", 0),
            )
            return run_id

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
    async def load_results() -> list[Dict[str, Any]]:
        """Load all experiment results from SQLite."""
        engine = get_async_engine()
        factory = get_session_factory(engine)
        async with factory() as session:
            store = EvaluationStore()
            runs = await store.get_runs(session)
            return [
                {
                    "run_id": r.id,
                    "experiment": r.retrieval_strategy,
                    "timestamp": r.timestamp,
                    "hit@1": r.hit_at_1,
                    "hit@3": r.hit_at_3,
                    "hit@5": r.hit_at_5,
                    "mrr": r.mrr,
                }
                for r in runs
            ]


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
