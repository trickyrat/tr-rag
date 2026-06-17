"""RAG evaluation runner — chunk → index → retrieve → evaluate.

Usage::

    uv run rag_runner.py                # uses config.yaml
    RAG_TOP_K=5 uv run rag_runner.py    # override via env vars

Results are saved to ``src/evaluation/results/*.json`` for
cross-run comparison (chunker × embedding × retrieval strategy).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

from langchain_core.documents import Document

from core.config import RAGConfig
from evaluation import EvaluationStore, RAGEvaluator
from evaluation.db import get_async_engine, get_session_factory
from rag import DocumentChunker, RetrievalOptimization, VectorStoreBuilder
from scripts.convert_testset_to_hash import find_best_chunks

# ── logging ─────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════

def _export_chunk_mapping(
    chunks: List[Document],
    path: str = "test/evaluation/chunk_mapping.jsonl",
) -> List[Dict[str, Any]]:
    """Write chunk_id + content_hash mapping to JSONL; return the records."""
    records: List[Dict[str, Any]] = []
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": chunk.metadata.get("chunk_id"),
                        "content_hash": chunk.metadata.get("content_hash"),
                        "source": chunk.metadata.get("source"),
                        "doc_name": chunk.metadata.get("doc_name"),
                        "primary_category": chunk.metadata.get("primary_category"),
                        "sub_category": chunk.metadata.get("sub_category"),
                        "content_preview": chunk.page_content[:500],
                        "chunk_size": len(chunk.page_content),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            records.append(
                {"content_hash": chunk.metadata.get("content_hash", ""),
                 "content_preview": chunk.page_content[:500]}
            )
    logger.info("Exported %d chunk mappings → %s", len(records), path)
    return records


def _sync_testset(
    source_path: str,
    target_path: str,
    chunk_records: List[Dict[str, Any]],
) -> None:
    """Regenerate ``relevant_content_hashes`` from keyword-overlap matching."""
    src = Path(source_path)
    if not src.exists():
        logger.info("Testset source not found at %s, skipping sync", source_path)
        return

    cases: List[Dict[str, Any]] = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    converted: List[Dict[str, Any]] = []
    for case in cases:
        q = case.get("query", "").strip()
        if not q:
            continue
        if "relevant_content_hashes" in case:
            converted.append(case)
            continue
        hashes = find_best_chunks(q, chunk_records, top_n=3, min_score=0.05)
        if hashes:
            new_case: Dict[str, Any] = {"query": q, "relevant_content_hashes": hashes}
            for fld in ("category", "difficulty", "query_type"):
                if fld in case:
                    new_case[fld] = case[fld]
            converted.append(new_case)

    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        for case in converted:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    logger.info("Testset synced: %d cases → %s", len(converted), target_path)


def _make_run_metadata(config: RAGConfig, num_chunks: int) -> Dict[str, Any]:
    """Build a structured metadata snapshot for the result file."""
    strategy = (
        "cross_encoder" if config.use_cross_encoder
        else "hybrid" if config.use_sparse
        else "vector_only"
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "chunker": "markdown_parser" if config.use_markdown_parser else "legacy",
            "embedding_model": Path(config.embedding_model).name,
            "retrieval_strategy": strategy,
            "use_cross_encoder": config.use_cross_encoder,
            "cross_encoder_model": (
                Path(config.cross_encoder_model).name
                if config.use_cross_encoder else None
            ),
            "use_sparse": config.use_sparse,
            "use_parent_expansion": config.use_parent_expansion,
            "top_k": config.top_k,
            "vector_k": config.vector_k,
            "sparse_k": config.sparse_k,
            "rrf_constant": config.rrf_constant,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "num_chunks": num_chunks,
        },
    }


async def _save_results_async(
    results: Dict[str, Any],
    run_meta: Dict[str, Any],
) -> int:
    """Persist evaluation results to SQLite via async EvaluationStore.

    Returns the ``run_id`` of the newly created row.
    """
    engine = get_async_engine()
    factory = get_session_factory(engine)
    async with factory() as session:
        store = EvaluationStore()
        run_id = await store.save_run(
            session=session,
            run_metadata=run_meta,
            aggregate_metrics=results.get("aggregate", {}),
            per_query=results.get("per_query", []),
            elapsed_seconds=results.get("elapsed_seconds", 0),
        )
    return run_id


def _save_results(
    results: Dict[str, Any],
    run_meta: Dict[str, Any],
) -> int:
    """Sync wrapper around ``_save_results_async``."""
    return asyncio.run(_save_results_async(results, run_meta))


def _print_summary(results: Dict[str, Any], run_id: int | None = None) -> None:
    """Thin console summary — key metrics only."""
    agg = results.get("aggregate", {})
    extra = f"  run_id={run_id}" if run_id else ""
    logger.info("Hit@1=%.4f  Hit@3=%.4f  MRR=%.4f  (%.1fs)%s",
                agg.get("hit@1", 0), agg.get("hit@3", 0),
                agg.get("mrr", 0), results.get("elapsed_seconds", 0), extra)


# ═══════════════════════════════════════════════════════════════════
# pipeline
# ═══════════════════════════════════════════════════════════════════

def build_pipeline(
    config: RAGConfig,
) -> tuple[RetrievalOptimization, Callable[[str, int], List[Document]], int]:
    """Chunk → index → build retriever.

    Returns ``(retriever, retrieve_fn, num_chunks)``.
    """
    logger.info("🚀 Initializing RAG system...")
    logger.info("Loading and chunking documents")

    # 1. Chunk
    chunker = DocumentChunker(
        config.knowledge_base_path,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        use_markdown_parser=config.use_markdown_parser,
    )
    chunker.load_documents()
    chunks = chunker.chunk_documents()
    logger.info(
        "Chunked into %d chunks (model=%s, chunk_size=%d, parent_exp=%s)",
        len(chunks), config.embedding_model, config.chunk_size,
        config.use_parent_expansion,
    )

    # 2. Export mapping + sync testset
    chunk_records = _export_chunk_mapping(chunks)
    logger.info("Syncing testset with current chunks...")
    _sync_testset(
        "test/evaluation/testset_new.jsonl",
        config.evaluation_testset_path,
        chunk_records,
    )

    # 3. Index
    builder = VectorStoreBuilder(
        model_name=config.embedding_model,
        index_save_path=config.index_save_path,
    )
    builder.reset_collection()
    builder.index_documents(chunks)
    logger.info("Indexed %d chunks", len(chunks))

    # 4. Retriever
    retriever = RetrievalOptimization(
        vectorstore=builder.get_vectorstore(),
        chunks=chunks,
        parent_docs=chunker.parent_docs,
        parent_child_map=chunker.parent_child_map,
        sparse_search_fn=builder.sparse_search,
        use_sparse=config.use_sparse,
        vector_k=config.vector_k,
        sparse_k=config.bm25_k,
        top_k=config.top_k,
        rrf_constant=config.rrf_constant,
        cross_encoder_model=(
            config.cross_encoder_model if config.use_cross_encoder else None
        ),
        cross_encoder_candidate_k=config.cross_encoder_top_k,
    )
    logger.info("Knowledge base ready")

    def retrieve_fn(query: str, top_k: int = 5) -> List[Document]:
        if config.use_parent_expansion:
            return retriever.search_with_parents(query, top_k=top_k)
        return retriever.search(query, top_k=top_k)

    return retriever, retrieve_fn, len(chunks)


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()
    config = RAGConfig()
    _retriever, _retrieve_fn, num_chunks = build_pipeline(config)
    pipeline_sec = time.perf_counter() - t_start

    run_meta = _make_run_metadata(config, num_chunks)
    run_meta["pipeline_seconds"] = round(pipeline_sec, 2)

    evaluator = RAGEvaluator(config.evaluation_testset_path)
    results = evaluator.evaluate(
        retrieve_fn=_retrieve_fn,
        top_k=config.top_k,
        k_list=[1, 3, 5],
        per_query_results=True,
        run_metadata=run_meta,
    )

    run_id = _save_results(results, run_meta)
    _print_summary(results, run_id)
    logger.info("Total wall time: %.1fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()