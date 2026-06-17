"""Async EvaluationStore — CRUD for RAG evaluation runs and per-query results."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import desc, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import Run, QueryResult

logger = logging.getLogger(__name__)

# Fields that define a unique "version" (config combination)
VERSION_FIELDS = [
    "chunker",
    "embedding_model",
    "retrieval_strategy",
    "use_cross_encoder",
    "cross_encoder_model",
    "use_sparse",
    "use_parent_expansion",
    "top_k",
    "vector_k",
    "sparse_k",
    "rrf_constant",
    "chunk_size",
    "chunk_overlap",
]

# Metric field names shared between Run and QueryResult
METRIC_FIELDS = [
    "mrr",
    "hit_at_1", "hit_at_3", "hit_at_5",
    "precision_at_1", "precision_at_3", "precision_at_5",
    "recall_at_1", "recall_at_3", "recall_at_5",
    "ndcg_at_1", "ndcg_at_3", "ndcg_at_5",
]


class EvaluationStore:
    """Async persistence layer for RAG evaluation results."""

    # ── Write ───────────────────────────────────────────────────

    @staticmethod
    async def save_run(
        session: AsyncSession,
        run_metadata: Dict[str, Any],
        aggregate_metrics: Dict[str, Any],
        per_query: List[Dict[str, Any]],
        elapsed_seconds: float,
    ) -> int:
        """Persist a complete evaluation run and return its ``run_id``.

        Args:
            session: Active ``AsyncSession``.
            run_metadata: Dict with ``config`` sub-dict (from
                ``_make_run_metadata``) plus ``timestamp``.
            aggregate_metrics: Dict of aggregated metric values
                (e.g. ``{"hit@1": 0.85, "mrr": 0.78, ...}``).
            per_query: List of per-query result dicts (the ``per_query``
                list from ``RAGEvaluator.evaluate``).
            elapsed_seconds: Wall time for the evaluation.
        """
        cfg: Dict[str, Any] = run_metadata.get("config", {})

        # Normalize metric keys: "hit@1" → "hit_at_1"
        norm_metrics = _normalize_metric_keys(aggregate_metrics)

        run = Run(
            chunker=cfg.get("chunker", "unknown"),
            embedding_model=cfg.get("embedding_model", "unknown"),
            retrieval_strategy=cfg.get("retrieval_strategy", "unknown"),
            use_cross_encoder=bool(cfg.get("use_cross_encoder", False)),
            cross_encoder_model=cfg.get("cross_encoder_model"),
            use_sparse=bool(cfg.get("use_sparse", False)),
            use_parent_expansion=bool(cfg.get("use_parent_expansion", False)),
            top_k=int(cfg.get("top_k", 5)),
            vector_k=int(cfg.get("vector_k", 30)),
            sparse_k=int(cfg.get("sparse_k", 30)),
            rrf_constant=int(cfg.get("rrf_constant", 60)),
            chunk_size=int(cfg.get("chunk_size", 512)),
            chunk_overlap=int(cfg.get("chunk_overlap", 64)),
            num_chunks=int(cfg.get("num_chunks", 0)),
            timestamp=str(run_metadata.get("timestamp", "")),
            elapsed_seconds=float(elapsed_seconds),
            total_queries=int(norm_metrics.get("total_queries", len(per_query))),
            mrr=float(norm_metrics.get("mrr", 0)),
            hit_at_1=float(norm_metrics.get("hit_at_1", 0)),
            hit_at_3=float(norm_metrics.get("hit_at_3", 0)),
            hit_at_5=float(norm_metrics.get("hit_at_5", 0)),
            precision_at_1=_maybe_float(norm_metrics.get("precision_at_1")),
            precision_at_3=_maybe_float(norm_metrics.get("precision_at_3")),
            precision_at_5=_maybe_float(norm_metrics.get("precision_at_5")),
            recall_at_1=_maybe_float(norm_metrics.get("recall_at_1")),
            recall_at_3=_maybe_float(norm_metrics.get("recall_at_3")),
            recall_at_5=_maybe_float(norm_metrics.get("recall_at_5")),
            ndcg_at_1=_maybe_float(norm_metrics.get("ndcg_at_1")),
            ndcg_at_3=_maybe_float(norm_metrics.get("ndcg_at_3")),
            ndcg_at_5=_maybe_float(norm_metrics.get("ndcg_at_5")),
        )
        session.add(run)
        await session.flush()  # populate run.id

        # Bulk insert per-query results
        qr_entries: List[QueryResult] = []
        for pq in per_query:
            pq_metrics = _normalize_metric_keys(pq.get("metrics", {}))
            qr_entries.append(
                QueryResult(
                    run_id=run.id,
                    query=pq.get("query", ""),
                    retrieved_ids=json.dumps(pq.get("retrieved_ids", []), ensure_ascii=False),
                    relevant_ids=json.dumps(pq.get("relevant_ids", []), ensure_ascii=False),
                    category=pq.get("category"),
                    difficulty=pq.get("difficulty"),
                    query_type=pq.get("query_type"),
                    mrr=_maybe_float(pq_metrics.get("mrr")),
                    hit_at_1=_maybe_float(pq_metrics.get("hit_at_1")),
                    hit_at_3=_maybe_float(pq_metrics.get("hit_at_3")),
                    hit_at_5=_maybe_float(pq_metrics.get("hit_at_5")),
                    precision_at_1=_maybe_float(pq_metrics.get("precision_at_1")),
                    precision_at_3=_maybe_float(pq_metrics.get("precision_at_3")),
                    precision_at_5=_maybe_float(pq_metrics.get("precision_at_5")),
                    recall_at_1=_maybe_float(pq_metrics.get("recall_at_1")),
                    recall_at_3=_maybe_float(pq_metrics.get("recall_at_3")),
                    recall_at_5=_maybe_float(pq_metrics.get("recall_at_5")),
                    ndcg_at_1=_maybe_float(pq_metrics.get("ndcg_at_1")),
                    ndcg_at_3=_maybe_float(pq_metrics.get("ndcg_at_3")),
                    ndcg_at_5=_maybe_float(pq_metrics.get("ndcg_at_5")),
                )
            )
        session.add_all(qr_entries)
        await session.commit()

        logger.info("Saved run_id=%d with %d query results", run.id, len(qr_entries))
        return run.id

    # ── Read: runs ──────────────────────────────────────────────

    @staticmethod
    async def get_runs(
        session: AsyncSession,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Run]:
        """List runs, optionally filtered by config fields, newest first."""
        stmt = select(Run).order_by(desc(Run.timestamp))
        if filters:
            for key, value in filters.items():
                if hasattr(Run, key) and value is not None:
                    stmt = stmt.where(getattr(Run, key) == value)
        stmt = stmt.limit(limit).offset(offset)
        result = await session.execute(stmt)
        return result.scalars().all()

    @staticmethod
    async def get_run(
        session: AsyncSession,
        run_id: int,
        *,
        include_queries: bool = False,
    ) -> Optional[Run]:
        """Get a single run, optionally eager-loading query results."""
        stmt = select(Run).where(Run.id == run_id)
        if include_queries:
            stmt = stmt.options(selectinload(Run.query_results))
        result = await session.execute(stmt)
        return result.scalars().first()

    @staticmethod
    async def get_run_count(session: AsyncSession) -> int:
        stmt = select(func.count(Run.id))
        result = await session.execute(stmt)
        return result.scalar() or 0

    # ── Read: versions ──────────────────────────────────────────

    @staticmethod
    async def get_versions(session: AsyncSession) -> List[Dict[str, Any]]:
        """Return distinct config combinations, each with latest metrics + run count."""
        # Sub-query: max(id) per config group
        version_cols = [getattr(Run, f) for f in VERSION_FIELDS]
        sub = (
            select(
                *version_cols,
                func.max(Run.id).label("latest_run_id"),
                func.count(Run.id).label("run_count"),
            )
            .group_by(*version_cols)
            .subquery()
        )

        stmt = (
            select(Run, sub.c.run_count)
            .join(sub, Run.id == sub.c.latest_run_id)
            .order_by(desc(Run.timestamp))
        )
        result = await session.execute(stmt)
        rows = result.all()

        versions: List[Dict[str, Any]] = []
        for run, count in rows:
            versions.append({
                "version": {f: getattr(run, f) for f in VERSION_FIELDS},
                "latest_run_id": run.id,
                "run_count": count,
                "metrics": _extract_metrics(run),
                "timestamp": run.timestamp,
                "elapsed_seconds": run.elapsed_seconds,
                "total_queries": run.total_queries,
            })
        return versions

    @staticmethod
    async def get_latest_by_version(session: AsyncSession) -> Sequence[Run]:
        """Return the latest Run for each config combination."""
        version_cols = [getattr(Run, f) for f in VERSION_FIELDS]
        sub = (
            select(
                *version_cols,
                func.max(Run.id).label("max_id"),
            )
            .group_by(*version_cols)
            .subquery()
        )
        stmt = (
            select(Run)
            .join(sub, Run.id == sub.c.max_id)
            .order_by(desc(Run.timestamp))
        )
        result = await session.execute(stmt)
        return result.scalars().all()

    # ── Read: compare ───────────────────────────────────────────

    @staticmethod
    async def compare_runs(
        session: AsyncSession, run_ids: List[int],
    ) -> Sequence[Run]:
        """Get multiple runs by ID for side-by-side comparison."""
        stmt = select(Run).where(Run.id.in_(run_ids)).order_by(Run.id)
        result = await session.execute(stmt)
        return result.scalars().all()

    # ── Delete ──────────────────────────────────────────────────

    @staticmethod
    async def delete_run(session: AsyncSession, run_id: int) -> bool:
        """Delete a run (cascades to query_results). Returns True if deleted."""
        run = await session.get(Run, run_id)
        if run is None:
            return False
        await session.delete(run)
        await session.commit()
        logger.info("Deleted run_id=%d", run_id)
        return True


# ── helpers ─────────────────────────────────────────────────────────

def _normalize_metric_keys(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ``hit@1`` → ``hit_at_1`` etc."""
    mapping = {
        "hit@1": "hit_at_1", "hit@3": "hit_at_3", "hit@5": "hit_at_5",
        "precision@1": "precision_at_1", "precision@3": "precision_at_3",
        "precision@5": "precision_at_5",
        "recall@1": "recall_at_1", "recall@3": "recall_at_3",
        "recall@5": "recall_at_5",
        "ndcg@1": "ndcg_at_1", "ndcg@3": "ndcg_at_3", "ndcg@5": "ndcg_at_5",
    }
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        out[mapping.get(k, k)] = v
    return out


def _maybe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_metrics(run: Run) -> Dict[str, float]:
    """Extract metric fields from a Run into a flat dict (with @ notation)."""
    return {
        "hit@1": run.hit_at_1,
        "hit@3": run.hit_at_3,
        "hit@5": run.hit_at_5,
        "mrr": run.mrr,
        "precision@1": run.precision_at_1,
        "precision@3": run.precision_at_3,
        "precision@5": run.precision_at_5,
        "recall@1": run.recall_at_1,
        "recall@3": run.recall_at_3,
        "recall@5": run.recall_at_5,
        "ndcg@1": run.ndcg_at_1,
        "ndcg@3": run.ndcg_at_3,
        "ndcg@5": run.ndcg_at_5,
    }
